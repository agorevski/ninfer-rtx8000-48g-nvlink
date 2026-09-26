#include "ops/linear/turing.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_rowsplit_gemm_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_mma.cuh"
#include "ops/linear/q6/q6_rowsplit_gemm_mma.cuh"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"
#include "ops/linear/turing_gemv.cuh"
#include "ops/linear/turing_batched.cuh"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kPreparedActivationMinTokens = 512;
constexpr int kPreparedActivationMaxTokens = 128 * kCudaGridYLimit;

struct PreparedActivations {
    Tensor values;
    Tensor masks;
    Tensor exponents;
};

template <class Allocator>
PreparedActivations allocate_prepared_activations(Allocator& workspace, int input_rows,
                                                  int tokens) {
    if (input_rows <= 0 || input_rows % 8 != 0 || tokens <= 0) {
        throw std::invalid_argument("Turing activation workspace: invalid projection extent");
    }
    return {workspace.alloc(DType::FP16, {input_rows, tokens}),
            workspace.alloc(DType::U8, {input_rows / 8, tokens}),
            workspace.alloc(DType::I32, {tokens})};
}

__global__ void prepare_activations(const __nv_bfloat16* __restrict__ input,
                                    __half* __restrict__ prepared,
                                    std::uint8_t* __restrict__ masks,
                                    int* __restrict__ exponents, int k) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    __shared__ unsigned maxima[8];
    __shared__ int exponent;
    unsigned maximum = 0;
    for (int chunk = tid; chunk < k / 8; chunk += blockDim.x) {
        const uint4 values = load_vec<uint4>(input + static_cast<std::int64_t>(token) * k + chunk * 8);
        const unsigned words[4] = {values.x, values.y, values.z, values.w};
#pragma unroll
        for (unsigned word : words) {
            maximum = max(maximum, word & 0x7fffu);
            maximum = max(maximum, (word >> 16) & 0x7fffu);
        }
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if (lane == 0) maxima[tid >> 5] = maximum;
    __syncthreads();
    if (tid < 32) {
        maximum = tid < 8 ? maxima[tid] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
        if (tid == 0) {
            int shift = 0;
            if (maximum != 0 && maximum < 0x4780u) {
                // BF16 subnormals are integer multiples of 2^-133.
                const int value_exponent =
                    maximum >= 0x80u ? int(maximum >> 7) - 127 : -133 + 31 - __clz(maximum);
                shift = max(0, 14 - value_exponent);
            }
            exponent = shift;
            exponents[token] = shift;
        }
    }
    __syncthreads();
    for (int chunk = tid; chunk < k / 8; chunk += blockDim.x) {
        const auto index = static_cast<std::int64_t>(token) * k + chunk * 8;
        const uint4 values = load_vec<uint4>(input + index);
        const unsigned words[4] = {values.x, values.y, values.z, values.w};
        union alignas(16) Packed {
            uint4 bits;
            __half2 pair[4];
        } packed;
        unsigned exceptional = 0;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float first = turing_scale_activation(__uint_as_float(words[pair] << 16), exponent);
            const float second = turing_scale_activation(
                __uint_as_float(words[pair] & 0xffff0000u), exponent);
            union PairBits { __half2 value; unsigned bits; } pair_bits;
            pair_bits.value = __floats2half2_rn(first, second);
            const float2 roundtrip = __half22float2(pair_bits.value);
            const bool bad0 = roundtrip.x != first;
            const bool bad1 = roundtrip.y != second;
            exceptional |= unsigned(bad0) << (pair * 2);
            exceptional |= unsigned(bad1) << (pair * 2 + 1);
            if (bad0) pair_bits.bits &= 0xffff0000u;
            if (bad1) pair_bits.bits &= 0x0000ffffu;
            packed.pair[pair] = pair_bits.value;
        }
        store_vec(prepared + index, packed.bits);
        masks[index / 8] = static_cast<std::uint8_t>(exceptional);
    }
}

template <int Bits, bool Add, bool SwiGlu>
__global__ __launch_bounds__(256, 1) void prepared_projection(
    const __nv_bfloat16* __restrict__ input, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    const __half* __restrict__ prepared, const std::uint8_t* __restrict__ masks,
    const int* __restrict__ exponents, __nv_bfloat16* __restrict__ output,
    int rows, int k, int padded_k, int tokens) {
    static_assert(Bits == 4 || Bits == 5);
    static_assert(!SwiGlu || (Bits == 4 && !Add));
    constexpr int WarpRows = SwiGlu ? 64 : 32;
    constexpr int WarpCols = SwiGlu ? 16 : 32;
    constexpr int MT = WarpRows / 16;
    constexpr int NT = WarpCols / 8;
    constexpr int OutputTile = SwiGlu ? 32 : 64;
    __shared__ __align__(16) A16Operand weights[64 * 64], activations[128 * 64];
    float acc[MT][NT][4] = {};
    const int row_begin = static_cast<int>(blockIdx.x) * OutputTile;
    const int token_begin = static_cast<int>(blockIdx.y) * 128;
    if constexpr (SwiGlu) {
        turing_rowsplit_tile<Bits, 64, 128, WarpRows, WarpCols, false, true>(
            input, codes, high, scales, weights, activations, acc, k, padded_k, tokens,
            token_begin, TuringPairedRows<32>{row_begin, rows / 2}, nullptr, prepared, masks, exponents);
    } else {
        turing_rowsplit_tile<Bits, 64, 128, WarpRows, WarpCols, false, true>(
            input, codes, high, scales, weights, activations, acc, k, padded_k, tokens,
            token_begin, TuringLinearRows{row_begin, rows}, nullptr, prepared, masks, exponents);
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_row = warp / (128 / WarpCols);
    const int warp_col = warp % (128 / WarpCols);
    const int output_rows = SwiGlu ? rows / 2 : rows;
#pragma unroll
    for (int mi = 0; mi < (SwiGlu ? MT / 2 : MT); ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int row = row_begin + warp_row * WarpRows + mi * 16 + (lane >> 2) +
                                (component / 2) * 8;
                const int token = token_begin + warp_col * WarpCols + ni * 8 +
                                  (lane & 3) * 2 + (component & 1);
                if (row < output_rows && token < tokens) {
                    const auto index = static_cast<std::int64_t>(token) * output_rows + row;
                    float value = turing_scale_activation(acc[mi][ni][component], -exponents[token]);
                    if constexpr (SwiGlu) {
                        value = silu(value) * turing_scale_activation(
                            acc[mi + MT / 2][ni][component], -exponents[token]);
                    }
                    if constexpr (Add) value += __bfloat162float(output[index]);
                    output[index] = __float2bfloat16_rn(value);
                }
            }
        }
    }
}

template <int Bits, bool Add, bool SwiGlu = false>
void launch_prepared_projection(const Tensor& x, const Weight& weight, Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    const auto prepared = allocate_prepared_activations(workspace, x.ne[0], x.ne[1]);
    prepare_activations<<<x.ne[1], 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__half*>(prepared.values.data),
        static_cast<std::uint8_t*>(prepared.masks.data),
        static_cast<int*>(prepared.exponents.data), x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
    constexpr int OutputTile = SwiGlu ? 32 : 64;
    const dim3 grid((out.ne[0] + OutputTile - 1) / OutputTile, (x.ne[1] + 127) / 128);
    prepared_projection<Bits, Add, SwiGlu><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales),
        static_cast<const __half*>(prepared.values.data),
        static_cast<const std::uint8_t*>(prepared.masks.data),
        static_cast<const int*>(prepared.exponents.data),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, weight.padded_shape[1],
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

struct AddEpilogue {
    __device__ __forceinline__ void operator()(__nv_bfloat16* destination, float value) const {
        *destination = __float2bfloat16_rn(__bfloat162float(*destination) + value);
    }
};

template <int Bits, bool Add>
void decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    turing_rowsplit_gemv_kernel<Bits, 4, 1, Add>
        <<<dim3((weight.n + 3) / 4, x.ne[1]), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, weight.padded_shape[1]);
}

template <int Bits, bool Add, int Columns>
void matrix(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int Rows = 64;
    constexpr int WarpRows = Columns <= 16 ? 16 : 32;
    constexpr int WarpColumns = Columns <= 16 ? Columns : (Columns == 32 ? 16 : 32);
    const dim3 grid((w.n + Rows - 1) / Rows, (x.ne[1] + Columns - 1) / Columns);
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    if constexpr (Bits == 4) {
        using Schedule = Q4RowSplitMmaGemmSchedule<
            Rows, Columns, 64, WarpRows, WarpColumns, 2, 1, Q4FragmentPipeline::Serial,
            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
        using Epilogue = std::conditional_t<Add, AddEpilogue, Q4MmaStoreEpilogue>;
        q4_rowsplit_gemm_mma_kernel<Schedule, false, Epilogue>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                input, codes, scales, output, w.n, w.k, x.ne[1], w.padded_shape[1]);
    } else if constexpr (Bits == 5) {
        using Schedule = Q5RowSplitMmaGemmSchedule<
            Rows, Columns, 64, WarpRows, WarpColumns, 2, 1, Q5FragmentPipeline::Serial,
            Cache::cg, Cache::cg, Q5ScaleLoad::Pair32>;
        constexpr auto Epilogue =
            Add ? Q5MmaEpilogue::AddResidual : Q5MmaEpilogue::Store;
        q5_rowsplit_gemm_mma_kernel<Schedule, false, Epilogue>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                input, codes, high, scales, output, w.n, w.k, x.ne[1],
                w.padded_shape[1]);
    } else if constexpr (Bits == 6) {
        static_assert(!Add);
        using Schedule = Q6RowSplitMmaGemmSchedule<
            Rows, Columns, 64, WarpRows, WarpColumns, 2, 1, Q6FragmentPipeline::Serial,
            Cache::cg, Cache::cg, Q6ScaleLoad::Pair32>;
        q6_rowsplit_gemm_mma_kernel<Schedule, false>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                input, codes, high, scales, output, w.n, w.k, x.ne[1], w.padded_shape[1]);
    } else {
        using Schedule = Q8RowSplitMmaGemmSchedule<Rows, Columns, WarpRows, WarpColumns, 1>;
        constexpr auto Epilogue = Add ? Q8Epilogue::Residual : Q8Epilogue::Store;
        q8_rowsplit_gemm_mma_kernel<Schedule, false, Epilogue>
            <<<grid, Schedule::THREADS, 0, stream>>>(
                input, codes, scales, Q8ContiguousOutput{output, w.n}, w.n, w.k, x.ne[1],
                w.padded_shape[1]);
    }
}

template <int Bits, bool Add>
void project(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 2 || (w.n == 2560 && x.ne[1] <= 8)) {
        decode<Bits, Add>(x, w, out, stream);
    } else if (x.ne[1] <= 8) {
        matrix<Bits, Add, 8>(x, w, out, stream);
    } else if (x.ne[1] <= 16) {
        matrix<Bits, Add, 16>(x, w, out, stream);
    } else if (x.ne[1] <= 64) {
        matrix<Bits, Add, 32>(x, w, out, stream);
    } else {
        matrix<Bits, Add, 128>(x, w, out, stream);
    }
}

template <bool Add>
bool dispatch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
              WorkspaceArena* workspace) {
    const bool shard = (weight.n == 17408 && weight.k == 5120) ||
                       (weight.n == 2560 && weight.k == 17408);
    if (x.ne[1] > 1 && x.ne[1] <= 32 && !shard) return false;
    if (weight.qtype != QType::Q4_G64_FP16 && weight.qtype != QType::Q5_G64_FP16 &&
        weight.qtype != QType::Q6_G64_FP16 && weight.qtype != QType::Q8_G32_FP16) return false;
    if constexpr (Add) {
        if (weight.qtype == QType::Q6_G64_FP16) return false;
    }
    for_each_token_slice(x.ne[1], 128, [&](int begin, int count) {
        const Tensor input = x.slice(1, begin, count);
        Tensor output = out.slice(1, begin, count);
        if (workspace != nullptr && count >= kPreparedActivationMinTokens) {
            if (weight.qtype == QType::Q4_G64_FP16) {
                launch_prepared_projection<4, Add>(input, weight, output, *workspace, stream);
                return;
            }
            if (weight.qtype == QType::Q5_G64_FP16) {
                launch_prepared_projection<5, Add>(input, weight, output, *workspace, stream);
                return;
            }
        }
        switch (weight.qtype) {
        case QType::Q4_G64_FP16: project<4, Add>(input, weight, output, stream); break;
        case QType::Q5_G64_FP16: project<5, Add>(input, weight, output, stream); break;
        case QType::Q6_G64_FP16:
            if constexpr (!Add) project<6, false>(input, weight, output, stream);
            break;
        case QType::Q8_G32_FP16: project<8, Add>(input, weight, output, stream); break;
        default: break;
        }
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int Bits, int Columns>
void swiglu(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    if (x.ne[1] == 1) {
        turing_rowsplit_swiglu_kernel<Bits>
            <<<(out.ne[0] + 3) / 4, 128, 0, stream>>>(
                input, codes, scales, output, out.ne[0], w.k);
    } else if constexpr (Bits == 4) {
        using Schedule = GemmCfg<64, Columns, 64, 64, 16, 2, 1, false, true, true>;
        const dim3 grid((out.ne[0] + 31) / 32, (x.ne[1] + Columns - 1) / Columns);
        q4_linear_swiglu_mma_split_half_pair_kernel<Schedule, false>
            <<<grid, Schedule::THREADS, 0, stream>>>(
                input, codes, scales, output, out.ne[0], w.k, x.ne[1], w.padded_shape[1]);
    } else {
        using Schedule = Q8RowSplitMmaGemmSchedule<64, Columns, 64, 16, 1>;
        const dim3 grid((out.ne[0] + 31) / 32, (x.ne[1] + Columns - 1) / Columns);
        q8_rowsplit_gemm_mma_kernel<Schedule, false, Q8Epilogue::SwiGluSplitHalf>
            <<<grid, Schedule::THREADS, 0, stream>>>(
                input, codes, scales, Q8ContiguousOutput{output, out.ne[0]}, w.n, w.k, x.ne[1],
                w.padded_shape[1]);
    }
}

} // namespace

std::size_t turing_linear_workspace_capacity_bytes(QType qtype, std::int32_t input_rows,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Turing activation workspace: invalid token interval");
    }
    if ((qtype != QType::Q4_G64_FP16 && qtype != QType::Q5_G64_FP16) ||
        max_tokens < kPreparedActivationMinTokens) return 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate_prepared_activations(layout, input_rows,
                                       std::min(max_tokens, kPreparedActivationMaxTokens));
    return layout.peak_bytes(1);
}

bool turing_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                    WorkspaceArena* workspace) {
    if (try_turing_batched_linear(x, weight, out, stream)) return true;
    return dispatch<false>(x, weight, out, stream, workspace);
}

void turing_linear_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (!turing_linear(x, weight, out, stream))
        throw std::invalid_argument("Turing Linear: unsupported projection profile");
}

bool turing_linear_add(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                        WorkspaceArena* workspace) {
    return dispatch<true>(x, weight, out, stream, workspace);
}

bool turing_linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                           WorkspaceArena* workspace) {
    if (try_turing_batched_swiglu(x, weight, out, stream)) return true;
    if (x.ne[1] > 1 && x.ne[1] <= 32) return false;
    if (weight.qtype != QType::Q4_G64_FP16 && weight.qtype != QType::Q8_G32_FP16) return false;
    for_each_token_slice(x.ne[1], 128, [&](int begin, int count) {
        const Tensor input = x.slice(1, begin, count);
        Tensor output = out.slice(1, begin, count);
        if (weight.qtype == QType::Q4_G64_FP16) {
            if (workspace != nullptr && count >= kPreparedActivationMinTokens)
                launch_prepared_projection<4, false, true>(input, weight, output, *workspace, stream);
            else if (count <= 64)
                swiglu<4, 32>(input, weight, output, stream);
            else
                swiglu<4, 128>(input, weight, output, stream);
        } else {
            if (count <= 64) swiglu<8, 32>(input, weight, output, stream);
            else swiglu<8, 128>(input, weight, output, stream);
        }
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ninfer::ops::detail
