#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/turing_gemv.cuh"

namespace ninfer::ops::detail {

template <int Bits>
__device__ __forceinline__ void turing_batched_codes(
    const std::uint8_t* codes, int column, float (&values)[8]) {
    static_assert(Bits == 4 || Bits == 8);
    if constexpr (Bits == 8) {
        const uint2 packed = load_vec<uint2>(codes + column);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
            values[2 * pair] = float(static_cast<std::int8_t>(word));
            values[2 * pair + 1] = float(static_cast<std::int8_t>(word >> 8));
        }
    } else {
        const unsigned packed = load_vec<unsigned>(codes + column / 2);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            values[i] = float((int((packed >> (4 * i)) & 15u) ^ 8) - 8);
    }
}

template <int Bits, int Columns, bool SwiGlu>
__global__ __launch_bounds__(128) void turing_batched_gemv_kernel(
    const __nv_bfloat16* input, const std::uint8_t* codes, const std::uint8_t* scales,
    __nv_bfloat16* output, int rows, int k, int padded_k, int tokens) {
    static_assert(!SwiGlu || Bits == 4);
    constexpr int group = Bits == 8 ? 32 : 64;
    constexpr int packing = Bits == 8 ? 1 : 2;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int output_rows = SwiGlu ? rows / 2 : rows;
    if (row >= output_rows) return;
    const auto row_offset = static_cast<std::int64_t>(row) * padded_k;
    const auto up_offset = static_cast<std::int64_t>(row + output_rows) * padded_k;
    const auto* row_codes = codes + row_offset / packing;
    const auto* row_scales = scales + row_offset / group * 2;
    float accumulator[Columns] = {};
    float up_accumulator[SwiGlu ? Columns : 1] = {};
#pragma unroll 4
    for (int column = lane * 8; column < k; column += 256) {
        float weight[8], up[8];
        turing_batched_codes<Bits>(row_codes, column, weight);
        const float scale = __half2float(__ushort_as_half(
            *reinterpret_cast<const std::uint16_t*>(row_scales + column / group * 2)));
        float up_scale = 0;
        if constexpr (SwiGlu) {
            turing_batched_codes<Bits>(codes + up_offset / packing, column, up);
            up_scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                scales + (up_offset + column) / group * 2)));
        }
#pragma unroll
        for (int token = 0; token < Columns; ++token) {
            if (token < tokens) {
                const uint4 activation =
                    load_ldg<uint4>(input + static_cast<std::int64_t>(token) * k + column);
                const float2 values[4] = {
                    bf16x2_bits_to_float2(activation.x), bf16x2_bits_to_float2(activation.y),
                    bf16x2_bits_to_float2(activation.z), bf16x2_bits_to_float2(activation.w)};
                float sum = 0, up_sum = 0;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    sum = fmaf(weight[2 * pair], values[pair].x, sum);
                    sum = fmaf(weight[2 * pair + 1], values[pair].y, sum);
                    if constexpr (SwiGlu) {
                        up_sum = fmaf(up[2 * pair], values[pair].x, up_sum);
                        up_sum = fmaf(up[2 * pair + 1], values[pair].y, up_sum);
                    }
                }
                accumulator[token] = fmaf(sum, scale, accumulator[token]);
                if constexpr (SwiGlu)
                    up_accumulator[token] = fmaf(up_sum, up_scale, up_accumulator[token]);
            }
        }
    }
#pragma unroll
    for (int token = 0; token < Columns; ++token) {
        if (token < tokens) {
            float value = warp_reduce_sum(accumulator[token]);
            if constexpr (SwiGlu)
                value = silu(value) * warp_reduce_sum(up_accumulator[token]);
            if (lane == 0)
                output[static_cast<std::int64_t>(token) * output_rows + row] =
                    __float2bfloat16_rn(value);
        }
    }
}

template <int Bits, bool SwiGlu>
void turing_batched_launch(const Tensor& x, const Weight& weight, Tensor& output,
                           cudaStream_t stream) {
    const auto launch = [&]<int Columns>() {
        turing_batched_gemv_kernel<Bits, Columns, SwiGlu>
            <<<(output.ne[0] + 3) / 4, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(output.data), weight.n, weight.k,
                weight.padded_shape[1], x.ne[1]);
    };
    if (x.ne[1] <= 2) launch.template operator()<2>();
    else if (x.ne[1] <= 4) launch.template operator()<4>();
    else launch.template operator()<8>();
    CUDA_CHECK(cudaGetLastError());
}

inline bool try_turing_batched_linear(const Tensor& x, const Weight& weight, Tensor& output,
                                      cudaStream_t stream) {
    if (x.ne[1] < 2 || weight.k != 5120 || weight.layout != QuantLayout::RowSplit) return false;
    if (weight.qtype == QType::Q4_G64_FP16 &&
        ((weight.n == 34816 && x.ne[1] <= 8) || (weight.n == 17408 && x.ne[1] <= 4))) {
        turing_batched_launch<4, false>(x, weight, output, stream);
        return true;
    }
    if (weight.qtype == QType::Q8_G32_FP16 && weight.n == 248320 && x.ne[1] <= 4) {
        turing_batched_launch<8, false>(x, weight, output, stream);
        return true;
    }
    return false;
}

inline bool try_turing_batched_swiglu(const Tensor& x, const Weight& weight, Tensor& output,
                                      cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 8 || weight.qtype != QType::Q4_G64_FP16 ||
        weight.n != 34816 || weight.k != 5120 || weight.layout != QuantLayout::RowSplit)
        return false;
    turing_batched_launch<4, true>(x, weight, output, stream);
    return true;
}

} // namespace ninfer::ops::detail
