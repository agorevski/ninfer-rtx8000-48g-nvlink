#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Bits, int KWarps>
__device__ __forceinline__ float turing_rowsplit_dot(
    const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* high,
    const std::uint8_t* scales, int k, int warp_in_row) {
    const int lane = threadIdx.x & 31;
    float accumulator = 0.0f;
#pragma unroll 4
    for (int column = warp_in_row * 256 + lane * 8; column < k; column += KWarps * 256) {
        const uint4 activation = load_ldg<uint4>(x + column);
        const float2 values[4] = {
            bf16x2_bits_to_float2(activation.x), bf16x2_bits_to_float2(activation.y),
            bf16x2_bits_to_float2(activation.z), bf16x2_bits_to_float2(activation.w)};
        const float scale = __half2float(__ushort_as_half(
            *reinterpret_cast<const std::uint16_t*>(scales + column / (Bits == 8 ? 32 : 64) * 2)));
        float dot = 0.0f;
        if constexpr (Bits == 4) {
            const unsigned packed = load_vec<unsigned>(codes + column / 2);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const unsigned byte = packed >> (pair * 8);
                const int first = (int(byte & 15u) ^ 8) - 8;
                const int second = (int((byte >> 4) & 15u) ^ 8) - 8;
                dot = fmaf(float(first), values[pair].x, dot);
                dot = fmaf(float(second), values[pair].y, dot);
            }
        } else if constexpr (Bits == 8) {
            const uint2 packed = load_vec<uint2>(codes + column);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
                dot = fmaf(float(static_cast<std::int8_t>(word)), values[pair].x, dot);
                dot = fmaf(float(static_cast<std::int8_t>(word >> 8)), values[pair].y, dot);
            }
        } else {
            const unsigned packed = load_vec<unsigned>(codes + column / 2);
            constexpr int HighBits = Bits - 4;
            const unsigned upper = Bits == 5 ? high[column / 8]
                                              : *reinterpret_cast<const std::uint16_t*>(
                                                    high + column / 4);
            constexpr int Mask = (1 << HighBits) - 1;
            constexpr int Sign = 1 << (Bits - 1);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const unsigned byte = packed >> (pair * 8);
                const int q0 = ((int(byte & 15u) |
                                 int(((upper >> (pair * 2 * HighBits)) & Mask) << 4)) ^ Sign) - Sign;
                const int q1 = ((int((byte >> 4) & 15u) |
                                 int(((upper >> ((pair * 2 + 1) * HighBits)) & Mask) << 4)) ^ Sign) - Sign;
                dot = fmaf(float(q0), values[pair].x, dot);
                dot = fmaf(float(q1), values[pair].y, dot);
            }
        }
        accumulator = fmaf(dot, scale, accumulator);
    }
    return warp_reduce_sum(accumulator);
}

template <int Bits, int RowsPerCta = 4, int KWarps = 1, bool Add = false>
__global__ void turing_rowsplit_gemv_kernel(
    const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* high,
    const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int padded_k) {
    x += static_cast<std::int64_t>(blockIdx.y) * k;
    out += static_cast<std::int64_t>(blockIdx.y) * n;
    const int warp = threadIdx.x >> 5;
    const int local_row = warp / KWarps;
    const int split = warp % KWarps;
    const int row = blockIdx.x * RowsPerCta + local_row;
    __shared__ float partial[RowsPerCta][KWarps];
    const auto offset = static_cast<std::int64_t>(row) * padded_k;
    const float sum = row < n
                          ? turing_rowsplit_dot<Bits, KWarps>(
                                x, codes + offset / (Bits == 8 ? 1 : 2),
                                high ? high + offset * (Bits - 4) / 8 : nullptr,
                                scales + offset / (Bits == 8 ? 32 : 64) * 2, k, split)
                          : 0.0f;
    if ((threadIdx.x & 31) == 0) partial[local_row][split] = sum;
    __syncthreads();
    if (threadIdx.x < RowsPerCta) {
        float value = 0.0f;
#pragma unroll
        for (int i = 0; i < KWarps; ++i) value += partial[threadIdx.x][i];
        const int output_row = blockIdx.x * RowsPerCta + threadIdx.x;
        if (output_row < n) {
            if constexpr (Add) {
                value += __bfloat162float(out[output_row]);
            }
            out[output_row] = __float2bfloat16_rn(value);
        }
    }
}

template <int Bits, int RowsPerCta = 4>
__global__ void turing_rowsplit_swiglu_kernel(
    const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
    __nv_bfloat16* out, int intermediate, int k) {
    const int row = blockIdx.x * RowsPerCta + (threadIdx.x >> 5);
    if (row >= intermediate) return;
    constexpr int Group = Bits == 8 ? 32 : 64;
    constexpr int Packing = Bits == 8 ? 1 : 2;
    const auto gate_offset = static_cast<std::int64_t>(row) * k;
    const auto up_offset = static_cast<std::int64_t>(row + intermediate) * k;
    const float gate = turing_rowsplit_dot<Bits, 1>(
        x, codes + gate_offset / Packing, nullptr, scales + gate_offset / Group * 2, k, 0);
    const float up = turing_rowsplit_dot<Bits, 1>(
        x, codes + up_offset / Packing, nullptr, scales + up_offset / Group * 2, k, 0);
    if ((threadIdx.x & 31) == 0) out[row] = __float2bfloat16_rn(silu(gate) * up);
}

} // namespace ninfer::ops::detail
