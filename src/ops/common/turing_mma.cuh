#pragma once

#include "ops/common/memory.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops::turing {

__device__ __forceinline__ unsigned half_pair(float x, float y) {
    const __half2 value = __floats2half2_rn(x, y);
    return load_vec<unsigned>(&value);
}

__device__ __forceinline__ unsigned bf16_to_half_pair(unsigned value) {
    return half_pair(__uint_as_float(value << 16), __uint_as_float(value & 0xffff0000u));
}

__device__ __forceinline__ void mma_half_k8(float& c0, float& c1, float& c2, float& c3,
                                           unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ bool bf16_pair_outside_half(unsigned value) {
    const unsigned low = value & 0x7fffu;
    const unsigned high = (value >> 16) & 0x7fffu;
    return (low != 0 && low - 0x3880u > 0x477fu - 0x3880u) ||
           (high != 0 && high - 0x3880u > 0x477fu - 0x3880u);
}

__device__ __forceinline__ void mma_bf16_k16(
    float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    constexpr unsigned mask = 0xffffffffu;
    const bool outside = bf16_pair_outside_half(a0) || bf16_pair_outside_half(a1) ||
                         bf16_pair_outside_half(a2) || bf16_pair_outside_half(a3) ||
                         bf16_pair_outside_half(b0) || bf16_pair_outside_half(b1);
    if (__any_sync(mask, outside)) {
        const int lane = threadIdx.x & 31;
        const int row = lane >> 2;
        const int column = lane & 3;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int inner_lane = (k & 7) >> 1;
            const unsigned top =
                __shfl_sync(mask, k < 8 ? a0 : a2, row * 4 + inner_lane);
            const unsigned bottom =
                __shfl_sync(mask, k < 8 ? a1 : a3, row * 4 + inner_lane);
            const unsigned left =
                __shfl_sync(mask, k < 8 ? b0 : b1, column * 8 + inner_lane);
            const unsigned right =
                __shfl_sync(mask, k < 8 ? b0 : b1, column * 8 + 4 + inner_lane);
            const int shift = (k & 1) * 16;
            const float av0 = __uint_as_float(((top >> shift) & 0xffffu) << 16);
            const float av1 = __uint_as_float(((bottom >> shift) & 0xffffu) << 16);
            const float bv0 = __uint_as_float(((left >> shift) & 0xffffu) << 16);
            const float bv1 = __uint_as_float(((right >> shift) & 0xffffu) << 16);
            c0 = fmaf(av0, bv0, c0);
            c1 = fmaf(av0, bv1, c1);
            c2 = fmaf(av1, bv0, c2);
            c3 = fmaf(av1, bv1, c3);
        }
        return;
    }
    mma_half_k8(c0, c1, c2, c3, bf16_to_half_pair(a0), bf16_to_half_pair(a1),
                bf16_to_half_pair(b0));
    mma_half_k8(c0, c1, c2, c3, bf16_to_half_pair(a2), bf16_to_half_pair(a3),
                bf16_to_half_pair(b1));
}

__device__ __forceinline__ void mma_int8_k16(int& c0, int& c1, unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a), "r"(b));
}

__device__ __forceinline__ void mma_float_k8(float& c0, float& c1, float& c2, float& c3,
                                            float a0, float a1, float a2, float a3,
                                            float b0, float b1) {
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int column = lane & 3;
    constexpr unsigned mask = 0xffffffffu;
    const auto outside_half = [](float x) {
        return x != 0.0f && (fabsf(x) < 0x1p-14f || fabsf(x) > 65504.0f);
    };
    if (__any_sync(mask, outside_half(a0) || outside_half(a1) || outside_half(a2) ||
                            outside_half(a3) || outside_half(b0) || outside_half(b1))) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float top = __shfl_sync(mask, k < 4 ? a0 : a2, group * 4 + (k & 3));
            const float bottom = __shfl_sync(mask, k < 4 ? a1 : a3, group * 4 + (k & 3));
            const float left = __shfl_sync(mask, k < 4 ? b0 : b1, column * 8 + (k & 3));
            const float right = __shfl_sync(mask, k < 4 ? b0 : b1, column * 8 + 4 + (k & 3));
            c0 = fmaf(top, left, c0);
            c1 = fmaf(top, right, c1);
            c2 = fmaf(bottom, left, c2);
            c3 = fmaf(bottom, right, c3);
        }
        return;
    }

    // Repack TF32's four scalar lanes into Turing's two adjacent FP16 lanes.
    const int source = group * 4 + 2 * (column & 1);
    const float top0 = __shfl_sync(mask, a0, source);
    const float top1 = __shfl_sync(mask, a0, source + 1);
    const float top2 = __shfl_sync(mask, a2, source);
    const float top3 = __shfl_sync(mask, a2, source + 1);
    const float bottom0 = __shfl_sync(mask, a1, source);
    const float bottom1 = __shfl_sync(mask, a1, source + 1);
    const float bottom2 = __shfl_sync(mask, a3, source);
    const float bottom3 = __shfl_sync(mask, a3, source + 1);
    const float right0 = __shfl_sync(mask, b0, source);
    const float right1 = __shfl_sync(mask, b0, source + 1);
    const float right2 = __shfl_sync(mask, b1, source);
    const float right3 = __shfl_sync(mask, b1, source + 1);
    const float x0 = column < 2 ? top0 : top2;
    const float x1 = column < 2 ? top1 : top3;
    const float x2 = column < 2 ? bottom0 : bottom2;
    const float x3 = column < 2 ? bottom1 : bottom3;
    const float y0 = column < 2 ? right0 : right2;
    const float y1 = column < 2 ? right1 : right3;
    const unsigned ah0 = half_pair(x0, x1);
    const unsigned ah1 = half_pair(x2, x3);
    const unsigned bh = half_pair(y0, y1);
    const float2 av0 = __half22float2(load_vec<__half2>(&ah0));
    const float2 av1 = __half22float2(load_vec<__half2>(&ah1));
    const float2 bv = __half22float2(load_vec<__half2>(&bh));
    const unsigned al0 = half_pair(x0 - av0.x, x1 - av0.y);
    const unsigned al1 = half_pair(x2 - av1.x, x3 - av1.y);
    const unsigned bl = half_pair(y0 - bv.x, y1 - bv.y);
    // GDN's FP32 state and triangular solve cannot cross a single FP16 rounding boundary.
    // Retain both halves of each operand and accumulate all four products in FP32.
    mma_half_k8(c0, c1, c2, c3, al0, al1, bl);
    mma_half_k8(c0, c1, c2, c3, ah0, ah1, bl);
    mma_half_k8(c0, c1, c2, c3, al0, al1, bh);
    mma_half_k8(c0, c1, c2, c3, ah0, ah1, bh);
}

} // namespace ninfer::ops::turing
