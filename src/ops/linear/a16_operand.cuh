#pragma once

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {

// Turing consumes FP16 operands directly. Public tensors and epilogue rounding stay BF16.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
using A16Operand = __half;
using A16OperandPair = __half2;
#else
using A16Operand = __nv_bfloat16;
using A16OperandPair = __nv_bfloat162;
#endif

__device__ __forceinline__ A16OperandPair a16_operand_pair(float first, float second) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __floats2half2_rn(first, second);
#else
    return __floats2bfloat162_rn(first, second);
#endif
}

// SM75 returns a mask of zero-staged values whose products need FP32 correction.
template <Cache Policy = Cache::ca>
__device__ __forceinline__ unsigned a16_stage_activation(A16Operand* destination,
                                                     const __nv_bfloat16* source,
                                                     int valid_elements = 8) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    union alignas(16) Pack {
        uint4 bits;
        __half2 pair[4];
    } result;
    unsigned exceptional = 0;
    if (valid_elements == 8) {
        const uint4 input = load_vec<uint4>(source);
        const unsigned words[4] = {input.x, input.y, input.z, input.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned low = words[i] & 0x7fffu;
            const unsigned high = (words[i] >> 16) & 0x7fffu;
            const bool exceptional0 = low != 0 && low - 0x3880u > 0x477fu - 0x3880u;
            const bool exceptional1 = high != 0 && high - 0x3880u > 0x477fu - 0x3880u;
            exceptional |= unsigned(exceptional0) << (2 * i);
            exceptional |= unsigned(exceptional1) << (2 * i + 1);
            result.pair[i] = __floats2half2_rn(
                exceptional0 ? 0.0f : __uint_as_float(words[i] << 16),
                exceptional1 ? 0.0f : __uint_as_float(words[i] & 0xffff0000u));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float first = 2 * i < valid_elements ? __bfloat162float(source[2 * i]) : 0.0f;
            const float second =
                2 * i + 1 < valid_elements ? __bfloat162float(source[2 * i + 1]) : 0.0f;
            const bool exceptional0 = first != 0.0f &&
                                      (fabsf(first) < 0x1p-14f || fabsf(first) > 65504.0f);
            const bool exceptional1 = second != 0.0f &&
                                      (fabsf(second) < 0x1p-14f || fabsf(second) > 65504.0f);
            exceptional |= unsigned(exceptional0) << (2 * i);
            exceptional |= unsigned(exceptional1) << (2 * i + 1);
            result.pair[i] = __floats2half2_rn(exceptional0 ? 0.0f : first,
                                               exceptional1 ? 0.0f : second);
        }
    }
    store_vec(destination, result.bits);
    return exceptional;
#else
    if (valid_elements == 8) cp_async<16, Policy>(destination, source);
    else cp_async_zfill<16, Policy>(destination, source, valid_elements * 2);
    return 0;
#endif
}

__device__ __forceinline__ void a16_mma(float& c0, float& c1, float& c2, float& c3,
                                        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a2), "r"(a3), "r"(b1));
#else
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
}

} // namespace ninfer::ops::detail
