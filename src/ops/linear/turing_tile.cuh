#pragma once

#include "ops/linear/a16_operand.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float turing_scale_activation(float value, int exponent) {
    if (exponent >= -126 && exponent <= 127)
        return __fmul_rn(value, __uint_as_float(static_cast<unsigned>(exponent + 127) << 23));
    return ldexpf(value, exponent);
}

struct TuringLinearRows {
    int begin;
    int rows;

    __device__ __forceinline__ int operator()(int row) const { return begin + row; }
    __device__ __forceinline__ bool valid(int row) const { return row < rows; }
};

template <int HalfTile>
struct TuringPairedRows {
    int begin;
    int intermediate;

    __device__ __forceinline__ int operator()(int row) const {
        const int local = begin + row % HalfTile;
        return local < intermediate ? local + (row / HalfTile) * intermediate : -1;
    }
    __device__ __forceinline__ bool valid(int row) const { return row >= 0; }
};

template <int Bits>
__device__ __forceinline__ float turing_weight_fp32(
    const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
    int padded_k, int row, int column) {
    const auto index = static_cast<std::int64_t>(row) * padded_k + column;
    int value;
    if constexpr (Bits == 8) {
        value = static_cast<std::int8_t>(codes[index]);
    } else {
        unsigned code = (codes[index / 2] >> ((index & 1) * 4)) & 15u;
        if constexpr (Bits > 4) {
            constexpr int HighBits = Bits - 4;
            constexpr int PerByte = 8 / HighBits;
            code |= ((high[index / PerByte] >> ((index % PerByte) * HighBits)) &
                     ((1u << HighBits) - 1)) << 4;
        }
        constexpr int Sign = 1 << (Bits - 1);
        value = (static_cast<int>(code) ^ Sign) - Sign;
    }
    constexpr int Group = Bits == 8 ? 32 : 64;
    return float(value) * __half2float(reinterpret_cast<const __half*>(scales)[index / Group]);
}

template <int Bits, int BM, int BN, int WM, int WN, bool ExternalMasks = false,
          bool PreparedActivation = false, class RowMap>
__device__ __forceinline__ void turing_rowsplit_tile(
    const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* high,
    const std::uint8_t* scales, A16Operand* weights, A16Operand* activations,
    float (&acc)[WM / 16][WN / 8][4], int k, int padded_k, int tokens, int token_begin,
    RowMap row_map, std::uint8_t* exception_storage = nullptr,
    const __half* prepared = nullptr, const std::uint8_t* prepared_masks = nullptr,
    const int* exponents = nullptr) {
    constexpr int BK = 64;
    constexpr int MT = WM / 16;
    constexpr int NT = WN / 8;
    constexpr int WarpsN = BN / WN;
    constexpr int Threads = (BM / WM) * WarpsN * 32;
    constexpr int Group = Bits == 8 ? 32 : 64;
    // Sparse FP32 corrections retain the BF16 exponent range without abandoning the tensor tile.
    if constexpr (!ExternalMasks) {
        __shared__ __align__(8) std::uint8_t exceptions[BN * (BK / 8) + BM * (BK / Group)];
        exception_storage = exceptions;
    }
    auto* activation_exceptions =
        reinterpret_cast<std::uint8_t (*)[BK / 8]>(exception_storage);
    auto* weight_exceptions =
        reinterpret_cast<std::uint8_t (*)[BK / Group]>(exception_storage + BN * (BK / 8));
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / WarpsN;
    const int warp_n = warp % WarpsN;
    const int groups = padded_k / Group;
    const auto swizzle = [](int row, int col) {
        return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
    };

    for (int k0 = 0; k0 < padded_k; k0 += BK) {
        bool outside = false;
        for (int item = tid; item < BN * (BK / 8); item += Threads) {
            const int token = item / (BK / 8);
            const int column = (item % (BK / 8)) * 8;
            const int global_token = token_begin + token;
            const int valid = global_token < tokens && k0 + column < k
                                  ? min(8, k - k0 - column)
                                  : 0;
            unsigned exceptional;
            if constexpr (PreparedActivation) {
                const std::int64_t index =
                    static_cast<std::int64_t>(global_token) * k + k0 + column;
                exceptional = valid ? prepared_masks[index / 8] : 0;
                store_vec(activations + token * BK + swizzle(token, column),
                          valid ? load_vec<uint4>(prepared + index) : make_uint4(0, 0, 0, 0));
            } else {
                exceptional = a16_stage_activation(
                    activations + token * BK + swizzle(token, column),
                    x + static_cast<std::int64_t>(valid ? global_token : 0) * k +
                        (valid ? k0 + column : 0),
                    valid);
            }
            activation_exceptions[token][column / 8] = static_cast<std::uint8_t>(exceptional);
            outside |= exceptional != 0;
        }
        for (int item = tid; item < BM * (BK / 8); item += Threads) {
            const int local_row = item / (BK / 8);
            const int column = (item % (BK / 8)) * 8;
            const int row = row_map(local_row);
            union alignas(16) Decoded {
                uint4 bits;
                __half2 pair[4];
            } decoded;
            decoded.bits = make_uint4(0, 0, 0, 0);
            bool overflow = false;
            if (row_map.valid(row)) {
                const std::int64_t index = static_cast<std::int64_t>(row) * padded_k + k0 + column;
                const __half scale = reinterpret_cast<const __half*>(scales)[
                    static_cast<std::int64_t>(row) * groups + (k0 + column) / Group];
                overflow = (__half_as_ushort(scale) & 0x7fffu) >
                           (0x7bffu - ((Bits - 1) << 10));
                outside |= overflow;
                const __half2 scale_pair = __half2half2(scale);
                if constexpr (Bits == 4) {
                    const unsigned word = *reinterpret_cast<const unsigned*>(codes + index / 2);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const unsigned byte = word >> (pair * 8);
                        const unsigned lanes =
                            ((byte & 15u) | ((byte & 240u) << 12)) ^ 0x00080008u;
                        const unsigned biased = lanes | 0x64006400u;
                        const __half2 integer = __hsub2(
                            *reinterpret_cast<const __half2*>(&biased),
                            __half2half2(__ushort_as_half(0x6408)));
                        decoded.pair[pair] = __hmul2(integer, scale_pair);
                    }
                } else if constexpr (Bits == 8) {
                    const uint2 words = load_vec<uint2>(codes + index);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const unsigned word =
                            (pair < 2 ? words.x : words.y) >> ((pair & 1) * 16);
                        const unsigned biased =
                            __byte_perm(word ^ 0x8080u, 0x64006400u, 0x7150);
                        const __half2 integer = __hsub2(
                            *reinterpret_cast<const __half2*>(&biased),
                            __half2half2(__ushort_as_half(0x6480)));
                        decoded.pair[pair] = __hmul2(integer, scale_pair);
                    }
                } else {
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        int q0, q1;
                        const unsigned packed = codes[index / 2 + pair];
                        const std::int64_t value_index = index + pair * 2;
                        constexpr int HighBits = Bits - 4;
                        constexpr int PerByte = 8 / HighBits;
                        const unsigned upper = high[value_index / PerByte];
                        const int shift = (value_index % PerByte) * HighBits;
                        constexpr int Mask = (1 << HighBits) - 1;
                        constexpr int Sign = 1 << (Bits - 1);
                        q0 = ((packed & 15) | (((upper >> shift) & Mask) << 4)) ^ Sign;
                        q1 = ((packed >> 4) |
                              (((upper >> (shift + HighBits)) & Mask) << 4)) ^ Sign;
                        q0 -= Sign;
                        q1 -= Sign;
                        decoded.pair[pair] =
                            __hmul2(__floats2half2_rn(float(q0), float(q1)), scale_pair);
                    }
                }
            }
            if (overflow) decoded.bits = make_uint4(0, 0, 0, 0);
            if ((column % Group) == 0)
                weight_exceptions[local_row][column / Group] = static_cast<std::uint8_t>(overflow);
            store_vec(weights + local_row * BK + swizzle(local_row, column), decoded.bits);
        }
        const bool exceptional_tile = __syncthreads_or(outside);

#pragma unroll
        for (int step = 0; step < BK; step += 16) {
            unsigned a[MT][4], b[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_m * WM + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int column = step + (lane >> 4) * 8;
                ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                            smem_addr(weights + row * BK + swizzle(row, column)));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_n * WN + ni * 8 + (lane & 7);
                const int column = step + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(b[ni][0], b[ni][1],
                            smem_addr(activations + row * BK + swizzle(row, column)));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    a16_mma(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                            a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[ni][0], b[ni][1]);
                }
            }
        }
        if (exceptional_tile) {
            const auto correct = [&](float& value, int local_row, int local_token) {
                const int row = row_map(local_row);
                const int token = token_begin + local_token;
                if (!row_map.valid(row) || token >= tokens) return;
                auto mask = load_vec<unsigned long long>(activation_exceptions[local_token]);
#pragma unroll
                for (int group = 0; group < BK / Group; ++group) {
                    if (weight_exceptions[local_row][group])
                        mask |= (~0ull >> (64 - Group)) << (group * Group);
                }
                while (mask) {
                    const int column = k0 + __ffsll(static_cast<long long>(mask)) - 1;
                    mask &= mask - 1;
                    if (column < k) {
                        const float weight =
                            turing_weight_fp32<Bits>(codes, high, scales, padded_k, row, column);
                        float activation =
                            __bfloat162float(x[static_cast<std::int64_t>(token) * k + column]);
                        if constexpr (PreparedActivation)
                            activation = turing_scale_activation(activation, exponents[token]);
                        value = fmaf(weight, activation, value);
                    }
                }
            };
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_m * WM + mi * 16 + (lane >> 2);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int token = warp_n * WN + ni * 8 + (lane & 3) * 2;
                    correct(acc[mi][ni][0], row, token);
                    correct(acc[mi][ni][1], row, token + 1);
                    correct(acc[mi][ni][2], row + 8, token);
                    correct(acc[mi][ni][3], row + 8, token + 1);
                }
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail
