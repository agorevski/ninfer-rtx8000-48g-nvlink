#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"

#include <array>
#include <cmath>
#include <iostream>

namespace {

__device__ unsigned pack_int8(const float* values) {
    unsigned result = 0;
    for (int i = 0; i < 4; ++i) {
        result |= static_cast<unsigned>(static_cast<unsigned char>(static_cast<int>(values[i])))
                  << (8 * i);
    }
    return result;
}

__global__ void product(const float* a, const float* b, float* c, int mode) {
    const int group = threadIdx.x >> 2;
    const int lane = threadIdx.x & 3;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (mode == 2) {
        unsigned ar[4], br[2];
        for (int half = 0; half < 2; ++half) {
            for (int row = 0; row < 2; ++row)
                ar[half * 2 + row] = pack_int8(a + (group + row * 8) * 32 + half * 16 + lane * 4);
            float values[4];
            for (int i = 0; i < 4; ++i) values[i] = b[(half * 16 + lane * 4 + i) * 8 + group];
            br[half] = pack_int8(values);
        }
        int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        ninfer::ops::mma_s8(i0, i1, i2, i3, ar[0], ar[1], ar[2], ar[3], br[0], br[1]);
        c0 = i0; c1 = i1; c2 = i2; c3 = i3;
    } else if (mode >= 3) {
        ninfer::ops::mma_tf32(c0, c1, c2, c3, a[group * 8 + lane],
                              a[(group + 8) * 8 + lane], a[group * 8 + lane + 4],
                              a[(group + 8) * 8 + lane + 4], b[lane * 8 + group],
                              b[(lane + 4) * 8 + group]);
    } else {
        unsigned ar[4], br[2];
        for (int half = 0; half < 2; ++half) {
            for (int row = 0; row < 2; ++row) {
                const int at = (group + row * 8) * 16 + half * 8 + lane * 2;
                ar[half * 2 + row] = mode == 0 ? ninfer::ops::pack_bf16x2(a[at], a[at + 1])
                                               : ninfer::ops::pack_f16x2(a[at], a[at + 1]);
            }
            const int at = (half * 8 + lane * 2) * 8 + group;
            br[half] = mode == 0 ? ninfer::ops::pack_bf16x2(b[at], b[at + 8])
                                  : ninfer::ops::pack_f16x2(b[at], b[at + 8]);
        }
        if (mode == 0)
            ninfer::ops::mma_bf16(c0, c1, c2, c3, ar[0], ar[1], ar[2], ar[3], br[0], br[1]);
        else
            ninfer::ops::mma_f16(c0, c1, c2, c3, ar[0], ar[1], ar[2], ar[3], br[0], br[1]);
    }
    c[group * 8 + lane * 2] = c0;
    c[group * 8 + lane * 2 + 1] = c1;
    c[(group + 8) * 8 + lane * 2] = c2;
    c[(group + 8) * 8 + lane * 2 + 1] = c3;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    ninfer::DeviceContext device;
    std::array<float, 512> a{};
    std::array<float, 256> b{};
    std::array<float, 128> actual{};
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, sizeof(a)));
    CUDA_CHECK(cudaMalloc(&db, sizeof(b)));
    CUDA_CHECK(cudaMalloc(&dc, sizeof(actual)));
    int failures = 0;
    for (int mode = 0; mode < 6; ++mode) {
        const int k = mode == 2 ? 32 : mode >= 3 ? 8 : 16;
        for (int i = 0; i < 16 * k; ++i)
            a[i] = (i * 13 % 31 - 15) * (mode == 2 ? 1.0f : mode >= 3 ? 0.00314159f : 0.0625f);
        for (int i = 0; i < k * 8; ++i)
            b[i] = (i * 7 % 29 - 14) * (mode == 2 ? 1.0f : mode >= 3 ? 0.00628318f : 0.125f);
        if (mode == 4) a[13] = 1e-9f;
        if (mode == 5) a[13] = 131072.0f;
        CUDA_CHECK(cudaMemcpy(da, a.data(), sizeof(a), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), sizeof(b), cudaMemcpyHostToDevice));
        product<<<1, 32, 0, device.stream>>>(da, db, dc, mode);
        CUDA_CHECK(cudaGetLastError());
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(actual.data(), dc, sizeof(actual), cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 8; ++col) {
                double reference = 0;
                for (int j = 0; j < k; ++j) reference += double(a[row * k + j]) * b[j * 8 + col];
                const double error = std::abs(actual[row * 8 + col] - reference);
                worst = std::max(worst, error);
                const double tolerance = mode < 3 ? 0 : 1e-7 + 2e-7 * std::abs(reference);
                if (!std::isfinite(actual[row * 8 + col]) || error > tolerance)
                    ++failures;
            }
        }
        std::cout << "mma mode=" << mode << " max_absolute_error=" << worst << '\n';
    }
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(da));
    return failures != 0;
}
