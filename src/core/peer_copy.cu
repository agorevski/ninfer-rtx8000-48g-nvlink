#include "core/peer_copy.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer {
namespace {

template <bool Vectorized>
__global__ void peer_copy_kernel(
    unsigned char* destination, std::size_t destination_pitch, const unsigned char* source,
    std::size_t source_pitch, std::size_t row_bytes, std::size_t units_per_row,
    std::size_t units) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t unit = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         unit < units; unit += stride) {
        const std::size_t row = unit / units_per_row;
        const std::size_t column = unit - row * units_per_row;
        auto* dst = destination + row * destination_pitch;
        const auto* src = source + row * source_pitch;
        if constexpr (Vectorized) {
            const std::size_t offset = column * sizeof(uint4);
            if (row_bytes - offset >= sizeof(uint4)) {
                *reinterpret_cast<uint4*>(dst + offset) =
                    *reinterpret_cast<const uint4*>(src + offset);
            } else {
                for (std::size_t byte = offset; byte < row_bytes; ++byte) dst[byte] = src[byte];
            }
        } else {
            dst[column] = src[column];
        }
    }
}

bool valid_extent(std::size_t pitch, std::size_t row_bytes, std::size_t rows) {
    return pitch >= row_bytes &&
           rows - 1 <= (std::numeric_limits<std::size_t>::max() - row_bytes) / pitch;
}

} // namespace

cudaError_t copy_peer_kernel_2d_async(
    void* destination, std::size_t destination_pitch, const void* source,
    std::size_t source_pitch, std::size_t row_bytes, std::size_t rows, cudaStream_t stream) {
    if (row_bytes == 0 || rows == 0) return cudaSuccess;
    if (destination == nullptr || source == nullptr ||
        !valid_extent(destination_pitch, row_bytes, rows) ||
        !valid_extent(source_pitch, row_bytes, rows)) return cudaErrorInvalidValue;

    const bool aligned = ((reinterpret_cast<std::uintptr_t>(destination) |
                           reinterpret_cast<std::uintptr_t>(source) |
                           (rows > 1 ? destination_pitch | source_pitch : 0)) &
                          (alignof(uint4) - 1)) == 0;
    const std::size_t units_per_row =
        aligned ? row_bytes / sizeof(uint4) + (row_bytes % sizeof(uint4) != 0) : row_bytes;
    const std::size_t units = units_per_row * rows;
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>(
        std::min<std::size_t>(units / threads + (units % threads != 0), 65535));
    if (aligned) {
        peer_copy_kernel<true><<<blocks, threads, 0, stream>>>(
            static_cast<unsigned char*>(destination), destination_pitch,
            static_cast<const unsigned char*>(source), source_pitch, row_bytes, units_per_row,
            units);
    } else {
        peer_copy_kernel<false><<<blocks, threads, 0, stream>>>(
            static_cast<unsigned char*>(destination), destination_pitch,
            static_cast<const unsigned char*>(source), source_pitch, row_bytes, units_per_row,
            units);
    }
    return cudaGetLastError();
}

cudaError_t copy_peer_kernel_async(void* destination, const void* source, std::size_t bytes,
                                    cudaStream_t stream) {
    return copy_peer_kernel_2d_async(destination, bytes, source, bytes, bytes, 1, stream);
}

} // namespace ninfer
