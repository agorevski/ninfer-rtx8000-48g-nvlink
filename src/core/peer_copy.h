#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer {

// Raw, non-overlapping byte copies submitted as CUDA kernel nodes. The current device and
// stream must own the destination; a peer source requires an already enabled UVA mapping.
// The caller owns ordering, allocation lifetime and peer enablement. Padding is untouched.
[[nodiscard]] cudaError_t copy_peer_kernel_async(void* destination, const void* source,
                                                 std::size_t bytes, cudaStream_t stream);
[[nodiscard]] cudaError_t copy_peer_kernel_2d_async(
    void* destination, std::size_t destination_pitch, const void* source,
    std::size_t source_pitch, std::size_t row_bytes, std::size_t rows, cudaStream_t stream);

} // namespace ninfer
