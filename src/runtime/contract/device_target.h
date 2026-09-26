#pragma once

#include <stdexcept>
#include <string>

namespace ninfer::runtime {

#if defined(NINFER_SM75)
inline constexpr int kCompiledComputeCapability = 75;
inline constexpr const char* kCompiledDeviceTarget = "SM75";
#else
inline constexpr int kCompiledComputeCapability = 120;
inline constexpr const char* kCompiledDeviceTarget = "SM120a";
#endif

inline void require_compiled_device(int compute_capability) {
    if (compute_capability != kCompiledComputeCapability) {
        throw std::invalid_argument(
            std::string("This NInfer binary targets ") + kCompiledDeviceTarget +
            ", but the selected CUDA device is SM" + std::to_string(compute_capability) +
            ". Use the rtx8000/rtx8000-dev build presets for SM75, or release/dev for SM120a.");
    }
}

} // namespace ninfer::runtime
