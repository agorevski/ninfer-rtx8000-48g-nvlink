#include "runtime/contract/device_target.h"

#include <iostream>
#include <string>

int main() {
#if defined(NINFER_SM75)
    static_assert(ninfer::runtime::kCompiledComputeCapability == 75);
#else
    static_assert(ninfer::runtime::kCompiledComputeCapability == 120);
#endif
    for (const int capability : {75, 80, 86, 89, 90, 100, 120, 121}) {
        bool rejected = false;
        try {
            ninfer::runtime::require_compiled_device(capability);
        } catch (const std::invalid_argument& error) {
            rejected = true;
            const std::string message = error.what();
            if (message.find(ninfer::runtime::kCompiledDeviceTarget) == std::string::npos ||
                message.find("SM" + std::to_string(capability)) == std::string::npos ||
                message.find("rtx8000") == std::string::npos ||
                message.find("release/dev") == std::string::npos) {
                std::cerr << "wrong-build error omitted its target/device/preset guidance\n";
                return 1;
            }
        }
        if (rejected == (capability == ninfer::runtime::kCompiledComputeCapability)) {
            std::cerr << "compiled target accepted or rejected the wrong device\n";
            return 1;
        }
    }
}
