#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        ninfer::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");

        ninfer::DeviceBuffer source(sizeof(std::uint32_t));
        ninfer::DecodeGraphDefinition copy;
        copy.capture(device.stream, [&] {
            CUDA_CHECK(ninfer::copy_device_async(storage.base(), source.p, source.bytes,
                                                  device.stream));
        });
        ninfer::DecodeGraphExecutable copy_executable;
        copy_executable.instantiate(copy);
        for (const std::uint32_t expected : {0x44444444U, 0x55555555U, 0x77777777U}) {
            source.copy_from_host(&expected, sizeof(expected));
            copy_executable.launch(device.stream);
            device.synchronize();
            failures += expect_value(storage.base(), expected, "captured device copy");
        }

        const std::array<std::uint32_t, 6> pitched_input{1, 2, 99, 3, 4, 99};
        ninfer::DeviceBuffer pitched_source(sizeof(pitched_input));
        ninfer::DeviceBuffer pitched_destination(8 * sizeof(std::uint32_t));
        pitched_source.copy_from_host(pitched_input.data(), sizeof(pitched_input));
        pitched_destination.fill(0);
        ninfer::DecodeGraphDefinition pitched_copy;
        pitched_copy.capture(device.stream, [&] {
            CUDA_CHECK(ninfer::copy_device_2d_async(
                pitched_destination.p, 4 * sizeof(std::uint32_t), pitched_source.p,
                3 * sizeof(std::uint32_t), 2 * sizeof(std::uint32_t), 2, device.stream));
        });
        ninfer::DecodeGraphExecutable pitched_executable;
        pitched_executable.instantiate(pitched_copy);
        pitched_executable.launch(device.stream);
        device.synchronize();
        std::array<std::uint32_t, 8> pitched_actual{};
        pitched_destination.copy_to_host(pitched_actual.data(), sizeof(pitched_actual));
        if (pitched_actual != std::array<std::uint32_t, 8>{1, 2, 0, 0, 3, 4, 0, 0}) {
            std::cerr << "captured pitched copy changed payload or padding\n";
            ++failures;
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
