#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/peer_copy.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

struct CopyCase {
    std::size_t width;
    std::size_t rows;
    std::size_t source_pitch;
    std::size_t destination_pitch;
    std::size_t source_offset;
    std::size_t destination_offset;
    bool flat;
};

int run_case(const ninfer::DeviceContext& source_device,
             const ninfer::DeviceContext& destination_device, CopyCase shape) {
    const std::size_t source_bytes = shape.source_offset + shape.rows * shape.source_pitch + 32;
    const std::size_t destination_bytes =
        shape.destination_offset + shape.rows * shape.destination_pitch + 32;
    source_device.bind_to_current_thread();
    ninfer::DeviceBuffer source(source_bytes);
    destination_device.bind_to_current_thread();
    ninfer::DeviceBuffer destination(destination_bytes);
    std::vector<unsigned char> input(source_bytes);
    std::vector<unsigned char> expected(destination_bytes, 0xa5);
    std::vector<unsigned char> actual(destination_bytes);
    const auto copy = [&] {
        void* dst = static_cast<unsigned char*>(destination.p) + shape.destination_offset;
        const void* src = static_cast<unsigned char*>(source.p) + shape.source_offset;
        if (shape.flat) {
            CUDA_CHECK(ninfer::copy_peer_kernel_async(dst, src, shape.width,
                                                       destination_device.stream));
        } else {
            CUDA_CHECK(ninfer::copy_peer_kernel_2d_async(
                dst, shape.destination_pitch, src, shape.source_pitch, shape.width, shape.rows,
                destination_device.stream));
        }
    };
    ninfer::DecodeGraphDefinition definition;
    definition.capture(destination_device.stream, copy);
    ninfer::DecodeGraphExecutable graph;
    graph.instantiate(definition);
    for (int replay = 0; replay < 4; ++replay) {
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<unsigned char>((i * 173 + replay * 79) ^ (i >> 8));
        std::fill(expected.begin(), expected.end(), 0xa5);
        source.copy_from_host(input.data(), input.size());
        destination.copy_from_host(expected.data(), expected.size());
        for (std::size_t row = 0; row < shape.rows; ++row) {
            std::copy_n(input.begin() + shape.source_offset + row * shape.source_pitch,
                        shape.width,
                        expected.begin() + shape.destination_offset + row * shape.destination_pitch);
        }
        if (replay == 0) copy();
        else graph.launch(destination_device.stream);
        destination_device.synchronize();
        destination.copy_to_host(actual.data(), actual.size());
        if (actual != expected) {
            std::cerr << "peer copy mismatch: source=" << source_device.device
                      << " destination=" << destination_device.device << " width=" << shape.width
                      << " rows=" << shape.rows << " replay=" << replay << '\n';
            return 1;
        }
        std::vector<unsigned char> unchanged(input.size());
        source.copy_to_host(unchanged.data(), unchanged.size());
        if (unchanged != input) {
            std::cerr << "peer copy modified its source\n";
            return 1;
        }
    }
    return 0;
}

int run_direction(int source_id, int destination_id) {
    ninfer::DeviceContext source(source_id);
    ninfer::DeviceContext destination(destination_id);
    if (source_id != destination_id) {
        int accessible = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, destination_id, source_id));
        if (!accessible) {
            std::cerr << "SKIP: peer access unavailable\n";
            return 77;
        }
        const auto status = cudaDeviceEnablePeerAccess(source_id, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
        else CUDA_CHECK(status);
    }
    int failures = 0;
    ninfer::DeviceBuffer invalid_storage(16);
    if (ninfer::copy_peer_kernel_2d_async(invalid_storage.p, 8, invalid_storage.p, 16, 16, 1,
                                         destination.stream) != cudaErrorInvalidValue ||
        ninfer::copy_peer_kernel_2d_async(
            invalid_storage.p, 16, invalid_storage.p, 16, 16,
            std::numeric_limits<std::size_t>::max(), destination.stream) != cudaErrorInvalidValue)
        return 1;
    for (const std::size_t width : {1, 15, 16, 17, 127, 128, 129, 511, 512, 513, 81920}) {
        failures += run_case(source, destination, {width, 1, width, width, 0, 0, true});
        failures += run_case(source, destination, {width, 1, width, width, 1, 3, true});
    }
    for (const CopyCase shape : {
             CopyCase{31, 8, 48, 64, 0, 0, false},
             CopyCase{10240, 8, 10256, 17408, 0, 0, false},
             CopyCase{129, 7, 145, 193, 1, 3, false},
             CopyCase{0, 8, 16, 32, 0, 0, false},
         }) failures += run_case(source, destination, shape);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const bool peer = argc == 2 && std::string_view(argv[1]) == "--peer";
    if (argc != 1 && !peer) {
        std::cerr << "usage: ninfer_peer_copy_test [--peer]\n";
        return 2;
    }
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices < (peer ? 2 : 1))) return 77;
    CUDA_CHECK(status);
    try {
        CUDA_CHECK(ninfer::copy_peer_kernel_async(nullptr, nullptr, 0, nullptr));
        CUDA_CHECK(ninfer::copy_peer_kernel_2d_async(nullptr, 0, nullptr, 0, 1, 0, nullptr));
        if (ninfer::copy_peer_kernel_async(nullptr, nullptr, 1, nullptr) != cudaErrorInvalidValue)
            return 1;
        int failures = run_direction(0, peer ? 1 : 0);
        if (failures == 77) return 77;
        if (peer) {
            const int reverse = run_direction(1, 0);
            if (reverse == 77) return 77;
            failures += reverse;
        }
        std::cout << (failures ? "FAIL" : "PASS") << " exact peer kernel copies\n";
        return failures != 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
