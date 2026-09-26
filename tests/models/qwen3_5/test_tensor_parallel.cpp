#include "artifact/materializer.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/execution/tensor_parallel.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;
namespace fixture = ninfer::test::quantized_weight;

float value(std::uint16_t word) {
    return std::bit_cast<float>(std::uint32_t(word) << 16);
}

std::uint16_t bf16(float x) {
    const auto bits = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

WeightView rows(const WeightParent& parent, int begin, int count) {
    const auto k = parent.geometry.shape[1];
    return {{static_cast<std::uint64_t>(count), k},
            {{&parent, begin * k, (begin + count) * k}}};
}

void check_row_copy(const WeightView& source, const artifact::MaterializedWeightRows& copy) {
    std::vector<std::byte> payload(copy.storage.bytes);
    copy.storage.copy_to_host(payload.data(), payload.size());
    const auto planes = weight_row_planes(contiguous_weight_region(source));
    const auto& g = copy.parent.geometry;
    if (std::memcmp(payload.data(), planes.codes, g.code_bytes) ||
        (g.high_bytes &&
         std::memcmp(payload.data() + g.high_offset, planes.high, g.high_bytes)) ||
        (g.scale_bytes &&
         std::memcmp(payload.data() + g.scale_offset, planes.scales, g.scale_bytes))) {
        throw std::runtime_error("peer row materialization changed stored weight planes");
    }
}

void check_materialization(DeviceContext& primary, DeviceContext& peer) {
    for (const auto format : {QType::BF16, QType::Q4_G64_FP16, QType::Q5_G64_FP16,
                              QType::Q8_G32_FP16}) {
        const std::array<std::uint64_t, 2> shape{512, 128};
        const auto layout = format == QType::BF16 ? QuantLayout::Contiguous : QuantLayout::RowSplit;
        const auto geometry = weight_geometry(format, layout, shape);
        std::vector<std::byte> payload(geometry.bytes);
        for (std::size_t i = 0; i < payload.size(); ++i) {
            payload[i] = std::byte((i * 31 + i / 251) & 255);
        }
        DeviceBuffer storage(payload.size());
        storage.copy_from_host(payload.data(), payload.size());
        const WeightParent source{geometry, static_cast<const std::byte*>(storage.p)};
        const WeightParent host{geometry, payload.data()};
        auto copy = artifact::materialize_weight_rows(rows(source, 192, 128), peer);
        {
            DeviceGuard guard(peer.device);
            check_row_copy(rows(host, 192, 128), *copy);
        }
        copy.reset();
        int current = -1;
        CUDA_CHECK(cudaGetDevice(&current));
        if (current != primary.device) {
            throw std::runtime_error("peer row materialization leaked device binding");
        }
    }
}

void check_constructor_binding(DeviceContext& primary, DeviceContext& peer) {
    for (const int caller : {primary.device, peer.device}) {
        DeviceGuard caller_guard(caller);
        {
            TensorParallelProjections execution(primary, peer.device, 256);
            int current = -1;
            CUDA_CHECK(cudaGetDevice(&current));
            if (current != caller) {
                throw std::runtime_error("tensor parallel constructor changed caller device");
            }
        }
        int current = -1;
        CUDA_CHECK(cudaGetDevice(&current));
        if (current != caller) {
            throw std::runtime_error("tensor parallel destructor changed caller device");
        }
    }
}

void compare(const std::vector<std::uint16_t>& actual, const std::vector<double>& expected,
             const char* label, bool swiglu_seam = false) {
    if (swiglu_seam) {
        std::vector<double> decoded(actual.size());
        std::transform(actual.begin(), actual.end(), decoded.begin(), value);
        if (ninfer::test::verify_reduction(label, decoded, expected,
                                           {3.3e-3, 5.0e-3, 6.3e-3})) {
            throw std::runtime_error("tensor parallel composition failed LinearSwiGLU A16 criterion");
        }
        return;
    }
    double error2 = 0;
    double reference2 = 0;
    double max_error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto x = value(actual[i]);
        if (!std::isfinite(x)) { throw std::runtime_error("non-finite tensor parallel output"); }
        error2 += std::pow(x - expected[i], 2);
        reference2 += expected[i] * expected[i];
        max_error = std::max(max_error, std::abs(x - expected[i]));
    }
    const auto rms = std::sqrt(reference2 / expected.size());
    const auto relative = std::sqrt(error2 / std::max(reference2, 1e-30));
    // Two BF16 projection results feed SiLU; the product and residual have BF16 storage.
    // A cancellation-aware gross criterion accompanies the aggregate reduction error.
    if (relative > 0.008) {
        throw std::runtime_error(std::string(label) + ": relative L2 " + std::to_string(relative));
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(value(actual[i]) - expected[i]) > 0.03 * (rms + std::abs(expected[i]))) {
            throw std::runtime_error(std::string(label) + ": gross numerical error");
        }
    }
    std::cout << label << " relative_l2=" << relative << " max_abs=" << max_error << '\n';
}

void projection_case(DeviceContext& primary, DeviceContext& peer, QType format,
                     int output_rows, int input_rows, bool add) {
    const auto packed = fixture::make_patterned_weight(
        format, output_rows, input_rows, 101,
        {.row_split_scale = fixture::RowSplitScalePattern::Small,
         .row_split_codes = fixture::RowSplitCodePattern::Hashed});
    const std::array<std::uint64_t, 2> shape{std::uint64_t(output_rows), std::uint64_t(input_rows)};
    const auto geometry = weight_geometry(format, QuantLayout::RowSplit, shape);
    DeviceGuard primary_guard(primary.device);
    DeviceBuffer weights(packed.payload.size());
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    const WeightParent parent{geometry, static_cast<const std::byte*>(weights.p)};
    auto upper = artifact::materialize_weight_rows(rows(parent, output_rows / 2, output_rows / 2),
                                                    peer);
    const TensorParallelProjectionParameters parameters{std::array{
        LinearParameters{native_weight(rows(parent, 0, output_rows / 2)), ops::LinearPolicy::A16Only},
        LinearParameters{native_weight(rows(upper->parent, 0, output_rows / 2)),
                         ops::LinearPolicy::A16Only}}};
    WorkspaceArena workspace(std::max<std::size_t>(
        256, tensor_parallel_projection_workspace_bytes(parameters, 0, add)));
    TensorParallelProjections execution(primary, peer.device,
        tensor_parallel_projection_workspace_bytes(parameters, 1, add));
    std::vector<std::uint16_t> normal(input_rows);
    for (int k = 0; k < input_rows; ++k) {
        normal[k] = bf16((int(fixture::detail::mix64(k + 57) & 255) - 128) / 512.0F);
    }
    std::vector<std::uint16_t> input(normal), staged(output_rows + 256, 0x5a5a);
    std::vector<std::uint16_t> actual(staged.size()), eager(output_rows);
    std::vector<float> represented(input_rows);
    std::vector<double> expected(output_rows), decoded(output_rows);
    DeviceBuffer input_storage(input_rows * 2);
    DeviceBuffer output_storage(staged.size() * 2);
    Tensor x(input_storage.p, DType::BF16, {input_rows, 1});
    Tensor output(static_cast<std::byte*>(output_storage.p) + 256, DType::BF16,
                  {output_rows, 1});
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    const auto project = [&] {
        if (add) { execution.project_add(x, parameters, output, workspace); }
        else { execution.project(x, parameters, output, workspace); }
    };
    const auto read_output = [&] {
        output_storage.copy_to_host(actual.data(), actual.size() * 2);
        for (int i = 0; i < 128; ++i) {
            if (actual[i] != 0x5a5a || actual[128 + output_rows + i] != 0x5a5a) {
                throw std::runtime_error("sharded output projection modified a guard region");
            }
        }
        for (int row = 0; row < output_rows; ++row) {
            decoded[row] = value(actual[128 + row]);
        }
    };
    for (int phase = 0; phase < 3; ++phase) {
        input = normal;
        if (phase == 1) { input[0] = bf16(1.0e30F); }
        if (phase == 2) {
            for (auto& word : input) { word = bf16(-value(word)); }
        }
        std::transform(input.begin(), input.end(), represented.begin(), value);
        for (int row = 0; row < output_rows; ++row) {
            const auto dot = fixture::dot_fp64(packed, row, represented.data(), input_rows);
            const auto residual = phase == 2 ? bf16(static_cast<float>(-dot))
                                             : bf16(float((row % 31) - 15) / 128.0F);
            staged[128 + row] = add ? residual : 0x7fc1;
            expected[row] = dot + (add ? value(residual) : 0.0);
        }
        input_storage.copy_from_host(input.data(), input.size() * 2);
        output_storage.copy_from_host(staged.data(), staged.size() * 2);
        project();
        primary.synchronize();
        read_output();
        if (ninfer::test::verify_reduction("sharded projection eager", decoded, expected,
                                           {1.0 / 256, 1.0 / 256, 2.0 / 256})) {
            throw std::runtime_error("sharded output projection failed the A16 oracle");
        }
        std::copy_n(actual.begin() + 128, output_rows, eager.begin());
        if (phase == 0) {
            definition.capture(primary.stream, project);
            executable.instantiate(definition);
        }
        execution.reset_memory_peak();
        output_storage.copy_from_host(staged.data(), staged.size() * 2);
        executable.launch(primary.stream);
        execution.record_graph_execution(1);
        primary.synchronize();
        read_output();
        if (!std::equal(eager.begin(), eager.end(), actual.begin() + 128) ||
            execution.workspace_summary().peak_used_bytes == 0) {
            throw std::runtime_error("sharded projection graph changed output or lost workspace usage");
        }
    }
    DecodeGraphDefinition updated;
    updated.capture(primary.stream, project);
    executable.update(updated);
    output_storage.copy_from_host(staged.data(), staged.size() * 2);
    executable.launch(primary.stream);
    primary.synchronize();
    read_output();
    if (!std::equal(eager.begin(), eager.end(), actual.begin() + 128)) {
        throw std::runtime_error("sharded projection graph update changed output");
    }
    std::cout << "PASS output projection " << output_rows << 'x' << input_rows
              << " add=" << add << " full-output FP64/range/cancellation/graph\n";
}

void run_case(DeviceContext& primary, DeviceContext& peer, int h, int intermediate, int tokens,
              QType down_type, bool swiglu_seam = false) {
    const auto pattern = fixture::PatternedWeightOptions{
        .row_split_scale = fixture::RowSplitScalePattern::Small,
        .row_split_codes = fixture::RowSplitCodePattern::Hashed};
    auto gu = fixture::make_patterned_weight(QType::Q4_G64_FP16, 2 * intermediate, h, 29, pattern);
    auto down = fixture::make_patterned_weight(down_type, h, intermediate, 37, pattern);
    if (swiglu_seam) {
        // An exact Q5 selector exposes the production FFN's actual SwiGLU product without a
        // debug hook or a second implementation. Each output row selects one channel with unit
        // weight; residual is zero. The oracle below then retains the unrounded FP64 ideal.
        std::fill(down.payload.begin(), down.payload.end(), std::uint8_t{0});
        for (std::size_t offset = 0; offset < down.scale_plane_bytes; offset += 2) {
            down.payload[down.scale_plane_offset + offset + 1] = 0x3c;
        }
        for (int row = 0; row < h; ++row) {
            const int channel = row * (intermediate - 1) / (h - 1);
            down.payload[std::size_t(row) * intermediate / 2 + channel / 2] =
                (channel & 1) ? 0x10 : 0x01;
        }
    }
    const std::array<std::uint64_t, 2> gu_shape{std::uint64_t(2 * intermediate), std::uint64_t(h)};
    const std::array<std::uint64_t, 2> down_shape{std::uint64_t(h), std::uint64_t(intermediate)};
    auto gu_geometry = weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, gu_shape);
    auto down_geometry = weight_geometry(down_type, QuantLayout::RowSplit, down_shape);
    DeviceGuard primary_guard(primary.device);
    DeviceBuffer gu_device(gu.payload.size()), down_device(down.payload.size());
    gu_device.copy_from_host(gu.payload.data(), gu.payload.size());
    down_device.copy_from_host(down.payload.data(), down.payload.size());
    WeightParent gu_parent{gu_geometry, static_cast<const std::byte*>(gu_device.p)};
    WeightParent down_parent{down_geometry, static_cast<const std::byte*>(down_device.p)};
    WeightParent gu_host{gu_geometry, reinterpret_cast<const std::byte*>(gu.payload.data())};
    WeightParent down_host{down_geometry, reinterpret_cast<const std::byte*>(down.payload.data())};
    auto remote_up = artifact::materialize_weight_rows(rows(gu_parent, intermediate, intermediate),
                                                        peer);
    auto remote_down = artifact::materialize_weight_rows(rows(down_parent, h / 2, h / 2), peer);
    check_row_copy(rows(gu_host, intermediate, intermediate), *remote_up);
    check_row_copy(rows(down_host, h / 2, h / 2), *remote_down);
    const TensorParallelDenseParameters parameters{
        {{{native_weight(rows(gu_parent, 0, intermediate)), ops::LinearPolicy::A16Only},
          {native_weight(rows(remote_up->parent, 0, intermediate)), ops::LinearPolicy::A16Only}}},
        {{{native_weight(rows(down_parent, 0, h / 2)), ops::LinearPolicy::A16Only},
          {native_weight(rows(remote_down->parent, 0, h / 2)), ops::LinearPolicy::A16Only}}}};

    WorkspaceArena workspace(tensor_parallel_ffn_workspace_bytes(parameters, 0, 1, tokens));
    TensorParallelProjections execution(primary, peer.device,
                               tensor_parallel_ffn_workspace_bytes(parameters, 1, 1, tokens));
    int constructor_device = -1;
    CUDA_CHECK(cudaGetDevice(&constructor_device));
    if (constructor_device != primary.device) {
        throw std::runtime_error("tensor parallel constructor did not retain primary binding");
    }
    std::vector<std::uint16_t> input(std::size_t(h) * tokens), residual(input.size());
    const auto dense_divisor = 512.0F * (tokens > 1 ? std::sqrt(float(h) / 4) : 1.0F);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto hash = fixture::detail::mix64(i + 13);
        input[i] = bf16((int(hash & 255) - 128) / dense_divisor);
        residual[i] = bf16((int((hash >> 8) & 127) - 64) / 4096.0F);
    }
    if (swiglu_seam) { std::fill(residual.begin(), residual.end(), std::uint16_t{0}); }
    // One dense column exercises every stored K. Sparse later columns have comparable norms,
    // keeping the independent oracle bounded without hiding column errors behind a large column.
    for (int token = 1; token < tokens; ++token) {
        std::fill_n(input.begin() + std::size_t(token) * h, h, std::uint16_t{0});
        for (int lane = 0; lane < 4; ++lane) {
            const auto k = (token * 131 + lane * 977) % h;
            input[std::size_t(token) * h + k] = bf16((lane - 1.5F) / 4);
        }
    }
    std::vector<std::uint16_t> mask_sentinels(tokens);
    for (int token = 0; token < tokens; ++token) {
        auto& sentinel = input[std::size_t(token) * h + (token * 131) % h];
        if (value(sentinel) == 0) { sentinel = bf16(0.125F); }
        mask_sentinels[token] = sentinel;
    }
    DeviceBuffer input_device(input.size() * 2), output_device(residual.size() * 2);
    input_device.copy_from_host(input.data(), input.size() * 2);
    Tensor x(input_device.p, DType::BF16, {h, tokens});
    Tensor output(output_device.p, DType::BF16, {h, tokens});

    // Decode stored codes/scales with the independent test codec, then evaluate the full
    // FP64 mathematical FFN. Only the product entering down is a semantic BF16 boundary.
    const auto decode = [](const fixture::PackedWeight& packed) {
        std::vector<double> result(std::size_t(packed.weight.n) * packed.weight.k);
        for (int row = 0; row < packed.weight.n; ++row) {
            for (int k = 0; k < packed.weight.k; ++k) {
                result[std::size_t(row) * packed.weight.k + k] =
                    fixture::logical_weight_fp64(packed, row, k);
            }
        }
        return result;
    };
    const auto gu_values = decode(gu);
    std::vector<int> oracle_rows;
    if (tokens > 8) {
        oracle_rows = {0, 1, h / 2 - 1, h / 2, h / 2 + 1, h - 1};
        for (int i = 0; i < 32; ++i) { oracle_rows.push_back(i * (h - 1) / 31); }
        std::sort(oracle_rows.begin(), oracle_rows.end());
        oracle_rows.erase(std::unique(oracle_rows.begin(), oracle_rows.end()), oracle_rows.end());
    } else {
        for (int row = 0; row < h; ++row) { oracle_rows.push_back(row); }
    }
    std::vector<double> down_values(oracle_rows.size() * intermediate);
    for (std::size_t i = 0; i < oracle_rows.size(); ++i) {
        for (int k = 0; k < intermediate; ++k) {
            down_values[i * intermediate + k] =
                fixture::logical_weight_fp64(down, oracle_rows[i], k);
        }
    }
    std::vector<double> expected(oracle_rows.size() * tokens), product(intermediate);
    const auto evaluate_oracle = [&] {
        for (int token = 0; token < tokens; ++token) {
            std::vector<std::pair<int, double>> active;
            for (int k = 0; k < h; ++k) {
                const auto activation = value(input[std::size_t(token) * h + k]);
                if (activation != 0) { active.emplace_back(k, activation); }
            }
            for (int row = 0; row < intermediate; ++row) {
                double gate = 0, up = 0;
                for (const auto& [k, activation] : active) {
                    gate += gu_values[std::size_t(row) * h + k] * activation;
                    up += gu_values[std::size_t(row + intermediate) * h + k] * activation;
                }
                const double ideal = gate / (1 + std::exp(-gate)) * up;
                product[row] = swiglu_seam ? ideal : value(bf16(static_cast<float>(ideal)));
            }
            for (std::size_t row = 0; row < oracle_rows.size(); ++row) {
                double result = value(residual[std::size_t(token) * h + oracle_rows[row]]);
                for (int k = 0; k < intermediate; ++k) {
                    result += down_values[std::size_t(row) * intermediate + k] * product[k];
                }
                expected[std::size_t(token) * oracle_rows.size() + row] = result;
            }
        }
    };
    evaluate_oracle();
    const auto samples = [&](const std::vector<std::uint16_t>& all) {
        std::vector<std::uint16_t> selected;
        selected.reserve(expected.size());
        for (const auto word : all) {
            if (!std::isfinite(value(word))) {
                throw std::runtime_error("non-finite output outside sampled oracle rows");
            }
        }
        for (int token = 0; token < tokens; ++token) {
            for (const auto row : oracle_rows) { selected.push_back(all[std::size_t(token) * h + row]); }
        }
        return selected;
    };
    std::vector<std::uint16_t> eager(residual.size()), replay(residual.size());
    output_device.copy_from_host(residual.data(), residual.size() * 2);
    const bool decode_phase = tokens <= static_cast<int>(kMaximumConcurrency);
    execution.ffn(x, parameters, output, workspace, decode_phase);
    primary.synchronize();
    output_device.copy_to_host(eager.data(), eager.size() * 2);
    compare(samples(eager), expected, "tensor-parallel eager", swiglu_seam);

    DecodeGraphDefinition graph;
    graph.capture(primary.stream, [&] {
        execution.ffn(x, parameters, output, workspace, decode_phase);
    });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    execution.reset_memory_peak();
    if (execution.workspace_summary().peak_used_bytes != 0) {
        throw std::runtime_error("peer workspace peak did not reset");
    }
    for (int iteration = 0; iteration < 3; ++iteration) {
        if (iteration != 0) {
            for (auto& word : input) { word = bf16(-value(word)); }
            for (auto& word : residual) { word = bf16(-0.5F * value(word)); }
            if (tokens >= 512) {
                for (int token = 0; token < tokens; ++token) {
                    input[std::size_t(token) * h + (token * 131) % h] =
                        iteration == 1 ? bf16(std::ldexp(1.0F, -32)) : mask_sentinels[token];
                }
            }
            input_device.copy_from_host(input.data(), input.size() * 2);
            output_device.copy_from_host(residual.data(), residual.size() * 2);
            execution.ffn(x, parameters, output, workspace, decode_phase);
            primary.synchronize();
            output_device.copy_to_host(eager.data(), eager.size() * 2);
            evaluate_oracle();
            compare(samples(eager), expected, "tensor-parallel changed-input eager", swiglu_seam);
        }
        output_device.copy_from_host(residual.data(), residual.size() * 2);
        executable.launch(primary.stream);
        if (tokens <= static_cast<int>(kMaximumConcurrency)) {
            execution.record_graph_execution(static_cast<std::uint32_t>(tokens));
            if (execution.workspace_summary().peak_used_bytes == 0) {
                throw std::runtime_error("graph-only execution omitted peer workspace usage");
            }
        }
        primary.synchronize();
        output_device.copy_to_host(replay.data(), replay.size() * 2);
        if (replay != eager) { throw std::runtime_error("multi-device graph replay differs from eager"); }
    }
    DecodeGraphDefinition updated;
    updated.capture(primary.stream, [&] {
        execution.ffn(x, parameters, output, workspace, decode_phase);
    });
    executable.update(updated);
    output_device.copy_from_host(residual.data(), residual.size() * 2);
    executable.launch(primary.stream);
    primary.synchronize();
    output_device.copy_to_host(replay.data(), replay.size() * 2);
    if (replay != eager) { throw std::runtime_error("multi-device graph update differs from eager"); }
    compare(samples(replay), expected, "tensor-parallel graph", swiglu_seam);
    int active = -1;
    CUDA_CHECK(cudaGetDevice(&active));
    if (active != primary.device) { throw std::runtime_error("peer execution leaked current device"); }
    std::cout << "PASS h=" << h << " intermediate=" << intermediate << " tokens=" << tokens
              << " oracle_rows=" << oracle_rows.size() << " swiglu_seam=" << swiglu_seam << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
            std::cout << "SKIP: two peer-connected CUDA devices required\n";
            return 77;
        }
        DeviceContext primary(0), peer(1);
        if (primary.compute_capability() != 75 || peer.compute_capability() != 75) {
            std::cout << "SKIP: tensor parallel qualification requires two SM75 devices\n";
            return 77;
        }
        for (const auto pair : {std::pair{0, 1}, std::pair{1, 0}}) {
            int available = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&available, pair.first, pair.second));
            if (!available) { return 77; }
            DeviceGuard guard(pair.first);
            const auto status = cudaDeviceEnablePeerAccess(pair.second, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); }
            else { CUDA_CHECK(status); }
        }
        primary.bind_to_current_thread();
        check_constructor_binding(primary, peer);
        check_materialization(primary, peer);
        if (argc == 2 && std::string(argv[1]) == "--projection-c1") {
            projection_case(primary, peer, QType::Q5_G64_FP16, 5120, 6144, true);
            projection_case(primary, peer, QType::Q8_G32_FP16, 248320, 5120, false);
        } else if (argc == 2 && std::string(argv[1]) == "--swiglu-seam") {
            for (const int tokens : {1, 8, 128, 1024}) {
                run_case(primary, peer, 5120, 17408, tokens, QType::Q5_G64_FP16, true);
            }
        } else if (argc == 2 && std::string(argv[1]) == "--production-shape") {
            for (const int tokens : {1, 3, 8, 128, 512, 1024}) {
                run_case(primary, peer, 5120, 17408, tokens, QType::Q5_G64_FP16);
            }
        } else {
            for (const int tokens : {1, 3, 8}) {
                run_case(primary, peer, 5120, 17408, tokens, QType::Q5_G64_FP16);
            }
            run_case(primary, peer, 5120, 17408, 1, QType::Q5_G64_FP16, true);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
