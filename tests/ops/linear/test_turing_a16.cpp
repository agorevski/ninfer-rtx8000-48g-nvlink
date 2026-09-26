#include "ops/linear/linear_test_common.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/weight_input.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>

namespace {

using namespace ninfer::test::linear;

constexpr std::array kTokens{
    Invocation{1}, Invocation{2}, Invocation{3}, Invocation{4},
    Invocation{5}, Invocation{6}, Invocation{7}, Invocation{8},
    Invocation{9}, Invocation{16}, Invocation{17}, Invocation{24},
    Invocation{25}, Invocation{32}, Invocation{33}, Invocation{64},
    Invocation{65}, Invocation{128}, Invocation{129}, Invocation{511},
    Invocation{512}, Invocation{513}, Invocation{1024}, Invocation{4096},
    Invocation{7, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
    Invocation{513, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
    Invocation{512, CallForm::A16Convenience},
};

ninfer::test::quantized_weight::PackedWeight large_q4_weight(int n, int k, std::uint32_t seed) {
    auto weight = make_q4_g64_fp16_weight(n, k, seed);
    for (std::size_t offset = weight.scale_plane_offset;
         offset < weight.scale_plane_offset + weight.scale_plane_bytes; offset += 2) {
        weight.payload[offset] = 0;
        weight.payload[offset + 1] = 0x78; // FP16 32768: decoded Q4 cannot fit FP16.
    }
    return weight;
}

int unavailable_formats() {
    using namespace ninfer;
    int failures = 0;
    const auto rejects = [&](const char* label, auto query) {
        try {
            (void)query();
            std::cerr << label << ": SM75 accepted an unavailable projection format\n";
            ++failures;
        } catch (const std::invalid_argument&) {
        }
    };
    for (const auto format : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16}) {
        for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                  ops::LinearPolicy::AllowA4}) {
            rejects("linear workspace", [&] {
                return ops::linear_workspace_capacity_bytes(format, 34816, 5120, policy, 1, 512);
            });
            rejects("linear_add workspace", [&] {
                return ops::linear_add_workspace_capacity_bytes(format, 5120, 17408, policy, 1, 512);
            });
            rejects("linear_swiglu workspace", [&] {
                return ops::linear_swiglu_workspace_capacity_bytes(format, 34816, 5120, policy, 1, 512);
            });
            rejects("attention projection workspace", [&] {
                return ops::attn_input_proj_workspace_capacity_bytes(format, 14336, 5120, policy, 1, 512);
            });
            rejects("GDN projection workspace", [&] {
                return ops::gdn_input_proj_workspace_capacity_bytes(format, 16384, 5120, policy, 1, 512);
            });
            rejects("GDN snapshot workspace", [&] {
                return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                    format, 16384, 5120, policy, 1, 1, 512);
            });
            rejects("GDN record workspace", [&] {
                return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                    format, 16384, 5120, policy, 1, 1, 512);
            });
        }
        rejects("linear_topk workspace", [&] {
            return ops::linear_topk_workspace_capacity_bytes(format, 248320, 5120, 1, 512);
        });
    }
    return failures;
}

int c1_row_shards() {
    using namespace ninfer;
    namespace qw = test::quantized_weight;
    int failures = 0;
    for (const auto [qtype, rows, k] :
         {std::tuple{QType::Q5_G64_FP16, 2560, 6144},
          std::tuple{QType::Q8_G32_FP16, 124160, 5120}}) {
        const bool add = qtype == QType::Q5_G64_FP16;
        const int parent_rows = rows * 2;
        auto packed = qw::make_patterned_weight(
            qtype, parent_rows, k, 7761,
            {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
        std::vector<int> oracle_rows{0, 1, rows - 2, rows - 1, rows, rows + 1,
                                     parent_rows - 2, parent_rows - 1};
        for (int i = 1; i < 16; ++i) oracle_rows.push_back((parent_rows - 1) * i / 16);
        std::sort(oracle_rows.begin(), oracle_rows.end());
        oracle_rows.erase(std::unique(oracle_rows.begin(), oracle_rows.end()), oracle_rows.end());
        const auto decoded = qw::materialize_rows_fp32(packed, oracle_rows);
        std::mt19937 random(7762);
        std::vector<std::uint16_t> activation(k * 2), residual(parent_rows);
        for (auto& bits : activation)
            bits = test::f32_to_bf16(float(int(random() % 512) - 256) / 512);
        for (auto& bits : residual)
            bits = test::f32_to_bf16(float(int(random() % 512) - 256) / 512);
        test::GuardedDeviceBuffer dw(packed.payload.size()), dx(activation.size() * 2),
            dy(residual.size() * 2);
        dw.copy_from_host(packed.payload.data(), dw.bytes());
        const std::array shape{static_cast<std::uint64_t>(parent_rows),
                               static_cast<std::uint64_t>(k)};
        const WeightParent parent{weight_geometry(qtype, QuantLayout::RowSplit, shape),
                                   static_cast<const std::byte*>(dw.data())};
        std::array<Weight, 2> weights;
        for (int part = 0; part < 2; ++part) {
            const auto begin = static_cast<std::uint64_t>(part) * rows * k;
            const WeightView view{{static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k)},
                                   {{&parent, begin, begin + static_cast<std::uint64_t>(rows) * k}}};
            weights[part] = ops::prepare_linear_weight({view}).weight;
        }
        const auto query = [&](int first, int last, ops::LinearPolicy policy) {
            return add ? ops::linear_add_workspace_capacity_bytes(qtype, rows, k, policy, first, last)
                       : ops::linear_workspace_capacity_bytes(qtype, rows, k, policy, first, last);
        };
        for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                  ops::LinearPolicy::AllowA4}) {
            if (query(1, 1, policy) != 0) ++failures;
            for (const auto [first, last] : {std::pair{1, 2}, std::pair{2, 2}, std::pair{1, 512}}) {
                try {
                    (void)query(first, last, policy);
                    std::cerr << "C1 row shard accepted a multi-column workspace interval\n";
                    ++failures;
                } catch (const std::invalid_argument&) {
                }
            }
        }
        WorkspaceArena workspace(256);
        DeviceContext context;
        const auto invoke = [&](int part, int tokens) {
            Tensor input(dx.data(), DType::BF16, {k, tokens});
            Tensor output(static_cast<std::uint16_t*>(dy.data()) + part * rows,
                          DType::BF16, {rows, tokens});
            if (add)
                ops::linear_add(input, weights[part], output, ops::LinearPolicy::A16Only,
                                 workspace, context.stream);
            else
                ops::linear(input, weights[part], output, ops::LinearPolicy::A16Only,
                             workspace, context.stream);
        };
        dy.fill(255);
        test::cuda_check(cudaDeviceSynchronize(), "finish C1 rejection setup");
        try {
            invoke(0, 2);
            std::cerr << "C1 row shard accepted multi-column execution\n";
            ++failures;
        } catch (const std::invalid_argument&) {
        }
        std::vector<std::uint16_t> unchanged(parent_rows);
        dy.copy_to_host(unchanged.data(), dy.bytes());
        if (std::any_of(unchanged.begin(), unchanged.end(), [](auto bits) { return bits != 0xffff; })) {
            std::cerr << "Rejected C1 row shard mutated output\n";
            ++failures;
        }
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        const auto launch = [&] { invoke(0, 1); invoke(1, 1); };
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
        for (int phase = 0; phase < 3; ++phase) {
            auto represented = activation;
            auto initial = residual;
            if (phase == 1) {
                for (auto& bits : represented)
                    bits = test::f32_to_bf16(std::ldexp(test::bf16_to_f32(bits), -32));
                std::fill(initial.begin(), initial.end(), 0);
            } else if (phase == 2) {
                for (auto& bits : represented) bits ^= 0x8000;
            }
            dx.copy_from_host(represented.data(), dx.bytes());
            if (add) dy.copy_from_host(initial.data(), dy.bytes());
            else dy.fill(255);
            test::cuda_check(cudaDeviceSynchronize(), "finish C1 graph input update");
            if (phase == 0) launch();
            else graph.launch(context.stream);
            context.synchronize();
            std::vector<std::uint16_t> merged(parent_rows);
            dy.copy_to_host(merged.data(), dy.bytes());
            if (std::any_of(merged.begin(), merged.end(), [](auto bits) {
                    return !std::isfinite(test::bf16_to_f32(bits));
                })) ++failures;
            std::vector<double> actual, reference;
            for (std::size_t r = 0; r < oracle_rows.size(); ++r) {
                double value = add ? test::bf16_to_f32(initial[oracle_rows[r]]) : 0;
                for (int column = 0; column < k; ++column)
                    value += double(decoded[r * k + column]) *
                             double(test::bf16_to_f32(represented[column]));
                reference.push_back(value);
                actual.push_back(test::bf16_to_f32(merged[oracle_rows[r]]));
            }
            failures += test::verify_reduction("C1 row-shard gather boundaries", actual, reference,
                                               {1.0 / 256, 1.0 / 256, 2.0 / 256});
        }
        if (workspace.used() != 0 || workspace.peak_used() != 0) ++failures;
        failures += dw.verify_guards("C1 row-shard weights");
        failures += dx.verify_guards("C1 row-shard input");
        failures += dy.verify_guards("C1 row-shard output");
    }
    return failures;
}

enum class PrefillOp { Linear, Add, SwiGlu };

struct PrefillProfile {
    ninfer::QType qtype;
    int n;
    int k;
    PrefillOp op;
    bool cancellation = false;
};

int prefill_workspace_and_replay(const PrefillProfile& profile) {
    using namespace ninfer;
    namespace qw = test::quantized_weight;
    constexpr int max_tokens = 1024;
    const int output_rows = profile.op == PrefillOp::SwiGlu ? profile.n / 2 : profile.n;
    auto packed = qw::make_patterned_weight(
        profile.qtype, profile.n, profile.k, 7731,
        {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    if (profile.cancellation) {
        std::fill_n(packed.payload.begin(), packed.code_plane_bytes, std::uint8_t{0x11});
        std::fill_n(packed.payload.begin() + packed.high_plane_offset,
                    packed.high_plane_bytes, std::uint8_t{0});
        for (std::size_t offset = packed.scale_plane_offset;
             offset < packed.scale_plane_offset + packed.scale_plane_bytes; offset += 2) {
            packed.payload[offset] = 0;
            packed.payload[offset + 1] = 0x30;
        }
    }
    std::vector<int> rows;
    for (int i = 0; i < 32; ++i)
        rows.push_back(static_cast<int>(static_cast<std::int64_t>(output_rows - 1) * i / 31));
    auto weight_rows = rows;
    if (profile.op == PrefillOp::SwiGlu)
        for (int row : rows) weight_rows.push_back(row + output_rows);
    const auto oracle_weight = qw::materialize_rows_fp32(packed, weight_rows);
    std::mt19937 random(7732);
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(profile.k) * max_tokens);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(output_rows) * max_tokens);
    for (auto& value : activation)
        value = test::f32_to_bf16(static_cast<float>(static_cast<int>(random() % 1024) - 512) / 1024);
    for (auto& value : residual)
        value = test::f32_to_bf16(static_cast<float>(static_cast<int>(random() % 1024) - 512) / 512);
    if (profile.cancellation) {
        std::fill(activation.begin(), activation.end(), test::f32_to_bf16(0.125f));
        for (int token = 0; token < max_tokens; ++token)
            activation[static_cast<std::size_t>(token) * profile.k] = test::f32_to_bf16(0.25f);
        std::fill(residual.begin(), residual.end(), test::f32_to_bf16(-float(profile.k) / 64));
    }
    test::GuardedDeviceBuffer dw(packed.payload.size()), dx(activation.size() * 2),
        dy(residual.size() * 2);
    dw.copy_from_host(packed.payload.data(), dw.bytes());
    const Weight weight = packed.device_weight(dw.data());
    const auto capacity = [&](int first, int last) {
        if (profile.op == PrefillOp::Linear)
            return ops::linear_workspace_capacity_bytes(
                profile.qtype, profile.n, profile.k, ops::LinearPolicy::A16Only, first, last);
        if (profile.op == PrefillOp::Add)
            return ops::linear_add_workspace_capacity_bytes(
                profile.qtype, profile.n, profile.k, ops::LinearPolicy::A16Only, first, last);
        return ops::linear_swiglu_workspace_capacity_bytes(
            profile.qtype, profile.n, profile.k, ops::LinearPolicy::A16Only, first, last);
    };
    int failures = 0;
    if (capacity(1, 511) != 0 || capacity(512, 512) == 0 ||
        capacity(511, 512) != capacity(512, 512) ||
        capacity(512, 513) != capacity(513, 513) ||
        capacity(1, max_tokens) != capacity(max_tokens, max_tokens)) {
        std::cerr << "Turing prefill workspace interval does not cover its exact route boundary\n";
        ++failures;
    }
    DeviceContext context;
    for (int tokens : {511, 512, 513, 1024}) {
        const std::size_t exact = capacity(tokens, tokens);
        test::GuardedDeviceBuffer scratch(std::max<std::size_t>(exact, 256));
        WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
        Tensor x(dx.data(), DType::BF16, {profile.k, tokens});
        Tensor output(dy.data(), DType::BF16, {output_rows, tokens});
        const auto launch = [&] {
            if (profile.op == PrefillOp::Linear)
                ops::linear(x, weight, output, ops::LinearPolicy::A16Only, workspace, context.stream);
            else if (profile.op == PrefillOp::Add)
                ops::linear_add(x, weight, output, ops::LinearPolicy::A16Only, workspace, context.stream);
            else
                ops::linear_swiglu(x, weight, output, ops::LinearPolicy::A16Only, workspace, context.stream);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
        for (int phase = 0; phase < (tokens >= 512 ? 6 : 3); ++phase) {
            auto represented = activation;
            auto initial_residual = residual;
            if (phase == 1) {
                for (int token = 0; token < tokens; ++token) {
                    const std::size_t begin = static_cast<std::size_t>(token) * profile.k;
                    represented[begin + token % profile.k] =
                        test::f32_to_bf16(profile.op == PrefillOp::SwiGlu ? 0x1p20f : 0x1p100f);
                    represented[begin + (token + 1) % profile.k] = test::f32_to_bf16(0x1p-100f);
                }
            }
            if (phase == 3 || phase == 4) {
                const int exponent =
                    phase == 4 ? (profile.op == PrefillOp::SwiGlu ? -48 : -120) : -32;
                for (auto& bits : represented)
                    bits = test::f32_to_bf16(std::ldexp(test::bf16_to_f32(bits), exponent));
                std::fill(initial_residual.begin(), initial_residual.end(), 0);
            }
            dx.copy_from_host(represented.data(), dx.bytes());
            if (profile.op == PrefillOp::Add)
                dy.copy_from_host(initial_residual.data(), dy.bytes());
            else
                dy.fill(255);
            test::cuda_check(cudaDeviceSynchronize(), "finish prefill graph input update");
            graph.launch(context.stream);
            context.synchronize();
            if (workspace.used() != 0 || workspace.peak_used() != exact) {
                std::cerr << "Turing prefill exact workspace high-water mismatch\n";
                ++failures;
            }
            std::vector<std::uint16_t> actual_bits(static_cast<std::size_t>(output_rows) * tokens);
            dy.copy_to_host(actual_bits.data(), actual_bits.size() * 2);
            for (auto bits : actual_bits) {
                if (!std::isfinite(test::bf16_to_f32(bits))) {
                    std::cerr << "Turing prefill produced nonfinite or unwritten output\n";
                    ++failures;
                    break;
                }
            }
            std::vector<double> actual, reference;
            for (int sample = 0; sample < 32; ++sample) {
                const int token = static_cast<int>(static_cast<std::int64_t>(tokens - 1) * sample / 31);
                for (int ri = 0; ri < 32; ++ri) {
                    double gate = 0, up = 0;
                    for (int col = 0; col < profile.k; ++col) {
                        const double value = test::bf16_to_f32(
                            represented[static_cast<std::size_t>(token) * profile.k + col]);
                        gate += static_cast<double>(oracle_weight[static_cast<std::size_t>(ri) * profile.k + col]) * value;
                        if (profile.op == PrefillOp::SwiGlu)
                            up += static_cast<double>(oracle_weight[static_cast<std::size_t>(ri + 32) * profile.k + col]) * value;
                    }
                    const std::size_t index = static_cast<std::size_t>(token) * output_rows + rows[ri];
                    if (profile.op == PrefillOp::SwiGlu) gate = gate / (1 + std::exp(-gate)) * up;
                    if (profile.op == PrefillOp::Add)
                        gate += test::bf16_to_f32(initial_residual[index]);
                    reference.push_back(gate);
                    actual.push_back(test::bf16_to_f32(actual_bits[index]));
                }
            }
            const auto criterion = profile.op == PrefillOp::SwiGlu
                                       ? test::ReductionCriterion{3.3e-3, 5e-3, 6.3e-3}
                                       : test::ReductionCriterion{1.0 / 256, 1.0 / 256, 2.0 / 256};
            const auto label = "Turing prefill op=" + std::to_string(static_cast<int>(profile.op)) +
                               " N=" + std::to_string(profile.n) + " K=" + std::to_string(profile.k) +
                               " T=" + std::to_string(tokens) + " phase=" + std::to_string(phase);
            failures += test::verify_reduction(label, actual, reference, criterion);
        }
        failures += scratch.verify_guards("Turing prefill workspace");
        failures += dy.verify_guards("Turing prefill output");
    }
    std::vector<std::uint16_t> after(activation.size());
    dx.copy_to_host(after.data(), dx.bytes());
    if (after != activation) {
        std::cerr << "Turing prefill changed its represented activation\n";
        ++failures;
    }
    std::vector<std::uint8_t> weight_after(packed.payload.size());
    dw.copy_to_host(weight_after.data(), dw.bytes());
    if (weight_after != packed.payload) {
        std::cerr << "Turing prefill changed its packed weights\n";
        ++failures;
    }
    failures += dx.verify_guards("Turing prefill input");
    failures += dw.verify_guards("Turing prefill weight");
    return failures;
}

int prefill_subnormal_products() {
    using namespace ninfer;
    namespace qw = test::quantized_weight;
    constexpr int rows = 1024, k = 5120, tokens = 512;
    constexpr std::array<std::uint16_t, 15> values{
        0x3880, 0x387f, 0x3800, 0x3780, 0x3700, 0x3701, 0x3680, 0x3681,
        0x3682, 0x3400, 0x3380, 0x3300, 0x0080, 0x0001, 0x7180};
    int failures = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto packed = qw::make_patterned_weight(QType::Q5_G64_FP16, rows, k, 7733);
        std::fill_n(packed.payload.begin(), packed.code_plane_bytes, std::uint8_t{0});
        std::fill_n(packed.payload.begin() + packed.high_plane_offset,
                    packed.high_plane_bytes, std::uint8_t{0});
        for (std::size_t offset = packed.scale_plane_offset;
             offset < packed.scale_plane_offset + packed.scale_plane_bytes; offset += 2) {
            packed.payload[offset] = 0;
            packed.payload[offset + 1] = mode == 1 ? 0x60 : 0x3c;
        }
        for (int row = 0; row < rows; ++row)
            packed.payload[static_cast<std::size_t>(row) * k / 2] = 0x10;
        std::vector<std::uint16_t> input(k * tokens, mode == 1 ? 1 : 0);
        if (mode == 0) {
            for (int token = 0; token < tokens; ++token) {
                input[static_cast<std::size_t>(token) * k] = test::f32_to_bf16(0x1p20f);
                input[static_cast<std::size_t>(token) * k + 1] = values[token % values.size()];
            }
        }
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::Q5_G64_FP16, rows, k, ops::LinearPolicy::A16Only, tokens, tokens);
        test::GuardedDeviceBuffer dw(packed.payload.size()), dx(input.size() * 2),
            dy(static_cast<std::size_t>(rows) * tokens * 2);
        dw.copy_from_host(packed.payload.data(), dw.bytes());
        dx.copy_from_host(input.data(), dx.bytes());
        WorkspaceArena workspace(capacity);
        const auto weight = packed.device_weight(dw.data());
        Tensor x(dx.data(), DType::BF16, {k, tokens});
        Tensor output(dy.data(), DType::BF16, {rows, tokens});
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight, output, ops::LinearPolicy::A16Only, workspace, context.stream);
        });
        graph.instantiate(definition);
        dy.fill(255);
        test::cuda_check(cudaDeviceSynchronize(), "finish subnormal fixture setup");
        graph.launch(context.stream);
        context.synchronize();
        std::vector<std::uint16_t> actual(static_cast<std::size_t>(rows) * tokens);
        dy.copy_to_host(actual.data(), dy.bytes());
        for (int token = 0; token < tokens; ++token) {
            double expected = 0;
            for (int column = 0; column < k; ++column)
                expected += qw::logical_weight_fp64(packed, 0, column) *
                            double(test::bf16_to_f32(input[static_cast<std::size_t>(token) * k + column]));
            for (int row = 0; row < rows; ++row) {
                if (double(test::bf16_to_f32(actual[static_cast<std::size_t>(token) * rows + row])) != expected) {
                    std::cerr << "Turing prefill lost an exact one-term subnormal product\n";
                    ++failures;
                    break;
                }
            }
        }
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << "Turing subnormal workspace high-water mismatch\n";
            ++failures;
        }
        failures += dy.verify_guards("Turing subnormal output");
    }
    return failures;
}

int qualify() {
    int failures = unavailable_formats();
    failures += c1_row_shards();
    failures += prefill_subnormal_products();
    for (const auto profile : {
             PrefillProfile{ninfer::QType::Q4_G64_FP16, 17408, 5120, PrefillOp::Linear},
             PrefillProfile{ninfer::QType::Q5_G64_FP16, 5120, 17408, PrefillOp::Linear},
             PrefillProfile{ninfer::QType::Q4_G64_FP16, 5120, 6144, PrefillOp::Add},
             PrefillProfile{ninfer::QType::Q5_G64_FP16, 5120, 17408, PrefillOp::Add},
             PrefillProfile{ninfer::QType::Q5_G64_FP16, 2560, 17408, PrefillOp::Add},
             PrefillProfile{ninfer::QType::Q4_G64_FP16, 34816, 5120, PrefillOp::SwiGlu},
             PrefillProfile{ninfer::QType::Q5_G64_FP16, 5120, 17408, PrefillOp::Add, true}}) {
        failures += prefill_workspace_and_replay(profile);
    }
    for (const auto [n, k] : {std::pair{1024, 5120}, std::pair{7168, 5120},
                               std::pair{34816, 5120}, std::pair{5120, 6144}}) {
        failures += run_shape("Turing Q4 A16", ActivationCompute::A16, make_q4_g64_fp16_weight,
                              {n, k, 751U, Comparison::Sampled, true, kTokens});
    }
    for (const auto [n, k] : {std::pair{7168, 5120}, std::pair{5120, 17408}}) {
        failures += run_shape("Turing Q5 A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                              {n, k, 752U, Comparison::Sampled, true, kTokens});
    }
    for (const auto [n, k] : {std::pair{14336, 5120}, std::pair{34816, 5120},
                               std::pair{5120, 17408}, std::pair{5120, 6144}}) {
        failures += run_shape("Turing Q8 A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {n, k, 753U, Comparison::Sampled, true, kTokens});
    }
    constexpr std::array kVocabularyTokens{
        Invocation{1}, Invocation{3}, Invocation{8}, Invocation{33}, Invocation{128},
    };
    constexpr std::array kVisionTokens{Invocation{4}, Invocation{8}, Invocation{32},
                                      Invocation{128}, Invocation{512}};
    failures += run_shape("Turing Q6 vision", ActivationCompute::A16, make_q6_g64_fp16_weight,
                          {1152, 1536, 754U, Comparison::Sampled, true, kVisionTokens});
    failures += run_shape("Turing Q8 output head", ActivationCompute::A16,
                          make_q8_g32_fp16_weight,
                          {248320, 5120, 765U, Comparison::Sampled, true, kVocabularyTokens});
    constexpr std::array kTailTokens{Invocation{4}, Invocation{36}, Invocation{68}, Invocation{516}};
    failures += run_shape("Turing Q5 padded K", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1152, 4304, 755U, Comparison::Sampled, true, kTailTokens});
    constexpr std::array kRowTailTokens{Invocation{4}, Invocation{36}, Invocation{68}, Invocation{516}};
    failures += run_shape("Turing Q4 row tail", ActivationCompute::A16, make_q4_g64_fp16_weight,
                          {4304, 1152, 756U, Comparison::Sampled, true, kRowTailTokens});
    failures += run_shape("Turing Q4 FFN row shard", ActivationCompute::A16,
                          make_q4_g64_fp16_weight,
                          {17408, 5120, 757U, Comparison::Sampled, true, kTokens});
    constexpr std::array kRangeTokens{Invocation{1}, Invocation{3}, Invocation{8},
                                     Invocation{33}, Invocation{65}};
    for (const float multiplier : {0x1p20f, 0x1p-32f}) {
        failures += run_shape("Turing BF16 exponent range", ActivationCompute::A16,
                              make_q4_g64_fp16_weight,
                              {1024, 5120, 760U, Comparison::Sampled, true, kRangeTokens,
                               multiplier});
        failures += run_shape("Turing Q8 BF16 exponent range", ActivationCompute::A16,
                              make_q8_g32_fp16_weight,
                              {1024, 5120, 761U, Comparison::Sampled, true, kRangeTokens,
                               multiplier});
    }
    failures += run_shape("Turing large FP16 weight scales", ActivationCompute::A16,
                          large_q4_weight,
                          {1024, 5120, 762U, Comparison::Sampled, true, kRangeTokens});
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const bool c1_only = argc == 2 && std::string_view(argv[1]) == "--c1-row-shards";
    if (argc != 1 && !c1_only) {
        std::cerr << "usage: ninfer_linear_turing_a16_test [--c1-row-shards]\n";
        return 2;
    }
    if (!ninfer::test::linear::cuda_available()) return 77;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 1;
    if (properties.major != 7 || properties.minor != 5) return 77;
    try {
        const int failures = c1_only ? c1_row_shards() : qualify();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Turing A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Turing A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
