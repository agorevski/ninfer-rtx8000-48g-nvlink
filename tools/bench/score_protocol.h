#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>

namespace ninfer::bench::scoring {

inline constexpr std::uint64_t kReferenceLogitsBudget = 128ULL * 1024 * 1024;

inline std::uint32_t reference_batch_rows(std::uint64_t vocabulary) {
    if (vocabulary == 0 || vocabulary > kReferenceLogitsBudget / sizeof(float)) {
        throw std::invalid_argument("reference vocabulary exceeds bounded logits budget");
    }
    auto rows = std::min<std::uint64_t>(1024, kReferenceLogitsBudget / (vocabulary * sizeof(float)));
    if (rows >= 32) { rows -= rows % 32; }
    return static_cast<std::uint32_t>(rows);
}

inline std::int64_t integer(const nlohmann::json& value) {
    if (!value.is_number_integer() || value < 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("scoring counts and IDs must be nonnegative int32 values");
    }
    return value.get<std::int64_t>();
}

inline void validate_report(const nlohmann::json& report, bool allow_plan = false) {
    const bool plan = report.is_object() &&
                      report.value("artifact_type", "") == "ninfer_paired_score_plan";
    if (!report.is_object() || report.at("schema_version") != (plan ? 1 : 2) ||
        (plan ? !allow_plan : report.at("artifact_type") != "ninfer_paired_scores")) {
        throw std::invalid_argument("invalid paired-score report identity");
    }
    const auto& execution = report.at("execution");
    const auto context = integer(execution.at("context"));
    const auto stride = integer(execution.at("stride"));
    const auto limit = integer(execution.at("max_tokens_per_stream"));
    if (context < 2 || stride < 1 || stride >= context || limit == 1 ||
        !execution.at("special_tokens_added").is_boolean() ||
        execution.at("special_tokens_added") != false) {
        throw std::invalid_argument("invalid raw-token scoring protocol");
    }
    if (!plan) {
        const auto backend = report.at("backend").get<std::string>();
        const auto kv = execution.at("kv_dtype").get<std::string>();
        std::string key, value;
        if (backend == "ninfer" && kv == "bf16") { key = "bf16"; value = "fp16"; }
        else if (backend == "ninfer" && kv == "int8") { key = value = "int8_g64"; }
        else if (backend == "llama.cpp" && kv == "bf16-fp16") { key = "bf16"; value = "fp16"; }
        else if (backend == "llama.cpp" && kv == "bf16") { key = value = "bf16"; }
        else if (backend == "llama.cpp" && kv == "fp16") { key = value = "fp16"; }
        else { throw std::invalid_argument("unknown scoring KV profile"); }
        if (execution.at("kv_key_dtype") != key || execution.at("kv_value_dtype") != value) {
            throw std::invalid_argument("scoring KV profile disagrees with physical K/V types");
        }
    }
    const auto& streams = report.at("streams");
    if (!streams.is_array() || streams.empty()) {
        throw std::invalid_argument("empty scoring streams");
    }
    std::set<std::string> identities;
    for (const auto& stream : streams) {
        for (const auto* key : {"id", "domain"}) {
            if (!stream.at(key).is_string() || stream.at(key).get_ref<const std::string&>().empty()) {
                throw std::invalid_argument("missing scoring stream identity/domain");
            }
        }
        if (!identities.insert(stream.at("id").get<std::string>()).second) {
            throw std::invalid_argument("duplicate scoring stream identity");
        }
        const auto& text = stream.at("text");
        if (!text.is_string() || text.get_ref<const std::string&>().empty() ||
            text.get_ref<const std::string&>().size() > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("invalid scoring text");
        }
        const auto& tokens = stream.at("tokens");
        const auto full_count = integer(stream.at("full_input_tokens"));
        const auto expected_count = limit ? std::min(limit, full_count) : full_count;
        if (!tokens.is_array() || tokens.size() < 2 ||
            tokens.size() != static_cast<std::uint64_t>(expected_count)) {
            throw std::invalid_argument("scoring token prefix is incomplete");
        }
        for (const auto& token : tokens) { (void)integer(token); }
        const auto& windows = stream.at("windows");
        if (!windows.is_array() || windows.empty()) {
            throw std::invalid_argument("missing scoring windows");
        }
        std::int64_t target = 1;
        bool first = true;
        for (const auto& window : windows) {
            const auto end = std::min(expected_count, first ? context : target + stride);
            const auto begin = std::max<std::int64_t>(0, end - context);
            if (integer(window.at("input_begin")) != begin ||
                integer(window.at("input_end")) != end ||
                integer(window.at("target_begin")) != target ||
                integer(window.at("target_end")) != end ||
                integer(window.at("first_target")) != target - begin) {
                throw std::invalid_argument("noncanonical scoring window");
            }
            if (plan) {
                if (window.contains("logprobs")) {
                    throw std::invalid_argument("unscored plan must not invent log probabilities");
                }
                target = end;
                first = false;
                continue;
            }
            const auto& scores = window.at("logprobs");
            if (!scores.is_array() || scores.empty() ||
                scores.size() != static_cast<std::uint64_t>(end - target)) {
                throw std::invalid_argument("missing scoring targets");
            }
            for (const auto& score : scores) {
                if (!score.is_number() || !std::isfinite(score.get<double>()) ||
                    score.get<double>() > 1e-5) {
                    throw std::invalid_argument("invalid causal log probability");
                }
            }
            target = end;
            first = false;
        }
        if (target != expected_count) { throw std::invalid_argument("incomplete scoring windows"); }
    }
}

} // namespace ninfer::bench::scoring
