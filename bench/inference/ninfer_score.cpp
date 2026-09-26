#include "ninfer/engine.h"
#include "apps/perplexity/corpus.h"
#include "apps/perplexity/evaluation.h"
#include "tools/bench/score_protocol.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;

int number(const std::string& value) {
    int result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result < 0) {
        throw std::invalid_argument("invalid nonnegative integer: " + value);
    }
    return result;
}

void run(int argc, char** argv) {
    std::string weights, corpus, output, kv = "int8";
    int context = 4096, stride = 2048, limit = 8192, device = 0;
    std::optional<int> tensor_parallel_device;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--quick") {
            quick = true;
            continue;
        }
        if (++i == argc) { throw std::invalid_argument("missing value for " + option); }
        const std::string value = argv[i];
        if (option == "--weights") { weights = value; }
        else if (option == "--corpus") { corpus = value; }
        else if (option == "--output") { output = value; }
        else if (option == "--context") { context = number(value); }
        else if (option == "--stride") { stride = number(value); }
        else if (option == "--max-tokens") { limit = number(value); }
        else if (option == "--device") { device = number(value); }
        else if (option == "--tensor-parallel-device") { tensor_parallel_device = number(value); }
        else if (option == "--kv-dtype") { kv = value; }
        else { throw std::invalid_argument("unknown option: " + option); }
    }
    if (weights.empty() || corpus.empty() || output.empty() || context < 2 ||
        stride < 1 || stride >= context || limit == 1 || (kv != "bf16" && kv != "int8")) {
        throw std::invalid_argument("required weights/corpus/output; context>=2, 1<=stride<context, "
                                    "max-tokens=0 or >=2; kv-dtype=bf16|int8");
    }
    if (std::filesystem::exists(output)) { throw std::runtime_error("output already exists"); }
    auto selection = ninfer::perplexity::load_corpus(corpus, quick);
    ninfer::EngineOptions options;
    options.artifact_path = weights;
    options.purpose = ninfer::EnginePurpose::CausalScoring;
    options.device = device;
    options.tensor_parallel_device = tensor_parallel_device;
    options.max_context = context;
    options.kv_cache = kv == "int8" ? ninfer::KvCacheStorage::Int8Group64
                                   : ninfer::KvCacheStorage::BFloat16;
    ninfer::Engine engine(std::move(options));
    const auto load = engine.load_summary();
    json streams = json::array();
    for (const auto& source : selection.streams) {
        auto tokens = engine.tokenize_text(source.text);
        const auto full_count = tokens.size();
        if (limit != 0 && tokens.size() > static_cast<std::size_t>(limit)) { tokens.resize(limit); }
        const auto windows = ninfer::perplexity::plan_windows(tokens.size(), context, stride);
        json scored = json::array();
        for (const auto& window : windows) {
            std::vector<ninfer::TokenId> input(tokens.begin() + window.input_begin,
                                               tokens.begin() + window.input_end);
            auto logprobs = engine.score_tokens(std::move(input), window.first_target);
            if (logprobs.size() != window.target_end - window.target_begin) {
                throw std::runtime_error("unexpected score count for " + source.id);
            }
            ninfer::perplexity::ScoreAggregate aggregate;
            aggregate.add(logprobs);
            scored.push_back({{"input_begin", window.input_begin}, {"input_end", window.input_end},
                              {"target_begin", window.target_begin}, {"target_end", window.target_end},
                              {"first_target", window.first_target}, {"logprobs", logprobs}});
        }
        streams.push_back({{"id", source.id}, {"domain", source.domain}, {"text", source.text},
                           {"tokens", tokens}, {"full_input_tokens", full_count},
                           {"windows", std::move(scored)}});
        std::cerr << "scored " << source.id << ": " << tokens.size() - 1 << " targets\n";
    }
    json report{
        {"schema_version", 2}, {"artifact_type", "ninfer_paired_scores"}, {"backend", "ninfer"},
        {"artifact", {{"path", std::filesystem::absolute(weights).string()},
                      {"name", load.model_name}, {"formats", load.weight_formats},
                      {"prefill_signature", load.prefill_signature}}},
        {"execution", {{"context", context}, {"stride", stride}, {"max_tokens_per_stream", limit},
                       {"device", device}, {"kv_dtype", kv}, {"purpose", "causal_scoring"},
                       {"kv_key_dtype", kv == "bf16" ? "bf16" : "int8_g64"},
                       {"kv_value_dtype", kv == "bf16" ? "fp16" : "int8_g64"},
                       {"prefill_chunk", 1024}, {"score_tile_tokens", 1024},
                       {"special_tokens_added", false}}},
        {"corpus", {{"id", selection.corpus_id}, {"mode", selection.mode},
                    {"path", selection.source.string()}}},
        {"streams", std::move(streams)}};
    report["execution"]["tensor_parallel_device"] = tensor_parallel_device
                                                        ? json(*tensor_parallel_device) : json(nullptr);
    ninfer::bench::scoring::validate_report(report);
    const std::filesystem::path path(output);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream file(path);
    if (!(file << report.dump(2) << '\n') || !file.flush()) {
        throw std::runtime_error("failed writing output: " + output);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "ninfer_score --weights MODEL --corpus MANIFEST --output JSON [--quick]\n"
                     "  [--context 4096] [--stride 2048] [--max-tokens 8192]\n"
                     "  [--device 0] [--tensor-parallel-device N] [--kv-dtype bf16|int8]\n"
                     "Public Engine paired-quality export. max-tokens=0 scores complete streams.\n";
        return 0;
    }
    try { run(argc, argv); return 0; }
    catch (const std::exception& error) {
        std::cerr << "ninfer_score: " << error.what() << '\n';
        return 1;
    }
}
