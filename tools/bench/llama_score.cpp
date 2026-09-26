// External reference adapter; never linked into NInfer or used for model execution by its Engine.
#include <llama.h>
#include <nlohmann/json.hpp>
#include "score_protocol.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "llama_score MODEL.gguf PLAN_OR_NINFER_SCORES.json OUTPUT.json [--kv-dtype bf16-fp16|bf16|fp16]\n"
                     "llama_score --check-plan PLAN_OR_NINFER_SCORES.json  (CPU-only)\n"
                     "Default: BF16 K / FP16 V, matching NInfer's bf16 profile. No BOS/template.\n";
        return 0;
    }
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-plan") {
            std::ifstream input(argv[2]);
            json plan;
            input >> plan;
            ninfer::bench::scoring::validate_report(plan, true);
            std::cout << "validated " << plan.at("streams").size() << " scoring streams\n";
            return 0;
        }
        if (argc != 4 && argc != 6) {
            throw std::runtime_error("expected MODEL.gguf INPUT.json OUTPUT.json [--kv-dtype bf16-fp16|bf16|fp16]");
        }
        const std::string kv = argc == 6 ? argv[5] : "bf16-fp16";
        if ((argc == 6 && std::string(argv[4]) != "--kv-dtype") ||
            (kv != "bf16-fp16" && kv != "bf16" && kv != "fp16")) {
            throw std::runtime_error("--kv-dtype must be bf16-fp16, bf16, or fp16");
        }
        if (std::filesystem::exists(argv[3])) { throw std::runtime_error("output already exists"); }
        std::ifstream input(argv[2]);
        json report;
        input >> report;
        if (report.at("backend") != "ninfer" && report.at("backend") != "tokenizer-only") {
            throw std::runtime_error("expected a native ninfer_score report or CPU token plan");
        }
        ninfer::bench::scoring::validate_report(report, true);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), &llama_model_free);
        if (!model) { throw std::runtime_error("failed loading GGUF"); }
        const auto* vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const auto score_rows = ninfer::bench::scoring::reference_batch_rows(n_vocab);
        auto cp = llama_context_default_params();
        cp.n_ctx = report.at("execution").at("context").get<uint32_t>();
        cp.n_batch = 1024;
        cp.n_ubatch = cp.n_batch;
        cp.n_outputs_max = score_rows;
        cp.n_outputs_max_per_seq = score_rows;
        cp.n_seq_max = 1;
        cp.type_k = kv == "fp16" ? GGML_TYPE_F16 : GGML_TYPE_BF16;
        cp.type_v = kv == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), &llama_free);
        if (!ctx) { throw std::runtime_error("failed creating reference context"); }
        for (auto& stream : report.at("streams")) {
            const auto text = stream.at("text").get<std::string>();
            const int required = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, false);
            if (required >= 0) { throw std::runtime_error("unexpected empty tokenizer output"); }
            std::vector<llama_token> encoded(-required);
            const int count = llama_tokenize(vocab, text.data(), text.size(), encoded.data(),
                                             encoded.size(), false, false);
            const auto expected = stream.at("tokens").get<std::vector<llama_token>>();
            if (count != stream.at("full_input_tokens").get<int>() ||
                expected.size() > encoded.size() ||
                !std::equal(expected.begin(), expected.end(), encoded.begin())) {
                throw std::runtime_error("tokenizer IDs differ for " + stream.at("id").get<std::string>());
            }
            for (auto& window : stream.at("windows")) {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                const int begin = window.at("input_begin");
                const int end = window.at("input_end");
                const int first = window.at("first_target");
                std::vector<double> logprobs;
                auto batch = llama_batch_init(cp.n_batch, 0, 1);
                try {
                    for (int offset = 0; offset < end - begin - 1;) {
                        const int prefix_left = std::max(0, first - 1 - offset);
                        const int rows = prefix_left ? std::min<int>(cp.n_batch, prefix_left)
                                                     : static_cast<int>(score_rows);
                        batch.n_tokens = std::min(rows, end - begin - 1 - offset);
                        for (int j = 0; j < batch.n_tokens; ++j) {
                            batch.token[j] = expected.at(begin + offset + j);
                            batch.pos[j] = offset + j;
                            batch.n_seq_id[j] = 1;
                            batch.seq_id[j][0] = 0;
                            batch.logits[j] = offset + j + 1 >= first;
                        }
                        if (llama_decode(ctx.get(), batch) != 0) {
                            throw std::runtime_error("reference llama_decode failed");
                        }
                        for (int j = 0; j < batch.n_tokens; ++j) {
                            if (!batch.logits[j]) { continue; }
                            const float* logits = llama_get_logits_ith(ctx.get(), j);
                            if (!logits) { throw std::runtime_error("missing reference logits"); }
                            const double maximum = *std::max_element(logits, logits + n_vocab);
                            double denominator = 0;
                            for (int k = 0; k < n_vocab; ++k) {
                                denominator += std::exp(static_cast<double>(logits[k]) - maximum);
                            }
                            const auto target = expected.at(begin + offset + j + 1);
                            if (target < 0 || target >= n_vocab) {
                                throw std::runtime_error("target outside reference vocabulary");
                            }
                            const double score = static_cast<double>(logits[target]) -
                                                 maximum - std::log(denominator);
                            if (!std::isfinite(score)) {
                                throw std::runtime_error("nonfinite reference score");
                            }
                            logprobs.push_back(score);
                        }
                        offset += batch.n_tokens;
                    }
                } catch (...) {
                    llama_batch_free(batch);
                    throw;
                }
                llama_batch_free(batch);
                if (logprobs.size() != window.at("target_end").get<size_t>() -
                                      window.at("target_begin").get<size_t>()) {
                    throw std::runtime_error("reference target count mismatch");
                }
                window["logprobs"] = std::move(logprobs);
            }
            std::cerr << "scored " << stream.at("id") << '\n';
        }
        report["backend"] = "llama.cpp";
        report["schema_version"] = 2;
        report["artifact_type"] = "ninfer_paired_scores";
        report["artifact"] = {{"path", std::filesystem::absolute(argv[1]).string()}};
        report["execution"]["kv_dtype"] = kv;
        report["execution"]["kv_key_dtype"] = kv == "fp16" ? "fp16" : "bf16";
        report["execution"]["kv_value_dtype"] = kv == "bf16" ? "bf16" : "fp16";
        report["execution"]["prefill_chunk"] = cp.n_batch;
        report["execution"]["device"] = 0;
        report["execution"]["tensor_parallel_device"] = nullptr;
        report["execution"]["log_softmax_accumulation"] = "fp64";
        report["execution"]["score_tile_tokens"] = score_rows;
        report["execution"]["logits_tile_bytes"] =
            std::uint64_t(score_rows) * n_vocab * sizeof(float);
        report["execution"]["logits_tile_budget_bytes"] =
            ninfer::bench::scoring::kReferenceLogitsBudget;
        ninfer::bench::scoring::validate_report(report);
        const auto output = std::filesystem::path(argv[3]);
        if (!output.parent_path().empty()) { std::filesystem::create_directories(output.parent_path()); }
        std::ofstream file(output);
        if (!(file << report.dump(2) << '\n') || !file.flush()) {
            throw std::runtime_error("failed writing reference scores");
        }
        ctx.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "llama_score: " << error.what() << '\n';
        return 1;
    }
}
