#include "ninfer/engine.h"
#include "models/qwen3_5/load.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;

void materialization_binding(const char* artifact) {
    DeviceContext primary(0);
    for (const int caller : {0, 1}) {
        DeviceGuard caller_guard(caller);
        models::LoadOptions options;
        options.tensor_parallel_device = 1;
        auto model = models::qwen3_5::load_model(artifact, options, primary);
        int current = -1;
        CUDA_CHECK(cudaGetDevice(&current));
        if (current != caller) {
            throw std::runtime_error("model materialization changed caller device");
        }
        cudaPointerAttributes attributes{};
        const auto& weights = model->weights().text;
        CUDA_CHECK(cudaPointerGetAttributes(&attributes,
            model->weight(weights.token_embedding).view.parts.front().parent->data));
        if (attributes.device != primary.device) {
            throw std::runtime_error("primary model weights were materialized on the peer");
        }
        CUDA_CHECK(cudaPointerGetAttributes(&attributes,
            model->tensor_parallel_weights().front().up->parent.data));
        if (attributes.device != 1) {
            throw std::runtime_error("peer model weights were materialized on the primary");
        }
        CUDA_CHECK(cudaPointerGetAttributes(&attributes,
            model->tensor_parallel_weights().front().output->parent.data));
        if (attributes.device != 1) {
            throw std::runtime_error("peer mixer output weights were materialized on the primary");
        }
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, model->tensor_parallel_head()->parent.data));
        if (attributes.device != 1) {
            throw std::runtime_error("peer vocabulary weights were materialized on the primary");
        }
        model.reset();
        CUDA_CHECK(cudaGetDevice(&current));
        if (current != caller) {
            throw std::runtime_error("model destruction changed caller device");
        }
    }
}

EngineOptions options(const char* artifact, bool parallel) {
    EngineOptions out;
    out.artifact_path = artifact;
    out.tensor_parallel_device = parallel ? std::optional<int>{1} : std::nullopt;
    out.max_context = 2048;
    out.kv_capacity = KvCapacityPolicy::explicit_capacity(4096);
    out.prefill_chunk = 256;
    out.kv_cache = KvCacheStorage::Int8Group64;
    out.context_cache.host_state_slots = 0;
    out.context_cache.host_kv_capacity_bytes = 0;
    return out;
}

struct ScoreResults {
    std::vector<float> window;
    float singleton = 0;
};

ScoreResults scores(const char* artifact, bool parallel, std::vector<TokenId>& tokens) {
    auto config = options(artifact, parallel);
    config.purpose = EnginePurpose::CausalScoring;
    Engine engine(config);
    if (tokens.empty()) {
        std::string text;
        while (tokens.size() < 1153) {
            text += "The same mathematical model is evaluated on one or two GPUs. "
                    "Independent requests must never share mutable inference state.\n";
            tokens = engine.tokenize_text(text);
        }
        tokens.resize(1153);
    }
    auto result = engine.score_tokens(tokens, 1);
    const auto repeated = engine.score_tokens(tokens, 1);
    if (result != repeated) { throw std::runtime_error("score windows leaked state"); }
    const auto singleton = engine.score_tokens({tokens[0], tokens[1]}, 1);
    if (singleton.size() != 1 ||
        singleton != engine.score_tokens({tokens[0], tokens[1]}, 1)) {
        throw std::runtime_error("single-column scoring returned an invalid shape or leaked state");
    }
    if (parallel && !engine.memory_summary().tensor_parallel) {
        throw std::runtime_error("Engine omitted peer residency");
    }
    return {std::move(result), singleton.front()};
}

RequestOptions request() {
    RequestOptions out;
    out.execution.requested_output_tokens = 8;
    out.execution.sampling.temperature = 0;
    out.execution.allow_prefix_reuse = false;
    return out;
}

using TokenBatches = std::vector<std::vector<TokenId>>;

TokenBatches eager_generation(const char* artifact, const TokenBatches& prompts) {
    auto config = options(artifact, true);
    config.max_context = 262144;
    config.kv_capacity = KvCapacityPolicy::explicit_capacity(config.max_context);
    config.use_cuda_graph = false;
    Engine engine(config);
    const auto capacity = engine.memory_summary();
    if (capacity.max_context != 262144 || capacity.kv_capacity < 262144 ||
        capacity.kv_payload_bytes == 0) {
        throw std::runtime_error("dual-GPU Engine did not allocate the full shared 262144-token KV pool");
    }
    std::cout << "shared KV capacity=" << capacity.kv_capacity
              << " payload_bytes=" << capacity.kv_payload_bytes << '\n';
    TokenBatches output;
    for (const auto& prompt : prompts) {
        output.push_back(
            engine.generate(engine.prepare_tokens(prompt), request()).generated_token_ids);
    }
    engine.reset_memory_peaks();
    const auto reset = engine.memory_summary();
    if (!reset.tensor_parallel || reset.tensor_parallel->workspace.peak_used_bytes != 0) {
        throw std::runtime_error("peer workspace peak was not reset");
    }
    const auto repeated = engine.generate(engine.prepare_tokens(prompts.front()), request());
    if (repeated.generated_token_ids != output.front() ||
        engine.memory_summary().tensor_parallel->workspace.peak_used_bytes == 0) {
        throw std::runtime_error("peer workspace was not reused after resetting memory peaks");
    }
    return output;
}

void graph_concurrency(const char* artifact, const TokenBatches& prompts,
                       const TokenBatches& reference) {
    auto config = options(artifact, true);
    config.max_context = 512;
    config.max_concurrency = 8;
    config.context_cache.device_state_slots = 4;
    Engine engine(config);
    for (unsigned batch = 1; batch <= 8; ++batch) {
        std::vector<GenerationHandle> handles;
        for (unsigned i = 0; i < batch; ++i) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompts[i]), request()));
        }
        {
            auto cancelled = engine.submit(engine.prepare_tokens(prompts.back()), request());
        }
        for (unsigned i = 0; i < batch; ++i) {
            const auto result = handles[i].wait();
            if (result.generated_token_ids != reference[i]) {
                throw std::runtime_error("dual-GPU graph/concurrent generation differs from eager");
            }
        }
    }
    auto cached_request = request();
    cached_request.execution.allow_prefix_reuse = true;
    cached_request.stop.include_model_defaults = false;
    const auto cached = engine.generate(engine.prepare_tokens(prompts.front()), cached_request);
    if (cached.generated_token_ids.size() != 8) {
        throw std::runtime_error("prefix head fixture did not generate eight tokens");
    }
    auto endpoint = prompts.front();
    endpoint.insert(endpoint.end(), cached.generated_token_ids.begin(),
                    cached.generated_token_ids.end() - 1);
    cached_request.execution.requested_output_tokens = 2;
    const auto resumed = engine.generate(engine.prepare_tokens(endpoint), cached_request);
    if (resumed.reused_prompt_tokens != endpoint.size() ||
        resumed.generated_token_ids.size() != 2 ||
        resumed.generated_token_ids.front() != cached.generated_token_ids.back()) {
        throw std::runtime_error("zero-suffix prefix reuse did not preserve full-vocabulary projection");
    }
    if (!engine.is_available()) { throw std::runtime_error("cancellation made Engine unavailable"); }
}
} // namespace

int main() {
    const auto* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact || !std::getenv("NINFER_TEST_TENSOR_PARALLEL")) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT and NINFER_TEST_TENSOR_PARALLEL=1\n";
        return 77;
    }
    try {
        materialization_binding(artifact);
        std::vector<TokenId> tokens;
        const auto single = scores(artifact, false, tokens);
        const auto parallel = scores(artifact, true, tokens);
        double error2 = 0, max_error = 0;
        if (single.window.size() != parallel.window.size()) {
            throw std::runtime_error("score length mismatch");
        }
        for (std::size_t i = 0; i < single.window.size(); ++i) {
            const auto error = std::abs(double(single.window[i]) - parallel.window[i]);
            if (!std::isfinite(parallel.window[i])) { throw std::runtime_error("non-finite score"); }
            error2 += error * error;
            max_error = std::max(max_error, error);
        }
        const auto rms = std::sqrt(error2 / single.window.size());
        const auto singleton_error = std::abs(single.singleton - parallel.singleton);
        std::cout << "single/dual logprob rms=" << rms << " max=" << max_error << '\n';
        std::cout << "single-column logprob absolute difference=" << singleton_error << '\n';
        if (rms > 0.03 || max_error > 0.2 || !std::isfinite(parallel.singleton) ||
            singleton_error > 0.2) {
            throw std::runtime_error("single/dual model logprob equivalence failed");
        }
        TokenBatches prompts;
        for (int lane = 0; lane < 8; ++lane) {
            prompts.emplace_back(tokens.begin() + 11 * lane, tokens.begin() + 11 * lane + 48);
        }
        const auto eager = eager_generation(artifact, prompts);
        graph_concurrency(artifact, prompts, eager);
        std::cout << "PASS dual-GPU Engine scoring, generation, graph, logical context, cancellation, B1..8\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
