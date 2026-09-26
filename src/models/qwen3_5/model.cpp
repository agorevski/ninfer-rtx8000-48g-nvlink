#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)) {}

Model::~Model() = default;

void Model::materialize_tensor_parallel(DeviceContext& primary) {
    if (!options_.tensor_parallel_device) { return; }
    if (!std::holds_alternative<DenseConfig>(config_.text.ffn)) {
        throw std::invalid_argument("tensor parallelism requires the dense Qwen architecture");
    }
    if (*options_.tensor_parallel_device == primary.device) {
        throw std::invalid_argument("tensor parallelism requires two distinct devices");
    }
    DeviceGuard restore(primary.device);
    DeviceContext peer = [&] {
        DeviceGuard guard(*options_.tensor_parallel_device);
        return DeviceContext(*options_.tensor_parallel_device);
    }();
    if (primary.compute_capability() != 75 || peer.compute_capability() != 75) {
        throw std::invalid_argument("dense tensor parallelism currently requires two SM75 devices");
    }
    for (const auto pair : {std::pair{primary.device, peer.device},
                            std::pair{peer.device, primary.device}}) {
        int accessible = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, pair.first, pair.second));
        if (!accessible) {
            throw std::invalid_argument("tensor parallel devices require bidirectional peer access");
        }
        DeviceGuard guard(pair.first);
        const auto status = cudaDeviceEnablePeerAccess(pair.second, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); }
        else { CUDA_CHECK(status); }
    }
    primary.synchronize();
    const auto copy_upper_rows = [&](WeightId id) {
        auto rows = weight(id).view;
        if (rows.shape.size() != 2 || rows.shape[0] % 2 != 0) {
            throw std::invalid_argument("tensor parallel projection requires even output rows");
        }
        const auto region = contiguous_weight_region(rows);
        rows.shape[0] /= 2;
        rows.parts = {
            {region.parent, region.begin + (region.end - region.begin) / 2, region.end}};
        return artifact::materialize_weight_rows(rows, peer);
    };
    tensor_parallel_weights_.reserve(weights_.text.layers.size());
    for (const auto& layer : weights_.text.layers) {
        const auto& dense = std::get<DenseWeights>(layer.ffn);
        const auto& up = weight(dense.up).view;
        const auto output = std::visit([](const auto& mixer) { return mixer.output; }, layer.mixer);
        tensor_parallel_weights_.push_back(
            {artifact::materialize_weight_rows(up, peer),
             copy_upper_rows(dense.down), copy_upper_rows(output)});
    }
    tensor_parallel_head_ = copy_upper_rows(weights_.text.output_head);
}

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor};
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace ninfer::models::qwen3_5
