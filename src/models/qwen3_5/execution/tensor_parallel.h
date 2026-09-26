#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t tensor_parallel_ffn_workspace_bytes(
    const TensorParallelDenseParameters& parameters, int rank, std::int32_t first,
    std::int32_t last);

[[nodiscard]] std::size_t tensor_parallel_projection_workspace_bytes(
    const TensorParallelProjectionParameters& parameters, int rank, bool add);
[[nodiscard]] std::size_t output_projection_workspace_bytes(
    const LinearParameters& full,
    const std::optional<TensorParallelProjectionParameters>& partition,
    bool add, std::int32_t first, std::int32_t last);

// No request state lives on the peer: every invocation joins the primary stream before
// returning its output. Workspace and stream ownership belong to the containing Program.
class TensorParallelProjections {
public:
    TensorParallelProjections(DeviceContext& primary, int peer_device,
                              std::size_t peer_workspace_bytes);
    ~TensorParallelProjections() noexcept;
    TensorParallelProjections(const TensorParallelProjections&) = delete;
    TensorParallelProjections& operator=(const TensorParallelProjections&) = delete;

    void ffn(const Tensor& hidden, const TensorParallelDenseParameters& parameters,
             Tensor& residual, WorkspaceArena& primary_workspace, bool decode);
    void project(const Tensor& hidden, const TensorParallelProjectionParameters& parameters,
                 Tensor& output, WorkspaceArena& primary_workspace);
    void project_add(const Tensor& hidden, const TensorParallelProjectionParameters& parameters,
                     Tensor& residual, WorkspaceArena& primary_workspace);
    void drain() noexcept;
    void record_graph_execution(std::uint32_t batch_size);
    [[nodiscard]] ArenaMemorySummary workspace_summary() const noexcept;
    void reset_memory_peak() noexcept;

private:
    void drain_after_failure() noexcept;
    void project_impl(const Tensor& hidden, const TensorParallelProjectionParameters& parameters,
                      Tensor& output, WorkspaceArena& primary_workspace, bool add);
    DeviceContext& primary_;
    DeviceContext peer_;
    std::unique_ptr<WorkspaceArena> peer_workspace_;
    CudaCompletionEvent input_ready_;
    CudaCompletionEvent gate_ready_;
    CudaCompletionEvent up_ready_;
    CudaCompletionEvent peer_done_;
    std::array<std::size_t, kMaximumConcurrency + 1> decode_workspace_bytes_{};
    std::size_t graph_workspace_peak_bytes_ = 0;
};

void project_text_head(const Tensor& hidden, const TextParameters& parameters, Tensor& logits,
                       WorkspaceArena& workspace, cudaStream_t stream,
                       TensorParallelProjections* parallel);

} // namespace ninfer::models::qwen3_5::execution
