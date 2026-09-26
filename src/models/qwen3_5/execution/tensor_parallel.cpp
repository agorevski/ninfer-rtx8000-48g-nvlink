#include "models/qwen3_5/execution/tensor_parallel.h"

#include "core/layout.h"
#include "core/peer_copy.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void copy(const Tensor& source, Tensor& destination, cudaStream_t stream, bool kernel) {
    CUDA_CHECK(kernel
        ? copy_peer_kernel_async(destination.data, source.data, destination.bytes(), stream)
        : copy_device_async(destination.data, source.data, destination.bytes(), stream));
}

void copy_rows(const Tensor& source, std::int32_t source_row, Tensor& destination,
               std::int32_t destination_row, std::int32_t rows, cudaStream_t stream,
               bool kernel = false) {
    const auto bytes = dtype_size(DType::BF16);
    auto* dst = static_cast<std::byte*>(destination.data) + destination_row * bytes;
    const auto* src = static_cast<const std::byte*>(source.data) + source_row * bytes;
    CUDA_CHECK(kernel
        ? copy_peer_kernel_2d_async(dst, destination.nb[1], src, source.nb[1], rows * bytes,
                                    source.ne[1], stream)
        : copy_device_2d_async(dst, destination.nb[1], src, source.nb[1], rows * bytes,
                               source.ne[1], stream));
}

} // namespace

std::size_t tensor_parallel_ffn_workspace_bytes(const TensorParallelDenseParameters& parameters,
                                               int rank, std::int32_t first,
                                               std::int32_t last) {
    if ((rank != 0 && rank != 1) || first <= 0 || last < first) {
        throw std::invalid_argument("tensor parallel FFN: invalid workspace extent");
    }
    const auto& input = parameters.input[rank];
    const auto& down = parameters.down[rank];
    WorkspaceLayoutBuilder layout;
    if (rank == 1) { (void)layout.alloc(DType::BF16, {input.weight.k, last}); }
    (void)layout.alloc(DType::BF16, {input.weight.n, last});
    (void)layout.alloc(DType::BF16, {input.weight.n, last});
    (void)layout.alloc(DType::BF16, {input.weight.n, last});
    (void)layout.alloc(DType::BF16, {down.weight.n, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            input.weight.qtype, input.weight.n, input.weight.k, input.policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down.weight.qtype, down.weight.n, down.weight.k, down.policy, first, last));
    }
    return layout.peak_bytes(1);
}

std::size_t tensor_parallel_projection_workspace_bytes(
    const TensorParallelProjectionParameters& parameters, int rank, bool add) {
    if (rank != 0 && rank != 1) {
        throw std::invalid_argument("tensor parallel projection: invalid rank");
    }
    const auto& p = parameters.ranks[rank];
    WorkspaceLayoutBuilder layout;
    if (rank == 1) {
        (void)layout.alloc(DType::BF16, {p.weight.k, 1});
        (void)layout.alloc(DType::BF16, {p.weight.n, 1});
    }
    (void)layout.alloc_bytes(add
        ? ops::linear_add_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                    p.policy, 1, 1)
        : ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                p.policy, 1, 1));
    return layout.peak_bytes(1);
}

std::size_t output_projection_workspace_bytes(
    const LinearParameters& full,
    const std::optional<TensorParallelProjectionParameters>& partition,
    bool add, std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("output projection: invalid column interval");
    }
    std::size_t bytes = 0;
    if (partition && first == 1) {
        bytes = tensor_parallel_projection_workspace_bytes(*partition, 0, add);
        first = 2;
    }
    if (first <= last) {
        bytes = std::max(bytes, add
            ? ops::linear_add_workspace_capacity_bytes(full.weight.qtype, full.weight.n,
                                                        full.weight.k, full.policy, first, last)
            : ops::linear_workspace_capacity_bytes(full.weight.qtype, full.weight.n,
                                                    full.weight.k, full.policy, first, last));
    }
    return bytes;
}

TensorParallelProjections::TensorParallelProjections(DeviceContext& primary, int peer_device,
                                                   std::size_t peer_workspace_bytes)
    : primary_(primary), peer_([peer_device] {
          DeviceGuard guard(peer_device);
          return DeviceContext(peer_device);
      }()), input_ready_(primary), gate_ready_(primary),
      up_ready_(peer_), peer_done_(peer_) {
    if (peer_device == primary.device || peer_workspace_bytes == 0) {
        throw std::invalid_argument("tensor parallel projections require a distinct peer and workspace");
    }
    DeviceGuard guard(peer_device);
    peer_workspace_ = std::make_unique<WorkspaceArena>(peer_workspace_bytes);
}

TensorParallelProjections::~TensorParallelProjections() noexcept {
    drain();
}

void TensorParallelProjections::drain() noexcept {
    (void)cudaStreamSynchronize(primary_.stream);
    (void)cudaStreamSynchronize(peer_.stream);
}

void TensorParallelProjections::drain_after_failure() noexcept {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(primary_.stream, &capture) == cudaSuccess &&
        capture == cudaStreamCaptureStatusNone) {
        drain();
    }
}

ArenaMemorySummary TensorParallelProjections::workspace_summary() const noexcept {
    return {peer_workspace_->capacity(), peer_workspace_->used(),
            std::max(peer_workspace_->peak_used(), graph_workspace_peak_bytes_)};
}

void TensorParallelProjections::reset_memory_peak() noexcept {
    peer_workspace_->reset_peak();
    graph_workspace_peak_bytes_ = 0;
}

void TensorParallelProjections::record_graph_execution(std::uint32_t batch_size) {
    if (batch_size == 0 || batch_size > kMaximumConcurrency ||
        decode_workspace_bytes_[batch_size] == 0) {
        throw std::logic_error("tensor parallel graph has no prepared workspace extent");
    }
    graph_workspace_peak_bytes_ =
        std::max(graph_workspace_peak_bytes_, decode_workspace_bytes_[batch_size]);
}

void TensorParallelProjections::ffn(const Tensor& hidden,
                            const TensorParallelDenseParameters& parameters,
                            Tensor& residual, WorkspaceArena& primary_workspace, bool decode) {
    DeviceGuard primary_guard(primary_.device);
    const auto columns = hidden.ne[1];
    const auto width = hidden.ne[0];
    const auto intermediate = parameters.input[0].weight.n;
    const auto half = parameters.down[0].weight.n;
    if (hidden.dtype != DType::BF16 || residual.dtype != DType::BF16 ||
        !hidden.is_contiguous() || !residual.is_contiguous() || columns <= 0 ||
        (decode && columns > static_cast<std::int32_t>(kMaximumConcurrency)) ||
        !hidden.data || !residual.data || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        residual.ne[2] != 1 || residual.ne[3] != 1 ||
        residual.ne[0] != width || residual.ne[1] != columns || width != 2 * half ||
        parameters.input[0].weight.k != width ||
        parameters.input[1].weight.k != width ||
        parameters.input[1].weight.n != intermediate ||
        parameters.down[0].weight.k != intermediate ||
        parameters.down[1].weight.k != intermediate ||
        parameters.down[1].weight.n != half) {
        throw std::invalid_argument("tensor parallel FFN: incompatible dense geometry");
    }
    if (columns <= static_cast<std::int32_t>(kMaximumConcurrency)) {
        auto& bytes = decode_workspace_bytes_[columns];
        bytes = std::max(bytes, tensor_parallel_ffn_workspace_bytes(parameters, 1, columns, columns));
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!decode) { CUDA_CHECK(cudaStreamIsCapturing(primary_.stream, &capture)); }
    // Peer transfers stay kernel nodes during capture; ordinary eager prefill retains DMA.
    const bool kernel_copies = decode || capture != cudaStreamCaptureStatusNone;
    auto primary_scope = primary_workspace.scope();
    auto peer_scope = peer_workspace_->scope();
    auto& remote = *peer_workspace_;
    Tensor gate = primary_workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor up = primary_workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor product = primary_workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor output = primary_workspace.alloc(DType::BF16, {half, columns});
    Tensor peer_hidden = remote.alloc(DType::BF16, {width, columns});
    Tensor peer_gate = remote.alloc(DType::BF16, {intermediate, columns});
    Tensor peer_up = remote.alloc(DType::BF16, {intermediate, columns});
    Tensor peer_product = remote.alloc(DType::BF16, {intermediate, columns});
    Tensor peer_output = remote.alloc(DType::BF16, {half, columns});

    try {
        input_ready_.record(primary_.stream);
        {
            DeviceGuard guard(peer_.device);
            input_ready_.wait(peer_.stream);
            copy(hidden, peer_hidden, peer_.stream, kernel_copies);
            copy_rows(residual, half, peer_output, 0, half, peer_.stream, kernel_copies);
            auto scratch = remote.scope();
            ops::linear(peer_hidden, parameters.input[1].weight, peer_up,
                         parameters.input[1].policy, remote, peer_.stream);
            up_ready_.record(peer_.stream);
        }
        copy_rows(residual, 0, output, 0, half, primary_.stream);
        {
            auto scratch = primary_workspace.scope();
            ops::linear(hidden, parameters.input[0].weight, gate, parameters.input[0].policy,
                         primary_workspace, primary_.stream);
        }
        gate_ready_.record(primary_.stream);
        {
            DeviceGuard guard(peer_.device);
            gate_ready_.wait(peer_.stream);
            copy(gate, peer_gate, peer_.stream, kernel_copies);
            ops::silu_mul(peer_gate, peer_up, peer_product, peer_.stream);
            auto scratch = remote.scope();
            ops::linear_add(peer_product, parameters.down[1].weight, peer_output,
                             parameters.down[1].policy, remote, peer_.stream);
            peer_done_.record(peer_.stream);
        }
        up_ready_.wait(primary_.stream);
        copy(peer_up, up, primary_.stream, kernel_copies);
        ops::silu_mul(gate, up, product, primary_.stream);
        {
            auto scratch = primary_workspace.scope();
            ops::linear_add(product, parameters.down[0].weight, output, parameters.down[0].policy,
                             primary_workspace, primary_.stream);
        }
        copy_rows(output, 0, residual, 0, half, primary_.stream);
        peer_done_.wait(primary_.stream);
        copy_rows(peer_output, 0, residual, half, half, primary_.stream, kernel_copies);
    } catch (...) {
        drain_after_failure();
        throw;
    }
}

void TensorParallelProjections::project(
    const Tensor& hidden, const TensorParallelProjectionParameters& parameters,
    Tensor& output, WorkspaceArena& primary_workspace) {
    project_impl(hidden, parameters, output, primary_workspace, false);
}

void TensorParallelProjections::project_add(
    const Tensor& hidden, const TensorParallelProjectionParameters& parameters,
    Tensor& residual, WorkspaceArena& primary_workspace) {
    project_impl(hidden, parameters, residual, primary_workspace, true);
}

void TensorParallelProjections::project_impl(
    const Tensor& hidden, const TensorParallelProjectionParameters& parameters,
    Tensor& output, WorkspaceArena& primary_workspace, bool add) {
    DeviceGuard primary_guard(primary_.device);
    const auto& local = parameters.ranks[0];
    const auto& remote_parameters = parameters.ranks[1];
    const auto half = local.weight.n;
    if (hidden.dtype != DType::BF16 || output.dtype != DType::BF16 ||
        !hidden.is_contiguous() || !output.is_contiguous() || !hidden.data || !output.data ||
        hidden.ne[1] != 1 || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        output.ne[1] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        hidden.ne[0] != local.weight.k || hidden.ne[0] != remote_parameters.weight.k ||
        half <= 0 || output.ne[0] != 2LL * half || half != remote_parameters.weight.n) {
        throw std::invalid_argument("tensor parallel output projection requires a single column");
    }
    decode_workspace_bytes_[1] = std::max(decode_workspace_bytes_[1],
        tensor_parallel_projection_workspace_bytes(parameters, 1, add));
    auto local_scope = primary_workspace.scope();
    auto remote_scope = peer_workspace_->scope();
    auto& remote = *peer_workspace_;
    Tensor remote_input = remote.alloc(DType::BF16, {hidden.ne[0], 1});
    Tensor remote_output = remote.alloc(DType::BF16, {half, 1});
    Tensor local_output = output.slice(0, 0, half);
    Tensor upper_output = output.slice(0, half, half);
    const auto launch = [add](const Tensor& input, const LinearParameters& p, Tensor& destination,
                              WorkspaceArena& scratch, cudaStream_t stream) {
        if (add) {
            ops::linear_add(input, p.weight, destination, p.policy, scratch, stream);
        } else {
            ops::linear(input, p.weight, destination, p.policy, scratch, stream);
        }
    };
    try {
        input_ready_.record(primary_.stream);
        {
            DeviceGuard guard(peer_.device);
            input_ready_.wait(peer_.stream);
            copy(hidden, remote_input, peer_.stream, true);
            if (add) { copy(upper_output, remote_output, peer_.stream, true); }
            launch(remote_input, remote_parameters, remote_output, remote, peer_.stream);
            peer_done_.record(peer_.stream);
        }
        launch(hidden, local, local_output, primary_workspace, primary_.stream);
        peer_done_.wait(primary_.stream);
        copy(remote_output, upper_output, primary_.stream, true);
    } catch (...) {
        drain_after_failure();
        throw;
    }
}

void project_text_head(const Tensor& hidden, const TextParameters& parameters, Tensor& logits,
                       WorkspaceArena& workspace, cudaStream_t stream,
                       TensorParallelProjections* parallel) {
    if (parallel && hidden.ne[1] == 1) {
        parallel->project(hidden, parameters.tensor_parallel_head.value(), logits, workspace);
    } else {
        auto scope = workspace.scope();
        const auto& p = parameters.output_head;
        ops::linear(hidden, p.weight, logits, p.policy, workspace, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
