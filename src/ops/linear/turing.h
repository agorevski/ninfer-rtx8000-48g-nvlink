#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

inline void require_linear_weight_support(QType qtype, const char* operation) {
#if defined(NINFER_SM75)
    if (qtype == QType::FP8_E4M3FN_ROW_BF16 || qtype == QType::NVFP4)
        throw std::invalid_argument(std::string(operation) +
                                    ": SM75 supports BF16 and groupwise integer projection weights");
#else
    (void)qtype;
    (void)operation;
#endif
}

std::size_t turing_linear_workspace_capacity_bytes(QType qtype, std::int32_t input_rows,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens);
bool turing_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                    WorkspaceArena* workspace = nullptr);
void turing_linear_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
bool turing_linear_add(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                        WorkspaceArena* workspace = nullptr);
bool turing_linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                           WorkspaceArena* workspace = nullptr);

} // namespace ninfer::ops::detail
