#include "ops/linear/linear_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer::test::linear;
    if (!cuda_available()) return 77;
    try {
        constexpr std::array columns{
            Invocation{1}, Invocation{2}, Invocation{3}, Invocation{4}, Invocation{5},
            Invocation{6}, Invocation{7}, Invocation{8}, Invocation{9},
            Invocation{3, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
            Invocation{7, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        };
        constexpr std::array head_columns{
            Invocation{1}, Invocation{2}, Invocation{3}, Invocation{4}, Invocation{5},
            Invocation{8},
            Invocation{3, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
            Invocation{4, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        };
        int failures = 0;
        for (const float multiplier : {1.0f, 0x1p20f, 0x1p-32f}) {
            failures += run_shape(
                "Turing batched Q4 Linear", ActivationCompute::A16, make_q4_g64_fp16_weight,
                {34816, 5120, 8121u, Comparison::Sampled, true, columns, multiplier});
            failures += run_shape(
                "Turing batched Q4 TP shard", ActivationCompute::A16, make_q4_g64_fp16_weight,
                {17408, 5120, 8124u, Comparison::Sampled, true, head_columns, multiplier});
            failures += run_shape(
                "Turing batched Q8 head", ActivationCompute::A16, make_q8_g32_fp16_weight,
                {248320, 5120, 8122u, Comparison::Sampled, true, head_columns, multiplier});
        }
        constexpr std::array<std::int32_t, 9> swiglu_columns{1, 2, 3, 4, 5, 6, 7, 8, 9};
        constexpr std::array<std::int32_t, 4> swiglu_graphs{2, 3, 7, 8};
        failures += ninfer::test::linear_swiglu::run_profile(
            "Turing batched Q4 SwiGLU",
            {ninfer::QType::Q4_G64_FP16, 34816, 5120, 17408, 8123u,
             ninfer::test::linear_swiglu::ActivationCompute::A16},
            swiglu_columns, swiglu_graphs);
        std::cout << (failures ? "FAIL" : "PASS") << " Turing batched public routes\n";
        return failures != 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
