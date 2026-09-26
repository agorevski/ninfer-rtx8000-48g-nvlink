#include "tools/bench/score_protocol.h"

#include <iostream>
#include <limits>

using json = nlohmann::json;
namespace scoring = ninfer::bench::scoring;

int main() {
    try {
        json report{
            {"schema_version", 2}, {"artifact_type", "ninfer_paired_scores"}, {"backend", "ninfer"},
            {"execution", {{"context", 4}, {"stride", 2}, {"max_tokens_per_stream", 5},
                           {"special_tokens_added", false}, {"kv_dtype", "bf16"},
                           {"kv_key_dtype", "bf16"}, {"kv_value_dtype", "fp16"}}},
            {"streams", json::array({
                {{"id", "sample"}, {"domain", "code"}, {"text", "a b c d e"},
                 {"tokens", {1, 2, 3, 4, 5}}, {"full_input_tokens", 5},
                 {"windows", json::array({
                     {{"input_begin", 0}, {"input_end", 4}, {"target_begin", 1}, {"target_end", 4},
                      {"first_target", 1}, {"logprobs", {-1.0, -2.0, -3.0}}},
                     {{"input_begin", 1}, {"input_end", 5}, {"target_begin", 4}, {"target_end", 5},
                      {"first_target", 3}, {"logprobs", {-4.0}}},
                 })}},
            })},
        };
        scoring::validate_report(report);
        const auto rejects = [](const json& invalid) {
            try { scoring::validate_report(invalid); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("malformed score report accepted");
        };
        for (const auto& value : {json(false), json(nullptr), json(0.1),
                                 json(std::numeric_limits<double>::quiet_NaN()),
                                 json(-std::numeric_limits<double>::infinity())}) {
            auto invalid = report;
            invalid["streams"][0]["windows"][0]["logprobs"][0] = value;
            rejects(invalid);
        }
        auto invalid = report;
        invalid["streams"][0]["tokens"].erase(4);
        invalid["streams"][0]["windows"].erase(1);
        rejects(invalid);
        invalid = report;
        invalid["streams"][0]["windows"][1]["first_target"] = 2;
        rejects(invalid);
        invalid = report;
        invalid["streams"][0]["tokens"][0] = 1.0;
        rejects(invalid);
        invalid = report;
        invalid["execution"]["context"] = 4.0;
        rejects(invalid);
        invalid = report;
        invalid["streams"].push_back(invalid["streams"][0]);
        rejects(invalid);
        auto plan = report;
        plan["schema_version"] = 1;
        plan["artifact_type"] = "ninfer_paired_score_plan";
        for (auto& window : plan["streams"][0]["windows"]) { window.erase("logprobs"); }
        scoring::validate_report(plan, true);
        rejects(plan);
        if (scoring::reference_batch_rows(248320) != 128 ||
            scoring::reference_batch_rows(32) != 1024 ||
            std::uint64_t(scoring::reference_batch_rows(1000000)) * 1000000 * sizeof(float) >
                scoring::kReferenceLogitsBudget) {
            throw std::runtime_error("reference logits budget is not bounded");
        }
        try {
            (void)scoring::reference_batch_rows(0);
            throw std::logic_error("empty vocabulary accepted");
        } catch (const std::invalid_argument&) {}
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
