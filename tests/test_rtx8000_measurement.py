from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.bench.compare_causal_scores import compare, validate_report
from tools.bench.prepare_score_plan import plan_windows
from tools.bench.run_ninfer_bench_matrix import (
    build_cases,
    load_bench_report,
    main as matrix_main,
    max_prompt_in_cases,
    run_command,
)


def scores(backend: str = "ninfer") -> dict:
    return {
        "schema_version": 2, "artifact_type": "ninfer_paired_scores", "backend": backend,
        "execution": {"context": 4, "stride": 2, "max_tokens_per_stream": 5,
                      "special_tokens_added": False,
                      "kv_dtype": "bf16" if backend == "ninfer" else "bf16-fp16",
                      "kv_key_dtype": "bf16", "kv_value_dtype": "fp16"},
        "streams": [{
            "id": "sample", "domain": "english", "text": "a b c d e",
            "tokens": [1, 2, 3, 4, 5], "full_input_tokens": 5,
            "windows": [
                {"input_begin": 0, "input_end": 4, "target_begin": 1, "target_end": 4,
                 "first_target": 1, "logprobs": [-1.0, -2.0, -3.0]},
                {"input_begin": 1, "input_end": 5, "target_begin": 4, "target_end": 5,
                 "first_target": 3, "logprobs": [-4.0]},
            ],
        }],
    }


def test_paired_scores_weight_targets_not_windows() -> None:
    result = compare(scores(), scores("llama.cpp"), min_scored_tokens=4)
    assert result["passed"]
    assert result["overall"]["candidate_mean_nll"] == 2.5
    assert result["overall"]["perplexity_ratio"] == 1


@pytest.mark.parametrize("kind", ["token", "text", "window", "stream", "incomplete", "duplicate"])
def test_protocol_mismatch_rejected(kind: str) -> None:
    reference = scores("llama.cpp")
    stream = reference["streams"][0]
    if kind == "token":
        stream["tokens"][2] = 6
    elif kind == "text":
        stream["text"] += "!"
    elif kind == "window":
        stream["windows"][1]["first_target"] = 2
    elif kind == "stream":
        stream["id"] = "other"
    elif kind == "incomplete":
        stream["windows"].pop()
    else:
        reference["streams"].append(copy.deepcopy(stream))
    with pytest.raises(ValueError):
        compare(scores(), reference, min_scored_tokens=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.1, False, True, None])
def test_invalid_log_probability_rejected(value: float) -> None:
    report = scores()
    report["streams"][0]["windows"][0]["logprobs"][0] = value
    with pytest.raises(ValueError):
        validate_report(report)


def test_precision_mismatch_requires_explicit_acknowledgement() -> None:
    reference = scores("llama.cpp")
    reference["execution"]["kv_dtype"] = "fp16"
    reference["execution"]["kv_key_dtype"] = "fp16"
    with pytest.raises(ValueError, match="precision"):
        compare(scores(), reference)
    result = compare(scores(), reference, allow_kv_mismatch=True, min_scored_tokens=4)
    assert result["passed"]
    assert result["kv_precision_mismatch_acknowledged"]


def test_same_logical_bf16_label_does_not_hide_different_value_storage() -> None:
    reference = scores("llama.cpp")
    reference["execution"].update(kv_dtype="bf16", kv_value_dtype="bf16")
    with pytest.raises(ValueError, match="precision"):
        compare(scores(), reference, min_scored_tokens=4)
    result = compare(scores(), reference, allow_kv_mismatch=True, min_scored_tokens=4)
    assert result["kv_precision_mismatch_acknowledged"]


def test_inconsistent_physical_storage_metadata_rejected() -> None:
    report = scores()
    report["execution"]["kv_value_dtype"] = "bf16"
    with pytest.raises(ValueError, match="physical"):
        validate_report(report)


def test_degradation_and_insufficient_work_are_failures() -> None:
    candidate = scores()
    candidate["streams"][0]["windows"][0]["logprobs"][0] -= 1
    result = compare(candidate, scores("llama.cpp"), min_scored_tokens=10,
                     max_domain_nll_delta=0.10)
    assert not result["passed"]
    assert len(result["failures"]) == 3


def test_domain_degradation_cannot_hide_in_improved_overall() -> None:
    candidate, reference = scores(), scores("llama.cpp")
    for report in (candidate, reference):
        second = copy.deepcopy(report["streams"][0])
        second.update(id="second", domain="code")
        report["streams"].append(second)
    candidate["streams"][0]["windows"][0]["logprobs"][0] -= 1
    candidate["streams"][1]["windows"][0]["logprobs"][0] += 1
    result = compare(candidate, reference, min_scored_tokens=8, max_domain_nll_delta=0.10)
    assert result["overall"]["mean_nll_delta"] == 0
    assert not result["passed"]


def test_default_quality_limit_is_two_percent_nll_and_reports_window_uncertainty() -> None:
    candidate = scores()
    candidate["streams"][0]["windows"][0]["logprobs"][0] -= 0.12
    result = compare(candidate, scores("llama.cpp"), min_scored_tokens=4)
    assert not result["passed"]
    assert result["thresholds"]["max_mean_nll_delta"] == 0.02
    assert result["thresholds"]["max_domain_nll_delta"] is None
    assert len(result["paired_windows"]) == 2
    uncertainty = result["overall"]["paired_window_uncertainty"]
    assert uncertainty["paired_windows"] == 2
    low, high = uncertainty["stratified_window_bootstrap_interval_95"]
    assert low <= result["overall"]["mean_nll_delta"] <= high
    assert uncertainty["bootstrap_repetitions"] == 2000


def test_one_window_never_claims_uncertainty_interval() -> None:
    candidate, reference = scores(), scores("llama.cpp")
    for report in (candidate, reference):
        stream = report["streams"][0]
        stream["tokens"] = stream["tokens"][:4]
        stream["full_input_tokens"] = 4
        stream["windows"] = stream["windows"][:1]
    result = compare(candidate, reference, min_scored_tokens=3)
    uncertainty = result["domains"]["english"]["paired_window_uncertainty"]
    assert uncertainty["stratified_window_bootstrap_interval_95"] is None
    assert uncertainty["bootstrap_repetitions"] == 0
    assert uncertainty["token_weighted_window_delta_stddev"] is None


@pytest.mark.parametrize("field,value", [
    ("context", 4.0), ("stride", True), ("max_tokens_per_stream", float("nan")),
    ("special_tokens_added", True), ("kv_dtype", None), ("context", 2**40),
])
def test_malformed_execution_protocol_rejected(field: str, value: object) -> None:
    report = scores()
    report["execution"][field] = value
    with pytest.raises(ValueError):
        validate_report(report)


def test_silently_truncated_prefix_cannot_pass_even_if_both_backends_agree() -> None:
    candidate, reference = scores(), scores("llama.cpp")
    for report in (candidate, reference):
        stream = report["streams"][0]
        stream["tokens"].pop()
        stream["windows"].pop()
    with pytest.raises(ValueError, match="declared token limit"):
        compare(candidate, reference, min_scored_tokens=1)


@pytest.mark.parametrize("value", [float("inf"), 5.0, True])
def test_invalid_full_stream_count_rejected(value: object) -> None:
    report = scores()
    report["streams"][0]["full_input_tokens"] = value
    with pytest.raises(ValueError):
        validate_report(report)


def test_float_window_bound_rejected() -> None:
    report = scores()
    report["streams"][0]["windows"][0]["first_target"] = 1.0
    with pytest.raises(ValueError, match="integers"):
        validate_report(report)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_imported_comparator_rejects_invalid_threshold(value: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        compare(scores(), scores("llama.cpp"), max_mean_nll_delta=value, min_scored_tokens=1)


def test_cpu_reference_first_plan_scores_each_target_once_at_tail_boundaries() -> None:
    for length in range(2, 20):
        for context in range(2, 8):
            for stride in range(1, context):
                windows = plan_windows(length, context, stride)
                targets = [position for window in windows
                           for position in range(window["target_begin"], window["target_end"])]
                assert targets == list(range(1, length))
                report = scores()
                report["execution"].update(context=context, stride=stride, max_tokens_per_stream=0)
                stream = report["streams"][0]
                stream.update(tokens=list(range(length)), full_input_tokens=length)
                stream["windows"] = [
                    {**window, "logprobs": [-1.0] * (window["target_end"] - window["target_begin"])}
                    for window in windows
                ]
                validate_report(report)


@pytest.mark.parametrize("tokens,context,stride", [(1, 4, 2), (5, 1, 1), (5, 4, 0), (5, 4, 4)])
def test_cpu_plan_invalid_window_protocol_rejected(tokens: int, context: int, stride: int) -> None:
    with pytest.raises(ValueError):
        plan_windows(tokens, context, stride)


@pytest.mark.parametrize("preset,tokens", [
    ("rtx8000", 4096), ("rtx8000-long", 32768), ("rtx8000-capacity", 16),
])
def test_target_presets_have_exact_work_and_no_speculation(preset: str, tokens: int) -> None:
    cases = build_cases(preset)
    assert max_prompt_in_cases(cases) == tokens
    assert all("--spec" not in case.args for case in cases)
    assert cases[0].repetitions == (1 if preset.endswith("capacity") else 5)
    if preset == "rtx8000-long":
        assert "32768,1024" in cases[0].args
        assert "262144" in cases[0].args


def test_matrix_dry_run_forwards_precision_and_records_provenance(tmp_path: Path) -> None:
    model, corpus = tmp_path / "model.ninfer", tmp_path / "corpus.ids"
    model.write_bytes(b"model")
    corpus.write_text("1 " * 4096)
    output = tmp_path / "results"
    assert matrix_main([
        "--preset", "rtx8000", "--weights", str(model), "--corpus", str(corpus),
        "--output-dir", str(output), "--kv-dtype", "int8", "--dry-run", "--no-build",
    ]) == 0
    report = json.loads((output / "manifest.json").read_text())
    assert report["primary_mtp_draft_tokens"] == 0
    assert len(report["corpus_sha256"]) == 64
    assert "--kv-dtype" in report["commands"][0]["command"]


def test_competing_gpu_workload_prevents_launch(tmp_path: Path) -> None:
    with patch("tools.bench.run_ninfer_bench_matrix.gpu_snapshot",
               return_value={"compute_processes": [["gpu", "999", "foreign", "20"]]}), \
         patch("tools.bench.run_ninfer_bench_matrix.subprocess.Popen") as launch:
        with pytest.raises(RuntimeError, match="competing"):
            run_command(["must-not-run"], tmp_path / "out", tmp_path / "err", "0")
        launch.assert_not_called()


def test_failed_rerun_does_not_retain_stale_native_report(tmp_path: Path) -> None:
    model, corpus = tmp_path / "model.ninfer", tmp_path / "corpus.ids"
    model.write_bytes(b"model")
    corpus.write_text("1 " * 4096)
    output = tmp_path / "results"
    stale = output / "json/native/pp512_pp4096_tg128.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"stale":true}')
    with patch("tools.bench.run_ninfer_bench_matrix.run_command", return_value=1):
        assert matrix_main([
            "--preset", "rtx8000", "--bench", "/usr/bin/false", "--weights", str(model),
            "--corpus", str(corpus), "--output-dir", str(output), "--no-build",
        ]) == 1
    assert not stale.exists()
    assert json.loads((output / "summary.json").read_text()) == []


@pytest.mark.parametrize("bad", ["empty", "missing_rep", "nan_rate", "missing_rate", "wrong_count"])
def test_incomplete_native_records_fail_visibly(tmp_path: Path, bad: str) -> None:
    report = {
        "schema_version": 16, "artifact_type": "ninfer_bench_report", "tool": "ninfer_bench",
        "config": {"repetitions": 1},
        "tests": [{"kind": "pp", "requested_output_tokens": 1, "prefill_tok_s_mean": 2.0,
                   "reps": [{"generated_output_tokens": 1}]}],
    }
    if bad == "empty":
        report["tests"] = []
    elif bad == "missing_rep":
        report["tests"][0]["reps"] = []
    elif bad == "nan_rate":
        report["tests"][0]["prefill_tok_s_mean"] = float("nan")
    elif bad == "missing_rate":
        del report["tests"][0]["prefill_tok_s_mean"]
    else:
        report["tests"][0]["reps"][0]["generated_output_tokens"] = 2
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_bench_report(path)


def test_tensor_parallel_scoring_and_concurrency_cli(tmp_path: Path) -> None:
    from tools.bench.run_serve_concurrency import build_points, parse_args, server_command, validate_args
    args = parse_args([
        "--artifact", "qwen=test.ninfer", "--mode", "mtp0", "--suite", "decode-saturation",
        "--concurrency", "1", "--concurrency", "8", "--tensor-parallel-device", "1",
        "--output", str(tmp_path),
    ])
    validate_args(args)
    points = build_points([("qwen", Path("test.ninfer"))], args)
    assert [point.concurrency for point in points] == [1, 8]
    command = server_command(Path("ninfer-serve"), points[1], tmp_path / "log", args)
    assert command[command.index("--tensor-parallel-device") + 1] == "1"
    assert command[command.index("--max-concurrency") + 1] == "8"
