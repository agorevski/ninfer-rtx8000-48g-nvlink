#!/usr/bin/env python3
"""Compare exact-token, exact-window public Engine scores against an independent reference."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


def validate_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(report, dict) or (
        report.get("schema_version") != 2 or report.get("artifact_type") != "ninfer_paired_scores"
    ):
        raise ValueError("unsupported score report")
    streams = {}
    context = report["execution"]["context"]
    stride = report["execution"]["stride"]
    limit = report["execution"]["max_tokens_per_stream"]
    if (type(context) is not int or type(stride) is not int or type(limit) is not int or
        not 2 <= context < 2**31 or not 1 <= stride < context or
        not 0 <= limit < 2**31 or limit == 1):
        raise ValueError("invalid context/stride")
    if report["execution"]["special_tokens_added"] is not False:
        raise ValueError("scoring protocol must not add special tokens")
    storage_profiles = {
        ("ninfer", "bf16"): ("bf16", "fp16"),
        ("ninfer", "int8"): ("int8_g64", "int8_g64"),
        ("llama.cpp", "bf16-fp16"): ("bf16", "fp16"),
        ("llama.cpp", "bf16"): ("bf16", "bf16"),
        ("llama.cpp", "fp16"): ("fp16", "fp16"),
    }
    storage = storage_profiles.get((report.get("backend"), report["execution"]["kv_dtype"]))
    if storage is None:
        raise ValueError("missing/unknown scoring precision")
    if (report["execution"]["kv_key_dtype"], report["execution"]["kv_value_dtype"]) != storage:
        raise ValueError("KV profile disagrees with physical K/V storage precision")
    if not isinstance(report["streams"], list):
        raise ValueError("streams must be a list")
    for stream in report["streams"]:
        if not isinstance(stream, dict) or any(
            not isinstance(stream.get(key), str) or not stream[key]
            for key in ("id", "domain", "text")
        ):
            raise ValueError("invalid stream identity/domain/text")
        if stream["id"] in streams:
            raise ValueError("duplicate stream ID")
        tokens = stream["tokens"]
        if not isinstance(tokens, list) or len(tokens) < 2 or any(
            type(token) is not int or not 0 <= token < 2**31 for token in tokens
        ):
            raise ValueError("invalid token IDs")
        full_count = stream["full_input_tokens"]
        if type(full_count) is not int or not len(tokens) <= full_count < 2**31:
            raise ValueError("invalid full token count")
        if len(tokens) != (min(full_count, limit) if limit else full_count):
            raise ValueError("stream length does not match its declared token limit")
        if not isinstance(stream["windows"], list):
            raise ValueError("windows must be a list")
        target = 1
        for index, window in enumerate(stream["windows"]):
            if not isinstance(window, dict) or any(
                type(window.get(key)) is not int for key in (
                    "input_begin", "input_end", "target_begin", "target_end", "first_target")
            ):
                raise ValueError("window bounds must be integers")
            end = min(len(tokens), context if index == 0 else target + stride)
            begin = max(0, end - context)
            if (window["input_begin"], window["input_end"], window["target_begin"],
                window["target_end"], window["first_target"]) != (
                    begin, end, target, end, target - begin):
                raise ValueError("invalid/noncanonical scoring window")
            scores = window["logprobs"]
            if not isinstance(scores, list) or len(scores) != end - target or not scores:
                raise ValueError("missing or duplicate scored targets")
            if any(type(value) not in (int, float) or not math.isfinite(value) or value > 1e-5
                   for value in scores):
                raise ValueError("invalid/nonfinite log probability")
            target = end
        if target != len(tokens):
            raise ValueError("incomplete scoring stream")
        streams[stream["id"]] = stream
    if not streams:
        raise ValueError("empty score report")
    return streams


def window_uncertainty(windows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        by_domain[window["domain"]].append(window)
    weight = sum(window["scored_tokens"] for window in windows)
    total = sum(window["scored_tokens"] * window["mean_nll_delta"] for window in windows)
    mean = total / weight
    denominator = weight - sum(window["scored_tokens"] ** 2 for window in windows) / weight
    deviation = (math.sqrt(sum(window["scored_tokens"] * (window["mean_nll_delta"] - mean) ** 2
                               for window in windows) / denominator)
                 if denominator > 0 else None)
    leave_one_out = [
        (total - window["scored_tokens"] * window["mean_nll_delta"]) /
        (weight - window["scored_tokens"])
        for window in windows if weight > window["scored_tokens"]
    ]
    interval = None
    enough_windows = all(len(domain) >= 2 for domain in by_domain.values())
    if enough_windows:
        generator = random.Random(0)
        estimates = []
        for _ in range(2000):
            sampled = [generator.choice(domain) for domain in by_domain.values()
                       for _ in range(len(domain))]
            estimates.append(
                sum(window["scored_tokens"] * window["mean_nll_delta"] for window in sampled) /
                sum(window["scored_tokens"] for window in sampled)
            )
        estimates.sort()
        interval = [estimates[49], estimates[1949]]
    return {
        "paired_windows": len(windows),
        "windows_per_domain": {name: len(domain) for name, domain in sorted(by_domain.items())},
        "token_weighted_window_delta_stddev": deviation,
        "leave_one_window_out_mean_delta_range": (
            [min(leave_one_out), max(leave_one_out)] if leave_one_out else None),
        "stratified_window_bootstrap_interval_95": interval,
        "bootstrap_repetitions": 2000 if enough_windows else 0,
        "bootstrap_seed": 0,
        "limitation": (
            "Descriptive paired-window resampling within domains, not a statistical quality "
            "guarantee. Shared streams and overlapping context correlate windows. "
            "The interval is unavailable if any domain has fewer than two windows."
        ),
    }


def compare(candidate: dict[str, Any], reference: dict[str, Any], *,
            max_mean_nll_delta: float = 0.02, max_domain_nll_delta: float | None = None,
            min_scored_tokens: int = 16000, allow_kv_mismatch: bool = False) -> dict[str, Any]:
    if (not all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (max_mean_nll_delta, max_domain_nll_delta) if value is not None)
            or max_mean_nll_delta is None or type(min_scored_tokens) is not int
            or min_scored_tokens < 1):
        raise ValueError("invalid quality acceptance thresholds")
    candidates, references = validate_report(candidate), validate_report(reference)
    if candidate.get("backend") != "ninfer" or reference.get("backend") != "llama.cpp":
        raise ValueError("expected native candidate and independent llama.cpp reference")
    if candidates.keys() != references.keys():
        raise ValueError("stream sets differ")
    for key in ("context", "stride", "max_tokens_per_stream", "special_tokens_added"):
        if candidate["execution"][key] != reference["execution"][key]:
            raise ValueError(f"execution protocol differs: {key}")
    kv_mismatch = any(candidate["execution"][key] != reference["execution"][key]
                      for key in ("kv_key_dtype", "kv_value_dtype"))
    if kv_mismatch and not allow_kv_mismatch:
        raise ValueError("KV precision differs; explicitly acknowledge with --allow-kv-mismatch")
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    paired_windows = []
    for name, current in candidates.items():
        expected = references[name]
        for key in ("domain", "tokens", "text", "full_input_tokens"):
            if current[key] != expected[key]:
                raise ValueError(f"{name}: {key} differs")
        if len(current["windows"]) != len(expected["windows"]):
            raise ValueError(f"{name}: window counts differ")
        for index, (lhs, rhs) in enumerate(zip(current["windows"], expected["windows"], strict=True)):
            for key in ("input_begin", "input_end", "target_begin", "target_end", "first_target"):
                if lhs[key] != rhs[key]:
                    raise ValueError(f"{name}: target windows differ")
            pairs[current["domain"]].extend(
                (-a, -b) for a, b in zip(lhs["logprobs"], rhs["logprobs"], strict=True)
            )
            candidate_nll = -statistics.fmean(lhs["logprobs"])
            reference_nll = -statistics.fmean(rhs["logprobs"])
            paired_windows.append({
                "stream_id": name, "domain": current["domain"], "window_index": index,
                **{key: lhs[key] for key in (
                    "input_begin", "input_end", "target_begin", "target_end", "first_target")},
                "scored_tokens": len(lhs["logprobs"]), "candidate_mean_nll": candidate_nll,
                "reference_mean_nll": reference_nll, "mean_nll_delta": candidate_nll - reference_nll,
            })

    def summary(values: list[tuple[float, float]]) -> dict[str, Any]:
        candidate_nll = statistics.fmean(a for a, _ in values)
        reference_nll = statistics.fmean(b for _, b in values)
        delta = candidate_nll - reference_nll
        absolute = sorted(abs(a - b) for a, b in values)
        return {
            "scored_tokens": len(values), "candidate_mean_nll": candidate_nll,
            "reference_mean_nll": reference_nll, "mean_nll_delta": delta,
            "candidate_perplexity": math.exp(candidate_nll),
            "reference_perplexity": math.exp(reference_nll), "perplexity_ratio": math.exp(delta),
            "token_abs_nll_delta_p50": absolute[int((len(absolute) - 1) * 0.50)],
            "token_abs_nll_delta_p95": absolute[int((len(absolute) - 1) * 0.95)],
            "token_abs_nll_delta_p99": absolute[int((len(absolute) - 1) * 0.99)],
            "token_abs_nll_delta_max": absolute[-1],
        }

    domains = {name: summary(values) for name, values in sorted(pairs.items())}
    overall = summary([pair for values in pairs.values() for pair in values])
    failures = []
    if overall["scored_tokens"] < min_scored_tokens:
        failures.append("insufficient scored targets")
    if overall["mean_nll_delta"] > max_mean_nll_delta:
        failures.append("overall mean NLL degradation exceeds tolerance")
    for name, item in domains.items():
        if max_domain_nll_delta is not None and item["mean_nll_delta"] > max_domain_nll_delta:
            failures.append(f"{name}: mean NLL degradation exceeds tolerance")
        item["paired_window_uncertainty"] = window_uncertainty(
            [window for window in paired_windows if window["domain"] == name])
    overall["paired_window_uncertainty"] = window_uncertainty(paired_windows)
    return {
        "schema_version": 2, "artifact_type": "ninfer_paired_quality_comparison",
        "passed": not failures, "failures": failures, "overall": overall, "domains": domains,
        "thresholds": {"max_mean_nll_delta": max_mean_nll_delta,
                       "max_domain_nll_delta": max_domain_nll_delta,
                       "min_scored_tokens": min_scored_tokens},
        "candidate_execution": candidate["execution"], "reference_execution": reference["execution"],
        "paired_windows": paired_windows,
        "kv_precision_mismatch_acknowledged": kv_mismatch and allow_kv_mismatch,
        "scope": "fixed corpus/window quantized-model non-inferiority screen, not an Op oracle "
                 "or proof of equal model quality; token-tail errors are diagnostics",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mean-nll-delta", type=float, default=0.02)
    parser.add_argument("--max-domain-nll-delta", type=float,
                        help="optional separately predeclared domain guard; all domains are reported")
    parser.add_argument("--min-scored-tokens", type=int, default=16000)
    parser.add_argument("--allow-kv-mismatch", action="store_true")
    args = parser.parse_args()
    if (not all(math.isfinite(value) and value >= 0 for value in (
        args.max_mean_nll_delta, args.max_domain_nll_delta) if value is not None)
            or args.min_scored_tokens < 1):
        parser.error("tolerances must be finite and nonnegative; minimum token count must be positive")
    try:
        candidate_bytes, reference_bytes = args.candidate.read_bytes(), args.reference.read_bytes()
        result = compare(
            json.loads(candidate_bytes), json.loads(reference_bytes),
            max_mean_nll_delta=args.max_mean_nll_delta,
            max_domain_nll_delta=args.max_domain_nll_delta,
            min_scored_tokens=args.min_scored_tokens, allow_kv_mismatch=args.allow_kv_mismatch,
        )
        result["inputs"] = {
            "candidate": {"path": str(args.candidate.resolve()),
                          "sha256": hashlib.sha256(candidate_bytes).hexdigest()},
            "reference": {"path": str(args.reference.resolve()),
                          "sha256": hashlib.sha256(reference_bytes).hexdigest()},
        }
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        result = {"passed": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result.get("overall", result), indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
