#!/usr/bin/env python3
"""Prepare exact artifact-tokenizer windows on CPU so independent reference scoring can run first."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.artifact.reader import Artifact


def plan_windows(tokens: int, context: int, stride: int) -> list[dict[str, int]]:
    if tokens < 2 or context < 2 or not 1 <= stride < context:
        raise ValueError("need tokens/context>=2 and 1<=stride<context")
    windows = []
    target = 1
    while target < tokens:
        end = min(tokens, context if not windows else target + stride)
        begin = max(0, end - context)
        windows.append({"input_begin": begin, "input_end": end, "target_begin": target,
                        "target_end": end, "first_target": target - begin})
        target = end
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 0 or args.max_tokens == 1:
        parser.error("--max-tokens must be zero or at least two")
    plan_windows(2, args.context, args.stride)
    if args.output.exists():
        parser.error("output already exists")
    from tokenizers import Tokenizer
    with Artifact(args.weights) as artifact:
        resource = artifact.directory.components["text"]["resources"]["tokenizer.json"]
        tokenizer_bytes = artifact.read_object(resource)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    manifest = json.loads(args.corpus.read_text(encoding="utf-8"))
    metadata = {stream["id"]: stream for stream in manifest["streams"]}
    mode = "quick" if args.quick else "full"
    streams = []
    for identity in manifest["modes"][mode]:
        item = metadata[identity]
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("corpus stream must be relative to manifest")
        text = (args.corpus.parent / path).read_bytes().decode("utf-8")
        if not text or "\0" in text:
            raise ValueError("corpus stream must be nonempty NUL-free UTF-8")
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        full_count = len(tokens)
        if args.max_tokens:
            tokens = tokens[:args.max_tokens]
        streams.append({"id": identity, "domain": item["domain"], "text": text, "tokens": tokens,
                        "full_input_tokens": full_count,
                        "windows": plan_windows(len(tokens), args.context, args.stride)})
    report = {
        "schema_version": 1, "artifact_type": "ninfer_paired_score_plan", "backend": "tokenizer-only",
        "tokenizer_artifact": str(args.weights.resolve()),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "execution": {"context": args.context, "stride": args.stride,
                      "max_tokens_per_stream": args.max_tokens, "special_tokens_added": False},
        "corpus": {"id": manifest["corpus_id"], "mode": mode, "path": str(args.corpus.resolve())},
        "streams": streams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(streams)} streams / {sum(len(s['tokens']) - 1 for s in streams)} targets")


if __name__ == "__main__":
    main()
