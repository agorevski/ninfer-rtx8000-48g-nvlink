#!/usr/bin/env python3
"""Build the independent scorer using an explicitly selected, existing llama.cpp build."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def build_command(source: Path, build: Path, output: Path) -> tuple[list[str], Path]:
    directory = build / "tools/llama-bench"
    link = directory / "CMakeFiles/llama-bench.dir/link.txt"
    command = shlex.split(link.read_text(encoding="utf-8"))
    objects = [part for part in command if part.endswith(".o")]
    if len(objects) != 1 or command.count("-o") != 1:
        raise ValueError("expected the existing CMake Makefiles llama-bench link command")
    command[command.index(objects[0])] = str(Path(__file__).with_name("llama_score.cpp"))
    command[command.index("-o") + 1] = str(output)
    command[1:1] = [
        "-std=c++17", "-I" + str(source / "include"),
        "-I" + str(source / "ggml/include"), "-I" + str(source / "vendor"),
    ]
    return command, directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    command, directory = build_command(args.source_dir.resolve(), args.build_dir.resolve(), output)
    revision = subprocess.run(
        ["git", "-C", str(args.source_dir.resolve()), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    output.with_suffix(".build.json").write_text(
        json.dumps({"command": command, "cwd": str(directory),
                    "source_revision": revision}, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(command, cwd=directory, check=True,
                   env={**os.environ, "TMPDIR": str(output.parent)})
    subprocess.run([str(output), "--help"], check=True)


if __name__ == "__main__":
    main()
