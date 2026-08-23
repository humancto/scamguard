#!/usr/bin/env python3
"""Record reproducible llama.cpp prompt/generation floors for a GGUF artifact."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scamguard.metrics import file_sha256

MAX_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)


def command_text(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({' '.join(command)}): {completed.stderr[-2000:]}")
    return completed.stdout.strip()


def parse_max_rss(stderr: str) -> int | None:
    matches = MAX_RSS.findall(stderr)
    return int(matches[-1]) if matches else None


def derive_measurement(row: dict[str, Any]) -> dict[str, Any]:
    samples = np.asarray(row.get("samples_ns", []), dtype=np.float64)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("llama-bench row has no finite timing samples")
    n_prompt = int(row.get("n_prompt", 0))
    n_gen = int(row.get("n_gen", 0))
    if (n_prompt > 0) == (n_gen > 0):
        raise ValueError("llama-bench row must measure prompt or generation, not both/neither")
    return {
        "kind": "prompt_processing" if n_prompt else "token_generation",
        "tokens": n_prompt or n_gen,
        "threads": int(row["n_threads"]),
        "gpu_layers": int(row["n_gpu_layers"]),
        "backend": str(row["backends"]),
        "mean_ms": float(np.mean(samples) / 1_000_000),
        "median_ms": float(np.median(samples) / 1_000_000),
        "p95_ms": float(np.percentile(samples, 95) / 1_000_000),
        "minimum_ms": float(np.min(samples) / 1_000_000),
        "maximum_ms": float(np.max(samples) / 1_000_000),
        "mean_tokens_per_second": float(row["avg_ts"]),
        "samples": int(samples.size),
    }


def benchmark(
    binary: Path,
    model: Path,
    prompt_lengths: list[int],
    generation_lengths: list[int],
    threads: list[int],
    gpu_layers: int,
    repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None, list[str]]:
    command = [
        str(binary),
        "--model",
        str(model),
        "--n-prompt",
        ",".join(map(str, prompt_lengths)),
        "--n-gen",
        ",".join(map(str, generation_lengths)),
        "--threads",
        ",".join(map(str, threads)),
        "--n-gpu-layers",
        str(gpu_layers),
        "--repetitions",
        str(repetitions),
        "--output",
        "json",
    ]
    timed_command = ["/usr/bin/time", "-l", *command]
    completed = subprocess.run(timed_command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"llama-bench failed: {completed.stderr[-4000:]}")
    raw = json.loads(completed.stdout)
    if not isinstance(raw, list) or not raw:
        raise ValueError("llama-bench emitted no JSON measurement rows")
    derived = [derive_measurement(row) for row in raw]
    return raw, derived, parse_max_rss(completed.stderr), command


def select_prompt_measurement(
    measurements: list[dict[str, Any]], prompt_tokens: int, gpu_layers: int, threads: int
) -> dict[str, Any]:
    matches = [
        row
        for row in measurements
        if row["kind"] == "prompt_processing"
        and row["tokens"] == prompt_tokens
        and row["gpu_layers"] == gpu_layers
        and row["threads"] == threads
    ]
    if len(matches) != 1:
        raise ValueError("reference prompt measurement was not found exactly once")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-bench", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 192, 256])
    parser.add_argument("--generation-lengths", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--metal-threads", type=int, nargs="+", default=[12])
    parser.add_argument("--cpu-threads", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--reference-prompt-tokens", type=int, default=192)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.llama_bench, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.llama_source.is_dir():
        raise FileNotFoundError(args.llama_source)
    model_sha256 = file_sha256(args.model)
    if model_sha256 != args.expected_model_sha256:
        raise ValueError(
            f"model SHA-256 differs: expected {args.expected_model_sha256}, got {model_sha256}"
        )

    source_revision = command_text(["git", "rev-parse", "HEAD"], args.llama_source)
    dirty_paths = command_text(
        ["git", "status", "--short", "--untracked-files=no"], args.llama_source
    ).splitlines()
    scenarios = []
    all_measurements: list[dict[str, Any]] = []
    for name, gpu_layers, threads in (
        ("metal", 99, args.metal_threads),
        ("cpu", 0, args.cpu_threads),
    ):
        raw, derived, maximum_rss, command = benchmark(
            args.llama_bench,
            args.model,
            args.prompt_lengths,
            args.generation_lengths,
            threads,
            gpu_layers,
            args.repetitions,
        )
        scenarios.append(
            {
                "name": name,
                "gpu_layers": gpu_layers,
                "threads": threads,
                "command": command,
                "maximum_resident_set_size_bytes": maximum_rss,
                "maximum_resident_set_size_scope": (
                    "whole llama-bench process from macOS /usr/bin/time -l"
                ),
                "measurements": derived,
                "raw_llama_bench_rows": raw,
            }
        )
        all_measurements.extend(derived)

    reference = select_prompt_measurement(
        all_measurements,
        args.reference_prompt_tokens,
        gpu_layers=99,
        threads=args.metal_threads[0],
    )
    result = {
        "artifact_schema_version": 1,
        "purpose": (
            "warm quantized base runtime floor; this is not trained-model quality or complete "
            "tokenizer-to-verdict latency"
        ),
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "sha256": model_sha256,
        },
        "llama_cpp": {
            "source_path": str(args.llama_source),
            "revision": source_revision,
            "tracked_worktree_dirty": bool(dirty_paths),
            "tracked_dirty_paths": dirty_paths,
            "binary": str(args.llama_bench),
            "binary_sha256": file_sha256(args.llama_bench),
            "binary_file_description": command_text(["file", str(args.llama_bench)]),
        },
        "host": {
            "os": platform.platform(),
            "python_architecture": platform.machine(),
            "cpu": command_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "physical_memory_bytes": int(command_text(["sysctl", "-n", "hw.memsize"])),
            "note": "M4 Max laptop measurements are not physical-phone measurements",
        },
        "protocol": {
            "prompt_lengths": args.prompt_lengths,
            "generation_lengths": args.generation_lengths,
            "repetitions": args.repetitions,
            "warmup": "llama-bench default warmup",
            "reference_prompt_tokens": args.reference_prompt_tokens,
        },
        "reference_prompt_floor": reference
        | {
            "formal_laptop_target_ms": 50.0,
            "formal_laptop_target_passed": reference["p95_ms"] < 50.0,
            "strict_fast_path_target_ms": 20.0,
            "strict_fast_path_target_passed": reference["p95_ms"] < 20.0,
            "scope_warning": "prompt processing only; a complete verdict is necessarily slower",
        },
        "scenarios": scenarios,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["reference_prompt_floor"], indent=2))


if __name__ == "__main__":
    main()
