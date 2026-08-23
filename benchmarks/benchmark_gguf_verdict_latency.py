#!/usr/bin/env python3
"""Measure the exact three-candidate GGUF verdict scorer without message text in reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scamguard.prompts import SYSTEM_PROMPT
from training.eval_gguf import LABELS, SCORE_LINE, read_jsonl, write_tasks

TIMESTAMP = r"(\d+)\.(\d{2})\.(\d{3})\.(\d{3})"
SCORE_START = re.compile(TIMESTAMP + r" I multiple_choice_score : calculating")
SCORE_END = re.compile(TIMESTAMP + r" I Final result:")
MAX_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)


def timestamp_microseconds(parts: tuple[str, str, str, str]) -> int:
    minutes, seconds, milliseconds, microseconds = (int(value) for value in parts)
    return ((minutes * 60 + seconds) * 1_000_000) + milliseconds * 1_000 + microseconds


def score_phase_seconds(output: str) -> float:
    starts = SCORE_START.findall(output)
    ends = SCORE_END.findall(output)
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("llama.cpp score-phase timestamps were not found exactly once")
    elapsed = timestamp_microseconds(ends[0]) - timestamp_microseconds(starts[0])
    if elapsed <= 0:
        raise ValueError("llama.cpp score-phase duration is not positive")
    return elapsed / 1_000_000


def run_once(command: list[str], expected_rows: int) -> dict[str, Any]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        ["/usr/bin/time", "-l", *command], text=True, capture_output=True, check=False
    )
    wall_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"llama.cpp verdict benchmark failed: {output[-4000:]}")
    if len(SCORE_LINE.findall(output)) != expected_rows:
        raise ValueError("llama.cpp verdict benchmark emitted an unexpected score-row count")
    phase_seconds = score_phase_seconds(output)
    rss = MAX_RSS.findall(completed.stderr)
    return {
        "process_wall_seconds": wall_seconds,
        "score_phase_seconds": phase_seconds,
        "mean_score_phase_ms_per_message": phase_seconds * 1_000 / expected_rows,
        "maximum_resident_set_size_bytes": int(rss[-1]) if rss else None,
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot summarize empty or non-finite measurements")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--processor", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--processor-revision", required=True)
    parser.add_argument("--ctx-size", type=int, default=640)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.llama_perplexity, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    split_path = args.data / f"{args.split}.jsonl"
    rows = read_jsonl(split_path)[: args.rows]
    if len(rows) != args.rows:
        raise ValueError(f"requested {args.rows} rows, found {len(rows)}")
    model_sha256 = file_sha256(args.model)
    if model_sha256 != args.expected_model_sha256:
        raise ValueError("model SHA-256 differs from the pinned control")

    processor = AutoProcessor.from_pretrained(
        args.processor, revision=args.processor_revision, local_files_only=True
    )
    questions: list[str] = []
    token_lengths: list[int] = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{row['text']}</message>",
            },
        ]
        question = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        question += '{"verdict":"'
        questions.append(question)
        token_lengths.append(
            max(
                len(
                    processor.tokenizer(
                        question + label + '"', add_special_tokens=False
                    )["input_ids"]
                )
                for label in LABELS
            )
        )

    with tempfile.TemporaryDirectory(prefix="scamguard-verdict-latency-") as directory:
        task_path = Path(directory) / "tasks.bin"
        write_tasks(task_path, questions, [str(row["label"]) for row in rows])
        command = [
            str(args.llama_perplexity),
            "--model",
            str(args.model),
            "--file",
            str(task_path),
            "--multiple-choice",
            "--ctx-size",
            str(args.ctx_size),
            "--batch-size",
            str(args.ctx_size),
            "--ubatch-size",
            str(args.ctx_size),
            "--parallel",
            "1",
            "--threads",
            str(args.threads),
            "--n-gpu-layers",
            str(args.n_gpu_layers),
        ]
        measurements = [run_once(command, len(rows)) for _ in range(args.repetitions)]

    per_message = [row["mean_score_phase_ms_per_message"] for row in measurements]
    rss_values = [
        int(row["maximum_resident_set_size_bytes"])
        for row in measurements
        if row["maximum_resident_set_size_bytes"] is not None
    ]
    identifier_payload = "\n".join(str(row["id"]) for row in rows).encode()
    result = {
        "artifact_schema_version": 1,
        "purpose": (
            "exact three-candidate verdict-score phase throughput at product batch one; "
            "not per-request latency and not full structured-output generation"
        ),
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "sha256": model_sha256,
        },
        "runtime": {
            "llama_perplexity": str(args.llama_perplexity),
            "binary_sha256": file_sha256(args.llama_perplexity),
            "source_contract": (
                "local llama.cpp patch removes the inserted answer space and emits one "
                "length-normalized log-probability per verdict candidate"
            ),
            "ctx_size": args.ctx_size,
            "threads": args.threads,
            "n_gpu_layers": args.n_gpu_layers,
            "parallel": 1,
        },
        "data": {
            "path": str(split_path),
            "sha256": file_sha256(split_path),
            "rows": len(rows),
            "selection": "first rows from frozen split, matching the BF16 latency protocol",
            "selected_ids_sha256": hashlib.sha256(identifier_payload).hexdigest(),
            "contains_message_text": False,
            "candidate_input_tokens": {
                "minimum": min(token_lengths),
                "p50": float(np.percentile(token_lengths, 50)),
                "p95": float(np.percentile(token_lengths, 95)),
                "maximum": max(token_lengths),
            },
        },
        "measurements": measurements,
        "score_phase_ms_per_message_across_run_means": summarize(per_message),
        "maximum_process_rss_bytes_across_runs": max(rss_values) if rss_values else None,
        "interpretation": {
            "unit": "each sample is one run's score-phase time divided by its 50 messages",
            "formal_laptop_target_ms": 50.0,
            "formal_target_passed_by_run_mean_p95": summarize(per_message)["p95"] < 50.0,
            "strict_fast_path_target_ms": 20.0,
            "strict_target_passed_by_run_mean_p95": summarize(per_message)["p95"] < 20.0,
            "limitation": (
                "run-mean p95 is not a per-message latency percentile because upstream "
                "llama.cpp does not emit per-task timing"
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["score_phase_ms_per_message_across_run_means"], indent=2))


if __name__ == "__main__":
    main()
