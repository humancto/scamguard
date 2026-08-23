#!/usr/bin/env python3
"""Compare uncached and fixed-prefix persistent GGUF verdict latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_routed_transformers_runtime import summarize
from scamguard.decision import calibrated_verdict
from scamguard.gguf_runtime import (
    LABELS,
    PersistentGGUFScorer,
    calibrated_probabilities,
)
from scamguard.metrics import file_sha256
from scamguard.prompts import SYSTEM_PROMPT


def render_question(processor: Any, text: str) -> str:
    prompt = processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{text}</message>",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + '{"verdict":"'


def fixed_prefix(processor: Any) -> str:
    sentinel = "SCAMGUARD_RUNTIME_MESSAGE_SENTINEL"
    rendered = render_question(processor, sentinel)
    marker = f"<message>{sentinel}"
    if rendered.count(marker) != 1:
        raise ValueError("Qwen chat template does not preserve the runtime message marker")
    return rendered.split(marker, maxsplit=1)[0]


def score_parity(
    reference: list[tuple[float, float, float]],
    candidate: list[tuple[float, float, float]],
    *,
    temperature: float,
    scam_threshold: float,
    safe_threshold: float,
) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("score parity requires equal non-empty inputs")
    reference_probabilities = [
        calibrated_probabilities(scores, temperature) for scores in reference
    ]
    candidate_probabilities = [
        calibrated_probabilities(scores, temperature) for scores in candidate
    ]

    def verdict(probabilities: tuple[float, float, float]) -> str:
        return calibrated_verdict(
            safe_probability=probabilities[0],
            scam_probability=probabilities[2],
            scam_probability_threshold=scam_threshold,
            safe_probability_threshold=safe_threshold,
            safe_max_scam_probability=None,
        )

    mismatches = [
        index
        for index, (expected, actual) in enumerate(
            zip(reference_probabilities, candidate_probabilities, strict=True)
        )
        if verdict(expected) != verdict(actual)
    ]
    return {
        "examples": len(reference),
        "maximum_absolute_raw_score_error": max(
            abs(expected - actual)
            for expected_row, actual_row in zip(reference, candidate, strict=True)
            for expected, actual in zip(expected_row, actual_row, strict=True)
        ),
        "maximum_absolute_probability_error": max(
            abs(expected - actual)
            for expected_row, actual_row in zip(
                reference_probabilities, candidate_probabilities, strict=True
            )
            for expected, actual in zip(expected_row, actual_row, strict=True)
        ),
        "calibrated_verdict_mismatch_count": len(mismatches),
        "calibrated_verdict_mismatch_indices": mismatches[:20],
        "release_gate_passed": not mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--processor", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--processor-revision", required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.repetitions < 1:
        parser.error("--rows and --repetitions must be positive")
    if file_sha256(args.runner) != args.runner_sha256:
        raise ValueError("native runner SHA-256 differs from --runner-sha256")
    if file_sha256(args.model) != args.model_sha256:
        raise ValueError("GGUF model SHA-256 differs from --model-sha256")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.processor,
        revision=args.processor_revision,
        local_files_only=True,
    )
    prefix = fixed_prefix(processor)
    split_path = args.data / f"{args.split}.jsonl"
    with split_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][: args.rows]
    if len(rows) != args.rows:
        raise ValueError(f"requested {args.rows} rows, found {len(rows)}")
    questions = [render_question(processor, str(row["text"])) for row in rows]
    scorer_options = {
        "runner": args.runner,
        "model": args.model,
        "ctx_size": args.ctx_size,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "threads": args.threads,
        "n_gpu_layers": args.n_gpu_layers,
    }
    with PersistentGGUFScorer(**scorer_options) as uncached:
        uncached.score("uncached-warmup", questions[0])
        reference_results = [
            uncached.score(f"uncached-{index}", question)
            for index, question in enumerate(questions)
        ]
    with PersistentGGUFScorer(**scorer_options, prefix=prefix) as cached:
        cached.score("cached-warmup", questions[0])
        cached_results = [
            cached.score(f"cached-{repetition}-{index}", question)
            for repetition in range(args.repetitions)
            for index, question in enumerate(questions)
        ]
        loaded_prefix_tokens = cached.loaded_prefix_tokens

    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    first_cached = cached_results[: len(rows)]
    parity = score_parity(
        [result.raw_scores for result in reference_results],
        [result.raw_scores for result in first_cached],
        temperature=float(calibration["temperature"]),
        scam_threshold=float(calibration["scam_threshold"]),
        safe_threshold=float(calibration["safe_threshold"]),
    )
    uncached_latency = summarize([result.round_trip_ms for result in reference_results])
    cached_latency = summarize([result.round_trip_ms for result in cached_results])
    identifier_digest = hashlib.sha256(
        "\n".join(str(row["id"]) for row in rows).encode()
    ).hexdigest()
    result = {
        "artifact_schema_version": 1,
        "purpose": "persistent per-request fixed-prefix GGUF latency control",
        "model": {
            "path": str(args.model),
            "sha256": args.model_sha256,
            "bytes": args.model.stat().st_size,
        },
        "processor": {
            "repository": args.processor,
            "revision": args.processor_revision,
        },
        "runtime": {
            "runner": str(args.runner),
            "runner_sha256": args.runner_sha256,
            "protocol_version": 2,
            "ctx_size_per_sequence": args.ctx_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "n_gpu_layers": args.n_gpu_layers,
            "message_batch_size": 1,
            "candidate_batch_size": len(LABELS),
            "sequence_bucket_size": 64,
            "prefix_tokens": loaded_prefix_tokens,
            "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        },
        "data": {
            "path": str(split_path),
            "sha256": file_sha256(split_path),
            "rows": len(rows),
            "repetitions": args.repetitions,
            "selected_ids_sha256": identifier_digest,
            "contains_message_text": False,
        },
        "calibration_report": {
            "path": str(args.calibration_report),
            "sha256": file_sha256(args.calibration_report),
        },
        "parity_vs_uncached_native": parity,
        "uncached_round_trip_latency": {
            **uncached_latency,
            "samples_ms": [result.round_trip_ms for result in reference_results],
        },
        "cached_round_trip_latency": {
            **cached_latency,
            "samples_ms": [result.round_trip_ms for result in cached_results],
        },
        "gates": {
            "cached_p95_under_50_ms": bool(
                cached_latency["p95_ms"] is not None
                and float(cached_latency["p95_ms"]) < 50.0
            ),
            "exact_calibrated_verdict_parity": parity["release_gate_passed"],
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "limitations": [
            "This upstream base-model control does not establish trained-model quality.",
            "Desktop Metal timing does not establish physical-phone latency.",
            "The complete routed release benchmark remains authoritative for product latency.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parity": parity, "cached_latency": cached_latency}, indent=2))


if __name__ == "__main__":
    main()
