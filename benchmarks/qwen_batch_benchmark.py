#!/usr/bin/env python3
"""Select an efficient, numerically equivalent Qwen evaluation batch size on MPS."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval_qwen import LABELS, read_jsonl, score_messages  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def representative_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select deterministic text-length quantiles, including both binary labels."""

    binary = [row for row in rows if row["label"] in {"SAFE", "SCAM"}]
    binary.sort(key=lambda row: (len(str(row["text"])), str(row["id"])))
    if count >= len(binary):
        return binary
    indices = np.linspace(0, len(binary) - 1, count, dtype=int)
    return [binary[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/dev.jsonl"))
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--report", type=Path, default=Path("reports/runs/qwen35-2b-batch-benchmark.json")
    )
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()

    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("--require-mps was set, but MPS is unavailable")
    device = torch.device("mps" if mps_available else "cpu")
    print(f"batch benchmark accelerator: {device}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
        low_cpu_mem_usage=True,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if args.revision and resolved_revision and args.revision != resolved_revision:
        raise RuntimeError(
            f"loaded base revision {resolved_revision} differs from requested {args.revision}"
        )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()
    rows = representative_rows(read_jsonl(args.data), args.rows)
    texts = [str(row["text"]) for row in rows]

    # Warm kernels and allocator before any timed measurement.
    score_messages(model, processor, texts[:1], device, batch_size=1)
    synchronize(device)

    reference: np.ndarray | None = None
    measurements: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        if device.type == "mps":
            torch.mps.empty_cache()
        synchronize(device)
        memory: dict[str, int] = {}
        started = time.perf_counter_ns()
        scores = score_messages(
            model,
            processor,
            texts,
            device,
            batch_size=batch_size,
            memory_telemetry=memory,
            progress_label=f"batch_size={batch_size}",
            progress_every=1,
        )
        synchronize(device)
        elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
        if reference is None:
            reference = scores
        maximum_delta = float(np.max(np.abs(scores - reference)))
        argmax_matches = int(np.sum(scores.argmax(axis=1) == reference.argmax(axis=1)))
        measurement = {
            "batch_size": batch_size,
            "rows": len(rows),
            "batches": int(np.ceil(len(rows) / batch_size)),
            "elapsed_seconds": elapsed_seconds,
            "rows_per_second": len(rows) / elapsed_seconds,
            "milliseconds_per_row": elapsed_seconds * 1_000 / len(rows),
            "maximum_absolute_score_delta_vs_batch_one": maximum_delta,
            "argmax_matches_vs_batch_one": argmax_matches,
            "argmax_total": len(rows),
            "numerically_equivalent": maximum_delta <= 0.02 and argmax_matches == len(rows),
            "memory": memory,
        }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)

    eligible = [row for row in measurements if row["numerically_equivalent"]]
    selected = max(eligible, key=lambda row: row["rows_per_second"])["batch_size"]
    result = {
        "purpose": "evaluation throughput only; product latency remains batch one",
        "model": args.model,
        "requested_revision": args.revision,
        "base_model_revision": resolved_revision or args.revision,
        "adapter": str(args.adapter),
        "device": str(device),
        "environment": {"torch": torch.__version__, "python_arch": platform.machine()},
        "selection": {
            "batch_size": selected,
            "criterion": "highest rows/sec among numerically equivalent candidates",
            "score_delta_tolerance": 0.02,
            "labels": list(LABELS),
        },
        "sample": {
            "method": "deterministic SAFE/SCAM message-length quantiles from frozen dev",
            "rows": len(rows),
            "minimum_characters": min(len(text) for text in texts),
            "maximum_characters": max(len(text) for text in texts),
        },
        "measurements": measurements,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"selected evaluation batch size: {selected}", flush=True)


if __name__ == "__main__":
    main()
