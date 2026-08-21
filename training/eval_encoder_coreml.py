#!/usr/bin/env python3
"""Evaluate one fixed-shape Core ML encoder without touching sealed data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
from eval_encoder_onnx import parity_report, read_jsonl, score, softmax
from transformers import AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import prepare_model_text


def directory_identity(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
    return {"files": len(files), "bytes": total, "tree_sha256": digest.hexdigest()}


def encode(tokenizer: Any, text: str, sequence_length: int) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        text,
        return_tensors="np",
        max_length=sequence_length,
        truncation=True,
        padding="max_length",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def infer(
    model: ct.models.MLModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    sequence_length: int,
    dialogue_policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    texts = [prepare_model_text(str(row["text"]), dialogue_policy) for row in rows]
    untruncated = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    token_lengths = [len(values) for values in untruncated["input_ids"]]
    logits = np.empty((len(rows), 3), dtype=np.float32)
    for index, text in enumerate(texts):
        prediction = model.predict(encode(tokenizer, text, sequence_length))
        logits[index] = np.asarray(prediction["logits"])[0]
    return logits, {
        "examples": len(rows),
        "token_length_max": max(token_lengths),
        "token_length_p95": float(np.percentile(token_lengths, 95)),
        "truncated_examples": sum(length > sequence_length for length in token_lengths),
    }


def latency(
    model: ct.models.MLModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    sequence_length: int,
    dialogue_policy: str,
) -> dict[str, Any]:
    samples = rows[: min(250, len(rows))]
    for row in samples[:8]:
        text = prepare_model_text(str(row["text"]), dialogue_policy)
        model.predict(encode(tokenizer, text, sequence_length))

    forward_ms: list[float] = []
    end_to_end_ms: list[float] = []
    for row in samples:
        text = prepare_model_text(str(row["text"]), dialogue_policy)
        started = time.perf_counter_ns()
        inputs = encode(tokenizer, text, sequence_length)
        forward_started = time.perf_counter_ns()
        logits = np.asarray(model.predict(inputs)["logits"])
        np.exp(logits - logits.max(axis=1, keepdims=True))
        finished = time.perf_counter_ns()
        forward_ms.append((finished - forward_started) / 1_000_000)
        end_to_end_ms.append((finished - started) / 1_000_000)
    return {
        "model_forward_median_ms": float(np.median(forward_ms)),
        "model_forward_p95_ms": float(np.percentile(forward_ms, 95)),
        "end_to_end_median_ms": float(np.median(end_to_end_ms)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end_ms, 95)),
        "samples": len(samples),
        "scope": (
            "batch-one fixed-shape tokenization plus Core ML inference and probability transform"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    package_identity = directory_identity(args.package)
    if package_identity != export_manifest["model_identity"]:
        raise ValueError("Core ML package differs from its export manifest")
    sequence_length = int(export_manifest["sequence_length"])
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    temperature = float(model_report["temperature"])
    threshold = float(model_report["scam_threshold"])
    dialogue_policy = str(export_manifest["input_transform"]["dialogue_policy"])
    if dialogue_policy != str(model_report["input_transform"]["dialogue_policy"]):
        raise ValueError("Core ML and reference dialogue policies differ")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = ct.models.MLModel(str(args.package), compute_units=ct.ComputeUnit.ALL)
    splits: dict[str, Any] = {}
    split_inputs = {
        "dev": (args.data_dir / "dev.jsonl", args.dev_predictions),
        "test": (args.data_dir / "test.jsonl", args.test_predictions),
    }
    test_rows: list[dict[str, Any]] = []
    for split, (data_path, reference_path) in split_inputs.items():
        if file_sha256(data_path) != str(model_report["data_sha256"][split]):
            raise ValueError(f"{split} data hash differs from frozen model report")
        rows = read_jsonl(data_path)
        logits, windows = infer(model, tokenizer, rows, sequence_length, dialogue_policy)
        probabilities = softmax(logits, temperature)
        splits[split] = {
            "data_sha256": file_sha256(data_path),
            "sequence_window": windows,
            "score": score(rows, probabilities, threshold),
            "reference_parity": parity_report(rows, probabilities, threshold, reference_path),
        }
        if split == "test":
            test_rows = rows

    report = {
        "format_version": 1,
        "model": str(args.package),
        "model_identity": package_identity,
        "export_manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "model_report_sha256": file_sha256(args.model_report),
        "sequence_length": sequence_length,
        "batch_size": 1,
        "temperature": temperature,
        "scam_threshold": threshold,
        "dialogue_policy": dialogue_policy,
        "runtime": {
            "coremltools": ct.__version__,
            "compute_units": "ALL",
            "platform": platform.platform(),
        },
        "splits": splits,
        "latency": latency(model, tokenizer, test_rows, sequence_length, dialogue_policy),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_scope": "whole evaluator process peak RSS on macOS; includes Python and tokenizer",
        "sealed_sources_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
