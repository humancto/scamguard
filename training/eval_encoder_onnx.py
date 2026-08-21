#!/usr/bin/env python3
"""Evaluate one static-shape ONNX encoder without touching prediction-sealed data."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from scamguard.metrics import binary_safety_metrics, file_sha256
from scamguard.policy import POLICY_VERSION, deterministic_override
from scamguard.preprocessing import prepare_model_text
from scamguard.signals import extract_signal_matches
from scamguard.taxonomy import Verdict

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    adjusted = logits / temperature
    adjusted -= adjusted.max(axis=1, keepdims=True)
    values = np.exp(adjusted)
    return values / values.sum(axis=1, keepdims=True)


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def infer(
    session: ort.InferenceSession,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    sequence_length: int,
    dialogue_policy: str,
    dynamic_sequence: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    texts = [prepare_model_text(str(row["text"]), dialogue_policy) for row in rows]
    untruncated = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    token_lengths = [len(values) for values in untruncated["input_ids"]]
    logits = np.empty((len(rows), len(LABELS)), dtype=np.float32)
    for index, text in enumerate(texts):
        encoded = tokenizer(
            text,
            return_tensors="np",
            max_length=sequence_length,
            truncation=True,
            padding=False if dynamic_sequence else "max_length",
        )
        result = session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )[0]
        logits[index] = result[0]
    return logits, {
        "examples": len(rows),
        "token_length_max": max(token_lengths),
        "token_length_p95": float(np.percentile(token_lengths, 95)),
        "truncated_examples": sum(length > sequence_length for length in token_lengths),
    }


def parity_report(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    reference_path: Path,
) -> dict[str, Any]:
    reference_rows = read_jsonl(reference_path)
    reference = {str(row["id"]): row for row in reference_rows}
    if len(reference) != len(rows) or set(reference) != {str(row["id"]) for row in rows}:
        raise ValueError("reference predictions do not match evaluation row IDs")
    current_probability = probabilities[:, LABEL_TO_ID["SCAM"]]
    reference_probability = np.array(
        [float(reference[str(row["id"])]["scam_probability"]) for row in rows]
    )
    current_argmax = [LABELS[index] for index in probabilities.argmax(axis=1)]
    reference_argmax = [str(reference[str(row["id"])]["argmax_label"]) for row in rows]
    current_verdict = current_probability >= threshold
    reference_verdict = np.array(
        [bool(reference[str(row["id"])]["scam_at_frozen_threshold"]) for row in rows]
    )
    absolute_error = np.abs(current_probability - reference_probability)
    return {
        "reference_sha256": file_sha256(reference_path),
        "max_abs_scam_probability_error": float(absolute_error.max()),
        "mean_abs_scam_probability_error": float(absolute_error.mean()),
        "argmax_agreement": float(np.mean(np.array(current_argmax) == np.array(reference_argmax))),
        "frozen_binary_verdict_agreement": float(np.mean(current_verdict == reference_verdict)),
        "frozen_binary_verdict_disagreements": int(np.sum(current_verdict != reference_verdict)),
    }


def score(
    rows: list[dict[str, Any]], probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    truth = [str(row["label"]) for row in rows]
    predicted_ids = probabilities.argmax(axis=1)
    predicted_labels = [LABELS[index] for index in predicted_ids]
    binary_indices = [index for index, label in enumerate(truth) if label in {"SAFE", "SCAM"}]
    binary_truth = np.array([truth[index] == "SCAM" for index in binary_indices], dtype=int)
    scam_probabilities = probabilities[binary_indices, LABEL_TO_ID["SCAM"]]
    base_binary = scam_probabilities >= threshold

    policy_binary = base_binary.copy()
    policy_labels = predicted_labels.copy()
    binary_position = {row_index: offset for offset, row_index in enumerate(binary_indices)}
    rule_counts: Counter[str] = Counter()
    rule_truth: Counter[str] = Counter()
    for index, row in enumerate(rows):
        signals = tuple(match.signal for match in extract_signal_matches(str(row["text"])))
        override = deterministic_override(str(row["text"]), signals)
        if override is None:
            continue
        policy_labels[index] = override.verdict.value
        rule_counts[override.rule_id] += 1
        rule_truth[f"{override.rule_id}:{row['label']}"] += 1
        if index in binary_position:
            policy_binary[binary_position[index]] = override.verdict is Verdict.SCAM

    base_metrics = binary_safety_metrics(binary_truth, scam_probabilities, threshold)
    policy_scores = policy_binary.astype(float)
    policy_metrics = binary_safety_metrics(binary_truth, policy_scores, 0.5)
    policy_metrics["threshold"] = "deterministic policy over frozen model verdict"

    scam_by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in rows if row["label"] == "SCAM"}):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["label"] == "SCAM" and row["category"] == category
        ]
        detected = int(
            sum(probabilities[index, LABEL_TO_ID["SCAM"]] >= threshold for index in indices)
        )
        scam_by_category[category] = {
            "examples": len(indices),
            "detected": detected,
            "recall": detected / len(indices),
        }

    return {
        "labels": dict(Counter(truth)),
        "macro_f1_argmax": float(
            f1_score(truth, predicted_labels, labels=LABELS, average="macro", zero_division=0)
        ),
        "binary_safety": base_metrics,
        "scam_by_category": scam_by_category,
        "policy": {
            "version": POLICY_VERSION,
            "binary_safety": policy_metrics,
            "macro_f1": float(
                f1_score(truth, policy_labels, labels=LABELS, average="macro", zero_division=0)
            ),
            "rule_counts": dict(rule_counts),
            "rule_truth": dict(rule_truth),
        },
    }


def latency(
    session: ort.InferenceSession,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    sequence_length: int,
    dialogue_policy: str,
    dynamic_sequence: bool,
) -> dict[str, Any]:
    samples = rows[: min(250, len(rows))]
    for row in samples[:8]:
        text = prepare_model_text(str(row["text"]), dialogue_policy)
        encoded = tokenizer(
            text,
            return_tensors="np",
            max_length=sequence_length,
            truncation=True,
            padding=False if dynamic_sequence else "max_length",
        )
        session.run(["logits"], dict(encoded))

    forward_ms: list[float] = []
    end_to_end_ms: list[float] = []
    for row in samples:
        text = prepare_model_text(str(row["text"]), dialogue_policy)
        started = time.perf_counter_ns()
        encoded = tokenizer(
            text,
            return_tensors="np",
            max_length=sequence_length,
            truncation=True,
            padding=False if dynamic_sequence else "max_length",
        )
        forward_started = time.perf_counter_ns()
        logits = session.run(["logits"], dict(encoded))[0]
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
            "batch-one variable-shape tokenization plus ONNX CPU inference and probability "
            "transform"
            if dynamic_sequence
            else "batch-one fixed-shape tokenization plus ONNX CPU inference and probability "
            "transform"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dev-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--dynamic-sequence", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.threads < 1:
        raise ValueError("threads must be positive")
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    temperature = float(model_report["temperature"])
    threshold = float(model_report["scam_threshold"])
    dialogue_policy = str(model_report["input_transform"]["dialogue_policy"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    session = make_session(args.onnx, args.threads)
    if {item.name for item in session.get_inputs()} != {"input_ids", "attention_mask"}:
        raise ValueError("unexpected ONNX input contract")

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
        logits, windows = infer(
            session,
            tokenizer,
            rows,
            args.sequence_length,
            dialogue_policy,
            args.dynamic_sequence,
        )
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
        "model": str(args.onnx),
        "model_sha256": file_sha256(args.onnx),
        "model_bytes": args.onnx.stat().st_size,
        "checkpoint": str(args.checkpoint),
        "model_report_sha256": file_sha256(args.model_report),
        "sequence_length": args.sequence_length,
        "dynamic_sequence": args.dynamic_sequence,
        "batch_size": 1,
        "temperature": temperature,
        "scam_threshold": threshold,
        "dialogue_policy": dialogue_policy,
        "runtime": {
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "threads": args.threads,
            "platform": platform.platform(),
        },
        "splits": splits,
        "latency": latency(
            session,
            tokenizer,
            test_rows,
            args.sequence_length,
            dialogue_policy,
            args.dynamic_sequence,
        ),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "memory_scope": "whole evaluator process peak RSS on macOS; includes Python and tokenizer",
        "sealed_sources_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
