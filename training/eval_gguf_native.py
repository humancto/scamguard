#!/usr/bin/env python3
"""Evaluate a quantized ScamGuard GGUF through the protocol-v3 native scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.gguf_runtime import (
    FROZEN_PROMPT_PREFIX,
    FROZEN_PROMPT_SUFFIX,
    GGUF_BACKEND_TYPE,
    GGUF_SCORING_VERSION,
    PersistentGGUFScorer,
)
from scamguard.metrics import choose_threshold_for_gates, file_sha256
from scamguard.prompts import SYSTEM_PROMPT
from training.eval_gguf import (
    prediction_ledger_records,
    quantization_parity,
    read_jsonl,
    resolve_split_path,
)
from training.eval_qwen import (
    LABELS,
    choose_safe_threshold,
    evaluate_slice,
    fit_temperature,
    softmax,
    validate_primary_test_v8,
)

PROTOCOL_VERSION = 3


def cache_identity(
    *,
    split: str,
    rows: int,
    data_sha256: str,
    model_sha256: str,
    runner_sha256: str,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    threads: int,
    n_gpu_layers: int,
) -> dict[str, Any]:
    return {
        "scoring_version": GGUF_SCORING_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "rows": rows,
        "data_sha256": data_sha256,
        "model_sha256": model_sha256,
        "runner_sha256": runner_sha256,
        "ctx_size": ctx_size,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "threads": threads,
        "n_gpu_layers": n_gpu_layers,
        "prompt_prefix_sha256": hashlib.sha256(FROZEN_PROMPT_PREFIX.encode()).hexdigest(),
        "prompt_suffix_sha256": hashlib.sha256(FROZEN_PROMPT_SUFFIX.encode()).hexdigest(),
    }


def load_cache(
    directory: Path, split: str, identity: dict[str, Any]
) -> tuple[np.ndarray, list[float]] | None:
    metadata_path = directory / f"{split}.json"
    scores_path = directory / f"{split}.npy"
    if not metadata_path.is_file() or not scores_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scores = np.load(scores_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    timings = metadata.get("round_trip_ms")
    if (
        metadata.get("identity") != identity
        or scores.shape != (identity["rows"], len(LABELS))
        or not np.isfinite(scores).all()
        or not isinstance(timings, list)
        or len(timings) != identity["rows"]
        or any(not isinstance(value, (int, float)) or value <= 0 for value in timings)
    ):
        return None
    return scores, [float(value) for value in timings]


def save_cache(
    directory: Path,
    split: str,
    identity: dict[str, Any],
    scores: np.ndarray,
    timings: list[float],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary_scores = directory / f".{split}.npy.tmp"
    temporary_metadata = directory / f".{split}.json.tmp"
    with temporary_scores.open("wb") as handle:
        np.save(handle, scores, allow_pickle=False)
    temporary_scores.replace(directory / f"{split}.npy")
    temporary_metadata.write_text(
        json.dumps({"identity": identity, "round_trip_ms": timings}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(directory / f"{split}.json")


def score_rows(
    scorer: PersistentGGUFScorer,
    split: str,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[float]]:
    scores: list[tuple[float, float, float]] = []
    timings: list[float] = []
    for index, row in enumerate(rows):
        question = (
            FROZEN_PROMPT_PREFIX
            + "<message>"
            + str(row["text"])
            + FROZEN_PROMPT_SUFFIX
        )
        result = scorer.score(f"{split}-{index}", question, timeout_seconds=120.0)
        scores.append(result.raw_scores)
        timings.append(result.round_trip_ms)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"{split}: {index + 1}/{len(rows)}", flush=True)
    return np.asarray(scores, dtype=np.float64), timings


def calibration_from_report(
    path: Path,
    *,
    model_sha256: str,
    runner_sha256: str,
    dev_sha256: str,
    max_fpr: float,
    min_recall: float,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    calibration = report.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("native calibration report lacks a calibration record")
    expected = {
        "backend_type": GGUF_BACKEND_TYPE,
        "model_sha256": model_sha256,
        "runner_sha256": runner_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "scoring_mode": "branch_token",
        "scoring_version": GGUF_SCORING_VERSION,
        "dev_data_sha256": dev_sha256,
        "maximum_safe_fpr": max_fpr,
        "minimum_dev_recall": min_recall,
    }
    actual = {field: calibration.get(field) for field in expected}
    if actual != expected:
        raise ValueError("native calibration differs from the frozen GGUF or policy")
    return calibration


def validate_final_declaration(
    path: Path,
    *,
    model: Path,
    runner: Path,
    calibration_report: Path,
    primary_test: Path,
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "artifact_schema_version": 1,
        "state": "FINAL_QUANTIZED_CANDIDATE_FROZEN",
        "quantization_frozen": True,
        "model_sha256": file_sha256(model),
        "runner_sha256": file_sha256(runner),
        "calibration_report_sha256": file_sha256(calibration_report),
        "primary_test_v8_sha256": file_sha256(primary_test),
        "protocol_version": PROTOCOL_VERSION,
        "scoring_version": GGUF_SCORING_VERSION,
    }
    actual = {field: record.get(field) for field in expected}
    if actual != expected:
        raise ValueError("final artifact declaration differs from the sealed evaluation inputs")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, default=Path("data/external"))
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--fit-calibration-on-dev", action="store_true")
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--primary-test-v8", type=Path)
    parser.add_argument("--final-artifact-declaration", type=Path)
    parser.add_argument("--ctx-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--min-recall", type=float, default=0.97)
    args = parser.parse_args()
    if args.fit_calibration_on_dev == (args.calibration_report is not None):
        parser.error("choose exactly one of --fit-calibration-on-dev or --calibration-report")
    if args.primary_test_v8 is not None and args.final_artifact_declaration is None:
        parser.error("sealed primary evaluation requires --final-artifact-declaration")
    if args.primary_test_v8 is not None and args.fit_calibration_on_dev:
        parser.error("sealed primary evaluation cannot fit calibration")
    for path in (args.model, args.runner):
        if not path.is_file():
            raise FileNotFoundError(path)

    split_paths = {
        split: resolve_split_path(args.data, args.external_data, split)
        for split in args.splits
        if split != "primary_test_v8"
    }
    primary_manifest = None
    if args.primary_test_v8 is not None:
        primary_manifest = validate_primary_test_v8(args.primary_test_v8)
        split_paths["primary_test_v8"] = args.primary_test_v8
        if "primary_test_v8" not in args.splits:
            args.splits.append("primary_test_v8")
    elif "primary_test_v8" in args.splits:
        parser.error("primary_test_v8 split requires --primary-test-v8")
    rows_by_split = {split: read_jsonl(split_paths[split]) for split in args.splits}
    model_sha256 = file_sha256(args.model)
    runner_sha256 = file_sha256(args.runner)
    data_sha256 = {split: file_sha256(path) for split, path in split_paths.items()}

    if args.fit_calibration_on_dev and "dev" not in rows_by_split:
        raise ValueError("native calibration fitting requires dev")
    if args.calibration_report is not None:
        calibration = calibration_from_report(
            args.calibration_report,
            model_sha256=model_sha256,
            runner_sha256=runner_sha256,
            dev_sha256=file_sha256(args.data / "dev.jsonl"),
            max_fpr=args.max_fpr,
            min_recall=args.min_recall,
        )
    else:
        calibration = {}
    final_declaration = None
    if args.primary_test_v8 is not None:
        assert args.calibration_report is not None
        final_declaration = validate_final_declaration(
            args.final_artifact_declaration,
            model=args.model,
            runner=args.runner,
            calibration_report=args.calibration_report,
            primary_test=args.primary_test_v8,
        )

    cache_dir = args.cache_dir or args.report.parent / f"{args.report.stem}.scores"
    scores_by_split: dict[str, np.ndarray] = {}
    timings_by_split: dict[str, list[float]] = {}
    with PersistentGGUFScorer(
        runner=args.runner,
        model=args.model,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        prefix=FROZEN_PROMPT_PREFIX,
        startup_timeout_seconds=120.0,
    ) as scorer:
        if scorer.protocol_version != PROTOCOL_VERSION:
            raise ValueError("native runner protocol differs from the branch-token contract")
        for split in args.splits:
            identity = cache_identity(
                split=split,
                rows=len(rows_by_split[split]),
                data_sha256=data_sha256[split],
                model_sha256=model_sha256,
                runner_sha256=runner_sha256,
                ctx_size=args.ctx_size,
                batch_size=args.batch_size,
                ubatch_size=args.ubatch_size,
                threads=args.threads,
                n_gpu_layers=args.n_gpu_layers,
            )
            cached = load_cache(cache_dir, split, identity)
            if cached is None:
                scores, timings = score_rows(scorer, split, rows_by_split[split])
                save_cache(cache_dir, split, identity, scores, timings)
            else:
                scores, timings = cached
                print(f"{split}: loaded {len(rows_by_split[split])} cached scores")
            scores_by_split[split] = scores
            timings_by_split[split] = timings

    if args.fit_calibration_on_dev:
        dev_truth = np.asarray(
            [LABELS.index(str(row["label"])) for row in rows_by_split["dev"]]
        )
        temperature = fit_temperature(scores_by_split["dev"], dev_truth)
        dev_probabilities = softmax(scores_by_split["dev"], temperature)
        binary_mask = np.asarray(
            [row["label"] in {"SAFE", "SCAM"} for row in rows_by_split["dev"]]
        )
        binary_truth = np.asarray(
            [int(row["label"] == "SCAM") for row in rows_by_split["dev"]]
        )[binary_mask]
        threshold = choose_threshold_for_gates(
            binary_truth,
            dev_probabilities[binary_mask, LABELS.index("SCAM")],
            min_recall=args.min_recall,
            max_fpr=args.max_fpr,
        )
        if threshold is None:
            raise ValueError("quantized development scores cannot satisfy recall/FPR gates")
        safe_threshold = choose_safe_threshold(dev_truth, dev_probabilities, threshold)
        calibration = {
            "artifact_schema_version": 1,
            "backend_type": GGUF_BACKEND_TYPE,
            "model_sha256": model_sha256,
            "runner_sha256": runner_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "scoring_mode": "branch_token",
            "scoring_version": GGUF_SCORING_VERSION,
            "labels": list(LABELS),
            "temperature": temperature,
            "scam_threshold": threshold,
            "safe_threshold": safe_threshold,
            "safe_threshold_semantics": "minimum_safe_probability",
            "dev_data_sha256": data_sha256["dev"],
            "maximum_safe_fpr": args.max_fpr,
            "minimum_dev_recall": args.min_recall,
            "sequence_bucket_size": 64,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        }
        calibration_output = args.calibration_output or args.report.with_suffix(
            ".calibration.json"
        )
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_output.write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
        calibration = calibration | {
            "path": str(calibration_output),
            "sha256": file_sha256(calibration_output),
        }

    temperature = float(calibration["temperature"])
    threshold = float(calibration["scam_threshold"])
    safe_threshold = float(calibration["safe_threshold"])
    result: dict[str, Any] = {
        "model": str(args.model),
        "model_sha256": model_sha256,
        "artifact_bytes": args.model.stat().st_size,
        "runner": str(args.runner),
        "runner_sha256": runner_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "scoring": "native first-divergent verdict-token log-probability",
        "scoring_mode": "branch_token",
        "scoring_version": GGUF_SCORING_VERSION,
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": safe_threshold,
        "safe_threshold_semantics": "minimum_safe_probability",
        "calibration": calibration,
        "final_artifact_declaration": final_declaration,
        "primary_test_v8_manifest": primary_manifest,
        "data_sha256": data_sha256,
        "data_manifest": json.loads((args.data / "manifest.json").read_text()),
        "environment": {
            "python_arch": platform.machine(),
            "measurement": "persistent local native runner",
        },
        "score_cache": {
            "directory": str(cache_dir),
            "message_batch_size": 1,
            "candidate_batch_size": 3,
            "model_sequence_batch_size": 1,
            "sequence_bucket_size": 64,
            "scoring_version": GGUF_SCORING_VERSION,
        },
        "runtime_config": {
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "n_gpu_layers": args.n_gpu_layers,
            "prefix_cache_enabled": True,
        },
        "latency": {
            split: {
                "samples": len(timings),
                "median_ms": float(np.median(timings)),
                "p95_ms": float(np.percentile(timings, 95)),
                "max_ms": max(timings),
            }
            for split, timings in timings_by_split.items()
        },
    }
    for split in args.splits:
        result[split] = evaluate_slice(
            rows_by_split[split],
            scores_by_split[split],
            temperature,
            threshold,
            safe_threshold=safe_threshold,
        )
    prediction_records = prediction_ledger_records(
        rows_by_split,
        scores_by_split,
        temperature,
        threshold,
        safe_threshold,
    )
    prediction_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in prediction_records),
        encoding="utf-8",
    )
    result["prediction_ledger"] = {
        "path": str(prediction_path),
        "sha256": file_sha256(prediction_path),
        "examples": len(prediction_records),
        "contains_message_text": False,
    }
    if args.reference_predictions is not None:
        result["quantization_parity"] = quantization_parity(
            prediction_records, read_jsonl(args.reference_predictions)
        ) | {
            "reference_predictions": str(args.reference_predictions),
            "reference_sha256": file_sha256(args.reference_predictions),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
