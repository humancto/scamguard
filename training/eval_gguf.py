#!/usr/bin/env python3
"""Rerun calibrated ScamBench verdict scoring through a quantized GGUF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoProcessor

from scamguard.metrics import choose_threshold, file_sha256
from scamguard.prompts import SYSTEM_PROMPT

try:
    from training.eval_qwen import (
        LABELS,
        choose_safe_threshold,
        evaluate_slice,
        fit_temperature,
        predict_with_abstention,
        softmax,
    )
except ModuleNotFoundError as error:
    if error.name != "training":
        raise
    # Direct script execution puts this directory, not the repository root,
    # first on sys.path.
    from eval_qwen import (  # type: ignore[no-redef]
        LABELS,
        choose_safe_threshold,
        evaluate_slice,
        fit_temperature,
        predict_with_abstention,
        softmax,
    )

SCORE_LINE = re.compile(
    r"^\s*(\d+)\t[0-9.]+\t(-?[0-9.eE+]+)\t(-?[0-9.eE+]+)\t(-?[0-9.eE+]+)\s*$",
    re.MULTILINE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_split_path(data: Path, external_data: Path, split: str) -> Path:
    candidates = [data / f"{split}.jsonl"]
    external_paths = {
        "ood_chichewa": external_data / "chichewa" / "ood_chichewa.jsonl",
        "scam_dialogue_validation": (
            external_data / "scam_dialogue" / "scam_dialogue_validation.jsonl"
        ),
        "taskmaster_validation": (
            external_data / "taskmaster" / "taskmaster_validation.jsonl"
        ),
    }
    if split in external_paths:
        candidates.append(external_paths[split])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no GGUF evaluation data found for split {split}")


def cache_identity(
    *,
    split: str,
    examples: int,
    data_sha256: str,
    model_sha256: str,
    llama_perplexity_sha256: str,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    parallel: int,
    n_gpu_layers: int,
) -> dict[str, Any]:
    return {
        "scoring_version": "gguf-verdict-likelihood-v2",
        "split": split,
        "examples": examples,
        "data_sha256": data_sha256,
        "model_sha256": model_sha256,
        "llama_perplexity_sha256": llama_perplexity_sha256,
        "ctx_size": ctx_size,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "parallel": parallel,
        "n_gpu_layers": n_gpu_layers,
        "labels": list(LABELS),
    }


def load_score_cache(
    cache_dir: Path, split: str, identity: dict[str, Any]
) -> tuple[np.ndarray, float] | None:
    metadata_path = cache_dir / f"{split}.json"
    scores_path = cache_dir / f"{split}.npy"
    if not metadata_path.is_file() or not scores_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scores = np.load(scores_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metadata.get("identity") != identity
        or scores.shape != (identity["examples"], len(LABELS))
        or not np.isfinite(scores).all()
    ):
        return None
    return scores, float(metadata["wall_seconds"])


def save_score_cache(
    cache_dir: Path,
    split: str,
    scores: np.ndarray,
    identity: dict[str, Any],
    wall_seconds: float,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores_path = cache_dir / f"{split}.npy"
    metadata_path = cache_dir / f"{split}.json"
    temporary_scores = cache_dir / f".{split}.npy.tmp"
    temporary_metadata = cache_dir / f".{split}.json.tmp"
    with temporary_scores.open("wb") as handle:
        np.save(handle, scores, allow_pickle=False)
    temporary_scores.replace(scores_path)
    temporary_metadata.write_text(
        json.dumps({"identity": identity, "wall_seconds": wall_seconds}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)


def packed_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def packed_answers(answers: list[str], labels: list[int]) -> bytes:
    return (
        struct.pack("<I", len(answers))
        + b"".join(packed_string(answer) for answer in answers)
        + struct.pack(f"<{len(labels)}i", *labels)
    )


def task_bytes(question: str, truth: str) -> bytes:
    answers = [label + '"' for label in LABELS]
    labels = [int(label == truth) for label in LABELS]
    return packed_string(question) + packed_answers(answers, labels) + packed_answers([], [])


def write_tasks(path: Path, questions: list[str], truths: list[str]) -> None:
    tasks = [task_bytes(question, truth) for question, truth in zip(questions, truths, strict=True)]
    header_size = 4 + 4 * len(tasks)
    positions = []
    offset = header_size
    for task in tasks:
        positions.append(offset)
        offset += len(task)
    path.write_bytes(
        struct.pack("<I", len(tasks))
        + struct.pack(f"<{len(positions)}I", *positions)
        + b"".join(tasks)
    )


def parse_scores(output: str, expected: int) -> np.ndarray:
    parsed = [
        (int(match[0]), *(float(value) for value in match[1:]))
        for match in SCORE_LINE.findall(output)
    ]
    if len(parsed) != expected:
        raise RuntimeError(f"llama.cpp emitted {len(parsed)} score rows; expected {expected}")
    indices = [row[0] for row in parsed]
    if indices != list(range(1, expected + 1)):
        raise RuntimeError("llama.cpp score row indices are not contiguous")
    return np.asarray([row[1:] for row in parsed], dtype=np.float64)


def prediction_ledger_records(
    rows_by_split: dict[str, list[dict[str, Any]]],
    scores_by_split: dict[str, np.ndarray],
    temperature: float,
    scam_threshold: float,
    safe_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        probabilities = softmax(scores_by_split[split], temperature)
        calibrated = predict_with_abstention(
            probabilities, scam_threshold, safe_threshold
        )
        for row, values, calibrated_index in zip(
            rows, probabilities, calibrated, strict=True
        ):
            truth_index = LABELS.index(str(row["label"]))
            records.append(
                {
                    "id": row["id"],
                    "split": split,
                    "source": row["source"],
                    "source_language": row.get("source_language"),
                    "category": row["category"],
                    "truth": row["label"],
                    "argmax": LABELS[int(values.argmax())],
                    "calibrated_verdict": LABELS[int(calibrated_index)],
                    "threshold_scam": bool(
                        values[LABELS.index("SCAM")] >= scam_threshold
                    ),
                    "negative_log_likelihood": float(
                        -math.log(max(float(values[truth_index]), 1e-9))
                    ),
                    "probabilities": {
                        label: float(values[index])
                        for index, label in enumerate(LABELS)
                    },
                }
            )
    return records


def quantization_parity(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> dict[str, Any]:
    def indexed(
        records: list[dict[str, Any]], name: str
    ) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            key = (str(record["split"]), str(record["id"]))
            if key in result:
                raise ValueError(f"duplicate {name} prediction key: {key}")
            result[key] = record
        return result

    candidate_by_key = indexed(candidate, "GGUF")
    reference_by_key = indexed(reference, "reference")
    if set(candidate_by_key) != set(reference_by_key):
        raise ValueError("GGUF and reference prediction keys differ")
    maximum_error = 0.0
    decision_mismatches: list[str] = []
    argmax_mismatches: list[str] = []
    for key in sorted(candidate_by_key):
        actual = candidate_by_key[key]
        expected = reference_by_key[key]
        for field in ("truth", "source", "source_language", "category"):
            if actual.get(field) != expected.get(field):
                raise ValueError(f"GGUF and reference metadata differ for {key}: {field}")
        maximum_error = max(
            maximum_error,
            *(
                abs(
                    float(actual["probabilities"][label])
                    - float(expected["probabilities"][label])
                )
                for label in LABELS
            ),
        )
        rendered = f"{key[0]}:{key[1]}"
        if actual["argmax"] != expected["argmax"]:
            argmax_mismatches.append(rendered)
        if actual["calibrated_verdict"] != expected["calibrated_verdict"]:
            decision_mismatches.append(rendered)
    return {
        "examples": len(candidate),
        "maximum_absolute_probability_error": maximum_error,
        "argmax_mismatch_count": len(argmax_mismatches),
        "calibrated_verdict_mismatch_count": len(decision_mismatches),
        "argmax_mismatch_sample": argmax_mismatches[:20],
        "calibrated_verdict_mismatch_sample": decision_mismatches[:20],
        "exact_argmax_parity": not argmax_mismatches,
        "exact_calibrated_verdict_parity": not decision_mismatches,
        "release_gate_passed": not decision_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--external-data", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "dev",
            "forum_validation",
            "test",
            "ood_financial",
            "ood_wspr",
            "ood_forum",
            "ood_forum_materialized",
            "adversarial",
            "ood_azsc",
        ],
    )
    parser.add_argument("--report", type=Path, default=Path("reports/runs/qwen35-2b-q4.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--fit-calibration-on-dev", action="store_true")
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=96)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    args = parser.parse_args()
    if args.fit_calibration_on_dev and args.reference_predictions:
        parser.error(
            "--reference-predictions requires frozen pre-quantization calibration; "
            "do not combine it with --fit-calibration-on-dev"
        )

    for required in (args.model, args.processor, args.calibration, args.llama_perplexity):
        if not required.exists():
            raise FileNotFoundError(required)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if tuple(calibration["labels"]) != LABELS:
        raise ValueError("calibration label order differs from GGUF evaluator")
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    split_paths = {
        split: resolve_split_path(args.data, args.external_data, split)
        for split in args.splits
    }
    rows_by_split = {}
    for split in args.splits:
        split_rows = read_jsonl(split_paths[split])
        if args.limit_per_split is not None:
            split_rows = split_rows[: args.limit_per_split]
        rows_by_split[split] = split_rows

    model_sha256 = file_sha256(args.model)
    llama_perplexity_sha256 = file_sha256(args.llama_perplexity)
    cache_dir = args.cache_dir or args.report.with_suffix(".scores")
    scores_by_split: dict[str, np.ndarray] = {}
    split_scoring: dict[str, dict[str, Any]] = {}
    for split, split_rows in rows_by_split.items():
        data_sha256 = file_sha256(split_paths[split])
        identity = cache_identity(
            split=split,
            examples=len(split_rows),
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            llama_perplexity_sha256=llama_perplexity_sha256,
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            parallel=args.parallel,
            n_gpu_layers=args.n_gpu_layers,
        )
        cached = load_score_cache(cache_dir, split, identity)
        if cached is not None:
            scores_by_split[split], duration = cached
            split_scoring[split] = {
                "examples": len(split_rows),
                "wall_seconds": duration,
                "cache_hit": True,
            }
            continue

        questions = []
        for row in split_rows:
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
            questions.append(question + '{"verdict":"')
        with tempfile.TemporaryDirectory(prefix=f"scamguard-gguf-{split}-") as directory:
            task_path = Path(directory) / "tasks.bin"
            write_tasks(
                task_path,
                questions,
                [str(row["label"]) for row in split_rows],
            )
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
                str(args.batch_size),
                "--ubatch-size",
                str(args.ubatch_size),
                "--parallel",
                str(args.parallel),
                "--n-gpu-layers",
                str(args.n_gpu_layers),
            ]
            started = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            duration = time.perf_counter() - started
            output = completed.stdout + "\n" + completed.stderr
            if completed.returncode:
                raise RuntimeError(
                    f"llama.cpp evaluation failed for {split}:\n{output[-4000:]}"
                )
            split_scores = parse_scores(output, len(split_rows))
        save_score_cache(cache_dir, split, split_scores, identity, duration)
        scores_by_split[split] = split_scores
        split_scoring[split] = {
            "examples": len(split_rows),
            "wall_seconds": duration,
            "cache_hit": False,
        }

    calibration_record: dict[str, Any]
    if args.fit_calibration_on_dev:
        if "dev" not in rows_by_split:
            raise ValueError("--fit-calibration-on-dev requires the dev split")
        if args.limit_per_split is not None:
            raise ValueError("cannot fit release calibration on a limited dev sample")
        dev_rows = rows_by_split["dev"]
        dev_truth = np.asarray([LABELS.index(str(row["label"])) for row in dev_rows])
        temperature = fit_temperature(scores_by_split["dev"], dev_truth)
        dev_probabilities = softmax(scores_by_split["dev"], temperature)
        binary_mask = np.asarray([row["label"] in {"SAFE", "SCAM"} for row in dev_rows])
        binary_truth = np.asarray([int(row["label"] == "SCAM") for row in dev_rows])[
            binary_mask
        ]
        threshold = choose_threshold(
            binary_truth,
            dev_probabilities[binary_mask, LABELS.index("SCAM")],
            args.max_fpr,
        )
        safe_threshold = choose_safe_threshold(dev_truth, dev_probabilities, threshold)
        calibration_record = {
            "backend_type": "llama_cpp_gguf_verdict_likelihood",
            "model": str(args.model),
            "model_sha256": model_sha256,
            "parent_bf16_calibration": str(args.calibration),
            "parent_bf16_calibration_sha256": file_sha256(args.calibration),
            "labels": list(LABELS),
            "temperature": temperature,
            "scam_threshold": threshold,
            "safe_threshold": safe_threshold,
            "safe_threshold_semantics": "minimum_safe_probability",
            "threshold_source": (
                "ScamBench dev scored through Q4_K_M GGUF: SCAM threshold from SAFE/SCAM "
                "subset under max FPR; SAFE threshold maximizes three-way macro-F1 after "
                "freezing the SCAM policy"
            ),
            "max_fpr": args.max_fpr,
            "scoring": "llama.cpp length-normalized teacher-forced verdict likelihood",
            "runtime_config": {
                "ctx_size": args.ctx_size,
                "batch_size": args.batch_size,
                "ubatch_size": args.ubatch_size,
                "parallel": args.parallel,
                "n_gpu_layers": args.n_gpu_layers,
            },
            "dev_data_sha256": file_sha256(split_paths["dev"]),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        }
        calibration_output = args.calibration_output or args.report.with_suffix(
            ".calibration.json"
        )
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_output.write_text(
            json.dumps(calibration_record, indent=2) + "\n", encoding="utf-8"
        )
        calibration_record["path"] = str(calibration_output)
        calibration_record["sha256"] = file_sha256(calibration_output)
    else:
        temperature = float(calibration["temperature"])
        threshold = float(calibration["scam_threshold"])
        safe_threshold = float(calibration["safe_threshold"])
        calibration_record = {
            "path": str(args.calibration),
            "sha256": file_sha256(args.calibration),
            "source": (
                "pre-quantization calibration; diagnostic only for a shifted GGUF score scale"
            ),
        }
    result: dict[str, Any] = {
        "model": str(args.model),
        "model_sha256": model_sha256,
        "artifact_bytes": args.model.stat().st_size,
        "llama_perplexity": str(args.llama_perplexity),
        "llama_perplexity_sha256": llama_perplexity_sha256,
        "scoring": "llama.cpp length-normalized teacher-forced verdict likelihood",
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": safe_threshold,
        "safe_threshold_semantics": "minimum_safe_probability",
        "calibration": calibration_record,
        "examples": sum(len(rows) for rows in rows_by_split.values()),
        "limit_per_split": args.limit_per_split,
        "runtime_config": {
            "ctx_size": args.ctx_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "parallel": args.parallel,
            "n_gpu_layers": args.n_gpu_layers,
        },
        "split_scoring": split_scoring,
        "cache_dir": str(cache_dir),
        "data_sha256": {split: file_sha256(split_paths[split]) for split in args.splits},
    }
    for split in args.splits:
        split_rows = rows_by_split[split]
        split_scores = scores_by_split[split]
        result[split] = evaluate_slice(
            split_rows,
            split_scores,
            temperature,
            threshold,
            safe_threshold=safe_threshold,
        )

    if "test" in result and args.limit_per_split is None:
        test_binary = result["test"]["binary_safety"]
        core_categories = {
            category: values
            for category, values in result["test"]["scam_by_category"].items()
            if values["examples"] >= 20
        }
        result["test_gates"] = {
            "recall": test_binary["scam_recall"] >= 0.97,
            "fpr": test_binary["false_positive_rate"] <= args.max_fpr,
            "core_category_recall": bool(core_categories)
            and all(values["recall"] >= 0.97 for values in core_categories.values()),
            "core_category_min_examples": 20,
            "core_categories_evaluated": sorted(core_categories),
            "macro_f1_stretch": result["test"]["calibrated_decision"]["macro_f1"] >= 0.94,
        }
    elif "test" in result:
        result["test_gates"] = {
            "evaluated": False,
            "reason": "limit_per_split makes this a benchmark sample, not the frozen test",
        }
    prediction_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_records = prediction_ledger_records(
        rows_by_split,
        scores_by_split,
        temperature,
        threshold,
        safe_threshold,
    )
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
    if args.reference_predictions:
        reference_records = read_jsonl(args.reference_predictions)
        result["quantization_parity"] = quantization_parity(
            prediction_records, reference_records
        ) | {
            "reference_predictions": str(args.reference_predictions),
            "reference_sha256": file_sha256(args.reference_predictions),
        }
    else:
        result["quantization_parity"] = {
            "evaluated": False,
            "release_gate_passed": False,
            "reason": "--reference-predictions was not supplied",
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": result["model"],
                "artifact_bytes": result["artifact_bytes"],
                "examples": result["examples"],
                "runtime_config": result["runtime_config"],
                "split_scoring": result["split_scoring"],
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
