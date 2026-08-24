#!/usr/bin/env python3
"""Evaluate Qwen verdict likelihoods under the same frozen ScamBench threshold policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.metrics import binary_safety_metrics, choose_threshold, file_sha256, wilson_interval
from scamguard.prompts import SYSTEM_PROMPT
from scamguard.qwen_scoring import bucketed_sequence_length, candidate_token_sequences

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
SCORING_VERSION = "qwen-verdict-likelihood-v2"


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def runtime_artifact_description(adapter: Path | None) -> str:
    return "BF16 reference checkpoint" + (" plus LoRA adapter" if adapter else "")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit]


def score_cache_identity(
    *,
    model: str,
    revision: str | None,
    adapter_sha256: str | None,
    data_sha256: str,
    examples: int,
    batch_size: int,
    sequence_bucket_size: int = 0,
) -> dict[str, Any]:
    return {
        "scoring_version": SCORING_VERSION,
        "model": model,
        "revision": revision,
        "adapter_sha256": adapter_sha256,
        "data_sha256": data_sha256,
        "examples": examples,
        "batch_size": batch_size,
        "sequence_bucket_size": sequence_bucket_size,
        "labels": list(LABELS),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }


def load_score_cache(
    cache_dir: Path, split: str, expected_identity: dict[str, Any]
) -> np.ndarray | None:
    metadata_path = cache_dir / f"{split}.json"
    scores_path = cache_dir / f"{split}.npy"
    if not metadata_path.is_file() or not scores_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scores = np.load(scores_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected_shape = (int(expected_identity["examples"]), len(LABELS))
    if (
        metadata != expected_identity
        or scores.shape != expected_shape
        or not np.isfinite(scores).all()
    ):
        return None
    return scores


def save_score_cache(
    cache_dir: Path,
    split: str,
    scores: np.ndarray,
    identity: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores_path = cache_dir / f"{split}.npy"
    metadata_path = cache_dir / f"{split}.json"
    temporary_scores = cache_dir / f".{split}.npy.tmp"
    temporary_metadata = cache_dir / f".{split}.json.tmp"
    with temporary_scores.open("wb") as handle:
        np.save(handle, scores, allow_pickle=False)
    temporary_scores.replace(scores_path)
    temporary_metadata.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    # Metadata is the commit marker: a partial score write can never be treated as valid cache.
    temporary_metadata.replace(metadata_path)


def sample_mps_memory(device: torch.device, telemetry: dict[str, int] | None) -> None:
    if device.type != "mps" or telemetry is None:
        return
    current = int(torch.mps.current_allocated_memory())
    driver = int(torch.mps.driver_allocated_memory())
    telemetry["sampled_current_allocated_peak_bytes"] = max(
        telemetry.get("sampled_current_allocated_peak_bytes", 0), current
    )
    telemetry["sampled_driver_allocated_peak_bytes"] = max(
        telemetry.get("sampled_driver_allocated_peak_bytes", 0), driver
    )


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted = logits / temperature
    adjusted -= adjusted.max(axis=1, keepdims=True)
    values = np.exp(adjusted)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, truth: np.ndarray) -> float:
    def negative_log_likelihood(log_temperature: float) -> float:
        probabilities = softmax(scores, math.exp(log_temperature))
        selected = probabilities[np.arange(len(truth)), truth]
        return float(-np.log(np.clip(selected, 1e-9, 1.0)).mean())

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(math.log(0.05), math.log(10.0)),
        method="bounded",
    )
    return float(math.exp(result.x))


def multiclass_calibration_metrics(
    truth: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 15
) -> dict[str, Any]:
    """Return reproducible multiclass calibration metrics.

    ECE is top-label expected calibration error with equal-width confidence
    bins over [0, 1]. Brier is the unscaled multiclass Brier score: the mean
    sum of squared probability error across all labels.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if probabilities.ndim != 2 or len(probabilities) != len(truth):
        raise ValueError("probabilities must be a row per truth label")
    if not len(truth):
        raise ValueError("calibration metrics require at least one example")

    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[truth]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    selected = np.clip(probabilities[np.arange(len(truth)), truth], 1e-9, 1.0)
    negative_log_likelihood = float(-np.log(selected).mean())

    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == truth).astype(np.float64)
    # A confidence of exactly 1.0 belongs in the last bin.
    bin_indices = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    maximum_calibration_error = 0.0
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = abs(accuracy - mean_confidence)
            ece += count / len(truth) * gap
            maximum_calibration_error = max(maximum_calibration_error, gap)
        else:
            mean_confidence = None
            accuracy = None
            gap = None
        bins.append(
            {
                "index": index,
                "lower_inclusive": lower,
                "upper_inclusive": index == n_bins - 1,
                "upper": upper,
                "examples": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return {
        "definition": "top_label_equal_width",
        "bins": n_bins,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(maximum_calibration_error),
        "multiclass_brier_score": brier,
        "negative_log_likelihood": negative_log_likelihood,
        "bin_details": bins,
    }


def predict_with_abstention(
    probabilities: np.ndarray, scam_threshold: float, safe_threshold: float
) -> np.ndarray:
    """Apply the product rule: SCAM first, SAFE only when confident, else abstain."""

    predicted = np.full(len(probabilities), LABELS.index("UNCERTAIN"), dtype=np.int64)
    predicted[probabilities[:, LABELS.index("SAFE")] >= safe_threshold] = LABELS.index("SAFE")
    predicted[probabilities[:, LABELS.index("SCAM")] >= scam_threshold] = LABELS.index("SCAM")
    return predicted


def choose_safe_threshold(
    truth: np.ndarray, probabilities: np.ndarray, scam_threshold: float
) -> float:
    """Fit the SAFE/UNCERTAIN abstention boundary on dev after freezing SCAM policy."""

    safe_probabilities = probabilities[:, LABELS.index("SAFE")]
    candidates = sorted({0.0, 1.0, *(float(value) for value in safe_probabilities)})
    ranked: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = predict_with_abstention(probabilities, scam_threshold, threshold)
        ranked.append(
            (
                float(f1_score(truth, predicted, average="macro", zero_division=0)),
                float(accuracy_score(truth, predicted)),
                threshold,
            )
        )
    # The final tie-break prefers a higher SAFE threshold: with equal measured
    # quality, abstention is safer than forcing an unsupported SAFE verdict.
    return max(ranked)[2]


def score_message(
    model: Any,
    processor: Any,
    text: str,
    device: torch.device,
    *,
    sequence_bucket_size: int = 0,
) -> np.ndarray:
    return score_messages(
        model,
        processor,
        [text],
        device,
        sequence_bucket_size=sequence_bucket_size,
    )[0]


def score_messages(
    model: Any,
    processor: Any,
    texts: list[str],
    device: torch.device,
    *,
    batch_size: int = 4,
    sequence_bucket_size: int = 0,
    memory_telemetry: dict[str, int] | None = None,
    progress_label: str | None = None,
    progress_every: int = 10,
) -> np.ndarray:
    all_scores: list[np.ndarray] = []
    total_batches = math.ceil(len(texts) / batch_size)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        sequences: list[list[int]] = []
        metadata: list[tuple[int, list[int]]] = []
        for text in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this message:\n<message>{text}</message>"},
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt += '{"verdict":"'
            candidates, common_prefix = candidate_token_sequences(
                processor.tokenizer, prompt, LABELS
            )
            for candidate in candidates:
                sequences.append(candidate)
                metadata.append((common_prefix, candidate[common_prefix:]))
        maximum = bucketed_sequence_length(sequences, sequence_bucket_size)
        # Left padding aligns every candidate suffix at the right edge. Qwen's
        # logits_to_keep can then avoid materializing a [batch, full_sequence,
        # 248k_vocab] tensor when only the positions predicting the verdict
        # suffix are required. Include one preceding position for the first
        # suffix token.
        kept_logits = max(len(tokens) + 1 for _, tokens in metadata)
        pad = processor.tokenizer.pad_token_id
        input_ids = torch.tensor(
            [[pad] * (maximum - len(sequence)) + sequence for sequence in sequences],
            device=device,
        )
        attention = torch.tensor(
            [[0] * (maximum - len(sequence)) + [1] * len(sequence) for sequence in sequences],
            device=device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=kept_logits,
            ).logits
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        flat_scores = []
        for row, (_prompt_length, tokens) in enumerate(metadata):
            token_scores = [
                log_probabilities[row, kept_logits - len(tokens) + offset - 1, token].item()
                for offset, token in enumerate(tokens)
            ]
            flat_scores.append(sum(token_scores) / len(token_scores))
        all_scores.extend(
            np.asarray(flat_scores, dtype=np.float64).reshape(len(batch), len(LABELS))
        )
        sample_mps_memory(device, memory_telemetry)
        completed_batches = start // batch_size + 1
        if progress_label and (
            completed_batches == total_batches or completed_batches % progress_every == 0
        ):
            print(
                f"{progress_label}: {completed_batches}/{total_batches} batches "
                f"({min(start + len(batch), len(texts))}/{len(texts)} rows)",
                flush=True,
            )
    return np.stack(all_scores)


def score_message_unbatched(
    model: Any, processor: Any, text: str, device: torch.device
) -> np.ndarray:
    """Reference implementation retained for scorer equivalence tests."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this message:\n<message>{text}</message>"},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += '{"verdict":"'
    sequences, common_prefix = candidate_token_sequences(processor.tokenizer, prompt, LABELS)
    maximum = max(len(sequence) for sequence in sequences)
    pad = processor.tokenizer.pad_token_id
    input_ids = torch.tensor(
        [sequence + [pad] * (maximum - len(sequence)) for sequence in sequences], device=device
    )
    attention = torch.tensor(
        [[1] * len(sequence) + [0] * (maximum - len(sequence)) for sequence in sequences],
        device=device,
    )
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    scores = []
    for row, sequence in enumerate(sequences):
        tokens = sequence[common_prefix:]
        token_scores = [
            log_probabilities[row, common_prefix + offset - 1, token].item()
            for offset, token in enumerate(tokens)
        ]
        scores.append(sum(token_scores) / len(token_scores))
    return np.array(scores)


def evaluate_slice(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    temperature: float,
    threshold: float,
    *,
    safe_threshold: float | None = None,
    include_sources: bool = True,
) -> dict[str, Any]:
    probabilities = softmax(scores, temperature)
    uncalibrated_probabilities = softmax(scores)
    truth = np.array([LABELS.index(str(row["label"])) for row in rows])
    predicted = probabilities.argmax(axis=1)
    binary_mask = np.array([row["label"] in {"SAFE", "SCAM"} for row in rows])
    binary_truth = np.array([int(row["label"] == "SCAM") for row in rows])[binary_mask]
    scam_probabilities = probabilities[binary_mask, LABELS.index("SCAM")]
    result: dict[str, Any] = {
        "examples": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "accuracy_argmax": float(accuracy_score(truth, predicted)),
        "macro_f1_argmax": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "confusion_argmax": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        "calibration": {
            "temperature": float(temperature),
            "before_temperature": multiclass_calibration_metrics(
                truth, uncalibrated_probabilities
            ),
            "after_temperature": multiclass_calibration_metrics(truth, probabilities),
        },
        "binary_safety": (
            binary_safety_metrics(binary_truth, scam_probabilities, threshold)
            if len(binary_truth)
            else None
        ),
    }
    if safe_threshold is not None:
        calibrated = predict_with_abstention(probabilities, threshold, safe_threshold)
        result["calibrated_decision"] = {
            "rule": (
                "SCAM if p_scam >= scam_threshold; else SAFE if p_safe >= "
                "safe_threshold; else UNCERTAIN"
            ),
            "safe_threshold": float(safe_threshold),
            "accuracy": float(accuracy_score(truth, calibrated)),
            "macro_f1": float(f1_score(truth, calibrated, average="macro", zero_division=0)),
            "confusion": confusion_matrix(truth, calibrated, labels=[0, 1, 2]).tolist(),
        }
    scam_categories = sorted({str(row["category"]) for row in rows if row["label"] == "SCAM"})
    result["scam_by_category"] = {}
    for category in scam_categories:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["label"] == "SCAM" and row["category"] == category
        ]
        category_probabilities = probabilities[indices, LABELS.index("SCAM")]
        detected = int(np.sum(category_probabilities >= threshold))
        result["scam_by_category"][category] = {
            "examples": len(indices),
            "detected": detected,
            "recall": detected / len(indices),
            "recall_ci95": wilson_interval(detected, len(indices)),
            "mean_scam_probability": float(category_probabilities.mean()),
        }
    if not len(binary_truth):
        result["binary_subset_empty"] = True
    elif not any(row["label"] == "SAFE" for row in rows):
        result["positive_only"] = True
    if include_sources:
        sources = sorted({str(row["source"]) for row in rows})
        if len(sources) > 1:
            result["by_source"] = {}
            for source in sources:
                indices = [index for index, row in enumerate(rows) if row["source"] == source]
                result["by_source"][source] = evaluate_slice(
                    [rows[index] for index in indices],
                    scores[indices],
                    temperature,
                    threshold,
                    safe_threshold=safe_threshold,
                    include_sources=False,
                )
        if any(row.get("source_language") for row in rows):
            languages = sorted({str(row.get("source_language") or "UNSPECIFIED") for row in rows})
            result["by_language"] = {}
            for language in languages:
                indices = [
                    index
                    for index, row in enumerate(rows)
                    if str(row.get("source_language") or "UNSPECIFIED") == language
                ]
                result["by_language"][language] = evaluate_slice(
                    [rows[index] for index in indices],
                    scores[indices],
                    temperature,
                    threshold,
                    safe_threshold=safe_threshold,
                    include_sources=False,
                )
        if any(row.get("source_domain") for row in rows):
            domains = sorted({str(row.get("source_domain") or "UNSPECIFIED") for row in rows})
            result["by_source_domain"] = {}
            for domain in domains:
                indices = [
                    index
                    for index, row in enumerate(rows)
                    if str(row.get("source_domain") or "UNSPECIFIED") == domain
                ]
                result["by_source_domain"][domain] = evaluate_slice(
                    [rows[index] for index in indices],
                    scores[indices],
                    temperature,
                    threshold,
                    safe_threshold=safe_threshold,
                    include_sources=False,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--external-data", type=Path, default=Path("data/external"))
    parser.add_argument("--report", type=Path, default=Path("reports/runs/qwen35-08b.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        help="ignored per-example ledger; defaults beside --report",
    )
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--sequence-bucket-size",
        type=int,
        default=0,
        help=(
            "Left-pad candidate sequences to a multiple of this many tokens; zero keeps "
            "dynamic shapes. Release evaluation uses 64 to match product runtime."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        help=(
            "Subset to score first; must include dev and test. Reuse --cache-dir for a later "
            "full run. Defaults to every available benchmark slice."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Resumable raw-score cache; defaults beside --report.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Fail before loading weights when Apple Metal is not visible.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.sequence_bucket_size < 0:
        parser.error("--sequence-bucket-size cannot be negative")

    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError(
            "--require-mps was set, but torch.backends.mps.is_available() is false. "
            "Run outside a restricted sandbox or choose an explicit CPU workflow."
        )
    device = torch.device("mps" if mps_available else "cpu")
    print(f"evaluation accelerator: {device}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if args.revision and resolved_revision and args.revision != resolved_revision:
        raise RuntimeError(
            f"loaded base revision {resolved_revision} differs from requested {args.revision}"
        )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device).eval()

    split_paths = {
        split: args.data / f"{split}.jsonl" for split in ("dev", "test", "ood_financial")
    }
    if (args.data / "forum_validation.jsonl").exists():
        split_paths["forum_validation"] = args.data / "forum_validation.jsonl"
    if (args.data / "ood_wspr.jsonl").exists():
        split_paths["ood_wspr"] = args.data / "ood_wspr.jsonl"
    if (args.data / "ood_forum.jsonl").exists():
        split_paths["ood_forum"] = args.data / "ood_forum.jsonl"
    if (args.data / "ood_forum_materialized.jsonl").exists():
        split_paths["ood_forum_materialized"] = args.data / "ood_forum_materialized.jsonl"
    if (args.data / "adversarial.jsonl").exists():
        split_paths["adversarial"] = args.data / "adversarial.jsonl"
    if (args.data / "ood_azsc.jsonl").exists():
        split_paths["ood_azsc"] = args.data / "ood_azsc.jsonl"
    for split in (
        "call_state_validation",
        "call_window_validation",
        "multidogo_call_validation",
        "multidogo_state_validation",
        "ftc_pattern_validation",
        "multidogo_annotation_dev",
        "multidogo_annotation_test",
    ):
        if (args.data / f"{split}.jsonl").exists():
            split_paths[split] = args.data / f"{split}.jsonl"
    chichewa_path = args.external_data / "chichewa" / "ood_chichewa.jsonl"
    if chichewa_path.exists():
        split_paths["ood_chichewa"] = chichewa_path
    scam_dialogue_path = (
        args.external_data / "scam_dialogue" / "scam_dialogue_validation.jsonl"
    )
    if scam_dialogue_path.exists():
        split_paths["scam_dialogue_validation"] = scam_dialogue_path
    taskmaster_path = args.external_data / "taskmaster" / "taskmaster_validation.jsonl"
    if taskmaster_path.exists():
        split_paths["taskmaster_validation"] = taskmaster_path
    available_splits = list(split_paths)
    splits = args.splits or available_splits
    unknown_splits = sorted(set(splits) - set(available_splits))
    if unknown_splits:
        raise ValueError(f"unknown or unavailable splits: {unknown_splits}")
    if not {"dev", "test"}.issubset(splits):
        raise ValueError("--splits must include dev and test for calibration and gates")
    rows = {split: read_jsonl(split_paths[split], args.limit) for split in splits}
    data_sha256 = {split: file_sha256(split_paths[split]) for split in splits}
    adapter_weights = args.adapter / "adapter_model.safetensors" if args.adapter else None
    adapter_sha256 = (
        file_sha256(adapter_weights) if adapter_weights and adapter_weights.is_file() else None
    )
    cache_dir = args.cache_dir or args.report.parent / f"{args.report.stem}.scores"
    all_scores: dict[str, np.ndarray] = {}
    memory_telemetry: dict[str, int] = {}
    for split, split_rows in rows.items():
        identity = score_cache_identity(
            model=args.model,
            revision=resolved_revision or args.revision,
            adapter_sha256=adapter_sha256,
            data_sha256=data_sha256[split],
            examples=len(split_rows),
            batch_size=args.batch_size,
            sequence_bucket_size=args.sequence_bucket_size,
        )
        cached = None if args.no_cache else load_score_cache(cache_dir, split, identity)
        if cached is not None:
            all_scores[split] = cached
            print(f"{split}: loaded {len(split_rows)} cached scores", flush=True)
            continue
        all_scores[split] = score_messages(
            model,
            processor,
            [str(row["text"]) for row in split_rows],
            device,
            batch_size=args.batch_size,
            sequence_bucket_size=args.sequence_bucket_size,
            memory_telemetry=memory_telemetry,
            progress_label=split,
        )
        if not args.no_cache:
            save_score_cache(cache_dir, split, all_scores[split], identity)
        print(f"{split}: scored {len(split_rows)}", flush=True)

    all_latencies: list[float] = []
    latency_input_tokens: list[int] = []
    for row in rows["test"][: min(50, len(rows["test"]))]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{row['text']}</message>",
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += '{"verdict":"'
        candidates, _ = candidate_token_sequences(processor.tokenizer, prompt, LABELS)
        latency_input_tokens.append(max(len(candidate) for candidate in candidates))
        started = time.perf_counter_ns()
        score_message(
            model,
            processor,
            str(row["text"]),
            device,
            sequence_bucket_size=args.sequence_bucket_size,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        all_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

    dev_truth = np.array([LABELS.index(str(row["label"])) for row in rows["dev"]])
    temperature = fit_temperature(all_scores["dev"], dev_truth)
    dev_probabilities = softmax(all_scores["dev"], temperature)
    binary_mask = np.array([row["label"] in {"SAFE", "SCAM"} for row in rows["dev"]])
    binary_truth = np.array([int(row["label"] == "SCAM") for row in rows["dev"]])[binary_mask]
    threshold = choose_threshold(
        binary_truth, dev_probabilities[binary_mask, LABELS.index("SCAM")], args.max_fpr
    )
    safe_threshold = choose_safe_threshold(dev_truth, dev_probabilities, threshold)
    external_data_manifests = {}
    for diagnostic in ("chichewa", "scam_dialogue", "taskmaster"):
        manifest_path = args.external_data / diagnostic / "manifest.json"
        if manifest_path.exists():
            external_data_manifests[diagnostic] = json.loads(manifest_path.read_text())
    result = {
        "model": args.model,
        "requested_revision": args.revision,
        "base_model_revision": resolved_revision or args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": adapter_sha256,
        "scoring": "length-normalized teacher-forced verdict likelihood",
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": safe_threshold,
        "safe_threshold_semantics": "minimum_safe_probability",
        "memory_footprint_bytes": model.get_memory_footprint(),
        "memory": memory_telemetry
        | {
            "peak_telemetry_available": bool(memory_telemetry),
            "measurement": (
                "sampled after every uncached scoring batch; a lower bound on instantaneous "
                "peak; absent when all requested splits load from score cache"
            )
        },
        "adapter_bytes": artifact_size(args.adapter) if args.adapter else None,
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": mps_available,
            "local_files_only": args.local_files_only,
        },
        "data_manifest": json.loads((args.data / "manifest.json").read_text()),
        "external_data_manifests": external_data_manifests,
        "data_sha256": data_sha256,
        "score_cache": {
            "enabled": not args.no_cache,
            "directory": str(cache_dir) if not args.no_cache else None,
            "identity_includes_batch_size": True,
            "message_batch_size": args.batch_size,
            "candidate_sequences_per_message": len(LABELS),
            "candidate_batch_size": args.batch_size * len(LABELS),
            "sequence_bucket_size": args.sequence_bucket_size,
            "scoring_version": SCORING_VERSION,
        },
        "latency": {
            "median_ms": float(np.median(all_latencies)),
            "p95_ms": float(np.percentile(all_latencies, 95)),
            "samples": len(all_latencies),
            "product_batch_size": 1,
            "internal_candidate_batch_size": len(LABELS),
            "sequence_bucket_size": args.sequence_bucket_size,
            "warmup": "all benchmark slices scored before timed loop",
            "quantization": runtime_artifact_description(args.adapter),
            "input_tokens": {
                "p50": int(np.percentile(latency_input_tokens, 50)),
                "p95": int(np.percentile(latency_input_tokens, 95)),
                "max": max(latency_input_tokens),
            },
        },
    }
    for split in splits:
        result[split] = evaluate_slice(
            rows[split],
            all_scores[split],
            temperature,
            threshold,
            safe_threshold=safe_threshold,
        )
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_records = []
    for split in splits:
        probabilities = softmax(all_scores[split], temperature)
        for row, values in zip(rows[split], probabilities, strict=True):
            truth_index = LABELS.index(str(row["label"]))
            calibrated_index = int(
                predict_with_abstention(values.reshape(1, -1), threshold, safe_threshold)[0]
            )
            prediction_records.append(
                {
                    "id": row["id"],
                    "split": split,
                    "source": row["source"],
                    "source_language": row.get("source_language"),
                    "category": row["category"],
                    "truth": row["label"],
                    "argmax": LABELS[int(values.argmax())],
                    "calibrated_verdict": LABELS[calibrated_index],
                    "threshold_scam": bool(values[LABELS.index("SCAM")] >= threshold),
                    "negative_log_likelihood": float(
                        -math.log(max(float(values[truth_index]), 1e-9))
                    ),
                    "probabilities": {
                        label: float(values[index]) for index, label in enumerate(LABELS)
                    },
                }
            )
    prediction_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prediction_records),
        encoding="utf-8",
    )
    result["prediction_ledger"] = {
        "path": str(prediction_path),
        "sha256": file_sha256(prediction_path),
        "examples": len(prediction_records),
        "contains_message_text": False,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.adapter:
        calibration = {
            "backend_type": "qwen_verdict_likelihood",
            "model_id": args.adapter.name,
            "adapter_sha256": adapter_sha256,
            "base_model": args.model,
            "base_model_revision": result["base_model_revision"],
            "labels": list(LABELS),
            "temperature": temperature,
            "scam_threshold": threshold,
            "safe_threshold": safe_threshold,
            "safe_threshold_semantics": "minimum_safe_probability",
            "threshold_source": (
                "ScamBench dev: SCAM threshold from SAFE/SCAM subset under max FPR; "
                "SAFE threshold maximizes three-way macro F1 after freezing SCAM policy"
            ),
            "scoring": result["scoring"],
            "sequence_bucket_size": args.sequence_bucket_size,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        }
        (args.adapter / "scamguard_calibration.json").write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
