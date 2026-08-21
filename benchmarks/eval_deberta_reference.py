#!/usr/bin/env python3
"""Rerun the public 89M DeBERTa v0.2.2 reference on ScamBench.

Architecture and feature extraction are reproduced from the model card under CC-BY-NC-4.0.
The checkpoint is downloaded for research evaluation only and is never redistributed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from scamguard.metrics import (
    binary_safety_metrics,
    choose_threshold,
    file_sha256,
    wilson_interval,
)

REPO_ID = "notd5a/deberta-v3-malicious-sms-mms-detector-v0.2.2"
REPO_REVISION = "237e993fb5ad62a44947aeadfc18503e860fd3b6"
BASE_MODEL = "microsoft/deberta-v3-base"
BASE_MODEL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
URGENCY_WORDS = {
    "urgent",
    "immediately",
    "expires",
    "verify",
    "confirm",
    "suspended",
    "locked",
    "alert",
    "action required",
    "limited time",
    "click here",
    "act now",
    "final notice",
    "winner",
    "prize",
    "claim",
    "free",
    "blocked",
    "deactivated",
    "unusual activity",
}
URL_PATTERN = re.compile(r"(https?://|www\.)\S+|\w+\.(com|net|org|io|co|uk)", re.I)
SHORTENED_DOMAINS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "smsg.io", "rb.gy"}
PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)
CURRENCY_PATTERN = re.compile(r"[$£€₹¥]|(usd|gbp|eur|inr)", re.I)
LEET_MAP = str.maketrans("013457@!", "oieastai")
OBFUSCATED_URL = re.compile(
    r"(https?(?:clue|[a-z]{4,}[a-z0-9]{2,})\b)"
    r"|(?:h\s*t\s*t\s*p)|(?:www\s*\.\s*\w)|(?:\w+\s*\.\s*(?:com|net|org|xyz|info|co)\b)",
    re.I,
)
SPACED_WORD = re.compile(r"\b(?:\w\s){3,}\w\b")


def extract_features(text: str) -> list[float]:
    words = text.split()
    letters = [character for character in text if character.isalpha()]
    chars = list(text)
    count = len(chars)
    original = [
        len(text),
        len(words),
        sum(len(word) for word in words) / max(len(words), 1),
        sum(1 for character in letters if character.isupper()) / max(len(letters), 1),
        sum(1 for character in text if character.isdigit()) / max(len(text), 1),
        sum(1 for character in text if not character.isalnum() and not character.isspace())
        / max(len(text), 1),
        text.count("!"),
        text.count("?"),
        int(bool(URL_PATTERN.search(text))),
        len(URL_PATTERN.findall(text)),
        int(any(domain in text.lower() for domain in SHORTENED_DOMAINS)),
        int(
            bool(
                [
                    match
                    for match in PHONE_PATTERN.findall(text)
                    if len(re.sub(r"\D", "", match)) >= 7
                ]
            )
        ),
        int(bool(EMAIL_PATTERN.search(text))),
        int(bool(CURRENCY_PATTERN.search(text))),
        sum(1 for word in URGENCY_WORDS if word in text.lower()),
    ]
    counts = Counter(text.lower())
    entropy = (
        -sum((value / count) * math.log2(value / count) for value in counts.values() if value > 0)
        if count > 0
        else 0.0
    )
    translated = text.translate(LEET_MAP)
    leet_changes = sum(
        1 for source, target in zip(text, translated, strict=True) if source != target
    )
    maximum_digit_run = current = 0
    for character in chars:
        if character.isdigit():
            current += 1
            maximum_digit_run = max(maximum_digit_run, current)
        else:
            current = 0
    repeats = sum(1 for index in range(1, count) if chars[index] == chars[index - 1])
    additional = [
        sum(1 for character in chars if ord(character) > 127) / max(count, 1),
        entropy,
        len(SPACED_WORD.findall(text)),
        leet_changes / max(count, 1),
        maximum_digit_run,
        repeats / max(count - 1, 1) if count > 1 else 0.0,
        len({word.lower() for word in words}) / max(len(words), 1),
        int(bool(OBFUSCATED_URL.search(text))),
    ]
    return original + additional


class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class DebertaWithFeaturesV2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deberta = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
        hidden = self.deberta.config.hidden_size
        self.attn_pool = AttentionPooling(hidden)
        self.feature_proj = torch.nn.Sequential(
            torch.nn.Linear(23, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
        )
        self.fc1 = torch.nn.Linear(2 * hidden + 128, 256)
        self.ln1 = torch.nn.LayerNorm(256)
        self.residual_block = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.output_head = torch.nn.Linear(256, 2)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, extra_features: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        combined = torch.cat(
            [
                hidden[:, 0, :],
                self.attn_pool(hidden, attention_mask),
                self.feature_proj(extra_features),
            ],
            dim=1,
        )
        value = torch.nn.functional.gelu(self.ln1(self.fc1(combined)))
        value = value + self.residual_block(value)
        return self.output_head(self.dropout(value))


def read_binary(path: Path) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row["label"] in {"SAFE", "SCAM"}]
    return (
        rows,
        [str(row["text"]) for row in rows],
        np.array([int(row["label"] == "SCAM") for row in rows]),
    )


def report_binary_slice(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    probabilities: np.ndarray,
    published_threshold: float,
    scambench_threshold: float,
    *,
    include_groups: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "examples": len(truth),
        "labels": dict(Counter("SCAM" if value else "SAFE" for value in truth)),
        "published_operating_point": binary_safety_metrics(
            truth, probabilities, published_threshold
        ),
        "scambench_operating_point": binary_safety_metrics(
            truth, probabilities, scambench_threshold
        ),
    }
    if not any(not value for value in truth):
        result["positive_only"] = True
    categories = sorted(
        {str(row["category"]) for row, value in zip(rows, truth, strict=True) if value}
    )
    result["scam_by_category"] = {}
    for category in categories:
        indices = np.array(
            [
                index
                for index, (row, value) in enumerate(zip(rows, truth, strict=True))
                if value and row["category"] == category
            ]
        )
        category_probabilities = probabilities[indices]
        operating_points = {}
        for name, threshold in (
            ("published", published_threshold),
            ("scambench", scambench_threshold),
        ):
            detected = int(np.sum(category_probabilities >= threshold))
            operating_points[name] = {
                "detected": detected,
                "recall": detected / len(indices),
                "recall_ci95": wilson_interval(detected, len(indices)),
            }
        result["scam_by_category"][category] = {
            "examples": len(indices),
            "mean_scam_probability": float(category_probabilities.mean()),
            "operating_points": operating_points,
        }
    if include_groups:
        sources = sorted({str(row["source"]) for row in rows})
        if len(sources) > 1:
            result["by_source"] = {}
            for source in sources:
                indices = np.array(
                    [index for index, row in enumerate(rows) if row["source"] == source]
                )
                result["by_source"][source] = report_binary_slice(
                    [rows[index] for index in indices],
                    truth[indices],
                    probabilities[indices],
                    published_threshold,
                    scambench_threshold,
                    include_groups=False,
                )
        if any(row.get("source_language") for row in rows):
            languages = sorted(
                {str(row.get("source_language") or "UNSPECIFIED") for row in rows}
            )
            result["by_language"] = {}
            for language in languages:
                indices = np.array(
                    [
                        index
                        for index, row in enumerate(rows)
                        if str(row.get("source_language") or "UNSPECIFIED") == language
                    ]
                )
                result["by_language"][language] = report_binary_slice(
                    [rows[index] for index in indices],
                    truth[indices],
                    probabilities[indices],
                    published_threshold,
                    scambench_threshold,
                    include_groups=False,
                )
    return result


def predict(
    model: DebertaWithFeaturesV2,
    tokenizer: Any,
    scaler: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    probabilities: list[float] = []
    latencies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            raw = np.array([extract_features(text) for text in batch], dtype=np.float32)
            # Apply the pinned scaler statistics directly. The public artifact
            # was serialized by scikit-learn 1.8 while ScamGuard's environment
            # may be newer; this formula is the complete StandardScaler transform
            # and avoids relying on version-sensitive estimator methods.
            scaled = (raw - np.asarray(scaler.mean_)) / np.asarray(scaler.scale_)
            features = torch.tensor(scaled, dtype=torch.float32, device=device)
            started = time.perf_counter_ns()
            logits = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                features,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            latencies.extend([elapsed / len(batch)] * len(batch))
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.array(probabilities), latencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--report", type=Path, default=Path("reports/runs/deberta-v022.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        selected_device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        selected_device = args.device
    if selected_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    device = torch.device(selected_device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    scaler_path = Path(hf_hub_download(REPO_ID, "scaler.pkl", revision=REPO_REVISION))
    threshold_path = Path(hf_hub_download(REPO_ID, "threshold.json", revision=REPO_REVISION))
    weights_path = Path(hf_hub_download(REPO_ID, "pytorch_model.pt", revision=REPO_REVISION))
    with warnings.catch_warnings(record=True) as scaler_warnings:
        warnings.simplefilter("always")
        scaler = joblib.load(scaler_path)
    if (
        not hasattr(scaler, "mean_")
        or not hasattr(scaler, "scale_")
        or len(scaler.mean_) != 23
        or len(scaler.scale_) != 23
    ):
        raise ValueError("pinned reference scaler is not a 23-feature StandardScaler")
    published_threshold = json.loads(threshold_path.read_text(encoding="utf-8"))[
        "optimal_threshold"
    ]
    model = DebertaWithFeaturesV2()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    # The published checkpoint mixes FP16 encoder tensors with FP32 custom-head
    # tensors. Normalize the complete scorer to FP32 so every matrix operation
    # has the same dtype on CPU (and to avoid an Apple MPS dtype assertion).
    model = model.float().to(device)

    paths = {
        "dev": args.data / "dev.jsonl",
        "test": args.data / "test.jsonl",
        "ood_financial": args.data / "ood_financial.jsonl",
        "ood_wspr": args.data / "ood_wspr.jsonl",
        "ood_forum": args.data / "ood_forum.jsonl",
        "ood_forum_materialized": args.data / "ood_forum_materialized.jsonl",
        "adversarial": args.data / "adversarial.jsonl",
    }
    binary_rows: dict[str, list[dict[str, Any]]] = {}
    truths: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    latencies: list[float] = []
    for split, path in paths.items():
        binary_rows[split], texts, truths[split] = read_binary(path)
        probabilities[split], split_latencies = predict(
            model, tokenizer, scaler, texts, device, args.batch_size
        )
        latencies.extend(split_latencies)
        print(f"{split}: scored {len(texts)}")

    scamguard_threshold = choose_threshold(truths["dev"], probabilities["dev"], args.max_fpr)
    result: dict[str, Any] = {
        "model": REPO_ID,
        "model_revision": REPO_REVISION,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "license": "CC-BY-NC-4.0; research evaluation only",
        "reference_artifacts": {
            "weights_sha256": file_sha256(weights_path),
            "scaler_sha256": file_sha256(scaler_path),
            "threshold_sha256": file_sha256(threshold_path),
            "scaler_features": len(scaler.mean_),
            "scaler_application": "manual (x - mean_) / scale_ from pinned artifact",
            "scaler_load_warnings": [str(item.message) for item in scaler_warnings],
        },
        "published_threshold": published_threshold,
        "scambench_threshold": scamguard_threshold,
        "latency_batched": {
            "median_ms_per_message": float(np.median(latencies)),
            "p95_ms_per_message": float(np.percentile(latencies, 95)),
            "batch_size": args.batch_size,
            "scope": "model forward pass only; not a batch-one product latency claim",
        },
        "model_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "data_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "requested_device": args.device,
            "device": str(device),
            "inference_dtype": str(next(model.parameters()).dtype),
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    for split in paths:
        result[split] = report_binary_slice(
            binary_rows[split],
            truths[split],
            probabilities[split],
            published_threshold,
            scamguard_threshold,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.predictions or args.report.with_suffix(".predictions.jsonl")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for split in paths:
        for row, truth, probability in zip(
            binary_rows[split], truths[split], probabilities[split], strict=True
        ):
            records.append(
                {
                    "id": row["id"],
                    "split": split,
                    "source": row["source"],
                    "source_language": row.get("source_language"),
                    "category": row["category"],
                    "truth": "SCAM" if truth else "SAFE",
                    "scam_probability": float(probability),
                    "published_threshold_scam": bool(probability >= published_threshold),
                    "scambench_threshold_scam": bool(probability >= scamguard_threshold),
                }
            )
    prediction_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    result["prediction_ledger"] = {
        "path": str(prediction_path),
        "sha256": file_sha256(prediction_path),
        "examples": len(records),
        "contains_message_text": False,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
