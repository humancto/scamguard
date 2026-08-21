#!/usr/bin/env python3
"""Audit Qwen's generated JSON contract separately from calibrated classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.metrics import file_sha256
from scamguard.prompts import SYSTEM_PROMPT
from scamguard.taxonomy import Category, RecommendedAction, Signal, Verdict

REQUIRED_KEYS = {
    "verdict",
    "category",
    "signals",
    "evidence",
    "recommended_action",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: str) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source"]), str(row["label"]))].append(row)

    def rank(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()

    ranked = {key: sorted(group_rows, key=rank) for key, group_rows in groups.items()}
    selected = []
    depth = 0
    while len(selected) < min(limit, len(rows)):
        added = False
        for key in sorted(ranked):
            if depth < len(ranked[key]):
                selected.append(ranked[key][depth])
                added = True
                if len(selected) == min(limit, len(rows)):
                    break
        if not added:
            break
        depth += 1
    return selected


def validate_output(text: str, generated: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    if (
        generated != generated.strip()
        or not generated.startswith("{")
        or not generated.endswith("}")
    ):
        errors.append("not_exact_json_envelope")
    try:
        payload = json.loads(generated)
    except json.JSONDecodeError:
        return None, errors + ["invalid_json"]
    if not isinstance(payload, dict):
        return None, errors + ["not_object"]
    if set(payload) != REQUIRED_KEYS:
        errors.append("wrong_keys")
    if payload.get("verdict") not in {item.value for item in Verdict}:
        errors.append("invalid_verdict")
    if payload.get("category") not in {item.value for item in Category}:
        errors.append("invalid_category")
    signals = payload.get("signals")
    if not isinstance(signals, list) or any(
        signal not in {item.value for item in Signal} for signal in signals
    ):
        errors.append("invalid_signals")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(span, str) or not span or span not in text for span in evidence
    ):
        errors.append("non_verbatim_evidence")
    if isinstance(signals, list) and isinstance(evidence, list) and len(signals) != len(evidence):
        errors.append("signal_evidence_length_mismatch")
    if payload.get("recommended_action") not in {item.value for item in RecommendedAction}:
        errors.append("invalid_action")
    verdict = payload.get("verdict")
    category = payload.get("category")
    action = payload.get("recommended_action")
    if verdict == Verdict.SAFE.value and category != Category.NONE.value:
        errors.append("safe_category_not_none")
    if verdict == Verdict.SAFE.value and action != RecommendedAction.NO_ACTION.value:
        errors.append("safe_action_not_none")
    if verdict == Verdict.SAFE.value and (signals or evidence):
        errors.append("safe_has_risk_evidence")
    if verdict in {Verdict.UNCERTAIN.value, Verdict.SCAM.value} and category == Category.NONE.value:
        errors.append("risk_category_none")
    if (
        verdict in {Verdict.UNCERTAIN.value, Verdict.SCAM.value}
        and action == RecommendedAction.NO_ACTION.value
    ):
        errors.append("risk_action_none")
    return payload, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", default="scamguard-generation-audit-v1")
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Fail before loading weights when Apple Metal is not visible.",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/runs/qwen35-2b-generation.json")
    )
    args = parser.parse_args()

    rows = sample_rows(read_jsonl(args.data / f"{args.split}.jsonl"), args.limit, args.seed)
    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError(
            "--require-mps was set, but torch.backends.mps.is_available() is false. "
            "Run outside a restricted sandbox or choose an explicit CPU workflow."
        )
    device = torch.device("mps" if mps_available else "cpu")
    print(f"generation accelerator: {device}")
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
        low_cpu_mem_usage=True,
    )
    resolved_revision = getattr(base.config, "_commit_hash", None)
    if args.revision and resolved_revision and args.revision != resolved_revision:
        raise RuntimeError(
            f"loaded base revision {resolved_revision} differs from requested {args.revision}"
        )
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()

    truth = []
    predicted = []
    failures = []
    error_counts: Counter[str] = Counter()
    latencies = []
    input_token_counts = []
    output_token_counts = []
    sampled_mps_current_peak = 0
    sampled_mps_driver_peak = 0
    json_valid = 0
    strict_valid = 0
    for index, row in enumerate(rows, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{row['text']}</message>",
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            device
        )
        input_token_counts.append(int(inputs["input_ids"].shape[1]))
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                use_cache=True,
            )
        if device.type == "mps":
            torch.mps.synchronize()
            sampled_mps_current_peak = max(
                sampled_mps_current_peak, int(torch.mps.current_allocated_memory())
            )
            sampled_mps_driver_peak = max(
                sampled_mps_driver_peak, int(torch.mps.driver_allocated_memory())
            )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        generated = processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        output_token_counts.append(int(output.shape[1] - inputs["input_ids"].shape[1]))
        payload, errors = validate_output(str(row["text"]), generated)
        json_valid += int(payload is not None)
        strict_valid += int(not errors)
        error_counts.update(errors)
        truth.append(str(row["label"]))
        predicted.append(payload.get("verdict") if payload else "INVALID")
        if errors and len(failures) < 25:
            failures.append(
                {"id": row["id"], "truth": row["label"], "errors": errors, "output": generated}
            )
        if index % 20 == 0:
            print(f"generated {index}/{len(rows)}")

    labels = [item.value for item in Verdict]
    valid_indices = [index for index, value in enumerate(predicted) if value in labels]
    result = {
        "model": args.model,
        "requested_revision": args.revision,
        "base_model_revision": resolved_revision or args.revision,
        "adapter": str(args.adapter),
        "split": args.split,
        "sample_strategy": "deterministic source-label stratified",
        "input_sha256": file_sha256(args.data / f"{args.split}.jsonl"),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in rows).encode()
        ).hexdigest(),
        "examples": len(rows),
        "labels": dict(Counter(truth)),
        "json_valid_rate": json_valid / len(rows),
        "strict_schema_rate": strict_valid / len(rows),
        "valid_verdict_rate": len(valid_indices) / len(rows),
        "error_counts": dict(error_counts),
        "latency": {
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "product_batch_size": 1,
            "warmup": "none; includes the first cold generation",
            "quantization": "BF16 reference checkpoint plus LoRA adapter",
            "input_tokens": {
                "p50": int(np.percentile(input_token_counts, 50)),
                "p95": int(np.percentile(input_token_counts, 95)),
                "max": max(input_token_counts),
            },
            "output_tokens": {
                "p50": int(np.percentile(output_token_counts, 50)),
                "p95": int(np.percentile(output_token_counts, 95)),
                "max": max(output_token_counts),
            },
        },
        "memory": {
            "sampled_current_allocated_peak_bytes": sampled_mps_current_peak,
            "sampled_driver_allocated_peak_bytes": sampled_mps_driver_peak,
            "measurement": (
                "sampled after every generation; a lower bound on instantaneous peak"
            ),
        },
        "failures": failures,
    }
    if valid_indices:
        valid_truth = [truth[index] for index in valid_indices]
        valid_predicted = [predicted[index] for index in valid_indices]
        result["verdict_accuracy_valid_only"] = float(accuracy_score(valid_truth, valid_predicted))
        result["verdict_macro_f1_valid_only"] = float(
            f1_score(
                valid_truth,
                valid_predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        result["confusion_valid_only"] = confusion_matrix(
            valid_truth, valid_predicted, labels=labels
        ).tolist()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
