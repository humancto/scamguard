#!/usr/bin/env python3
"""Merge a verified Qwen LoRA adapter into a portable Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.metrics import file_sha256
from scamguard.prompts import SYSTEM_PROMPT
from scamguard.qwen_scoring import bucketed_sequence_length, candidate_token_sequences

LABELS = ("SAFE", "UNCERTAIN", "SCAM")

VERIFY_MESSAGES = (
    "Dinner is ready at six.",
    "Urgent: verify your bank login at account.example now.",
    "The courier says I owe a redelivery fee. Is this legitimate?",
)


def load_release_calibration(
    adapter: Path, base: str, revision: str | None
) -> tuple[dict[str, object], Path]:
    path = adapter / "scamguard_calibration.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluated adapter calibration: {path}")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("base_model") != base:
        raise ValueError("adapter calibration identifies a different base model")
    if revision and calibration.get("base_model_revision") != revision:
        raise ValueError("adapter calibration identifies a different base revision")
    if calibration.get("labels") != list(LABELS):
        raise ValueError("adapter calibration has an incompatible label order")
    if calibration.get("safe_threshold_semantics") != "minimum_safe_probability":
        raise ValueError("adapter calibration has incompatible SAFE semantics")
    if calibration.get("sequence_bucket_size") != 64:
        raise ValueError("release adapter calibration must use 64-token sequence buckets")
    expected_prompt = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if calibration.get("system_prompt_sha256") != expected_prompt:
        raise ValueError("adapter calibration was fitted with a different system prompt")
    return calibration, path


def candidate_scores(
    model: object,
    processor: object,
    device: torch.device,
    *,
    sequence_bucket_size: int,
) -> torch.Tensor:
    """Return the exact verdict scores used by evaluation, not full-vocabulary logits."""

    rows: list[torch.Tensor] = []
    for text in VERIFY_MESSAGES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this message:\n<message>{text}</message>"},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += '{"verdict":"'
        candidates, common_prefix = candidate_token_sequences(
            processor.tokenizer, prompt, LABELS
        )
        suffixes = [candidate[common_prefix:] for candidate in candidates]
        maximum = bucketed_sequence_length(candidates, sequence_bucket_size)
        kept_logits = max(len(suffix) + 1 for suffix in suffixes)
        pad = processor.tokenizer.pad_token_id
        input_ids = torch.tensor(
            [[pad] * (maximum - len(candidate)) + candidate for candidate in candidates],
            device=device,
        )
        attention = torch.tensor(
            [[0] * (maximum - len(candidate)) + [1] * len(candidate) for candidate in candidates],
            device=device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=kept_logits,
            ).logits
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        scores = []
        for row, suffix in enumerate(suffixes):
            token_scores = [
                log_probabilities[row, kept_logits - len(suffix) + offset - 1, token]
                for offset, token in enumerate(suffix)
            ]
            scores.append(torch.stack(token_scores).mean())
        rows.append(torch.stack(scores).cpu())
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    calibration, calibration_path = load_release_calibration(
        args.adapter, args.base, args.revision
    )
    adapter_weights = args.adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"missing adapter weights: {adapter_weights}")

    selected_device = args.device
    if selected_device == "auto":
        selected_device = "mps" if torch.backends.mps.is_available() else "cpu"
    if selected_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(selected_device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.base, revision=args.revision)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base,
        revision=args.revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    adapted = PeftModel.from_pretrained(base, args.adapter).eval()
    sequence_bucket_size = int(calibration["sequence_bucket_size"])
    before = candidate_scores(
        adapted,
        processor,
        device,
        sequence_bucket_size=sequence_bucket_size,
    )
    merged = adapted.merge_and_unload().eval()
    after = candidate_scores(
        merged,
        processor,
        device,
        sequence_bucket_size=sequence_bucket_size,
    )
    maximum_delta = float((before - after).abs().max())
    argmax_equal = bool(torch.equal(before.argmax(dim=-1), after.argmax(dim=-1)))
    if not argmax_equal or maximum_delta > 0.05:
        raise RuntimeError(
            f"adapter merge equivalence failed: argmax_equal={argmax_equal}, "
            f"max_abs_verdict_score_delta={maximum_delta}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(args.output)
    output_calibration = args.output / "scamguard_calibration.json"
    output_calibration.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "base_model": args.base,
        "base_model_revision": getattr(merged.config, "_commit_hash", args.revision),
        "adapter": str(args.adapter),
        "adapter_weights_sha256": file_sha256(adapter_weights),
        "calibration_source": str(calibration_path),
        "calibration_source_sha256": file_sha256(calibration_path),
        "merged_calibration": str(output_calibration),
        "merged_calibration_sha256": file_sha256(output_calibration),
        "sequence_bucket_size": sequence_bucket_size,
        "verification_messages": len(VERIFY_MESSAGES),
        "argmax_equal": argmax_equal,
        "max_abs_verdict_score_delta": maximum_delta,
        "verification_scoring": "length-normalized teacher-forced verdict likelihood",
        "device": str(device),
        "dtype": str(next(merged.parameters()).dtype),
    }
    (args.output / "scamguard_merge.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
