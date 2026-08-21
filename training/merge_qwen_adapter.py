#!/usr/bin/env python3
"""Merge a verified Qwen LoRA adapter into a portable Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.prompts import SYSTEM_PROMPT
from scamguard.qwen_scoring import candidate_token_sequences

LABELS = ("SAFE", "UNCERTAIN", "SCAM")

VERIFY_MESSAGES = (
    "Dinner is ready at six.",
    "Urgent: verify your bank login at account.example now.",
    "The courier says I owe a redelivery fee. Is this legitimate?",
)


def candidate_scores(model: object, processor: object, device: torch.device) -> torch.Tensor:
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
        maximum = max(map(len, candidates))
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
    before = candidate_scores(adapted, processor, device)
    merged = adapted.merge_and_unload().eval()
    after = candidate_scores(merged, processor, device)
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
    report = {
        "base_model": args.base,
        "base_model_revision": getattr(merged.config, "_commit_hash", args.revision),
        "adapter": str(args.adapter),
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
