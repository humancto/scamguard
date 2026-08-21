#!/usr/bin/env python3
"""Create strict, evidence-grounded chat examples for Qwen adapter training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scamguard.prompts import SYSTEM_PROMPT
from scamguard.signals import choose_action, extract_signal_matches, infer_category
from scamguard.taxonomy import Category, RecommendedAction, Signal, Verdict

FORUM_CATEGORY_MAP = {
    "banking": Category.FINANCIAL_IMPERSONATION,
    "delivery": Category.DELIVERY_TOLL_PARKING,
    "government": Category.GOVERNMENT_LEGAL,
    "telecom": Category.CREDENTIAL_MFA,
    "wrong number": Category.ROMANCE_RELATIONSHIP,
    "hey mum/dad": Category.FAMILY_EXECUTIVE,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_for(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row["text"])
    matches = extract_signal_matches(text)
    signals = tuple(match.signal for match in matches)
    if row["label"] == "SAFE":
        matches = ()
        signals = ()
    action = choose_action(signals)
    if row["label"] == "SAFE":
        action = RecommendedAction.NO_ACTION
    elif action is RecommendedAction.NO_ACTION:
        action = RecommendedAction.VERIFY_OFFICIAL_CHANNEL
    if row["label"] == "SAFE":
        category = Category.NONE.value
    elif row.get("source") == "imc25_public_forum_smishing":
        category = FORUM_CATEGORY_MAP.get(
            str(row["source_label"]), infer_category(text, signals)
        ).value
    else:
        category = infer_category(text, signals).value
    return {
        "verdict": row["label"],
        "category": category,
        "signals": [signal.value for signal in signals],
        "evidence": [match.evidence.text for match in matches],
        "recommended_action": action.value,
    }


def validate_target(target: dict[str, Any], text: str) -> None:
    """Fail dataset construction if supervision violates the runtime contract."""
    Verdict(target["verdict"])
    Category(target["category"])
    RecommendedAction(target["recommended_action"])
    for signal in target["signals"]:
        Signal(signal)
    if any(not evidence or evidence not in text for evidence in target["evidence"]):
        raise ValueError("Qwen evidence must be a non-empty verbatim substring")
    if target["verdict"] == Verdict.SAFE.value and (
        target["category"] != Category.NONE.value
        or target["recommended_action"] != RecommendedAction.NO_ACTION.value
        or target["signals"]
        or target["evidence"]
    ):
        raise ValueError("SAFE supervision must clear risk metadata")
    if target["verdict"] == Verdict.SCAM.value and not target["evidence"]:
        raise ValueError("SCAM supervision requires at least one verbatim evidence span")


def convert(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row["text"])
    target = target_for(row)
    validate_target(target, text)
    return {
        "id": row["id"],
        "family_id": row["family_id"],
        "source": row["source"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{text}</message>",
            },
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/qwen_sft"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        rows = [convert(row) for row in read_jsonl(args.data / f"{split}.jsonl")]
        destination = args.output / f"{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{split}: wrote {len(rows)} chat examples to {destination}")


if __name__ == "__main__":
    main()
