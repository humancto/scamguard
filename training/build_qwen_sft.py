#!/usr/bin/env python3
"""Create strict, evidence-grounded chat examples for Qwen adapter training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
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


def convert_supported_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude SCAM rows whose text cannot support the required evidence contract."""
    converted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        target = target_for(row)
        if target["verdict"] == Verdict.SCAM.value and not target["evidence"]:
            excluded.append(row)
            continue
        validate_target(target, str(row["text"]))
        converted.append(convert(row))
    return converted, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/qwen_sft"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        source_path = args.data / f"{split}.jsonl"
        source_rows = read_jsonl(source_path)
        rows, excluded = convert_supported_rows(source_rows)
        destination = args.output / f"{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        excluded_ids = sorted(str(row["id"]) for row in excluded)
        split_reports[split] = {
            "input_rows": len(source_rows),
            "output_rows": len(rows),
            "excluded_unsupported_scam_rows": len(excluded),
            "excluded_unsupported_scam_by_source": dict(
                Counter(str(row["source"]) for row in excluded)
            ),
            "excluded_ids_sha256": hashlib.sha256(
                "\n".join(excluded_ids).encode("utf-8")
            ).hexdigest(),
            "input_sha256": file_sha256(source_path),
            "output_sha256": file_sha256(destination),
        }
        print(
            f"{split}: wrote {len(rows)} chat examples to {destination}; "
            f"excluded {len(excluded)} unsupported SCAM rows"
        )
    source_manifest = args.data / "manifest.json"
    manifest = {
        "artifact_schema_version": 1,
        "input_directory": str(args.data),
        "input_manifest_sha256": (
            file_sha256(source_manifest) if source_manifest.is_file() else None
        ),
        "policy": {
            "safe_rows_require_empty_risk_metadata": True,
            "scam_rows_require_verbatim_runtime_evidence": True,
            "unsupported_scam_rows_excluded_from_sft": True,
            "unsupported_scam_rows_relabelled": False,
            "all_non_scam_rows_retained": True,
        },
        "splits": split_reports,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
