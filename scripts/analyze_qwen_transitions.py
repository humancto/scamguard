#!/usr/bin/env python3
"""Compare two text-free Qwen ledgers against an exact open reference split.

The emitted audit is safe to track: it includes immutable IDs, approved metadata,
derived surface cues, and text hashes, but never copies message text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
FORBIDDEN_TEXT_FIELDS = {"text", "message", "prompt", "conversation", "transcript"}
REFERENCE_METADATA_FIELDS = (
    "family_id",
    "source_case",
    "source_dialogue_type",
    "source_length_category",
    "source_variation_style",
    "source_target_exchanges",
    "source_batch_number",
    "label_policy",
    "provenance_class",
    "is_synthetic",
)
CUE_PATTERNS = {
    "press_digit": r"\bpress\s+(?:one|two|three|[0-9])\b",
    "money_or_charge": r"[$£€]|\b(?:charged?|payment|purchase|refund|dollars?)\b",
    "credential_request": (
        r"\b(?:password|passcode|pin|verification code|account information|"
        r"social security number)\b"
    ),
    "government_claim": (
        r"\b(?:social security|irs|tax|government|police|officer|warrant|court)\b"
    ),
    "support_claim": r"\b(?:support|customer service|technician|computer)\b",
    "urgency_or_threat": (
        r"\b(?:immediately|urgent|suspend(?:ed|ing)?|arrest|legal action|"
        r"compromised|suspicious activit(?:y|ies))\b"
    ),
    "safer_channel": (
        r"\b(?:call back|official (?:app|site|website|number)|in person|"
        r"reference code|secure portal)\b"
    ),
    "generation_artifact": r"\b(?:i will stop here|exchange requirement)\b",
}


def forbidden_text_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_TEXT_FIELDS:
                found.add(str(key))
            found.update(forbidden_text_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(forbidden_text_fields(child))
    return found


def read_prediction_split(path: Path, split: str) -> dict[str, dict[str, Any]]:
    required = {
        "id",
        "split",
        "source",
        "category",
        "truth",
        "calibrated_verdict",
        "threshold_scam",
        "probabilities",
    }
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not required <= set(record):
                raise ValueError(f"{path}:{line_number}: invalid prediction schema")
            forbidden = forbidden_text_fields(record)
            if forbidden:
                raise ValueError(
                    f"{path}:{line_number}: text-bearing fields are forbidden: "
                    f"{sorted(forbidden)}"
                )
            if record["split"] != split:
                continue
            if record["truth"] not in LABELS or record["calibrated_verdict"] not in LABELS:
                raise ValueError(f"{path}:{line_number}: invalid verdict")
            probabilities = record["probabilities"]
            if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
                raise ValueError(f"{path}:{line_number}: invalid probability labels")
            values = [probabilities[label] for label in LABELS]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0.0 <= value <= 1.0
                for value in values
            ) or not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"{path}:{line_number}: invalid probabilities")
            identifier = str(record["id"])
            if not identifier or identifier in records:
                raise ValueError(f"{path}:{line_number}: duplicate or empty ID")
            records[identifier] = record
    if not records:
        raise ValueError(f"{path}: split {split!r} is absent or empty")
    return records


def read_reference(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not {"id", "text", "label"} <= set(record):
                raise ValueError(f"{path}:{line_number}: invalid reference schema")
            identifier = str(record["id"])
            if not identifier or identifier in records:
                raise ValueError(f"{path}:{line_number}: duplicate or empty ID")
            if not isinstance(record["text"], str) or record["label"] not in LABELS:
                raise ValueError(f"{path}:{line_number}: invalid reference text or label")
            records[identifier] = record
    if not records:
        raise ValueError(f"{path}: reference is empty")
    return records


def surface_profile(text: str) -> dict[str, Any]:
    normalized = text.casefold()
    return {
        "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "words": len(text.split()),
        "speaker_turn_markers": len(re.findall(r"(?im)^(?:caller|receiver)\s*:", text)),
        "cues": {
            name: bool(re.search(pattern, normalized, flags=re.IGNORECASE))
            for name, pattern in CUE_PATTERNS.items()
        },
    }


def correctness(record: dict[str, Any]) -> bool:
    return record["calibrated_verdict"] == record["truth"]


def binary_correct(record: dict[str, Any]) -> bool | None:
    if record["truth"] == "SCAM":
        return bool(record["threshold_scam"])
    if record["truth"] == "SAFE":
        return not bool(record["threshold_scam"])
    return None


def outcome(before: bool, after: bool) -> str:
    if before and not after:
        return "regressed"
    if not before and after:
        return "improved"
    if before:
        return "retained_correct"
    return "retained_error"


def compare(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    if set(baseline) != set(candidate) or set(baseline) != set(reference):
        raise ValueError("baseline, candidate, and reference ID sets must match exactly")

    rows: list[dict[str, Any]] = []
    verdict_outcomes: Counter[str] = Counter()
    binary_outcomes: Counter[str] = Counter()
    verdict_transitions: Counter[str] = Counter()
    threshold_transitions: Counter[str] = Counter()
    for identifier in sorted(baseline):
        before = baseline[identifier]
        after = candidate[identifier]
        source = reference[identifier]
        for field in ("truth", "source", "category"):
            if before.get(field) != after.get(field):
                raise ValueError(f"prediction metadata mismatch for {identifier}: {field}")
        if source["label"] != before["truth"]:
            raise ValueError(f"reference label mismatch for {identifier}")

        before_correct = correctness(before)
        after_correct = correctness(after)
        verdict_result = outcome(before_correct, after_correct)
        verdict_outcomes[verdict_result] += 1
        before_binary = binary_correct(before)
        after_binary = binary_correct(after)
        binary_result = None
        if before_binary is not None and after_binary is not None:
            binary_result = outcome(before_binary, after_binary)
            binary_outcomes[binary_result] += 1

        verdict_transition = (
            f"{before['calibrated_verdict']}->{after['calibrated_verdict']}"
        )
        threshold_transition = (
            f"{bool(before['threshold_scam'])}->{bool(after['threshold_scam'])}"
        )
        verdict_transitions[verdict_transition] += 1
        threshold_transitions[threshold_transition] += 1
        if (
            before["calibrated_verdict"] == after["calibrated_verdict"]
            and bool(before["threshold_scam"]) == bool(after["threshold_scam"])
        ):
            continue

        metadata = {
            field: source[field]
            for field in REFERENCE_METADATA_FIELDS
            if field in source
        }
        rows.append(
            {
                "id": identifier,
                "split": split,
                "source": before["source"],
                "category": before["category"],
                "truth": before["truth"],
                "baseline": {
                    "verdict": before["calibrated_verdict"],
                    "threshold_scam": bool(before["threshold_scam"]),
                    "probabilities": before["probabilities"],
                },
                "candidate": {
                    "verdict": after["calibrated_verdict"],
                    "threshold_scam": bool(after["threshold_scam"]),
                    "probabilities": after["probabilities"],
                },
                "verdict_outcome": verdict_result,
                "binary_outcome": binary_result,
                "metadata": metadata,
                "surface_profile": surface_profile(source["text"]),
            }
        )

    return {
        "artifact_schema_version": 1,
        "split": split,
        "contains_message_text": False,
        "examples": len(baseline),
        "changed_examples": len(rows),
        "verdict_outcomes": dict(sorted(verdict_outcomes.items())),
        "binary_outcomes": dict(sorted(binary_outcomes.items())),
        "verdict_transitions": dict(sorted(verdict_transitions.items())),
        "threshold_transitions": dict(sorted(threshold_transitions.items())),
        "changed_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare(
        read_prediction_split(args.baseline, args.split),
        read_prediction_split(args.candidate, args.split),
        read_reference(args.reference),
        split=args.split,
    )
    result["inputs"] = {
        "baseline": {"path": str(args.baseline), "sha256": file_sha256(args.baseline)},
        "candidate": {"path": str(args.candidate), "sha256": file_sha256(args.candidate)},
        "reference": {"path": str(args.reference), "sha256": file_sha256(args.reference)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "split": result["split"],
                "examples": result["examples"],
                "changed_examples": result["changed_examples"],
                "verdict_outcomes": result["verdict_outcomes"],
                "binary_outcomes": result["binary_outcomes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
