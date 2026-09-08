#!/usr/bin/env python3
"""Triage publisher phone labels for action-risk conflicts without opening test.

This audit never changes a label automatically. It identifies rows whose caller
behavior conflicts with a SAFE designation, weak-evidence SCAM rows, and obvious
generation artifacts for independent review or quarantine.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
from scamguard.signals import extract_signal_matches

LABELS = {"SAFE", "SCAM"}
SPEAKER_RE = re.compile(r"\b(caller|receiver)\s*:\s*", re.IGNORECASE)
GENERATION_ARTIFACT_PATTERNS = {
    "explicit_generation_instruction": re.compile(
        r"\b(?:i will stop here|exchange requirement|as requested|as per the prompt|"
        r"to meet the .{0,24} requirement)\b",
        re.IGNORECASE,
    ),
    "assistant_preface": re.compile(
        r"^\s*(?:sure|certainly|here(?:'s| is)|below is)\b", re.IGNORECASE
    ),
}
RISK_PATTERNS = {
    "caller_requests_sensitive_value": re.compile(
        r"\b(?:(?:can|could|would|will) you(?: please)?|please|"
        r"(?:i|we)(?: would|'d)? (?:need|require|ask) you to)\s+"
        r"(?:provide|give|share|confirm|verify|enter|submit|read|tell|send)\b"
        r".{0,90}\b(?:password|passcode|pin|otp|one[- ]time (?:code|password)|"
        r"verification code|confirmation code|social security number|last four digits|"
        r"credit card|bank account|routing number|email address|shipping address|"
        r"account (?:details|information)|identity)\b",
        re.IGNORECASE,
    ),
    "caller_sends_or_directs_link": re.compile(
        r"\b(?:send|email|click|follow|open|use|access)\b.{0,70}"
        r"\b(?:link|portal|website|url)\b|"
        r"\b(?:link|portal|website|url)\b.{0,70}\b(?:enter|verify|confirm|log in)\b",
        re.IGNORECASE,
    ),
    "caller_requests_code_return": re.compile(
        r"\b(?:(?:can|could|would|will) you(?: please)?|please)\s+"
        r"(?:provide|read|give|tell|share)\b.{0,90}"
        r"\b(?:verification|confirmation|reference|security) code\b|"
        r"\b(?:verification|confirmation|reference|security) code\b.{0,60}"
        r"\b(?:to me|to us|over the phone|back to me|back to us)\b",
        re.IGNORECASE,
    ),
    "caller_requests_payment": re.compile(
        r"\b(?:pay|transfer|send|deposit|purchase|buy)\b.{0,80}"
        r"(?:[$£€]|\b(?:money|funds?|fee|gift cards?|bitcoin|crypto|wire)\b)",
        re.IGNORECASE,
    ),
    "caller_requests_remote_access": re.compile(
        r"\b(?:install|download|open|run|enable|allow)\b.{0,70}"
        r"\b(?:anydesk|teamviewer|remote (?:access|desktop|support)|screen shar(?:e|ing))\b",
        re.IGNORECASE,
    ),
}


def parse_turns(text: str) -> list[tuple[str, str]]:
    matches = list(SPEAKER_RE.finditer(text))
    turns: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        turns.append((match.group(1).casefold(), text[match.end() : end].strip()))
    return turns


def caller_risk_reasons(text: str) -> list[str]:
    caller_text = "\n".join(
        value for speaker, value in parse_turns(text) if speaker == "caller"
    )
    if not caller_text:
        return []
    return [name for name, pattern in RISK_PATTERNS.items() if pattern.search(caller_text)]


def generation_artifacts(text: str) -> list[str]:
    return [
        name for name, pattern in GENERATION_ARTIFACT_PATTERNS.items() if pattern.search(text)
    ]


def safe_conflict_priority(reasons: list[str]) -> str | None:
    high_risk = {
        "caller_requests_code_return",
        "caller_requests_payment",
        "caller_requests_remote_access",
    }
    if high_risk.intersection(reasons):
        return "high"
    if {
        "caller_requests_sensitive_value",
        "caller_sends_or_directs_link",
    } <= set(reasons):
        return "medium"
    return None


def prediction_alignment(
    path: Path,
    labels_by_id: dict[str, str],
    high_risk_ids: set[str],
    *,
    prediction_split: str = "phone_scam_validation",
) -> dict[str, Any]:
    expected_ids = set(labels_by_id)
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: invalid prediction row")
            if row.get("split") != prediction_split:
                continue
            required = {"id", "truth", "threshold_scam", "calibrated_verdict"}
            if not required <= set(row) or any(key in row for key in ("text", "message")):
                raise ValueError(f"{path}:{line_number}: invalid prediction schema")
            identifier = str(row["id"])
            if identifier in records:
                raise ValueError(f"{path}:{line_number}: duplicate prediction ID")
            records[identifier] = row
    if set(records) != expected_ids:
        raise ValueError(f"{path}: prediction IDs differ from audited validation IDs")
    for identifier, row in records.items():
        if row["truth"] != labels_by_id[identifier]:
            raise ValueError(f"{path}: prediction truth differs for {identifier}")

    safe_ids = {identifier for identifier, label in labels_by_id.items() if label == "SAFE"}
    false_positive_ids = {
        identifier
        for identifier in safe_ids
        if bool(records[identifier]["threshold_scam"])
    }
    flagged_safe_ids = safe_ids & high_risk_ids
    unflagged_safe_ids = safe_ids - high_risk_ids
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "prediction_split": prediction_split,
        "safe_rows": len(safe_ids),
        "safe_rows_with_high_risk_caller_action": len(flagged_safe_ids),
        "safe_rows_without_high_risk_caller_action": len(unflagged_safe_ids),
        "safe_false_positives": len(false_positive_ids),
        "false_positives_with_high_risk_caller_action": len(
            false_positive_ids & flagged_safe_ids
        ),
        "false_positives_without_high_risk_caller_action": len(
            false_positive_ids & unflagged_safe_ids
        ),
        "high_risk_action_trigger_rate": (
            len(false_positive_ids & flagged_safe_ids) / len(flagged_safe_ids)
            if flagged_safe_ids
            else None
        ),
        "other_safe_trigger_rate": (
            len(false_positive_ids & unflagged_safe_ids) / len(unflagged_safe_ids)
            if unflagged_safe_ids
            else None
        ),
    }


def audit(
    paths: list[Path], predictions: dict[str, Path] | None = None
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one --input is required")
    counts: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    styles: defaultdict[str, Counter[str]] = defaultdict(Counter)
    risk_reason_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    findings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    validation_labels: dict[str, str] = {}
    validation_text_hashes: dict[str, str] = {}
    inputs: list[dict[str, Any]] = []
    for path in paths:
        path_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {"id", "text", "label", "split", "source"}
                if not isinstance(row, dict) or not required <= set(row):
                    raise ValueError(f"{path}:{line_number}: invalid phone row schema")
                identifier = str(row["id"])
                if not identifier or identifier in seen_ids:
                    raise ValueError(f"{path}:{line_number}: duplicate or empty ID")
                seen_ids.add(identifier)
                if row["label"] not in LABELS or not isinstance(row["text"], str):
                    raise ValueError(f"{path}:{line_number}: invalid phone label or text")
                if row["split"] == "test":
                    raise ValueError(
                        "phone test is prediction-sealed and must not enter this audit"
                    )

                path_rows += 1
                split = str(row["split"])
                label = str(row["label"])
                text = str(row["text"])
                counts[f"{split}:{label}"] += 1
                if split == "validation":
                    validation_labels[identifier] = label
                    validation_text_hashes[identifier] = sha256(
                        text.encode("utf-8")
                    ).hexdigest()
                style = str(row.get("source_variation_style", "UNKNOWN"))
                styles[split][f"{label}:{style}"] += 1
                risk = caller_risk_reasons(text)
                for reason in risk:
                    risk_reason_counts[f"{split}:{label}"][reason] += 1
                artifacts = generation_artifacts(text)
                signal_names = sorted(
                    match.signal.value for match in extract_signal_matches(text)
                )
                row_flags = []
                conflict_priority = safe_conflict_priority(risk) if label == "SAFE" else None
                if conflict_priority:
                    row_flags.append("publisher_safe_has_high_risk_caller_action")
                if label == "SCAM" and not risk and not signal_names:
                    row_flags.append("publisher_scam_lacks_detected_runtime_evidence")
                if artifacts:
                    row_flags.append("generation_artifact")
                if not parse_turns(text):
                    row_flags.append("speaker_parse_failed")
                for flag in row_flags:
                    flags[flag] += 1
                if not row_flags:
                    continue
                findings.append(
                    {
                        "id": identifier,
                        "split": split,
                        "publisher_label": label,
                        "review_priority": conflict_priority,
                        "flags": row_flags,
                        "caller_risk_reasons": risk,
                        "generation_artifacts": artifacts,
                        "deterministic_signals": signal_names,
                        "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                        "characters": len(text),
                        "words": len(text.split()),
                        "family_id": row.get("family_id"),
                        "source_dialogue_type": row.get("source_dialogue_type"),
                        "source_length_category": row.get("source_length_category"),
                        "source_variation_style": row.get("source_variation_style"),
                        "source_batch_number": row.get("source_batch_number"),
                    }
                )
        inputs.append({"path": str(path), "sha256": file_sha256(path), "rows": path_rows})

    result = {
        "artifact_schema_version": 1,
        "contains_message_text": False,
        "policy": {
            "automatic_relabeling_allowed": False,
            "test_inspected": False,
            "safe_action_conflicts_require_review_or_quarantine": True,
            "weak_scam_evidence_requires_review_or_quarantine": True,
            "generation_artifacts_require_quarantine": True,
        },
        "inputs": inputs,
        "rows": sum(item["rows"] for item in inputs),
        "counts": dict(sorted(counts.items())),
        "variation_style_counts": {
            split: dict(sorted(values.items())) for split, values in sorted(styles.items())
        },
        "risk_reason_counts": {
            split_label: dict(sorted(values.items()))
            for split_label, values in sorted(risk_reason_counts.items())
        },
        "validation_index": [
            {
                "id": identifier,
                "publisher_label": validation_labels[identifier],
                "text_sha256": validation_text_hashes[identifier],
            }
            for identifier in sorted(validation_labels)
        ],
        "flag_counts": dict(sorted(flags.items())),
        "findings": sorted(findings, key=lambda row: (row["split"], row["id"])),
    }
    high_risk_validation_ids = {
        str(row["id"])
        for row in findings
        if row["split"] == "validation"
        and "publisher_safe_has_high_risk_caller_action" in row["flags"]
    }
    result["prediction_alignment"] = {
        name: prediction_alignment(path, validation_labels, high_risk_validation_ids)
        for name, path in sorted((predictions or {}).items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="optional text-free phone-validation prediction ledger",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions: dict[str, Path] = {}
    for value in args.prediction:
        if "=" not in value:
            raise ValueError("--prediction must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in predictions:
            raise ValueError("--prediction names and paths must be non-empty and unique")
        predictions[name] = Path(raw_path)
    result = audit(args.input, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "counts": result["counts"],
                "flag_counts": result["flag_counts"],
                "prediction_alignment": result["prediction_alignment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
