#!/usr/bin/env python3
"""Fail closed until the stratified label audit has independent decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABELS = {"SAFE", "UNCERTAIN", "SCAM"}
BOOLEANS = {"true": True, "false": False, "yes": True, "no": False}


def audit_summary(path: Path) -> tuple[dict[str, object], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    complete = 0
    agreements = 0
    audited_labels: Counter[str] = Counter()
    for index, row in enumerate(rows, start=2):
        auditor_label = str(row.get("auditor_label", "")).strip().upper()
        correct_text = str(row.get("label_correct", "")).strip().casefold()
        sensitive_text = str(row.get("contains_sensitive_data", "")).strip().casefold()
        populated = bool(auditor_label or correct_text or sensitive_text)
        if not populated:
            continue
        if auditor_label not in LABELS:
            errors.append(f"row {index}: invalid auditor_label {auditor_label!r}")
            continue
        if correct_text not in BOOLEANS:
            errors.append(f"row {index}: invalid label_correct {correct_text!r}")
            continue
        if sensitive_text not in BOOLEANS:
            errors.append(
                f"row {index}: invalid contains_sensitive_data {sensitive_text!r}"
            )
            continue
        agrees = auditor_label == str(row.get("label", "")).strip().upper()
        if BOOLEANS[correct_text] != agrees:
            errors.append(
                f"row {index}: label_correct conflicts with label/auditor_label agreement"
            )
            continue
        complete += 1
        agreements += int(agrees)
        audited_labels[auditor_label] += 1
    total = len(rows)
    return (
        {
            "path": str(path),
            "rows": total,
            "complete_rows": complete,
            "incomplete_rows": total - complete,
            "agreement": agreements / complete if complete else None,
            "auditor_labels": dict(audited_labels),
            "release_gate_passed": total > 0 and complete == total and not errors,
        },
        errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("data/audit/label_audit.csv"))
    args = parser.parse_args()
    summary, errors = audit_summary(args.audit)
    print(json.dumps({**summary, "errors": errors[:20]}, indent=2))
    if not summary["release_gate_passed"]:
        raise SystemExit("independent label-audit release gate is incomplete")


if __name__ == "__main__":
    main()
