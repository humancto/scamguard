from __future__ import annotations

import csv
from pathlib import Path

from scripts.check_audit_completion import audit_summary


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_audit_summary_requires_every_independent_decision(tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    write_audit(
        path,
        [
            {
                "label": "SCAM",
                "auditor_label": "SCAM",
                "label_correct": "yes",
                "contains_sensitive_data": "no",
            },
            {
                "label": "SAFE",
                "auditor_label": "",
                "label_correct": "",
                "contains_sensitive_data": "",
            },
        ],
    )

    summary, errors = audit_summary(path)

    assert not errors
    assert summary["complete_rows"] == 1
    assert summary["incomplete_rows"] == 1
    assert not summary["release_gate_passed"]


def test_audit_summary_rejects_inconsistent_correctness(tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    write_audit(
        path,
        [
            {
                "label": "SCAM",
                "auditor_label": "SAFE",
                "label_correct": "yes",
                "contains_sensitive_data": "no",
            }
        ],
    )

    summary, errors = audit_summary(path)

    assert errors
    assert not summary["release_gate_passed"]
