#!/usr/bin/env python3
"""Audit deterministic override coverage without loading a model or exposing message text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
from scamguard.policy import POLICY_VERSION, deterministic_override
from scamguard.signals import extract_signal_matches

DEFAULT_SPLITS = (
    Path("data/processed/dev.jsonl"),
    Path("data/processed/test.jsonl"),
    Path("data/processed/ood_financial.jsonl"),
    Path("data/processed/ood_wspr.jsonl"),
    Path("data/processed/forum_validation.jsonl"),
    Path("data/processed/ood_forum.jsonl"),
    Path("data/processed/ood_forum_materialized.jsonl"),
    Path("data/processed/adversarial.jsonl"),
    Path("data/processed/ood_azsc.jsonl"),
    Path("data/external/chichewa/ood_chichewa.jsonl"),
    Path("data/external/scam_dialogue/scam_dialogue_validation.jsonl"),
    Path("data/external/taskmaster/taskmaster_validation.jsonl"),
)


def audit(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    rules: Counter[str] = Counter()
    rule_truth: Counter[str] = Counter()
    for row in rows:
        signals = tuple(
            match.signal for match in extract_signal_matches(str(row["text"]))
        )
        override = deterministic_override(str(row["text"]), signals)
        if override:
            rules[override.rule_id] += 1
            rule_truth[f"{override.rule_id}:{row['label']}"] += 1
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "rules": dict(rules),
        "rule_truth": dict(rule_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("paths", type=Path, nargs="*")
    args = parser.parse_args()
    paths = tuple(args.paths) or DEFAULT_SPLITS
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"policy audit inputs are missing: {missing}")
    result = {
        "policy_version": POLICY_VERSION,
        "text_in_report": False,
        "splits": [audit(path) for path in paths],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
