#!/usr/bin/env python3
"""Create a deterministic, source-stratified label audit workbook as CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/audit/label_audit.csv"))
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--seed", default="scamguard-audit-v1")
    args = parser.parse_args()

    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split in (
        "train",
        "dev",
        "test",
        "ood_financial",
        "ood_wspr",
        "forum_validation",
        "ood_forum",
    ):
        for row in read_jsonl(args.data / f"{split}.jsonl"):
            copy = row | {"dataset_split": split}
            groups[(str(row["source"]), str(row["label"]))].append(copy)

    selected = []
    for stratum, rows in sorted(groups.items()):
        ranked = sorted(rows, key=lambda row: audit_key(row, args.seed))
        selected.extend(ranked[: args.per_stratum])
        print(f"{stratum}: selected {min(args.per_stratum, len(ranked))}/{len(ranked)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "id",
        "dataset_split",
        "source",
        "source_label",
        "label",
        "category",
        "text",
        "auditor_label",
        "label_correct",
        "contains_sensitive_data",
        "notes",
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(selected, key=lambda item: (item["source"], item["label"], item["id"])):
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fields
                    if field
                    not in {"auditor_label", "label_correct", "contains_sensitive_data", "notes"}
                }
                | {
                    "auditor_label": "",
                    "label_correct": "",
                    "contains_sensitive_data": "",
                    "notes": "",
                }
            )
    print(f"wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
