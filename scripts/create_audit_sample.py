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

from scamguard.metrics import file_sha256

try:
    from scripts.audit_protocol import AUDIT_PROTOCOL_VERSION, audit_protocol_sha256
    from scripts.check_audit_completion import audit_ids_sha256, audit_input_sha256
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from audit_protocol import (  # type: ignore[no-redef]
        AUDIT_PROTOCOL_VERSION,
        audit_protocol_sha256,
    )
    from check_audit_completion import (  # type: ignore[no-redef]
        audit_ids_sha256,
        audit_input_sha256,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/audit/label_audit.csv"))
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--seed", default="scamguard-audit-v1")
    parser.add_argument(
        "--extra-split",
        action="append",
        default=[],
        help="Additional filename stem to audit, such as a schema-specific held slice.",
    )
    args = parser.parse_args()

    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    split_names = [
        "train",
        "dev",
        "test",
        "ood_financial",
        "ood_wspr",
        "forum_validation",
        "ood_forum",
        *args.extra_split,
    ]
    all_rows: list[dict[str, Any]] = []
    for split in split_names:
        for row in read_jsonl(args.data / f"{split}.jsonl"):
            copy = row | {"dataset_split": split}
            groups[(str(row["source"]), str(row["label"]))].append(copy)
            all_rows.append(copy)

    selected_by_id: dict[str, dict[str, Any]] = {}
    for stratum, rows in sorted(groups.items()):
        ranked = sorted(rows, key=lambda row: audit_key(row, args.seed))
        for row in ranked[: args.per_stratum]:
            selected_by_id[str(row["id"])] = row
        print(f"{stratum}: selected {min(args.per_stratum, len(ranked))}/{len(ranked)}")

    schema24_groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    extra_splits = set(args.extra_split)
    for row in all_rows:
        if row.get("schema24_admitted") is True or row["dataset_split"] in extra_splits:
            schema24_groups[
                (
                    str(row.get("source_domain", "unknown")),
                    str(row.get("annotation_stratum", "unstratified")),
                    str(row["dataset_split"]),
                )
            ].append(row)
    for stratum, rows in sorted(schema24_groups.items()):
        ranked = sorted(rows, key=lambda row: audit_key(row, args.seed + ":schema24"))
        for row in ranked[: args.per_stratum]:
            selected_by_id[str(row["id"])] = row
        print(
            f"schema24 {stratum}: selected {min(args.per_stratum, len(ranked))}/{len(ranked)}"
        )
    selected = list(selected_by_id.values())

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
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")
    data_manifest = args.data / "manifest.json"
    source_files = {
        f"{split}.jsonl": {
            "sha256": file_sha256(args.data / f"{split}.jsonl"),
            "rows": len(read_jsonl(args.data / f"{split}.jsonl")),
        }
        for split in split_names
    }
    manifest = {
        "artifact_schema_version": 2,
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "audit_protocol_sha256": audit_protocol_sha256(),
        "audit_path": str(args.output),
        "audit_template_sha256": file_sha256(args.output),
        "selected_inputs_sha256": audit_input_sha256(args.output),
        "data_manifest_path": str(data_manifest),
        "data_manifest_sha256": file_sha256(data_manifest),
        "source_files": source_files,
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "selected_rows": len(selected),
        "selected_ids_sha256": audit_ids_sha256(args.output),
        "strata": {
            f"{source}:{label}": min(args.per_stratum, len(rows))
            for (source, label), rows in sorted(groups.items())
        },
        "schema24_strata": {
            ":".join(stratum): min(args.per_stratum, len(rows))
            for stratum, rows in sorted(schema24_groups.items())
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} rows to {args.output}")
    print(f"wrote audit manifest to {manifest_output}")


if __name__ == "__main__":
    main()
