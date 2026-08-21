#!/usr/bin/env python3
"""Replace research-corpus placeholders with safe, realistic-looking text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_ -]*)>")
REPLACEMENTS = {
    "URL": "https://secure-check.example/verify",
    "PHONE": "the number shown in the message",
    "PHONE_NUMBER": "the number shown in the message",
    "ACCOUNT_NUMBER": "the account reference shown in the message",
    "US_BANK_NUMBER": "account ending 0199",
    "EMAIL": "support@example.org",
    "EMAIL_ADDRESS": "support@example.org",
    "NAMED_ENTITY": "Northstar Services",
    "LOCATION": "Riverton",
    "DATE_TIME": "today at 2:30 PM",
    "DATE": "today",
    "TIME": "2:30 PM",
    "US_DRIVER_LICENSE": "reference number AB-0199",
    "NRP": "customer reference 0199",
    "UK_NHS": "NHS reference 0199",
    "IP_ADDRESS": "192.0.2.10",
    "US_PASSPORT": "passport ending 0199",
    "IBAN_CODE": "IBAN ending 0199",
    "US_SSN": "SSN ending 0199",
    "MEDICAL_LICENSE": "medical licence 0199",
    "IC": "identity number ending 0199",
    "NO_KERETA": "vehicle number 0199",
    "CREDIT_CARD": "card ending 0199",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(text: str) -> tuple[str, list[str]]:
    replaced: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip().replace(" ", "_")
        value = REPLACEMENTS.get(key)
        if value is None:
            return match.group(0)
        replaced.append(key)
        return value

    return PLACEHOLDER_RE.sub(replace, text), replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/ood_forum.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/ood_forum_materialized.jsonl")
    )
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    replacement_counts: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text, replacements = materialize(str(row["text"]))
            if not replacements:
                continue
            replacement_counts.update(replacements)
            output_rows.append(
                row
                | {
                    "id": f"materialized-{row['id']}",
                    "text": text,
                    "source": "imc25_forum_placeholder_materialization_v1",
                    "source_label": str(row["source_label"]),
                    "parent_id": row["id"],
                    "transform": "safe_placeholder_materialization_v1",
                }
            )

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "parent_path": str(args.input),
        "parent_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
        "labels": dict(Counter(str(row["label"]) for row in output_rows)),
        "replacement_counts": dict(replacement_counts),
        "training_or_selection_use": False,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(output_rows)} materialized forum rows; "
        f"replacements={dict(replacement_counts)}"
    )


if __name__ == "__main__":
    main()
