#!/usr/bin/env python3
"""Build a non-release schema-v24 overlay from the completed internal blind review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256
from scamguard.signals import extract_signal_matches, infer_category

try:
    from scripts.blind_audit import blind_review_id
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from blind_audit import blind_review_id  # type: ignore[no-redef]

LABELS = {"SAFE", "UNCERTAIN", "SCAM"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def apply_decision(
    row: dict[str, Any], decision: dict[str, str]
) -> tuple[dict[str, Any] | None, str]:
    if decision["contains_sensitive_data"].strip().casefold() == "yes":
        return None, "quarantined_sensitive"
    auditor_label = decision["auditor_label"].strip().upper()
    if auditor_label not in LABELS:
        raise ValueError(f"invalid internal auditor label: {auditor_label!r}")
    original_label = str(row.get("label", "")).strip().upper()
    if original_label == auditor_label:
        return row, "confirmed"
    updated = dict(row)
    updated["schema24_ai_internal_original_label"] = original_label
    updated["label"] = auditor_label
    updated["label_policy"] = "ai_internal_blind_review_v1"
    if auditor_label == "SAFE":
        updated["category"] = "NONE"
    else:
        matches = extract_signal_matches(str(updated.get("text", "")))
        updated["category"] = infer_category(
            str(updated.get("text", "")),
            tuple(match.signal for match in matches),
        ).value
    return updated, "relabelled"


def load_decisions(
    decisions_path: Path,
    canonical_audit_path: Path,
) -> dict[str, dict[str, str]]:
    with decisions_path.open(encoding="utf-8", newline="") as handle:
        blind_rows = list(csv.DictReader(handle))
    blind_by_id = {str(row["id"]): row for row in blind_rows}
    if len(blind_by_id) != len(blind_rows):
        raise ValueError("internal blind review contains duplicate IDs")
    with canonical_audit_path.open(encoding="utf-8", newline="") as handle:
        canonical_rows = list(csv.DictReader(handle))
    decisions: dict[str, dict[str, str]] = {}
    for row in canonical_rows:
        identifier = str(row["id"])
        decision = blind_by_id.get(blind_review_id(identifier))
        if decision is None:
            raise ValueError(f"internal blind review is missing {identifier!r}")
        if not decision["auditor_label"].strip():
            raise ValueError(f"internal blind review is incomplete for {identifier!r}")
        decisions[identifier] = decision
    if len(decisions) != len(blind_rows):
        raise ValueError("internal blind review does not map one-to-one to the canonical audit")
    return decisions


def build_overlay(
    source: Path,
    output: Path,
    decisions_path: Path,
    canonical_audit_path: Path,
    internal_report_path: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AI-internal overlay: {output}")
    source_manifest_path = source / "manifest.json"
    internal_report = json.loads(internal_report_path.read_text(encoding="utf-8"))
    if (
        internal_report.get("review_kind") != "ai_internal_blind"
        or internal_report.get("complete_rows") != internal_report.get("rows")
        or internal_report.get("independent_human_review") is not False
        or internal_report.get("release_gate_passed") is not False
        or internal_report.get("publication_authorized") is not False
        or internal_report.get("returned_blind_audit_sha256")
        != file_sha256(decisions_path)
    ):
        raise ValueError("AI-internal audit report is incomplete or release semantics changed")
    decisions = load_decisions(decisions_path, canonical_audit_path)

    output.mkdir(parents=True)
    split_reports: dict[str, dict[str, object]] = {}
    observed_decisions: Counter[str] = Counter()
    for source_path in sorted(source.iterdir()):
        if source_path.name in {"manifest.json", "qwen_sft"}:
            continue
        destination = output / source_path.name
        if not source_path.is_file() or source_path.suffix != ".jsonl":
            if source_path.is_file():
                shutil.copy2(source_path, destination)
            continue
        input_rows = read_jsonl(source_path)
        output_rows: list[dict[str, Any]] = []
        outcomes: Counter[str] = Counter()
        label_changes: Counter[str] = Counter()
        for row in input_rows:
            identifier = str(row.get("id", ""))
            decision = decisions.get(identifier)
            if decision is None:
                output_rows.append(row)
                continue
            revised, outcome = apply_decision(row, decision)
            observed_decisions[identifier] += 1
            outcomes[outcome] += 1
            if revised is not None:
                if outcome == "relabelled":
                    label_changes[
                        f"{row.get('label')}->{revised.get('label')}"
                    ] += 1
                output_rows.append(revised)
        write_jsonl(destination, output_rows)
        split_reports[source_path.stem] = {
            "input_rows": len(input_rows),
            "output_rows": len(output_rows),
            "confirmed_audit_rows": outcomes["confirmed"],
            "relabelled_rows": outcomes["relabelled"],
            "quarantined_sensitive_rows": outcomes["quarantined_sensitive"],
            "label_changes": dict(sorted(label_changes.items())),
            "output_sha256": file_sha256(destination),
        }

    duplicated = sorted(identifier for identifier, count in observed_decisions.items() if count > 1)
    if duplicated:
        raise ValueError(f"audited rows occur in multiple processed splits: {duplicated[:5]}")
    missing = sorted(set(decisions) - set(observed_decisions))
    if missing:
        raise ValueError(f"audited rows are missing from processed data: {missing[:5]}")

    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "schema_version": 24,
        "experiment_kind": "ai_internal_exploratory_correction_overlay",
        "release_eligible": False,
        "publication_authorized": False,
        "source_processed_directory": str(source),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "internal_ai_audit_report_path": str(internal_report_path),
        "internal_ai_audit_report_sha256": file_sha256(internal_report_path),
        "internal_ai_decisions_path": str(decisions_path),
        "internal_ai_decisions_sha256": file_sha256(decisions_path),
        "canonical_audit_path": str(canonical_audit_path),
        "canonical_audit_sha256": file_sha256(canonical_audit_path),
        "audited_rows": len(decisions),
        "agreement": internal_report.get("agreement"),
        "cohen_kappa": internal_report.get("cohen_kappa"),
        "split_reports": split_reports,
        "limitations": [
            "Labels were reviewed by the same AI system assisting model development.",
            "Sensitive rows identified by that review were removed from this overlay.",
            "This overlay may be used for exploratory training and diagnostics only.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--internal-report", type=Path, required=True)
    args = parser.parse_args()
    result = build_overlay(
        args.source,
        args.output,
        args.decisions,
        args.canonical_audit,
        args.internal_report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
