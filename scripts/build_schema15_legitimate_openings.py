#!/usr/bin/env python3
"""Append a balanced dose of original legitimate-call openings to schema v14."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.generate_legitimate_call_openings import (
        SCENARIOS,
        SOURCE,
        STRUCTURES,
        SYNTHETIC_METHOD,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from generate_legitimate_call_openings import (  # type: ignore[no-redef]
        SCENARIOS,
        SOURCE,
        STRUCTURES,
        SYNTHETIC_METHOD,
    )

PRESERVED_FILES = (
    "dev.jsonl",
    "test.jsonl",
    "ood_financial.jsonl",
    "ood_wspr.jsonl",
    "forum_validation.jsonl",
    "ood_forum.jsonl",
    "ood_azsc.jsonl",
    "quarantine_label_conflicts.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def select_balanced_dose(
    rows: list[dict[str, object]], dose_per_scenario: int
) -> list[dict[str, object]]:
    if dose_per_scenario < len(STRUCTURES) or dose_per_scenario % len(STRUCTURES):
        raise ValueError("dose per scenario must be a positive multiple of four")
    per_structure = dose_per_scenario // len(STRUCTURES)
    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scenario")), str(row.get("dialogue_structure")))].append(row)
    expected_groups = {(scenario, structure) for scenario in SCENARIOS for structure in STRUCTURES}
    if set(grouped) != expected_groups:
        missing = sorted(expected_groups - set(grouped))
        unexpected = sorted(set(grouped) - expected_groups)
        raise ValueError(
            f"legitimate-call source groups differ; missing={missing}, unexpected={unexpected}"
        )
    selected: list[dict[str, object]] = []
    for group in sorted(expected_groups):
        candidates = sorted(grouped[group], key=lambda row: str(row["id"]))
        if len(candidates) < per_structure:
            raise ValueError(f"insufficient legitimate-call rows for {group}: {len(candidates)}")
        selected.extend(candidates[:per_structure])
    return sorted(selected, key=lambda row: str(row["id"]))


def build(
    parent: Path,
    source_data: Path,
    source_manifest_path: Path,
    output: Path,
    *,
    dose_per_scenario: int = 16,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v15 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 14:
        raise ValueError("schema-v15 parent must be schema version 14")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("source") != SOURCE:
        raise ValueError("legitimate-call generator manifest has an unexpected source")
    if file_sha256(source_data) != source_manifest.get("sha256"):
        raise ValueError("legitimate-call artifact differs from its generator manifest")

    source_rows = read_jsonl(source_data)
    for row in source_rows:
        if (
            row.get("source") != SOURCE
            or row.get("license") != "Apache-2.0"
            or row.get("label") != "SAFE"
            or row.get("category") != "NONE"
            or row.get("split") != "train"
            or row.get("is_synthetic") is not True
            or row.get("synthetic_method") != SYNTHETIC_METHOD
            or row.get("external_benchmark_text_copied") is not False
        ):
            raise ValueError(f"unexpected legitimate-call row contract: {row.get('id')}")
    selected = select_balanced_dose(source_rows, dose_per_scenario)
    parent_train = read_jsonl(parent / "train.jsonl")
    parent_ids = {str(row["id"]) for row in parent_train}
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected_ids) != len(selected) or parent_ids & selected_ids:
        raise ValueError("legitimate-call increment has duplicate or parent-colliding IDs")

    output.mkdir(parents=True)
    combined_train = parent_train + selected
    write_jsonl(output / "train.jsonl", combined_train)
    for filename in PRESERVED_FILES:
        source_path = parent / filename
        if source_path.is_file():
            shutil.copy2(source_path, output / filename)

    counts = dict(parent_manifest["counts"])
    counts["train"] = len(combined_train)
    development_rows = combined_train.copy()
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    labels = dict(Counter(str(row["label"]) for row in development_rows))
    sources = dict(Counter(str(row["source"]) for row in development_rows))
    selected_scenarios = dict(Counter(str(row["scenario"]) for row in selected))
    selected_structures = dict(
        Counter(str(row["dialogue_structure"]) for row in selected)
    )
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": 15,
            "counts": counts,
            "labels": labels,
            "sources": sources,
            "parent": {
                "schema_version": 14,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema15_increment": {
                "source": SOURCE,
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_data_sha256": file_sha256(source_data),
                "rows": len(selected),
                "families": len({str(row["family_id"]) for row in selected}),
                "dose_per_scenario": dose_per_scenario,
                "scenarios": selected_scenarios,
                "dialogue_structures": selected_structures,
                "label": "SAFE",
                "license": "Apache-2.0",
                "provenance": "original deterministic synthetic legitimate service-call openings",
                "selection_signal": (
                    "AppTek open-slice aggregate and metadata metrics; no benchmark text copied"
                ),
                "used_for_fitting": True,
                "used_for_threshold": False,
                "apptek_rows_used_for_fitting": 0,
                "apptek_ood_opened": False,
            },
            "preserved_parent_artifacts": {
                filename: {
                    "sha256": file_sha256(output / filename),
                    "byte_identical_to_parent": file_sha256(output / filename)
                    == file_sha256(parent / filename),
                }
                for filename in PRESERVED_FILES
                if (output / filename).is_file()
            },
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("data/experiments/schema14-natural-dialogue/processed"),
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("data/generated/legitimate_call_openings.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/generated/legitimate_call_openings_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema15-legitimate-openings-dose16/processed"),
    )
    parser.add_argument("--dose-per-scenario", type=int, default=16)
    args = parser.parse_args()
    build(
        args.parent,
        args.source_data,
        args.source_manifest,
        args.output,
        dose_per_scenario=args.dose_per_scenario,
    )


if __name__ == "__main__":
    main()
