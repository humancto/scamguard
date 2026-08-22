#!/usr/bin/env python3
"""Build schema v21 by adding human banking calls and grounded action states."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.build_harper_valley_calls import LICENSE, SOURCE, STATE_SOURCE
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.generate_call_action_states import CONTRAST_STATES, TARGET_KEYS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_harper_valley_calls import (  # type: ignore[no-redef]
        LICENSE,
        SOURCE,
        STATE_SOURCE,
    )
    from build_schema19_call_windows import (  # type: ignore[no-redef]
        read_jsonl,
        write_jsonl,
    )
    from generate_call_action_states import (  # type: ignore[no-redef]
        CONTRAST_STATES,
        TARGET_KEYS,
    )


SCHEMA_VERSION = 21


def artifact_rows(
    manifest: dict[str, object], name: str, path: Path
) -> list[dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise ValueError(f"HarperValleyBank manifest is missing {name}")
    contract = artifacts[name]
    if file_sha256(path) != contract.get("sha256"):
        raise ValueError(f"HarperValleyBank {name} differs from its manifest")
    rows = read_jsonl(path)
    if len(rows) != contract.get("rows"):
        raise ValueError(f"HarperValleyBank {name} count differs from its manifest")
    return rows


def validate_rows(
    real_train: list[dict[str, object]],
    state_train: list[dict[str, object]],
    real_validation: list[dict[str, object]],
    state_validation: list[dict[str, object]],
) -> None:
    for row in real_train + real_validation:
        if (
            row.get("source") != SOURCE
            or row.get("license") != LICENSE
            or row.get("label") != "SAFE"
            or row.get("is_synthetic") is not False
            or row.get("action_verdict_weight") != 1.0
            or tuple(row.get("action_targets", {})) != TARGET_KEYS
        ):
            raise ValueError(f"invalid HarperValleyBank real row: {row.get('id')}")
    for row in state_train + state_validation:
        if (
            row.get("source") != STATE_SOURCE
            or row.get("license") != LICENSE
            or row.get("is_synthetic") is not True
            or row.get("human_grounded") is not True
            or row.get("contrast_state") not in CONTRAST_STATES
            or tuple(row.get("action_targets", {})) != TARGET_KEYS
        ):
            raise ValueError(f"invalid HarperValleyBank state row: {row.get('id')}")
    train_families = {
        str(row["family_id"]) for row in real_train + state_train
    }
    validation_families = {
        str(row["family_id"]) for row in real_validation + state_validation
    }
    if train_families & validation_families:
        raise ValueError("HarperValleyBank family crosses train and validation")
    for rows, expected_split in (
        (real_train + state_train, "train"),
        (real_validation + state_validation, "validation"),
    ):
        if any(row.get("split") != expected_split for row in rows):
            raise ValueError(f"HarperValleyBank row has wrong {expected_split} split")


def build(parent: Path, harper: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v21 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 20:
        raise ValueError("schema-v21 parent must be schema version 20")
    harper_manifest_path = harper / "manifest.json"
    harper_manifest = json.loads(harper_manifest_path.read_text(encoding="utf-8"))
    if harper_manifest.get("source") != SOURCE:
        raise ValueError("HarperValleyBank source differs from schema-v21 contract")

    real_train = artifact_rows(
        harper_manifest, "real_train", harper / "harper_real_train.jsonl"
    )
    state_train = artifact_rows(
        harper_manifest, "state_train", harper / "harper_state_train.jsonl"
    )
    real_validation = artifact_rows(
        harper_manifest, "call_validation", harper / "harper_call_validation.jsonl"
    )
    state_validation = artifact_rows(
        harper_manifest, "state_validation", harper / "harper_state_validation.jsonl"
    )
    validate_rows(real_train, state_train, real_validation, state_validation)

    parent_train = read_jsonl(parent / "train.jsonl")
    increment = sorted(real_train + state_train, key=lambda row: str(row["id"]))
    all_ids = [str(row["id"]) for row in parent_train + increment]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("schema-v21 increment has a duplicate or parent-colliding ID")

    output.mkdir(parents=True)
    combined_train = parent_train + increment
    write_jsonl(output / "train.jsonl", combined_train)
    write_jsonl(output / "harper_call_validation.jsonl", real_validation)
    write_jsonl(output / "harper_state_validation.jsonl", state_validation)
    preserved_files: list[str] = []
    for source_path in sorted(parent.glob("*.jsonl")):
        if source_path.name == "train.jsonl":
            continue
        shutil.copy2(source_path, output / source_path.name)
        preserved_files.append(source_path.name)

    development_rows = list(combined_train)
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    counts = dict(parent_manifest["counts"])
    counts.update(
        {
            "train": len(combined_train),
            "harper_call_validation": len(real_validation),
            "harper_state_validation": len(state_validation),
        }
    )
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "labels": dict(Counter(str(row["label"]) for row in development_rows)),
            "sources": dict(Counter(str(row["source"]) for row in development_rows)),
            "parent": {
                "schema_version": 20,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema21_increment": {
                "source": SOURCE,
                "state_source": STATE_SOURCE,
                "license": LICENSE,
                "source_manifest_sha256": file_sha256(harper_manifest_path),
                "source_revision": harper_manifest["revision"],
                "human_call_train_rows": len(real_train),
                "human_call_validation_rows": len(real_validation),
                "human_grounded_state_train_rows": len(state_train),
                "human_grounded_state_validation_rows": len(state_validation),
                "train_families": len({str(row["family_id"]) for row in increment}),
                "validation_families": len(
                    {
                        str(row["family_id"])
                        for row in real_validation + state_validation
                    }
                ),
                "contrast_states": list(CONTRAST_STATES),
                "action_target_keys": list(TARGET_KEYS),
                "apptek_rows_used_for_fitting": 0,
                "bothbosu_rows_used_for_fitting": 0,
                "apptek_ood_opened": False,
                "bothbosu_ood_opened": False,
                "youtube_ood_opened": False,
                "moz_holdout_opened": False,
            },
            "preserved_parent_artifacts": {
                filename: {
                    "sha256": file_sha256(output / filename),
                    "byte_identical_to_parent": file_sha256(output / filename)
                    == file_sha256(parent / filename),
                }
                for filename in preserved_files
            },
        }
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("data/experiments/schema20-action-states/processed"),
    )
    parser.add_argument(
        "--harper",
        type=Path,
        default=Path("data/external/harper_valley"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema21-human-calls/processed"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.harper, args.output), indent=2))


if __name__ == "__main__":
    main()
