#!/usr/bin/env python3
"""Build a split-safe second-stage Qwen call-robustness curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.audit_source_overlap import near_overlap_indices
from training.build_qwen_sft import convert, convert_supported_rows

MULTIDOGO_SOURCE = "multidogo_human_service_dialogues"
LONG_CALL_SOURCE = "scamguard_synthetic_long_call_action_states_v1"
DIALOGUE_SOURCE = "scamguard_synthetic_dialogue_v2"
TASKMASTER_LONG_WINDOW = "recent_complete_turns_long"
SELECTION_SALT = "scamguard-qwen08-call-robustness-stage2-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ranked_sample(rows: list[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["label"] == label]
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SALT}:{label}:{row['id']}".encode()
        ).hexdigest(),
    )[:count]


def verified_artifact(
    manifest: dict[str, Any], name: str, path: Path
) -> list[dict[str, Any]]:
    contract = manifest.get("artifacts", {}).get(name)
    if not isinstance(contract, dict):
        raise ValueError(f"MultiDoGO manifest is missing {name}")
    if file_sha256(path) != contract.get("sha256"):
        raise ValueError(f"MultiDoGO {name} hash differs from its manifest")
    rows = read_jsonl(path)
    if len(rows) != contract.get("rows"):
        raise ValueError(f"MultiDoGO {name} count differs from its manifest")
    return rows


def build(
    parent: Path,
    multidogo: Path,
    output: Path,
    *,
    multidogo_repetitions: int = 3,
    core_per_label: int = 1_000,
    full_parent_replay: bool = False,
    supplement: Path | None = None,
    supplement_manifest: Path | None = None,
    overlap_references: tuple[Path, ...] = (),
    stage_name: str = "stage2",
    uncertain_repetitions: int = 0,
    synthetic_safe_repetitions: int = 0,
    supplement_safe_repetitions: int = 0,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite curriculum: {output}")
    if multidogo_repetitions < 1 or core_per_label < 0:
        raise ValueError("curriculum repetition and sample counts must be non-negative")
    if stage_name not in {"stage2", "stage3", "stage4", "stage5"}:
        raise ValueError("stage_name must be stage2, stage3, stage4, or stage5")
    if (
        uncertain_repetitions < 0
        or synthetic_safe_repetitions < 0
        or supplement_safe_repetitions < 0
    ):
        raise ValueError("targeted replay repetitions must be non-negative")
    if stage_name not in {"stage4", "stage5"} and (
        uncertain_repetitions or synthetic_safe_repetitions or supplement_safe_repetitions
    ):
        raise ValueError("targeted replay is reserved for stage4 and stage5")
    if stage_name != "stage5" and supplement_safe_repetitions:
        raise ValueError("supplement SAFE replay is reserved for stage5")
    if (supplement is None) != (supplement_manifest is None):
        raise ValueError("supplement and supplement_manifest must be supplied together")

    parent_manifest_path = parent / "manifest.json"
    parent_sft = parent / "qwen_sft"
    parent_sft_manifest_path = parent_sft / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if (
        parent_manifest.get("schema_version") != 24
        or parent_manifest.get("release_eligible") is not False
        or parent_manifest.get("publication_authorized") is not False
    ):
        raise ValueError("parent must be the frozen non-release schema-24 overlay")
    parent_sft_manifest = json.loads(
        parent_sft_manifest_path.read_text(encoding="utf-8")
    )
    if parent_sft_manifest.get("input_manifest_sha256") != file_sha256(
        parent_manifest_path
    ):
        raise ValueError("parent Qwen SFT artifact is not bound to its manifest")

    parent_raw_train = read_jsonl(parent / "train.jsonl")
    parent_sft_train = read_jsonl(parent_sft / "train.jsonl")
    parent_sft_dev = read_jsonl(parent_sft / "dev.jsonl")
    raw_by_id = {str(row["id"]): row for row in parent_raw_train}
    if len(raw_by_id) != len(parent_raw_train):
        raise ValueError("parent training IDs are not unique")
    if any(str(row["id"]) not in raw_by_id for row in parent_sft_train):
        raise ValueError("parent SFT row lacks a raw training parent")

    multidogo_manifest_path = multidogo / "manifest.json"
    multidogo_manifest = json.loads(
        multidogo_manifest_path.read_text(encoding="utf-8")
    )
    real_train = verified_artifact(
        multidogo_manifest, "real_train", multidogo / "multidogo_real_train.jsonl"
    )
    held_calls = verified_artifact(
        multidogo_manifest,
        "call_validation",
        multidogo / "multidogo_call_validation.jsonl",
    )
    complete_train_calls = [
        row for row in real_train if row.get("source_window") == "recent_complete_turns"
    ]
    if not complete_train_calls or any(
        row.get("label") != "SAFE" or row.get("source") != MULTIDOGO_SOURCE
        for row in complete_train_calls
    ):
        raise ValueError("invalid MultiDoGO complete-call training partition")
    train_families = {str(row["family_id"]) for row in complete_train_calls}
    held_families = {str(row["family_id"]) for row in held_calls}
    if train_families & held_families:
        raise ValueError("MultiDoGO complete-call family crosses training and validation")

    if full_parent_replay:
        replay = list(parent_sft_train)
    else:
        replay = []
        replay_ids: set[str] = set()
        for row in parent_sft_train:
            raw = raw_by_id[str(row["id"])]
            target = json.loads(row["messages"][-1]["content"])
            if (
                raw.get("source") in {LONG_CALL_SOURCE, DIALOGUE_SOURCE}
                or raw.get("source_window") == TASKMASTER_LONG_WINDOW
                or target.get("verdict") == "UNCERTAIN"
            ):
                replay.append(row)
                replay_ids.add(str(row["id"]))

        remaining = [
            raw_by_id[str(row["id"])]
            for row in parent_sft_train
            if str(row["id"]) not in replay_ids
        ]
        core_rows = ranked_sample(remaining, core_per_label, "SAFE") + ranked_sample(
            remaining, core_per_label, "SCAM"
        )
        sft_by_id = {str(row["id"]): row for row in parent_sft_train}
        replay.extend(sft_by_id[str(row["id"])] for row in core_rows)

    targeted_replay: list[dict[str, Any]] = []
    if uncertain_repetitions or synthetic_safe_repetitions:
        for row in parent_sft_train:
            target = json.loads(row["messages"][-1]["content"])
            verdict = target.get("verdict")
            repetitions = 0
            replay_kind = ""
            if verdict == "UNCERTAIN":
                repetitions = uncertain_repetitions
                replay_kind = "uncertain"
            elif row.get("source") == "scamguard_synthetic_v5" and verdict == "SAFE":
                repetitions = synthetic_safe_repetitions
                replay_kind = "synthetic-safe"
            for repetition in range(repetitions):
                repeated = dict(row)
                repeated["curriculum_parent_id"] = row["id"]
                repeated["curriculum_repetition"] = repetition + 1
                repeated["id"] = (
                    f"{stage_name}-{replay_kind}-r{repetition + 1}-{row['id']}"
                )
                targeted_replay.append(repeated)

    supplement_rows: list[dict[str, Any]] = []
    supplement_safe_replay: list[dict[str, Any]] = []
    supplement_contract: dict[str, Any] | None = None
    if supplement is not None and supplement_manifest is not None:
        supplement_contract = json.loads(supplement_manifest.read_text(encoding="utf-8"))
        raw_supplement = read_jsonl(supplement)
        if (
            file_sha256(supplement) != supplement_contract.get("sha256")
            or len(raw_supplement) != supplement_contract.get("rows")
            or supplement_contract.get("held_rows_copied") != 0
        ):
            raise ValueError("supplement differs from its non-held manifest contract")
        supplement_rows, excluded = convert_supported_rows(raw_supplement)
        if excluded or len(supplement_rows) != len(raw_supplement):
            raise ValueError("supplement contains unsupported grounded supervision")
        for row in supplement_rows:
            target = json.loads(row["messages"][-1]["content"])
            if target.get("verdict") != "SAFE":
                continue
            for repetition in range(supplement_safe_repetitions):
                repeated = dict(row)
                repeated["curriculum_parent_id"] = row["id"]
                repeated["curriculum_repetition"] = repetition + 1
                repeated["id"] = (
                    f"{stage_name}-supplement-safe-r{repetition + 1}-{row['id']}"
                )
                supplement_safe_replay.append(repeated)
        reference_rows: list[dict[str, Any]] = []
        reference_contracts: list[dict[str, Any]] = []
        for reference in overlap_references:
            rows = read_jsonl(reference)
            reference_rows.extend(rows)
            reference_contracts.append(
                {"path": str(reference), "rows": len(rows), "sha256": file_sha256(reference)}
            )
        overlaps = (
            near_overlap_indices(raw_supplement, reference_rows, 6) if reference_rows else set()
        )
        if overlaps:
            raise ValueError("supplement near-overlaps a held or evaluation reference")
        supplement_contract = {
            "path": str(supplement),
            "manifest_path": str(supplement_manifest),
            "manifest_sha256": file_sha256(supplement_manifest),
            "rows": len(supplement_rows),
            "sha256": file_sha256(supplement),
            "held_rows_copied": 0,
            "supplement_safe_repetitions": supplement_safe_repetitions,
            "supplement_safe_replay_rows": len(supplement_safe_replay),
            "near_overlap_radius": 6,
            "near_overlap_rows": 0,
            "overlap_references": reference_contracts,
        }

    call_rows: list[dict[str, Any]] = []
    for repetition in range(multidogo_repetitions):
        for raw in complete_train_calls:
            row = convert(raw)
            row["curriculum_parent_id"] = row["id"]
            row["curriculum_repetition"] = repetition + 1
            row["id"] = f"{stage_name}-md-r{repetition + 1}-{row['id']}"
            call_rows.append(row)

    train = sorted(
        replay + targeted_replay + call_rows + supplement_rows + supplement_safe_replay,
        key=lambda row: str(row["id"]),
    )
    train_ids = [str(row["id"]) for row in train]
    dev_ids = [str(row["id"]) for row in parent_sft_dev]
    if len(train_ids) != len(set(train_ids)) or set(train_ids) & set(dev_ids):
        raise ValueError("curriculum IDs are duplicated or cross development")
    dev_families = {str(row["family_id"]) for row in parent_sft_dev}
    if {str(row["family_id"]) for row in train} & dev_families:
        raise ValueError("curriculum family crosses parent development")

    output_sft = output / "qwen_sft"
    output_sft.mkdir(parents=True)
    train_path = output_sft / "train.jsonl"
    dev_path = output_sft / "dev.jsonl"
    write_jsonl(train_path, train)
    shutil.copy2(parent_sft / "dev.jsonl", dev_path)

    verdict_counts = Counter(
        json.loads(row["messages"][-1]["content"])["verdict"] for row in train
    )
    source_counts = Counter(str(row["source"]) for row in train)
    manifest: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment_kind": {
            "stage2": "qwen_call_robustness_stage2_curriculum",
            "stage3": "qwen_boundary_recovery_stage3_curriculum",
            "stage4": "qwen_boundary_separation_stage4_curriculum",
            "stage5": "qwen_precision_recovery_stage5_curriculum",
        }[stage_name],
        "schema_version": 24,
        "release_eligible": False,
        "publication_authorized": False,
        "parent": {
            "directory": str(parent),
            "manifest_sha256": file_sha256(parent_manifest_path),
            "sft_manifest_sha256": file_sha256(parent_sft_manifest_path),
            "train_sha256": file_sha256(parent_sft / "train.jsonl"),
            "dev_sha256": file_sha256(parent_sft / "dev.jsonl"),
        },
        "multidogo": {
            "directory": str(multidogo),
            "manifest_sha256": file_sha256(multidogo_manifest_path),
            "revision": multidogo_manifest.get("revision"),
            "license": multidogo_manifest.get("license"),
            "complete_training_calls": len(complete_train_calls),
            "training_families": len(train_families),
            "held_validation_calls_read_for_identity_only": len(held_calls),
            "held_validation_rows_used_for_fitting": 0,
            "family_cross_split": False,
            "repetitions": multidogo_repetitions,
        },
        "selection": {
            "policy": (
                "replay the complete parent SFT corpus"
                if full_parent_replay
                else (
                    "replay all existing synthetic long-call, dialogue, Taskmaster long-window, "
                    "and UNCERTAIN rows; add deterministic SAFE/SCAM core controls"
                )
            ),
            "salt": SELECTION_SALT,
            "core_per_label": core_per_label,
            "full_parent_replay": full_parent_replay,
            "parent_replay_rows": len(replay),
            "targeted_replay": {
                "training_rows_only": True,
                "uncertain_repetitions": uncertain_repetitions,
                "synthetic_safe_repetitions": synthetic_safe_repetitions,
                "supplement_safe_repetitions": supplement_safe_repetitions,
                "rows": len(targeted_replay),
                "supplement_safe_rows": len(supplement_safe_replay),
            },
            "new_complete_call_presentations": len(call_rows),
            "supplement": supplement_contract,
            "primary_test_rows_used": 0,
            "bothbosu_rows_used_for_fitting": 0,
        },
        "splits": {
            "train": {
                "rows": len(train),
                "families": len({str(row["family_id"]) for row in train}),
                "sha256": file_sha256(train_path),
                "verdicts": dict(sorted(verdict_counts.items())),
                "sources": dict(sorted(source_counts.items())),
            },
            "dev": {
                "rows": len(parent_sft_dev),
                "families": len(dev_families),
                "sha256": file_sha256(dev_path),
                "byte_identical_to_parent": file_sha256(dev_path)
                == file_sha256(parent_sft / "dev.jsonl"),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sft_manifest = {
        "artifact_schema_version": 1,
        "input_directory": str(output),
        "input_manifest_sha256": file_sha256(manifest_path),
        "policy": {
            "continuation_replay_only": True,
            "complete_call_source_partition": "publisher training only",
            "held_rows_used_for_fitting": 0,
        },
        "splits": manifest["splits"],
    }
    (output_sft / "manifest.json").write_text(
        json.dumps(sft_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--multidogo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multidogo-repetitions", type=int, default=3)
    parser.add_argument("--core-per-label", type=int, default=1_000)
    parser.add_argument("--full-parent-replay", action="store_true")
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--supplement-manifest", type=Path)
    parser.add_argument("--overlap-reference", type=Path, action="append", default=[])
    parser.add_argument(
        "--stage-name", choices=("stage2", "stage3", "stage4", "stage5"), default="stage2"
    )
    parser.add_argument("--uncertain-repetitions", type=int, default=0)
    parser.add_argument("--synthetic-safe-repetitions", type=int, default=0)
    parser.add_argument("--supplement-safe-repetitions", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.parent,
                args.multidogo,
                args.output,
                multidogo_repetitions=args.multidogo_repetitions,
                core_per_label=args.core_per_label,
                full_parent_replay=args.full_parent_replay,
                supplement=args.supplement,
                supplement_manifest=args.supplement_manifest,
                overlap_references=tuple(args.overlap_reference),
                stage_name=args.stage_name,
                uncertain_repetitions=args.uncertain_repetitions,
                synthetic_safe_repetitions=args.synthetic_safe_repetitions,
                supplement_safe_repetitions=args.supplement_safe_repetitions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
