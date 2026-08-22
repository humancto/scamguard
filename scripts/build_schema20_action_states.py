#!/usr/bin/env python3
"""Build schema v20 with licensed call windows and long action-state contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.build_schema19_call_windows import (
        PRESERVED_FILES,
        TASKMASTER_SOURCE,
        YOUTUBE_SOURCE,
        build_taskmaster_long_rows,
        build_youtube_long_rows,
        read_jsonl,
        shared_long_extension,
        validate_youtube_recent,
        write_jsonl,
    )
    from scripts.generate_call_action_states import (
        CONTRAST_STATES,
        HOLDOUT_SCENARIOS,
        SOURCE,
        TARGET_KEYS,
        validate_action_state_rows,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_schema19_call_windows import (  # type: ignore[no-redef]
        PRESERVED_FILES,
        TASKMASTER_SOURCE,
        YOUTUBE_SOURCE,
        build_taskmaster_long_rows,
        build_youtube_long_rows,
        read_jsonl,
        shared_long_extension,
        validate_youtube_recent,
        write_jsonl,
    )
    from generate_call_action_states import (  # type: ignore[no-redef]
        CONTRAST_STATES,
        HOLDOUT_SCENARIOS,
        SOURCE,
        TARGET_KEYS,
        validate_action_state_rows,
    )

SCHEMA_VERSION = 20
LONG_STATE_SOURCE = "scamguard_synthetic_long_call_action_states_v1"
LONG_STATE_GENERATOR_VERSION = 1


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def expand_long_action_states(
    source_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Length-match each four-state family while preserving its action-only contrast."""
    validate_action_state_rows(source_rows)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["contrast_id"])].append(row)

    rows: list[dict[str, object]] = []
    for parent_contrast_id, contrast in sorted(grouped.items()):
        extension = shared_long_extension(contrast[0])
        contrast_id = "long-call-action-state-" + short_hash(
            f"v{LONG_STATE_GENERATOR_VERSION}:{parent_contrast_id}:{extension}"
        )
        contexts: set[str] = set()
        for source_row in contrast:
            original_context, ending = str(source_row["text"]).rsplit("\n", 1)
            context = f"{original_context}\n{extension}"
            contexts.add(context)
            state = str(source_row["contrast_state"])
            row = dict(source_row)
            row.update(
                {
                    "id": f"{contrast_id}-{state}",
                    "text": f"{context}\n{ending}",
                    "source": LONG_STATE_SOURCE,
                    "source_label": f"synthetic_long_call_action_state_{state}",
                    "family_id": (
                        f"synthetic:long_call_action_state:{source_row['scenario']}:"
                        f"{source_row['dialogue_structure']}:"
                        f"{source_row['context_frame']}:{source_row['risk_mechanism']}:"
                        f"v{LONG_STATE_GENERATOR_VERSION}"
                    ),
                    "contrast_id": contrast_id,
                    "parent_contrast_id": parent_contrast_id,
                    "generator_version": LONG_STATE_GENERATOR_VERSION,
                    "shared_context_sha256": hashlib.sha256(
                        context.encode("utf-8")
                    ).hexdigest(),
                    "context_window_curriculum": (
                        "long_shared_history_before_four_way_final_action_state"
                    ),
                    "selection_signal": (
                        "schema19 passed length and action-pair gates but failed absolute "
                        "real-dialogue calibration; no benchmark text copied"
                    ),
                }
            )
            rows.append(row)
        if len(contexts) != 1:
            raise ValueError(f"long action-state context differs: {parent_contrast_id}")
    rows = sorted(rows, key=lambda row: str(row["id"]))
    validate_action_state_rows(rows)
    return rows


def build(
    parent: Path,
    state_data: Path,
    state_manifest_path: Path,
    taskmaster_raw: Path,
    taskmaster_train: Path,
    taskmaster_validation: Path,
    taskmaster_manifest_path: Path,
    youtube_train: Path,
    youtube_raw: Path,
    youtube_manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v20 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 14:
        raise ValueError("schema-v20 parent must be schema version 14")

    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    if state_manifest.get("source") != SOURCE:
        raise ValueError("action-state generator source differs from schema-v20 contract")
    if file_sha256(state_data) != state_manifest.get("sha256"):
        raise ValueError("action-state artifact differs from its generator manifest")
    long_states = expand_long_action_states(read_jsonl(state_data))
    holdouts = set(HOLDOUT_SCENARIOS)
    state_train = [
        row | {"split": "train"}
        for row in long_states
        if str(row["scenario"]) not in holdouts
    ]
    state_validation = [
        row | {"split": "validation"}
        for row in long_states
        if str(row["scenario"]) in holdouts
    ]

    taskmaster_manifest = json.loads(taskmaster_manifest_path.read_text(encoding="utf-8"))
    taskmaster_artifacts = taskmaster_manifest.get("artifacts", {})
    if file_sha256(taskmaster_train) != taskmaster_artifacts.get("train", {}).get("sha256"):
        raise ValueError("Taskmaster train artifact differs from its manifest")
    if (
        file_sha256(taskmaster_validation)
        != taskmaster_artifacts.get("validation", {}).get("sha256")
    ):
        raise ValueError("Taskmaster validation artifact differs from its manifest")
    taskmaster_train_long = build_taskmaster_long_rows(
        taskmaster_raw, read_jsonl(taskmaster_train), "train"
    )
    taskmaster_validation_long = build_taskmaster_long_rows(
        taskmaster_raw, read_jsonl(taskmaster_validation), "validation"
    )
    if {str(row["family_id"]) for row in taskmaster_train_long} & {
        str(row["family_id"]) for row in taskmaster_validation_long
    }:
        raise ValueError("Taskmaster family crosses train and long-window validation")

    youtube_manifest = json.loads(youtube_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(youtube_train) != youtube_manifest.get("artifacts", {}).get(
        "train", {}
    ).get("sha256"):
        raise ValueError("YouTube train artifact differs from its manifest")
    youtube_rows = read_jsonl(youtube_train)
    youtube_recent = [
        row for row in youtube_rows if row.get("source_window") == "recent"
    ]
    validate_youtube_recent(youtube_recent)
    youtube_long = build_youtube_long_rows(youtube_raw, youtube_rows)

    parent_train = read_jsonl(parent / "train.jsonl")
    increments = taskmaster_train_long + youtube_recent + youtube_long + state_train
    all_ids = [str(row["id"]) for row in parent_train + increments]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("schema-v20 increment has a duplicate or parent-colliding ID")

    output.mkdir(parents=True)
    combined_train = parent_train + sorted(increments, key=lambda row: str(row["id"]))
    write_jsonl(output / "train.jsonl", combined_train)
    write_jsonl(output / "call_state_validation.jsonl", state_validation)
    write_jsonl(output / "call_window_validation.jsonl", taskmaster_validation_long)
    for filename in PRESERVED_FILES:
        source_path = parent / filename
        if source_path.is_file():
            shutil.copy2(source_path, output / filename)

    development_rows = list(combined_train)
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    counts = dict(parent_manifest["counts"])
    counts.update(
        {
            "train": len(combined_train),
            "call_state_validation": len(state_validation),
            "call_window_validation": len(taskmaster_validation_long),
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
                "schema_version": 14,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema20_increment": {
                "long_state_source": LONG_STATE_SOURCE,
                "state_source_data_sha256": file_sha256(state_data),
                "state_source_manifest_sha256": file_sha256(state_manifest_path),
                "state_train_rows": len(state_train),
                "state_train_families": len(
                    {str(row["contrast_id"]) for row in state_train}
                ),
                "state_validation_rows": len(state_validation),
                "state_validation_families": len(
                    {str(row["contrast_id"]) for row in state_validation}
                ),
                "contrast_states": list(CONTRAST_STATES),
                "action_target_keys": list(TARGET_KEYS),
                "taskmaster_long_train_rows": len(taskmaster_train_long),
                "taskmaster_long_validation_rows": len(taskmaster_validation_long),
                "taskmaster_raw_sha256": file_sha256(taskmaster_raw),
                "taskmaster_manifest_sha256": file_sha256(taskmaster_manifest_path),
                "youtube_recent_train_rows": len(youtube_recent),
                "youtube_long_train_rows": len(youtube_long),
                "youtube_raw_sha256": file_sha256(youtube_raw),
                "youtube_manifest_sha256": file_sha256(youtube_manifest_path),
                "licenses": {
                    LONG_STATE_SOURCE: "Apache-2.0",
                    TASKMASTER_SOURCE: "CC-BY-4.0",
                    YOUTUBE_SOURCE: "CC0-1.0",
                },
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
        "--state-data",
        type=Path,
        default=Path("data/generated/call_action_states_v1.jsonl"),
    )
    parser.add_argument(
        "--state-manifest",
        type=Path,
        default=Path("data/generated/call_action_states_v1_manifest.json"),
    )
    parser.add_argument(
        "--taskmaster-raw",
        type=Path,
        default=Path("data/raw/taskmaster1_woz_dialogues.json"),
    )
    parser.add_argument(
        "--taskmaster-train",
        type=Path,
        default=Path("data/generated/taskmaster_safe_train.jsonl"),
    )
    parser.add_argument(
        "--taskmaster-validation",
        type=Path,
        default=Path("data/external/taskmaster/taskmaster_validation.jsonl"),
    )
    parser.add_argument(
        "--taskmaster-manifest",
        type=Path,
        default=Path("data/external/taskmaster/manifest.json"),
    )
    parser.add_argument(
        "--youtube-train",
        type=Path,
        default=Path("data/external/youtube_scam_calls/youtube_scam_train.jsonl"),
    )
    parser.add_argument(
        "--youtube-raw",
        type=Path,
        default=Path("data/raw/youtube_scam_calls_v2.zip"),
    )
    parser.add_argument(
        "--youtube-manifest",
        type=Path,
        default=Path("data/external/youtube_scam_calls/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema20-action-states/processed"),
    )
    args = parser.parse_args()
    build(
        args.parent,
        args.state_data,
        args.state_manifest,
        args.taskmaster_raw,
        args.taskmaster_train,
        args.taskmaster_validation,
        args.taskmaster_manifest,
        args.youtube_train,
        args.youtube_raw,
        args.youtube_manifest,
        args.output,
    )


if __name__ == "__main__":
    main()
