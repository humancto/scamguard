#!/usr/bin/env python3
"""Build schema v23 from schema 20 plus bounded, evidence-focused increments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.audit_source_overlap import near_overlap_signatures, read_reference_rows
    from scripts.build_dataset import family_skeleton, normalized, simhash64
    from scripts.build_multidogo_dialogues import LICENSE as MULTIDOGO_LICENSE
    from scripts.build_multidogo_dialogues import SOURCE as MULTIDOGO_SOURCE
    from scripts.build_multidogo_dialogues import STATE_SOURCE as MULTIDOGO_STATE_SOURCE
    from scripts.build_multidogo_dialogues import TRAIN_STATE_DOMAINS, VALIDATION_STATE_DOMAINS
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.generate_call_action_states import CONTRAST_STATES, TARGET_KEYS
    from scripts.generate_ftc_pattern_action_states import (
        LICENSE as FTC_LICENSE,
    )
    from scripts.generate_ftc_pattern_action_states import (
        SOURCE as FTC_SOURCE,
    )
    from scripts.generate_ftc_pattern_action_states import (
        VALIDATION_PATTERNS as FTC_VALIDATION_PATTERNS,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from audit_source_overlap import (  # type: ignore[no-redef]
        near_overlap_signatures,
        read_reference_rows,
    )
    from build_dataset import family_skeleton, normalized, simhash64  # type: ignore[no-redef]
    from build_multidogo_dialogues import LICENSE as MULTIDOGO_LICENSE  # type: ignore[no-redef]
    from build_multidogo_dialogues import SOURCE as MULTIDOGO_SOURCE  # type: ignore[no-redef]
    from build_multidogo_dialogues import (
        STATE_SOURCE as MULTIDOGO_STATE_SOURCE,  # type: ignore[no-redef]
    )
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        TRAIN_STATE_DOMAINS,
        VALIDATION_STATE_DOMAINS,
    )
    from build_schema19_call_windows import read_jsonl, write_jsonl  # type: ignore[no-redef]
    from generate_call_action_states import CONTRAST_STATES, TARGET_KEYS  # type: ignore[no-redef]
    from generate_ftc_pattern_action_states import (  # type: ignore[no-redef]
        LICENSE as FTC_LICENSE,
    )
    from generate_ftc_pattern_action_states import SOURCE as FTC_SOURCE  # type: ignore[no-redef]
    from generate_ftc_pattern_action_states import (  # type: ignore[no-redef]
        VALIDATION_PATTERNS as FTC_VALIDATION_PATTERNS,
    )

SCHEMA_VERSION = 23
PARTITION_SALT = "scamguard-schema23-multidogo-action-calibration-v1"
CALIBRATION_FAMILIES_PER_DOMAIN = 20
MULTIDOGO_REAL_VERDICT_WEIGHT = 0.25
MULTIDOGO_DELAYED_TURNS_AFTER_ACTION = 2


def artifact_rows(manifest: dict[str, object], name: str, path: Path) -> list[dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise ValueError(f"MultiDoGO manifest is missing {name}")
    contract = artifacts[name]
    if file_sha256(path) != contract.get("sha256"):
        raise ValueError(f"MultiDoGO {name} differs from its manifest")
    rows = read_jsonl(path)
    if len(rows) != contract.get("rows"):
        raise ValueError(f"MultiDoGO {name} count differs from its manifest")
    return rows


def partition_multidogo(
    real_train: list[dict[str, object]], state_train: list[dict[str, object]]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    state_by_domain: defaultdict[str, set[str]] = defaultdict(set)
    for row in state_train:
        state_by_domain[str(row["source_domain"])].add(str(row["family_id"]))
    calibration_families: set[str] = set()
    for domain in TRAIN_STATE_DOMAINS:
        ranked = sorted(
            state_by_domain[domain],
            key=lambda family: hashlib.sha256(
                f"{PARTITION_SALT}:{domain}:{family}".encode()
            ).hexdigest(),
        )
        if len(ranked) <= CALIBRATION_FAMILIES_PER_DOMAIN:
            raise ValueError(f"MultiDoGO {domain} lacks schema-23 train families")
        calibration_families.update(ranked[:CALIBRATION_FAMILIES_PER_DOMAIN])

    fit_states = [
        row | {"split": "train"}
        for row in state_train
        if str(row["family_id"]) not in calibration_families
    ]
    calibration_states = [
        row | {"split": "action_calibration"}
        for row in state_train
        if str(row["family_id"]) in calibration_families
    ]
    fit_families = {str(row["family_id"]) for row in fit_states}
    calibration_real_families = {str(row["family_id"]) for row in calibration_states}
    fit_real = [
        row
        | {
            "split": "train",
            "action_verdict_weight": MULTIDOGO_REAL_VERDICT_WEIGHT,
        }
        for row in real_train
        if str(row["family_id"]) in fit_families
        and row.get("source_window") == "highest_risk_agent_turn"
    ]
    calibration_real = [
        row | {"split": "action_calibration"}
        for row in real_train
        if str(row["family_id"]) in calibration_real_families
        and row.get("source_window") == "highest_risk_agent_turn"
    ]
    if len(fit_real) != len(fit_families):
        raise ValueError("MultiDoGO fit family does not map to one human agent turn")
    if len(calibration_real) != len(calibration_real_families):
        raise ValueError("MultiDoGO calibration family does not map to one human agent turn")
    return (
        sorted(fit_states + fit_real, key=lambda row: str(row["id"])),
        sorted(calibration_states + calibration_real, key=lambda row: str(row["id"])),
        sorted(calibration_states, key=lambda row: str(row["id"])),
    )


def validate_state_rows(rows: list[dict[str, object]], source: str) -> None:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("source") != source
            or tuple(row.get("action_targets", {})) != TARGET_KEYS
            or row.get("contrast_state") not in CONTRAST_STATES
        ):
            raise ValueError(f"invalid action-state row: {row.get('id')}")
        grouped[str(row["contrast_id"])].append(row)
    for contrast_id, family in grouped.items():
        if len(family) != 4 or {str(row["contrast_state"]) for row in family} != set(
            CONTRAST_STATES
        ):
            raise ValueError(f"incomplete action-state family: {contrast_id}")


def shape_multidogo_state_context(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep licensed context, the decisive turn, and two shared delayed turns.

    The upstream artifact deliberately includes up to 650 characters both before and after the
    selected human-grounded action. That can make an action-supervision label invisible after the
    frozen mobile token limit. Schema 23 retains the full common prefix and two common turns after
    the action, making the supervised action observable without adding a marker to model input.
    The original source artifact and its digest remain immutable provenance.
    """

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["contrast_id"])].append(row)
    shaped: list[dict[str, object]] = []
    for contrast_id, family in grouped.items():
        ordered = sorted(family, key=lambda row: str(row["contrast_state"]))
        lines = [str(row["text"]).splitlines() for row in ordered]
        if len({len(item) for item in lines}) != 1:
            raise ValueError(f"MultiDoGO state line counts differ: {contrast_id}")
        changing = [
            index
            for index in range(len(lines[0]))
            if len({item[index] for item in lines}) > 1
        ]
        if len(changing) != 1:
            raise ValueError(
                f"MultiDoGO state family changes {len(changing)} lines: {contrast_id}"
            )
        decisive_index = changing[0]
        suffix_start = len(lines[0]) - MULTIDOGO_DELAYED_TURNS_AFTER_ACTION
        if suffix_start <= decisive_index:
            raise ValueError(f"MultiDoGO state family lacks delayed turns: {contrast_id}")
        shared_suffixes = {
            tuple(item[suffix_start:])
            for item in lines
        }
        if len(shared_suffixes) != 1:
            raise ValueError(f"MultiDoGO delayed turns differ: {contrast_id}")
        for row, row_lines in zip(ordered, lines, strict=True):
            source_text = str(row["text"])
            text = "\n".join(
                row_lines[: decisive_index + 1] + row_lines[suffix_start:]
            )
            shaped.append(
                row
                | {
                    "text": text,
                    "schema23_source_text_sha256": hashlib.sha256(
                        source_text.encode("utf-8")
                    ).hexdigest(),
                    "schema23_context_policy": (
                        "full_common_prefix_plus_decisive_action_plus_two_shared_delayed_turns"
                    ),
                }
            )
    return sorted(shaped, key=lambda row: str(row["id"]))


def remove_reference_overlap_families(
    candidate_rows: list[dict[str, object]], reference_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    candidate_signatures = [
        simhash64(family_skeleton(str(row["text"]))) for row in candidate_rows
    ]
    reference_signatures = [
        simhash64(family_skeleton(str(row["text"]))) for row in reference_rows
    ]
    overlap_indices = near_overlap_signatures(
        candidate_signatures, reference_signatures, radius=6
    )
    exact_reference = {normalized(str(row["text"])) for row in reference_rows}
    exact_overlap_indices = {
        index
        for index, row in enumerate(candidate_rows)
        if normalized(str(row["text"])) in exact_reference
    }
    removed_families = {
        str(candidate_rows[index]["family_id"]) for index in overlap_indices
    }
    kept = [
        row for row in candidate_rows if str(row["family_id"]) not in removed_families
    ]
    return kept, {
        "candidate_rows_before_overlap_control": len(candidate_rows),
        "exact_overlap_rows": len(exact_overlap_indices),
        "near_overlap_rows_including_exact": len(overlap_indices),
        "families_removed_for_any_near_overlap": len(removed_families),
        "rows_removed_with_overlap_families": len(candidate_rows) - len(kept),
        "candidate_rows_after_overlap_control": len(kept),
        "near_hamming_max": 6,
        "reference_rows": len(reference_rows),
    }


def build(
    parent: Path,
    multidogo: Path,
    ftc: Path,
    external_overlap_dir: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v23 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 20:
        raise ValueError("schema-v23 parent must be schema version 20")

    multidogo_manifest_path = multidogo / "manifest.json"
    multidogo_manifest = json.loads(multidogo_manifest_path.read_text(encoding="utf-8"))
    real_train = artifact_rows(
        multidogo_manifest, "real_train", multidogo / "multidogo_real_train.jsonl"
    )
    state_train = artifact_rows(
        multidogo_manifest, "state_train", multidogo / "multidogo_state_train.jsonl"
    )
    call_validation = artifact_rows(
        multidogo_manifest,
        "call_validation",
        multidogo / "multidogo_call_validation.jsonl",
    )
    state_validation = artifact_rows(
        multidogo_manifest,
        "state_validation",
        multidogo / "multidogo_state_validation.jsonl",
    )
    validate_state_rows(state_train, MULTIDOGO_STATE_SOURCE)
    validate_state_rows(state_validation, MULTIDOGO_STATE_SOURCE)
    state_train = shape_multidogo_state_context(state_train)
    state_validation = shape_multidogo_state_context(state_validation)
    multidogo_increment, action_calibration, calibration_states = partition_multidogo(
        real_train, state_train
    )

    ftc_manifest_path = ftc / "manifest.json"
    ftc_manifest = json.loads(ftc_manifest_path.read_text(encoding="utf-8"))
    ftc_train_path = ftc / "train.jsonl"
    ftc_validation_path = ftc / "validation.jsonl"
    if (
        ftc_manifest.get("source") != FTC_SOURCE
        or ftc_manifest.get("license") != FTC_LICENSE
        or ftc_manifest.get("external_transcript_text_copied") is not False
        or file_sha256(ftc_train_path) != ftc_manifest.get("train_sha256")
        or file_sha256(ftc_validation_path) != ftc_manifest.get("validation_sha256")
    ):
        raise ValueError("FTC pattern artifact differs from the schema-23 contract")
    ftc_train_source = read_jsonl(ftc_train_path)
    ftc_validation_source = read_jsonl(ftc_validation_path)
    parent_reference_rows = read_reference_rows(parent)
    external_reference_rows = read_reference_rows(external_overlap_dir)
    ftc_controlled, ftc_overlap_stats = remove_reference_overlap_families(
        ftc_train_source + ftc_validation_source,
        parent_reference_rows + external_reference_rows,
    )
    ftc_train = [row for row in ftc_controlled if row.get("split") == "train"]
    ftc_validation = [
        row for row in ftc_controlled if row.get("split") == "validation"
    ]
    validate_state_rows(ftc_train, FTC_SOURCE)
    validate_state_rows(ftc_validation, FTC_SOURCE)

    parent_train = read_jsonl(parent / "train.jsonl")
    increment = sorted(multidogo_increment + ftc_train, key=lambda row: str(row["id"]))
    all_fit_ids = [str(row["id"]) for row in parent_train + increment]
    if len(all_fit_ids) != len(set(all_fit_ids)):
        raise ValueError("schema-v23 fit rows have duplicate or parent-colliding IDs")
    held_ids = [
        str(row["id"])
        for row in action_calibration + call_validation + state_validation + ftc_validation
    ]
    if set(all_fit_ids) & set(held_ids) or len(held_ids) != len(set(held_ids)):
        raise ValueError("schema-v23 held row ID overlaps fitting or another held artifact")
    fit_families = {str(row.get("family_id")) for row in increment}
    held_families = {
        str(row.get("family_id"))
        for row in action_calibration + call_validation + state_validation + ftc_validation
    }
    if fit_families & held_families:
        raise ValueError("schema-v23 increment family crosses fitting and held artifacts")

    output.mkdir(parents=True)
    combined_train = parent_train + increment
    write_jsonl(output / "train.jsonl", combined_train)
    write_jsonl(output / "action_calibration.jsonl", action_calibration)
    write_jsonl(output / "multidogo_call_validation.jsonl", call_validation)
    write_jsonl(output / "multidogo_state_validation.jsonl", state_validation)
    write_jsonl(output / "ftc_pattern_validation.jsonl", ftc_validation)
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
            "action_calibration": len(action_calibration),
            "multidogo_call_validation": len(call_validation),
            "multidogo_state_validation": len(state_validation),
            "ftc_pattern_validation": len(ftc_validation),
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
            "schema23_increment": {
                "input_policy": "speaker-neutral-evidence-recent-v2",
                "multidogo_source": MULTIDOGO_SOURCE,
                "multidogo_state_source": MULTIDOGO_STATE_SOURCE,
                "multidogo_license": MULTIDOGO_LICENSE,
                "multidogo_source_manifest_sha256": file_sha256(multidogo_manifest_path),
                "multidogo_source_revision": multidogo_manifest["revision"],
                "multidogo_fit_rows": len(multidogo_increment),
                "multidogo_fit_state_rows": len(
                    [row for row in multidogo_increment if row["source"] == MULTIDOGO_STATE_SOURCE]
                ),
                "multidogo_fit_human_agent_turn_rows": len(
                    [row for row in multidogo_increment if row["source"] == MULTIDOGO_SOURCE]
                ),
                "multidogo_fit_families": len(
                    {str(row["family_id"]) for row in multidogo_increment}
                ),
                "multidogo_action_calibration_rows": len(action_calibration),
                "multidogo_action_calibration_state_rows": len(calibration_states),
                "multidogo_action_calibration_families": len(
                    {str(row["family_id"]) for row in action_calibration}
                ),
                "multidogo_calibration_families_per_domain": CALIBRATION_FAMILIES_PER_DOMAIN,
                "multidogo_train_state_domains": list(TRAIN_STATE_DOMAINS),
                "multidogo_validation_state_domains": list(VALIDATION_STATE_DOMAINS),
                "multidogo_real_verdict_weight": MULTIDOGO_REAL_VERDICT_WEIGHT,
                "multidogo_state_context_policy": (
                    "full_common_prefix_plus_decisive_action_plus_two_shared_delayed_turns"
                ),
                "multidogo_delayed_turns_after_action": (
                    MULTIDOGO_DELAYED_TURNS_AFTER_ACTION
                ),
                "ftc_pattern_source": FTC_SOURCE,
                "ftc_pattern_license": FTC_LICENSE,
                "ftc_pattern_manifest_sha256": file_sha256(ftc_manifest_path),
                "ftc_pattern_source_train_rows": len(ftc_train_source),
                "ftc_pattern_source_validation_rows": len(ftc_validation_source),
                "ftc_pattern_overlap_control": ftc_overlap_stats,
                "ftc_pattern_overlap_references": {
                    "parent_directory": str(parent),
                    "external_directory": str(external_overlap_dir),
                    "external_role": "BothBosu exact and SimHash contamination audit only",
                    "external_rows_used_for_fitting_or_threshold": 0,
                },
                "ftc_pattern_fit_rows": len(ftc_train),
                "ftc_pattern_fit_families": len(
                    {str(row["contrast_id"]) for row in ftc_train}
                ),
                "ftc_pattern_validation_rows": len(ftc_validation),
                "ftc_pattern_validation_families": len(
                    {str(row["contrast_id"]) for row in ftc_validation}
                ),
                "ftc_pattern_validation_scenarios": list(FTC_VALIDATION_PATTERNS),
                "ftc_external_transcript_text_copied": False,
                "action_target_keys": list(TARGET_KEYS),
                "apptek_rows_used_for_fitting": 0,
                "bothbosu_rows_used_for_fitting": 0,
                "reddit_rows_directly_scraped": 0,
                "teleantifraud_rows_used_for_fitting": 0,
                "bothbosu_ood_opened": True,
                "bothbosu_used_for_schema23_design": False,
                "sealed_ood_opened": False,
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
    parser.add_argument("--multidogo", type=Path, default=Path("data/external/multidogo"))
    parser.add_argument(
        "--ftc", type=Path, default=Path("data/generated/ftc_pattern_action_states_v1")
    )
    parser.add_argument(
        "--external-overlap-dir",
        type=Path,
        default=Path("data/external/scam_dialogue"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema23-evidence-compaction/processed"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.parent,
                args.multidogo,
                args.ftc,
                args.external_overlap_dir,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
