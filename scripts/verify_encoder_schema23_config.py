#!/usr/bin/env python3
"""Fail closed when the schema-23 evidence-compaction contract drifts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import EVIDENCE_RECENT_MAX_CHARS, prepare_model_text

try:
    from scripts.audit_source_overlap import read_reference_rows
    from scripts.build_schema19_call_windows import read_jsonl
    from scripts.build_schema23_evidence_compaction import (
        FTC_SOURCE,
        MULTIDOGO_SOURCE,
        MULTIDOGO_STATE_SOURCE,
        remove_reference_overlap_families,
    )
    from scripts.generate_call_action_states import CONTRAST_STATES
    from scripts.verify_encoder_pair_config import model_file
    from training.train_encoder import ACTION_TARGETS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from audit_source_overlap import read_reference_rows  # type: ignore[no-redef]
    from build_schema19_call_windows import read_jsonl  # type: ignore[no-redef]
    from build_schema23_evidence_compaction import (  # type: ignore[no-redef]
        FTC_SOURCE,
        MULTIDOGO_SOURCE,
        MULTIDOGO_STATE_SOURCE,
        remove_reference_overlap_families,
    )
    from generate_call_action_states import (  # type: ignore[no-redef]
        CONTRAST_STATES,
    )
    from generate_call_action_states import (
        TARGET_KEYS as ACTION_TARGETS,
    )
    from verify_encoder_pair_config import model_file  # type: ignore[no-redef]

CONFIG_PATH = Path(
    "configs/encoder-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right.json"
)


def contains_subsequence(values: list[object], subsequence: list[object]) -> bool:
    if not subsequence:
        return False
    width = len(subsequence)
    return any(
        values[index : index + width] == subsequence
        for index in range(len(values) - width + 1)
    )


def word_sequence(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def decisive_visibility_failures(
    rows: list[dict[str, object]], tokenizer: object, dialogue_policy: str
) -> list[str]:
    failures: list[str] = []
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("contrast_id"))].append(row)
    for contrast_id, family in grouped.items():
        if len(family) != len(CONTRAST_STATES):
            failures.append(f"incomplete state family: {contrast_id}")
            continue
        lines = [str(row["text"]).splitlines() for row in family]
        if len({len(item) for item in lines}) != 1:
            failures.append(f"state line counts differ: {contrast_id}")
            continue
        changing = [
            index
            for index in range(len(lines[0]))
            if len({item[index] for item in lines}) > 1
        ]
        if len(changing) != 1:
            failures.append(f"state family changes {len(changing)} lines: {contrast_id}")
            continue
        decisive_index = changing[0]
        for row, row_lines in zip(family, lines, strict=True):
            decisive = row_lines[decisive_index].partition(": ")[2]
            prepared = prepare_model_text(str(row["text"]), dialogue_policy)
            if not contains_subsequence(word_sequence(prepared), word_sequence(decisive)):
                failures.append(f"decisive turn omitted by compactor: {row['id']}")
                continue
            encoded = tokenizer(  # type: ignore[operator]
                prepared,
                truncation=True,
                max_length=256,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]
            visible = tokenizer.decode(encoded, skip_special_tokens=True)  # type: ignore[operator]
            if not contains_subsequence(word_sequence(visible), word_sequence(decisive)):
                failures.append(f"decisive turn is outside compacted window: {row['id']}")
            if (
                prepared.startswith(("EVIDENCE:", "RECENT:"))
                and len(prepared) > EVIDENCE_RECENT_MAX_CHARS
            ):
                failures.append(f"compacted dialogue exceeds character bound: {row['id']}")
    return failures


def verify(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    data_dir = Path(data["directory"])
    teacher = config["teacher"]
    initialization = config["initialization"]
    expected_hashes = {
        data_dir / "manifest.json": data["manifest_sha256"],
        data_dir / "train.jsonl": data["train_sha256"],
        data_dir / "dev.jsonl": data["dev_sha256"],
        data_dir / "test.jsonl": data["test_sha256"],
        data_dir / "call_state_validation.jsonl": data["call_state_validation_sha256"],
        data_dir / "call_window_validation.jsonl": data["call_window_validation_sha256"],
        data_dir / "action_calibration.jsonl": data["action_calibration_sha256"],
        data_dir / "multidogo_call_validation.jsonl": data[
            "multidogo_call_validation_sha256"
        ],
        data_dir / "multidogo_state_validation.jsonl": data[
            "multidogo_state_validation_sha256"
        ],
        data_dir / "ftc_pattern_validation.jsonl": data[
            "ftc_pattern_validation_sha256"
        ],
        Path(data["multidogo_source_manifest"]): data[
            "multidogo_source_manifest_sha256"
        ],
        Path(data["ftc_pattern_source_manifest"]): data[
            "ftc_pattern_source_manifest_sha256"
        ],
        Path(teacher["ledger"]): teacher["ledger_sha256"],
        Path(teacher["manifest"]): teacher["manifest_sha256"],
        model_file(Path(initialization["checkpoint"])): initialization["model_sha256"],
    }
    failures: list[str] = []
    for path, expected in expected_hashes.items():
        if not path.is_file():
            failures.append(f"missing frozen artifact: {path}")
        elif file_sha256(path) != expected:
            failures.append(f"hash drift: {path}")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train = read_jsonl(data_dir / "train.jsonl")
    call_states = read_jsonl(data_dir / "call_state_validation.jsonl")
    call_windows = read_jsonl(data_dir / "call_window_validation.jsonl")
    action_calibration = read_jsonl(data_dir / "action_calibration.jsonl")
    multidogo_calls = read_jsonl(data_dir / "multidogo_call_validation.jsonl")
    multidogo_states = read_jsonl(data_dir / "multidogo_state_validation.jsonl")
    ftc_validation = read_jsonl(data_dir / "ftc_pattern_validation.jsonl")
    if manifest.get("schema_version") != 23 or len(train) != data["train_rows"]:
        failures.append("schema or training count differs")
    if len({str(row["id"]) for row in train}) != len(train):
        failures.append("training IDs are not unique")

    licensed = [row for row in train if not bool(row.get("is_synthetic"))]
    roleplay = [
        row
        for row in train
        if row.get("source") in {"taskmaster1_woz_dialogues", MULTIDOGO_SOURCE}
    ]
    action_rows = [row for row in train if isinstance(row.get("action_targets"), dict)]
    if len(licensed) != data["licensed_source_train_rows"]:
        failures.append("licensed-source count differs")
    if len(train) - len(licensed) != data["synthetic_train_rows"]:
        failures.append("synthetic count differs")
    if len(roleplay) != data["human_authored_or_spoken_roleplay_train_rows"]:
        failures.append("human roleplay count differs")
    if len(action_rows) != data["action_supervised_train_rows"]:
        failures.append("action-supervised count differs")
    target_counts = {
        name: sum(bool(row["action_targets"][name]) for row in action_rows)
        for name in ACTION_TARGETS
    }
    if target_counts != config["training"]["action_target_positive_counts"]:
        failures.append("action-target positive counts differ")
    for row in action_rows + action_calibration + multidogo_states + ftc_validation:
        if tuple(row.get("action_targets", {})) != ACTION_TARGETS:
            failures.append(f"action target schema differs: {row.get('id')}")

    multidogo_fit = [
        row for row in train if row.get("source") in {MULTIDOGO_SOURCE, MULTIDOGO_STATE_SOURCE}
    ]
    ftc_fit = [row for row in train if row.get("source") == FTC_SOURCE]
    if len(multidogo_fit) != data["multidogo_fit_rows"]:
        failures.append("MultiDoGO fit count differs")
    if len(ftc_fit) != data["ftc_pattern_fit_rows"]:
        failures.append("FTC fit count differs")
    if len(action_calibration) != data["multidogo_action_calibration_rows"]:
        failures.append("action calibration count differs")
    if len(ftc_validation) != data["ftc_pattern_validation_rows"]:
        failures.append("FTC validation count differs")
    if len(call_windows) != 447 or {str(row["label"]) for row in call_windows} != {"SAFE"}:
        failures.append("long-call SAFE validation contract differs")

    fit_families = {str(row.get("family_id")) for row in multidogo_fit}
    calibration_families = {str(row.get("family_id")) for row in action_calibration}
    held_families = {str(row.get("family_id")) for row in multidogo_calls + multidogo_states}
    if (
        fit_families & calibration_families
        or fit_families & held_families
        or calibration_families & held_families
    ):
        failures.append("MultiDoGO family crosses fit, action calibration, or held validation")
    if len(calibration_families) != data["multidogo_action_calibration_families"]:
        failures.append("action calibration family count differs")
    if len({str(row.get("family_id")) for row in ftc_fit}) != data["ftc_pattern_fit_families"]:
        failures.append("FTC fit family count differs")
    if len({str(row.get("family_id")) for row in ftc_validation}) != data[
        "ftc_pattern_validation_families"
    ]:
        failures.append("FTC validation family count differs")
    if {str(row.get("scenario")) for row in ftc_validation} != set(
        data["ftc_pattern_validation_scenarios"]
    ):
        failures.append("FTC validation scenarios differ")
    if any(
        row.get("external_transcript_text_copied") is not False
        or row.get("external_benchmark_text_copied") is not False
        for row in ftc_fit + ftc_validation
    ):
        failures.append("FTC row does not explicitly prohibit copied external text")

    references = read_reference_rows(Path("data/experiments/schema20-action-states/processed"))
    references.extend(read_reference_rows(Path("data/external/scam_dialogue")))
    overlap_kept, overlap_stats = remove_reference_overlap_families(
        ftc_fit + ftc_validation, references
    )
    if len(overlap_kept) != len(ftc_fit) + len(ftc_validation):
        failures.append("admitted FTC family still overlaps parent or BothBosu references")

    teacher_manifest = json.loads(Path(teacher["manifest"]).read_text(encoding="utf-8"))
    teacher_records = read_jsonl(Path(teacher["ledger"]))
    parent_ids = {
        str(row["id"])
        for row in read_jsonl(Path("data/experiments/schema20-action-states/processed/train.jsonl"))
    }
    if (
        teacher_manifest.get("contains_text") is not False
        or teacher_manifest.get("logit_scope") != "first three verdict logits only"
        or teacher_manifest.get("dialogue_policy") != config["training"]["dialogue_policy"]
        or len(teacher_records) != teacher["anchor_rows"]
        or {str(row["id"]) for row in teacher_records} != parent_ids
        or any(
            set(row) != {"id", "logits"}
            or not isinstance(row["logits"], list)
            or len(row["logits"]) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in row["logits"]
            )
            for row in teacher_records
        )
    ):
        failures.append("teacher retention ledger contract differs")

    model = AutoModelForSequenceClassification.from_pretrained(
        initialization["checkpoint"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(initialization["checkpoint"], local_files_only=True)
    saved_targets = tuple(getattr(model.config, "scamguard_action_targets", ()))
    if saved_targets != ACTION_TARGETS or int(model.config.num_labels) != 10:
        failures.append("schema-20 initialization action-head contract differs")
    tokenizer.truncation_side = config["training"]["truncation_side"]
    visibility_rows = (
        call_states
        + [row for row in multidogo_fit if row.get("source") == MULTIDOGO_STATE_SOURCE]
        + multidogo_states
        + ftc_fit
        + ftc_validation
    )
    failures.extend(
        decisive_visibility_failures(
            visibility_rows, tokenizer, config["training"]["dialogue_policy"]
        )
    )

    expected_recipe = {
        "epochs": 1.0,
        "batch_size": 16,
        "optimizer_steps": math.ceil(len(train) / 16),
        "learning_rate": 2e-6,
        "max_length": 256,
        "truncation_side": "right",
        "dialogue_policy": "speaker-neutral-evidence-recent-v2",
        "retention_weight": 4.0,
        "retention_temperature": 2.0,
        "action_loss_weight": 0.5,
        "default_action_verdict_weight": 0.25,
        "seed": 20260821,
    }
    for key, value in expected_recipe.items():
        if config["training"].get(key) != value:
            failures.append(f"frozen recipe differs: {key}")
    if config.get("predeclared_before_training") is not True:
        failures.append("experiment is not marked predeclared")

    return {
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "passed": not failures,
        "failures": failures,
        "counts": {
            "train": len(train),
            "licensed": len(licensed),
            "synthetic": len(train) - len(licensed),
            "action_supervised": len(action_rows),
            "action_calibration": len(action_calibration),
            "ftc_fit": len(ftc_fit),
            "ftc_validation": len(ftc_validation),
        },
        "ftc_admitted_overlap_recheck": overlap_stats,
        "decisive_visibility_rows": len(visibility_rows),
        "teacher_rows": len(teacher_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    result = verify(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
