#!/usr/bin/env python3
"""Fail closed when the schema-21 human-call experiment contract drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import prepare_model_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_harper_valley_calls import (  # noqa: E402
    HOLDOUT_TASKS,
    LICENSE,
    SOURCE,
    STATE_SOURCE,
)
from scripts.generate_call_action_states import CONTRAST_STATES  # noqa: E402
from scripts.verify_encoder_pair_config import model_file, read_jsonl  # noqa: E402
from scripts.verify_encoder_schema20_config import state_failures  # noqa: E402
from training.train_encoder import (  # noqa: E402
    ACTION_TARGETS,
    expand_classifier_for_action_targets,
)


def token_summary(
    rows: list[dict[str, object]], tokenizer: object, dialogue_policy: str
) -> dict[str, int]:
    encoded = tokenizer(  # type: ignore[operator]
        [prepare_model_text(str(row["text"]), dialogue_policy) for row in rows],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    lengths = np.array([len(values) for values in encoded["input_ids"]])
    return {
        "token_min": int(np.min(lengths)),
        "token_p50": int(np.percentile(lengths, 50)),
        "token_p95": int(np.percentile(lengths, 95)),
        "token_max": int(np.max(lengths)),
    }


def validate_harper_families(
    real_rows: list[dict[str, object]],
    state_rows: list[dict[str, object]],
    split: str,
) -> list[str]:
    failures: list[str] = []
    real_families = {str(row.get("family_id")) for row in real_rows}
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in state_rows:
        grouped[str(row.get("family_id"))].append(row)
    if real_families != set(grouped):
        failures.append(f"{split} Harper real/state family sets differ")
    for row in real_rows:
        targets = row.get("action_targets")
        if (
            row.get("source") != SOURCE
            or row.get("license") != LICENSE
            or row.get("label") != "SAFE"
            or row.get("is_synthetic") is not False
            or row.get("action_verdict_weight") != 1.0
            or not isinstance(targets, dict)
            or tuple(targets) != ACTION_TARGETS
        ):
            failures.append(f"invalid {split} Harper real row: {row.get('id')}")
    for family_id, contrast in grouped.items():
        if len(contrast) != 4 or {
            str(row.get("contrast_state")) for row in contrast
        } != set(CONTRAST_STATES):
            failures.append(f"incomplete {split} Harper state family: {family_id}")
            continue
        contexts = [str(row.get("text", "")).rsplit("\nAGENT:", 1)[0] for row in contrast]
        context_hash = hashlib.sha256(contexts[0].encode()).hexdigest()
        if len(set(contexts)) != 1 or {
            str(row.get("shared_context_sha256")) for row in contrast
        } != {context_hash}:
            failures.append(f"invalid {split} Harper shared context: {family_id}")
        for row in contrast:
            targets = row.get("action_targets")
            if (
                row.get("source") != STATE_SOURCE
                or row.get("license") != LICENSE
                or row.get("is_synthetic") is not True
                or row.get("human_grounded") is not True
                or not isinstance(targets, dict)
                or tuple(targets) != ACTION_TARGETS
            ):
                failures.append(f"invalid {split} Harper state row: {row.get('id')}")
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
        data_dir / "call_state_validation.jsonl": data[
            "call_state_validation_sha256"
        ],
        data_dir / "call_window_validation.jsonl": data[
            "call_window_validation_sha256"
        ],
        data_dir / "harper_call_validation.jsonl": data[
            "harper_call_validation_sha256"
        ],
        data_dir / "harper_state_validation.jsonl": data[
            "harper_state_validation_sha256"
        ],
        Path(data["harper_source_manifest"]): data["harper_source_manifest_sha256"],
        Path(teacher["ledger"]): teacher["ledger_sha256"],
        Path(teacher["manifest"]): teacher["manifest_sha256"],
        model_file(Path(initialization["checkpoint"])): initialization["model_sha256"],
    }
    failures: list[str] = []
    for path, expected in expected_hashes.items():
        if not path.is_file():
            failures.append(f"missing frozen artifact: {path}")
        elif file_sha256(path) != expected:
            failures.append(f"{path}: expected {expected}, found {file_sha256(path)}")

    train_rows = read_jsonl(data_dir / "train.jsonl")
    long_state_validation = read_jsonl(data_dir / "call_state_validation.jsonl")
    window_validation = read_jsonl(data_dir / "call_window_validation.jsonl")
    harper_call_validation = read_jsonl(data_dir / "harper_call_validation.jsonl")
    harper_state_validation = read_jsonl(data_dir / "harper_state_validation.jsonl")
    if len(train_rows) != data["train_rows"]:
        failures.append("training row count differs from config")
    if len(window_validation) != 447 or {
        str(row.get("label")) for row in window_validation
    } != {"SAFE"}:
        failures.append("preserved call-window validation contract differs")
    licensed_rows = [row for row in train_rows if not bool(row.get("is_synthetic"))]
    if len(licensed_rows) != data["licensed_source_train_rows"]:
        failures.append("licensed-source training count differs from config")
    if len(train_rows) - len(licensed_rows) != data["synthetic_train_rows"]:
        failures.append("synthetic training count differs from config")

    long_state_train = [
        row
        for row in train_rows
        if row.get("source") == "scamguard_synthetic_long_call_action_states_v1"
    ]
    failures.extend(
        state_failures(
            long_state_train,
            data["long_state_train_rows"],
            data["long_state_train_families"],
            {"SAFE": 3072, "UNCERTAIN": 1536, "SCAM": 1536},
            "train",
        )
    )
    failures.extend(
        state_failures(
            long_state_validation,
            2048,
            512,
            {"SAFE": 1024, "UNCERTAIN": 512, "SCAM": 512},
            "validation",
        )
    )

    harper_real_train = [row for row in train_rows if row.get("source") == SOURCE]
    harper_state_train = [row for row in train_rows if row.get("source") == STATE_SOURCE]
    expected_counts = {
        "harper_real_train_rows": len(harper_real_train),
        "harper_real_train_families": len(
            {str(row.get("family_id")) for row in harper_real_train}
        ),
        "harper_state_train_rows": len(harper_state_train),
        "harper_state_train_families": len(
            {str(row.get("family_id")) for row in harper_state_train}
        ),
        "harper_call_validation_rows": len(harper_call_validation),
        "harper_state_validation_rows": len(harper_state_validation),
        "harper_validation_families": len(
            {str(row.get("family_id")) for row in harper_call_validation}
        ),
    }
    for key, actual in expected_counts.items():
        if data.get(key) != actual:
            failures.append(f"{key}: expected {data.get(key)}, found {actual}")
    failures.extend(validate_harper_families(harper_real_train, harper_state_train, "train"))
    failures.extend(
        validate_harper_families(
            harper_call_validation, harper_state_validation, "validation"
        )
    )
    train_tasks = {str(row.get("source_task")) for row in harper_real_train}
    validation_tasks = {str(row.get("source_task")) for row in harper_call_validation}
    if train_tasks != set(data["harper_train_tasks"]):
        failures.append("Harper training tasks differ from config")
    if validation_tasks != set(data["harper_holdout_tasks"]) or validation_tasks != HOLDOUT_TASKS:
        failures.append("Harper holdout tasks differ from frozen task-disjoint split")
    if train_tasks & validation_tasks:
        failures.append("Harper task appears in both train and validation")

    action_rows = [row for row in train_rows if isinstance(row.get("action_targets"), dict)]
    if len(action_rows) != data["action_supervised_train_rows"]:
        failures.append("action-supervised row count differs from config")
    action_positive_counts = {
        name: sum(int(bool(row["action_targets"][name])) for row in action_rows)
        for name in ACTION_TARGETS
    }
    if action_positive_counts != config["training"]["action_target_positive_counts"]:
        failures.append("action-target positive counts differ from config")

    tokenizer = AutoTokenizer.from_pretrained(
        Path(initialization["checkpoint"]), local_files_only=True
    )
    policy = config["training"]["dialogue_policy"]
    for prefix, split_rows in (
        ("harper_real_train", harper_real_train),
        ("harper_state_train", harper_state_train),
        ("harper_call_validation", harper_call_validation),
        ("harper_state_validation", harper_state_validation),
    ):
        for suffix, actual in token_summary(split_rows, tokenizer, policy).items():
            key = f"{prefix}_{suffix}"
            if data.get(key) != actual:
                failures.append(f"{key}: expected {data.get(key)}, found {actual}")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    increment = manifest.get("schema21_increment", {})
    harper_source_manifest = json.loads(
        Path(data["harper_source_manifest"]).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != data["schema_version"]:
        failures.append("processed schema version differs from config")
    if increment.get("source_manifest_sha256") != data["harper_source_manifest_sha256"]:
        failures.append("Harper source manifest hash differs from processed manifest")
    if increment.get("source_revision") != data["harper_source_revision"]:
        failures.append("Harper source revision differs from config")
    if (
        harper_source_manifest.get("transcript_tree_sha256")
        != data["harper_transcript_tree_sha256"]
        or harper_source_manifest.get("metadata_tree_sha256")
        != data["harper_metadata_tree_sha256"]
    ):
        failures.append("Harper transcript or metadata tree differs from config")
    for flag in (
        "apptek_ood_opened",
        "bothbosu_ood_opened",
        "moz_holdout_opened",
        "youtube_ood_opened",
    ):
        if increment.get(flag) is not False:
            failures.append(f"{flag} is not recorded as sealed")

    teacher_rows = read_jsonl(Path(teacher["ledger"]))
    teacher_ids = {str(row.get("id")) for row in teacher_rows}
    train_ids = {str(row.get("id")) for row in train_rows}
    if len(teacher_rows) != teacher["anchor_rows"] or len(teacher_ids) != teacher["anchor_rows"]:
        failures.append("teacher ledger count differs or contains duplicate IDs")
    if not teacher_ids <= train_ids:
        failures.append("teacher ledger contains IDs absent from training")
    if len(train_ids - teacher_ids) != teacher["unanchored_rows"]:
        failures.append("unanchored row count differs from config")
    teacher_manifest = json.loads(Path(teacher["manifest"]).read_text(encoding="utf-8"))
    if teacher_manifest.get("contains_text") is not False or teacher["contains_text"] is not False:
        failures.append("teacher cache is not explicitly text-free")
    if teacher_manifest.get("checkpoint_model_sha256") != initialization["model_sha256"]:
        failures.append("teacher checkpoint differs from initialization checkpoint")

    frozen_training = {
        "epochs": 1.0,
        "batch_size": 16,
        "gradient_accumulation": 1,
        "optimizer_steps": 1662,
        "learning_rate": 0.000005,
        "max_length": 256,
        "truncation_side": "left",
        "dialogue_policy": "speaker-neutral-v1",
        "binary_loss_weight": 1.0,
        "retention_weight": 4.0,
        "retention_temperature": 2.0,
        "action_loss_weight": 0.5,
        "default_action_verdict_weight": 0.25,
        "real_harper_action_verdict_weight": 1.0,
        "action_target_names": list(ACTION_TARGETS),
        "action_target_positive_counts": action_positive_counts,
        "action_positive_weight_policy": "square root of negative count divided by positive count",
        "pair_loss_weight": 0.0,
        "source_balance_alpha": 0.0,
        "seed": 20260820,
        "checkpoint_selection": "development recall at the frozen 2-percent FPR cap",
        "primary_alert_score": (
            "calibrated probability from the preserved three verdict logits; action logits "
            "remain auxiliary diagnostics in this experiment"
        ),
    }
    if config.get("training") != frozen_training:
        failures.append("training recipe differs from the frozen schema-21 contract")
    expected_steps = math.ceil(len(train_rows) / frozen_training["batch_size"])
    if expected_steps != frozen_training["optimizer_steps"]:
        failures.append("optimizer-step count differs from the training size")

    model = AutoModelForSequenceClassification.from_pretrained(
        Path(initialization["checkpoint"]), local_files_only=True
    )
    original_weight = model.classifier.weight.detach().clone()
    original_bias = model.classifier.bias.detach().clone()
    expand_classifier_for_action_targets(model, ACTION_TARGETS, frozen_training["seed"])
    if model.classifier.out_features != 3 + len(ACTION_TARGETS):
        failures.append("expanded classifier output count differs")
    if not torch.equal(model.classifier.weight[:3], original_weight):
        failures.append("classifier expansion changed verdict weights before training")
    if not torch.equal(model.classifier.bias[:3], original_bias):
        failures.append("classifier expansion changed verdict bias before training")

    quality = config.get("quality_acceptance", {})
    if quality.get("harper_call_validation_fpr_max") != 0.02:
        failures.append("Harper real-call validation does not enforce the 2-percent FPR cap")
    if quality.get("bothbosu_latest_window_fpr_max") != 0.02:
        failures.append("BothBosu does not enforce the frozen 2-percent FPR cap")
    if failures:
        raise SystemExit("schema-21 experiment preflight failed:\n" + "\n".join(failures))

    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "expected_optimizer_steps": expected_steps,
        "teacher_anchor_rows": len(teacher_rows),
        "unanchored_rows": len(train_ids - teacher_ids),
        "harper_train_families": len({str(row["family_id"]) for row in harper_real_train}),
        "harper_validation_families": len(
            {str(row["family_id"]) for row in harper_call_validation}
        ),
        "classifier_outputs": model.classifier.out_features,
        "verdict_head_preserved_before_training": True,
        "sealed_artifacts_opened": False,
        "status": "preflight_passed",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/encoder-schema21-human-calls-actionheads-ret4-aw05-vw025-left.json"
        ),
    )
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
