#!/usr/bin/env python3
"""Fail closed when the schema-20 action-head experiment contract drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import prepare_model_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.train_encoder import (  # noqa: E402
    ACTION_TARGETS,
    expand_classifier_for_action_targets,
)

try:
    from scripts.build_schema20_action_states import LONG_STATE_SOURCE
    from scripts.generate_call_action_states import CONTRAST_STATES
    from scripts.verify_encoder_pair_config import model_file, read_jsonl
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_schema20_action_states import LONG_STATE_SOURCE  # type: ignore[no-redef]
    from generate_call_action_states import CONTRAST_STATES  # type: ignore[no-redef]
    from verify_encoder_pair_config import model_file, read_jsonl  # type: ignore[no-redef]


def token_lengths(
    rows: list[dict[str, object]], tokenizer: object, dialogue_policy: str
) -> list[int]:
    encoded = tokenizer(  # type: ignore[operator]
        [prepare_model_text(str(row["text"]), dialogue_policy) for row in rows],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    return [len(values) for values in encoded["input_ids"]]


def state_failures(
    rows: list[dict[str, object]],
    expected_rows: int,
    expected_families: int,
    expected_labels: dict[str, int],
    split: str,
) -> list[str]:
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"{split} state rows: expected {expected_rows}, found {len(rows)}")
    if Counter(str(row.get("label")) for row in rows) != Counter(expected_labels):
        failures.append(f"{split} state labels differ from config")
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("source") != LONG_STATE_SOURCE:
            failures.append(f"{split} state row has unexpected source: {row.get('id')}")
        if (
            row.get("context_window_curriculum")
            != "long_shared_history_before_four_way_final_action_state"
        ):
            failures.append(f"{split} state row lacks long-window contract: {row.get('id')}")
        targets = row.get("action_targets")
        if (
            not isinstance(targets, dict)
            or tuple(targets) != ACTION_TARGETS
            or not all(isinstance(value, bool) for value in targets.values())
        ):
            failures.append(f"{split} state row has invalid action targets: {row.get('id')}")
        grouped[str(row.get("contrast_id", ""))].append(row)
    if "" in grouped:
        failures.append(f"{split} contains an empty contrast ID")
    if len(grouped) != expected_families:
        failures.append(
            f"{split} state families: expected {expected_families}, found {len(grouped)}"
        )
    for contrast_id, contrast in grouped.items():
        if len(contrast) != 4 or {
            str(row.get("contrast_state")) for row in contrast
        } != set(CONTRAST_STATES):
            failures.append(f"{split} contrast is not a complete four-state family: {contrast_id}")
            continue
        contexts = [str(row.get("text", "")).rsplit("\nAGENT:", 1)[0] for row in contrast]
        if len(set(contexts)) != 1:
            failures.append(f"{split} contrast has different preceding turns: {contrast_id}")
            continue
        context_hash = hashlib.sha256(contexts[0].encode("utf-8")).hexdigest()
        if {str(row.get("shared_context_sha256")) for row in contrast} != {context_hash}:
            failures.append(f"{split} contrast has an invalid context hash: {contrast_id}")
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
        Path(data["state_source_data"]): data["state_source_data_sha256"],
        Path(data["state_source_manifest"]): data["state_source_manifest_sha256"],
        Path(teacher["ledger"]): teacher["ledger_sha256"],
        Path(teacher["manifest"]): teacher["manifest_sha256"],
        model_file(Path(initialization["checkpoint"])): initialization["model_sha256"],
    }
    failures: list[str] = []
    for path, expected in expected_hashes.items():
        if not path.is_file():
            failures.append(f"missing frozen artifact: {path}")
        elif file_sha256(path) != expected:
            failures.append(
                f"{path}: expected {expected}, found {file_sha256(path)}"
            )

    train_rows = read_jsonl(data_dir / "train.jsonl")
    state_validation = read_jsonl(data_dir / "call_state_validation.jsonl")
    window_validation = read_jsonl(data_dir / "call_window_validation.jsonl")
    if len(train_rows) != data["train_rows"]:
        failures.append(f"train rows: expected {data['train_rows']}, found {len(train_rows)}")
    if len(window_validation) != data["call_window_validation_rows"]:
        failures.append("call-window validation row count differs from config")
    if Counter(str(row.get("label")) for row in window_validation) != {
        "SAFE": len(window_validation)
    }:
        failures.append("call-window validation is not SAFE-only")
    licensed_rows = sum(not bool(row.get("is_synthetic")) for row in train_rows)
    if licensed_rows != data["licensed_source_train_rows"]:
        failures.append("licensed-source training count differs from config")
    if len(train_rows) - licensed_rows != data["synthetic_train_rows"]:
        failures.append("synthetic training count differs from config")

    state_train = [row for row in train_rows if row.get("source") == LONG_STATE_SOURCE]
    failures.extend(
        state_failures(
            state_train,
            data["state_train_rows"],
            data["state_train_families"],
            data["state_train_labels"],
            "train",
        )
    )
    failures.extend(
        state_failures(
            state_validation,
            data["state_validation_rows"],
            data["state_validation_families"],
            data["state_validation_labels"],
            "validation",
        )
    )
    holdouts = set(data["holdout_scenarios"])
    if {str(row.get("scenario")) for row in state_validation} != holdouts:
        failures.append("state validation scenarios differ from config")
    if {str(row.get("scenario")) for row in state_train} & holdouts:
        failures.append("state holdout scenarios appear in training")

    action_positive_counts = {
        name: sum(int(bool(row["action_targets"][name])) for row in state_train)
        for name in ACTION_TARGETS
    }
    if action_positive_counts != config["training"]["action_target_positive_counts"]:
        failures.append("action-target positive counts differ from config")

    tokenizer = AutoTokenizer.from_pretrained(
        Path(initialization["checkpoint"]), local_files_only=True
    )
    policy = config["training"]["dialogue_policy"]
    train_lengths = token_lengths(state_train, tokenizer, policy)
    validation_lengths = token_lengths(state_validation, tokenizer, policy)
    window_lengths = token_lengths(window_validation, tokenizer, policy)
    length_expectations = {
        "state_train_token_min": int(np.min(train_lengths)),
        "state_train_token_p50": int(np.percentile(train_lengths, 50)),
        "state_train_token_p95": int(np.percentile(train_lengths, 95)),
        "state_train_token_max": int(np.max(train_lengths)),
        "state_validation_token_min": int(np.min(validation_lengths)),
        "state_validation_token_p50": int(np.percentile(validation_lengths, 50)),
        "state_validation_token_p95": int(np.percentile(validation_lengths, 95)),
        "state_validation_token_max": int(np.max(validation_lengths)),
        "call_window_validation_token_p50": int(np.percentile(window_lengths, 50)),
    }
    for key, actual in length_expectations.items():
        if data.get(key) != actual:
            failures.append(f"{key}: expected {data.get(key)}, found {actual}")
    if any(length <= config["training"]["max_length"] for length in train_lengths):
        failures.append("not every state training row exercises latest-window truncation")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    increment = manifest.get("schema20_increment", {})
    if manifest.get("schema_version") != data["schema_version"]:
        failures.append("processed schema version differs from config")
    if increment.get("state_train_families") != data["state_train_families"]:
        failures.append("manifest train state-family count differs from config")
    if increment.get("state_validation_families") != data["state_validation_families"]:
        failures.append("manifest validation state-family count differs from config")
    for flag in (
        "apptek_ood_opened",
        "bothbosu_ood_opened",
        "moz_holdout_opened",
        "youtube_ood_opened",
    ):
        if increment.get(flag) is not False:
            failures.append(f"{flag} is not recorded as sealed")
    if increment.get("apptek_rows_used_for_fitting") != 0:
        failures.append("AppTek rows were used for fitting")
    if increment.get("bothbosu_rows_used_for_fitting") != 0:
        failures.append("BothBosu rows were used for fitting")

    teacher_rows = read_jsonl(Path(teacher["ledger"]))
    teacher_ids = {str(row.get("id")) for row in teacher_rows}
    train_ids = {str(row.get("id")) for row in train_rows}
    if len(teacher_rows) != teacher["anchor_rows"] or len(teacher_ids) != teacher[
        "anchor_rows"
    ]:
        failures.append("teacher ledger count differs or contains duplicate IDs")
    if not teacher_ids <= train_ids:
        failures.append("teacher ledger contains IDs absent from training")
    if len(train_ids - teacher_ids) != teacher["unanchored_rows"]:
        failures.append("unanchored row count differs from config")
    if any(set(row) != {"id", "logits"} for row in teacher_rows):
        failures.append("teacher ledger is not limited to IDs and logits")
    teacher_manifest = json.loads(Path(teacher["manifest"]).read_text(encoding="utf-8"))
    if teacher_manifest.get("contains_text") is not False or teacher["contains_text"] is not False:
        failures.append("teacher cache is not explicitly text-free")
    if teacher_manifest.get("checkpoint_model_sha256") != initialization["model_sha256"]:
        failures.append("teacher checkpoint differs from initialization checkpoint")

    frozen_training = {
        "epochs": 1.0,
        "batch_size": 16,
        "gradient_accumulation": 1,
        "optimizer_steps": 1328,
        "learning_rate": 0.000005,
        "max_length": 256,
        "truncation_side": "left",
        "dialogue_policy": "speaker-neutral-v1",
        "binary_loss_weight": 1.0,
        "retention_weight": 4.0,
        "retention_temperature": 2.0,
        "action_loss_weight": 0.5,
        "action_verdict_weight": 0.25,
        "action_target_names": list(ACTION_TARGETS),
        "action_target_positive_counts": action_positive_counts,
        "action_positive_weight_policy": (
            "square root of negative count divided by positive count"
        ),
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
        failures.append("training recipe differs from the frozen schema-20 contract")
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
    if quality.get("state_verified_safe_fpr_max") != 0.02:
        failures.append("verified SAFE state does not enforce the 2-percent FPR cap")
    if quality.get("action_target_macro_auc_min") != 0.97:
        failures.append("action targets do not enforce the frozen macro-AUC gate")
    if failures:
        raise SystemExit("schema-20 experiment preflight failed:\n" + "\n".join(failures))

    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "expected_optimizer_steps": expected_steps,
        "teacher_anchor_rows": len(teacher_rows),
        "unanchored_rows": len(train_ids - teacher_ids),
        "state_train_families": data["state_train_families"],
        "state_validation_families": data["state_validation_families"],
        "call_window_validation_rows": len(window_validation),
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
        default=Path("configs/encoder-schema20-actionheads-ret4-aw05-vw025-left.json"),
    )
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
