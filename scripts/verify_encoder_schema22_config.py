#!/usr/bin/env python3
"""Fail closed when the schema-22 service-evidence experiment contract drifts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import prepare_model_text

try:
    from scripts.build_multidogo_dialogues import (
        LICENSE,
        SOURCE,
        STATE_SOURCE,
        TRAIN_STATE_DOMAINS,
        VALIDATION_STATE_DOMAINS,
    )
    from scripts.fetch_multidogo import LICENSE_SHA256, REVISION
    from scripts.generate_call_action_states import CONTRAST_STATES
    from scripts.verify_encoder_pair_config import model_file, read_jsonl
    from scripts.verify_encoder_schema20_config import state_failures
    from scripts.verify_encoder_schema21_config import token_summary
    from training.train_encoder import ACTION_TARGETS, expand_classifier_for_action_targets
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        LICENSE,
        SOURCE,
        STATE_SOURCE,
        TRAIN_STATE_DOMAINS,
        VALIDATION_STATE_DOMAINS,
    )
    from fetch_multidogo import LICENSE_SHA256, REVISION  # type: ignore[no-redef]
    from generate_call_action_states import CONTRAST_STATES  # type: ignore[no-redef]
    from verify_encoder_pair_config import model_file, read_jsonl  # type: ignore[no-redef]
    from verify_encoder_schema20_config import state_failures  # type: ignore[no-redef]
    from verify_encoder_schema21_config import token_summary  # type: ignore[no-redef]

    from training.train_encoder import (  # type: ignore[no-redef]
        ACTION_TARGETS,
        expand_classifier_for_action_targets,
    )


CONFIG_PATH = Path(
    "configs/encoder-schema22-service-evidence-actionheads-ret4-aw05-vw025-left.json"
)


def decisive_window_failures(
    rows: list[dict[str, object]], tokenizer: object, dialogue_policy: str
) -> list[str]:
    """Prove the only state-changing line survives the deployed left window."""

    failures: list[str] = []
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family_id"))].append(row)
    for family_id, family_rows in grouped.items():
        source_lines = [str(row["text"]).splitlines() for row in family_rows]
        if len({len(lines) for lines in source_lines}) != 1:
            failures.append(f"state variants have different line counts: {family_id}")
            continue
        varying = [
            index
            for index in range(len(source_lines[0]))
            if len({lines[index] for lines in source_lines}) > 1
        ]
        if len(varying) != 1:
            failures.append(f"state family does not change exactly one line: {family_id}")
            continue
        decisive_index = varying[0]
        for row, lines in zip(family_rows, source_lines, strict=True):
            prepared = prepare_model_text(str(row["text"]), dialogue_policy)
            encoded = tokenizer(  # type: ignore[operator]
                prepared,
                truncation=True,
                max_length=256,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]
            decisive_text = lines[decisive_index].partition(": ")[2]
            visible_text = tokenizer.decode(  # type: ignore[operator]
                encoded, skip_special_tokens=True
            )
            if decisive_text.casefold() not in visible_text.casefold():
                failures.append(
                    f"decisive state line falls outside 256-token left window: {row.get('id')}"
                )
    return failures


def multidogo_failures(
    real_rows: list[dict[str, object]],
    state_rows: list[dict[str, object]],
    split: str,
) -> list[str]:
    failures: list[str] = []
    real_by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    state_by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in real_rows:
        real_by_family[str(row.get("family_id"))].append(row)
    for row in state_rows:
        state_by_family[str(row.get("family_id"))].append(row)

    for family_id, family_rows in real_by_family.items():
        if len(family_rows) != 2 or {
            str(row.get("source_window")) for row in family_rows
        } != {"recent_complete_turns", "highest_risk_agent_turn"}:
            failures.append(f"invalid {split} MultiDoGO real family: {family_id}")
    if not set(state_by_family) <= set(real_by_family):
        failures.append(f"{split} MultiDoGO state family lacks its real source family")

    for row in real_rows:
        targets = row.get("action_targets")
        if (
            row.get("source") != SOURCE
            or row.get("license") != LICENSE
            or row.get("label") != "SAFE"
            or row.get("is_synthetic") is not False
            or row.get("naturally_occurring_communication") is not False
            or row.get("provenance_class")
            != "human_customer_and_trained_agent_roleplay"
            or row.get("action_verdict_weight") != 0.5
            or row.get("source_revision") != REVISION
            or not isinstance(targets, dict)
            or tuple(targets) != ACTION_TARGETS
        ):
            failures.append(f"invalid {split} MultiDoGO real row: {row.get('id')}")

    expected_state_labels = {
        "routine_safe": "SAFE",
        "verified_safe": "SAFE",
        "unresolved": "UNCERTAIN",
        "harmful_scam": "SCAM",
    }
    for family_id, family_rows in state_by_family.items():
        if len(family_rows) != 4 or {
            str(row.get("contrast_state")) for row in family_rows
        } != set(CONTRAST_STATES):
            failures.append(f"incomplete {split} MultiDoGO state family: {family_id}")
            continue
        if len({str(row.get("shared_context_sha256")) for row in family_rows}) != 1:
            failures.append(f"invalid {split} MultiDoGO shared context: {family_id}")
        for row in family_rows:
            targets = row.get("action_targets")
            state = str(row.get("contrast_state"))
            if (
                row.get("source") != STATE_SOURCE
                or row.get("license") != LICENSE
                or row.get("label") != expected_state_labels.get(state)
                or row.get("is_synthetic") is not True
                or row.get("human_grounded") is not True
                or row.get("decisive_action_precedes_shared_continuation") is not True
                or row.get("external_benchmark_text_copied") is not False
                or row.get("source_revision") != REVISION
                or not isinstance(targets, dict)
                or tuple(targets) != ACTION_TARGETS
            ):
                failures.append(f"invalid {split} MultiDoGO state row: {row.get('id')}")
    return failures


def verify(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    data_dir = Path(data["directory"])
    teacher = config["teacher"]
    initialization = config["initialization"]
    source_manifest_path = Path(data["multidogo_source_manifest"])
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
        data_dir / "multidogo_call_validation.jsonl": data[
            "multidogo_call_validation_sha256"
        ],
        data_dir / "multidogo_state_validation.jsonl": data[
            "multidogo_state_validation_sha256"
        ],
        source_manifest_path: data["multidogo_source_manifest_sha256"],
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
    multidogo_call_validation = read_jsonl(
        data_dir / "multidogo_call_validation.jsonl"
    )
    multidogo_state_validation = read_jsonl(
        data_dir / "multidogo_state_validation.jsonl"
    )
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
    roleplay_rows = [
        row
        for row in train_rows
        if row.get("source") in {"taskmaster1_woz_dialogues", SOURCE}
    ]
    if len(roleplay_rows) != data["human_authored_or_spoken_roleplay_train_rows"]:
        failures.append("human roleplay training count differs from config")

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

    multidogo_real_train = [row for row in train_rows if row.get("source") == SOURCE]
    multidogo_state_train = [
        row for row in train_rows if row.get("source") == STATE_SOURCE
    ]
    actual_counts = {
        "multidogo_real_train_rows": len(multidogo_real_train),
        "multidogo_train_families": len(
            {str(row.get("family_id")) for row in multidogo_real_train}
        ),
        "multidogo_state_train_rows": len(multidogo_state_train),
        "multidogo_state_train_families": len(
            {str(row.get("family_id")) for row in multidogo_state_train}
        ),
        "multidogo_call_validation_rows": len(multidogo_call_validation),
        "multidogo_validation_families": len(
            {str(row.get("family_id")) for row in multidogo_call_validation}
        ),
        "multidogo_state_validation_rows": len(multidogo_state_validation),
        "multidogo_state_validation_families": len(
            {str(row.get("family_id")) for row in multidogo_state_validation}
        ),
    }
    for key, actual in actual_counts.items():
        if data.get(key) != actual:
            failures.append(f"{key}: expected {data.get(key)}, found {actual}")
    failures.extend(
        multidogo_failures(multidogo_real_train, multidogo_state_train, "train")
    )
    failures.extend(
        multidogo_failures(
            multidogo_call_validation, multidogo_state_validation, "validation"
        )
    )

    train_state_domains = {
        str(row.get("source_domain")) for row in multidogo_state_train
    }
    validation_state_domains = {
        str(row.get("source_domain")) for row in multidogo_state_validation
    }
    if train_state_domains != set(TRAIN_STATE_DOMAINS) or train_state_domains != set(
        data["multidogo_state_train_domains"]
    ):
        failures.append("MultiDoGO state training domains differ from config")
    if validation_state_domains != set(
        VALIDATION_STATE_DOMAINS
    ) or validation_state_domains != set(data["multidogo_state_validation_domains"]):
        failures.append("MultiDoGO state validation domains differ from config")
    if train_state_domains & validation_state_domains:
        failures.append("MultiDoGO state domain appears in train and validation")

    train_families = {
        str(row.get("family_id"))
        for row in multidogo_real_train + multidogo_state_train
    }
    validation_families = {
        str(row.get("family_id"))
        for row in multidogo_call_validation + multidogo_state_validation
    }
    if train_families & validation_families:
        failures.append("MultiDoGO conversation family crosses train and validation")
    all_increment_rows = (
        multidogo_real_train
        + multidogo_state_train
        + multidogo_call_validation
        + multidogo_state_validation
    )
    texts = [str(row.get("text")) for row in all_increment_rows]
    ids = [str(row.get("id")) for row in all_increment_rows]
    if len(texts) != len(set(texts)):
        failures.append("MultiDoGO increment contains exact duplicate text")
    if len(ids) != len(set(ids)):
        failures.append("MultiDoGO increment contains duplicate IDs")

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
    tokenizer.truncation_side = config["training"]["truncation_side"]
    failures.extend(
        decisive_window_failures(
            multidogo_state_train + multidogo_state_validation, tokenizer, policy
        )
    )
    for prefix, split_rows in (
        ("multidogo_real_train", multidogo_real_train),
        ("multidogo_state_train", multidogo_state_train),
        ("multidogo_call_validation", multidogo_call_validation),
        ("multidogo_state_validation", multidogo_state_validation),
    ):
        for suffix, actual in token_summary(split_rows, tokenizer, policy).items():
            key = f"{prefix}_{suffix}"
            if data.get(key) != actual:
                failures.append(f"{key}: expected {data.get(key)}, found {actual}")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    increment = manifest.get("schema22_increment", {})
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != data["schema_version"]:
        failures.append("processed schema version differs from config")
    if increment.get("source_manifest_sha256") != data[
        "multidogo_source_manifest_sha256"
    ]:
        failures.append("MultiDoGO source manifest hash differs from processed manifest")
    if increment.get("source_revision") != data["multidogo_source_revision"]:
        failures.append("MultiDoGO source revision differs from config")
    if (
        source_manifest.get("revision") != REVISION
        or source_manifest.get("dialogue_tree_sha256")
        != data["multidogo_dialogue_tree_sha256"]
        or source_manifest.get("license_sha256") != LICENSE_SHA256
        or source_manifest.get("license_sha256") != data["multidogo_license_sha256"]
    ):
        failures.append("MultiDoGO raw revision, dialogue tree, or license differs")
    source_policy = source_manifest.get("policy", {})
    for flag in (
        "audio_downloaded",
        "direct_reddit_scrape",
        "model_rows_redistributed",
        "conversation_family_cross_split",
    ):
        if source_policy.get(flag) is not False:
            failures.append(f"MultiDoGO source policy {flag} is not false")
    if (
        source_policy.get("source_conversations_are_human_human_roleplay") is not True
        or source_policy.get("one_conversation_per_near_template_family") is not True
    ):
        failures.append("MultiDoGO human-roleplay or template-family policy differs")
    if increment.get("bothbosu_ood_opened") is not True or increment.get(
        "bothbosu_used_for_schema22_design"
    ) is not True:
        failures.append("prior-open BothBosu design use is not disclosed")
    for flag in ("apptek_ood_opened", "moz_holdout_opened", "youtube_ood_opened"):
        if increment.get(flag) is not False:
            failures.append(f"{flag} is not recorded as sealed")

    parent = Path("data/experiments/schema20-action-states/processed")
    for filename, contract in manifest.get("preserved_parent_artifacts", {}).items():
        if (
            not contract.get("byte_identical_to_parent")
            or file_sha256(data_dir / filename) != file_sha256(parent / filename)
        ):
            failures.append(f"preserved parent artifact drifted: {filename}")

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
    teacher_manifest = json.loads(Path(teacher["manifest"]).read_text(encoding="utf-8"))
    if teacher_manifest.get("contains_text") is not False or teacher[
        "contains_text"
    ] is not False:
        failures.append("teacher cache is not explicitly text-free")
    if teacher_manifest.get("checkpoint_model_sha256") != initialization["model_sha256"]:
        failures.append("teacher checkpoint differs from initialization checkpoint")

    frozen_training = {
        "epochs": 1.0,
        "batch_size": 16,
        "gradient_accumulation": 1,
        "optimizer_steps": 1513,
        "learning_rate": 0.000005,
        "max_length": 256,
        "truncation_side": "left",
        "dialogue_policy": "speaker-neutral-v1",
        "binary_loss_weight": 1.0,
        "retention_weight": 4.0,
        "retention_temperature": 2.0,
        "action_loss_weight": 0.5,
        "default_action_verdict_weight": 0.25,
        "real_multidogo_action_verdict_weight": 0.5,
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
        failures.append("training recipe differs from the frozen schema-22 contract")
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
    for key, expected in (
        ("multidogo_call_validation_fpr_max", 0.02),
        ("multidogo_call_domain_fpr_max", 0.03),
        ("bothbosu_latest_window_fpr_max", 0.02),
    ):
        if quality.get(key) != expected:
            failures.append(f"quality gate differs: {key}")
    if failures:
        raise SystemExit("schema-22 experiment preflight failed:\n" + "\n".join(failures))

    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "expected_optimizer_steps": expected_steps,
        "licensed_train_rows": len(licensed_rows),
        "teacher_anchor_rows": len(teacher_rows),
        "unanchored_rows": len(train_ids - teacher_ids),
        "multidogo_train_families": len(
            {str(row["family_id"]) for row in multidogo_real_train}
        ),
        "multidogo_validation_families": len(
            {str(row["family_id"]) for row in multidogo_call_validation}
        ),
        "held_out_state_domains": sorted(validation_state_domains),
        "action_positive_counts": action_positive_counts,
        "state_train_labels": dict(
            Counter(str(row["label"]) for row in multidogo_state_train)
        ),
        "state_validation_labels": dict(
            Counter(str(row["label"]) for row in multidogo_state_validation)
        ),
        "classifier_outputs": model.classifier.out_features,
        "verdict_head_preserved_before_training": True,
        "bothbosu_status": "prior-open regression diagnostic",
        "sealed_artifacts_opened": False,
        "status": "preflight_passed",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
