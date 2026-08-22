#!/usr/bin/env python3
"""Fail closed when the schema-19 call-window experiment contract drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from scamguard.metrics import file_sha256
from scamguard.preprocessing import prepare_model_text

try:
    from scripts.build_schema19_call_windows import LONG_PAIR_SOURCE
    from scripts.verify_encoder_pair_config import model_file, read_jsonl
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_schema19_call_windows import LONG_PAIR_SOURCE  # type: ignore[no-redef]
    from verify_encoder_pair_config import model_file, read_jsonl  # type: ignore[no-redef]


def pair_failures(
    rows: list[dict[str, object]],
    expected_rows: int,
    expected_families: int,
    expected_labels: dict[str, int],
    split: str,
) -> list[str]:
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"{split} rows: expected {expected_rows}, found {len(rows)}")
    if Counter(str(row.get("label")) for row in rows) != Counter(expected_labels):
        failures.append(f"{split} labels differ from config")
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("source") != LONG_PAIR_SOURCE:
            failures.append(f"{split} row has unexpected source: {row.get('id')}")
        if row.get("context_window_curriculum") != "long_shared_history_before_final_action":
            failures.append(f"{split} row lacks the long-window contract: {row.get('id')}")
        grouped[str(row.get("pair_id", ""))].append(row)
    if "" in grouped:
        failures.append(f"{split} contains an empty pair ID")
    if len(grouped) != expected_families:
        failures.append(
            f"{split} pair families: expected {expected_families}, found {len(grouped)}"
        )
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or {str(row.get("label")) for row in pair} != {"SAFE", "SCAM"}:
            failures.append(f"{split} pair {pair_id} is not one SAFE plus one SCAM")
            continue
        contexts = [str(row.get("text", "")).rsplit("\nAGENT:", 1)[0] for row in pair]
        if len(set(contexts)) != 1:
            failures.append(f"{split} pair {pair_id} has different preceding turns")
            continue
        context_hash = hashlib.sha256(contexts[0].encode("utf-8")).hexdigest()
        if {str(row.get("shared_context_sha256")) for row in pair} != {context_hash}:
            failures.append(f"{split} pair {pair_id} has an invalid context hash")
    return failures


def token_lengths(
    rows: list[dict[str, object]], tokenizer: object, dialogue_policy: str
) -> list[int]:
    return [
        len(
            tokenizer(  # type: ignore[operator]
                prepare_model_text(str(row["text"]), dialogue_policy),
                add_special_tokens=True,
            )["input_ids"]
        )
        for row in rows
    ]


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
        data_dir / "call_pair_validation.jsonl": data["call_pair_validation_sha256"],
        data_dir / "call_window_validation.jsonl": data["call_window_validation_sha256"],
        Path(data["pair_source_data"]): data["pair_source_data_sha256"],
        Path(data["pair_source_manifest"]): data["pair_source_manifest_sha256"],
        Path(teacher["ledger"]): teacher["ledger_sha256"],
        Path(teacher["manifest"]): teacher["manifest_sha256"],
        model_file(Path(initialization["checkpoint"])): initialization["model_sha256"],
    }
    failures = [
        f"{path}: expected {expected}, found {file_sha256(path)}"
        for path, expected in expected_hashes.items()
        if file_sha256(path) != expected
    ]

    train_rows = read_jsonl(data_dir / "train.jsonl")
    pair_validation = read_jsonl(data_dir / "call_pair_validation.jsonl")
    window_validation = read_jsonl(data_dir / "call_window_validation.jsonl")
    if len(train_rows) != data["train_rows"]:
        failures.append(f"train rows: expected {data['train_rows']}, found {len(train_rows)}")
    if len(window_validation) != data["call_window_validation_rows"]:
        failures.append("call-window validation row count differs from config")
    window_labels = Counter(str(row.get("label")) for row in window_validation)
    if window_labels != {"SAFE": len(window_validation)}:
        failures.append("call-window validation is not SAFE-only")
    source_counts = Counter(str(row.get("source")) for row in train_rows)
    for source, expected in data["supervised_source_totals"].items():
        if source_counts[source] != expected:
            failures.append(f"source {source}: expected {expected}, found {source_counts[source]}")
    window_counts = Counter(str(row.get("source_window")) for row in train_rows)
    increments = data["schema19_increment_rows"]
    if window_counts["recent"] != increments["youtube_recent"]:
        failures.append("YouTube recent-window count differs from config")
    if window_counts["recent_long"] != increments["youtube_recent_long"]:
        failures.append("YouTube long-window count differs from config")
    if window_counts["recent_complete_turns_long"] != increments["taskmaster_recent_long"]:
        failures.append("Taskmaster long-window count differs from config")

    train_pairs = [row for row in train_rows if row.get("source") == LONG_PAIR_SOURCE]
    failures.extend(
        pair_failures(
            train_pairs,
            increments["long_action_pairs"],
            data["pair_train_families"],
            data["pair_train_labels"],
            "train",
        )
    )
    failures.extend(
        pair_failures(
            pair_validation,
            data["call_pair_validation_rows"],
            data["pair_validation_families"],
            data["pair_validation_labels"],
            "validation",
        )
    )
    holdouts = set(data["holdout_scenarios"])
    if {str(row.get("scenario")) for row in pair_validation} != holdouts:
        failures.append("validation scenarios differ from config")
    if {str(row.get("scenario")) for row in train_pairs} & holdouts:
        failures.append("holdout scenarios appear in training")

    tokenizer = AutoTokenizer.from_pretrained(
        Path(initialization["checkpoint"]), local_files_only=True
    )
    policy = config["training"]["dialogue_policy"]
    if min(token_lengths(train_pairs, tokenizer, policy)) != data["pair_train_token_min"]:
        failures.append("training-pair token minimum differs from config")
    if (
        min(token_lengths(pair_validation, tokenizer, policy))
        != data["pair_validation_token_min"]
    ):
        failures.append("validation-pair token minimum differs from config")
    if (
        statistics.median(token_lengths(window_validation, tokenizer, policy))
        != data["call_window_validation_token_p50"]
    ):
        failures.append("call-window token median differs from config")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    increment = manifest.get("schema19_increment", {})
    if manifest.get("schema_version") != data["schema_version"]:
        failures.append("processed schema version differs from config")
    if increment.get("pair_train_families") != data["pair_train_families"]:
        failures.append("manifest train pair count differs from config")
    if increment.get("pair_validation_families") != data["pair_validation_families"]:
        failures.append("manifest validation pair count differs from config")
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
    if len(teacher_rows) != teacher["anchor_rows"] or len(teacher_ids) != teacher["anchor_rows"]:
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
        "learning_rate": 0.000005,
        "max_length": 256,
        "truncation_side": "left",
        "dialogue_policy": "speaker-neutral-v1",
        "binary_loss_weight": 1.0,
        "retention_weight": 4.0,
        "retention_temperature": 2.0,
        "pair_loss_weight": 1.0,
        "pair_margin": 3.0,
        "pair_repeats": 1,
        "pair_sampler": "every row once; every complete pair contained in one even-sized batch",
        "source_balance_alpha": 0.0,
        "seed": 20260820,
        "checkpoint_selection": "development recall at the frozen 2-percent FPR cap",
    }
    if config.get("training") != frozen_training:
        failures.append("training recipe differs from the frozen schema-19 contract")
    quality = config.get("quality_acceptance", {})
    if quality.get("call_pair_validation_order_accuracy_min") != 1.0:
        failures.append("paired validation does not require perfect ordering")
    if quality.get("call_window_validation_fpr_max") != 0.02:
        failures.append("long SAFE-call validation does not enforce the 2-percent FPR cap")
    expected_steps = math.ceil(len(train_rows) / frozen_training["batch_size"])
    if failures:
        raise SystemExit("schema-19 experiment preflight failed:\n" + "\n".join(failures))

    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "effective_sample_rows": len(train_rows),
        "expected_optimizer_steps": expected_steps,
        "teacher_anchor_rows": len(teacher_rows),
        "unanchored_rows": len(train_ids - teacher_ids),
        "pair_train_families": data["pair_train_families"],
        "pair_validation_families": data["pair_validation_families"],
        "call_window_validation_rows": len(window_validation),
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
        default=Path("configs/encoder-schema19-windowmix-ret4-w1-m3-left.json"),
    )
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
