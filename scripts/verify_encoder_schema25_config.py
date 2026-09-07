#!/usr/bin/env python3
"""Fail closed on the predeclared schema-25 encoder experiment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from scripts.build_phone_scam_synthetic import SOURCE as PHONE_SOURCE
    from scripts.build_schema19_call_windows import read_jsonl
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_phone_scam_synthetic import SOURCE as PHONE_SOURCE  # type: ignore[no-redef]
    from build_schema19_call_windows import read_jsonl  # type: ignore[no-redef]

CONFIG_PATH = Path("configs/encoder-schema25-fullcall-phone-ret4-aw05-vw025-lr1e6-right.json")
MULTIDOGO_SOURCE = "multidogo_human_service_dialogues"


def check_equal(failures: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{name}: expected {expected!r}, got {actual!r}")


def verify(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    check_equal(
        failures,
        "predeclared_before_training",
        config.get("predeclared_before_training"),
        True,
    )
    check_equal(
        failures,
        "training_recipe_version",
        config.get("training_recipe_version"),
        25,
    )

    data = config["data"]
    data_dir = Path(data["directory"])
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, filename in (
        ("manifest_sha256", "manifest.json"),
        ("train_sha256", "train.jsonl"),
        ("dev_sha256", "dev.jsonl"),
        ("test_sha256", "test.jsonl"),
        ("phone_scam_validation_sha256", "phone_scam_validation.jsonl"),
    ):
        check_equal(failures, name, file_sha256(data_dir / filename), data[name])
    check_equal(failures, "schema_version", manifest.get("schema_version"), data["schema_version"])
    check_equal(failures, "train_rows", manifest["counts"].get("train"), data["train_rows"])
    check_equal(
        failures,
        "phone_scam_validation_rows",
        manifest["counts"].get("phone_scam_validation"),
        data["phone_scam_validation_rows"],
    )
    increment = manifest.get("schema25_increment", {})
    for config_key, manifest_key in (
        ("multidogo_full_calls_admitted", "multidogo_complete_calls_admitted"),
        ("phone_synthetic_train_rows", "phone_train_admitted"),
        ("schema24_unaudited_rows_used_for_fitting", "schema24_unaudited_rows_used_for_fitting"),
        ("phone_test_predictions_opened", "phone_test_predictions_opened"),
        ("sealed_primary_holdouts_opened", "sealed_primary_holdouts_opened"),
    ):
        check_equal(failures, config_key, increment.get(manifest_key), data[config_key])

    initialization = config["initialization"]
    model_path = Path(initialization["checkpoint"]) / "model.safetensors"
    check_equal(
        failures,
        "initialization model",
        file_sha256(model_path),
        initialization["model_sha256"],
    )

    teacher = config["teacher"]
    teacher_manifest_path = Path(teacher["manifest"])
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    check_equal(
        failures,
        "teacher ledger",
        file_sha256(Path(teacher["ledger"])),
        teacher["ledger_sha256"],
    )
    check_equal(
        failures,
        "teacher manifest",
        file_sha256(teacher_manifest_path),
        teacher["manifest_sha256"],
    )
    check_equal(failures, "teacher rows", teacher_manifest.get("rows"), teacher["anchor_rows"])
    check_equal(failures, "teacher contains_text", teacher_manifest.get("contains_text"), False)
    check_equal(
        failures,
        "teacher checkpoint",
        teacher_manifest.get("checkpoint_model_sha256"),
        initialization["model_sha256"],
    )

    train = read_jsonl(data_dir / "train.jsonl")
    teacher_ids = {str(row["id"]) for row in read_jsonl(Path(teacher["ledger"]))}
    train_ids = {str(row["id"]) for row in train}
    if not teacher_ids <= train_ids:
        failures.append("teacher ledger contains IDs outside schema-25 training data")
    check_equal(
        failures,
        "unanchored_new_rows",
        len(train_ids - teacher_ids),
        teacher["unanchored_new_rows"],
    )
    phone_rows = [row for row in train if row.get("source") == PHONE_SOURCE]
    full_calls = [
        row
        for row in train
        if row.get("source") == MULTIDOGO_SOURCE
        and row.get("schema25_role") == "licensed_human_spoken_roleplay_full_call_safe_evidence"
    ]
    check_equal(
        failures,
        "phone training rows",
        len(phone_rows),
        data["phone_synthetic_train_rows"],
    )
    check_equal(
        failures,
        "full-call training rows",
        len(full_calls),
        data["multidogo_full_calls_admitted"],
    )
    if any(isinstance(row.get("action_targets"), dict) for row in full_calls):
        failures.append("schema-25 full calls retain weak action targets")
    if any(row.get("schema24_admitted") is True for row in train):
        failures.append("schema-25 train retains unaudited schema-24 rows")
    if (data_dir / "phone_scam_test.jsonl").exists():
        failures.append("phone publisher test was copied into the experiment directory")

    if failures:
        raise RuntimeError("schema-25 preflight failed:\n- " + "\n- ".join(failures))
    result: dict[str, object] = {
        "status": "passed",
        "experiment_id": config["experiment_id"],
        "train_rows": len(train),
        "teacher_anchor_rows": len(teacher_ids),
        "unanchored_new_rows": len(train_ids - teacher_ids),
        "phone_training_rows": len(phone_rows),
        "multidogo_full_call_rows": len(full_calls),
        "publisher_test_present_in_experiment": False,
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
