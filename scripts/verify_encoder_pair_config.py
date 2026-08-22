#!/usr/bin/env python3
"""Fail closed when the schema-17 paired-call experiment drifts from its contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256

PAIR_SOURCE = "scamguard_synthetic_call_minimal_pairs_v1"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def model_file(checkpoint: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing model weights in {checkpoint}")


def pair_failures(
    rows: list[dict[str, object]],
    expected_rows: int,
    expected_families: int,
    expected_labels: dict[str, int],
    split: str,
) -> list[str]:
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"{split} pair rows: expected {expected_rows}, found {len(rows)}")
    labels = Counter(str(row.get("label")) for row in rows)
    if labels != Counter(expected_labels):
        failures.append(f"{split} pair labels differ from config: {dict(labels)}")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("source") != PAIR_SOURCE:
            failures.append(f"{split} pair row has unexpected source: {row.get('id')}")
        grouped[str(row.get("pair_id", ""))].append(row)
    if "" in grouped:
        failures.append(f"{split} pair data contains an empty pair_id")
    if len(grouped) != expected_families:
        failures.append(
            f"{split} pair families: expected {expected_families}, found {len(grouped)}"
        )
    for pair_id, members in grouped.items():
        if len(members) != 2 or {str(row.get("label")) for row in members} != {
            "SAFE",
            "SCAM",
        }:
            failures.append(f"{split} pair {pair_id} is not one SAFE plus one SCAM")
            continue
        contexts = []
        declared_hashes = set()
        for row in members:
            text = str(row.get("text", ""))
            if "\nAGENT:" not in text:
                failures.append(f"{split} pair {pair_id} lacks a final AGENT turn")
                continue
            context = text.rsplit("\nAGENT:", 1)[0]
            contexts.append(context)
            declared_hashes.add(str(row.get("shared_context_sha256", "")))
        if len(set(contexts)) != 1:
            failures.append(f"{split} pair {pair_id} does not share exact preceding turns")
        if len(contexts) == 2:
            actual_hash = hashlib.sha256(contexts[0].encode("utf-8")).hexdigest()
            if declared_hashes != {actual_hash}:
                failures.append(f"{split} pair {pair_id} has an invalid context hash")
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
        data_dir / "call_pair_validation.jsonl": data["call_pair_validation_sha256"],
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
    validation_rows = read_jsonl(data_dir / "call_pair_validation.jsonl")
    if len(train_rows) != data["train_rows"]:
        failures.append(f"train rows: expected {data['train_rows']}, found {len(train_rows)}")
    source_counts = Counter(str(row.get("source")) for row in train_rows)
    for source, expected in data["new_supervised_rows"].items():
        if source_counts[source] != expected:
            failures.append(f"source {source}: expected {expected}, found {source_counts[source]}")

    train_pairs = [row for row in train_rows if row.get("source") == PAIR_SOURCE]
    failures.extend(
        pair_failures(
            train_pairs,
            data["new_supervised_rows"][PAIR_SOURCE],
            data["pair_train_families"],
            data["pair_train_labels"],
            "train",
        )
    )
    failures.extend(
        pair_failures(
            validation_rows,
            data["call_pair_validation_rows"],
            data["pair_validation_families"],
            data["pair_validation_labels"],
            "validation",
        )
    )
    train_scenarios = {str(row.get("scenario")) for row in train_pairs}
    validation_scenarios = {str(row.get("scenario")) for row in validation_rows}
    holdout_scenarios = set(data["holdout_scenarios"])
    if validation_scenarios != holdout_scenarios:
        failures.append("validation scenarios differ from the frozen holdout scenarios")
    if train_scenarios & holdout_scenarios:
        failures.append("held-out pair scenarios appear in training")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    increment = manifest.get("schema17_increment", {})
    if manifest.get("schema_version") != data["schema_version"]:
        failures.append("processed data schema version differs from config")
    if increment.get("train_pair_families") != data["pair_train_families"]:
        failures.append("manifest training pair-family count differs from config")
    if increment.get("validation_pair_families") != data["pair_validation_families"]:
        failures.append("manifest validation pair-family count differs from config")
    for flag in (
        "apptek_ood_opened",
        "bothbosu_ood_opened",
        "moz_holdout_opened",
        "youtube_ood_opened",
    ):
        if increment.get(flag) is not False:
            failures.append(f"{flag} is not recorded as sealed")
    if manifest.get("schema14_increment", {}).get("sealed_ood_opened") is not False:
        failures.append("YouTube-call OOD is not recorded as sealed")

    teacher_rows = read_jsonl(Path(teacher["ledger"]))
    teacher_ids = {str(row.get("id")) for row in teacher_rows}
    train_ids = {str(row.get("id")) for row in train_rows}
    if len(teacher_rows) != teacher["anchor_rows"] or len(teacher_ids) != teacher["anchor_rows"]:
        failures.append("teacher ledger row count differs from config or contains duplicate IDs")
    if not teacher_ids <= train_ids:
        failures.append("teacher ledger contains IDs absent from schema-17 training")
    if len(train_ids - teacher_ids) != teacher["unanchored_rows"]:
        failures.append("unanchored training row count differs from config")
    if any(set(row) != {"id", "logits"} for row in teacher_rows):
        failures.append("teacher ledger is not limited to IDs and logits")
    teacher_manifest = json.loads(Path(teacher["manifest"]).read_text(encoding="utf-8"))
    if teacher_manifest.get("rows") != teacher["anchor_rows"]:
        failures.append("teacher manifest anchor count differs from config")
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
        "dialogue_policy": "speaker-neutral-v1",
        "binary_loss_weight": 1.0,
        "retention_weight": 2.0,
        "retention_temperature": 2.0,
        "pair_loss_weight": 0.5,
        "pair_margin": 2.0,
        "pair_sampler": (
            "every row once per epoch; each complete pair contained in one even-sized batch"
        ),
        "source_balance_alpha": 0.0,
        "seed": 20260820,
        "checkpoint_selection": "development recall at the frozen 2-percent FPR cap",
    }
    if config.get("training") != frozen_training:
        failures.append("training recipe differs from the frozen schema-17 contract")
    acceptance = config.get("acceptance", {})
    if acceptance.get("call_pair_validation_order_accuracy_min") != 1.0:
        failures.append("paired validation does not require perfect ordering")
    if config.get("failure_policy") != (
        "reject without export or sealed evaluation if any acceptance gate fails"
    ):
        failures.append("failure policy differs from the frozen reject-without-export rule")
    if failures:
        raise SystemExit("paired experiment preflight failed:\n" + "\n".join(failures))

    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "teacher_anchor_rows": len(teacher_rows),
        "unanchored_rows": len(train_ids - teacher_ids),
        "pair_train_rows": len(train_pairs),
        "pair_train_families": data["pair_train_families"],
        "pair_validation_rows": len(validation_rows),
        "pair_validation_families": data["pair_validation_families"],
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
        default=Path("configs/encoder-schema17-pair-retention-w05-m2.json"),
    )
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
