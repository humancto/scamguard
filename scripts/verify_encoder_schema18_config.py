#!/usr/bin/env python3
"""Fail closed when the schema-18 evidence-action experiment contract drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.generate_call_evidence_pairs import SOURCE
    from scripts.verify_encoder_pair_config import model_file, read_jsonl
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the repo on sys.path.
    from generate_call_evidence_pairs import SOURCE  # type: ignore[no-redef]
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
        if row.get("source") != SOURCE:
            failures.append(f"{split} row has unexpected source: {row.get('id')}")
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
        Path(data["source_data"]): data["source_data_sha256"],
        Path(data["source_manifest"]): data["source_manifest_sha256"],
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

    train_pairs = [row for row in train_rows if row.get("source") == SOURCE]
    failures.extend(
        pair_failures(
            train_pairs,
            data["new_supervised_rows"][SOURCE],
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
    if {str(row.get("context_frame")) for row in train_pairs + validation_rows} != set(
        data["context_frames"]
    ):
        failures.append("context frames differ from config")
    if {str(row.get("risk_mechanism")) for row in train_pairs + validation_rows} != set(
        data["risk_mechanisms"]
    ):
        failures.append("risk mechanisms differ from config")
    holdouts = set(data["holdout_scenarios"])
    if {str(row.get("scenario")) for row in validation_rows} != holdouts:
        failures.append("validation scenarios differ from config")
    if {str(row.get("scenario")) for row in train_pairs} & holdouts:
        failures.append("holdout scenarios appear in training")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    increment = manifest.get("schema18_increment", {})
    if manifest.get("schema_version") != data["schema_version"]:
        failures.append("processed schema version differs from config")
    if increment.get("train_pair_families") != data["pair_train_families"]:
        failures.append("manifest train pair count differs from config")
    if increment.get("validation_pair_families") != data["pair_validation_families"]:
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
        "pair_loss_weight": 2.0,
        "pair_margin": 3.0,
        "pair_repeats": 2,
        "pair_sampler": (
            "legacy rows once; paired rows twice; every complete pair contained in one "
            "even-sized batch"
        ),
        "source_balance_alpha": 0.0,
        "seed": 20260820,
        "checkpoint_selection": "development recall at the frozen 2-percent FPR cap",
    }
    if config.get("training") != frozen_training:
        failures.append("training recipe differs from the frozen schema-18 contract")
    quality = config.get("quality_acceptance", {})
    if quality.get("call_pair_validation_order_accuracy_min") != 1.0:
        failures.append("paired validation does not require perfect ordering")
    expected_sample_rows = len(train_rows) - len(train_pairs) + 2 * len(train_pairs)
    expected_steps = math.ceil(expected_sample_rows / frozen_training["batch_size"])
    if failures:
        raise SystemExit("schema-18 experiment preflight failed:\n" + "\n".join(failures))

    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "effective_sample_rows": expected_sample_rows,
        "expected_optimizer_steps": expected_steps,
        "teacher_anchor_rows": len(teacher_rows),
        "unanchored_rows": len(train_ids - teacher_ids),
        "pair_train_families": data["pair_train_families"],
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
        default=Path("configs/encoder-schema18-action-pairx2-ret4-w2-m3-left.json"),
    )
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
