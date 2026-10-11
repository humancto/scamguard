#!/usr/bin/env python3
"""One bounded Ettin v1 continuation with frozen, train-only targeted replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from training import run_bounded_encoder as bounded
from training.train_encoder import LABELS, EncodedDataset, WeightedTrainer

PARENT_MODEL_SHA256 = "d7ee45bb5d2cc12e0b0f26171cca45234cb9a0a7c517237971e31c2766f06803"
TRAIN_SHA256 = "d4e3e100e97a937a789aa984d4ce75b545592886099a0027dbd758bbf0b2e614"
DEV_SHA256 = "04201ee86325306cf1751dbf848ddca360386de707069c0c4bba0c9c28a21c07"
SERVICE_SOURCES = frozenset(
    {"multidogo_human_service_dialogues", "shakeleoatmeal_phone_scam_synthetic"}
)
TARGET_COUNTS = {"identity_scam": 1206, "service_safe": 1400}


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def replay_group(row: dict[str, Any]) -> str:
    if row["label"] == "SCAM" and row.get("category") == "IDENTITY_IMPERSONATION":
        return "identity_scam"
    if row["label"] == "SAFE" and row["source"] in SERVICE_SOURCES:
        return "service_safe"
    return "ordinary"


def build_replay(
    rows: list[dict], seed: int, extra_target_repeats: int = 2
) -> tuple[list[int], dict]:
    """Freeze indices, not copied rows: every original row once, targets three times."""
    if not rows or extra_target_repeats != 2:
        raise ValueError("replay requires nonempty training rows and exactly two extra exposures")
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("replay input IDs must be unique")
    groups = {name: [] for name in ("identity_scam", "service_safe", "ordinary")}
    for index, row in enumerate(rows):
        groups[replay_group(row)].append(index)
    target_indices = sorted(groups["identity_scam"] + groups["service_safe"])
    indices = list(range(len(rows))) + target_indices * extra_target_repeats
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    indices = [indices[position] for position in order]
    group_metadata = {
        name: {
            "unique_rows": len(members),
            "presentations": len(members) * (1 if name == "ordinary" else 3),
            "selected_ids_sha256": json_sha256(sorted(identifiers[index] for index in members)),
        }
        for name, members in groups.items()
    }
    return indices, {
        "protocol": "ettin-targeted-full-replay-v1",
        "seed": seed,
        "replacement_sampling": False,
        "every_training_row_included": True,
        "extra_target_repeats": extra_target_repeats,
        "unique_training_rows": len(rows),
        "presentations": len(indices),
        "index_order_sha256": json_sha256(indices),
        "row_id_order_sha256": json_sha256([identifiers[index] for index in indices]),
        "groups": group_metadata,
        "presentation_labels": dict(Counter(rows[index]["label"] for index in indices)),
        "presentation_sources": dict(Counter(rows[index]["source"] for index in indices)),
        "source_texts_or_labels_modified": False,
    }


class FrozenReplaySampler(Sampler[int]):
    def __init__(self, indices: list[int], dataset_size: int):
        if not indices or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < dataset_size
            for index in indices
        ):
            raise ValueError("invalid frozen replay indices")
        self.indices = tuple(indices)
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


class ReplayTrainer(WeightedTrainer):
    def __init__(self, *args, replay_indices: list[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_sampler = FrozenReplaySampler(replay_indices, len(self.train_dataset))

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if len(dataset) != self.replay_sampler.dataset_size:
            raise ValueError("frozen replay dataset size changed")
        return self.replay_sampler


def validate_config(config: dict) -> None:
    settings = config["training"]
    # Reuse v1's exact model/data field validation without relaxing its frozen implementation.
    proxy = dict(config)
    proxy["training"] = {key: settings[key] for key in bounded.TRAINING_FIELDS if key in settings}
    proxy["training"].setdefault("epochs", 1)
    proxy["training"].setdefault("binary_loss_weight", 0.0)
    proxy["outputs"] = {
        key: value for key, value in config["outputs"].items() if key != "replay_manifest"
    }
    bounded.validate_config(proxy)
    expected = {
        "batch_size": 16,
        "learning_rate": 5e-6,
        "max_steps": 1853,
        "max_length": 256,
        "truncation_side": "right",
        "dialogue_policy": "speaker-neutral-evidence-recent-v2",
        "seed": 20261011,
        "extra_target_repeats": 2,
        "expected_presentations": 29648,
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"frozen recovery setting differs: {key}")
    if settings.get("epochs", 1) != 1 or settings.get("binary_loss_weight", 0.0) != 0:
        raise ValueError("only one replay pass with three-class cross entropy is authorized")
    if settings["max_wall_seconds"] > 2700:
        raise ValueError("recovery wall bound may not exceed 45 minutes")
    if settings.get("expected_target_counts") != TARGET_COUNTS:
        raise ValueError("recovery target counts differ from the frozen schema25 inventory")
    if (
        config["data"]["train"]["sha256"] != TRAIN_SHA256
        or config["data"]["dev"]["sha256"] != DEV_SHA256
    ):
        raise ValueError("recovery requires the unchanged schema25 training/development data")
    if config["data"]["train"]["rows"] != 24436 or config["data"]["dev"]["rows"] != 2634:
        raise ValueError("recovery requires the unchanged schema25 row counts")
    initialization = config["initialization"]
    if initialization.get("model_sha256") != PARENT_MODEL_SHA256:
        raise ValueError("recovery must start from the exact completed Ettin v1 weights")
    for forbidden in ("resume_from_checkpoint", "resume_or_continuation_allowed"):
        if settings.get(forbidden):
            raise ValueError("optimizer resume or additional continuation is not authorized")
    for objective in (
        "retention_weight",
        "pair_loss_weight",
        "action_loss_weight",
        "source_balance_alpha",
    ):
        if settings.get(objective, 0) != 0:
            raise ValueError("recovery must not add objectives or source-balanced sampling")
    if settings.get("hyperparameter_runs", 1) != 1:
        raise ValueError("recovery authorizes one fit only")
    paths = artifact_paths(config)
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise ValueError("recovery outputs must be distinct")
    if paths["checkpoint"].resolve() == Path(initialization["checkpoint"]).resolve():
        raise ValueError("recovery must not overwrite its parent checkpoint")


def artifact_paths(config: dict) -> dict[str, Path]:
    paths = bounded.artifact_paths(config)
    paths["replay_manifest"] = Path(
        config["outputs"].get(
            "replay_manifest", str(paths["report"].with_name(paths["report"].stem + ".replay.json"))
        )
    )
    return paths


def verify_parent(initialization: dict) -> dict:
    parent_report_path = Path(initialization["parent_report"])
    if file_sha256(parent_report_path) != initialization["parent_report_sha256"]:
        raise ValueError("parent training receipt hash differs")
    receipt = json.loads(parent_report_path.read_text())
    if (
        receipt.get("status") != "completed"
        or receipt.get("training_completed") is not True
        or receipt.get("global_step") != 3056
        or receipt.get("stop_reason") != "step_bound_reached"
        or receipt.get("checkpoint_model_sha256") != initialization["model_sha256"]
    ):
        raise ValueError("parent receipt is not the exact complete v1 fit")
    checkpoint = Path(initialization["checkpoint"])
    expected = initialization["files_sha256"]
    if expected != receipt.get("checkpoint_files_sha256"):
        raise ValueError("parent checkpoint file contract differs from its training receipt")
    required = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "scamguard_calibration.json",
    }
    if not required <= set(expected) or any(Path(name).name != name for name in expected):
        raise ValueError("parent checkpoint file contract is incomplete or unsafe")
    actual = {name: file_sha256(checkpoint / name) for name in sorted(expected)}
    if actual != expected or actual["model.safetensors"] != initialization["model_sha256"]:
        raise ValueError("parent checkpoint files changed")
    return {
        "checkpoint": str(checkpoint),
        "parent_report": str(parent_report_path),
        "parent_report_sha256": initialization["parent_report_sha256"],
        "files_sha256": actual,
        "optimizer_state_reused": False,
    }


def source_identity() -> dict[str, str]:
    result = bounded.source_identity()
    root = Path(__file__).resolve().parents[1]
    for name in ("training/run_ettin_recovery.py", "training/ettin_calibration.py"):
        result[name] = file_sha256(root / name)
    return result


def verify_replay_receipt(path: Path, expected: dict) -> str:
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("recovery requires an unchanged, preflight-frozen replay manifest")
    return file_sha256(path)


def run(config_path: Path, *, preflight: bool = False) -> int:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    paths = artifact_paths(config)
    bounded.ensure_fresh_outputs(paths, preflight=preflight)
    rows, metadata = bounded.load_rows(config)
    settings = config["training"]
    parent = verify_parent(config["initialization"])
    indices, replay = build_replay(
        rows["train"], settings["seed"], settings["extra_target_repeats"]
    )
    if replay["presentations"] != settings["expected_presentations"] or any(
        replay["groups"][group]["unique_rows"] != count for group, count in TARGET_COUNTS.items()
    ):
        raise ValueError("replay inventory differs from the frozen config")
    steps = math.ceil(len(indices) / settings["batch_size"])
    if steps != settings["max_steps"]:
        raise ValueError("replay step count differs from the frozen config")
    sources = source_identity()
    replay_receipt = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": sources,
        "train_sha256": config["data"]["train"]["sha256"],
        "parent": parent,
        "replay": replay,
        "steps": steps,
        "torch_version": torch.__version__,
    }
    if preflight:
        if paths["replay_manifest"].exists():
            raise FileExistsError("refusing to replace an existing replay manifest")
        bounded.write_json(paths["replay_manifest"], replay_receipt, exclusive=True)
        bounded.write_json(
            paths["preflight"],
            {
                "status": "passed",
                "mode": "CPU_metadata_and_replay_only",
                "config_sha256": replay_receipt["config_sha256"],
                "parent": parent,
                "data": metadata,
                "replay": replay,
                "effective_max_steps": steps,
                "replay_manifest_sha256": file_sha256(paths["replay_manifest"]),
                "model_forward_performed": False,
                "training_performed": False,
                "release_authorized": False,
            },
            exclusive=True,
        )
        print(
            json.dumps({"status": "passed", "steps": steps, "presentations": len(indices)}),
            flush=True,
        )
        return 0
    replay_receipt_sha = verify_replay_receipt(paths["replay_manifest"], replay_receipt)
    if not paths["preflight"].is_file():
        raise ValueError("recovery requires its successful preflight receipt")
    preflight_receipt = json.loads(paths["preflight"].read_text())
    if (
        preflight_receipt.get("status") != "passed"
        or preflight_receipt.get("replay_manifest_sha256") != replay_receipt_sha
        or preflight_receipt.get("config_sha256") != replay_receipt["config_sha256"]
    ):
        raise ValueError("preflight receipt no longer matches the replay contract")

    started = time.monotonic()
    result = {
        "artifact_schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": "one_fresh_optimizer_continuation_with_frozen_full_replay",
        "config": config,
        "config_sha256": replay_receipt["config_sha256"],
        "source_sha256": sources,
        "environment": bounded.environment(),
        "data": metadata,
        "parent": parent,
        "checkpoint": str(paths["checkpoint"]),
        "effective_max_steps": steps,
        "replay": replay,
        "replay_manifest": {"path": str(paths["replay_manifest"]), "sha256": replay_receipt_sha},
        "training_completed": False,
        "stop_reason": None,
        "release_authorized": False,
        "other_splits_opened": False,
        "checkpoint_selection": "fixed terminal replay step",
        "fitting_objective": "original-row square-root class-weighted three-class CE only",
    }
    result["environment_sha256"] = json_sha256(result["environment"])
    bounded.write_json(paths["progress"], {"phase": "loading", "global_step": 0}, exclusive=True)
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("recovery requires MPS; no CPU fallback")
        set_seed(settings["seed"])
        checkpoint = config["initialization"]["checkpoint"]
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        tokenizer.truncation_side = settings["truncation_side"]
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to("mps")
        if model.config.num_labels != 3 or [
            model.config.id2label[index] for index in range(3)
        ] != list(LABELS):
            raise ValueError("parent classifier does not have the exact three-way verdict head")
        projected = {split: bounded.verdict_rows(values) for split, values in rows.items()}
        dataset = EncodedDataset(
            projected["train"],
            tokenizer,
            settings["max_length"],
            dialogue_policy=settings["dialogue_policy"],
        )
        counts = Counter(row["label"] for row in rows["train"])
        weights = torch.tensor(
            [math.sqrt(len(rows["train"]) / (3 * counts[label])) for label in LABELS],
            dtype=torch.float32,
        )
        result["class_weights"] = weights.tolist()
        result["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        result["warmup_steps"] = math.ceil(steps * 0.08)
        progress = bounded.BoundProgress(
            paths["progress"], steps, settings["max_wall_seconds"], started
        )
        args = TrainingArguments(
            output_dir=str(paths["trainer"]),
            learning_rate=settings["learning_rate"],
            per_device_train_batch_size=settings["batch_size"],
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_steps=steps,
            weight_decay=0.01,
            warmup_steps=result["warmup_steps"],
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_steps=25,
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            train_sampling_strategy="random",
            dataloader_drop_last=False,
            seed=settings["seed"],
            data_seed=settings["seed"],
        )
        result["training_arguments"] = args.to_dict()
        trainer = ReplayTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
            processing_class=tokenizer,
            class_weights=weights,
            binary_loss_weight=0.0,
            binary_positive_weight=1.0,
            replay_indices=indices,
            callbacks=[progress],
        )
        if trainer.model.device.type != "mps" or args.world_size != 1:
            raise RuntimeError("recovery requires one local MPS process")
        if time.monotonic() - started >= settings["max_wall_seconds"]:
            raise TimeoutError("wall limit reached before training")
        result["train"] = trainer.train().metrics
        result["global_step"] = trainer.state.global_step
        result["stop_reason"] = progress.stop_reason
        complete = (
            trainer.state.global_step == steps and progress.stop_reason == "step_bound_reached"
        )
        result["training_completed"] = complete
        if not complete:
            result.update(status="rejected_incomplete_fit", wall_seconds=time.monotonic() - started)
            bounded.write_json(paths["report"], result, exclusive=True)
            progress.record(trainer.state, result["status"])
            return 1
        trainer.save_model(paths["checkpoint"])
        tokenizer.save_pretrained(paths["checkpoint"])
        result["checkpoint_model_sha256"] = file_sha256(paths["checkpoint"] / "model.safetensors")
        progress.record(trainer.state, "development_serving_calibration")
        from training.ettin_calibration import calibrate_serving_dev

        calibration, metrics, records = calibrate_serving_dev(
            model,
            tokenizer,
            projected["dev"],
            settings["max_length"],
            settings["dialogue_policy"],
        )
        calibration.update(
            model_id=config["experiment_id"],
            input_transform={
                "dialogue_policy": settings["dialogue_policy"],
                "truncation_side": settings["truncation_side"],
                "max_length": settings["max_length"],
            },
            checkpoint_model_sha256=result["checkpoint_model_sha256"],
            dev_data_sha256=DEV_SHA256,
        )
        calibration_path = paths["checkpoint"] / "scamguard_calibration.json"
        bounded.write_json(calibration_path, calibration, exclusive=True)
        result["calibration_sha256"] = file_sha256(calibration_path)
        result["checkpoint_files_sha256"] = {
            path.name: file_sha256(path)
            for path in sorted(paths["checkpoint"].iterdir())
            if path.is_file()
        }
        with paths["predictions"].open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        result.update(calibration=calibration, dev=metrics)
        result["dev_gates"] = {
            "recall": metrics["binary_safety"]["scam_recall"] >= 0.97,
            "safe_fpr": metrics["binary_safety"]["false_positive_rate"] <= 0.02,
        }
        result["development_quality_passed"] = all(result["dev_gates"].values())
        result["development_macro_f1_stretch"] = {
            "target": 0.94,
            "actual": metrics["calibrated"]["macro_f1"],
            "required": False,
        }
        result["predictions"] = {
            "path": str(paths["predictions"]),
            "sha256": file_sha256(paths["predictions"]),
            "rows": len(records),
            "contains_text": False,
        }
        for split, entry in config["data"].items():
            if file_sha256(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"{split} data changed during recovery")
        verify_parent(config["initialization"])
        verify_replay_receipt(paths["replay_manifest"], replay_receipt)
        result.update(status="completed", wall_seconds=time.monotonic() - started)
        progress.record(trainer.state, "completed")
        bounded.write_json(paths["report"], result, exclusive=True)
        return 0
    except BaseException as error:
        result.update(
            status="failed",
            stop_reason=type(error).__name__,
            error=str(error)[:1000],
            wall_seconds=time.monotonic() - started,
        )
        if not paths["report"].exists():
            bounded.write_json(paths["report"], result, exclusive=True)
        bounded.write_json(
            paths["progress"],
            {
                "phase": "failed",
                "stop_reason": type(error).__name__,
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preflight", action="store_true", help="CPU-only frozen replay and metadata audit"
    )
    args = parser.parse_args()
    raise SystemExit(run(args.config, preflight=args.preflight))


if __name__ == "__main__":
    main()
