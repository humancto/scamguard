#!/usr/bin/env python3
"""One fresh, offline Ettin fit with train/dev access and a fixed terminal checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import choose_threshold, file_sha256, verdict_alert_metrics
from scamguard.preprocessing import DIALOGUE_POLICIES, prepare_model_text
from training.ettin_calibration import SERVING_CALIBRATION_VERSION, calibrate_serving_dev
from training.train_encoder import (
    LABEL_TO_ID,
    LABELS,
    EncodedDataset,
    WeightedTrainer,
    binary_subset,
    fit_temperature,
    predict_in_dataset_order,
    prediction_ledger_records,
    report_slice,
    softmax,
)

MODEL_REPOSITORY = "jhu-clsp/ettin-encoder-150m"
MODEL_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
SYNTHETIC_PREFLIGHT_TEXT = "Your dentist appointment is tomorrow. No payment or code is requested."
TRAINING_FIELDS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "max_steps",
    "max_wall_seconds",
    "max_length",
    "truncation_side",
    "dialogue_policy",
    "seed",
    "binary_loss_weight",
}
TRAINING_METADATA_FIELDS = {
    "checkpoint_selection",
    "class_weights",
    "initialization",
    "auxiliary_fields",
    "hyperparameter_runs",
    "resume_or_continuation_allowed",
}
BASE_FILES = {
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    checkpoint = Path(config["outputs"]["checkpoint"])
    report = Path(config["outputs"]["report"])
    return {
        "checkpoint": checkpoint,
        "report": report,
        "trainer": checkpoint.with_name(checkpoint.name + ".trainer"),
        "progress": report.with_name(report.stem + ".progress.json"),
        "preflight": report.with_name(report.stem + ".preflight.json"),
        "predictions": report.with_name(report.stem + ".dev.predictions.jsonl"),
    }


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("experiment_id"), str) or not config["experiment_id"]:
        raise ValueError("experiment_id is required")
    model = config.get("model", {})
    if model.get("repository") != MODEL_REPOSITORY or model.get("revision") != MODEL_REVISION:
        raise ValueError("this bounded experiment requires the pinned fresh Ettin base")
    files = model.get("files_sha256", {})
    if set(files) != BASE_FILES or any(
        not re.fullmatch(r"[a-f0-9]{64}", str(value)) for value in files.values()
    ):
        raise ValueError("the five pinned base model/tokenizer file hashes are required")
    if set(config.get("data", {})) != {"train", "dev"}:
        raise ValueError("only train and dev data may be configured")
    for split, entry in config["data"].items():
        if set(entry) != {"path", "sha256", "rows"}:
            raise ValueError(f"{split} must declare exactly path, sha256, and rows")
        if (
            isinstance(entry["rows"], bool)
            or not isinstance(entry["rows"], int)
            or entry["rows"] < 1
        ):
            raise ValueError(f"{split} requires a positive row count")
        if Path(entry["path"]).name != f"{split}.jsonl":
            raise ValueError(f"{split} must point to {split}.jsonl")
        if not re.fullmatch(r"[a-f0-9]{64}", str(entry["sha256"])):
            raise ValueError(f"{split} requires an exact SHA-256")
    settings = config.get("training", {})
    if (
        not TRAINING_FIELDS <= set(settings)
        or set(settings) - TRAINING_FIELDS - TRAINING_METADATA_FIELDS
    ):
        raise ValueError("training fields differ from the bounded experiment contract")
    if (
        settings.get("hyperparameter_runs", 1) != 1
        or settings.get("resume_or_continuation_allowed", False) is not False
    ):
        raise ValueError("only one fresh fit is authorized; no resume or sweep")
    for name in ("epochs", "batch_size", "max_steps", "max_wall_seconds", "max_length", "seed"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["epochs"] > 2 or settings["max_steps"] > 3200:
        raise ValueError("the fit is limited to two epochs and 3200 steps")
    if settings["max_wall_seconds"] > 3600:
        raise ValueError("the fit wall limit must not exceed 3600 seconds")
    rate = settings["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate < 1:
        raise ValueError("learning_rate must be finite and between zero and one")
    if settings["binary_loss_weight"] != 0.0:
        raise ValueError("only three-class weighted cross entropy is authorized")
    if settings["truncation_side"] not in {"left", "right"}:
        raise ValueError("invalid truncation side")
    if settings["dialogue_policy"] not in DIALOGUE_POLICIES:
        raise ValueError("unknown dialogue policy")
    calibration = config.get("calibration", {})
    if not isinstance(calibration, dict) or (
        "method" in calibration and calibration["method"] != SERVING_CALIBRATION_VERSION
    ):
        raise ValueError("unknown calibration method; omit method for legacy calibration")
    outputs = config.get("outputs", {})
    if not {"checkpoint", "report"} <= set(outputs) or set(outputs) - {
        "checkpoint",
        "report",
        "evaluation",
        "predictions",
    }:
        raise ValueError("checkpoint and report outputs are required")
    paths = artifact_paths(config)
    output_paths = [
        *paths.values(),
        *(Path(outputs[key]) for key in ("evaluation", "predictions") if key in outputs),
    ]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("experiment output paths must be distinct")


def ensure_fresh_outputs(paths: dict[str, Path], *, preflight: bool) -> None:
    names = ("checkpoint", "report", "trainer", "progress", "predictions")
    if preflight:
        names += ("preflight",)
    occupied = [str(paths[name]) for name in names if paths[name].exists()]
    if occupied:
        raise FileExistsError(f"refusing to resume or overwrite prior outputs: {occupied}")


def load_rows(config: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    rows_by_split: dict[str, list[dict]] = {}
    metadata: dict[str, dict] = {}
    for split in ("train", "dev"):
        entry = config["data"][split]
        payload = Path(entry["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"{split} data differs from its frozen hash")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{split} data is empty")
        if len(rows) != entry["rows"]:
            raise ValueError(f"{split} row count differs from the frozen count")
        identifiers: list[str] = []
        for row in rows:
            if not isinstance(row.get("text"), str) or row.get("label") not in LABELS:
                raise ValueError(f"{split} row has invalid text or verdict label")
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"{split} row has no stable ID")
            identifiers.append(row["id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{split} contains duplicate IDs")
        counts = Counter(row["label"] for row in rows)
        if set(counts) != set(LABELS):
            raise ValueError(f"{split} must contain all three verdict classes")
        rows_by_split[split] = rows
        metadata[split] = {
            **entry,
            "rows": len(rows),
            "labels": dict(counts),
            "sources": dict(Counter(str(row.get("source", "UNKNOWN")) for row in rows)),
            "families": len({row["family_id"] for row in rows if row.get("family_id")}),
            "auxiliary_rows_projected": sum("action_targets" in row for row in rows),
        }
    if {row["id"] for row in rows_by_split["train"]} & {row["id"] for row in rows_by_split["dev"]}:
        raise ValueError("train and dev IDs overlap")
    return rows_by_split, metadata


def verdict_rows(rows: list[dict]) -> list[dict]:
    """Drop auxiliary supervision in memory, leaving the pinned source bytes untouched."""
    projected = []
    for row in rows:
        value = {
            key: val
            for key, val in row.items()
            if key not in {"action_targets", "action_verdict_weight"}
        }
        value.setdefault("source", "UNKNOWN")
        value.setdefault("category", "UNKNOWN")
        projected.append(value)
    return projected


def step_bound(rows: int, settings: dict[str, Any]) -> int:
    return min(settings["max_steps"], math.ceil(rows / settings["batch_size"]) * settings["epochs"])


class BoundProgress(TrainerCallback):
    def __init__(self, path: Path, steps: int, seconds: int, started: float, clock=time.monotonic):
        self.path, self.steps, self.seconds = path, steps, seconds
        self.started, self.clock = started, clock
        self.stop_reason: str | None = None

    def record(self, state: Any, phase: str) -> None:
        write_json(
            self.path,
            {
                "phase": phase,
                "global_step": int(state.global_step),
                "effective_max_steps": self.steps,
                "max_wall_seconds": self.seconds,
                "wall_seconds": self.clock() - self.started,
                "stop_reason": self.stop_reason,
                "release_authorized": False,
            },
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.record(state, "training")

    def on_step_end(self, args, state, control, **kwargs):
        if self.clock() - self.started >= self.seconds:
            self.stop_reason = "wall_limit_reached"
            control.should_training_stop = True
        elif state.global_step >= self.steps:
            self.stop_reason = "step_bound_reached"
            control.should_training_stop = True
        if state.global_step % 25 == 0 or self.stop_reason:
            self.record(state, "training")

    def on_train_end(self, args, state, control, **kwargs):
        if self.stop_reason is None:
            self.stop_reason = (
                "step_bound_reached" if state.global_step >= self.steps else "stopped_early"
            )
        self.record(state, "fit_finished")


def source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/run_bounded_encoder.py",
        "training/train_encoder.py",
        "training/ettin_calibration.py",
        "benchmarks/eval_bounded_encoder.py",
        "src/scamguard/metrics.py",
        "src/scamguard/decision.py",
        "src/scamguard/model.py",
        "src/scamguard/scanner.py",
        "src/scamguard/signals.py",
        "src/scamguard/preprocessing.py",
    )
    return {name: file_sha256(root / name) for name in names}


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "device": "mps",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "accelerate", "numpy")
        },
        "mps_available": torch.backends.mps.is_available(),
    }


def verify_model_files(snapshot: Path, expected: dict[str, str]) -> dict[str, str]:
    actual = {name: file_sha256(snapshot / name) for name in sorted(expected)}
    if actual != expected:
        raise ValueError("cached base model/tokenizer files differ from their frozen hashes")
    return actual


def local_model_files(config: dict[str, Any]) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=config["model"]["repository"],
            revision=config["model"]["revision"],
            allow_patterns=sorted(BASE_FILES),
            local_files_only=True,
        )
    )
    return verify_model_files(snapshot, config["model"]["files_sha256"])


def calibrate_dev(rows: list[dict], logits: np.ndarray) -> tuple[dict, dict, list[dict]]:
    if logits.shape != (len(rows), 3) or not np.isfinite(logits).all():
        raise ValueError("development logits must be finite and exactly three-class")
    truth = np.array([LABEL_TO_ID[row["label"]] for row in rows])
    temperature = fit_temperature(logits, truth)
    binary_truth, scam_scores = binary_subset(rows, softmax(logits, temperature))
    threshold = choose_threshold(binary_truth, scam_scores, max_fpr=0.02)
    calibration = {
        "temperature": temperature,
        "scam_threshold": threshold,
        "safe_threshold": 0.20,
        "safe_threshold_semantics": "maximum_safe_risk",
        "labels": list(LABELS),
        "threshold_source": "dev SAFE/SCAM only",
        "safe_threshold_source": "fixed existing encoder policy; not fitted",
    }
    records = prediction_ledger_records(
        {"dev": rows}, {"dev": logits}, temperature, threshold, 0.20
    )
    for record, scores in zip(records, logits, strict=True):
        record["logits"] = scores.tolist()
    predicted = np.array([LABEL_TO_ID[row["calibrated_verdict"]] for row in records])
    metrics = report_slice(rows, logits, temperature, threshold)
    metrics["calibrated"] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "confusion": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        "labels": list(LABELS),
        "alert_composition": verdict_alert_metrics(truth, predicted),
    }
    return calibration, metrics, records


def calibrate_checkpoint_dev(
    config: dict[str, Any], model: Any, tokenizer: Any, rows: list[dict],
    trainer: Any, dataset: Any,
) -> tuple[dict, dict, list[dict]]:
    """Explicit serving calibration opt-in; old configs retain their original path."""
    settings = config["training"]
    if config.get("calibration", {}).get("method") == SERVING_CALIBRATION_VERSION:
        return calibrate_serving_dev(
            model, tokenizer, rows, settings["max_length"], settings["dialogue_policy"]
        )
    trainer.args.train_sampling_strategy = "random"
    logits = predict_in_dataset_order(trainer, dataset, rows)
    return calibrate_dev(rows, logits)


def run(
    config_path: Path, *, preflight: bool = False, metadata_preflight: bool = False,
) -> int:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    serving_calibration = (
        config.get("calibration", {}).get("method") == SERVING_CALIBRATION_VERSION
    )
    if preflight and metadata_preflight:
        raise ValueError("choose only one preflight mode")
    if preflight and serving_calibration:
        raise ValueError("serving-calibrated fits require --metadata-preflight, not --preflight")
    paths = artifact_paths(config)
    ensure_fresh_outputs(paths, preflight=preflight or metadata_preflight)
    rows, data_identity = load_rows(config)
    settings = config["training"]
    steps = step_bound(len(rows["train"]), settings)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    sources = source_identity()
    base_files = None
    if metadata_preflight or serving_calibration:
        # Offline file verification only: no MPS probing, model load, or forward pass.
        base_files = local_model_files(config)
        expected_preflight = {
            "status": "passed",
            "mode": "CPU_metadata_only",
            "config_sha256": config_sha256,
            "source_sha256": sources,
            "data": data_identity,
            "base_model_files_sha256": base_files,
            "effective_max_steps": steps,
            "model_forward_performed": False,
            "training_performed": False,
            "release_authorized": False,
        }
        if metadata_preflight:
            write_json(paths["preflight"], expected_preflight, exclusive=True)
            return 0
        # The same preflight path is exclusive for preparation, but its unchanged
        # receipt must already exist when the one serving-calibrated fit begins.
        if (
            not paths["preflight"].is_file()
            or json.loads(paths["preflight"].read_text()) != expected_preflight
        ):
            raise ValueError("serving fit requires its unchanged metadata preflight receipt")
    started = time.monotonic()
    result: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": "preflight" if preflight else "fresh_fixed_terminal_fit",
        "config": config,
        "config_sha256": config_sha256,
        "source_sha256": sources,
        "data": data_identity,
        "environment": environment(),
        "effective_max_steps": steps,
        "checkpoint_selection": "fixed terminal step; no validation selection or resume",
        "fitting_objective": "square-root class-weighted three-class cross entropy only",
        "other_splits_opened": False,
        "release_authorized": False,
        "training_completed": False,
        "stop_reason": None,
        "checkpoint": str(paths["checkpoint"]),
    }
    if serving_calibration:
        result["metadata_preflight"] = {
            "path": str(paths["preflight"]),
            "sha256": file_sha256(paths["preflight"]),
        }
    result["environment_sha256"] = hashlib.sha256(
        json.dumps(result["environment"], sort_keys=True).encode()
    ).hexdigest()
    destination = paths["preflight"] if preflight else paths["report"]
    if not preflight:
        write_json(paths["progress"], {"phase": "loading", "global_step": 0}, exclusive=True)
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("this local experiment requires MPS; CPU fallback is prohibited")
        set_seed(settings["seed"])
        result["base_model_files_sha256"] = (
            base_files if base_files is not None else local_model_files(config)
        )
        load_args = {
            "revision": config["model"]["revision"],
            "local_files_only": True,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["repository"], **load_args)
        tokenizer.truncation_side = settings["truncation_side"]
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model"]["repository"],
            **load_args,
            num_labels=3,
            id2label=dict(enumerate(LABELS)),
            label2id=LABEL_TO_ID,
            use_safetensors=False,
            weights_only=True,
        ).to("mps")
        result["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        if preflight:
            model.eval()
            text = prepare_model_text(SYNTHETIC_PREFLIGHT_TEXT, settings["dialogue_policy"])
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=settings["max_length"]
            )
            with torch.inference_mode():
                logits = model(**{name: value.to("mps") for name, value in encoded.items()}).logits
            torch.mps.synchronize()
            if logits.shape != (1, 3) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("preflight did not produce three finite logits")
            result.update(
                status="passed",
                stop_reason="synthetic_preflight_complete",
                synthetic_fixture_sha256=hashlib.sha256(text.encode()).hexdigest(),
                synthetic_tokens=int(encoded["input_ids"].shape[1]),
            )
            write_json(destination, result, exclusive=True)
            return 0

        projected = {split: verdict_rows(values) for split, values in rows.items()}
        datasets = {
            split: EncodedDataset(
                values,
                tokenizer,
                settings["max_length"],
                dialogue_policy=settings["dialogue_policy"],
            )
            for split, values in projected.items()
        }
        counts = Counter(row["label"] for row in rows["train"])
        weights = torch.tensor(
            [math.sqrt(len(rows["train"]) / (3 * counts[label])) for label in LABELS],
            dtype=torch.float32,
        )
        result["class_weights"] = weights.tolist()
        result["warmup_steps"] = math.ceil(steps * 0.08)
        progress = BoundProgress(paths["progress"], steps, settings["max_wall_seconds"], started)
        args = TrainingArguments(
            output_dir=str(paths["trainer"]),
            learning_rate=settings["learning_rate"],
            per_device_train_batch_size=settings["batch_size"],
            per_device_eval_batch_size=settings["batch_size"],
            gradient_accumulation_steps=1,
            num_train_epochs=settings["epochs"],
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
            train_sampling_strategy="group_by_length",
            seed=settings["seed"],
            data_seed=settings["seed"],
        )
        result["training_arguments"] = args.to_dict()
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=datasets["train"],
            data_collator=DataCollatorWithPadding(tokenizer),
            processing_class=tokenizer,
            class_weights=weights,
            binary_loss_weight=0.0,
            binary_positive_weight=1.0,
            callbacks=[progress],
        )
        if trainer.model.device.type != "mps" or args.world_size != 1:
            raise RuntimeError("bounded fit requires one local MPS process")
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
            result["status"] = "rejected_incomplete_fit"
            result["wall_seconds"] = time.monotonic() - started
            progress.record(trainer.state, result["status"])
            write_json(destination, result, exclusive=True)
            return 1
        trainer.save_model(paths["checkpoint"])
        tokenizer.save_pretrained(paths["checkpoint"])
        result["checkpoint_files_sha256"] = {
            path.name: file_sha256(path)
            for path in sorted(paths["checkpoint"].iterdir())
            if path.is_file()
        }
        result["checkpoint_model_sha256"] = result["checkpoint_files_sha256"]["model.safetensors"]

        progress.record(trainer.state, "development_calibration")
        calibration, metrics, records = calibrate_checkpoint_dev(
            config, model, tokenizer, projected["dev"], trainer, datasets["dev"]
        )
        calibration.update(
            model_id=config["experiment_id"],
            input_transform={
                "dialogue_policy": settings["dialogue_policy"],
                "truncation_side": settings["truncation_side"],
                "max_length": settings["max_length"],
            },
        )
        calibration["checkpoint_model_sha256"] = result["checkpoint_model_sha256"]
        calibration["dev_data_sha256"] = config["data"]["dev"]["sha256"]
        write_json(paths["checkpoint"] / "scamguard_calibration.json", calibration, exclusive=True)
        result["calibration_sha256"] = file_sha256(
            paths["checkpoint"] / "scamguard_calibration.json"
        )
        result["checkpoint_files_sha256"]["scamguard_calibration.json"] = result[
            "calibration_sha256"
        ]
        with paths["predictions"].open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        result.update(calibration=calibration, dev=metrics, status="completed")
        result["dev_gates"] = {
            "recall": metrics["binary_safety"]["scam_recall"] >= 0.97,
            "safe_fpr": metrics["binary_safety"]["false_positive_rate"] <= 0.02,
        }
        result["development_quality_passed"] = all(result["dev_gates"].values())
        result["development_macro_f1_stretch"] = {
            "target": 0.94, "actual": metrics["calibrated"]["macro_f1"], "required": False
        }
        result["predictions"] = {
            "path": str(paths["predictions"]),
            "sha256": file_sha256(paths["predictions"]),
            "rows": len(records),
            "contains_text": False,
        }
        for split, entry in config["data"].items():
            if file_sha256(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"{split} data changed during the experiment")
        result["wall_seconds"] = time.monotonic() - started
        progress.record(trainer.state, "completed")
        write_json(destination, result, exclusive=True)
        # Quality failure is reported, not hidden behind a process error: open
        # benchmarks can still evaluate this frozen checkpoint independently.
        return 0
    except BaseException as error:
        result.update(
            status="failed",
            stop_reason=type(error).__name__,
            error=str(error)[:1000],
            wall_seconds=time.monotonic() - started,
        )
        if not destination.exists():
            write_json(destination, result, exclusive=True)
        if not preflight:
            write_json(
                paths["progress"],
                {
                    "phase": "failed",
                    "stop_reason": type(error).__name__,
                    "error": str(error)[:1000],
                    "wall_seconds": time.monotonic() - started,
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument(
        "--metadata-preflight", action="store_true",
        help="Verify immutable inputs offline without loading a model; exclusive receipt.",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(run(
            args.config, preflight=args.preflight, metadata_preflight=args.metadata_preflight
        ))
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
