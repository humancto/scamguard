#!/usr/bin/env python3
"""Train a deployable Qwen LoRA on the exact verdict-branch decision."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from training.qwen_branch import (
        LABELS,
        BranchCollator,
        BranchDataset,
        branch_focal_kl_loss,
        load_teacher_cache,
    )
    from training.train_qwen_lora import (
        LANGUAGE_LORA_TARGETS,
        adapter_identity,
        experiment_config_errors,
        final_eval_metrics,
        installed_transformers_revision,
        require_loaded_revision,
        sha256,
    )
except ModuleNotFoundError:  # Direct execution: python training/train_qwen_branch_lora.py
    from qwen_branch import (
        LABELS,
        BranchCollator,
        BranchDataset,
        branch_focal_kl_loss,
        load_teacher_cache,
    )
    from train_qwen_lora import (
        LANGUAGE_LORA_TARGETS,
        adapter_identity,
        experiment_config_errors,
        final_eval_metrics,
        installed_transformers_revision,
        require_loaded_revision,
        sha256,
    )


def branch_experiment_errors(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    errors = experiment_config_errors(args, config)
    objective = config.get("objective")
    expected = {
        "type": "branch_token_focal_kl_v1",
        "labels": list(LABELS),
        "class_weights": [float(value) for value in args.class_weights],
        "focal_gamma": args.focal_gamma,
        "kl_weight": args.kl_weight,
        "kl_temperature": args.kl_temperature,
        "deployment_head_added": False,
    }
    if not isinstance(objective, dict):
        errors.append("objective declaration is missing")
        return errors
    for key, value in expected.items():
        if objective.get(key) != value:
            errors.append(f"objective.{key}: config {objective.get(key)!r}, command {value!r}")
    cache = objective.get("teacher_cache")
    if not isinstance(cache, dict):
        errors.append("objective.teacher_cache declaration is missing")
        return errors
    identities = {
        args.teacher_cache: cache.get("sha256"),
        args.teacher_cache_manifest: cache.get("manifest_sha256"),
    }
    for path, expected_hash in identities.items():
        if not path.is_file():
            errors.append(f"missing frozen teacher-cache artifact: {path}")
        elif sha256(path) != expected_hash:
            errors.append(f"teacher-cache hash mismatch: {path}")
    if cache.get("path") != str(args.teacher_cache):
        errors.append("objective.teacher_cache.path differs from command")
    if cache.get("manifest_path") != str(args.teacher_cache_manifest):
        errors.append("objective.teacher_cache.manifest_path differs from command")
    if args.teacher_cache_manifest.is_file():
        try:
            manifest = json.loads(args.teacher_cache_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("teacher-cache manifest is invalid JSON")
        else:
            if manifest.get("contains_message_text") is not False:
                errors.append("teacher cache must be text-free")
            if manifest.get("base_model_revision") != args.revision:
                errors.append("teacher cache uses a different base revision")
            if manifest.get("adapter") != config.get("initial_adapter"):
                errors.append("teacher cache uses a different initial adapter")
            manifest_data = manifest.get("data", {})
            for split in ("train", "dev"):
                path = args.data / f"{split}.jsonl"
                if manifest_data.get(split, {}).get("sha256") != sha256(path):
                    errors.append(f"teacher cache uses different {split} data")
    return errors


class BranchTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: list[float],
        focal_gamma: float,
        kl_weight: float,
        kl_temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.branch_class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.focal_gamma = focal_gamma
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        del num_items_in_batch
        batch = dict(inputs)
        batch.pop("sample_ids", None)
        branch_tokens = batch.pop("branch_token_ids")
        targets = batch.pop("targets")
        teacher_logits = batch.pop("teacher_logits")
        outputs = model(**batch, logits_to_keep=1)
        vocabulary_logits = outputs.logits[:, -1, :]
        student_logits = vocabulary_logits.gather(1, branch_tokens)
        loss, _parts = branch_focal_kl_loss(
            student_logits,
            targets,
            teacher_logits,
            class_weights=self.branch_class_weights,
            focal_gamma=self.focal_gamma,
            kl_weight=self.kl_weight,
            kl_temperature=self.kl_temperature,
        )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--teacher-cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--class-weights", type=float, nargs=3, default=(1.0, 3.0, 1.0))
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=5.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("random", "group_by_length"),
        default="group_by_length",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the frozen config and artifact hashes, then exit before loading weights.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    mismatches = branch_experiment_errors(args, config)
    if mismatches:
        raise RuntimeError("experiment config preflight failed:\n" + "\n".join(mismatches))
    if args.preflight_only:
        print("branch experiment preflight passed")
        return
    set_seed(args.seed)
    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("--require-mps was set but MPS is unavailable")
    print(f"training accelerator: {'mps' if mps_available else 'cpu'}")

    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if mps_available else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    base.config.use_cache = False
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()
    base_revision = getattr(base.config, "_commit_hash", None)
    require_loaded_revision(requested=args.revision, loaded=base_revision)
    model = PeftModel.from_pretrained(base, args.initial_adapter, is_trainable=True)
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    visual = [name for name, _parameter in trainable if "visual" in name.casefold()]
    if visual:
        raise RuntimeError(f"text-only adapter unexpectedly targets visual modules: {visual[:5]}")
    model.print_trainable_parameters()

    cache = load_teacher_cache(args.teacher_cache)
    train = BranchDataset(
        args.data / "train.jsonl",
        processor,
        args.max_length,
        split="train",
        teacher_cache=cache,
    )
    dev = BranchDataset(
        args.data / "dev.jsonl",
        processor,
        args.max_length,
        split="dev",
        teacher_cache=cache,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output.parent / (args.output.name + "-trainer")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        warmup_steps=0.05,
        weight_decay=0.01,
        eval_strategy="no" if args.skip_eval else "epoch",
        save_strategy="no" if args.skip_eval else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not args.skip_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        train_sampling_strategy=args.sampling_strategy,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = BranchTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=None if args.skip_eval else dev,
        data_collator=BranchCollator(processor.tokenizer.pad_token_id),
        processing_class=processor.tokenizer,
        class_weights=[float(value) for value in args.class_weights],
        focal_gamma=args.focal_gamma,
        kl_weight=args.kl_weight,
        kl_temperature=args.kl_temperature,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    receipt = {
        "base_model": args.model,
        "base_model_revision": base_revision,
        "initial_adapter": adapter_identity(args.initial_adapter),
        "experiment_id": config.get("experiment_id"),
        "experiment_config": {
            "path": str(args.experiment_config),
            "sha256": sha256(args.experiment_config),
            "run_kind": config.get("run_kind"),
        },
        "objective": {
            "type": "branch_token_focal_kl_v1",
            "labels": list(LABELS),
            "class_weights": [float(value) for value in args.class_weights],
            "focal_gamma": args.focal_gamma,
            "kl_weight": args.kl_weight,
            "kl_temperature": args.kl_temperature,
            "deployment_head_added": False,
            "teacher_cache": {
                "path": str(args.teacher_cache),
                "sha256": sha256(args.teacher_cache),
                "manifest_path": str(args.teacher_cache_manifest),
                "manifest_sha256": sha256(args.teacher_cache_manifest),
            },
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "sampling_strategy": args.sampling_strategy,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trainer_eval": not args.skip_eval,
        "mps_required": args.require_mps,
        "local_files_only": args.local_files_only,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "data": {
            "train_path": str(args.data / "train.jsonl"),
            "train_sha256": sha256(args.data / "train.jsonl"),
            "dev_path": str(args.data / "dev.jsonl"),
            "dev_sha256": sha256(args.data / "dev.jsonl"),
        },
        "trainable_parameters": sum(parameter.numel() for _name, parameter in trainable),
        "trainable_tensors": len(trainable),
        "lora_target_modules": LANGUAGE_LORA_TARGETS,
        "metrics": result.metrics,
        "eval_metrics": final_eval_metrics(trainer.state.log_history),
        "training_path": "Transformers+PEFT verdict-branch focal/KL (no deployment head)",
        "environment": {
            "python_arch": platform.machine(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "transformers_revision": installed_transformers_revision(),
            "peft": importlib.metadata.version("peft"),
            "mps_available": mps_available,
        },
    }
    (args.output / "training_run.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
