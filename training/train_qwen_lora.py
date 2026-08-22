#!/usr/bin/env python3
"""LoRA fine-tune Qwen3.5 through Transformers/PEFT, avoiding current MLX adapter bugs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

LANGUAGE_LORA_TARGETS = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def experiment_config_errors(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    """Bind command-line training parameters to a frozen experiment declaration."""

    errors: list[str] = []
    expected = {
        "base_model": args.model,
        "base_model_revision": args.revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "sampling": args.sampling_strategy,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
    }
    for field, actual in expected.items():
        if config.get(field) != actual:
            errors.append(f"{field}: config {config.get(field)!r}, command {actual!r}")
    if "trainer_eval" in config and bool(config["trainer_eval"]) == args.skip_eval:
        errors.append(
            f"trainer_eval: config {config['trainer_eval']!r}, command {not args.skip_eval!r}"
        )
    checkpoint_output = config.get("checkpoint_output")
    if checkpoint_output is not None and Path(str(checkpoint_output)) != args.output:
        errors.append(
            f"checkpoint_output: config {checkpoint_output!r}, command {str(args.output)!r}"
        )

    lora = config.get("lora")
    expected_lora = {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": LANGUAGE_LORA_TARGETS,
        "scope": "language tower only; asserted at runtime",
    }
    if lora != expected_lora:
        errors.append("lora declaration differs from the runtime adapter contract")

    data = config.get("data")
    if not isinstance(data, dict):
        errors.append("data declaration is missing")
        return errors
    processed = args.data.parent
    declared_directory = data.get("processed_directory")
    if declared_directory is not None and Path(str(declared_directory)) != processed:
        errors.append(
            f"processed_directory: config {declared_directory!r}, command {str(processed)!r}"
        )
    identities = {
        processed / "manifest.json": data.get("manifest_sha256"),
        args.data / "train.jsonl": data.get("train_jsonl_sha256"),
        args.data / "dev.jsonl": data.get("dev_jsonl_sha256"),
    }
    for path, expected_hash in identities.items():
        if not path.is_file():
            errors.append(f"missing frozen data artifact: {path}")
        elif sha256(path) != expected_hash:
            errors.append(f"data hash mismatch: {path}")
    counts = {
        "train_examples": args.data / "train.jsonl",
        "dev_examples": args.data / "dev.jsonl",
    }
    for field, path in counts.items():
        if path.is_file() and line_count(path) != data.get(field):
            errors.append(
                f"{field}: config {data.get(field)!r}, data {line_count(path)!r}"
            )
    if config.get("run_kind") == "full":
        if data.get("schema_version") != 24:
            errors.append("full run requires data.schema_version 24")
        token_audit = data.get("token_length_audit")
        if not isinstance(token_audit, dict):
            errors.append("full run requires a token_length_audit declaration")
        else:
            report_path = Path(str(token_audit.get("report_path", "")))
            report_sha256 = token_audit.get("report_sha256")
            if not report_path.is_file():
                errors.append(f"missing frozen token audit: {report_path}")
            elif sha256(report_path) != report_sha256:
                errors.append(f"token audit hash mismatch: {report_path}")
            if token_audit.get("full_over_max_length") != 0:
                errors.append("full run token audit reports truncated examples")
            minimum_supervised = token_audit.get("minimum_supervised_tokens")
            if (
                not isinstance(minimum_supervised, int)
                or isinstance(minimum_supervised, bool)
                or minimum_supervised <= 0
            ):
                errors.append("full run token audit lacks positive supervised-token coverage")
        evidence_audit = data.get("evidence_audit")
        if not isinstance(evidence_audit, dict) or evidence_audit.get("coverage") != 1.0:
            errors.append("full run requires complete verbatim-evidence coverage")
        label_audit = data.get("label_audit")
        if not isinstance(label_audit, dict):
            errors.append("full run requires an independent label-audit declaration")
        else:
            label_report = Path(str(label_audit.get("report_path", "")))
            if not label_report.is_file():
                errors.append(f"missing frozen label audit: {label_report}")
            elif sha256(label_report) != label_audit.get("report_sha256"):
                errors.append(f"label audit hash mismatch: {label_report}")
            if label_audit.get("release_gate_passed") is not True:
                errors.append("full run label audit release gate is not passed")
            if label_audit.get("data_manifest_sha256") != data.get("manifest_sha256"):
                errors.append("full run label audit is bound to a different data manifest")
    return errors


def completion_token_start(
    prompt: str,
    complete: str,
    offsets: list[tuple[int, int]] | list[list[int]],
) -> int:
    """Return the first token wholly inside the completion-only character range.

    Tokenizing ``prompt`` separately is unsafe at a BPE boundary: a tokenizer may merge the
    prompt's final character with the completion's first character. Mask that crossing token and
    supervise only tokens whose complete-string offset begins at or after the exact boundary.
    """

    if not complete.startswith(prompt):
        raise ValueError("chat-template prompt is not an exact string prefix of the completion")
    boundary = len(prompt)
    for index, (start, end) in enumerate(offsets):
        if end > start and start >= boundary:
            return index
    raise ValueError("truncated example has no token wholly inside the assistant completion")


class ChatDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path, processor: Any, max_length: int) -> None:
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self.rows[index]["messages"]
        prompt = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        complete = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.processor.tokenizer(
            complete,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"][0]
        completion_start = completion_token_start(prompt, complete, offsets)
        labels = input_ids.clone()
        labels[:completion_start] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels,
        }


class CompletionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        def padded(key: str, value: int) -> torch.Tensor:
            rows = []
            for feature in features:
                tensor = feature[key]
                padding = torch.full((max_length - len(tensor),), value, dtype=tensor.dtype)
                rows.append(torch.cat((tensor, padding)))
            return torch.stack(rows)

        return {
            "input_ids": padded("input_ids", self.pad_token_id),
            "attention_mask": padded("attention_mask", 0),
            "labels": padded("labels", -100),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help="Verify model, hyperparameters, data identities, and output before loading weights.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/qwen_sft"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/checkpoints/qwen35-08b-lora")
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade extra compute for lower activation memory.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("random", "group_by_length"),
        default="group_by_length",
        help=(
            "Length grouping reduces dynamic-padding work. Random remains available "
            "as a deterministic comparison strategy."
        ),
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Fail before loading weights when Apple Metal is not visible.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    experiment_config: dict[str, Any] | None = None
    if args.experiment_config is not None:
        experiment_config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
        mismatches = experiment_config_errors(args, experiment_config)
        if mismatches:
            raise RuntimeError("experiment config preflight failed:\n" + "\n".join(mismatches))
    set_seed(args.seed)

    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError(
            "--require-mps was set, but torch.backends.mps.is_available() is false. "
            "Run outside a restricted sandbox or choose an explicit CPU workflow."
        )
    print(f"training accelerator: {'mps' if mps_available else 'cpu'}")

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16 if mps_available else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    base_model_revision = getattr(model.config, "_commit_hash", None)
    if args.revision and base_model_revision and args.revision != base_model_revision:
        raise RuntimeError(
            f"loaded base revision {base_model_revision} differs from requested {args.revision}"
        )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LANGUAGE_LORA_TARGETS,
        revision=args.revision,
    )
    model = get_peft_model(model, lora)
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    visual_trainable = [name for name, _ in trainable if "visual" in name.casefold()]
    if visual_trainable:
        raise RuntimeError(
            f"text-only adapter unexpectedly targets visual modules: {visual_trainable[:5]}"
        )
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    model.print_trainable_parameters()

    train = ChatDataset(args.data / "train.jsonl", processor, args.max_length)
    dev = ChatDataset(args.data / "dev.jsonl", processor, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output.parent / (args.output.name + "-trainer")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        # Transformers main consolidated warmup_ratio into warmup_steps; a
        # float in [0, 1) is interpreted as a fraction of total update steps.
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=None if args.skip_eval else dev,
        data_collator=CompletionCollator(processor.tokenizer.pad_token_id),
        # The multimodal processor advertises pixel_values first, which makes
        # Trainer's length grouper inspect the wrong field for text-only SFT.
        processing_class=processor.tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    (args.output / "training_run.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "base_model_revision": base_model_revision,
                "experiment_id": (
                    experiment_config.get("experiment_id") if experiment_config else None
                ),
                "experiment_config": (
                    {
                        "path": str(args.experiment_config),
                        "sha256": sha256(args.experiment_config),
                        "run_kind": experiment_config.get("run_kind"),
                    }
                    if experiment_config is not None and args.experiment_config is not None
                    else None
                ),
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
                "evaluation_policy": (
                    "trainer development loss at epoch end"
                    if not args.skip_eval
                    else "dedicated frozen-slice evaluation after training"
                ),
                "mps_required": args.require_mps,
                "train_examples": len(train),
                "dev_examples": len(dev),
                "data": {
                    "train_path": str(args.data / "train.jsonl"),
                    "train_sha256": sha256(args.data / "train.jsonl"),
                    "dev_path": str(args.data / "dev.jsonl"),
                    "dev_sha256": sha256(args.data / "dev.jsonl"),
                },
                "trainable_parameters": trainable_parameters,
                "trainable_tensors": len(trainable),
                "lora_target_modules": LANGUAGE_LORA_TARGETS,
                "metrics": result.metrics,
                "training_path": "Transformers+PEFT (not MLX)",
                "environment": {
                    "python_arch": platform.machine(),
                    "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "peft": importlib.metadata.version("peft"),
                    "mps_available": mps_available,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
