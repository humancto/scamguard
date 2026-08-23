#!/usr/bin/env python3
"""Stress the frozen Qwen3.5-0.8B LoRA microbatch geometry without updating weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.metrics import file_sha256

try:
    from scripts.preflight_qwen08_training import (
        BASE_MODEL,
        BASE_REVISION,
        SEED,
        TRANSFORMERS_REVISION,
        _training_example,
        manifest_sha256,
        snapshot_manifest,
    )
    from training.train_qwen_lora import (
        LANGUAGE_LORA_TARGETS,
        installed_transformers_revision,
        require_loaded_revision,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.preflight_qwen08_training import (
        BASE_MODEL,
        BASE_REVISION,
        SEED,
        TRANSFORMERS_REVISION,
        _training_example,
        manifest_sha256,
        snapshot_manifest,
    )
    from training.train_qwen_lora import (
        LANGUAGE_LORA_TARGETS,
        installed_transformers_revision,
        require_loaded_revision,
    )


def expanded_batch(
    encoded: dict[str, torch.Tensor], *, batch_size: int, sequence_length: int
) -> tuple[dict[str, torch.Tensor], int]:
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    source_ids = encoded["input_ids"][0]
    source_labels = encoded["labels"][0]
    supervised_source = source_labels[source_labels != -100]
    if source_ids.numel() == 0 or supervised_source.numel() == 0:
        raise ValueError("synthetic source example is empty or unsupervised")
    repeats = math.ceil(sequence_length / source_ids.numel())
    row = source_ids.repeat(repeats)[:sequence_length]
    input_ids = row.unsqueeze(0).repeat(batch_size, 1)
    labels = torch.full_like(input_ids, -100)
    supervised_tokens = min(int(supervised_source.numel()), sequence_length)
    labels[:, -supervised_tokens:] = input_ids[:, -supervised_tokens:]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }, supervised_tokens * batch_size


def run_preflight(
    *, output: Path, batch_size: int, sequence_length: int, local_files_only: bool
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite batch preflight: {output}")
    if platform.machine() != "arm64" or not torch.backends.mps.is_available():
        raise RuntimeError("Qwen batch preflight requires native arm64 MPS")
    torch.manual_seed(SEED)
    transformers_revision = installed_transformers_revision()
    if transformers_revision != TRANSFORMERS_REVISION:
        raise RuntimeError("installed Transformers commit differs from the frozen revision")

    snapshot = Path(
        snapshot_download(
            repo_id=BASE_MODEL,
            revision=BASE_REVISION,
            local_files_only=local_files_only,
        )
    ).resolve()
    if snapshot.name != BASE_REVISION:
        raise RuntimeError("resolved Qwen snapshot differs from the frozen revision")
    files = snapshot_manifest(snapshot)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    require_loaded_revision(
        requested=BASE_REVISION,
        loaded=getattr(model.config, "_commit_hash", None),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LANGUAGE_LORA_TARGETS,
            revision=BASE_REVISION,
        ),
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("visual" in name.casefold() for name, _parameter in trainable):
        raise RuntimeError("batch preflight LoRA scope is empty or enters the visual tower")

    source, _source_supervised = _training_example(processor)
    batch, supervised_tokens = expanded_batch(
        source,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    model.to("mps")
    device_batch = {name: tensor.to("mps") for name, tensor in batch.items()}
    torch.mps.empty_cache()
    torch.mps.synchronize()
    backward_started = time.perf_counter()
    loss = model(**device_batch).loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError("Qwen batch preflight produced a non-finite loss")
    loss.backward()
    torch.mps.synchronize()
    backward_elapsed = time.perf_counter() - backward_started
    gradients = [
        parameter.grad
        for _name, parameter in trainable
        if parameter.grad is not None
    ]
    if len(gradients) != len(trainable):
        raise RuntimeError("Qwen batch preflight did not populate every adapter gradient")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("Qwen batch preflight produced a non-finite adapter gradient")

    repository = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "artifact_schema_version": 1,
        "measurement_kind": "qwen08_no_update_mps_batch_geometry_preflight",
        "passed": True,
        "parameter_update_performed": False,
        "contains_training_or_audit_rows": False,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "transformers_revision": transformers_revision,
        "snapshot_manifest_sha256": manifest_sha256(files),
        "seed": SEED,
        "geometry": {
            "microbatch_size": batch_size,
            "sequence_length": sequence_length,
            "tokens_per_microbatch": batch_size * sequence_length,
            "gradient_accumulation": 1,
            "effective_batch_size": batch_size,
            "supervised_tokens": supervised_tokens,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "trainable_parameters": sum(parameter.numel() for _name, parameter in trainable),
            "trainable_tensors": len(trainable),
            "gradient_tensors": len(gradients),
            "visual_trainable_tensors": 0,
        },
        "probe": {
            "loss": float(loss.detach().cpu().item()),
            "forward_backward_seconds": backward_elapsed,
            "complete_seconds": time.perf_counter() - started,
        },
        "environment": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "mps_available": True,
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
        },
        "source_bindings": {
            "batch_preflight_sha256": file_sha256(Path(__file__).resolve()),
            "base_preflight_sha256": file_sha256(
                repository / "scripts" / "preflight_qwen08_training.py"
            ),
            "training_launcher_sha256": file_sha256(
                repository / "training" / "train_qwen_lora.py"
            ),
            "experiment_freezer_sha256": file_sha256(
                repository / "scripts" / "freeze_qwen08_full_experiment.py"
            ),
            "uv_lock_sha256": file_sha256(repository / "uv.lock"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=640)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/runs/qwen35-08b-batch-preflight.json"),
    )
    args = parser.parse_args()
    report = run_preflight(
        output=args.output,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
