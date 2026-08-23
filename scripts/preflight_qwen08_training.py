#!/usr/bin/env python3
"""Run a no-update MPS backward pass through the pinned Qwen3.5-0.8B LoRA path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Final

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from scamguard.metrics import file_sha256

try:
    from training.train_qwen_lora import (
        LANGUAGE_LORA_TARGETS,
        completion_token_start,
        installed_transformers_revision,
        require_loaded_revision,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.train_qwen_lora import (
        LANGUAGE_LORA_TARGETS,
        completion_token_start,
        installed_transformers_revision,
        require_loaded_revision,
    )

BASE_MODEL: Final[str] = "Qwen/Qwen3.5-0.8B"
BASE_REVISION: Final[str] = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_REVISION: Final[str] = "0c92811846095910816a87aca50050d10c545270"
SEED: Final[int] = 20260820
SYSTEM_PROMPT: Final[str] = (
    "You are ScamGuard. Return exactly one JSON object with verdict, confidence, "
    "evidence, rationale, and recommended_action."
)


def snapshot_manifest(snapshot: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(snapshot.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not records:
        raise ValueError("resolved Qwen snapshot contains no files")
    return records


def manifest_sha256(records: list[dict[str, object]]) -> str:
    payload = "".join(
        f"{record['name']}\t{record['bytes']}\t{record['sha256']}\n" for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _training_example(processor: Any) -> tuple[dict[str, torch.Tensor], int]:
    message = "Your account is locked. Send the one-time code now to restore access."
    target = json.dumps(
        {
            "verdict": "SCAM",
            "confidence": 0.99,
            "evidence": ["Send the one-time code now"],
            "rationale": "Requests a one-time code through an untrusted message.",
            "recommended_action": "Do not share the code; contact the bank independently.",
        },
        separators=(",", ":"),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Classify this message:\n<message>{message}</message>",
        },
        {"role": "assistant", "content": target},
    ]
    prompt = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    complete = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = processor.tokenizer(
        complete,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    completion_start = completion_token_start(prompt, complete, offsets)
    labels = encoded["input_ids"].clone()
    labels[:, :completion_start] = -100
    supervised_tokens = int((labels != -100).sum().item())
    if supervised_tokens <= 0:
        raise ValueError("preflight example contains no supervised completion tokens")
    encoded["labels"] = labels
    return encoded, supervised_tokens


def run_preflight(*, output: Path, local_files_only: bool) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training preflight: {output}")
    if platform.machine() != "arm64":
        raise RuntimeError("Qwen MPS preflight requires a native arm64 Python environment")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Qwen MPS preflight requires torch.backends.mps.is_available()")
    torch.manual_seed(SEED)
    transformers_revision = installed_transformers_revision()
    if transformers_revision != TRANSFORMERS_REVISION:
        raise RuntimeError(
            "installed Transformers revision differs from the frozen training revision: "
            f"{transformers_revision!r}"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=BASE_MODEL,
            revision=BASE_REVISION,
            local_files_only=local_files_only,
        )
    ).resolve()
    if snapshot.name != BASE_REVISION:
        raise RuntimeError(
            f"resolved snapshot {snapshot.name!r} differs from revision {BASE_REVISION}"
        )
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
    if not trainable:
        raise RuntimeError("LoRA preflight resolved no trainable parameters")
    visual_trainable = [name for name, _parameter in trainable if "visual" in name.casefold()]
    if visual_trainable:
        raise RuntimeError(
            f"LoRA preflight unexpectedly targets visual tensors: {visual_trainable[:5]}"
        )
    missing_targets = [
        target
        for target in LANGUAGE_LORA_TARGETS
        if not any(f".{target}." in name for name, _parameter in trainable)
    ]
    if missing_targets:
        raise RuntimeError(f"LoRA target modules were not resolved: {missing_targets}")

    encoded, supervised_tokens = _training_example(processor)
    input_tokens = int(encoded["input_ids"].shape[-1])
    model.to("mps")
    batch = {name: tensor.to("mps") for name, tensor in encoded.items()}
    torch.mps.synchronize()
    backward_started = time.perf_counter()
    loss = model(**batch).loss
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError("Qwen LoRA preflight produced a non-finite loss")
    loss.backward()
    torch.mps.synchronize()
    backward_elapsed = time.perf_counter() - backward_started
    gradients = [
        parameter.grad
        for _name, parameter in trainable
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("Qwen LoRA preflight produced missing or non-finite gradients")

    repository = Path(__file__).resolve().parents[1]
    report = {
        "artifact_schema_version": 1,
        "measurement_kind": "qwen08_no_update_mps_training_preflight",
        "passed": True,
        "parameter_update_performed": False,
        "contains_training_or_audit_rows": False,
        "seed": SEED,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "snapshot_manifest_sha256": manifest_sha256(files),
        "snapshot_files": files,
        "transformers_revision": transformers_revision,
        "source_bindings": {
            "preflight_script_sha256": file_sha256(Path(__file__).resolve()),
            "training_launcher_sha256": file_sha256(
                repository / "training" / "train_qwen_lora.py"
            ),
            "pyproject_sha256": file_sha256(repository / "pyproject.toml"),
            "uv_lock_sha256": file_sha256(repository / "uv.lock"),
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LANGUAGE_LORA_TARGETS,
            "trainable_parameters": sum(parameter.numel() for _name, parameter in trainable),
            "trainable_tensors": len(trainable),
            "gradient_tensors": len(gradients),
            "visual_trainable_tensors": 0,
        },
        "probe": {
            "input_tokens": input_tokens,
            "supervised_tokens": supervised_tokens,
            "loss": float(loss.detach().cpu().item()),
            "forward_backward_seconds": backward_elapsed,
            "complete_seconds": time.perf_counter() - started,
            "message_sha256": hashlib.sha256(
                b"Your account is locked. Send the one-time code now to restore access."
            ).hexdigest(),
            "contains_message_text": False,
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
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/runs/qwen35-08b-training-preflight.json"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    report = run_preflight(output=args.output, local_files_only=args.local_files_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
