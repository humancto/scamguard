#!/usr/bin/env python3
"""Build a self-contained, hash-bound ScamGuard GGUF desktop runtime pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from scamguard.gguf_runtime import (
    FROZEN_PROMPT_PREFIX,
    FROZEN_PROMPT_SUFFIX,
    LABELS,
    PACK_MANIFEST_NAME,
    QWEN35_08B_PROCESSOR,
    QWEN35_08B_PROCESSOR_REVISION,
)
from scamguard.metrics import file_sha256
from scamguard.prompts import SYSTEM_PROMPT

PACK_SCHEMA_VERSION = 1


def prompt_fragments(processor: Any) -> tuple[str, str]:
    sentinel = "SCAMGUARD_RUNTIME_MESSAGE_SENTINEL"
    rendered = processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{sentinel}</message>",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    question = rendered + '{"verdict":"'
    marker = f"<message>{sentinel}"
    if question.count(marker) != 1:
        raise ValueError("Qwen chat template does not preserve the runtime message sentinel")
    prefix, suffix = question.split(marker, maxsplit=1)
    if not prefix or not suffix or not suffix.startswith("</message>"):
        raise ValueError("Qwen chat template produced an incompatible message boundary")
    return prefix, suffix


def normalized_calibration(source: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    score_cache = source.get("score_cache")
    if not isinstance(score_cache, dict):
        score_cache = {}
    sequence_bucket_size = source.get(
        "sequence_bucket_size", score_cache.get("sequence_bucket_size")
    )
    record = {
        "artifact_schema_version": 1,
        "backend_type": "qwen_gguf_verdict_likelihood",
        "labels": list(source.get("labels") or LABELS),
        "temperature": source.get("temperature"),
        "scam_threshold": source.get("scam_threshold"),
        "safe_threshold": source.get("safe_threshold"),
        "safe_threshold_semantics": source.get("safe_threshold_semantics"),
        "sequence_bucket_size": sequence_bucket_size,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "source_report_sha256": source_sha256,
    }
    if tuple(record["labels"]) != LABELS:
        raise ValueError("calibration source has an incompatible ordered label set")
    if record["safe_threshold_semantics"] != "minimum_safe_probability":
        raise ValueError("calibration source has incompatible SAFE-threshold semantics")
    if record["sequence_bucket_size"] != 64:
        raise ValueError("calibration source must bind the frozen 64-token bucket")
    for field in ("temperature", "scam_threshold", "safe_threshold"):
        if not isinstance(record[field], (int, float)):
            raise ValueError(f"calibration source lacks numeric {field}")
    return record


def macos_dependency_report(runner: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["otool", "-L", str(runner)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"cannot inspect native runner dependencies: {completed.stderr.strip()}")
    dependencies = [
        line.strip().split(" (", maxsplit=1)[0]
        for line in completed.stdout.splitlines()[1:]
        if line.strip()
    ]
    non_system = [
        dependency
        for dependency in dependencies
        if not dependency.startswith(("/usr/lib/", "/System/Library/"))
    ]
    if non_system:
        raise ValueError(f"native runner has non-system dependencies: {non_system}")
    return {
        "inspection": "otool-L",
        "dependencies": dependencies,
        "portable_system_dependencies_only": True,
    }


def linux_dependency_report(runner: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ldd", str(runner)],
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode or "not found" in output:
        raise ValueError(f"cannot verify native runner dependencies: {output.strip()}")
    dependencies = [line.strip() for line in output.splitlines() if line.strip()]
    non_system: list[str] = []
    for dependency in dependencies:
        resolved = dependency.split("=>", maxsplit=1)[-1].strip().split(" ", maxsplit=1)[0]
        if resolved.startswith("/") and not resolved.startswith(
            ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")
        ):
            non_system.append(resolved)
    if non_system:
        raise ValueError(f"native runner has non-system dependencies: {non_system}")
    return {
        "inspection": "ldd",
        "dependencies": dependencies,
        "portable_system_dependencies_only": True,
    }


def runner_dependency_report(runner: Path) -> dict[str, Any]:
    system = platform.system()
    if system == "Darwin":
        return macos_dependency_report(runner)
    if system == "Linux":
        return linux_dependency_report(runner)
    raise ValueError(f"portable GGUF runtime packs are not implemented for {system}")


def build_pack(
    *,
    model: Path,
    runner: Path,
    calibration_source: Path,
    output: Path,
    prompt_prefix: str,
    prompt_suffix: str,
    processor_repository: str,
    processor_revision: str,
    purpose: str,
    dependency_report: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GGUF runtime pack: {output}")
    for path in (model, runner, calibration_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if model.suffix.casefold() != ".gguf":
        raise ValueError("runtime-pack model must use the .gguf extension")
    if not runner.stat().st_mode & 0o111:
        raise PermissionError(f"GGUF runner is not executable: {runner}")
    if dependency_report.get("portable_system_dependencies_only") is not True:
        raise ValueError("GGUF runner portability has not been established")
    if purpose not in {"release_candidate", "upstream_base_control"}:
        raise ValueError("unsupported GGUF pack purpose")
    if (
        processor_repository != QWEN35_08B_PROCESSOR
        or processor_revision != QWEN35_08B_PROCESSOR_REVISION
        or prompt_prefix != FROZEN_PROMPT_PREFIX
        or prompt_suffix != FROZEN_PROMPT_SUFFIX
    ):
        raise ValueError("runtime pack differs from the frozen Qwen prompt identity")

    source_sha256 = file_sha256(calibration_source)
    source = json.loads(calibration_source.read_text(encoding="utf-8"))
    calibration = normalized_calibration(source, source_sha256)
    output.mkdir(parents=True)
    model_output = output / model.name
    runner_output = output / "scamguard-gguf-verdict"
    calibration_output = output / "scamguard_calibration.json"
    shutil.copy2(model, model_output)
    shutil.copy2(runner, runner_output)
    calibration_output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest: dict[str, Any] = {
        "artifact_schema_version": PACK_SCHEMA_VERSION,
        "backend_type": "qwen_gguf_verdict_likelihood",
        "purpose": purpose,
        "publication_authorized": False,
        "model": {
            "path": model_output.name,
            "sha256": file_sha256(model_output),
            "bytes": model_output.stat().st_size,
        },
        "runner": {
            "path": runner_output.name,
            "sha256": file_sha256(runner_output),
            "bytes": runner_output.stat().st_size,
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **dependency_report,
        },
        "calibration": {
            "path": calibration_output.name,
            "sha256": file_sha256(calibration_output),
            "bytes": calibration_output.stat().st_size,
            "source_report": str(calibration_source),
            "source_report_sha256": source_sha256,
        },
        "prompt": {
            "prefix": prompt_prefix,
            "message_open": "<message>",
            "suffix": prompt_suffix,
            "prefix_sha256": hashlib.sha256(prompt_prefix.encode()).hexdigest(),
            "suffix_sha256": hashlib.sha256(prompt_suffix.encode()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "processor_repository": processor_repository,
            "processor_revision": processor_revision,
        },
        "runtime": {
            "protocol_version": 2,
            "ctx_size": 640,
            "batch_size": 640,
            "ubatch_size": 128,
            "threads": 4,
            "n_gpu_layers": 99,
            "message_batch_size": 1,
            "candidate_batch_size": 3,
            "sequence_bucket_size": 64,
            "prefix_cache_enabled": True,
        },
    }
    manifest_path = output / PACK_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--processor", default=QWEN35_08B_PROCESSOR)
    parser.add_argument(
        "--processor-revision", default=QWEN35_08B_PROCESSOR_REVISION
    )
    parser.add_argument(
        "--purpose",
        choices=("release_candidate", "upstream_base_control"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.processor,
        revision=args.processor_revision,
        local_files_only=True,
    )
    prefix, suffix = prompt_fragments(processor)
    manifest = build_pack(
        model=args.model,
        runner=args.runner,
        calibration_source=args.calibration_source,
        output=args.output,
        prompt_prefix=prefix,
        prompt_suffix=suffix,
        processor_repository=args.processor,
        processor_revision=args.processor_revision,
        purpose=args.purpose,
        dependency_report=runner_dependency_report(args.runner),
    )
    print(
        json.dumps(
            {
                "pack": str(args.output),
                "model_bytes": manifest["model"]["bytes"],
                "runner_bytes": manifest["runner"]["bytes"],
                "publication_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
