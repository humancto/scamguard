#!/usr/bin/env python3
"""Fail closed unless a ScamGuard Qwen3.5-0.8B release is fully evidenced."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ALLOWED_QUANTIZATIONS = {"Q4_K_M", "Q5_K_M", "Q8_0"}
ALLOWED_ROLES = {"fast_path", "routed_specialist"}
REQUIRED_INTERNAL_GATES = 39
REQUIRED_ARTIFACT_ROLES = {"merged_model", "gguf_model", "tokenizer"}
REQUIRED_REPORT_ROLES = {
    "data_manifest",
    "mobile_benchmark",
    "model_card",
    "quality",
    "quantized_quality",
    "runtime",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _require_true(section: dict[str, Any], field: str, errors: list[str]) -> None:
    if section.get(field) is not True:
        errors.append(f"{field} must be true")


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _validate_runtime_device(
    runtime: dict[str, Any], name: str, errors: list[str]
) -> dict[str, Any]:
    device = _mapping(runtime.get(name), f"runtime.{name}", errors)
    _require_true(device, "measured", errors)
    if not str(device.get("device", "")).strip():
        errors.append(f"runtime.{name}.device must be recorded")
    for field in (
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "maximum_ms",
        "peak_memory_bytes",
        "samples",
    ):
        if not _positive_number(device.get(field)):
            errors.append(f"runtime.{name}.{field} must be positive")
    if (
        _positive_number(device.get("p50_ms"))
        and _positive_number(device.get("p95_ms"))
        and device["p50_ms"] > device["p95_ms"]
    ):
        errors.append(f"runtime.{name}.p50_ms cannot exceed p95_ms")
    if (
        _positive_number(device.get("p95_ms"))
        and _positive_number(device.get("p99_ms"))
        and device["p95_ms"] > device["p99_ms"]
    ):
        errors.append(f"runtime.{name}.p95_ms cannot exceed p99_ms")
    if (
        _positive_number(device.get("p99_ms"))
        and _positive_number(device.get("maximum_ms"))
        and device["p99_ms"] > device["maximum_ms"]
    ):
        errors.append(f"runtime.{name}.p99_ms cannot exceed maximum_ms")
    return device


def _validate_evidence_files(
    entries: Any,
    section_name: str,
    required_roles: set[str],
    repo_root: Path,
    errors: list[str],
) -> None:
    if not isinstance(entries, list) or not entries:
        errors.append(f"{section_name} must be a non-empty list")
        return

    roles: set[str] = set()
    paths: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"{section_name}[{index}]", errors)
        role = str(entry.get("role", "")).strip()
        relative_path = str(entry.get("path", "")).strip()
        expected_hash = str(entry.get("sha256", "")).strip().lower()
        expected_bytes = entry.get("size_bytes")
        if not role:
            errors.append(f"{section_name}[{index}].role must be recorded")
        else:
            roles.add(role)
        if not relative_path:
            errors.append(f"{section_name}[{index}].path must be recorded")
            continue
        if relative_path in paths:
            errors.append(f"{section_name}[{index}].path is duplicated: {relative_path}")
        paths.add(relative_path)
        declared = Path(relative_path)
        if declared.is_absolute():
            errors.append(f"{section_name}[{index}].path must be repository-relative")
            continue
        path = (repo_root / declared).resolve()
        if not path.is_relative_to(repo_root):
            errors.append(f"{section_name}[{index}].path escapes the repository")
            continue
        if not path.is_file():
            errors.append(f"{section_name}[{index}].path is not a file: {relative_path}")
            continue
        if not SHA256_PATTERN.fullmatch(expected_hash):
            errors.append(f"{section_name}[{index}].sha256 must be a lowercase SHA-256")
        elif file_sha256(path) != expected_hash:
            errors.append(f"{section_name}[{index}] SHA-256 mismatch: {relative_path}")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
            errors.append(f"{section_name}[{index}].size_bytes must be an integer")
        elif path.stat().st_size != expected_bytes:
            errors.append(f"{section_name}[{index}] size mismatch: {relative_path}")

    missing_roles = required_roles - roles
    if missing_roles:
        errors.append(f"{section_name} missing required roles: {', '.join(sorted(missing_roles))}")


def validate_release_manifest(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    """Return every release-blocking error in deterministic order."""

    errors: list[str] = []
    repo_root = repo_root.resolve()
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if manifest.get("publication_status") != "approved":
        errors.append("publication_status must equal approved")

    repository = _mapping(manifest.get("repository"), "repository", errors)
    model_id = str(repository.get("model_id", "")).strip()
    gguf_id = str(repository.get("gguf_id", "")).strip()
    if "/" not in model_id:
        errors.append("repository.model_id must be a namespaced Hugging Face repository ID")
    if "/" not in gguf_id:
        errors.append("repository.gguf_id must be a namespaced Hugging Face repository ID")
    if model_id == gguf_id:
        errors.append("repository.model_id and repository.gguf_id must be different")
    if repository.get("visibility") != "public":
        errors.append("repository.visibility must equal public")

    model = _mapping(manifest.get("model"), "model", errors)
    if model.get("base_model") != BASE_MODEL:
        errors.append(f"model.base_model must equal {BASE_MODEL}")
    if model.get("base_revision") != BASE_REVISION:
        errors.append(f"model.base_revision must equal {BASE_REVISION}")
    if model.get("base_license") != "Apache-2.0":
        errors.append("model.base_license must equal Apache-2.0")
    if model.get("run_kind") != "full":
        errors.append("model.run_kind must equal full; smoke runs cannot be published")
    if model.get("deployment_role") not in ALLOWED_ROLES:
        errors.append("model.deployment_role must be fast_path or routed_specialist")
    if not str(model.get("experiment_id", "")).strip():
        errors.append("model.experiment_id must be recorded")
    for field in ("training_examples", "evaluation_examples"):
        if not isinstance(model.get(field), int) or isinstance(model.get(field), bool):
            errors.append(f"model.{field} must be an integer")
        elif model[field] <= 5:
            errors.append(f"model.{field} must exceed the five-row smoke-test size")

    quality = _mapping(manifest.get("quality"), "quality", errors)
    internal = _mapping(quality.get("internal_gates"), "quality.internal_gates", errors)
    passed = internal.get("passed")
    total = internal.get("total")
    if (
        not isinstance(passed, int)
        or isinstance(passed, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or passed != total
    ):
        errors.append("quality.internal_gates must record a positive passed == total")
    elif total != REQUIRED_INTERNAL_GATES:
        errors.append(
            f"quality.internal_gates.total must equal {REQUIRED_INTERNAL_GATES}"
        )
    for field in (
        "external_selection_passed",
        "human_label_audit_passed",
        "multilingual_claims_reviewed",
        "sealed_evaluation_passed",
    ):
        _require_true(quality, field, errors)

    quantization = _mapping(manifest.get("quantization"), "quantization", errors)
    if quantization.get("format") != "GGUF":
        errors.append("quantization.format must equal GGUF")
    if quantization.get("type") not in ALLOWED_QUANTIZATIONS:
        errors.append("quantization.type must be Q4_K_M, Q5_K_M, or Q8_0")
    for field in (
        "merge_equivalence_passed",
        "post_quantization_evaluation_passed",
        "frozen_calibration_reused",
    ):
        _require_true(quantization, field, errors)

    runtime = _mapping(manifest.get("runtime"), "runtime", errors)
    scoring_contract = _mapping(
        runtime.get("scoring_contract"), "runtime.scoring_contract", errors
    )
    if scoring_contract.get("message_batch_size") != 1:
        errors.append("runtime.scoring_contract.message_batch_size must equal 1")
    if scoring_contract.get("candidate_batch_size") != 3:
        errors.append("runtime.scoring_contract.candidate_batch_size must equal 3")
    if scoring_contract.get("sequence_bucket_size") != 64:
        errors.append("runtime.scoring_contract.sequence_bucket_size must equal 64")
    _require_true(scoring_contract, "exact_decision_parity", errors)
    _require_true(scoring_contract, "quantized_decision_parity", errors)
    desktop = _validate_runtime_device(runtime, "desktop", errors)
    _validate_runtime_device(runtime, "mobile", errors)
    role = model.get("deployment_role")
    if role == "fast_path" and _positive_number(desktop.get("p95_ms")):
        if desktop["p95_ms"] > 20:
            errors.append("fast_path desktop p95_ms must be at most 20")
    if role == "routed_specialist":
        routed = _mapping(runtime.get("routed"), "runtime.routed", errors)
        _require_true(routed, "measured", errors)
        escalation_rate = routed.get("escalation_rate")
        if (
            not isinstance(escalation_rate, (int, float))
            or isinstance(escalation_rate, bool)
            or not 0 <= escalation_rate <= 1
        ):
            errors.append("runtime.routed.escalation_rate must be between 0 and 1")
        for field in ("p50_ms", "p95_ms", "p99_ms", "maximum_ms", "escalated_p95_ms"):
            if not _positive_number(routed.get(field)):
                errors.append(f"runtime.routed.{field} must be positive")
        if _positive_number(routed.get("p95_ms")) and routed["p95_ms"] > 20:
            errors.append("runtime.routed.p95_ms must be at most 20")
        if (
            _positive_number(routed.get("escalated_p95_ms"))
            and routed["escalated_p95_ms"] >= 50
        ):
            errors.append("runtime.routed.escalated_p95_ms must be under 50")
        if (
            _positive_number(routed.get("p50_ms"))
            and _positive_number(routed.get("p95_ms"))
            and routed["p50_ms"] > routed["p95_ms"]
        ):
            errors.append("runtime.routed.p50_ms cannot exceed p95_ms")
        if (
            _positive_number(routed.get("p95_ms"))
            and _positive_number(routed.get("p99_ms"))
            and routed["p95_ms"] > routed["p99_ms"]
        ):
            errors.append("runtime.routed.p95_ms cannot exceed p99_ms")
        if (
            _positive_number(routed.get("p99_ms"))
            and _positive_number(routed.get("maximum_ms"))
            and routed["p99_ms"] > routed["maximum_ms"]
        ):
            errors.append("runtime.routed.p99_ms cannot exceed maximum_ms")

    governance = _mapping(manifest.get("governance"), "governance", errors)
    for field in (
        "base_license_notice_included",
        "data_redistribution_audit_passed",
        "pii_scan_passed",
        "secrets_scan_passed",
    ):
        _require_true(governance, field, errors)
    if governance.get("release_contains_training_rows") is not False:
        errors.append("governance.release_contains_training_rows must be false")
    if governance.get("direct_reddit_training_rows") is not False:
        errors.append("governance.direct_reddit_training_rows must be false")

    _validate_evidence_files(
        manifest.get("artifacts"), "artifacts", REQUIRED_ARTIFACT_ROLES, repo_root, errors
    )
    _validate_evidence_files(
        manifest.get("reports"), "reports", REQUIRED_REPORT_ROLES, repo_root, errors
    )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_release_manifest(manifest, args.repo_root)
    result = {
        "manifest": str(args.manifest),
        "publication_authorized": not errors,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
