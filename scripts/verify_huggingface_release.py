#!/usr/bin/env python3
"""Fail closed unless a ScamGuard Qwen3.5-0.8B release is fully evidenced."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LLAMA_CPP_REVISION = "521a64cd01979bb5b1a466152c576a9d809b068d"
ALLOWED_QUANTIZATIONS = {"Q4_K_M", "Q5_K_M", "Q8_0"}
ALLOWED_ROLES = {"fast_path", "routed_specialist"}
REQUIRED_INTERNAL_GATES = 39
REQUIRED_ARTIFACT_ROLES = {
    "merged_model",
    "gguf_model",
    "runtime_binary",
    "tokenizer",
}
REQUIRED_REPORT_ROLES = {
    "bf16_quality",
    "data_manifest",
    "mobile_benchmark",
    "model_card",
    "quality_gates",
    "quantized_quality",
    "routed_trace",
    "routed_runtime",
    "runtime_source",
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
) -> dict[str, Path]:
    role_paths: dict[str, Path] = {}
    if not isinstance(entries, list) or not entries:
        errors.append(f"{section_name} must be a non-empty list")
        return role_paths

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
        elif role in roles:
            errors.append(f"{section_name}[{index}].role is duplicated: {role}")
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
        if role and role not in role_paths:
            role_paths[role] = path
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
    return role_paths


def _json_report(path: Path | None, role: str, errors: list[str]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"reports role {role} must be valid JSON")
        return {}
    return _mapping(value, f"reports role {role}", errors)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _trace_summary(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(record[field]) for record in records]
    return {
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "maximum_ms": max(values),
    }


def _validate_routed_trace(
    routed: dict[str, Any], trace_path: Path | None, errors: list[str]
) -> None:
    if trace_path is None:
        return
    trace_ledger = _mapping(
        routed.get("trace_ledger"), "routed_runtime.trace_ledger", errors
    )
    if trace_ledger.get("sha256") != file_sha256(trace_path):
        errors.append("routed runtime trace SHA-256 differs from routed_trace evidence")
    if trace_ledger.get("contains_message_text") is not False:
        errors.append("routed runtime trace must declare contains_message_text false")
    records: list[dict[str, Any]] = []
    try:
        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                records.append(value)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"routed_trace must be valid JSONL: {error}")
        return
    if not records:
        errors.append("routed_trace must contain at least one request")
        return
    if trace_ledger.get("rows") != len(records):
        errors.append("routed runtime trace row count differs from routed_trace evidence")
    required = {
        "id",
        "repetition",
        "escalated",
        "router_ms",
        "specialist_ms",
        "routing_overhead_ms",
        "total_ms",
    }
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            errors.append(
                f"routed_trace[{index}] missing fields: {', '.join(sorted(missing))}"
            )
            return
        if "text" in record:
            errors.append(f"routed_trace[{index}] must not contain message text")
            return
        if not str(record["id"]).strip():
            errors.append(f"routed_trace[{index}].id must be recorded")
            return
        if not isinstance(record["repetition"], int) or isinstance(
            record["repetition"], bool
        ):
            errors.append(f"routed_trace[{index}].repetition must be an integer")
            return
        if not isinstance(record["escalated"], bool):
            errors.append(f"routed_trace[{index}].escalated must be boolean")
            return
        for field in ("router_ms", "specialist_ms", "routing_overhead_ms", "total_ms"):
            value = record[field]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                errors.append(f"routed_trace[{index}].{field} must be finite and non-negative")
                return
        component_total = (
            float(record["router_ms"])
            + float(record["specialist_ms"])
            + float(record["routing_overhead_ms"])
        )
        if not math.isclose(
            float(record["total_ms"]), component_total, rel_tol=1e-9, abs_tol=1e-6
        ):
            errors.append(f"routed_trace[{index}] component timings do not sum to total_ms")
            return

    latency = _mapping(routed.get("latency"), "routed_runtime.latency", errors)
    overall = _mapping(latency.get("overall"), "routed_runtime.latency.overall", errors)
    escalated_records = [record for record in records if record["escalated"] is True]
    if not escalated_records:
        errors.append("routed_trace must contain escalated requests")
        return
    escalated = _mapping(
        latency.get("escalated_path_total"),
        "routed_runtime.latency.escalated_path_total",
        errors,
    )
    bindings = (
        ("overall", overall, _trace_summary(records, "total_ms")),
        ("escalated", escalated, _trace_summary(escalated_records, "total_ms")),
    )
    for name, reported, computed in bindings:
        for field, value in computed.items():
            reported_value = reported.get(field)
            if (
                not isinstance(reported_value, (int, float))
                or isinstance(reported_value, bool)
                or not math.isfinite(reported_value)
            ):
                errors.append(f"routed runtime {name} {field} must be finite")
            elif not math.isclose(
                float(reported_value), value, rel_tol=1e-9, abs_tol=1e-6
            ):
                errors.append(
                    f"routed runtime {name} {field} differs from routed_trace evidence"
                )
    if overall.get("samples") != len(records):
        errors.append("routed runtime overall sample count differs from routed_trace evidence")
    if escalated.get("samples") != len(escalated_records):
        errors.append("routed runtime escalated sample count differs from routed_trace evidence")


def _validate_report_contents(
    manifest: dict[str, Any],
    artifact_paths: dict[str, Path],
    report_paths: dict[str, Path],
    errors: list[str],
) -> None:
    bf16 = _json_report(report_paths.get("bf16_quality"), "bf16_quality", errors)
    gates = _json_report(report_paths.get("quality_gates"), "quality_gates", errors)
    quantized = _json_report(
        report_paths.get("quantized_quality"), "quantized_quality", errors
    )
    routed = _json_report(report_paths.get("routed_runtime"), "routed_runtime", errors)
    if routed.get("measurement_mode") != "interleaved_persistent_per_request":
        errors.append(
            "routed runtime measurement_mode must equal interleaved_persistent_per_request"
        )
    _validate_routed_trace(routed, report_paths.get("routed_trace"), errors)

    score_cache = _mapping(bf16.get("score_cache"), "bf16_quality.score_cache", errors)
    expected_shape = {
        "message_batch_size": 1,
        "candidate_batch_size": 3,
        "sequence_bucket_size": 64,
    }
    for field, expected in expected_shape.items():
        if score_cache.get(field) != expected:
            errors.append(f"bf16_quality.score_cache.{field} must equal {expected}")
    bf16_ledger = _mapping(
        bf16.get("prediction_ledger"), "bf16_quality.prediction_ledger", errors
    )
    bf16_ledger_sha256 = str(bf16_ledger.get("sha256", ""))
    if not SHA256_PATTERN.fullmatch(bf16_ledger_sha256):
        errors.append("bf16_quality prediction ledger SHA-256 is invalid")

    if gates.get("quality_status") != "passed":
        errors.append("quality_gates.quality_status must equal passed")
    if gates.get("passed_gates") != REQUIRED_INTERNAL_GATES:
        errors.append(f"quality_gates.passed_gates must equal {REQUIRED_INTERNAL_GATES}")
    if gates.get("total_gates") != REQUIRED_INTERNAL_GATES:
        errors.append(f"quality_gates.total_gates must equal {REQUIRED_INTERNAL_GATES}")
    if gates.get("failed_gates") != []:
        errors.append("quality_gates.failed_gates must be empty")
    if gates.get("quantization_authorized") is not True:
        errors.append("quality_gates.quantization_authorized must be true")

    parity = _mapping(
        quantized.get("quantization_parity"),
        "quantized_quality.quantization_parity",
        errors,
    )
    if parity.get("reference_sha256") != bf16_ledger_sha256:
        errors.append("quantized quality does not reference the frozen BF16 ledger")
    if parity.get("exact_calibrated_verdict_parity") is not True:
        errors.append("quantized exact calibrated-verdict parity must be true")
    if parity.get("release_gate_passed") is not True:
        errors.append("quantized decision-parity release gate must pass")
    test_gates = _mapping(
        quantized.get("test_gates"), "quantized_quality.test_gates", errors
    )
    for field in ("recall", "fpr", "core_category_recall", "macro_f1_stretch"):
        if test_gates.get(field) is not True:
            errors.append(f"quantized_quality.test_gates.{field} must be true")
    gguf_path = artifact_paths.get("gguf_model")
    if gguf_path and quantized.get("model_sha256") != file_sha256(gguf_path):
        errors.append("quantized quality model SHA-256 differs from the GGUF artifact")

    specialist = _mapping(routed.get("specialist"), "routed_runtime.specialist", errors)
    quantization = _mapping(manifest.get("quantization"), "quantization", errors)
    if specialist.get("quantization") != quantization.get("type"):
        errors.append("routed runtime quantization differs from the selected GGUF type")
    bf16_path = report_paths.get("bf16_quality")
    if bf16_path and specialist.get("calibration_report_sha256") != file_sha256(bf16_path):
        errors.append("routed runtime calibration report differs from BF16 quality evidence")
    quantized_path = report_paths.get("quantized_quality")
    if quantized_path and specialist.get("quantized_quality_report_sha256") != file_sha256(
        quantized_path
    ):
        errors.append("routed runtime quantized report differs from quantized quality evidence")
    for section in ("frozen_quality_scoring", "runtime_scoring"):
        scoring = _mapping(
            specialist.get(section), f"routed_runtime.specialist.{section}", errors
        )
        for field, expected in expected_shape.items():
            if scoring.get(field) != expected:
                errors.append(
                    f"routed_runtime.specialist.{section}.{field} must equal {expected}"
                )
    runtime_parity = _mapping(routed.get("parity"), "routed_runtime.parity", errors)
    if runtime_parity.get("release_gate_passed") is not True:
        errors.append("routed runtime exact decision parity must pass")
    latency = _mapping(routed.get("latency"), "routed_runtime.latency", errors)
    latency_gates = _mapping(latency.get("gates"), "routed_runtime.latency.gates", errors)
    if latency_gates.get("overall_p95_under_20_ms") is not True:
        errors.append("routed runtime overall p95 gate must pass")
    if latency_gates.get("escalated_path_p95_under_50_ms") is not True:
        errors.append("routed runtime escalated-path p95 gate must pass")
    native_runtime = _mapping(
        specialist.get("native_runtime"), "routed_runtime.specialist.native_runtime", errors
    )
    runtime_binary = artifact_paths.get("runtime_binary")
    if runtime_binary and native_runtime.get("runner_sha256") != file_sha256(runtime_binary):
        errors.append("routed native runner SHA-256 differs from runtime_binary artifact")
    if gguf_path and native_runtime.get("model_sha256") != file_sha256(gguf_path):
        errors.append("routed native model SHA-256 differs from GGUF artifact")
    if native_runtime.get("protocol_version") != 1:
        errors.append("routed native runtime protocol_version must equal 1")
    for field, expected in expected_shape.items():
        if native_runtime.get(field) != expected:
            errors.append(f"routed native runtime {field} must equal {expected}")
    environment = _mapping(
        routed.get("environment"), "routed_runtime.environment", errors
    )
    if environment.get("llama_cpp_revision") != LLAMA_CPP_REVISION:
        errors.append(f"routed llama.cpp revision must equal {LLAMA_CPP_REVISION}")
    runtime_source = report_paths.get("runtime_source")
    if runtime_source and environment.get("runner_source_sha256") != file_sha256(
        runtime_source
    ):
        errors.append("routed runner source SHA-256 differs from runtime_source evidence")
    routed_manifest = _mapping(
        _mapping(manifest.get("runtime"), "runtime", errors).get("routed"),
        "runtime.routed",
        errors,
    )
    overall = _mapping(latency.get("overall"), "routed_runtime.latency.overall", errors)
    escalated = _mapping(
        latency.get("escalated_path_total"),
        "routed_runtime.latency.escalated_path_total",
        errors,
    )
    metric_bindings = {
        "escalation_rate": latency.get("escalation_rate"),
        "p50_ms": overall.get("p50_ms"),
        "p95_ms": overall.get("p95_ms"),
        "p99_ms": overall.get("p99_ms"),
        "maximum_ms": overall.get("maximum_ms"),
        "escalated_p95_ms": escalated.get("p95_ms"),
        "peak_memory_bytes": routed.get("process_peak_rss_bytes"),
    }
    for field, evidence_value in metric_bindings.items():
        if routed_manifest.get(field) != evidence_value:
            errors.append(f"runtime.routed.{field} differs from routed runtime evidence")


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
        for field in (
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "maximum_ms",
            "escalated_p95_ms",
            "peak_memory_bytes",
        ):
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

    artifact_paths = _validate_evidence_files(
        manifest.get("artifacts"), "artifacts", REQUIRED_ARTIFACT_ROLES, repo_root, errors
    )
    report_paths = _validate_evidence_files(
        manifest.get("reports"), "reports", REQUIRED_REPORT_ROLES, repo_root, errors
    )
    _validate_report_contents(manifest, artifact_paths, report_paths, errors)
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
