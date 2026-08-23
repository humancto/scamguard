#!/usr/bin/env python3
"""Verify raw-trace-backed physical iOS and Android benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Final

try:
    from scamguard.metrics import file_sha256
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from scamguard.metrics import file_sha256

SCHEMA_VERSION: Final[int] = 1
PLATFORMS: Final[tuple[str, ...]] = ("iOS", "Android")
LABELS: Final[set[str]] = {"SAFE", "UNCERTAIN", "SCAM"}
SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
MIN_WARMUP_REQUESTS: Final[int] = 5
MIN_UNIQUE_ROWS_PER_DEVICE: Final[int] = 100
MIN_MEASURED_REQUESTS_PER_DEVICE: Final[int] = 100
MOBILE_ID_DOMAIN: Final[bytes] = b"scamguard-mobile-benchmark-id-v1\0"
MOBILE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"sgm-[0-9a-f]{32}")
SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "maximum_ms",
)


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "maximum_ms": max(values),
    }


def selected_ids_sha256(identifiers: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()


def mobile_sample_id(prediction_ledger_id: str) -> str:
    digest = hashlib.sha256(MOBILE_ID_DOMAIN + prediction_ledger_id.encode()).hexdigest()
    return f"sgm-{digest[:32]}"


def load_reference_predictions(path: Path) -> tuple[dict[str, str], list[str]]:
    references: dict[str, str] = {}
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    errors.append(f"prediction ledger line {line_number} is not an object")
                    continue
                identifier = str(value.get("id", "")).strip()
                verdict = str(value.get("calibrated_verdict", "")).strip().upper()
                if not identifier or verdict not in LABELS:
                    errors.append(
                        f"prediction ledger line {line_number} has invalid identity/verdict"
                    )
                    continue
                opaque_id = mobile_sample_id(identifier)
                if opaque_id in references:
                    errors.append(f"prediction ledger line {line_number} duplicates an ID")
                    continue
                references[opaque_id] = verdict
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"prediction ledger must be valid JSONL: {error}")
    if not references:
        errors.append("prediction ledger must contain at least one valid decision")
    return references, errors


def sampled_reference_sha256(reference_by_id: dict[str, str]) -> str:
    ledger = "\n".join(
        f"{identifier}\t{reference_by_id[identifier]}" for identifier in sorted(reference_by_id)
    )
    return hashlib.sha256(ledger.encode()).hexdigest()


def _check_summary(
    reported: dict[str, Any], computed: dict[str, float], name: str, errors: list[str]
) -> None:
    for field, expected in computed.items():
        actual = reported.get(field)
        if not _positive_number(actual):
            errors.append(f"{name}.{field} must be finite and positive")
        elif not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-6):
            errors.append(f"{name}.{field} differs from raw timing samples")


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_run(
    run: dict[str, Any],
    *,
    index: int,
    rows: int,
    repetitions: int,
    expected_ids_sha256: str,
    expected_package_sha256: str | None,
    expected_reference_verdicts: dict[str, str] | None,
    errors: list[str],
) -> tuple[set[str], dict[str, float], int] | None:
    prefix = f"runs[{index}]"
    platform_name = run.get("platform")
    if platform_name not in PLATFORMS:
        errors.append(f"{prefix}.platform must be iOS or Android")
    if run.get("physical_device") is not True:
        errors.append(f"{prefix}.physical_device must be true")
    if run.get("simulator") is not False:
        errors.append(f"{prefix}.simulator must be false")
    if not _valid_utc_timestamp(run.get("measured_at_utc")):
        errors.append(f"{prefix}.measured_at_utc must be an ISO-8601 UTC timestamp")
    device = _mapping(run.get("device"), f"{prefix}.device", errors)
    for field in (
        "manufacturer",
        "model",
        "hardware_identifier",
        "architecture",
        "form_factor",
        "os_name",
        "os_version",
        "thermal_state_before",
        "thermal_state_after",
    ):
        if not str(device.get(field, "")).strip():
            errors.append(f"{prefix}.device.{field} must be recorded")
    if device.get("form_factor") != "phone":
        errors.append(f"{prefix}.device.form_factor must equal phone")
    expected_os = "iOS" if platform_name == "iOS" else "Android"
    if device.get("os_name") != expected_os:
        errors.append(f"{prefix}.device.os_name must equal {expected_os}")
    runtime = _mapping(run.get("runtime"), f"{prefix}.runtime", errors)
    if not str(runtime.get("backend", "")).strip():
        errors.append(f"{prefix}.runtime.backend must be recorded")
    if not str(runtime.get("runtime_revision", "")).strip():
        errors.append(f"{prefix}.runtime.runtime_revision must be recorded")
    if not str(runtime.get("accelerator", "")).strip():
        errors.append(f"{prefix}.runtime.accelerator must be recorded")
    if runtime.get("offline") is not True:
        errors.append(f"{prefix}.runtime.offline must be true")
    if runtime.get("protocol_version") != 2:
        errors.append(f"{prefix}.runtime.protocol_version must equal 2")
    if runtime.get("prefix_cache_enabled") is not True:
        errors.append(f"{prefix}.runtime.prefix_cache_enabled must be true")
    if (
        not isinstance(runtime.get("threads"), int)
        or isinstance(runtime.get("threads"), bool)
        or runtime.get("threads", 0) <= 0
    ):
        errors.append(f"{prefix}.runtime.threads must be a positive integer")
    package_sha256 = str(runtime.get("runtime_package_sha256", ""))
    if not SHA256_PATTERN.fullmatch(package_sha256):
        errors.append(f"{prefix}.runtime.runtime_package_sha256 is invalid")
    elif expected_package_sha256 is not None and package_sha256 != expected_package_sha256:
        errors.append(f"{prefix}.runtime package differs from release artifact")
    if run.get("latency_unit") != "ms":
        errors.append(f"{prefix}.latency_unit must equal ms")
    if run.get("measurement_scope") != "complete_local_tokenization_to_verdict":
        errors.append(
            f"{prefix}.measurement_scope must equal complete_local_tokenization_to_verdict"
        )
    if not str(run.get("monotonic_clock", "")).strip():
        errors.append(f"{prefix}.monotonic_clock must be recorded")
    warmup = run.get("warmup_requests")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < MIN_WARMUP_REQUESTS:
        errors.append(f"{prefix}.warmup_requests must be at least {MIN_WARMUP_REQUESTS}")
    peak_memory = run.get("peak_memory_bytes")
    if not isinstance(peak_memory, int) or isinstance(peak_memory, bool) or peak_memory <= 0:
        errors.append(f"{prefix}.peak_memory_bytes must be a positive integer")
        peak_memory = 0
    if not _positive_number(run.get("startup_ms")):
        errors.append(f"{prefix}.startup_ms must be finite and positive")

    samples = run.get("samples")
    if not isinstance(samples, list):
        errors.append(f"{prefix}.samples must be an array")
        return None
    expected_requests = rows * repetitions
    if len(samples) != expected_requests or len(samples) < MIN_MEASURED_REQUESTS_PER_DEVICE:
        errors.append(
            f"{prefix}.samples must contain rows*repetitions and at least "
            f"{MIN_MEASURED_REQUESTS_PER_DEVICE} requests"
        )
    identifiers: set[str] = set()
    pairs: set[tuple[str, int]] = set()
    reference_by_id: dict[str, str] = {}
    timings: list[float] = []
    for sample_index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"{prefix}.samples[{sample_index}]", errors)
        if "text" in sample or "message" in sample:
            errors.append(f"{prefix}.samples[{sample_index}] must not contain message text")
        identifier = str(sample.get("id", "")).strip()
        repetition = sample.get("repetition")
        verdict = str(sample.get("verdict", "")).strip().upper()
        reference = str(sample.get("reference_verdict", "")).strip().upper()
        elapsed = sample.get("elapsed_ms")
        if not identifier:
            errors.append(f"{prefix}.samples[{sample_index}].id must be recorded")
        elif not MOBILE_ID_PATTERN.fullmatch(identifier):
            errors.append(f"{prefix}.samples[{sample_index}].id must be an opaque mobile ID")
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition < 0
            or repetition >= repetitions
        ):
            errors.append(f"{prefix}.samples[{sample_index}].repetition is invalid")
            repetition = -1
        if verdict not in LABELS or reference not in LABELS:
            errors.append(f"{prefix}.samples[{sample_index}] has an invalid verdict")
        elif verdict != reference:
            errors.append(f"{prefix}.samples[{sample_index}] differs from reference verdict")
        if expected_reference_verdicts is not None:
            expected_reference = expected_reference_verdicts.get(identifier)
            if expected_reference is None:
                errors.append(f"{prefix}.samples[{sample_index}] is absent from prediction ledger")
            elif reference != expected_reference:
                errors.append(
                    f"{prefix}.samples[{sample_index}] reference differs from prediction ledger"
                )
        if not _positive_number(elapsed):
            errors.append(f"{prefix}.samples[{sample_index}].elapsed_ms must be positive")
        else:
            timings.append(float(elapsed))
        if not isinstance(sample.get("escalated"), bool):
            errors.append(f"{prefix}.samples[{sample_index}].escalated must be boolean")
        elif sample["escalated"] is True and sample.get("specialist_prefix_reused") is not True:
            errors.append(f"{prefix}.samples[{sample_index}] must reuse specialist prefix")
        pair = (identifier, repetition)
        if pair in pairs:
            errors.append(f"{prefix}.samples[{sample_index}] duplicates an id/repetition pair")
        pairs.add(pair)
        identifiers.add(identifier)
        prior_reference = reference_by_id.setdefault(identifier, reference)
        if prior_reference != reference:
            errors.append(
                f"{prefix}.samples[{sample_index}] changes reference verdict by repetition"
            )
    if len(identifiers) != rows:
        errors.append(f"{prefix} unique sample IDs differ from corpus rows")
    if selected_ids_sha256(identifiers) != expected_ids_sha256:
        errors.append(f"{prefix} selected IDs differ from corpus binding")
    if timings:
        computed = summarize(timings)
        _check_summary(
            _mapping(run.get("summary"), f"{prefix}.summary", errors),
            computed,
            f"{prefix}.summary",
            errors,
        )
    else:
        computed = {field: 0.0 for field in SUMMARY_FIELDS}
    reported_summary = _mapping(run.get("summary"), f"{prefix}.summary", errors)
    if reported_summary.get("requests") != len(samples):
        errors.append(f"{prefix}.summary.requests differs from raw timing samples")
    if run.get("sampled_reference_ledger_sha256") != sampled_reference_sha256(reference_by_id):
        errors.append(f"{prefix}.sampled_reference_ledger_sha256 differs from samples")
    return identifiers, computed, int(peak_memory)


def validate_mobile_benchmark(
    report: dict[str, Any],
    *,
    model_sha256: str | None = None,
    calibration_sha256: str | None = None,
    quantized_quality_sha256: str | None = None,
    source_prediction_ledger_sha256: str | None = None,
    runtime_package_sha256: dict[str, str] | None = None,
    reference_verdicts: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if report.get("artifact_schema_version") != SCHEMA_VERSION:
        errors.append(f"artifact_schema_version must equal {SCHEMA_VERSION}")
    if report.get("measurement_kind") != "physical_mobile_device_matrix":
        errors.append("measurement_kind must equal physical_mobile_device_matrix")
    if report.get("contains_message_text") is not False:
        errors.append("contains_message_text must be false")
    bindings = {
        "model_sha256": model_sha256,
        "calibration_sha256": calibration_sha256,
        "quantized_quality_report_sha256": quantized_quality_sha256,
        "source_prediction_ledger_sha256": source_prediction_ledger_sha256,
    }
    for field, expected in bindings.items():
        actual = str(report.get(field, ""))
        if not SHA256_PATTERN.fullmatch(actual):
            errors.append(f"{field} must be a lowercase SHA-256")
        elif expected is not None and actual != expected:
            errors.append(f"{field} differs from release evidence")
    corpus = _mapping(report.get("corpus"), "corpus", errors)
    rows = corpus.get("rows")
    repetitions = corpus.get("repetitions")
    ids_hash = str(corpus.get("selected_ids_sha256", ""))
    if (
        not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows < MIN_UNIQUE_ROWS_PER_DEVICE
    ):
        errors.append(
            f"corpus.rows must be at least {MIN_UNIQUE_ROWS_PER_DEVICE} unique messages"
        )
        rows = 0
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions <= 0:
        errors.append("corpus.repetitions must be a positive integer")
        repetitions = 0
    if not SHA256_PATTERN.fullmatch(ids_hash):
        errors.append("corpus.selected_ids_sha256 must be a lowercase SHA-256")
    if corpus.get("contains_message_text") is not False:
        errors.append("corpus.contains_message_text must be false")

    runs = report.get("runs")
    if not isinstance(runs, list):
        errors.append("runs must be an array")
        runs = []
    platforms = [run.get("platform") for run in runs if isinstance(run, dict)]
    if sorted(platforms) != sorted(PLATFORMS):
        errors.append("runs must contain exactly one physical iOS and one physical Android run")
    computed_runs: list[tuple[str, dict[str, float], int, int]] = []
    identity_sets: list[set[str]] = []
    expected_packages = runtime_package_sha256 or {}
    if rows and repetitions and SHA256_PATTERN.fullmatch(ids_hash):
        for index, raw_run in enumerate(runs):
            run = _mapping(raw_run, f"runs[{index}]", errors)
            result = _validate_run(
                run,
                index=index,
                rows=rows,
                repetitions=repetitions,
                expected_ids_sha256=ids_hash,
                expected_package_sha256=expected_packages.get(str(run.get("platform"))),
                expected_reference_verdicts=reference_verdicts,
                errors=errors,
            )
            if result is not None:
                identifiers, run_summary, peak_memory = result
                identity_sets.append(identifiers)
                computed_runs.append(
                    (str(run.get("platform")), run_summary, peak_memory, rows * repetitions)
                )
    if len(identity_sets) == 2 and identity_sets[0] != identity_sets[1]:
        errors.append("iOS and Android runs must measure the same selected IDs")

    computed_summary: dict[str, Any] = {}
    if len(computed_runs) == 2:
        computed_summary = {
            "device_matrix": "; ".join(
                f"{run.get('platform')} {run.get('device', {}).get('model')}"
                for run in runs
                if isinstance(run, dict) and isinstance(run.get("device"), dict)
            ),
            "devices": 2,
            "samples": sum(item[3] for item in computed_runs),
            "p50_ms": max(item[1]["p50_ms"] for item in computed_runs),
            "p95_ms": max(item[1]["p95_ms"] for item in computed_runs),
            "p99_ms": max(item[1]["p99_ms"] for item in computed_runs),
            "maximum_ms": max(item[1]["maximum_ms"] for item in computed_runs),
            "peak_memory_bytes": max(item[2] for item in computed_runs),
        }
        summary = _mapping(report.get("summary"), "summary", errors)
        for field, expected in computed_summary.items():
            actual = summary.get(field)
            if isinstance(expected, float):
                if not _positive_number(actual) or not math.isclose(
                    float(actual), expected, rel_tol=1e-9, abs_tol=1e-6
                ):
                    errors.append(f"summary.{field} differs from device runs")
            elif actual != expected:
                errors.append(f"summary.{field} differs from device runs")
    if report.get("release_gate_passed") is not True:
        errors.append("release_gate_passed must be true")
    return errors, computed_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--quantized-quality", type=Path, required=True)
    parser.add_argument("--prediction-ledger", type=Path, required=True)
    parser.add_argument("--ios-runtime-package", type=Path, required=True)
    parser.add_argument("--android-runtime-package", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    quantized = json.loads(args.quantized_quality.read_text(encoding="utf-8"))
    ledger = quantized.get("prediction_ledger")
    source_ledger_sha256 = ledger.get("sha256") if isinstance(ledger, dict) else None
    references, reference_errors = load_reference_predictions(args.prediction_ledger)
    if source_ledger_sha256 != file_sha256(args.prediction_ledger):
        reference_errors.append("prediction ledger file differs from quantized quality report")
    if isinstance(ledger, dict) and ledger.get("examples") != len(references):
        reference_errors.append("prediction ledger row count differs from quantized quality report")
    errors, computed = validate_mobile_benchmark(
        report,
        model_sha256=file_sha256(args.gguf),
        calibration_sha256=file_sha256(args.calibration),
        quantized_quality_sha256=file_sha256(args.quantized_quality),
        source_prediction_ledger_sha256=source_ledger_sha256,
        runtime_package_sha256={
            "iOS": file_sha256(args.ios_runtime_package),
            "Android": file_sha256(args.android_runtime_package),
        },
        reference_verdicts=references,
    )
    errors = reference_errors + errors
    print(
        json.dumps({"valid": not errors, "computed_summary": computed, "errors": errors}, indent=2)
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
