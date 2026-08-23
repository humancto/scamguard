from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_mobile_benchmark import (
    load_reference_predictions,
    mobile_sample_id,
    sampled_reference_sha256,
    selected_ids_sha256,
    summarize,
    validate_mobile_benchmark,
)

HASHES = {
    "model": "a" * 64,
    "calibration": "b" * 64,
    "quality": "c" * 64,
    "ledger": "d" * 64,
    "iOS": "e" * 64,
    "Android": "f" * 64,
}


def valid_report() -> dict[str, object]:
    identifiers = [mobile_sample_id(f"canonical-row-{index:03d}") for index in range(100)]
    reference = {
        identifier: ("SAFE" if index % 3 == 0 else "SCAM")
        for index, identifier in enumerate(identifiers)
    }
    runs = []
    for platform_index, platform_name in enumerate(("iOS", "Android")):
        samples = []
        for repetition in range(1):
            for index, identifier in enumerate(identifiers):
                samples.append(
                    {
                        "id": identifier,
                        "repetition": repetition,
                        "elapsed_ms": 20.0 + platform_index * 5 + index / 10,
                        "verdict": reference[identifier],
                        "reference_verdict": reference[identifier],
                        "escalated": index % 10 == 0,
                        "specialist_prefix_reused": index % 10 == 0,
                    }
                )
        timings = [sample["elapsed_ms"] for sample in samples]
        run_summary = {"requests": len(samples), **summarize(timings)}
        runs.append(
            {
                "platform": platform_name,
                "physical_device": True,
                "simulator": False,
                "measured_at_utc": "2026-08-22T23:00:00Z",
                "device": {
                    "manufacturer": "Apple" if platform_name == "iOS" else "Google",
                    "model": "iPhone 16 Pro" if platform_name == "iOS" else "Pixel 9",
                    "hardware_identifier": "iPhone17,1" if platform_name == "iOS" else "tokay",
                    "architecture": "arm64",
                    "form_factor": "phone",
                    "os_name": platform_name,
                    "os_version": "18.6" if platform_name == "iOS" else "16",
                    "thermal_state_before": "nominal",
                    "thermal_state_after": "fair",
                },
                "runtime": {
                    "backend": "llama.cpp Metal" if platform_name == "iOS" else "llama.cpp CPU",
                    "runtime_revision": "521a64cd01979bb5b1a466152c576a9d809b068d",
                    "accelerator": "Metal" if platform_name == "iOS" else "CPU",
                    "runtime_package_sha256": HASHES[platform_name],
                    "offline": True,
                    "protocol_version": 2,
                    "prefix_cache_enabled": True,
                    "threads": 4,
                },
                "latency_unit": "ms",
                "measurement_scope": "complete_local_tokenization_to_verdict",
                "monotonic_clock": "continuous monotonic nanoseconds",
                "warmup_requests": 5,
                "startup_ms": 400.0 + platform_index * 50,
                "peak_memory_bytes": 700_000_000 + platform_index * 50_000_000,
                "sampled_reference_ledger_sha256": sampled_reference_sha256(reference),
                "samples": samples,
                "summary": run_summary,
            }
        )
    return {
        "artifact_schema_version": 1,
        "measurement_kind": "physical_mobile_device_matrix",
        "contains_message_text": False,
        "model_sha256": HASHES["model"],
        "calibration_sha256": HASHES["calibration"],
        "quantized_quality_report_sha256": HASHES["quality"],
        "source_prediction_ledger_sha256": HASHES["ledger"],
        "corpus": {
            "rows": len(identifiers),
            "repetitions": 1,
            "selected_ids_sha256": selected_ids_sha256(set(identifiers)),
            "contains_message_text": False,
        },
        "runs": runs,
        "summary": {
            "device_matrix": "iOS iPhone 16 Pro; Android Pixel 9",
            "devices": 2,
            "samples": 200,
            "p50_ms": runs[1]["summary"]["p50_ms"],  # type: ignore[index]
            "p95_ms": runs[1]["summary"]["p95_ms"],  # type: ignore[index]
            "p99_ms": runs[1]["summary"]["p99_ms"],  # type: ignore[index]
            "maximum_ms": runs[1]["summary"]["maximum_ms"],  # type: ignore[index]
            "peak_memory_bytes": 750_000_000,
        },
        "release_gate_passed": True,
    }


def validate(report: dict[str, object]) -> tuple[list[str], dict[str, object]]:
    references = {
        sample["id"]: sample["reference_verdict"]
        for sample in report["runs"][0]["samples"]  # type: ignore[index]
    }
    return validate_mobile_benchmark(
        report,
        model_sha256=HASHES["model"],
        calibration_sha256=HASHES["calibration"],
        quantized_quality_sha256=HASHES["quality"],
        source_prediction_ledger_sha256=HASHES["ledger"],
        runtime_package_sha256={"iOS": HASHES["iOS"], "Android": HASHES["Android"]},
        reference_verdicts=references,
    )


def test_valid_physical_device_matrix_recomputes_worst_case_summary() -> None:
    errors, computed = validate(valid_report())

    assert errors == []
    assert computed["device_matrix"] == "iOS iPhone 16 Pro; Android Pixel 9"
    assert computed["samples"] == 200
    assert computed["peak_memory_bytes"] == 750_000_000


def test_simulator_or_missing_platform_is_rejected() -> None:
    report = valid_report()
    report["runs"][0]["physical_device"] = False  # type: ignore[index]
    report["runs"][0]["simulator"] = True  # type: ignore[index]
    report["runs"][1]["platform"] = "iOS"  # type: ignore[index]

    errors, _ = validate(report)

    assert "runs must contain exactly one physical iOS and one physical Android run" in errors
    assert "runs[0].physical_device must be true" in errors
    assert "runs[0].simulator must be false" in errors


def test_raw_trace_tampering_and_verdict_drift_are_rejected() -> None:
    report = valid_report()
    sample = report["runs"][0]["samples"][0]  # type: ignore[index]
    sample["elapsed_ms"] = 999.0
    sample["verdict"] = "UNCERTAIN"
    sample["text"] = "must never appear"

    errors, _ = validate(report)

    assert any("must not contain message text" in error for error in errors)
    assert any("differs from reference verdict" in error for error in errors)
    assert any("summary.p50_ms differs from raw timing samples" in error for error in errors)
    assert any("summary.maximum_ms differs from raw timing samples" in error for error in errors)


def test_artifact_binding_and_prefix_reuse_are_required() -> None:
    report = valid_report()
    report["model_sha256"] = hashlib.sha256(b"other model").hexdigest()
    report["runs"][1]["runtime"]["runtime_package_sha256"] = "0" * 64  # type: ignore[index]
    report["runs"][1]["samples"][0]["specialist_prefix_reused"] = False  # type: ignore[index]

    errors, _ = validate(report)

    assert "model_sha256 differs from release evidence" in errors
    assert "runs[1].runtime package differs from release artifact" in errors
    assert "runs[1].samples[0] must reuse specialist prefix" in errors


def test_reference_verdicts_are_loaded_from_quantized_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "predictions.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"canonical-{index}",
                    "calibrated_verdict": "SAFE" if index == 0 else "SCAM",
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )

    references, errors = load_reference_predictions(ledger)

    assert errors == []
    assert references == {
        mobile_sample_id("canonical-0"): "SAFE",
        mobile_sample_id("canonical-1"): "SCAM",
    }


def test_self_consistent_but_forged_reference_is_rejected() -> None:
    report = valid_report()
    references = {
        sample["id"]: sample["reference_verdict"]
        for sample in report["runs"][0]["samples"]  # type: ignore[index]
    }
    forged_id = next(iter(references))
    forged = "UNCERTAIN" if references[forged_id] != "UNCERTAIN" else "SAFE"
    for run in report["runs"]:  # type: ignore[union-attr]
        for sample in run["samples"]:
            if sample["id"] == forged_id:
                sample["verdict"] = forged
                sample["reference_verdict"] = forged

    errors, _ = validate_mobile_benchmark(report, reference_verdicts=references)

    assert any("reference differs from prediction ledger" in error for error in errors)
