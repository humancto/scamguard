from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_huggingface_release import validate_release_manifest


def evidence(path: Path, role: str, root: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def valid_manifest(tmp_path: Path) -> dict[str, object]:
    artifacts = []
    artifact_paths: dict[str, Path] = {}
    for role in ("merged_model", "gguf_model", "runtime_binary", "tokenizer"):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(f"artifact:{role}".encode())
        artifact_paths[role] = path
        artifacts.append(evidence(path, role, tmp_path))

    report_paths: dict[str, Path] = {}
    for role in ("data_manifest", "mobile_benchmark", "model_card", "runtime_source"):
        path = tmp_path / f"{role}.json"
        path.write_text(f'{{"report":"{role}"}}', encoding="utf-8")
        report_paths[role] = path
    routed_trace = tmp_path / "routed_runtime.traces.jsonl"
    trace_records = []
    for index in range(100):
        escalated = index >= 95
        if escalated:
            router_ms, specialist_ms, overhead_ms = 8.0, 36.9, 0.1
        elif index >= 90:
            router_ms, specialist_ms, overhead_ms = 14.9, 0.0, 0.1
        else:
            router_ms, specialist_ms, overhead_ms = 7.9, 0.0, 0.1
        trace_records.append(
            {
                "id": f"row-{index}",
                "repetition": 0,
                "escalated": escalated,
                "router_ms": router_ms,
                "specialist_ms": specialist_ms,
                "specialist_prefix_reused": escalated,
                "specialist_prefix_tokens": 100 if escalated else 0,
                "routing_overhead_ms": overhead_ms,
                "total_ms": router_ms + specialist_ms + overhead_ms,
            }
        )
    routed_trace.write_text(
        "".join(json.dumps(record) + "\n" for record in trace_records),
        encoding="utf-8",
    )
    report_paths["routed_trace"] = routed_trace
    frozen_ledger_sha256 = "a" * 64
    bf16 = tmp_path / "bf16_quality.json"
    bf16.write_text(
        json.dumps(
            {
                "score_cache": {
                    "message_batch_size": 1,
                    "candidate_batch_size": 3,
                    "sequence_bucket_size": 64,
                },
                "prediction_ledger": {"sha256": frozen_ledger_sha256},
            }
        ),
        encoding="utf-8",
    )
    report_paths["bf16_quality"] = bf16
    quality_gates = tmp_path / "quality_gates.json"
    quality_gates.write_text(
        json.dumps(
            {
                "quality_status": "passed",
                "passed_gates": 39,
                "total_gates": 39,
                "failed_gates": [],
                "quantization_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    report_paths["quality_gates"] = quality_gates
    quantized = tmp_path / "quantized_quality.json"
    quantized.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(
                    artifact_paths["gguf_model"].read_bytes()
                ).hexdigest(),
                "test_gates": {
                    "recall": True,
                    "fpr": True,
                    "core_category_recall": True,
                    "macro_f1_stretch": True,
                },
                "quantization_parity": {
                    "reference_sha256": frozen_ledger_sha256,
                    "exact_calibrated_verdict_parity": True,
                    "release_gate_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    report_paths["quantized_quality"] = quantized
    routed_runtime = tmp_path / "routed_runtime.json"
    routed_runtime.write_text(
        json.dumps(
            {
                "measurement_mode": "interleaved_persistent_per_request",
                "specialist": {
                    "quantization": "Q4_K_M",
                    "calibration_report_sha256": hashlib.sha256(
                        bf16.read_bytes()
                    ).hexdigest(),
                    "quantized_quality_report_sha256": hashlib.sha256(
                        quantized.read_bytes()
                    ).hexdigest(),
                    "frozen_quality_scoring": {
                        "message_batch_size": 1,
                        "candidate_batch_size": 3,
                        "sequence_bucket_size": 64,
                    },
                    "runtime_scoring": {
                        "message_batch_size": 1,
                        "candidate_batch_size": 3,
                        "sequence_bucket_size": 64,
                    },
                    "native_runtime": {
                        "runner_sha256": hashlib.sha256(
                            artifact_paths["runtime_binary"].read_bytes()
                        ).hexdigest(),
                        "model_sha256": hashlib.sha256(
                            artifact_paths["gguf_model"].read_bytes()
                        ).hexdigest(),
                        "protocol_version": 2,
                        "message_batch_size": 1,
                        "candidate_batch_size": 3,
                        "sequence_bucket_size": 64,
                        "prefix_cache_enabled": True,
                        "prefix_tokens": 100,
                        "prefix_sha256": "b" * 64,
                    },
                },
                "parity": {"release_gate_passed": True},
                "latency": {
                    "escalation_rate": 0.05,
                    "overall": {
                        "samples": 100,
                        "p50_ms": 8.0,
                        "p95_ms": 16.5,
                        "p99_ms": 45.0,
                        "maximum_ms": 45.0,
                    },
                    "escalated_path_total": {
                        "samples": 5,
                        "p50_ms": 45.0,
                        "p95_ms": 45.0,
                        "p99_ms": 45.0,
                        "maximum_ms": 45.0,
                    },
                    "gates": {
                        "overall_p95_under_20_ms": True,
                        "escalated_path_p95_under_50_ms": True,
                    },
                },
                "trace_ledger": {
                    "path": str(routed_trace.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(routed_trace.read_bytes()).hexdigest(),
                    "rows": len(trace_records),
                    "contains_message_text": False,
                },
                "process_peak_rss_bytes": 1_100_000_000,
                "environment": {
                    "llama_cpp_revision": "521a64cd01979bb5b1a466152c576a9d809b068d",
                    "runner_source_sha256": hashlib.sha256(
                        report_paths["runtime_source"].read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    report_paths["routed_runtime"] = routed_runtime
    reports = [
        evidence(path, role, tmp_path) for role, path in report_paths.items()
    ]
    return {
        "schema_version": 1,
        "publication_status": "approved",
        "repository": {
            "model_id": "humancto/scamguard-qwen3.5-0.8b",
            "gguf_id": "humancto/scamguard-qwen3.5-0.8b-GGUF",
            "visibility": "public",
        },
        "model": {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "base_license": "Apache-2.0",
            "run_kind": "full",
            "deployment_role": "routed_specialist",
            "experiment_id": "sg-qwen35-08b-schema24",
            "training_examples": 20_000,
            "evaluation_examples": 5_000,
        },
        "quality": {
            "internal_gates": {"passed": 39, "total": 39},
            "external_selection_passed": True,
            "human_label_audit_passed": True,
            "multilingual_claims_reviewed": True,
            "sealed_evaluation_passed": True,
        },
        "quantization": {
            "format": "GGUF",
            "type": "Q4_K_M",
            "merge_equivalence_passed": True,
            "post_quantization_evaluation_passed": True,
            "frozen_calibration_reused": True,
        },
        "runtime": {
            "scoring_contract": {
                "message_batch_size": 1,
                "candidate_batch_size": 3,
                "sequence_bucket_size": 64,
                "exact_decision_parity": True,
                "quantized_decision_parity": True,
            },
            "desktop": {
                "measured": True,
                "device": "MacBook Pro M4 Max",
                "p50_ms": 30.0,
                "p95_ms": 45.0,
                "p99_ms": 55.0,
                "maximum_ms": 70.0,
                "peak_memory_bytes": 900_000_000,
                "samples": 1_000,
            },
            "mobile": {
                "measured": True,
                "device": "physical iPhone",
                "p50_ms": 70.0,
                "p95_ms": 100.0,
                "p99_ms": 120.0,
                "maximum_ms": 150.0,
                "peak_memory_bytes": 900_000_000,
                "samples": 1_000,
            },
            "routed": {
                "measured": True,
                "escalation_rate": 0.05,
                "p50_ms": 8.0,
                "p95_ms": 16.5,
                "p99_ms": 45.0,
                "maximum_ms": 45.0,
                "escalated_p95_ms": 45.0,
                "peak_memory_bytes": 1_100_000_000,
            },
        },
        "governance": {
            "base_license_notice_included": True,
            "data_redistribution_audit_passed": True,
            "pii_scan_passed": True,
            "secrets_scan_passed": True,
            "release_contains_training_rows": False,
            "direct_reddit_training_rows": False,
        },
        "artifacts": artifacts,
        "reports": reports,
    }


def test_complete_release_is_authorized(tmp_path: Path) -> None:
    assert validate_release_manifest(valid_manifest(tmp_path), tmp_path) == []


def test_smoke_run_and_failed_gate_are_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    manifest["model"]["run_kind"] = "smoke"  # type: ignore[index]
    manifest["model"]["evaluation_examples"] = 5  # type: ignore[index]
    manifest["quality"]["external_selection_passed"] = False  # type: ignore[index]

    errors = validate_release_manifest(manifest, tmp_path)

    assert any("smoke runs" in error for error in errors)
    assert any("five-row smoke-test" in error for error in errors)
    assert "external_selection_passed must be true" in errors


def test_missing_physical_mobile_evidence_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    manifest["runtime"]["mobile"]["measured"] = False  # type: ignore[index]
    manifest["runtime"]["mobile"]["device"] = ""  # type: ignore[index]

    errors = validate_release_manifest(manifest, tmp_path)

    assert "measured must be true" in errors
    assert "runtime.mobile.device must be recorded" in errors


def test_tampered_artifact_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    (tmp_path / "gguf_model.bin").write_bytes(b"tampered")

    errors = validate_release_manifest(manifest, tmp_path)

    assert any("SHA-256 mismatch" in error for error in errors)
    assert any("size mismatch" in error for error in errors)


def test_tampered_routed_trace_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    with (tmp_path / "routed_runtime.traces.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "unevidenced"}) + "\n")

    errors = validate_release_manifest(manifest, tmp_path)

    assert any("SHA-256 mismatch" in error for error in errors)
    assert any("size mismatch" in error for error in errors)
    assert any("trace SHA-256" in error for error in errors)
    assert any("trace row count" in error for error in errors)


def test_missing_native_prefix_cache_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    report_path = tmp_path / "routed_runtime.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["specialist"]["native_runtime"]["prefix_cache_enabled"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    for entry in manifest["reports"]:  # type: ignore[union-attr]
        if entry["role"] == "routed_runtime":
            entry["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
            entry["size_bytes"] = report_path.stat().st_size

    errors = validate_release_manifest(manifest, tmp_path)

    assert "routed native runtime prefix_cache_enabled must be true" in errors


def test_uncached_escalated_trace_row_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    trace_path = tmp_path / "routed_runtime.traces.jsonl"
    trace_records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    trace_records[95]["specialist_prefix_reused"] = False
    trace_records[95]["specialist_prefix_tokens"] = 0
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace_records),
        encoding="utf-8",
    )
    report_path = tmp_path / "routed_runtime.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["trace_ledger"]["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    for entry in manifest["reports"]:  # type: ignore[union-attr]
        path = report_path if entry["role"] == "routed_runtime" else trace_path
        if entry["role"] in {"routed_runtime", "routed_trace"}:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            entry["size_bytes"] = path.stat().st_size

    errors = validate_release_manifest(manifest, tmp_path)

    assert "routed_trace[95] must reuse the specialist prefix cache" in errors


def test_non_numeric_routed_summary_is_rejected_without_crashing(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    report_path = tmp_path / "routed_runtime.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["latency"]["overall"]["p95_ms"] = "not-a-number"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    for entry in manifest["reports"]:  # type: ignore[union-attr]
        if entry["role"] == "routed_runtime":
            entry["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
            entry["size_bytes"] = report_path.stat().st_size

    errors = validate_release_manifest(manifest, tmp_path)

    assert "routed runtime overall p95_ms must be finite" in errors


def test_training_row_or_direct_reddit_release_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    manifest["governance"]["release_contains_training_rows"] = True  # type: ignore[index]
    manifest["governance"]["direct_reddit_training_rows"] = True  # type: ignore[index]

    errors = validate_release_manifest(manifest, tmp_path)

    assert "governance.release_contains_training_rows must be false" in errors
    assert "governance.direct_reddit_training_rows must be false" in errors


def test_routed_tail_or_scoring_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    manifest["runtime"]["scoring_contract"]["sequence_bucket_size"] = 0  # type: ignore[index]
    manifest["runtime"]["routed"]["escalated_p95_ms"] = 50.0  # type: ignore[index]

    errors = validate_release_manifest(manifest, tmp_path)

    assert "runtime.scoring_contract.sequence_bucket_size must equal 64" in errors
    assert "runtime.routed.escalated_p95_ms must be under 50" in errors


def test_stale_internal_gate_count_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    manifest["quality"]["internal_gates"] = {"passed": 36, "total": 36}  # type: ignore[index]

    errors = validate_release_manifest(manifest, tmp_path)

    assert "quality.internal_gates.total must equal 39" in errors
