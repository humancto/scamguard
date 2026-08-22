from __future__ import annotations

import hashlib
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
    for role in ("merged_model", "gguf_model", "tokenizer"):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(f"artifact:{role}".encode())
        artifacts.append(evidence(path, role, tmp_path))
    reports = []
    for role in (
        "data_manifest",
        "mobile_benchmark",
        "model_card",
        "quality",
        "quantized_quality",
    ):
        path = tmp_path / f"{role}.json"
        path.write_text(f'{{"report":"{role}"}}', encoding="utf-8")
        reports.append(evidence(path, role, tmp_path))
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
            "internal_gates": {"passed": 36, "total": 36},
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
            "desktop": {
                "measured": True,
                "device": "MacBook Pro M4 Max",
                "p50_ms": 30.0,
                "p95_ms": 45.0,
                "peak_memory_bytes": 900_000_000,
                "samples": 1_000,
            },
            "mobile": {
                "measured": True,
                "device": "physical iPhone",
                "p50_ms": 70.0,
                "p95_ms": 100.0,
                "peak_memory_bytes": 900_000_000,
                "samples": 1_000,
            },
            "routed": {
                "measured": True,
                "escalation_rate": 0.08,
                "p50_ms": 8.0,
                "p95_ms": 55.0,
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


def test_training_row_or_direct_reddit_release_is_rejected(tmp_path: Path) -> None:
    manifest = valid_manifest(tmp_path)
    manifest["governance"]["release_contains_training_rows"] = True  # type: ignore[index]
    manifest["governance"]["direct_reddit_training_rows"] = True  # type: ignore[index]

    errors = validate_release_manifest(manifest, tmp_path)

    assert "governance.release_contains_training_rows must be false" in errors
    assert "governance.direct_reddit_training_rows must be false" in errors
