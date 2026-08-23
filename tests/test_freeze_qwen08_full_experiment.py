from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.audit_protocol import AUDIT_PROTOCOL_VERSION, audit_protocol_sha256
from scripts.freeze_qwen08_full_experiment import freeze


def sft_row(identifier: str, family: str, verdict: str) -> dict[str, object]:
    text = "Urgent: share your verification code." if verdict == "SCAM" else "Hello there."
    target = (
        {
            "verdict": "SCAM",
            "category": "CREDENTIAL_MFA",
            "signals": ["credential_request"],
            "evidence": ["verification code"],
            "recommended_action": "DO_NOT_SHARE_CODE",
        }
        if verdict == "SCAM"
        else {
            "verdict": "SAFE",
            "category": "NONE",
            "signals": [],
            "evidence": [],
            "recommended_action": "NO_ACTION",
        }
    )
    return {
        "id": identifier,
        "family_id": family,
        "source": "fixture",
        "messages": [
            {"role": "system", "content": "Classify safely."},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{text}</message>",
            },
            {"role": "assistant", "content": json.dumps(target)},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def schema24_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed = tmp_path / "data" / "experiments" / "schema24" / "processed"
    sft = processed / "qwen_sft"
    sft.mkdir(parents=True)
    raw_train = [{"id": "train", "label": "SCAM"}]
    raw_dev = [{"id": "dev", "label": "SAFE"}]
    write_jsonl(processed / "train.jsonl", raw_train)
    write_jsonl(processed / "dev.jsonl", raw_dev)
    write_jsonl(processed / "test.jsonl", [{"id": "test", "label": "SCAM"}])
    write_jsonl(sft / "train.jsonl", [sft_row("train", "family-train", "SCAM")])
    write_jsonl(sft / "dev.jsonl", [sft_row("dev", "family-dev", "SAFE")])
    curriculum = tmp_path / "data" / "external" / "multidogo_annotated" / "manifest.json"
    curriculum.parent.mkdir(parents=True)
    curriculum.write_text('{"artifact_schema_version":1}', encoding="utf-8")
    manifest = {
        "schema_version": 24,
        "counts": {"train": 1, "dev": 1, "test": 1},
        "schema24_increment": {
            "paper_dev_test_rows_used_for_fitting": False,
            "annotation_train_rows": 1,
            "annotation_dev_rows": 1,
            "annotation_test_rows": 1,
            "annotation_curriculum_manifest": str(curriculum),
            "annotation_curriculum_manifest_sha256": file_sha256(curriculum),
        },
        "schema24_privacy": {
            "revision": "contextual_sensitive_values_v1",
            "access_codes_are_never_training_features": True,
            "applied_before_overlap_control": True,
        },
    }
    data_manifest_path = processed / "manifest.json"
    data_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sft_manifest = {
        "artifact_schema_version": 1,
        "input_manifest_sha256": file_sha256(data_manifest_path),
        "policy": {
            "safe_rows_require_empty_risk_metadata": True,
            "scam_rows_require_verbatim_runtime_evidence": True,
            "unsupported_scam_rows_excluded_from_sft": True,
            "unsupported_scam_rows_relabelled": False,
            "all_non_scam_rows_retained": True,
        },
        "splits": {
            "train": {
                "input_rows": 1,
                "output_rows": 1,
                "excluded_unsupported_scam_rows": 0,
                "output_sha256": file_sha256(sft / "train.jsonl"),
            },
            "dev": {
                "input_rows": 1,
                "output_rows": 1,
                "excluded_unsupported_scam_rows": 0,
                "output_sha256": file_sha256(sft / "dev.jsonl"),
            },
        },
    }
    (sft / "manifest.json").write_text(json.dumps(sft_manifest), encoding="utf-8")
    token_audit = tmp_path / "token-audit.json"
    token_audit.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "max_length": 640,
                "examples": 2,
                "split_counts": {"train": 1, "dev": 1},
                "full_tokens": {"p95": 90, "p99": 95, "max": 100},
                "supervised_tokens": {"min": 10},
                "full_over_max_length": 0,
            }
        ),
        encoding="utf-8",
    )
    label_audit = tmp_path / "label-audit.json"
    label_audit.write_text(
        json.dumps(
            {
                "release_gate_passed": True,
                "rows": 635,
                "complete_rows": 635,
                "incomplete_rows": 0,
                "incorrect_label_rows": 0,
                "sensitive_data_rows": 0,
                "agreement": 1.0,
                "agreement_wilson_95_lower": 0.994,
                "cohen_kappa": 1.0,
                "audit_sha256": "a" * 64,
                "audit_manifest_sha256": "b" * 64,
                "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
                "audit_protocol_sha256": audit_protocol_sha256(),
                "imported_from_blind_bundle": True,
                "blind_bundle_sha256": "c" * 64,
                "returned_blind_audit_sha256": "d" * 64,
                "data_manifest_sha256": file_sha256(data_manifest_path),
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return processed, token_audit, label_audit


def test_freeze_writes_full_hash_bound_experiment(tmp_path: Path) -> None:
    processed, token_audit, label_audit = schema24_fixture(tmp_path)
    output = tmp_path / "qwen08.json"
    checkpoint = tmp_path / "qwen08-checkpoint"

    config = freeze(
        processed,
        token_audit,
        label_audit,
        output,
        checkpoint,
        "sg-qwen08-schema24",
    )

    assert config["run_kind"] == "full"
    assert config["base_model"] == "Qwen/Qwen3.5-0.8B"
    assert config["batch_size"] == 4
    assert config["gradient_accumulation"] == 4
    assert config["batch_size"] * config["gradient_accumulation"] == 16
    assert config["data"]["label_audit"]["imported_from_blind_bundle"] is True  # type: ignore[index]
    assert config["data"]["schema_version"] == 24  # type: ignore[index]
    assert config["data"]["evidence_audit"]["coverage"] == 1.0  # type: ignore[index]
    assert output.is_file()


def test_freeze_rejects_incomplete_human_audit(tmp_path: Path) -> None:
    processed, token_audit, label_audit = schema24_fixture(tmp_path)
    report = json.loads(label_audit.read_text(encoding="utf-8"))
    report["incorrect_label_rows"] = 1
    report["release_gate_passed"] = False
    label_audit.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="human label audit"):
        freeze(
            processed,
            token_audit,
            label_audit,
            tmp_path / "qwen08.json",
            tmp_path / "checkpoint",
            "sg-qwen08-schema24",
        )


def test_freeze_rejects_non_blind_label_audit(tmp_path: Path) -> None:
    processed, token_audit, label_audit = schema24_fixture(tmp_path)
    report = json.loads(label_audit.read_text(encoding="utf-8"))
    report["imported_from_blind_bundle"] = False
    label_audit.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="human label audit"):
        freeze(
            processed,
            token_audit,
            label_audit,
            tmp_path / "qwen08.json",
            tmp_path / "checkpoint",
            "sg-qwen08-schema24",
        )


def test_freeze_rejects_stale_human_audit_rubric(tmp_path: Path) -> None:
    processed, token_audit, label_audit = schema24_fixture(tmp_path)
    report = json.loads(label_audit.read_text(encoding="utf-8"))
    report["audit_protocol_sha256"] = "0" * 64
    label_audit.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="human label audit"):
        freeze(
            processed,
            token_audit,
            label_audit,
            tmp_path / "qwen08.json",
            tmp_path / "checkpoint",
            "sg-qwen08-schema24",
        )


def test_freeze_rejects_token_truncation(tmp_path: Path) -> None:
    processed, token_audit, label_audit = schema24_fixture(tmp_path)
    report = json.loads(token_audit.read_text(encoding="utf-8"))
    report["full_over_max_length"] = 1
    token_audit.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="token audit"):
        freeze(
            processed,
            token_audit,
            label_audit,
            tmp_path / "qwen08.json",
            tmp_path / "checkpoint",
            "sg-qwen08-schema24",
        )


def test_freeze_rejects_unselected_batch_geometry(tmp_path: Path) -> None:
    processed, token_audit, label_audit = schema24_fixture(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    selection = json.loads(
        (repository / "reports/QWEN08_BATCH_GEOMETRY_SELECTION.json").read_text(
            encoding="utf-8"
        )
    )
    selection["selected"]["microbatch_size"] = 2
    selection_path = tmp_path / "tampered-batch-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="batch-geometry selection"):
        freeze(
            processed,
            token_audit,
            label_audit,
            tmp_path / "qwen08.json",
            tmp_path / "checkpoint",
            "sg-qwen08-schema24",
            selection_path,
        )
