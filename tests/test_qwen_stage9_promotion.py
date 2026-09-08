from __future__ import annotations

import json
from pathlib import Path

from scamguard.metrics import file_sha256
from scripts.check_qwen_stage9_promotion import check
from tests.test_qwen_stage8_promotion import split


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def prediction_rows(extra_other_safe_alarm: bool = False) -> list[dict]:
    decisions = {
        "flagged-safe": ("SAFE", True),
        "other-safe-1": ("SAFE", True),
        "other-safe-2": ("SAFE", extra_other_safe_alarm),
        "scam-1": ("SCAM", True),
        "scam-2": ("SCAM", True),
    }
    return [
        {
            "id": identifier,
            "split": "phone_scam_validation",
            "truth": truth,
            "threshold_scam": alarm,
            "calibrated_verdict": "SCAM" if alarm else truth,
        }
        for identifier, (truth, alarm) in decisions.items()
    ]


def create_inputs(tmp_path: Path, *, regression: bool = False) -> tuple[Path, ...]:
    candidate_predictions = tmp_path / "candidate.predictions.jsonl"
    stage7_predictions = tmp_path / "stage7.predictions.jsonl"
    write_jsonl(candidate_predictions, prediction_rows(regression))
    write_jsonl(stage7_predictions, prediction_rows(False))

    candidate = tmp_path / "candidate.json"
    stage7 = tmp_path / "stage7.json"
    ppone = tmp_path / "ppone.json"
    audit = tmp_path / "audit.json"
    output = tmp_path / "gates.json"
    write_json(
        candidate,
        {
            "selection_screen_only": True,
            "release_gate_report": False,
            "frozen_calibration_source": {"path": "dev.json"},
            "prediction_ledger": {
                "path": str(candidate_predictions),
                "sha256": file_sha256(candidate_predictions),
            },
            "dev": split(0.98, 0.01, 0.81),
            "phone_scam_validation": split(0.82, 0.16, 0.55),
            "ppone_validation": split(1.0, 0.0, 0.30),
        },
    )
    write_json(
        stage7,
        {
            "prediction_ledger": {
                "path": str(stage7_predictions),
                "sha256": file_sha256(stage7_predictions),
            },
            "dev": split(0.97, 0.01, 0.79),
            "phone_scam_validation": split(0.78, 0.16, 0.54),
        },
    )
    write_json(ppone, {"ppone_validation": split(0.92, 0.0, 0.24, uncertain_hits=1)})
    validation_index = [
        {
            "id": identifier,
            "publisher_label": truth,
            "text_sha256": "a" * 64,
        }
        for identifier, truth in (
            ("flagged-safe", "SAFE"),
            ("other-safe-1", "SAFE"),
            ("other-safe-2", "SAFE"),
            ("scam-1", "SCAM"),
            ("scam-2", "SCAM"),
        )
    ]
    write_json(
        audit,
        {
            "contains_message_text": False,
            "policy": {"automatic_relabeling_allowed": False, "test_inspected": False},
            "validation_index": validation_index,
            "findings": [
                {
                    "id": "flagged-safe",
                    "split": "validation",
                    "publisher_label": "SAFE",
                    "flags": ["publisher_safe_has_high_risk_caller_action"],
                }
            ],
        },
    )
    return (
        candidate,
        candidate_predictions,
        stage7,
        stage7_predictions,
        ppone,
        audit,
        output,
    )


def test_stage9_promotion_ignores_only_predeclared_risky_safe_stratum(
    tmp_path: Path,
) -> None:
    inputs = create_inputs(tmp_path)
    result = check(*inputs)
    assert result["passed"] is True
    assert result["phone_label_audit"]["publisher_labels_changed"] == 0
    assert result["candidate_metrics"]["phone_other_safe_fpr"] == 0.5
    assert result["candidate_metrics"]["phone_high_risk_safe_alarm_rate_diagnostic"] == 1.0


def test_stage9_promotion_rejects_new_alarm_on_other_safe_stratum(
    tmp_path: Path,
) -> None:
    inputs = create_inputs(tmp_path, regression=True)
    result = check(*inputs)
    assert result["passed"] is False
    assert result["gates"]["phone_other_safe_fpr_non_regression"] is False
