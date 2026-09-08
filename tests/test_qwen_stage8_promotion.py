from __future__ import annotations

import json
from pathlib import Path

from scripts.check_qwen_stage8_promotion import check


def split(recall: float, fpr: float, macro: float, uncertain_hits: int = 3) -> dict:
    return {
        "binary_safety": {"scam_recall": recall, "false_positive_rate": fpr},
        "calibrated_decision": {
            "macro_f1": macro,
            "confusion": [[5, 0, 0], [18 - uncertain_hits, uncertain_hits, 0], [0, 0, 12]],
        },
    }


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_stage8_promotion_passes_only_joint_improvement(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    stage7 = tmp_path / "stage7.json"
    ppone = tmp_path / "ppone.json"
    output = tmp_path / "gate.json"
    write(
        candidate,
        {
            "development_screen_only": False,
            "selection_screen_only": True,
            "release_gate_report": False,
            "frozen_calibration_source": {"path": "dev.json"},
            "dev": split(0.98, 0.01, 0.80),
            "phone_scam_validation": split(0.82, 0.14, 0.56),
            "ppone_validation": split(1.0, 0.0, 0.30),
        },
    )
    write(
        stage7,
        {
            "dev": split(0.97, 0.01, 0.79),
            "phone_scam_validation": split(0.78, 0.16, 0.54),
        },
    )
    write(ppone, {"ppone_validation": split(0.92, 0.0, 0.24, uncertain_hits=1)})

    result = check(candidate, stage7, ppone, output)
    assert result["passed"] is True
    assert result["next_action"] == "run frozen full regression without opening PPoNE test"


def test_stage8_promotion_rejects_uncertainty_regression(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    stage7 = tmp_path / "stage7.json"
    ppone = tmp_path / "ppone.json"
    output = tmp_path / "gate.json"
    write(
        candidate,
        {
            "development_screen_only": False,
            "selection_screen_only": True,
            "release_gate_report": False,
            "frozen_calibration_source": {"path": "dev.json"},
            "dev": split(0.98, 0.01, 0.80),
            "phone_scam_validation": split(0.82, 0.14, 0.56),
            "ppone_validation": split(1.0, 0.0, 0.28, uncertain_hits=1),
        },
    )
    write(
        stage7,
        {
            "dev": split(0.97, 0.01, 0.79),
            "phone_scam_validation": split(0.78, 0.16, 0.54),
        },
    )
    write(ppone, {"ppone_validation": split(0.92, 0.0, 0.24, uncertain_hits=1)})

    result = check(candidate, stage7, ppone, output)
    assert result["passed"] is False
    assert result["gates"]["ppone_uncertain_recall"] is False
