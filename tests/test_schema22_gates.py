from __future__ import annotations

import json
from pathlib import Path

from scripts.check_encoder_schema22_gates import CONFIG_PATH, evaluate_gates


def binary(recall: float = 1.0, fpr: float = 0.0) -> dict[str, float]:
    return {"scam_recall": recall, "false_positive_rate": fpr}


def state_slice() -> dict[str, object]:
    return {
        "state_verdict_metrics": {
            "by_state": {
                "harmful_scam": {"threshold_scam_rate": 1.0},
                "routine_safe": {"threshold_scam_rate": 0.0},
                "verified_safe": {"threshold_scam_rate": 0.0},
                "unresolved": {"threshold_scam_rate": 0.0},
            },
            "ordered_contrast_rate": 1.0,
        },
        "action_target_metrics": {
            "macro_roc_auc": 1.0,
            "exact_match_at_0_5": 1.0,
        },
    }


def passing_report() -> dict[str, object]:
    return {
        "dev": {"binary_safety": binary()},
        "test": {"binary_safety": binary()},
        "call_state_validation": state_slice(),
        "multidogo_state_validation": state_slice(),
        "multidogo_call_validation": {
            "binary_safety": binary(),
            "by_source_domain": {
                "insurance": {"binary_safety": binary()},
                "software": {"binary_safety": binary()},
            },
        },
        "call_window_validation": {"binary_safety": binary()},
        "taskmaster_validation": {"binary_safety": binary()},
        "scam_dialogue_validation": {"binary_safety": binary()},
    }


def config() -> dict[str, object]:
    return json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))


def test_schema22_gate_checker_passes_complete_contract() -> None:
    result = evaluate_gates(config(), passing_report())

    assert result["quality_status"] == "passed"
    assert result["passed_gates"] == result["total_gates"]
    assert result["external_selection_authorized"] is True
    assert result["sealed_evaluation_authorized"] is False


def test_schema22_gate_checker_rejects_one_bad_domain() -> None:
    report = passing_report()
    report["multidogo_call_validation"]["by_source_domain"]["insurance"][  # type: ignore[index]
        "binary_safety"
    ]["false_positive_rate"] = 0.04  # type: ignore[index]

    result = evaluate_gates(config(), report)

    assert result["quality_status"] == "rejected"
    assert "MultiDoGO insurance SAFE FPR" in result["failed_gates"]
    assert result["distillation_or_export_authorized"] is False
