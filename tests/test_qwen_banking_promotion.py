from __future__ import annotations

import json
from pathlib import Path

from scripts.check_qwen_banking_promotion import check
from tests.test_qwen_stage8_promotion import split
from tests.test_qwen_stage9_promotion import create_inputs, write_json


def banking_split(fp: int, fpr: float, upper: float) -> dict:
    value = split(0.0, fpr, 0.33)
    value["binary_safety"].update(
        {"fp": fp, "false_positive_rate_ci95": [0.0, upper]}
    )
    return value


def prepare(tmp_path: Path, *, banking_regression: bool = False) -> tuple[Path, ...]:
    (
        candidate,
        candidate_predictions,
        stage7,
        stage7_predictions,
        ppone,
        audit,
        _,
    ) = create_inputs(tmp_path)
    candidate_report = json.loads(candidate.read_text())
    candidate_report.update(
        {
            "banking77_validation": banking_split(
                4 if banking_regression else 3,
                0.0052 if banking_regression else 0.0039,
                0.013 if banking_regression else 0.0115,
            ),
            "vystadial_safe": split(0.0, 0.0, 0.33),
            "international_robocalls": split(0.80, 0.0, 0.30),
        }
    )
    write_json(candidate, candidate_report)

    banking = tmp_path / "stage7-banking.json"
    vystadial = tmp_path / "stage7-vystadial.json"
    international = tmp_path / "stage7-international.json"
    output = tmp_path / "stage13-gates.json"
    write_json(banking, {"banking77_validation": banking_split(8, 0.0104, 0.0204)})
    write_json(vystadial, {"vystadial_safe": split(0.0, 0.0, 0.33)})
    write_json(international, {"international_robocalls": split(0.79, 0.0, 0.30)})
    return (
        candidate,
        candidate_predictions,
        stage7,
        stage7_predictions,
        ppone,
        banking,
        vystadial,
        international,
        audit,
        output,
    )


def test_stage13_promotion_requires_joint_banking_gain_and_retention(
    tmp_path: Path,
) -> None:
    result = check(*prepare(tmp_path))
    assert result["passed"] is True
    assert result["candidate_metrics"]["banking_false_positives"] == 3
    assert result["gates"]["vystadial_fpr_non_regression"] is True
    assert result["next_action"] == (
        "run frozen full regression without opening BANKING77 official test"
    )


def test_stage13_promotion_rejects_four_banking_false_positives(
    tmp_path: Path,
) -> None:
    result = check(*prepare(tmp_path, banking_regression=True))
    assert result["passed"] is False
    assert result["gates"]["banking_fpr_absolute"] is False
    assert result["gates"]["banking_fpr_ci95_upper"] is False
    assert result["gates"]["banking_false_positive_budget"] is False
