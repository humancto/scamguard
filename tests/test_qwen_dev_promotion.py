from __future__ import annotations

import json
from pathlib import Path

from scripts.check_qwen_dev_promotion import check
from tests.test_qwen_stage8_promotion import split


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def candidate_report(macro: float) -> dict:
    return {
        "development_screen_only": True,
        "selection_screen_only": False,
        "release_gate_report": False,
        "frozen_calibration_source": None,
        "dev": split(0.971, 0.003, macro),
    }


def test_dev_promotion_passes_before_selection(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "gates.json"
    write(candidate, candidate_report(0.79))
    write(baseline, {"dev": split(0.971, 0.0025, 0.785)})
    result = check(candidate, baseline, output)
    assert result["passed"] is True
    assert result["next_action"] == "score frozen open selection splits"


def test_dev_promotion_blocks_macro_regression(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "gates.json"
    write(candidate, candidate_report(0.75))
    write(baseline, {"dev": split(0.971, 0.0025, 0.785)})
    result = check(candidate, baseline, output)
    assert result["passed"] is False
    assert result["gates"]["dev_macro_non_regression"] is False
    assert result["next_action"] == (
        "reject candidate without scoring open selection splits"
    )
