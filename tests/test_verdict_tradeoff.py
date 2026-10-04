from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scamguard.decision import calibrated_verdict
from scamguard.metrics import binary_safety_metrics, file_sha256
from scripts.audit_verdict_tradeoff import LABELS, audit, fixed_scam_oracle_ceiling, main


def fixture(tmp_path: Path) -> tuple[Path, Path, dict, list[dict]]:
    truth = np.array([0, 0, 1, 1, 2, 2, 2])
    probabilities = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.50, 0.40],
            [0.10, 0.55, 0.35],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.60, 0.30, 0.10],
            [0.10, 0.80, 0.10],
        ]
    )
    rows = []
    for index, (actual, probability) in enumerate(zip(truth, probabilities, strict=True)):
        verdict = calibrated_verdict(
            safe_probability=probability[0],
            scam_probability=probability[2],
            scam_probability_threshold=0.30,
            safe_probability_threshold=0.50,
            safe_max_scam_probability=None,
        )
        rows.append(
            {
                "id": f"example-{index}",
                "split": "dev",
                "source": "synthetic-fixture",
                "truth": LABELS[actual],
                "argmax": LABELS[probability.argmax()],
                "calibrated_verdict": verdict,
                "threshold_scam": verdict == "SCAM",
                "negative_log_likelihood": float(-np.log(probability[actual])),
                "probabilities": dict(zip(LABELS, probability.tolist(), strict=True)),
            }
        )
    predicted = np.array([LABELS.index(row["calibrated_verdict"]) for row in rows])
    argmax = probabilities.argmax(axis=1)
    binary = truth != 1
    report = {
        "development_screen_only": True,
        "selection_screen_only": False,
        "release_gate_report": False,
        "frozen_calibration_source": None,
        "temperature": 1.1,
        "scam_threshold": 0.30,
        "safe_threshold": 0.50,
        "safe_threshold_semantics": "minimum_safe_probability",
        "dev": {
            "examples": len(rows),
            "labels": dict(Counter(LABELS[index] for index in truth)),
            "accuracy_argmax": accuracy_score(truth, argmax),
            "macro_f1_argmax": f1_score(truth, argmax, average="macro"),
            "confusion_argmax": confusion_matrix(truth, argmax, labels=[0, 1, 2]).tolist(),
            "calibrated_decision": {
                "accuracy": accuracy_score(truth, predicted),
                "macro_f1": f1_score(truth, predicted, average="macro"),
                "confusion": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
                "safe_threshold": 0.50,
            },
            "binary_safety": binary_safety_metrics(
                (truth[binary] == 2).astype(int),
                probabilities[binary, 2],
                0.30,
            ),
            "calibration": {
                "temperature": 1.1,
                "after_temperature": {
                    "negative_log_likelihood": float(
                        -np.log(probabilities[np.arange(len(truth)), truth]).mean()
                    ),
                },
            },
        },
    }
    report_path, ledger_path = tmp_path / "dev.json", tmp_path / "ledger.jsonl"
    write_bound(report_path, ledger_path, report, rows)
    return report_path, ledger_path, report, rows


def write_bound(report_path: Path, ledger_path: Path, report: dict, rows: list[dict]) -> None:
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report["prediction_ledger"] = {
        "path": str(ledger_path),
        "sha256": file_sha256(ledger_path),
        "examples": len(rows),
        "contains_message_text": False,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_bound_audit_counts_alerts_and_never_processes_non_dev(tmp_path: Path) -> None:
    report_path, ledger_path, report, rows = fixture(tmp_path)
    # Deliberately unusable predictions and a colliding ID in another split:
    # the diagnostic must not validate, score, or output these records.
    rows.append({"split": "sealed", "id": "example-0", "probabilities": "DO NOT PROCESS"})
    write_bound(report_path, ledger_path, report, rows)
    before = (file_sha256(report_path), file_sha256(ledger_path))
    result = audit(report_path, ledger_path)
    assert result["scope"]["non_dev_records_ignored"] == 1
    assert result["scope"]["dev_examples"] == 7
    assert result["alerts"]["alerts_by_reference_label"] == {"SAFE": 1, "UNCERTAIN": 1, "SCAM": 1}
    assert result["alerts"]["binary_subset_precision"] == 0.5
    assert result["alerts"]["alert_fraction_by_reference_label"]["SCAM"] == pytest.approx(1 / 3)
    assert result["reference_uncertain"]["argmax_to_decision_confusion"][1] == [0, 1, 1]
    assert result["diagnostic_only"] is True
    assert result["eligibility_authorization"] is False
    assert "DO NOT PROCESS" not in json.dumps(result)
    assert "example-0" not in json.dumps(result)
    assert before == (file_sha256(report_path), file_sha256(ledger_path))


def test_oracle_matches_all_possible_non_scam_assignments() -> None:
    rng = np.random.default_rng(37)
    for _ in range(35):
        truth = rng.integers(0, 3, 6)
        predicted = rng.integers(0, 3, 6)
        fixed = predicted == 2
        actual = fixed_scam_oracle_ceiling(confusion_matrix(truth, predicted, labels=[0, 1, 2]))
        oracle = []
        for decisions in itertools.product((0, 1), repeat=int((~fixed).sum())):
            candidate = predicted.copy()
            candidate[~fixed] = decisions
            oracle.append(f1_score(truth, candidate, average="macro", zero_division=0))
        assert actual["macro_f1_upper_bound"] == pytest.approx(max(oracle))


def test_observed_stage7_matrix_has_tight_ceiling() -> None:
    result = fixed_scam_oracle_ceiling(np.array([[2001, 2, 5], [1, 34, 77], [12, 3, 499]]))
    assert result["macro_f1_upper_bound"] == pytest.approx(0.7942127638062962)
    assert result["oracle_confusion"] == [[2003, 0, 5], [0, 35, 77], [15, 0, 499]]


@pytest.mark.parametrize("kind", ["missing", "empty", "duplicate", "conflicting", "missing_row"])
def test_rejects_missing_or_nonunique_development_identity(tmp_path: Path, kind: str) -> None:
    report_path, ledger_path, report, rows = fixture(tmp_path)
    if kind == "missing":
        rows[0].pop("id")
    elif kind == "empty":
        rows[0]["id"] = ""
    elif kind == "duplicate":
        rows.append(dict(rows[0]))
    elif kind == "conflicting":
        rows[1]["id"] = rows[0]["id"]
    else:
        rows.pop()
    write_bound(report_path, ledger_path, report, rows)
    with pytest.raises(ValueError, match="IDs|dev.examples"):
        audit(report_path, ledger_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9", 0.89])
def test_rejects_invalid_probabilities(tmp_path: Path, value: object) -> None:
    report_path, ledger_path, report, rows = fixture(tmp_path)
    rows[0]["probabilities"]["SAFE"] = value
    write_bound(report_path, ledger_path, report, rows)
    with pytest.raises(ValueError, match="finite|sum to one"):
        audit(report_path, ledger_path)


def test_rejects_unbound_ledger_tamper(tmp_path: Path) -> None:
    report_path, ledger_path, _, rows = fixture(tmp_path)
    rows[0]["id"] = "tampered"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="bound"):
        audit(report_path, ledger_path)


@pytest.mark.parametrize(
    "field", ["calibrated_verdict", "argmax", "threshold_scam", "negative_log_likelihood"]
)
def test_rejects_conflicting_prediction_fields(tmp_path: Path, field: str) -> None:
    report_path, ledger_path, report, rows = fixture(tmp_path)
    rows[0][field] = {
        "calibrated_verdict": "SCAM",
        "argmax": "SCAM",
        "threshold_scam": True,
        "negative_log_likelihood": 0.8,
    }[field]
    write_bound(report_path, ledger_path, report, rows)
    with pytest.raises(ValueError, match="differs"):
        audit(report_path, ledger_path)


@pytest.mark.parametrize("field", ["confusion", "labels", "threshold", "temperature", "macro"])
def test_rejects_conflicting_report_fields(tmp_path: Path, field: str) -> None:
    report_path, ledger_path, report, rows = fixture(tmp_path)
    if field == "confusion":
        report["dev"]["calibrated_decision"]["confusion"][0][0] += 1
    elif field == "labels":
        report["dev"]["labels"]["SAFE"] += 1
    elif field == "threshold":
        report["dev"]["binary_safety"]["threshold"] = 0.31
    elif field == "temperature":
        report["dev"]["calibration"]["temperature"] = 2.0
    else:
        report["dev"]["macro_f1_argmax"] = 0.999
    write_bound(report_path, ledger_path, report, rows)
    with pytest.raises(ValueError, match="differs"):
        audit(report_path, ledger_path)


def test_accepts_explicit_legacy_missing_selection_flag_only(tmp_path: Path) -> None:
    report_path, ledger_path, report, rows = fixture(tmp_path)
    report.pop("selection_screen_only")
    write_bound(report_path, ledger_path, report, rows)
    assert audit(report_path, ledger_path)["scope"]["legacy_report_without_selection_screen_only"]
    report["selection_screen_only"] = None
    write_bound(report_path, ledger_path, report, rows)
    with pytest.raises(ValueError, match="development-only"):
        audit(report_path, ledger_path)


@pytest.mark.parametrize("target", ["report", "ledger", "existing"])
def test_cli_preserves_inputs_and_existing_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    report_path, ledger_path, _, _ = fixture(tmp_path)
    output = {"report": report_path, "ledger": ledger_path}.get(target, tmp_path / "receipt.json")
    if target == "existing":
        output.write_text("existing receipt", encoding="utf-8")
    original = output.read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_verdict_tradeoff",
            "--report",
            str(report_path),
            "--predictions",
            str(ledger_path),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="overwrite an input|already exists"):
        main()
    assert output.read_bytes() == original
