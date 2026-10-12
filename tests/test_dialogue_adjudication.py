"""CPU-only synthetic fixtures for the diagnostic, non-relabeling blind packet."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from scripts import prepare_dialogue_adjudication as audit
from scripts.blind_audit import BLIND_FIELDS, validate_bundle_manifest
from scripts.check_audit_completion import validate_audit_binding


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def dump_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    sources = {split: tmp_path / f"{split}.jsonl" for split in audit.SPLITS}
    ledger_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "open.json"
    source_rows = {}
    predictions = []
    counter = 0
    vectors = {
        "SAFE": {"SAFE": 0.9, "UNCERTAIN": 0.08, "SCAM": 0.02},
        "SCAM": {"SAFE": 0.1, "UNCERTAIN": 0.1, "SCAM": 0.8},
        "UNCERTAIN": {"SAFE": 0.2, "UNCERTAIN": 0.7, "SCAM": 0.1},
    }
    for split in audit.SPLITS:
        rows = []
        for truth, verdict, count in (
            ("SAFE", "SCAM", audit.EXPECTED_TARGETS[split, "SAFE"]),
            ("SCAM", "SAFE", audit.EXPECTED_TARGETS[split, "SCAM"]),
            ("SAFE", "SAFE", 60),
            ("SCAM", "SCAM", 60),
            ("SAFE", "UNCERTAIN", 1),
        ):
            for _ in range(count):
                counter += 1
                row = {
                    "id": f"source-bearing-id-{counter}",
                    "text": f" Synthetic message {counter}. ",
                    "label": truth,
                    "source": audit.SOURCE_NAMES[split],
                    "source_label": truth,
                    "category": "NONE" if truth == "SAFE" else "CREDENTIAL_THEFT",
                    "source_language": "English",
                    "source_dialogue_type": "support",
                    "split": "validation",
                    "family_id": f"family-{counter}",
                }
                rows.append(row)
                predictions.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "truth": truth,
                        "verdict": verdict,
                        "text_sha256": audit.text_sha256(row["text"]),
                        "probabilities": dict(vectors[verdict]),
                        "threshold_scam": verdict == "SCAM",
                        **{
                            field: row[field] for field in ("source", "category", "source_language")
                        },
                    }
                )
        source_rows[split] = rows
        dump_rows(sources[split], rows)
    # Other-split metadata is allowed in the frozen ledger, but is never selected or joined.
    predictions.append({"id": "never-joined", "split": "some_other_open_split"})
    dump_rows(ledger_path, predictions)
    report = {
        "status": "completed",
        "experiment_id": "sg-ettin150m-recovery-v2",
        "sealed_tests_opened": False,
        "data_sha256": {split: audit.file_sha256(path) for split, path in sources.items()},
        "prediction_ledger": {
            "path": str(ledger_path),
            "sha256": audit.file_sha256(ledger_path),
            "rows": len(predictions),
            "examples": len(predictions),
        },
    }
    dump(report_path, report)
    monkeypatch.setattr(
        audit,
        "EXPECTED_HASHES",
        {
            "report": audit.file_sha256(report_path),
            "ledger": audit.file_sha256(ledger_path),
            **{split: audit.file_sha256(path) for split, path in sources.items()},
        },
    )
    monkeypatch.setattr(audit, "EXPECTED_SOURCE_ROWS", {k: len(v) for k, v in source_rows.items()})
    return report_path, ledger_path, sources, source_rows, predictions


def refresh_binding(inputs, monkeypatch):
    report_path, ledger_path, sources, source_rows, predictions = inputs
    for split, rows in source_rows.items():
        dump_rows(sources[split], rows)
    dump_rows(ledger_path, predictions)
    report = json.loads(report_path.read_text())
    report["data_sha256"] = {split: audit.file_sha256(path) for split, path in sources.items()}
    report["prediction_ledger"].update(
        sha256=audit.file_sha256(ledger_path), rows=len(predictions), examples=len(predictions)
    )
    dump(report_path, report)
    monkeypatch.setattr(
        audit,
        "EXPECTED_HASHES",
        {
            "report": audit.file_sha256(report_path),
            "ledger": audit.file_sha256(ledger_path),
            **{split: audit.file_sha256(path) for split, path in sources.items()},
        },
    )


def validate(inputs):
    return audit.validate_inputs(*inputs[:3])


def test_exact_targets_controls_and_deterministic_order(inputs):
    rows = validate(inputs)
    selected, manifest = audit.select_rows(rows)
    assert len(selected) == len({row["id"] for row in selected}) == 96
    assert Counter(row["selection_role"] for row in manifest["rows"]) == {
        "target": 47,
        "concordant_control": 49,
    }
    assert audit.select_rows(list(reversed(rows))) == (selected, manifest)
    target_counts = Counter(
        (row["dataset_split"], row["reference_label"])
        for row in manifest["rows"]
        if row["selection_role"] == "target"
    )
    assert target_counts == audit.EXPECTED_TARGETS
    assert all(
        row["reference_label"] == row["model_verdict"]
        for row in manifest["rows"]
        if row["selection_role"] == "concordant_control"
    )
    assert sum(row["selected_controls"] for row in manifest["control_allocations"]) == 49


def test_source_hash_tamper_fails_before_output(inputs, tmp_path):
    inputs[2][audit.SPLITS[0]].write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(inputs)


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate_prediction", "duplicate dialogue"),
        ("missing_prediction", "missing or extra"),
        ("duplicate_source", "duplicate source"),
        ("truth", "truth/text"),
        ("text", "truth/text"),
        ("probability", "probability"),
        ("sum", "probability"),
        ("verdict", "serving decision"),
        ("mask", "serving decision"),
        ("sealed_source", "source row"),
    ],
)
def test_semantic_binding_guards(inputs, monkeypatch, change, match):
    row = inputs[4][0]
    if change == "duplicate_prediction":
        inputs[4].append(dict(row))
    elif change == "missing_prediction":
        inputs[4].pop(0)
    elif change == "duplicate_source":
        source = inputs[3][audit.SPLITS[0]]
        source[1] = dict(source[0])
    elif change == "truth":
        row["truth"] = "SCAM"
    elif change == "text":
        row["text_sha256"] = "0" * 64
    elif change == "probability":
        row["probabilities"]["SCAM"] = float("nan")
    elif change == "sum":
        row["probabilities"]["SAFE"] = 0.8
    elif change == "verdict":
        row["verdict"] = "SAFE"
    elif change == "mask":
        row["threshold_scam"] = False
    elif change == "sealed_source":
        inputs[3][audit.SPLITS[0]][0]["split"] = "test"
    refresh_binding(inputs, monkeypatch)
    with pytest.raises(ValueError, match=match):
        validate(inputs)


def test_changed_target_inventory_and_insufficient_controls_fail(inputs):
    rows = validate(inputs)
    with pytest.raises(ValueError, match="target inventory"):
        audit.select_rows(rows[1:])
    targets = [r for r in rows if r["label"] != r["prediction"]["verdict"]]
    with pytest.raises(ValueError, match="no concordant control"):
        audit.select_rows(targets)


def test_actual_bundle_has_no_answer_key_and_preserves_original_text(inputs, tmp_path):
    output = tmp_path / "private"
    public = tmp_path / "public.json"
    app = Path(__file__).parents[1] / "scripts" / "review_blind_audit.py"
    before = {path: path.read_bytes() for path in [inputs[0], inputs[1], *inputs[2].values()]}
    receipt = audit.prepare(*inputs[:3], output, public, app)
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert receipt["role"] == "benchmark_label_adjudication"
    assert receipt["independent_human_review"] is False
    assert receipt["zero_disagreement_release_gate_applicable"] is False
    assert receipt["benchmark_labels_changed"] is False
    assert receipt["existing_635_row_training_audit_unchanged_and_still_required"] is True
    assert output.stat().st_mode & 0o777 == 0o700
    assert (
        validate_audit_binding(
            output / "private-canonical.csv", output / "private-canonical.manifest.json"
        )[1]
        == []
    )
    with zipfile.ZipFile(output / "dialogue-adjudication-blind.zip") as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "review.py",
            "scamguard_blind_audit.csv",
            "scamguard_blind_audit.manifest.json",
        }
        payload = archive.read("scamguard_blind_audit.csv").decode()
        reader = csv.DictReader(io.StringIO(payload))
        assert tuple(reader.fieldnames) == BLIND_FIELDS
        blind_rows = list(reader)
        assert len(blind_rows) == 96
        all_text = {row["text"] for group in inputs[3].values() for row in group}
        assert all(row["text"] in all_text for row in blind_rows)
        assert all(
            row["id"].startswith("sg-") and "source-bearing" not in row["id"] for row in blind_rows
        )
        assert all(
            not row[field]
            for row in blind_rows
            for field in ("auditor_label", "contains_sensitive_data", "notes")
        )
        extraction = tmp_path / "reviewer"
        archive.extractall(extraction)
    manifest = json.loads((extraction / "scamguard_blind_audit.manifest.json").read_text())
    assert (
        validate_bundle_manifest(
            manifest, extraction / "scamguard_blind_audit.csv", require_complete=False
        )[1]
        == []
    )
    public_text = public.read_text()
    assert "Synthetic message" not in public_text and "source-bearing-id" not in public_text
    assert "model_verdict" not in public_text and "reference_label" not in public_text
    with pytest.raises(FileExistsError, match="overwrite"):
        audit.prepare(*inputs[:3], output, public, app)


def test_public_receipt_must_stay_outside_private_directory(inputs, tmp_path):
    output = tmp_path / "private"
    with pytest.raises(ValueError, match="outside"):
        audit.prepare(*inputs[:3], output, output / "receipt.json", Path(__file__))
    assert not output.exists()
