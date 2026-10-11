import hashlib
import json

import pytest

from benchmarks.eval_bounded_encoder import metrics, read_pinned_rows, write_new


def record(truth, verdict, category="CREDENTIAL_MFA"):
    return {"truth": truth, "verdict": verdict, "argmax": verdict,
            "source": "fixture", "category": category, "source_language": "en",
            "invalid_evidence_spans": 0}


def test_uncertainty_not_counted_as_binary_false_positive():
    result = metrics([record("SCAM", "SCAM"), record("SAFE", "SAFE"),
                      record("UNCERTAIN", "SCAM")])
    assert result["scam_recall"] == 1
    assert result["safe_fpr"] == 0
    assert result["binary_precision"] == 1
    assert result["known_scam_fraction_of_alerts"] == 0.5
    assert result["uncertain_recall"] == 0
    assert result["macro_f1"] < 1
    assert result["by_category"]["CREDENTIAL_MFA"]["examples"] == 3


def test_positive_only_does_not_claim_zero_fpr():
    result = metrics([record("SCAM", "SAFE"), record("SCAM", "UNCERTAIN")])
    assert result["safe_fpr"] is None
    assert result["scam_recall"] == 0
    assert result["scams_marked_safe"] == 1
    assert result["scams_deferred"] == 1


def test_pinned_inputs_reject_mismatch_duplicates_and_empty_text(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [{"id": "a", "label": "SAFE", "text": "hello"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    binding = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "rows": 1}
    assert read_pinned_rows(binding) == rows
    with pytest.raises(ValueError, match="hash mismatch"):
        read_pinned_rows({**binding, "sha256": "0" * 64})
    path.write_text("\n".join(json.dumps(r) for r in rows * 2))
    binding.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=2)
    with pytest.raises(ValueError, match="duplicate"):
        read_pinned_rows(binding)
    path.write_text(json.dumps({"id": "b", "label": "SAFE", "text": " "}))
    binding.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=1)
    with pytest.raises(ValueError, match="empty input"):
        read_pinned_rows(binding)


def test_reports_cannot_overwrite_failed_evidence(tmp_path):
    path = tmp_path / "report.json"
    write_new(path, {"status": "incomplete"})
    with pytest.raises(FileExistsError):
        write_new(path, {"status": "completed"})
    assert json.loads(path.read_text())["status"] == "incomplete"
