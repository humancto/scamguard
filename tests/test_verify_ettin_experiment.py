"""Synthetic receipts only: no model, original data, network, or GPU access."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from scripts import verify_ettin_experiment as verifier


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        source_names = (verifier.REQUIRED_TRAIN_SOURCES | verifier.REQUIRED_EVAL_SOURCES
                        | {verifier.EVALUATOR})
        for name in source_names:
            self.write(name, "# synthetic source\n")
        self.sources = {name: verifier.sha256(root / name) for name in source_names}
        corpus = [{"id": f"fixture-{i}", "text": "Private synthetic fixture text",
                   "label": label} for i, label in enumerate(verifier.LABELS)]
        for split in ("train", "dev", "test"):
            self.write_lines(f"data/{split}.jsonl", corpus)
        self.config = {
            "experiment_id": "synthetic-ettin-v3",
            "data": {split: self.binding(f"data/{split}.jsonl", 3)
                     for split in ("train", "dev")},
            "training": {"dialogue_policy": "speaker-neutral-action-preserving-v3",
                         "truncation_side": "right", "max_length": 256},
            "outputs": {"report": "reports/train.json", "evaluation": "reports/open.json",
                        "checkpoint": "artifacts/checkpoint", "predictions": "reports/open.jsonl"},
            "evaluation": {"primary_and_other_sealed_tests_may_open": False,
                           "splits": {"test": self.binding("data/test.jsonl", 3)}},
        }
        self.write_json("configs/fixture.json", self.config)
        self.config_path = root / "configs/fixture.json"
        for name in verifier.REQUIRED_CHECKPOINT_FILES - {"scamguard_calibration.json"}:
            self.write(f"artifacts/checkpoint/{name}", "synthetic model metadata, never loaded")
        self.calibration = {
            "model_id": self.config["experiment_id"], "labels": list(verifier.LABELS),
            "input_transform": self.config["training"],
            "dev_data_sha256": self.config["data"]["dev"]["sha256"],
            "checkpoint_model_sha256": verifier.sha256(
                root / "artifacts/checkpoint/model.safetensors"
            ),
        }
        self.write_json("artifacts/checkpoint/scamguard_calibration.json", self.calibration)
        checkpoint_hashes = {name: verifier.sha256(root / "artifacts/checkpoint" / name)
                             for name in verifier.REQUIRED_CHECKPOINT_FILES}
        self.dev_rows = []
        for index, label in enumerate(verifier.LABELS):
            self.dev_rows.append({
                "id": f"fixture-{index}", "split": "dev", "truth": label,
                "argmax": label, "calibrated_verdict": label,
                "threshold_scam": label == "SCAM", "text_sha256": "1" * 64,
                "probabilities": {key: 0.8 if key == label else 0.1
                                  for key in verifier.LABELS},
            })
        self.open_rows = []
        for split in ("dev", "test"):
            for dev in self.dev_rows:
                row = copy.deepcopy(dev)
                row["split"] = split
                row["verdict"] = row.pop("calibrated_verdict")
                self.open_rows.append(row)
        self.write_lines("reports/dev.jsonl", self.dev_rows)
        self.write_lines("reports/open.jsonl", self.open_rows)
        self.train = {
            "status": "completed", "training_completed": True,
            "experiment_id": self.config["experiment_id"],
            "config": self.config, "config_sha256": verifier.sha256(self.config_path),
            "release_authorized": False, "data": self.config["data"],
            "source_sha256": self.sources,
            "checkpoint_files_sha256": checkpoint_hashes,
            "checkpoint_model_sha256": checkpoint_hashes["model.safetensors"],
            "calibration_sha256": checkpoint_hashes["scamguard_calibration.json"],
            "calibration": self.calibration,
            "predictions": self.binding("reports/dev.jsonl", 3),
        }
        self.write_json("reports/train.json", self.train)
        self.opened = {
            "status": "completed", "experiment_id": self.config["experiment_id"],
            "config_sha256": verifier.sha256(self.config_path), "release_authorized": False,
            "sealed_tests_opened": False,
            "training_report_sha256": verifier.sha256(root / "reports/train.json"),
            "checkpoint_model_sha256": checkpoint_hashes["model.safetensors"],
            "calibration_sha256": checkpoint_hashes["scamguard_calibration.json"],
            "data_sha256": {split: self.binding(f"data/{split}.jsonl", 3)["sha256"]
                            for split in ("dev", "test")},
            "evaluation_code_sha256": self.sources[verifier.EVALUATOR],
            "source_code_sha256": {name: self.sources[name]
                                   for name in verifier.REQUIRED_EVAL_SOURCES},
            "input_transform": self.config["training"]["dialogue_policy"],
            "prediction_ledger": self.binding("reports/open.jsonl", 6),
            "metrics": {split: {"examples": 3} for split in ("dev", "test")},
        }
        self.write_json("reports/open.json", self.opened)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, name, value):
        self.write(name, json.dumps(value, indent=2) + "\n")

    def write_lines(self, name, rows):
        self.write(name, "".join(json.dumps(row) + "\n" for row in rows))

    def binding(self, name, rows):
        return {"path": name, "sha256": verifier.sha256(self.root / name), "rows": rows}

    def refresh_train(self):
        self.write_json("reports/train.json", self.train)
        self.opened["training_report_sha256"] = verifier.sha256(self.root / "reports/train.json")
        self.refresh_open()

    def refresh_open(self):
        self.write_json("reports/open.json", self.opened)

    def refresh_open_ledger(self):
        self.write_lines("reports/open.jsonl", self.open_rows)
        self.opened["prediction_ledger"] = self.binding("reports/open.jsonl", len(self.open_rows))
        self.refresh_open()

    def verify(self):
        return verifier.verify_experiment(self.config_path, self.root)


@pytest.fixture
def experiment(tmp_path):
    return Fixture(tmp_path)


def test_success_is_exact_parity_not_release_or_accuracy(experiment):
    result = experiment.verify()
    assert result["status"] == "verified"
    assert result["development_exact_parity_rows"] == 3
    assert result["population_rows"] == {"dev": 3, "test": 3}
    assert result["release_authorized"] is result["production_ready"] is False
    assert result["model_inference_performed"] is result["sealed_tests_opened"] is False
    assert "Private synthetic fixture text" not in json.dumps(result)
    assert "fixture-0" not in json.dumps(result)


@pytest.mark.parametrize("name", [
    "configs/fixture.json", "data/train.jsonl", "data/dev.jsonl", "data/test.jsonl",
    "reports/train.json", "reports/dev.jsonl", "reports/open.jsonl",
    "artifacts/checkpoint/model.safetensors", "artifacts/checkpoint/tokenizer.json",
    "artifacts/checkpoint/scamguard_calibration.json", "training/run_bounded_encoder.py",
    "training/ettin_calibration.py", "src/scamguard/preprocessing.py",
    "src/scamguard/model.py", "src/scamguard/scanner.py", verifier.EVALUATOR,
])
def test_hash_drift_is_rejected(experiment, name):
    path = experiment.root / name
    path.write_text(path.read_text() + " \n")
    with pytest.raises(ValueError, match="hash"):
        experiment.verify()


def test_one_float_ulp_difference_fails_without_tolerance(experiment):
    value = experiment.open_rows[0]["probabilities"]["SAFE"]
    experiment.open_rows[0]["probabilities"]["SAFE"] = math.nextafter(value, 1)
    experiment.refresh_open_ledger()
    with pytest.raises(ValueError, match="probabilities differ"):
        experiment.verify()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("id", "different-id", "IDs differ"),
    ("verdict", "SCAM", "verdicts differ"),
    ("truth", "SCAM", "metadata differs"),
    ("text_sha256", "2" * 64, "metadata differs"),
    ("argmax", "SCAM", "metadata differs"),
    ("threshold_scam", True, "metadata differs"),
])
def test_dev_mismatch_is_not_hidden_by_updated_ledger_hash(experiment, field, value, message):
    experiment.open_rows[0][field] = value
    experiment.refresh_open_ledger()
    with pytest.raises(ValueError, match=message):
        experiment.verify()


def test_missing_evaluation_row_rejected(experiment):
    experiment.open_rows.pop()
    experiment.refresh_open_ledger()
    with pytest.raises(ValueError, match="population coverage"):
        experiment.verify()


@pytest.mark.parametrize("target", ["dev", "open"])
def test_duplicate_id_rejected_even_when_ledger_hash_updated(experiment, target):
    if target == "dev":
        experiment.dev_rows[1] = copy.deepcopy(experiment.dev_rows[0])
        experiment.write_lines("reports/dev.jsonl", experiment.dev_rows)
        experiment.train["predictions"] = experiment.binding("reports/dev.jsonl", 3)
        experiment.refresh_train()
    else:
        experiment.open_rows[1] = copy.deepcopy(experiment.open_rows[0])
        experiment.refresh_open_ledger()
    with pytest.raises(ValueError, match="duplicate prediction ID"):
        experiment.verify()


def test_required_source_receipt_cannot_be_omitted(experiment):
    del experiment.train["source_sha256"]["training/ettin_calibration.py"]
    experiment.refresh_train()
    with pytest.raises(ValueError, match="missing recorded source"):
        experiment.verify()


def test_missing_artifact_rejected(experiment):
    (experiment.root / "artifacts/checkpoint/config.json").unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        experiment.verify()


def test_extra_checkpoint_file_not_silently_unverified(experiment):
    experiment.write("artifacts/checkpoint/unrecorded.txt", "extra")
    with pytest.raises(ValueError, match="unrecorded checkpoint"):
        experiment.verify()


@pytest.mark.parametrize("key", ["training_completed", "status"])
def test_incomplete_fit_rejected(experiment, key):
    experiment.train[key] = False if key == "training_completed" else "running"
    experiment.refresh_train()
    with pytest.raises(ValueError, match="not completed|incomplete"):
        experiment.verify()


def test_wrong_corpus_receipt_rejected(experiment):
    experiment.opened["data_sha256"]["dev"] = "0" * 64
    experiment.refresh_open()
    with pytest.raises(ValueError, match="corpus receipt mismatch"):
        experiment.verify()


def test_wrong_calibration_internal_binding_rejected(experiment):
    experiment.calibration["dev_data_sha256"] = "0" * 64
    experiment.write_json("artifacts/checkpoint/scamguard_calibration.json", experiment.calibration)
    digest = verifier.sha256(experiment.root / "artifacts/checkpoint/scamguard_calibration.json")
    experiment.train["checkpoint_files_sha256"]["scamguard_calibration.json"] = digest
    experiment.train["calibration_sha256"] = digest
    experiment.opened["calibration_sha256"] = digest
    experiment.refresh_train()
    with pytest.raises(ValueError, match="calibration model/development"):
        experiment.verify()


def add_metadata_preflight(experiment):
    experiment.config["model"] = {"files_sha256": {"pytorch_model.bin": "3" * 64}}
    experiment.write_json("configs/fixture.json", experiment.config)
    digest = verifier.sha256(experiment.config_path)
    experiment.train["config_sha256"] = digest
    experiment.opened["config_sha256"] = digest
    experiment.train["base_model_files_sha256"] = experiment.config["model"]["files_sha256"]
    experiment.train["effective_max_steps"] = 4
    preflight = {key: experiment.train[key] for key in (
        "config_sha256", "source_sha256", "data", "base_model_files_sha256",
        "effective_max_steps",
    )}
    preflight.update(status="passed", mode="CPU_metadata_only", model_forward_performed=False,
                     training_performed=False, release_authorized=False)
    experiment.write_json("reports/preflight.json", preflight)
    experiment.train["metadata_preflight"] = {
        "path": "reports/preflight.json",
        "sha256": verifier.sha256(experiment.root / "reports/preflight.json"),
    }
    experiment.refresh_train()
    return preflight


def test_metadata_preflight_and_base_tokenizer_receipts_bound(experiment):
    add_metadata_preflight(experiment)
    assert experiment.verify()["status"] == "verified"


def test_preflight_hash_drift_rejected(experiment):
    add_metadata_preflight(experiment)
    experiment.write("reports/preflight.json", "{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        experiment.verify()


def test_preflight_cannot_change_recipe_with_updated_hash(experiment):
    preflight = add_metadata_preflight(experiment)
    preflight["effective_max_steps"] = 5
    experiment.write_json("reports/preflight.json", preflight)
    experiment.train["metadata_preflight"]["sha256"] = verifier.sha256(
        experiment.root / "reports/preflight.json"
    )
    experiment.refresh_train()
    with pytest.raises(ValueError, match="preflight/fit binding"):
        experiment.verify()


def test_base_weight_receipt_must_match_frozen_config(experiment):
    add_metadata_preflight(experiment)
    experiment.train["base_model_files_sha256"] = {"pytorch_model.bin": "4" * 64}
    experiment.refresh_train()
    with pytest.raises(ValueError, match="base model/tokenizer receipt"):
        experiment.verify()


def test_source_symlink_rejected(experiment):
    path = experiment.root / "training/ettin_calibration.py"
    path.unlink()
    path.symlink_to(experiment.root / "training/train_encoder.py")
    with pytest.raises(ValueError, match="symlink"):
        experiment.verify()


def test_sealed_switch_stops_before_corpus_access(experiment, monkeypatch):
    experiment.config["evaluation"]["primary_and_other_sealed_tests_may_open"] = True
    experiment.write_json("configs/fixture.json", experiment.config)
    checked = []
    original = verifier.sha256

    def recorded(path):
        checked.append(path)
        return original(path)

    monkeypatch.setattr(verifier, "sha256", recorded)
    with pytest.raises(ValueError, match="sealed evaluation forbidden"):
        experiment.verify()
    assert checked == [experiment.config_path]


def test_cli_exclusive_receipt_never_overwrites_evidence(experiment):
    output = experiment.root / "verified.json"
    argv = ["--config", str(experiment.config_path), "--output", str(output),
            "--repo-root", str(experiment.root)]
    assert verifier.main(argv) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        verifier.main(argv)
    assert output.read_bytes() == original


def test_failure_receipt_nonzero_and_no_private_content(experiment):
    experiment.open_rows[0]["text"] = "secret fixture must not enter receipt"
    experiment.refresh_open_ledger()
    output = experiment.root / "failed.json"
    assert verifier.main(["--config", str(experiment.config_path), "--output", str(output),
                          "--repo-root", str(experiment.root)]) == 1
    record = json.loads(output.read_text())
    assert record["status"] == "failed"
    assert record["release_authorized"] is False
    assert "secret fixture" not in output.read_text()
    assert "fixture-0" not in output.read_text()


def test_duplicate_json_fields_rejected():
    with pytest.raises(ValueError, match="duplicate JSON"):
        verifier.parse_json('{"status":"completed","status":"failed"}')


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_probability_rejected(experiment, value):
    experiment.open_rows[0]["probabilities"]["SAFE"] = value
    experiment.refresh_open_ledger()
    with pytest.raises(ValueError, match="probability|non-finite"):
        experiment.verify()
