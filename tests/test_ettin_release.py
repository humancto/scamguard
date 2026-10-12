"""CPU-only fail-closed staging tests; no model loading, auth, network or inference."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import stage_ettin_release as release


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def ref(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": release.file_sha256(path)}


@pytest.fixture
def package_case(tmp_path, monkeypatch):
    root = tmp_path
    checkpoint = root / "artifacts/checkpoint"
    checkpoint.mkdir(parents=True)
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header
                                                  + struct.pack("<f", 0.5))
    write_json(checkpoint / "config.json", {
        "model_type": "modernbert", "architectures": ["ModernBertForSequenceClassification"],
        "id2label": {str(i): label for i, label in enumerate(release.LABELS)},
        "label2id": {label: i for i, label in enumerate(release.LABELS)},
    })
    write_json(checkpoint / "tokenizer.json", {"version": "1.0"})
    write_json(checkpoint / "tokenizer_config.json", {"truncation_side": "right"})
    transform = {"dialogue_policy": "speaker-neutral-evidence-recent-v2", "max_length": 256,
                 "truncation_side": "right"}
    write_json(checkpoint / "scamguard_calibration.json", {
        "model_id": "ettin-test", "labels": release.LABELS, "temperature": 1.5,
        "scam_threshold": 0.25, "safe_threshold": 0.2,
        "safe_threshold_semantics": "maximum_safe_risk", "input_transform": transform,
        "calibration_version": "ettin-serving-batch1-torch-fp32-v1",
        "checkpoint_model_sha256": release.file_sha256(checkpoint / "model.safetensors"),
        "dev_data_sha256": "b" * 64,
        "serving_contract": {**transform, "message_batch_size": 1, "padding": False,
                             "logits_dtype": "float32", "softmax_vector_shape": [3],
                             "text_preprocessing": "strip then prepare_model_text",
                             "softmax": "torch.softmax((logits[0, :3] / temperature), dim=-1)"},
    })
    (checkpoint / "training_args.bin").write_bytes(b"never-copy-or-unpickle")
    for name in release.SERVING_SOURCES | {"src/scamguard/decision.py",
                                           "benchmarks/eval_bounded_encoder.py"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixed synthetic source {name}\n")
    frozen = {name: release.file_sha256(root / name)
              for name in release.SERVING_SOURCES | {"src/scamguard/decision.py",
                                                      "benchmarks/eval_bounded_encoder.py"}}
    monkeypatch.setattr(release, "frozen_source_hash", lambda _root, _rev, name: frozen[name])
    (root / "LICENSE").write_text("Apache License Version 2.0\n")
    (root / "base-MIT.txt").write_text("MIT License\nPermission is hereby granted.\n")
    monkeypatch.setattr(release, "BASE_LICENSE_SHA256", release.file_sha256(root / "base-MIT.txt"))
    (root / "DATA_NOTICE.md").write_text("Source rights require independent owner review.\n")
    policy = {
        "dev_scam_recall_min": 0.97, "dev_safe_fpr_max": 0.02,
        "test_scam_recall_min": 0.97, "test_safe_fpr_max": 0.02,
        "test_core_category_recall_min": 0.97, "test_category_min_examples": 20,
        "required_dialogue_splits": ["phone"], "dialogue_scam_recall_min": 0.97,
        "dialogue_safe_fpr_max": 0.02, "macro_f1_stretch": 0.94,
    }
    config = {
        "experiment_id": "ettin-test",
        "model": {"repository": release.BASE, "revision": release.REVISION, "license": "MIT"},
        "outputs": {"checkpoint": "artifacts/checkpoint"}, "training": {"max_steps": 2},
        "data": {"train": {"sha256": "a" * 64, "rows": 3},
                 "dev": {"sha256": "b" * 64, "rows": 2}},
        "evaluation": {"splits": {"test": {"sha256": "c" * 64},
                                   "phone": {"sha256": "d" * 64}},
                       "latency": {"p95_ms_max": 20}}, "quality_promotion": policy,
    }
    write_json(root / "config.json", config)
    hashes = {p.name: release.file_sha256(p) for p in checkpoint.iterdir()}
    train = {
        "experiment_id": "ettin-test", "status": "completed", "training_completed": True,
        "global_step": 2, "effective_max_steps": 2,
        "config_sha256": release.file_sha256(root / "config.json"),
        "checkpoint_model_sha256": hashes["model.safetensors"],
        "calibration_sha256": hashes["scamguard_calibration.json"],
        "checkpoint_files_sha256": hashes,
        "source_sha256": {"src/scamguard/decision.py": frozen["src/scamguard/decision.py"]},
        "data": {"train": {**config["data"]["train"], "sources": {"synthetic_fixture": 3}},
                 "dev": config["data"]["dev"]},
    }
    write_json(root / "train.json", train)
    good = {"scam_recall": 1.0, "safe_fpr": 0.0, "invalid_evidence_spans": 0,
            "truth_counts": {"SCAM": 30, "SAFE": 30}, "macro_f1": 1.0}
    metrics = {"dev": dict(good), "test": {**good, "by_category": {"IDENTITY": good}},
               "phone": {**good, "safe_fpr": 0.1}}
    latency = {"p95_ms": 10.0, "limit_ms": 20}
    opened = {
        "status": "completed", "experiment_id": "ettin-test",
        "config_sha256": train["config_sha256"],
        "training_report_sha256": release.file_sha256(root / "train.json"),
        "checkpoint_model_sha256": train["checkpoint_model_sha256"],
        "calibration_sha256": train["calibration_sha256"], "release_authorized": False,
        "sealed_tests_opened": False,
        "source_code_sha256": {name: frozen[name] for name in release.SERVING_SOURCES},
        "evaluation_code_sha256": frozen["benchmarks/eval_bounded_encoder.py"],
        "data_sha256": {"dev": "b" * 64, "test": "c" * 64, "phone": "d" * 64},
        "metrics": metrics, "latency": latency,
        "promotion": release.check_promotion(metrics, latency, policy),
    }
    write_json(root / "open.json", opened)
    write_json(root / "provenance.json", {
        "schema_version": 1, "release_authorized": False, "rights_review_status": "review_required",
        "data_sha256": {"train": "a" * 64, "dev": "b" * 64}, "base_model": config["model"],
        "sources": [{"id": "synthetic_fixture", "rows": 3}],
    })
    names = {"experiment_config": "config.json", "training_report": "train.json",
             "open_report": "open.json", "base_license": "base-MIT.txt",
             "code_license": "LICENSE", "data_notice": "DATA_NOTICE.md",
             "provenance_report": "provenance.json"}
    spec = {"schema_version": 1, "runtime_revision": "a" * 40,
            "inputs": {role: ref(root / name, root) for role, name in names.items()}}
    write_json(root / "stage.json", spec)
    return root, spec, checkpoint


def refresh(root, spec, role):
    spec["inputs"][role] = ref(root / spec["inputs"][role]["path"], root)


def test_stage_is_blocked_minimal_and_verified(package_case):
    root, _, _ = package_case
    result = release.stage(root / "stage.json", root, "dist/research")
    assert result["release_authorized"] is False
    assert result["production_ready"] is False
    assert "open_quality:phone safe_fpr" in result["blockers"]
    assert "independent_human_returns_missing" in result["blockers"]
    assert result["final_validation"]["existing_open_report_cannot_substitute"] is True
    assert "training_args.bin" not in result["files"]
    assert not (root / "dist/research/training_args.bin").exists()
    assert release.verify_package(root / "dist/research") == result
    assert "/Users/" not in (root / "dist/research" / release.MANIFEST).read_text()
    with pytest.raises(ValueError, match="overwrite"):
        release.stage(root / "stage.json", root, "dist/research")


@pytest.mark.parametrize("filename", sorted(release.RUNTIME_FILES))
def test_tampered_checkpoint_rejected(package_case, filename):
    root, spec, checkpoint = package_case
    with (checkpoint / filename).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError):
        release.inspect_inputs(spec, root)


@pytest.mark.parametrize("labels", [["SCAM", "UNCERTAIN", "SAFE"], ["SAFE", "SCAM"]])
def test_wrong_ordered_labels_rejected(package_case, labels):
    _, _, checkpoint = package_case
    path = checkpoint / "scamguard_calibration.json"
    record = release.read_json(path)
    record["labels"] = labels
    write_json(path, record)
    with pytest.raises(ValueError, match="ordered labels"):
        release.check_runtime(checkpoint)


@pytest.mark.parametrize("field", [
    "final_validation", "publication_authorized", "provenance_passed",
])
def test_no_synthetic_approval_or_open_as_final(package_case, field):
    root, spec, _ = package_case
    spec[field] = {"passed": True, "report": spec["inputs"]["open_report"]}
    with pytest.raises(ValueError, match="input contract"):
        release.inspect_inputs(spec, root)


@pytest.mark.parametrize("role", ["base_license", "provenance_report", "data_notice"])
def test_missing_license_or_provenance_rejected(package_case, role):
    root, spec, _ = package_case
    del spec["inputs"][role]
    with pytest.raises(ValueError, match="input roles"):
        release.inspect_inputs(spec, root)


def test_wrong_base_license_rejected_even_when_rehashed(package_case):
    root, spec, _ = package_case
    (root / "base-MIT.txt").write_text("Apache License, Version 2.0")
    refresh(root, spec, "base_license")
    with pytest.raises(ValueError, match="upstream MIT"):
        release.inspect_inputs(spec, root)


@pytest.mark.parametrize("review", [{"passed": True}, {"returned_csv": {"path": "missing.csv",
                                                                      "sha256": "a" * 64}}])
def test_human_approval_requires_real_returns(package_case, review):
    root, spec, _ = package_case
    spec["human_review"] = review
    with pytest.raises(ValueError, match="returned-audit"):
        release.inspect_inputs(spec, root)


def test_provenance_cannot_self_clear(package_case):
    root, spec, _ = package_case
    path = root / "provenance.json"
    record = release.read_json(path)
    record["rights_review_status"] = "cleared"
    write_json(path, record)
    refresh(root, spec, "provenance_report")
    with pytest.raises(ValueError, match="unverified clearance"):
        release.inspect_inputs(spec, root)


def test_provenance_must_cover_exact_train_sources(package_case):
    root, spec, _ = package_case
    path = root / "provenance.json"
    record = release.read_json(path)
    record["sources"][0]["rows"] = 2
    write_json(path, record)
    refresh(root, spec, "provenance_report")
    with pytest.raises(ValueError, match="source inventory"):
        release.inspect_inputs(spec, root)


def test_runtime_drift_is_blocker_not_claim_of_validation(package_case):
    root, spec, _ = package_case
    (root / "src/scamguard/preprocessing.py").write_text("# new unmeasured policy\n")
    result, _, _ = release.inspect_inputs(spec, root)
    assert result["current_runtime_source_drift"] == ["src/scamguard/preprocessing.py"]
    assert "current_sdk_differs_from_frozen_evaluation_runtime" in result["blockers"]


def test_wrong_frozen_runtime_rejected(package_case, monkeypatch):
    root, spec, _ = package_case
    monkeypatch.setattr(release, "frozen_source_hash", lambda *_: "f" * 64)
    with pytest.raises(ValueError, match="frozen serving source"):
        release.inspect_inputs(spec, root)


@pytest.mark.parametrize("path", ["../secret", "/tmp/secret", "dir/../../secret", "a\\b"])
def test_traversal_rejected(package_case, path):
    root, spec, _ = package_case
    spec["inputs"]["data_notice"]["path"] = path
    with pytest.raises(ValueError):
        release.inspect_inputs(spec, root)


def test_symlink_source_and_output_rejected(package_case):
    root, spec, _ = package_case
    (root / "notice-link").symlink_to(root / "DATA_NOTICE.md")
    spec["inputs"]["data_notice"]["path"] = "notice-link"
    with pytest.raises(ValueError, match="symlink"):
        release.inspect_inputs(spec, root)
    (root / "dist-link").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        release.stage(root / "stage.json", root, "dist-link/staged")


@pytest.mark.parametrize("extra", ["weights.bin", "custom_model.py", "train.jsonl", ".env"])
def test_checkpoint_extras_rejected(package_case, extra):
    root, spec, checkpoint = package_case
    (checkpoint / extra).write_text("unapproved")
    with pytest.raises(ValueError, match="unexpected files"):
        release.inspect_inputs(spec, root)


def test_custom_code_mapping_rejected(package_case):
    _, _, checkpoint = package_case
    path = checkpoint / "config.json"
    config = release.read_json(path)
    config["auto_map"] = {"AutoModel": "unsafe.Model"}
    write_json(path, config)
    with pytest.raises(ValueError, match="custom code"):
        release.check_runtime(checkpoint)


def test_pickle_named_safetensors_rejected(package_case):
    _, _, checkpoint = package_case
    (checkpoint / "model.safetensors").write_bytes(b"\x80\x04not-safetensors")
    with pytest.raises(ValueError, match="header"):
        release.check_safetensors(checkpoint / "model.safetensors")


def test_secret_like_notice_rejected(package_case):
    root, spec, _ = package_case
    (root / "DATA_NOTICE.md").write_text("hf_" + "X" * 30)
    refresh(root, spec, "data_notice")
    with pytest.raises(ValueError, match="credential"):
        release.inspect_inputs(spec, root)


def test_cannot_flip_authorization_on_staged_package(package_case):
    root, _, _ = package_case
    release.stage(root / "stage.json", root, "dist/staged")
    path = root / "dist/staged" / release.MANIFEST
    record = release.read_json(path)
    record["publication_authorized"] = True
    write_json(path, record)
    with pytest.raises(ValueError, match="cannot authorize"):
        release.verify_package(path.parent)


def test_staged_extra_and_tamper_rejected(package_case):
    root, _, _ = package_case
    release.stage(root / "stage.json", root, "dist/staged")
    package = root / "dist/staged"
    (package / "training_args.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="unexpected or missing"):
        release.verify_package(package)
    (package / "training_args.bin").unlink()
    with (package / "README.md").open("a") as handle:
        handle.write("tampered")
    with pytest.raises(ValueError, match="hash/size"):
        release.verify_package(package)


def test_even_open_pass_never_authorizes_release(package_case):
    root, spec, _ = package_case
    opened = release.read_json(root / "open.json")
    opened["metrics"]["phone"]["safe_fpr"] = 0.0
    config = release.read_json(root / "config.json")
    opened["promotion"] = release.check_promotion(
        opened["metrics"], opened["latency"], config["quality_promotion"])
    write_json(root / "open.json", opened)
    refresh(root, spec, "open_report")
    result, _, _ = release.inspect_inputs(spec, root)
    assert result["open_quality"]["passed"] is True
    assert result["release_authorized"] is False
    assert "encoder_final_validation_not_performed" in result["blockers"]


def test_forged_passed_summary_rejected(package_case):
    root, spec, _ = package_case
    opened = release.read_json(root / "open.json")
    opened["promotion"]["passed"] = True
    write_json(root / "open.json", opened)
    refresh(root, spec, "open_report")
    with pytest.raises(ValueError, match="summary is inconsistent"):
        release.inspect_inputs(spec, root)


def test_open_report_cannot_claim_sealed_test(package_case):
    root, spec, _ = package_case
    opened = release.read_json(root / "open.json")
    opened["sealed_tests_opened"] = True
    write_json(root / "open.json", opened)
    refresh(root, spec, "open_report")
    with pytest.raises(ValueError, match="misrepresents"):
        release.inspect_inputs(spec, root)


def test_incomplete_training_rejected(package_case):
    root, spec, _ = package_case
    train = release.read_json(root / "train.json")
    train["global_step"] = 1
    write_json(root / "train.json", train)
    refresh(root, spec, "training_report")
    with pytest.raises(ValueError, match="step bound"):
        release.inspect_inputs(spec, root)


def test_wrong_backend_dispatch_rejected(package_case):
    _, _, checkpoint = package_case
    path = checkpoint / "scamguard_calibration.json"
    calibration = release.read_json(path)
    calibration["backend_type"] = "qwen_verdict_likelihood"
    write_json(path, calibration)
    with pytest.raises(ValueError, match="backend/calibration"):
        release.check_runtime(checkpoint)


def test_invalid_runtime_revision_rejected_before_git(tmp_path):
    with pytest.raises(ValueError, match="full Git SHA"):
        release.frozen_source_hash(tmp_path, "main", "src/scamguard/model.py")


def test_direct_cli_help_needs_no_model_or_credentials():
    result = subprocess.run([sys.executable, str(Path(release.__file__)), "--help"],
                            capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "stage,verify,smoke" in result.stdout
