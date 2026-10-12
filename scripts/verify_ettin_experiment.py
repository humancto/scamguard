#!/usr/bin/env python3
"""Read-only receipt/serving-parity verification; never infer, train, or authorize release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
REQUIRED_TRAIN_SOURCES = {
    "training/run_bounded_encoder.py",
    "training/train_encoder.py",
    "training/ettin_calibration.py",
    "benchmarks/eval_bounded_encoder.py",
    "src/scamguard/metrics.py",
    "src/scamguard/decision.py",
    "src/scamguard/preprocessing.py",
    "src/scamguard/model.py",
    "src/scamguard/scanner.py",
    "src/scamguard/signals.py",
}
REQUIRED_EVAL_SOURCES = {
    "src/scamguard/model.py", "src/scamguard/scanner.py",
    "src/scamguard/signals.py", "src/scamguard/preprocessing.py",
}
REQUIRED_CHECKPOINT_FILES = {
    "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
    "scamguard_calibration.json",
}
EVALUATOR = "benchmarks/eval_bounded_encoder.py"


class VerificationError(ValueError):
    """A fixed, text-free diagnostic safe to include in a public receipt."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def reject_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def parse_json(payload: str) -> Any:
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(path: Path) -> dict[str, Any]:
    value = parse_json(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON receipt must be an object")
    return value


def resolve_file(root: Path, value: str | Path) -> Path:
    require(isinstance(value, (str, Path)) and bool(str(value)), "missing artifact path")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    require(".." not in path.parts, "parent traversal in artifact path")
    try:
        relative = path.relative_to(root)
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path outside repository") from error
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), "symlink artifact path")
    require(path.is_file(), "missing artifact file")
    return path


def verify_hash(root: Path, path: str | Path, expected: str) -> Path:
    require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected) is not None,
            "invalid SHA-256 binding")
    resolved = resolve_file(root, path)
    require(sha256(resolved) == expected, "artifact hash mismatch")
    return resolved


def source_map(root: Path, recorded: Any, required: set[str]) -> dict[str, str]:
    require(isinstance(recorded, dict) and required <= set(recorded),
            "missing recorded source identity")
    for name, expected in recorded.items():
        verify_hash(root, name, expected)
    return dict(recorded)


def ledger(root: Path, binding: dict[str, Any]) -> list[dict[str, Any]]:
    path = verify_hash(root, binding["path"], binding["sha256"])
    rows = [parse_json(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    require(rows and len(rows) == binding["rows"], "prediction ledger row count mismatch")
    if "examples" in binding:
        require(binding["examples"] == len(rows), "prediction ledger examples mismatch")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        require(isinstance(row, dict) and "text" not in row, "invalid/text-bearing ledger row")
        require(isinstance(row.get("id"), str) and bool(row["id"]), "invalid ledger ID")
        require(isinstance(row.get("split"), str) and bool(row["split"]),
                "missing ledger split")
        key = (row["split"], row["id"])
        require(key not in seen, "duplicate prediction ID within split")
        seen.add(key)
        probabilities = row.get("probabilities")
        require(isinstance(probabilities, dict) and set(probabilities) == set(LABELS),
                "missing probability triple")
        require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for value in probabilities.values()), "invalid probability value")
        require(row.get("truth") in LABELS, "invalid reference label")
        require(isinstance(row.get("text_sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", row["text_sha256"]) is not None,
                "missing text hash binding")
    return rows


def verify_experiment(config_path: Path, root: Path) -> dict[str, Any]:
    """Check complete frozen receipts without loading model weights or corpus text."""
    root = root.resolve()
    config_path = resolve_file(root, config_path)
    config = read_json(config_path)
    config_hash = sha256(config_path)
    require(config["evaluation"]["primary_and_other_sealed_tests_may_open"] is False,
            "sealed evaluation forbidden")
    require(set(config["data"]) == {"train", "dev"}, "unexpected training data split")
    bindings = {"dev": config["data"]["dev"], **config["evaluation"]["splits"]}
    require("dev" not in config["evaluation"]["splits"], "duplicate development binding")
    for name, binding in {"train": config["data"]["train"], **bindings}.items():
        require(not any(word in name.lower() or word in str(binding["path"]).lower()
                        for word in ("sealed", "primary")), "sealed corpus access forbidden")
        require(type(binding.get("rows")) is int and binding["rows"] > 0,
                "invalid corpus row count")
        verify_hash(root, binding["path"], binding["sha256"])

    train_path = resolve_file(root, config["outputs"]["report"])
    open_path = resolve_file(root, config["outputs"]["evaluation"])
    train, opened = read_json(train_path), read_json(open_path)
    for record in (train, opened):
        require(record.get("status") == "completed", "experiment receipt incomplete")
        require(record.get("config_sha256") == config_hash, "config hash binding mismatch")
        require(record.get("experiment_id") == config["experiment_id"],
                "experiment ID mismatch")
        require(record.get("release_authorized") is False, "unexpected release authorization")
    require(train.get("training_completed") is True, "fit not completed")
    require(train.get("config") == config, "embedded training config mismatch")
    require(opened.get("sealed_tests_opened") is False, "sealed evaluation reported")
    require(opened.get("training_report_sha256") == sha256(train_path),
            "training receipt hash mismatch")
    for split, expected in config["data"].items():
        actual = train["data"][split]
        require(all(actual.get(key) == expected[key] for key in ("path", "sha256", "rows")),
                "training corpus receipt mismatch")
    require(opened.get("data_sha256") == {name: entry["sha256"]
                                          for name, entry in bindings.items()},
            "evaluation corpus receipt mismatch")

    train_sources = source_map(root, train.get("source_sha256"), REQUIRED_TRAIN_SOURCES)
    eval_sources = source_map(root, opened.get("source_code_sha256"), REQUIRED_EVAL_SOURCES)
    verify_hash(root, EVALUATOR, opened["evaluation_code_sha256"])
    for name in set(train_sources) & set(eval_sources):
        require(train_sources[name] == eval_sources[name], "train/evaluation source mismatch")
    if EVALUATOR in train_sources:
        require(train_sources[EVALUATOR] == opened["evaluation_code_sha256"],
                "evaluator changed since fitting")
    if "model" in config:
        require(train.get("base_model_files_sha256") == config["model"]["files_sha256"],
                "base model/tokenizer receipt mismatch")
    if "metadata_preflight" in train:
        preflight_ref = train["metadata_preflight"]
        preflight = read_json(verify_hash(root, preflight_ref["path"], preflight_ref["sha256"]))
        require(preflight.get("status") == "passed"
                and preflight.get("mode") == "CPU_metadata_only"
                and preflight.get("model_forward_performed") is False
                and preflight.get("training_performed") is False
                and preflight.get("release_authorized") is False,
                "invalid metadata preflight")
        require(all(preflight.get(key) == train.get(key) for key in (
            "config_sha256", "source_sha256", "data", "base_model_files_sha256",
            "effective_max_steps",
        )), "metadata preflight/fit binding mismatch")

    files = train.get("checkpoint_files_sha256")
    require(isinstance(files, dict) and REQUIRED_CHECKPOINT_FILES <= set(files),
            "incomplete checkpoint file bindings")
    checkpoint = Path(config["outputs"]["checkpoint"])
    for name, expected in files.items():
        require(isinstance(name, str) and Path(name).name == name and name not in (".", ".."),
                "invalid checkpoint filename")
        verify_hash(root, checkpoint / name, expected)
    checkpoint_path = resolve_file(root, checkpoint / "model.safetensors").parent
    require(set(files) == {path.name for path in checkpoint_path.iterdir() if path.is_file()},
            "unrecorded checkpoint file")
    for record in (train, opened):
        require(record.get("checkpoint_model_sha256") == files["model.safetensors"],
                "checkpoint model receipt mismatch")
        require(record.get("calibration_sha256") == files["scamguard_calibration.json"],
                "calibration receipt mismatch")
    calibration = read_json(checkpoint_path / "scamguard_calibration.json")
    require(train.get("calibration") == calibration, "embedded calibration mismatch")
    require(calibration.get("checkpoint_model_sha256") == files["model.safetensors"]
            and calibration.get("dev_data_sha256") == config["data"]["dev"]["sha256"],
            "calibration model/development binding mismatch")
    require(calibration.get("model_id") == config["experiment_id"]
            and calibration.get("labels") == list(LABELS), "calibration identity mismatch")
    transform = {key: config["training"][key]
                 for key in ("dialogue_policy", "truncation_side", "max_length")}
    require(calibration.get("input_transform") == transform, "calibration transform mismatch")
    require(opened.get("input_transform") == transform["dialogue_policy"],
            "evaluation transform mismatch")

    dev_rows = ledger(root, train["predictions"])
    open_binding = opened["prediction_ledger"]
    require(resolve_file(root, open_binding["path"])
            == resolve_file(root, config["outputs"]["predictions"]),
            "open ledger path mismatch")
    open_rows = ledger(root, open_binding)
    expected_counts = {name: entry["rows"] for name, entry in bindings.items()}
    require(Counter(row["split"] for row in open_rows) == expected_counts,
            "evaluation population coverage mismatch")
    require(set(opened["metrics"]) == set(bindings)
            and all(opened["metrics"][name]["examples"] == count
                    for name, count in expected_counts.items()),
            "reported population coverage mismatch")
    require(len(dev_rows) == expected_counts["dev"]
            and all(row["split"] == "dev" for row in dev_rows),
            "development ledger coverage mismatch")
    training_dev = {row["id"]: row for row in dev_rows}
    evaluated_dev = {row["id"]: row for row in open_rows if row["split"] == "dev"}
    require(set(training_dev) == set(evaluated_dev), "development IDs differ")
    for identifier, row in training_dev.items():
        evaluated = evaluated_dev[identifier]
        require(row["probabilities"] == evaluated["probabilities"],
                "development probabilities differ (exact comparison)")
        require(row.get("calibrated_verdict") in LABELS
                and row["calibrated_verdict"] == evaluated.get("verdict"),
                "development calibrated verdicts differ")
        require(all(row.get(key) == evaluated.get(key)
                    for key in ("truth", "text_sha256", "argmax", "threshold_scam")),
                "development prediction metadata differs")
    require(all(row.get("verdict") in LABELS for row in open_rows),
            "invalid evaluation verdict")
    return {
        "schema_version": 1, "status": "verified", "experiment_id": config["experiment_id"],
        "scope": ("receipt integrity and exact development serving parity only; "
                  "not accuracy or rights review"),
        "config_sha256": config_hash, "training_report_sha256": sha256(train_path),
        "open_report_sha256": sha256(open_path), "checkpoint_files_sha256": files,
        "training_ledger_sha256": train["predictions"]["sha256"],
        "open_ledger_sha256": open_binding["sha256"],
        "corpus_sha256": {"train": config["data"]["train"]["sha256"],
                          **{name: entry["sha256"] for name, entry in bindings.items()}},
        "source_sha256": {**train_sources, **eval_sources,
                          EVALUATOR: opened["evaluation_code_sha256"]},
        "development_exact_parity_rows": len(dev_rows), "population_rows": expected_counts,
        "release_authorized": False, "production_ready": False,
        "model_inference_performed": False, "sealed_tests_opened": False,
        "verifier_sha256": sha256(Path(__file__)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before verification; an existing success/failure receipt is never replaced.
    with args.output.open("x", encoding="utf-8") as handle:
        try:
            result = verify_experiment(args.config, args.repo_root)
            code = 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {"schema_version": 1, "status": "failed", "error_type": type(error).__name__,
                      "failure_reason": str(error) if isinstance(error, VerificationError)
                      else "Malformed, missing, or inaccessible experiment artifact",
                      "scope": "receipt integrity and exact development serving parity only",
                      "release_authorized": False, "production_ready": False,
                      "model_inference_performed": False, "sealed_tests_opened": False}
            # Do not serialize malformed source data, row identifiers, or raw exception content.
            code = 1
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
