#!/usr/bin/env python3
"""Evaluate one frozen encoder through the real SDK on declared OPEN slices only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from scamguard.metrics import file_sha256, wilson_interval
from scamguard.model import TransformersBackend
from scamguard.scanner import Scanner

LABELS = ("SAFE", "UNCERTAIN", "SCAM")


def write_new(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_pinned_rows(binding: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(binding["path"])
    if file_sha256(path) != binding["sha256"]:
        raise ValueError(f"data hash mismatch: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != binding["rows"] or not rows:
        raise ValueError(f"row count mismatch: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs: {path}")
    if any(row["label"] not in LABELS or not str(row["text"]).strip() for row in rows):
        raise ValueError(f"invalid verdict or empty input: {path}")
    return rows


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(records: list[dict[str, Any]], *, slices: bool = True) -> dict[str, Any]:
    if not records:
        raise ValueError("empty metric population")
    truth = [row["truth"] for row in records]
    verdicts = [row["verdict"] for row in records]
    scams = [row for row in records if row["truth"] == "SCAM"]
    safe = [row for row in records if row["truth"] == "SAFE"]
    uncertain = [row for row in records if row["truth"] == "UNCERTAIN"]
    tp = sum(row["verdict"] == "SCAM" for row in scams)
    fp = sum(row["verdict"] == "SCAM" for row in safe)
    alerts = sum(row["verdict"] == "SCAM" for row in records)
    result: dict[str, Any] = {
        "examples": len(records),
        "truth_counts": dict(Counter(truth)),
        "verdict_counts": dict(Counter(verdicts)),
        "macro_f1": float(f1_score(truth, verdicts, labels=LABELS,
                                    average="macro", zero_division=0)),
        "macro_f1_argmax": float(f1_score(truth, [r["argmax"] for r in records],
                                           labels=LABELS, average="macro", zero_division=0)),
        "confusion_labels": list(LABELS),
        "confusion": confusion_matrix(truth, verdicts, labels=LABELS).tolist(),
        "scam_recall": safe_rate(tp, len(scams)),
        "scam_recall_ci95": wilson_interval(tp, len(scams)) if scams else None,
        "safe_fpr": safe_rate(fp, len(safe)),
        "safe_fpr_ci95": wilson_interval(fp, len(safe)) if safe else None,
        "binary_precision": safe_rate(tp, tp + fp),
        "known_scam_fraction_of_alerts": safe_rate(tp, alerts),
        "scams_marked_safe": sum(row["verdict"] == "SAFE" for row in scams),
        "scams_deferred": sum(row["verdict"] == "UNCERTAIN" for row in scams),
        "safe_deferred": sum(row["verdict"] == "UNCERTAIN" for row in safe),
        "uncertain_recall": safe_rate(
            sum(row["verdict"] == "UNCERTAIN" for row in uncertain), len(uncertain)
        ),
        "uncertain_marked_scam": sum(row["verdict"] == "SCAM" for row in uncertain),
        "invalid_evidence_spans": sum(row["invalid_evidence_spans"] for row in records),
    }
    if slices:
        for field in ("source", "category", "source_language"):
            values = sorted({str(row.get(field) or "UNSPECIFIED") for row in records})
            result[f"by_{field}"] = {
                value: metrics([r for r in records if
                                str(r.get(field) or "UNSPECIFIED") == value], slices=False)
                for value in values
            }
    return result


def check_promotion(
    summaries: dict[str, Any], latency: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    def gate(name: str, actual: Any, threshold: float, operator: str) -> None:
        numeric = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        passed = numeric and math.isfinite(actual) and (
            actual >= threshold if operator == ">=" else actual <= threshold
        )
        gates.append({"name": name, "actual": actual, "operator": operator,
                      "threshold": threshold, "passed": bool(passed)})

    for split in ("dev", "test"):
        gate(f"{split} scam recall", summaries[split]["scam_recall"],
             policy[f"{split}_scam_recall_min"], ">=")
        gate(f"{split} SAFE FPR", summaries[split]["safe_fpr"],
             policy[f"{split}_safe_fpr_max"], "<=")
    categories = summaries["test"]["by_category"]
    eligible = {name: value for name, value in categories.items()
                if value["truth_counts"].get("SCAM", 0) >= policy["test_category_min_examples"]}
    if not eligible:
        gate("core category coverage", None, 1, ">=")
    for name, value in eligible.items():
        gate(f"test category {name} recall", value["scam_recall"],
             policy["test_core_category_recall_min"], ">=")
    for split in policy["required_dialogue_splits"]:
        summary = summaries[split]
        for metric, truth, threshold, operator in (
            ("scam_recall", "SCAM", policy["dialogue_scam_recall_min"], ">="),
            ("safe_fpr", "SAFE", policy["dialogue_safe_fpr_max"], "<="),
        ):
            if summary["truth_counts"].get(truth, 0):
                gate(f"{split} {metric}", summary[metric], threshold, operator)
    gate("desktop full SDK p95 ms", latency["p95_ms"], latency["limit_ms"], "<=")
    gate("invalid evidence spans", sum(s["invalid_evidence_spans"] for s in summaries.values()),
         0, "<=")
    return {
        "passed": all(g["passed"] for g in gates), "gates": gates,
        "failed_gates": [g["name"] for g in gates if not g["passed"]],
        "macro_f1_stretch": {"required": False, "target": policy["macro_f1_stretch"],
                             "actual": summaries["test"]["macro_f1"]},
        "meaning": "open-screen eligibility only; not final quality or release authorization",
    }


class RecordingBackend:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.last_scores: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def predict(self, text: str) -> Any:
        self.last_scores = self.backend.predict(text)
        return self.last_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report_path = Path(config["outputs"]["evaluation"])
    ledger_path = Path(config["outputs"]["predictions"])
    for path in (report_path, ledger_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite experiment evidence: {path}")
    train_report_path = Path(config["outputs"]["report"])
    train = json.loads(train_report_path.read_text())
    checkpoint = Path(config["outputs"]["checkpoint"])
    config_sha = file_sha256(args.config)
    model_sha = file_sha256(checkpoint / "model.safetensors")
    calibration_sha = file_sha256(checkpoint / "scamguard_calibration.json")
    if (train.get("status") != "completed" or train.get("config_sha256") != config_sha
            or train.get("checkpoint_model_sha256") != model_sha
            or train.get("calibration_sha256") != calibration_sha):
        raise ValueError("training completion/config/checkpoint/calibration binding mismatch")
    checkpoint_files = train.get("checkpoint_files_sha256", {})
    required_files = {"config.json", "tokenizer.json", "tokenizer_config.json",
                      "model.safetensors", "scamguard_calibration.json"}
    if not required_files <= set(checkpoint_files) or any(
        file_sha256(checkpoint / name) != expected for name, expected in checkpoint_files.items()
    ):
        raise ValueError("serving checkpoint/tokenizer files differ from the training receipt")
    if config["evaluation"]["primary_and_other_sealed_tests_may_open"] is not False:
        raise ValueError("this evaluator cannot authorize sealed tests")
    bindings = {"dev": config["data"]["dev"], **config["evaluation"]["splits"]}
    if any("primary" in name or "sealed" in name for name in bindings):
        raise ValueError("sealed split in open evaluator")
    rows = {name: read_pinned_rows(binding) for name, binding in bindings.items()}
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("frozen MPS benchmark device unavailable")
    started = time.monotonic()
    deadline = started + config["evaluation"]["max_wall_seconds"]
    backend = RecordingBackend(TransformersBackend(checkpoint, device="mps"))
    scanner = Scanner(backend=backend)
    summaries: dict[str, Any] = {}
    all_counts = 0
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"], "config_sha256": config_sha,
        "training_report_sha256": file_sha256(train_report_path),
        "checkpoint_model_sha256": model_sha, "calibration_sha256": calibration_sha,
        "data_sha256": {name: binding["sha256"] for name, binding in bindings.items()},
        "evaluation_code_sha256": file_sha256(Path(__file__)),
        "source_code_sha256": {str(p): file_sha256(p) for p in
                               map(Path, ("src/scamguard/model.py", "src/scamguard/scanner.py",
                                          "src/scamguard/signals.py",
                                          "src/scamguard/preprocessing.py"))},
        "environment": {"python": platform.python_version(), "arch": platform.machine(),
                        "torch": torch.__version__, "device": "mps"},
        "release_authorized": False, "sota_established": False,
        "sealed_tests_opened": False, "status": "running",
        "input_transform": config["training"]["dialogue_policy"],
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ledger_path.open("x", encoding="utf-8") as ledger:
            for split, split_rows in rows.items():
                records = []
                for row in split_rows:
                    if time.monotonic() > deadline:
                        raise TimeoutError("frozen open evaluation wall budget exhausted")
                    text = str(row["text"])
                    scan = scanner.scan(text)
                    scores = backend.last_scores
                    probabilities = {"SAFE": scores.safe, "UNCERTAIN": scores.uncertain,
                                     "SCAM": scores.scam}
                    if not all(math.isfinite(v) and 0 <= v <= 1
                               for v in probabilities.values()):
                        raise ValueError("non-finite/out-of-range probability")
                    stripped = text.strip()
                    invalid = sum(stripped[span.start:span.end] != span.text
                                  for span in scan.evidence_spans)
                    record = {
                        "id": str(row["id"]), "split": split, "truth": row["label"],
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "verdict": scan.verdict.value, "probabilities": probabilities,
                        "argmax": max(probabilities, key=probabilities.get),
                        "threshold_scam": scan.verdict.value == "SCAM",
                        "source": row.get("source"), "category": row.get("category"),
                        "source_language": row.get("source_language"),
                        "invalid_evidence_spans": invalid,
                    }
                    records.append(record)
                    ledger.write(json.dumps(record, allow_nan=False) + "\n")
                    all_counts += 1
                ledger.flush()
                summaries[split] = metrics(records)
                print(json.dumps({"split": split, "rows": len(records),
                                  "scam_recall": summaries[split]["scam_recall"],
                                  "safe_fpr": summaries[split]["safe_fpr"],
                                  "macro_f1": summaries[split]["macro_f1"]}), flush=True)
            settings = config["evaluation"]["latency"]
            pool = sorted(rows[settings["source_split"]], key=lambda row: str(row["id"]))
            selected = random.Random(settings["seed"]).sample(pool, settings["rows"])
            for row in selected[:settings["warmup"]]:
                scanner.scan(str(row["text"]))
            torch.mps.synchronize()
            trace = []
            for round_index in range(settings["rounds"]):
                for row in selected:
                    if time.monotonic() > deadline:
                        raise TimeoutError("frozen latency wall budget exhausted")
                    torch.mps.synchronize()
                    tick = time.perf_counter_ns()
                    scanner.scan(str(row["text"]))
                    torch.mps.synchronize()
                    trace.append({"id": str(row["id"]), "round": round_index,
                                  "milliseconds": (time.perf_counter_ns() - tick) / 1e6})
            times = [point["milliseconds"] for point in trace]
            latency = {"p50_ms": float(np.percentile(times, 50)),
                       "p95_ms": float(np.percentile(times, 95)),
                       "p99_ms": float(np.percentile(times, 99)),
                       "limit_ms": settings["p95_ms_max"], "trace": trace,
                       "scope": settings["scope"], "physical_mobile": False}
            result.update({"status": "completed", "metrics": summaries, "latency": latency,
                           "promotion": check_promotion(summaries, latency,
                                                        config["quality_promotion"])})
    except Exception as error:
        result.update({"status": "incomplete", "error": str(error), "metrics": summaries})
        raise
    finally:
        scanner.close()
        result.update({"elapsed_seconds": time.monotonic() - started,
                       "prediction_ledger": {"path": str(ledger_path), "rows": all_counts,
                                             "examples": all_counts,
                                             "sha256": file_sha256(ledger_path)},
                       "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "artifact_bytes": sum(p.stat().st_size for p in checkpoint.rglob("*")
                                             if p.is_file())})
        write_new(report_path, result)
    print(json.dumps({"report": str(report_path), "status": result["status"],
                      "promotion_passed": result["promotion"]["passed"]}))


if __name__ == "__main__":
    main()
