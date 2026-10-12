#!/usr/bin/env python3
"""Freeze and supervise one offline experiment; exclusive attempts prohibit retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def bindings(config_path: Path) -> dict[str, str]:
    config = read(config_path)
    result = {str(config_path): sha(config_path)}
    for entry in [*config["data"].values(), *config["evaluation"]["splits"].values()]:
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("frozen corpus or open evaluation input changed")
        result[entry["path"]] = entry["sha256"]
    corpus = config["corpus_receipt"]
    if sha(Path(corpus["path"])) != corpus["sha256"]:
        raise ValueError("corpus receipt changed")
    result[corpus["path"]] = corpus["sha256"]
    return result


def validate_limits(config: dict) -> None:
    settings, execution = config["training"], config["execution"]
    if any(type(value) is not int or value <= 0 for value in execution.values()):
        raise ValueError("execution bounds must be positive integers")
    steps = math.ceil(config["data"]["train"]["rows"] / settings["batch_size"])
    steps = min(settings["max_steps"], steps * settings["epochs"])
    if (execution["effective_max_steps"] != steps
            or execution["train_process_before_terminal_step_seconds"] > 3600
            or execution["post_fit_process_seconds"] > 900
            or execution["train_process_total_seconds"] > 4500
            or execution["evaluation_process_seconds"] > 1800):
        raise ValueError("execution limits exceed the final experiment contract")


def check_open_separation(config: dict) -> dict:
    """Inspect only already-open corpora; no text or IDs enter the receipt."""
    def keys(binding):
        rows = [json.loads(line) for line in Path(binding["path"]).read_text().splitlines()]
        if len(rows) != binding["rows"]:
            raise ValueError("corpus row count mismatch")
        return {
            "ids": {str(row["id"]) for row in rows},
            "normalized_texts": {
                " ".join(unicodedata.normalize("NFKC", row["text"]).casefold().split())
                for row in rows},
            "families": {str(row[key]) for row in rows
                         for key in ("family_id", "near_template_family_id") if row.get(key)},
        }

    if config["evaluation"]["primary_and_other_sealed_tests_may_open"] is not False:
        raise ValueError("sealed evaluation is forbidden")
    train = keys(config["data"]["train"])
    counts = {}
    for name, binding in {"dev": config["data"]["dev"],
                          **config["evaluation"]["splits"]}.items():
        if any(word in name.lower() or word in binding["path"].lower()
               for word in ("primary", "sealed")):
            raise ValueError("sealed input forbidden")
        other = keys(binding)
        counts[name] = {field: len(values & other[field]) for field, values in train.items()}
        if any(counts[name].values()):
            raise ValueError(f"training overlap with declared {name}")
    return counts


def freeze(config_path: Path, destination: Path) -> None:
    config = read(config_path)
    validate_limits(config)
    report = Path(config["outputs"]["report"])
    preflight = report.with_name(report.stem + ".preflight.json")
    receipt = read(preflight)
    if receipt.get("status") != "passed" or receipt.get("config_sha256") != sha(config_path):
        raise ValueError("matching successful metadata preflight required")
    input_bindings = bindings(config_path)
    separation = check_open_separation(config)
    sources = sorted({
        *Path("src/scamguard").glob("*.py"),
        *Path("training").glob("*.py"),
        Path("benchmarks/eval_bounded_encoder.py"),
        Path("scripts/build_ettin_v3_corpus.py"),
        Path("scripts/verify_ettin_experiment.py"),
        Path(__file__).relative_to(Path.cwd()),
    })
    write_new(destination, {
        "experiment_id": config["experiment_id"],
        "created_at_unix": time.time(),
        "config_path": str(config_path),
        "files_sha256": {
            **input_bindings, str(preflight): sha(preflight),
            **{str(path): sha(path) for path in sources},
        },
        "release_authorized": False,
        "sealed_tests_opened": False,
        "training_overlap_counts": separation,
        "overlap_scope": "exact IDs, NFKC casefold whitespace-normalized text and family IDs; "
                         "not semantic or pretraining contamination clearance",
    })


def verify_freeze(config_path: Path, freeze_path: Path) -> None:
    frozen = read(freeze_path)
    if frozen["config_path"] != str(config_path):
        raise ValueError("freeze configuration path mismatch")
    if frozen["files_sha256"].get(str(config_path)) != sha(config_path):
        raise ValueError("freeze configuration content mismatch")
    for name, expected in frozen["files_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"frozen input/source changed: {name}")


def terminal_step(path: Path, steps: int) -> bool:
    try:
        progress = read(path)
    except (OSError, ValueError):
        return False
    return (progress.get("global_step") == steps
            and progress.get("stop_reason") == "step_bound_reached")


def require_completed_training(report: Path, freeze_path: Path, execution: dict) -> None:
    attempt = read(report.with_name(report.stem + ".attempt.json"))
    supervisor = read(report.with_name(report.stem + ".supervisor.json"))
    train = read(report)
    if (train.get("status") != "completed" or train.get("training_completed") is not True
            or train.get("global_step") != execution["effective_max_steps"]
            or supervisor.get("status") != "completed" or supervisor.get("returncode") != 0
            or not 0 <= supervisor.get("elapsed_seconds", float("inf"))
            <= execution["train_process_total_seconds"]
            or supervisor.get("report_sha256") != sha(report)
            or supervisor.get("freeze_sha256") != sha(freeze_path)
            or attempt.get("freeze_sha256") != sha(freeze_path)):
        raise ValueError("successful bounded training supervisor evidence required")


def supervise(config_path: Path, freeze_path: Path, phase: str) -> int:
    verify_freeze(config_path, freeze_path)
    config = read(config_path)
    validate_limits(config)
    execution = config["execution"]
    train_report = Path(config["outputs"]["report"])
    report = (train_report if phase == "train" else Path(config["outputs"]["evaluation"]))
    attempt = report.with_name(report.stem + ".attempt.json")
    final = report.with_name(report.stem + ".supervisor.json")
    log = report.with_name(report.stem + ".log")
    if report.exists() or log.exists() or final.exists():
        raise FileExistsError("refusing to overwrite experiment evidence")
    if phase == "evaluate":
        require_completed_training(train_report, freeze_path, execution)
    command = [sys.executable, "-u", "-m",
               "training.run_bounded_encoder" if phase == "train"
               else "benchmarks.eval_bounded_encoder", "--config", str(config_path)]
    start = time.monotonic()
    write_new(attempt, {"command": command, "started_at_unix": time.time(),
                        "freeze_sha256": sha(freeze_path), "retry_allowed": False})
    budget = (execution["train_process_before_terminal_step_seconds"] if phase == "train"
              else execution["evaluation_process_seconds"])
    deadline = start + budget
    terminal_seen = False
    timed_out = False
    result = {"phase": phase, "release_authorized": False, "status": "failed",
              "freeze_sha256": sha(freeze_path)}
    process = None
    try:
        with log.open("x") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True, env={
                                           **os.environ, "HF_HUB_OFFLINE": "1",
                                           "TRANSFORMERS_OFFLINE": "1",
                                           "TOKENIZERS_PARALLELISM": "false",
                                       })
            progress = train_report.with_name(train_report.stem + ".progress.json")
            while process.poll() is None:
                now = time.monotonic()
                # A terminal step must be observed before the original deadline.
                if (phase == "train" and not terminal_seen and now < deadline
                        and terminal_step(progress, execution["effective_max_steps"])):
                    terminal_seen = True
                    deadline = min(start + execution["train_process_total_seconds"],
                                   now + execution["post_fit_process_seconds"])
                if now >= deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.25)
            returncode = process.wait()
        verify_freeze(config_path, freeze_path)
        if not timed_out and returncode == 0:
            if not report.exists() or read(report).get("status") != "completed":
                raise ValueError("child returned success without completed evidence")
            result["report_sha256"] = sha(report)
        status = "timeout" if timed_out else ("completed" if returncode == 0 else "failed")
        result.update(status=status,
                      returncode=returncode, terminal_step_observed=terminal_seen,
                      log_sha256=sha(log))
        return 124 if timed_out else returncode
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result["elapsed_seconds"] = time.monotonic() - start
        write_new(final, result)
        print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "train", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.config, args.freeze)
    else:
        raise SystemExit(supervise(args.config, args.freeze, args.phase))


if __name__ == "__main__":
    main()
