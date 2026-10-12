import hashlib
import json
import subprocess
import sys

import pytest

from scripts.run_ettin_v3_once import (
    check_open_separation,
    require_completed_training,
    sha,
    terminal_step,
    validate_limits,
    verify_freeze,
    write_new,
)


def test_execution_limits_are_enforced():
    config = {"training": {"batch_size": 16, "max_steps": 2852, "epochs": 2},
              "data": {"train": {"rows": 22811}}, "execution": {
                  "effective_max_steps": 2852,
                  "train_process_before_terminal_step_seconds": 3600,
                  "post_fit_process_seconds": 900, "train_process_total_seconds": 4500,
                  "evaluation_process_seconds": 1800}}
    validate_limits(config)
    config["execution"]["effective_max_steps"] += 1
    with pytest.raises(ValueError, match="limits"):
        validate_limits(config)
    config["execution"]["effective_max_steps"] -= 1
    config["execution"]["train_process_total_seconds"] = 4501
    with pytest.raises(ValueError, match="limits"):
        validate_limits(config)


def test_exclusive_attempt_is_not_overwritten(tmp_path):
    path = tmp_path / "attempt.json"
    write_new(path, {"retry_allowed": False})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {})
    assert path.read_bytes() == before


@pytest.mark.parametrize("value,passed", [
    ({"global_step": 10, "stop_reason": "step_bound_reached"}, True),
    ({"global_step": 10, "stop_reason": "wall_limit_reached"}, False),
    ({"global_step": 9, "stop_reason": "step_bound_reached"}, False),
    ({"global_step": 10}, False),
])
def test_terminal_step_requires_exact_success(tmp_path, value, passed):
    path = tmp_path / "progress.json"
    write_new(path, value)
    assert terminal_step(path, 10) is passed


def test_missing_or_partial_progress_is_not_terminal(tmp_path):
    path = tmp_path / "progress.json"
    assert not terminal_step(path, 10)
    path.write_text("{")
    assert not terminal_step(path, 10)


def test_freeze_detects_source_and_config_drift(tmp_path):
    config = tmp_path / "config.json"
    source = tmp_path / "source.py"
    config.write_text("{}")
    source.write_text("# frozen")
    frozen = tmp_path / "freeze.json"
    frozen.write_text(json.dumps({"config_path": str(config), "files_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (config, source)
    }}))
    verify_freeze(config, frozen)
    source.write_text("# changed")
    with pytest.raises(ValueError, match="source changed"):
        verify_freeze(config, frozen)
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="configuration content"):
        verify_freeze(config, frozen)


def test_supervisor_kills_hung_child_and_refuses_retry(tmp_path, monkeypatch):
    from scripts import run_ettin_v3_once as runner

    config_path, frozen = tmp_path / "config.json", tmp_path / "freeze.json"
    report = tmp_path / "run.train.json"
    write_new(frozen, {})
    write_new(config_path, {
        "outputs": {"report": str(report)},
        "training": {"batch_size": 16, "max_steps": 2, "epochs": 2},
        "data": {"train": {"rows": 8}},
        "execution": {"effective_max_steps": 2,
                      "train_process_before_terminal_step_seconds": 1,
                      "post_fit_process_seconds": 1, "train_process_total_seconds": 2,
                      "evaluation_process_seconds": 1},
    })
    monkeypatch.setattr(runner, "verify_freeze", lambda *_: None)
    original_popen = subprocess.Popen
    children = []

    def inert_child(_command, **kwargs):
        child = original_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", inert_child)
    assert runner.supervise(config_path, frozen, "train") == 124
    assert len(children) == 1 and children[0].poll() is not None
    receipt = json.loads((tmp_path / "run.train.supervisor.json").read_text())
    assert receipt["status"] == "timeout"
    with pytest.raises(FileExistsError):
        runner.supervise(config_path, frozen, "train")
    assert len(children) == 1


@pytest.mark.parametrize("mutation", ["timeout", "returncode", "elapsed", "freeze", "report"])
def test_evaluation_rejects_incomplete_or_stale_supervision(tmp_path, mutation):
    report, frozen = tmp_path / "run.train.json", tmp_path / "freeze.json"
    write_new(report, {"status": "completed", "training_completed": True, "global_step": 10})
    write_new(frozen, {})
    write_new(tmp_path / "run.train.attempt.json", {"freeze_sha256": sha(frozen)})
    supervisor = {"status": "completed", "returncode": 0, "elapsed_seconds": 20,
                  "report_sha256": sha(report), "freeze_sha256": sha(frozen)}
    path = tmp_path / "run.train.supervisor.json"
    write_new(path, supervisor)
    limits = {"effective_max_steps": 10, "train_process_total_seconds": 100}
    require_completed_training(report, frozen, limits)
    field, value = {"timeout": ("status", "timeout"), "returncode": ("returncode", 1),
                    "elapsed": ("elapsed_seconds", 101), "freeze": ("freeze_sha256", "bad"),
                    "report": ("report_sha256", "bad")}[mutation]
    supervisor[field] = value
    path.write_text(json.dumps(supervisor))
    with pytest.raises(ValueError, match="supervisor evidence"):
        require_completed_training(report, frozen, limits)


def test_open_separation_checks_normalized_text(tmp_path):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps({"id": "t", "text": "HELLO  there", "family_id": "t"}) + "\n")
    dev.write_text(json.dumps({"id": "d", "text": "hello there", "family_id": "d"}) + "\n")
    config = {"data": {"train": {"path": str(train), "rows": 1},
                       "dev": {"path": str(dev), "rows": 1}},
              "evaluation": {"primary_and_other_sealed_tests_may_open": False, "splits": {}}}
    with pytest.raises(ValueError, match="training overlap"):
        check_open_separation(config)
