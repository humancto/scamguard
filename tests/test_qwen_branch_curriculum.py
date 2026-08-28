from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_qwen_branch_curriculum import build


def row(identifier: str, source: str, label: str) -> dict[str, object]:
    return {
        "id": identifier,
        "family_id": f"family-{identifier}",
        "source": source,
        "messages": [
            {"role": "user", "content": f"message {identifier}"},
            {"role": "assistant", "content": json.dumps({"verdict": label})},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_branch_curriculum_uses_training_only_target_rows_and_unchanged_dev(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    train = [
        row("u1", "mendeley_sms_phishing", "UNCERTAIN"),
        row("u2", "mendeley_sms_phishing", "UNCERTAIN"),
        row("s1", "mendeley_sms_phishing", "SAFE"),
        row("s2", "mendeley_sms_phishing", "SAFE"),
        row("s3", "mendeley_sms_phishing", "SAFE"),
        row("x1", "mendeley_sms_phishing", "SCAM"),
        row("r-safe", "other", "SAFE"),
        row("r-uncertain", "other", "UNCERTAIN"),
        row("r-scam", "other", "SCAM"),
    ]
    dev = [row("held", "mendeley_sms_phishing", "UNCERTAIN")]
    write_jsonl(parent / "train.jsonl", train)
    write_jsonl(parent / "dev.jsonl", dev)
    (parent / "manifest.json").write_text('{"schema_version":24}', encoding="utf-8")

    output = tmp_path / "output"
    manifest = build(
        parent,
        output,
        target_sources=("mendeley_sms_phishing",),
        retention_per_label=1,
        salt="fixture",
    )

    selected = {
        json.loads(line)["id"]
        for line in (output / "qwen_sft/train.jsonl").read_text().splitlines()
    }
    assert {"u1", "u2", "x1"}.issubset(selected)
    assert len(selected & {"s1", "s2", "s3"}) == 2
    assert "held" not in selected
    assert manifest["selection"]["held_rows_used_for_fitting"] == 0
    assert manifest["splits"]["dev"]["byte_identical_to_parent"] is True
    assert (output / "qwen_sft/dev.jsonl").read_bytes() == (parent / "dev.jsonl").read_bytes()


def test_branch_curriculum_is_deterministic(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    train = [row(f"safe-{index}", "other", "SAFE") for index in range(5)]
    train += [row(f"uncertain-{index}", "other", "UNCERTAIN") for index in range(5)]
    train += [row(f"scam-{index}", "other", "SCAM") for index in range(5)]
    write_jsonl(parent / "train.jsonl", train)
    write_jsonl(parent / "dev.jsonl", [row("dev", "other", "SAFE")])
    (parent / "manifest.json").write_text('{"schema_version":24}', encoding="utf-8")

    first = tmp_path / "first"
    second = tmp_path / "second"
    build(parent, first, target_sources=(), retention_per_label=2, salt="same")
    build(parent, second, target_sources=(), retention_per_label=2, salt="same")

    assert (first / "qwen_sft/train.jsonl").read_bytes() == (
        second / "qwen_sft/train.jsonl"
    ).read_bytes()


def test_branch_curriculum_rejects_train_dev_family_crossing(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    training = row("train", "other", "SAFE")
    held = row("dev", "other", "UNCERTAIN")
    held["family_id"] = training["family_id"]
    write_jsonl(parent / "train.jsonl", [training])
    write_jsonl(parent / "dev.jsonl", [held])
    (parent / "manifest.json").write_text('{"schema_version":24}', encoding="utf-8")

    with pytest.raises(ValueError, match="families"):
        build(parent, tmp_path / "output", target_sources=(), retention_per_label=1)
