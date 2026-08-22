from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from training.cache_encoder_teacher_logits import existing_cache


def write_complete_cache(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"frozen model fixture")
    data = tmp_path / "train.jsonl"
    data.write_text('{"id":"row"}\n')
    ledger = tmp_path / "teacher.jsonl"
    ledger.write_text('{"id":"row","logits":[1.0,0.0,-1.0]}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ledger_sha256": file_sha256(ledger),
                "checkpoint_model_sha256": file_sha256(
                    checkpoint / "model.safetensors"
                ),
                "data_sha256": file_sha256(data),
                "dialogue_policy": "speaker-neutral-v1",
            }
        )
    )
    return checkpoint, data, ledger, manifest


def test_existing_teacher_cache_is_reused_only_when_all_inputs_match(tmp_path: Path) -> None:
    checkpoint, data, ledger, manifest = write_complete_cache(tmp_path)

    loaded = existing_cache(
        ledger,
        manifest,
        checkpoint,
        data,
        "speaker-neutral-v1",
    )

    assert loaded is not None
    assert loaded["ledger_sha256"] == file_sha256(ledger)
    data.write_text('{"id":"changed"}\n')
    with pytest.raises(RuntimeError, match="differs from request"):
        existing_cache(
            ledger,
            manifest,
            checkpoint,
            data,
            "speaker-neutral-v1",
        )


def test_partial_teacher_cache_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    ledger = tmp_path / "teacher.jsonl"
    ledger.write_text("{}\n")

    with pytest.raises(RuntimeError, match="partial"):
        existing_cache(
            ledger,
            tmp_path / "missing-manifest.json",
            checkpoint,
            data,
            "speaker-neutral-v1",
        )
