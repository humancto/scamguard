from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.select_qwen08_batch_geometry import select_geometry


def matrix() -> list[dict[str, object]]:
    return [
        {
            "microbatch_size": microbatch,
            "gradient_accumulation": 16 // microbatch,
            "forward_backward_seconds": elapsed,
            "mps_driver_allocated_bytes": memory,
            "memory_gate_passed": memory <= 50,
        }
        for microbatch, elapsed, memory in (
            (1, 10.8, 12),
            (2, 9.9, 22),
            (4, 9.7, 43),
            (8, 10.1, 82),
            (16, 132.0, 158),
        )
    ]


def test_selects_fastest_candidate_within_memory_gate() -> None:
    selected = select_geometry(matrix())

    assert selected["microbatch_size"] == 4
    assert selected["gradient_accumulation"] == 4


def test_rejects_incomplete_matrix_or_no_memory_pass() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        select_geometry(matrix()[:-1])
    candidates = matrix()
    for candidate in candidates:
        candidate["memory_gate_passed"] = False
    with pytest.raises(ValueError, match="no batch geometry"):
        select_geometry(candidates)


def test_tracked_selection_is_bound_and_preserves_effective_batch() -> None:
    repository = Path(__file__).resolve().parents[1]
    report = json.loads(
        (repository / "reports" / "QWEN08_BATCH_GEOMETRY_SELECTION.json").read_text(
            encoding="utf-8"
        )
    )
    selected = report["selected"]
    bindings = report["source_bindings"]

    assert selected["microbatch_size"] == 4
    assert selected["gradient_accumulation"] == 4
    assert selected["effective_batch_size"] == 16
    assert report["quality_contract"]["optimizer_semantics_changed"] is False
    assert bindings["source_commit"] == "33bb42c8f355eaade20a8094d0fe4409528c7a69"
    assert bindings["selector_sha256"] == file_sha256(
        repository / "scripts" / "select_qwen08_batch_geometry.py"
    )
    assert bindings["batch_preflight_sha256"] == file_sha256(
        repository / "scripts" / "preflight_qwen08_batch.py"
    )
    assert bindings["experiment_freezer_before_selection_sha256"] == (
        "6d3545786f7136ac63a110b421be8145d798f8b9f2646e2f40269981461e17a4"
    )
    assert bindings["training_launcher_before_selection_sha256"] == (
        "06e4a51ef7c682b4a35a0c5247e0360fe6f8147927cc7b737ab6c96347423663"
    )
