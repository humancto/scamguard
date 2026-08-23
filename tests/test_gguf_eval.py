"""Binary task serialization and patched llama.cpp score parsing are deterministic."""

import struct

import numpy as np
import pytest

from training.eval_gguf import (
    parse_scores,
    quantization_parity,
    resolve_split_path,
    write_tasks,
)


def test_multiple_choice_task_header_contains_absolute_offsets(tmp_path) -> None:
    path = tmp_path / "tasks.bin"
    write_tasks(path, ["prompt-a", "prompt-b"], ["SAFE", "SCAM"])
    data = path.read_bytes()

    task_count, first, second = struct.unpack("<III", data[:12])

    assert task_count == 2
    assert first == 12
    assert first < second < len(data)


def test_parse_llama_scores() -> None:
    output = "noise\n1\t100.00000000\t-0.1\t-2.0\t-3.0\n2\t50.00000000\t-3\t-2\t-0.1\n"

    scores = parse_scores(output, expected=2)

    np.testing.assert_allclose(scores, [[-0.1, -2.0, -3.0], [-3.0, -2.0, -0.1]])


def ledger_record(identifier: str, verdict: str, safe: float) -> dict[str, object]:
    return {
        "id": identifier,
        "split": "test",
        "truth": "SAFE",
        "source": "fixture",
        "source_language": "en",
        "category": "NONE",
        "argmax": verdict,
        "calibrated_verdict": verdict,
        "probabilities": {
            "SAFE": safe,
            "UNCERTAIN": 0.1,
            "SCAM": 0.9 - safe,
        },
    }


def test_quantization_parity_requires_exact_calibrated_decisions() -> None:
    reference = [ledger_record("a", "SAFE", 0.8)]
    candidate = [ledger_record("a", "UNCERTAIN", 0.79)]

    result = quantization_parity(candidate, reference)

    assert result["maximum_absolute_probability_error"] == pytest.approx(0.01)
    assert result["calibrated_verdict_mismatch_count"] == 1
    assert result["release_gate_passed"] is False


def test_quantization_parity_rejects_metadata_or_key_drift() -> None:
    reference = [ledger_record("a", "SAFE", 0.8)]
    wrong_key = [ledger_record("b", "SAFE", 0.8)]
    with pytest.raises(ValueError, match="prediction keys differ"):
        quantization_parity(wrong_key, reference)

    wrong_metadata = [ledger_record("a", "SAFE", 0.8)]
    wrong_metadata[0]["source"] = "other"
    with pytest.raises(ValueError, match="metadata differ"):
        quantization_parity(wrong_metadata, reference)


def test_split_resolution_includes_external_holdouts(tmp_path) -> None:
    data = tmp_path / "data"
    external = tmp_path / "external"
    data.mkdir()
    (data / "test.jsonl").write_text("{}\n", encoding="utf-8")
    chichewa = external / "chichewa" / "ood_chichewa.jsonl"
    chichewa.parent.mkdir(parents=True)
    chichewa.write_text("{}\n", encoding="utf-8")

    assert resolve_split_path(data, external, "test") == data / "test.jsonl"
    assert resolve_split_path(data, external, "ood_chichewa") == chichewa
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_split_path(data, external, "missing")
