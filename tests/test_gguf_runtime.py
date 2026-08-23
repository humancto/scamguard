from __future__ import annotations

import hashlib
import os
import textwrap
from pathlib import Path

import pytest

from benchmarks.benchmark_routed_gguf_runtime import validate_quantized_evidence
from scamguard.gguf_runtime import PersistentGGUFScorer, calibrated_probabilities


def test_calibrated_probabilities_are_stable_and_normalized() -> None:
    probabilities = calibrated_probabilities((1_000.0, 999.0, 998.0), 2.0)

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_persistent_scorer_validates_and_parses_native_protocol(tmp_path: Path) -> None:
    runner = tmp_path / "fake-runner"
    runner.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys

            prefix_enabled = "--prefix-hex" in sys.argv
            prefix_tokens = 7 if prefix_enabled else 0
            print(f"READY\\t2\\t1234\\t640\\t{prefix_tokens}", flush=True)
            for line in sys.stdin:
                line = line.rstrip("\\n")
                if line == "QUIT":
                    break
                identifier, _question = line.split("\\t", 1)
                reused = 1 if prefix_enabled else 0
                print(
                    f"RESULT\\t{identifier}\\t-1.0\\t-2.0\\t-3.0\\t1234\\t42"
                    f"\\t{reused}\\t{prefix_tokens}",
                    flush=True,
                )
            """
        ),
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")

    with PersistentGGUFScorer(runner=runner, model=model) as scorer:
        first_pid = scorer.process_id
        first = scorer.score("first", "question one")
        second = scorer.score("second", "question two")

        assert scorer.process_id == first_pid
        assert first.raw_scores == (-1.0, -2.0, -3.0)
        assert first.native_elapsed_ms == pytest.approx(1.234)
        assert first.maximum_sequence_tokens == 42
        assert first.prefix_reused is False
        assert first.prefix_tokens == 0
        assert first.round_trip_ms > 0.0
        assert second.raw_scores == first.raw_scores
        with pytest.raises(ValueError, match="unsupported characters"):
            scorer.score("bad id", "question")

    with PersistentGGUFScorer(runner=runner, model=model, prefix="fixed prefix") as scorer:
        cached = scorer.score("cached", "fixed prefix and request")

        assert cached.prefix_reused is True
        assert cached.prefix_tokens == 7


def test_quantized_evidence_must_bind_model_ledger_calibration_and_runtime(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    predictions = tmp_path / "predictions.jsonl"
    calibration = tmp_path / "calibration.json"
    model.write_bytes(b"model")
    predictions.write_text("{}\n", encoding="utf-8")
    calibration.write_text(
        '{"temperature":2.0,"scam_threshold":0.4,"safe_threshold":0.6}\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    report = {
        "model": str(model),
        "model_sha256": digest,
        "temperature": 2.0,
        "scam_threshold": 0.4,
        "safe_threshold": 0.6,
        "safe_threshold_semantics": "minimum_safe_probability",
        "prediction_ledger": {
            "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()
        },
        "calibration": {
            "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest()
        },
        "quantization_parity": {
            "exact_calibrated_verdict_parity": True,
            "release_gate_passed": True,
        },
        "runtime_config": {
            "ctx_size": 640,
            "batch_size": 640,
            "ubatch_size": 128,
            "n_gpu_layers": 99,
            "parallel": 1,
        },
    }

    validate_quantized_evidence(
        report,
        model=model,
        model_sha256=digest,
        predictions=predictions,
        calibration=calibration,
        ctx_size=640,
        batch_size=640,
        ubatch_size=128,
        n_gpu_layers=99,
    )
    report["quantization_parity"]["release_gate_passed"] = False
    with pytest.raises(ValueError, match="exact frozen verdict parity"):
        validate_quantized_evidence(
            report,
            model=model,
            model_sha256=digest,
            predictions=predictions,
            calibration=calibration,
            ctx_size=640,
            batch_size=640,
            ubatch_size=128,
            n_gpu_layers=99,
        )
