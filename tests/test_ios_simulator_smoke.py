from __future__ import annotations

from scripts.verify_ios_simulator_smoke import validate_simulator_result


def valid_result() -> dict[str, object]:
    return {
        "artifact_schema_version": 1,
        "diagnostic_only": True,
        "physical_device": False,
        "simulator": True,
        "verdict": "SAFE",
        "safe_probability": 0.3,
        "uncertain_probability": 0.4,
        "scam_probability": 0.3,
        "raw_safe_score": -3.0,
        "raw_uncertain_score": -2.0,
        "raw_scam_score": -3.0,
        "startup_ms": 7000.0,
        "complete_elapsed_ms": 90.0,
        "native_elapsed_ms": 89.0,
        "prefix_reused": True,
        "prefix_tokens": 141,
        "model_tensor_bytes": 552_074_496,
        "protocol_version": 2,
    }


def validate(result: dict[str, object]) -> list[str]:
    return validate_simulator_result(
        result,
        raw_scores=(-3.0, -2.0, -3.0),
        probabilities=(0.3, 0.4, 0.3),
        verdict="SAFE",
        protocol_version=2,
        model_tensor_bytes=552_074_496,
        prefix_tokens=141,
    )


def test_matching_cpu_parity_diagnostic_passes() -> None:
    assert validate(valid_result()) == []


def test_physical_claim_or_probability_drift_is_rejected() -> None:
    result = valid_result()
    result["physical_device"] = True
    result["diagnostic_only"] = False
    result["safe_probability"] = 0.31

    errors = validate(result)

    assert "physical_device must equal False" in errors
    assert "diagnostic_only must equal True" in errors
    assert "safe_probability differs from the host CPU reference" in errors


def test_timing_prefix_and_protocol_are_required() -> None:
    result = valid_result()
    result["native_elapsed_ms"] = 0.0
    result["prefix_reused"] = False
    result["protocol_version"] = 1

    errors = validate(result)

    assert "native_elapsed_ms must be finite and positive" in errors
    assert "prefix_reused must equal True" in errors
    assert "protocol_version must equal 2" in errors


def test_unexpected_message_text_field_is_rejected() -> None:
    result = valid_result()
    result["message"] = "sensitive input"

    errors = validate(result)

    assert "simulator result has unexpected fields: ['message']" in errors
