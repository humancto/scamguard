from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.verify_android_physical_smoke import validate_physical_result, verify_apk


def valid_result() -> dict[str, object]:
    return {
        "abi": "arm64-v8a",
        "android_api": 35,
        "artifact_schema_version": 1,
        "backend": "llama.cpp CPU",
        "complete_elapsed_ms": 90.0,
        "diagnostic_only": True,
        "manufacturer": "Google",
        "model": "Pixel",
        "model_tensor_bytes": 552_074_496,
        "native_elapsed_ms": 89.0,
        "passed": True,
        "physical_device": True,
        "platform": "Android",
        "prefix_reused": True,
        "prefix_tokens": 141,
        "protocol_version": 3,
        "raw_safe_score": -3.0,
        "raw_scam_score": -3.0,
        "raw_uncertain_score": -2.0,
        "safe_probability": 0.3,
        "scam_probability": 0.3,
        "simulator": False,
        "startup_ms": 700.0,
        "uncertain_probability": 0.4,
        "verdict": "SAFE",
    }


def validate(result: dict[str, object]) -> list[str]:
    return validate_physical_result(
        result,
        raw_scores=(-3.0, -2.0, -3.0),
        probabilities=(0.3, 0.4, 0.3),
        verdict="SAFE",
        protocol_version=3,
        model_tensor_bytes=552_074_496,
        prefix_tokens=141,
    )


def test_matching_physical_arm64_result_passes() -> None:
    assert validate(valid_result()) == []


def test_emulator_or_score_drift_is_rejected() -> None:
    result = valid_result()
    result["physical_device"] = False
    result["simulator"] = True
    result["raw_safe_score"] = -2.9

    errors = validate(result)

    assert "physical_device must equal True" in errors
    assert "simulator must equal False" in errors
    assert "raw_safe_score differs from the host CPU reference" in errors


def test_message_text_and_wrong_abi_are_rejected() -> None:
    result = valid_result()
    result["message"] = "sensitive input"
    result["abi"] = "x86_64"

    errors = validate(result)

    assert "Android result has unexpected fields: ['message']" in errors
    assert "abi must equal 'arm64-v8a'" in errors


def test_apk_must_embed_the_selected_arm64_jni_only(tmp_path: Path) -> None:
    jni = tmp_path / "libscamguard-jni.so"
    jni.write_bytes(b"jni-runtime")
    apk = tmp_path / "smoke.apk"
    with zipfile.ZipFile(apk, "w") as archive:
        archive.writestr("AndroidManifest.xml", b"manifest")
        archive.writestr("classes.dex", b"dex")
        archive.writestr("lib/arm64-v8a/libscamguard-jni.so", jni.read_bytes())

    binding = verify_apk(apk, jni)

    assert binding["jni_bytes"] == len(b"jni-runtime")


def test_apk_rejects_extra_abi(tmp_path: Path) -> None:
    jni = tmp_path / "libscamguard-jni.so"
    jni.write_bytes(b"jni-runtime")
    apk = tmp_path / "smoke.apk"
    with zipfile.ZipFile(apk, "w") as archive:
        archive.writestr("AndroidManifest.xml", b"manifest")
        archive.writestr("classes.dex", b"dex")
        archive.writestr("lib/arm64-v8a/libscamguard-jni.so", jni.read_bytes())
        archive.writestr("lib/x86_64/libscamguard-jni.so", b"wrong")

    with pytest.raises(ValueError, match="only the arm64-v8a"):
        verify_apk(apk, jni)
