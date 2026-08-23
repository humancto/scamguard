#!/usr/bin/env python3
"""Verify iOS Simulator scoring parity against the same host CPU runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Final

try:
    from scamguard.gguf_runtime import QwenGGUFVerdictBackend, calibrated_probabilities
    from scamguard.metrics import file_sha256
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from scamguard.gguf_runtime import QwenGGUFVerdictBackend, calibrated_probabilities
    from scamguard.metrics import file_sha256

SCHEMA_VERSION: Final[int] = 1
LABELS: Final[tuple[str, ...]] = ("SAFE", "UNCERTAIN", "SCAM")
ABSOLUTE_TOLERANCE: Final[float] = 1e-9
RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_schema_version",
        "complete_elapsed_ms",
        "diagnostic_only",
        "model_tensor_bytes",
        "native_elapsed_ms",
        "physical_device",
        "prefix_reused",
        "prefix_tokens",
        "protocol_version",
        "raw_safe_score",
        "raw_scam_score",
        "raw_uncertain_score",
        "safe_probability",
        "scam_probability",
        "simulator",
        "startup_ms",
        "uncertain_probability",
        "verdict",
    }
)


def _load_object(path: Path, role: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _member(root: Path, record: dict[str, Any], role: str) -> Path:
    relative = Path(str(record.get("path", "")))
    candidate = (root / relative).resolve()
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or not candidate.is_relative_to(root)
        or not candidate.is_file()
    ):
        raise ValueError(f"runtime-pack {role} is missing or escapes the pack")
    if file_sha256(candidate) != record.get("sha256"):
        raise ValueError(f"runtime-pack {role} SHA-256 differs from its manifest")
    return candidate


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def expected_verdict(
    probabilities: tuple[float, float, float],
    *,
    scam_threshold: float,
    safe_threshold: float,
) -> str:
    if probabilities[2] >= scam_threshold:
        return "SCAM"
    if probabilities[0] >= safe_threshold:
        return "SAFE"
    return "UNCERTAIN"


def validate_simulator_result(
    result: dict[str, Any],
    *,
    raw_scores: tuple[float, float, float],
    probabilities: tuple[float, float, float],
    verdict: str,
    protocol_version: int,
    model_tensor_bytes: int,
    prefix_tokens: int,
) -> list[str]:
    errors: list[str] = []
    unexpected_fields = sorted(set(result) - RESULT_FIELDS)
    if unexpected_fields:
        errors.append(f"simulator result has unexpected fields: {unexpected_fields}")
    required_flags = {
        "artifact_schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "physical_device": False,
        "simulator": True,
        "prefix_reused": True,
        "protocol_version": protocol_version,
        "model_tensor_bytes": model_tensor_bytes,
        "prefix_tokens": prefix_tokens,
        "verdict": verdict,
    }
    for field, expected in required_flags.items():
        if result.get(field) != expected:
            errors.append(f"{field} must equal {expected!r}")

    comparisons = (
        ("raw_safe_score", raw_scores[0]),
        ("raw_uncertain_score", raw_scores[1]),
        ("raw_scam_score", raw_scores[2]),
        ("safe_probability", probabilities[0]),
        ("uncertain_probability", probabilities[1]),
        ("scam_probability", probabilities[2]),
    )
    for field, expected in comparisons:
        actual = result.get(field)
        if not _finite_number(actual):
            errors.append(f"{field} must be finite")
        elif not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
            errors.append(f"{field} differs from the host CPU reference")

    for field in ("startup_ms", "complete_elapsed_ms", "native_elapsed_ms"):
        value = result.get(field)
        if not _finite_number(value) or float(value) <= 0:
            errors.append(f"{field} must be finite and positive")
    return errors


def verify(
    *, result_path: Path, request_path: Path, runtime_pack: Path
) -> tuple[list[str], dict[str, Any]]:
    result = _load_object(result_path, "simulator result")
    request = _load_object(request_path, "smoke request")
    root = runtime_pack.expanduser().resolve()
    manifest_path = root / "scamguard_gguf_pack.json"
    manifest = _load_object(manifest_path, "runtime-pack manifest")
    if manifest.get("publication_authorized") is not False:
        raise ValueError("simulator smoke cannot use a publication-authorizing pack")
    if manifest.get("purpose") != "upstream_base_control":
        raise ValueError("simulator smoke requires the explicit upstream base control")
    model_record = manifest.get("model")
    runner_record = manifest.get("runner")
    calibration_record = manifest.get("calibration")
    prompt = manifest.get("prompt")
    runtime = manifest.get("runtime")
    sections = (model_record, runner_record, calibration_record, prompt, runtime)
    if not all(isinstance(item, dict) for item in sections):
        raise ValueError("runtime-pack manifest is incomplete")
    assert isinstance(model_record, dict)
    assert isinstance(runner_record, dict)
    assert isinstance(calibration_record, dict)
    assert isinstance(prompt, dict)
    assert isinstance(runtime, dict)
    model = _member(root, model_record, "model")
    runner = _member(root, runner_record, "runner")
    calibration = _member(root, calibration_record, "calibration")
    if request.get("model") != model.name:
        raise ValueError("smoke request model differs from the runtime pack")
    if request.get("packManifest") != manifest_path.name:
        raise ValueError("smoke request manifest differs from the runtime pack")
    if request.get("calibration") != calibration.name:
        raise ValueError("smoke request calibration differs from the runtime pack")
    message = request.get("message")
    if not isinstance(message, str) or not message:
        raise ValueError("smoke request message must be non-empty")

    backend = QwenGGUFVerdictBackend(
        runner=runner,
        model=model,
        prompt_prefix=str(prompt["prefix"]),
        prompt_suffix=str(prompt["suffix"]),
        calibration=calibration,
        expected_model_sha256=str(model_record["sha256"]),
        expected_runner_sha256=str(runner_record["sha256"]),
        ctx_size=int(runtime["ctx_size"]),
        batch_size=int(runtime["batch_size"]),
        ubatch_size=int(runtime["ubatch_size"]),
        threads=int(runtime["threads"]),
        n_gpu_layers=0,
    )
    try:
        question = backend.cached_prefix + "<message>" + message + backend.prompt_suffix
        reference = backend.scorer.score("ios-simulator-parity", question)
        probabilities = calibrated_probabilities(reference.raw_scores, backend.temperature)
        verdict = expected_verdict(
            probabilities,
            scam_threshold=backend.scam_threshold,
            safe_threshold=backend.safe_threshold,
        )
        errors = validate_simulator_result(
            result,
            raw_scores=reference.raw_scores,
            probabilities=probabilities,
            verdict=verdict,
            protocol_version=backend.scorer.protocol_version,
            model_tensor_bytes=backend.scorer.loaded_model_bytes,
            prefix_tokens=backend.scorer.loaded_prefix_tokens,
        )
    finally:
        backend.close()

    evidence = {
        "artifact_schema_version": SCHEMA_VERSION,
        "measurement_kind": "ios_simulator_cpu_parity_diagnostic",
        "diagnostic_only": True,
        "physical_device": False,
        "simulator": True,
        "contains_message_text": False,
        "passed": not errors,
        "errors": errors,
        "bindings": {
            "request_sha256": file_sha256(request_path),
            "runtime_pack_manifest_sha256": file_sha256(manifest_path),
            "model_sha256": file_sha256(model),
            "runner_sha256": file_sha256(runner),
            "calibration_sha256": file_sha256(calibration),
        },
        "simulator_result": {field: result.get(field) for field in sorted(RESULT_FIELDS)},
        "host_cpu_reference": {
            "raw_scores": dict(zip(LABELS, reference.raw_scores, strict=True)),
            "probabilities": dict(zip(LABELS, probabilities, strict=True)),
            "verdict": verdict,
            "native_elapsed_ms": reference.native_elapsed_ms,
            "prefix_reused": reference.prefix_reused,
            "prefix_tokens": reference.prefix_tokens,
        },
    }
    return errors, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime-pack", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path)
    args = parser.parse_args()
    try:
        errors, evidence = verify(
            result_path=args.result,
            request_path=args.request,
            runtime_pack=args.runtime_pack,
        )
        rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if args.evidence_output is not None:
            if args.evidence_output.exists():
                raise FileExistsError(f"refusing to overwrite {args.evidence_output}")
            args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 1 if errors else 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"iOS Simulator smoke verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
