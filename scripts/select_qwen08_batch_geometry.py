#!/usr/bin/env python3
"""Select the fastest memory-bounded Qwen 0.8B MPS geometry at effective batch 16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

from scamguard.metrics import file_sha256

EXPECTED_GEOMETRIES: Final[set[tuple[int, int]]] = {
    (1, 16),
    (2, 8),
    (4, 4),
    (8, 2),
    (16, 1),
}
MAX_RECOMMENDED_MEMORY_FRACTION: Final[float] = 0.5
EFFECTIVE_BATCH_SIZE: Final[int] = 16
SEQUENCE_LENGTH: Final[int] = 640
TOKENS_PER_EFFECTIVE_BATCH: Final[int] = EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH


def _candidate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"batch geometry report must be an object: {path}")
    geometry = report.get("geometry")
    environment = report.get("environment")
    probe = report.get("probe")
    lora = report.get("lora")
    if not all(isinstance(item, dict) for item in (geometry, environment, probe, lora)):
        raise ValueError(f"batch geometry report is incomplete: {path}")
    assert isinstance(geometry, dict)
    assert isinstance(environment, dict)
    assert isinstance(probe, dict)
    assert isinstance(lora, dict)
    microbatch = geometry.get("microbatch_size")
    accumulation = geometry.get("gradient_accumulation")
    if (microbatch, accumulation) not in EXPECTED_GEOMETRIES:
        raise ValueError(f"batch geometry is outside the frozen matrix: {path}")
    effective_batch = geometry.get("effective_batch_size")
    sequence_length = geometry.get("sequence_length")
    tokens = geometry.get(
        "tokens_per_effective_batch",
        int(geometry.get("tokens_per_microbatch", 0)) * int(accumulation),
    )
    if (
        effective_batch != EFFECTIVE_BATCH_SIZE
        or sequence_length != SEQUENCE_LENGTH
        or tokens != TOKENS_PER_EFFECTIVE_BATCH
    ):
        raise ValueError(f"batch geometry does not preserve the frozen workload: {path}")
    if (
        report.get("passed") is not True
        or report.get("parameter_update_performed") is not False
        or report.get("contains_training_or_audit_rows") is not False
        or report.get("base_model") != "Qwen/Qwen3.5-0.8B"
        or report.get("base_model_revision")
        != "2fc06364715b967f1860aea9cf38778875588b17"
        or report.get("transformers_revision")
        != "0c92811846095910816a87aca50050d10c545270"
        or report.get("snapshot_manifest_sha256")
        != "c4a2b9f20a7aa8cd3137ccf2726ba01840b310dbcf6da865506b9c83406f8b08"
        or lora.get("trainable_parameters") != 10_822_656
        or lora.get("trainable_tensors") != 372
        or lora.get("gradient_tensors") != 372
        or lora.get("visual_trainable_tensors") != 0
    ):
        raise ValueError(f"batch geometry report differs from the frozen model contract: {path}")
    driver_memory = environment.get("mps_driver_allocated_bytes")
    recommended_memory = environment.get("mps_recommended_max_bytes")
    elapsed = probe.get("forward_backward_seconds")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in (driver_memory, recommended_memory, elapsed)
    ):
        raise ValueError(f"batch geometry report has invalid memory or timing: {path}")
    memory_fraction = float(driver_memory) / float(recommended_memory)
    return {
        "report": path.name,
        "report_sha256": file_sha256(path),
        "microbatch_size": int(microbatch),
        "gradient_accumulation": int(accumulation),
        "effective_batch_size": int(effective_batch),
        "sequence_length": int(sequence_length),
        "tokens_per_effective_batch": int(tokens),
        "forward_backward_seconds": float(elapsed),
        "mps_driver_allocated_bytes": int(driver_memory),
        "mps_recommended_max_bytes": int(recommended_memory),
        "recommended_memory_fraction": memory_fraction,
        "memory_gate_passed": memory_fraction <= MAX_RECOMMENDED_MEMORY_FRACTION,
    }


def select_geometry(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    observed = {
        (candidate["microbatch_size"], candidate["gradient_accumulation"])
        for candidate in candidates
    }
    if observed != EXPECTED_GEOMETRIES:
        raise ValueError("batch geometry matrix is incomplete or duplicated")
    eligible = [candidate for candidate in candidates if candidate["memory_gate_passed"]]
    if not eligible:
        raise ValueError("no batch geometry stays within the frozen memory ceiling")
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["forward_backward_seconds"],
            candidate["mps_driver_allocated_bytes"],
        ),
    )
    return selected


def build_selection(paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite batch selection: {output}")
    candidates = [_candidate(path) for path in paths]
    selected = select_geometry(candidates)
    ordered = sorted(candidates, key=lambda candidate: candidate["microbatch_size"])
    for candidate in ordered:
        if candidate is selected:
            candidate["decision"] = "selected_fastest_within_memory_gate"
        elif candidate["memory_gate_passed"]:
            candidate["decision"] = "eligible_but_slower"
        else:
            candidate["decision"] = "rejected_memory_ceiling"
    repository = Path(__file__).resolve().parents[1]
    record = {
        "artifact_schema_version": 1,
        "decision_kind": "qwen08_mps_batch_geometry_selection",
        "selection_data_only": "synthetic fixed-token stress; no fitting or audit rows",
        "quality_contract": {
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "tokens_per_effective_batch": TOKENS_PER_EFFECTIVE_BATCH,
            "optimizer_semantics_changed": False,
        },
        "policy": {
            "maximum_recommended_memory_fraction": MAX_RECOMMENDED_MEMORY_FRACTION,
            "selection": "minimum measured forward_backward_seconds among memory-gate passes",
        },
        "candidates": ordered,
        "selected": {
            "microbatch_size": selected["microbatch_size"],
            "gradient_accumulation": selected["gradient_accumulation"],
            "effective_batch_size": selected["effective_batch_size"],
            "forward_backward_seconds": selected["forward_backward_seconds"],
            "mps_driver_allocated_bytes": selected["mps_driver_allocated_bytes"],
            "recommended_memory_fraction": selected["recommended_memory_fraction"],
        },
        "source_bindings": {
            "selector_sha256": file_sha256(Path(__file__).resolve()),
            "batch_preflight_sha256": file_sha256(
                repository / "scripts" / "preflight_qwen08_batch.py"
            ),
            "experiment_freezer_before_selection_sha256": file_sha256(
                repository / "scripts" / "freeze_qwen08_full_experiment.py"
            ),
            "training_launcher_before_selection_sha256": file_sha256(
                repository / "training" / "train_qwen_lora.py"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = build_selection(args.report, args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
