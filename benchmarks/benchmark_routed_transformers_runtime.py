#!/usr/bin/env python3
"""Measure the frozen encoder-to-Qwen route with both models persistently loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.decision import calibrated_verdict
from scamguard.metrics import file_sha256
from scamguard.model import (
    ModelScores,
    QwenBaseVerdictBackend,
    QwenVerdictBackend,
    TransformersBackend,
)
from scamguard.prompts import SYSTEM_PROMPT
from training.eval_routed import (
    confidence_margin,
    evaluate_records,
    join_split,
    read_prediction_ledger,
    route_records,
)


class RuntimeBackend(Protocol):
    model_id: str
    scam_threshold: float
    safe_probability_threshold: float
    safe_max_scam_probability: float | None

    def predict(self, text: str) -> ModelScores: ...


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "maximum_ms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "samples": len(values),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "maximum_ms": float(np.max(array)),
    }


def backend_verdict(backend: RuntimeBackend, scores: ModelScores) -> str:
    return calibrated_verdict(
        safe_probability=scores.safe,
        scam_probability=scores.scam,
        scam_probability_threshold=backend.scam_threshold,
        safe_probability_threshold=backend.safe_probability_threshold,
        safe_max_scam_probability=backend.safe_max_scam_probability,
    )


def scores_dict(scores: ModelScores) -> dict[str, float]:
    return {"SAFE": scores.safe, "UNCERTAIN": scores.uncertain, "SCAM": scores.scam}


def maximum_probability_error(scores: ModelScores, expected: dict[str, Any]) -> float:
    return max(
        abs(actual - float(expected["probabilities"][label]))
        for label, actual in scores_dict(scores).items()
    )


def trace_requests(
    rows: list[dict[str, Any]],
    expected_router: dict[str, dict[str, Any]],
    expected_specialist: dict[str, dict[str, Any]],
    expected_final: dict[str, dict[str, Any]],
    router: RuntimeBackend,
    specialist: RuntimeBackend,
    margin_max: float,
    repetitions: int,
    probability_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    router_max_error = 0.0
    specialist_max_error = 0.0
    router_probability_drift: set[str] = set()
    specialist_probability_drift: set[str] = set()
    router_decision_mismatch: set[str] = set()
    specialist_decision_mismatch: set[str] = set()
    route_mismatch: set[str] = set()
    final_decision_mismatch: set[str] = set()
    for repetition in range(repetitions):
        for row in rows:
            identifier = str(row["id"])
            expected_router_record = expected_router[identifier]
            expected_specialist_record = expected_specialist[identifier]
            expected_final_record = expected_final[identifier]

            request_started = time.perf_counter_ns()
            router_started = request_started
            router_scores = router.predict(str(row["text"]))
            router_finished = time.perf_counter_ns()
            router_verdict = backend_verdict(router, router_scores)
            router_error = maximum_probability_error(router_scores, expected_router_record)
            router_max_error = max(router_max_error, router_error)
            if router_error > probability_tolerance:
                router_probability_drift.add(identifier)
            if router_verdict != expected_router_record["calibrated_verdict"]:
                router_decision_mismatch.add(identifier)

            router_runtime_record = {"probabilities": scores_dict(router_scores)}
            escalated = (
                router_verdict == "UNCERTAIN"
                or confidence_margin(router_runtime_record) <= margin_max
            )
            specialist_ms = 0.0
            specialist_verdict: str | None = None
            if escalated:
                specialist_started = time.perf_counter_ns()
                specialist_scores = specialist.predict(str(row["text"]))
                specialist_finished = time.perf_counter_ns()
                specialist_ms = (specialist_finished - specialist_started) / 1_000_000
                specialist_verdict = backend_verdict(specialist, specialist_scores)
                specialist_error = maximum_probability_error(
                    specialist_scores, expected_specialist_record
                )
                specialist_max_error = max(specialist_max_error, specialist_error)
                if specialist_error > probability_tolerance:
                    specialist_probability_drift.add(identifier)
                if specialist_verdict != expected_specialist_record["calibrated_verdict"]:
                    specialist_decision_mismatch.add(identifier)
            request_finished = time.perf_counter_ns()
            final_verdict = specialist_verdict if escalated else router_verdict
            if escalated != expected_final_record["escalated"]:
                route_mismatch.add(identifier)
            if final_verdict != expected_final_record["final_verdict"]:
                final_decision_mismatch.add(identifier)
            router_ms = (router_finished - router_started) / 1_000_000
            total_ms = (request_finished - request_started) / 1_000_000
            traces.append(
                {
                    "id": identifier,
                    "split": row["split"],
                    "source": row.get("source"),
                    "source_language": row.get("source_language"),
                    "category": row.get("category"),
                    "truth": row.get("label"),
                    "repetition": repetition,
                    "router_verdict": router_verdict,
                    "expected_router_verdict": expected_router_record[
                        "calibrated_verdict"
                    ],
                    "specialist_verdict": specialist_verdict,
                    "expected_specialist_verdict": (
                        expected_specialist_record["calibrated_verdict"]
                        if escalated
                        else None
                    ),
                    "final_verdict": final_verdict,
                    "expected_final_verdict": expected_final_record["final_verdict"],
                    "escalated": escalated,
                    "expected_escalated": expected_final_record["escalated"],
                    "router_ms": router_ms,
                    "specialist_ms": specialist_ms,
                    "routing_overhead_ms": max(0.0, total_ms - router_ms - specialist_ms),
                    "total_ms": total_ms,
                }
            )
    return traces, {
        "router_max_absolute_probability_error": router_max_error,
        "specialist_max_absolute_probability_error": specialist_max_error,
        "probability_tolerance": probability_tolerance,
        "unique_router_probability_drift_ids": len(router_probability_drift),
        "unique_specialist_probability_drift_ids": len(specialist_probability_drift),
        "unique_router_decision_mismatch_ids": len(router_decision_mismatch),
        "unique_specialist_decision_mismatch_ids": len(specialist_decision_mismatch),
        "unique_route_mismatch_ids": len(route_mismatch),
        "unique_final_decision_mismatch_ids": len(final_decision_mismatch),
        "router_decision_mismatch_id_sample": sorted(router_decision_mismatch)[:10],
        "specialist_decision_mismatch_id_sample": sorted(specialist_decision_mismatch)[:10],
        "route_mismatch_id_sample": sorted(route_mismatch)[:10],
        "final_decision_mismatch_id_sample": sorted(final_decision_mismatch)[:10],
        "all_router_decisions_match": not router_decision_mismatch,
        "all_specialist_decisions_match_when_escalated": not specialist_decision_mismatch,
        "all_routes_match": not route_mismatch,
        "all_final_decisions_match": not final_decision_mismatch,
        "release_gate_passed": not (
            router_decision_mismatch
            or specialist_decision_mismatch
            or route_mismatch
            or final_decision_mismatch
        ),
    }


def latency_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    escalated = [record for record in traces if record["escalated"]]
    fast = [record for record in traces if not record["escalated"]]
    overall = summarize([float(record["total_ms"]) for record in traces])
    escalated_total = summarize([float(record["total_ms"]) for record in escalated])
    return {
        "scope": (
            "batch-one tokenizer plus router inference plus frozen route decision plus selective "
            "specialist inference; models are persistently loaded; excludes evidence extraction, "
            "SDK result construction, I/O, and cold load"
        ),
        "requests": len(traces),
        "escalated_requests": len(escalated),
        "escalation_rate": len(escalated) / len(traces),
        "overall": overall,
        "fast_path_total": summarize([float(record["total_ms"]) for record in fast]),
        "escalated_path_total": escalated_total,
        "router_component": summarize([float(record["router_ms"]) for record in traces]),
        "specialist_component": summarize(
            [float(record["specialist_ms"]) for record in escalated]
        ),
        "routing_overhead": summarize(
            [float(record["routing_overhead_ms"]) for record in traces]
        ),
        "gates": {
            "overall_p95_under_20_ms": bool(
                overall["p95_ms"] is not None and float(overall["p95_ms"]) <= 20.0
            ),
            "escalated_path_p95_under_50_ms": bool(
                escalated_total["p95_ms"] is not None
                and float(escalated_total["p95_ms"]) < 50.0
            ),
            "tail_disclosure_complete": True,
        },
    }


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def verify_score_cache_identity(
    metadata: dict[str, Any], expected: dict[str, Any]
) -> None:
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise ValueError(f"specialist score-cache metadata mismatch: {field}")
    if not isinstance(metadata.get("batch_size"), int) or metadata["batch_size"] < 1:
        raise ValueError("specialist score-cache metadata has invalid batch_size")


def adapter_identity(
    specialist_report: dict[str, Any], adapter: Path | None
) -> tuple[str | None, dict[str, Any]]:
    """Bind the runtime adapter path and immutable PEFT weights to the evaluation report."""

    reported_adapter = specialist_report.get("adapter")
    if adapter is None:
        if reported_adapter is not None:
            raise ValueError("specialist report requires a LoRA adapter")
        return None, {"kind": "base_control", "adapter": None, "adapter_sha256": None}
    resolved = adapter.expanduser().resolve()
    weights = resolved / "adapter_model.safetensors"
    if not resolved.is_dir() or not weights.is_file():
        raise FileNotFoundError(f"missing LoRA adapter weights: {weights}")
    if reported_adapter is None or Path(str(reported_adapter)).expanduser().resolve() != resolved:
        raise ValueError("specialist report identifies a different LoRA adapter")
    digest = file_sha256(weights)
    return digest, {
        "kind": "lora_adapter",
        "adapter": str(adapter),
        "adapter_weights": str(weights),
        "adapter_sha256": digest,
        "adapter_artifact_bytes": artifact_size(resolved),
    }


def verify_backend_calibration(
    backend: RuntimeBackend,
    report: dict[str, Any],
    *,
    model: str,
    revision: str,
) -> None:
    if report.get("model") != model or report.get("base_model_revision") != revision:
        raise ValueError("specialist report identifies a different pinned base model")
    expected = {
        "temperature": float(report["temperature"]),
        "scam_threshold": float(report["scam_threshold"]),
        "safe_probability_threshold": float(report["safe_threshold"]),
        "sequence_bucket_size": int(report["score_cache"]["sequence_bucket_size"]),
    }
    for field, value in expected.items():
        if getattr(backend, field) != value:
            raise ValueError(f"runtime calibration differs from specialist report: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--specialist-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--specialist-revision", required=True)
    parser.add_argument("--specialist-adapter", type=Path)
    parser.add_argument("--specialist-report", type=Path, required=True)
    parser.add_argument("--specialist-score-cache-metadata", type=Path, required=True)
    parser.add_argument("--specialist-predictions", type=Path, required=True)
    parser.add_argument("--routed-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=5e-3,
        help="Maximum BF16 repeat-run probability drift; decisions must still match exactly.",
    )
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.probability_tolerance <= 0.0:
        parser.error("--probability-tolerance must be positive")

    import torch

    if args.require_mps and not torch.backends.mps.is_available():
        raise RuntimeError("--require-mps was set but Metal is unavailable")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    routed_report = json.loads(args.routed_report.read_text(encoding="utf-8"))
    specialist_report = json.loads(args.specialist_report.read_text(encoding="utf-8"))
    specialist_score_metadata = json.loads(
        args.specialist_score_cache_metadata.read_text(encoding="utf-8")
    )
    adapter_sha256, adapter_record = adapter_identity(
        specialist_report, args.specialist_adapter
    )
    if routed_report["inputs"]["router_sha256"] != file_sha256(args.router_predictions):
        raise ValueError("router ledger differs from routed policy report")
    if routed_report["inputs"]["specialist_sha256"] != file_sha256(
        args.specialist_predictions
    ):
        raise ValueError("specialist ledger differs from routed policy report")
    margin_max = float(routed_report["policy"]["margin_max"])
    expected_score_identity = {
        "scoring_version": specialist_report["score_cache"]["scoring_version"],
        "model": args.specialist_model,
        "revision": args.specialist_revision,
        "adapter_sha256": adapter_sha256,
        "data_sha256": specialist_report["data_sha256"][args.split],
        "examples": specialist_report[args.split]["examples"],
        "labels": ["SAFE", "UNCERTAIN", "SCAM"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "sequence_bucket_size": specialist_report["score_cache"][
            "sequence_bucket_size"
        ],
    }
    verify_score_cache_identity(specialist_score_metadata, expected_score_identity)

    router_ledger = read_prediction_ledger(args.router_predictions)
    specialist_ledger = read_prediction_ledger(args.specialist_predictions)
    joined = join_split(router_ledger, specialist_ledger, args.split)
    expected_routed = route_records(joined, margin_max)
    expected_router = {str(router["id"]): router for router, _specialist in joined}
    expected_specialist = {
        str(specialist["id"]): specialist for _router, specialist in joined
    }
    expected_final = {str(record["id"]): record for record in expected_routed}

    data_path = args.data / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError("runtime split is empty")
    row_ids = [str(row["id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(expected_router):
        raise ValueError("runtime data IDs differ from the frozen joined ledger")
    for row in rows:
        row["split"] = args.split
        expected = expected_router[str(row["id"])]
        for field in ("label", "source", "source_language", "category"):
            ledger_field = "truth" if field == "label" else field
            if row.get(field) != expected.get(ledger_field):
                raise ValueError(f"runtime data metadata mismatch for {row['id']}: {field}")

    router = TransformersBackend(args.router_checkpoint, device=device)
    if args.specialist_adapter is None:
        specialist = QwenBaseVerdictBackend(
            args.specialist_model,
            args.specialist_revision,
            args.specialist_report,
            device=device,
        )
    else:
        specialist = QwenVerdictBackend(args.specialist_adapter, device=device)
    verify_backend_calibration(
        specialist,
        specialist_report,
        model=args.specialist_model,
        revision=args.specialist_revision,
    )
    first_escalated = next(
        (record for record in expected_routed if record["escalated"]), None
    )
    row_by_id = {str(row["id"]): row for row in rows}
    router.predict(str(rows[0]["text"]))
    if first_escalated is not None:
        specialist.predict(str(row_by_id[str(first_escalated["id"])]["text"]))

    traces, parity = trace_requests(
        rows,
        expected_router,
        expected_specialist,
        expected_final,
        router,
        specialist,
        margin_max,
        args.repetitions,
        args.probability_tolerance,
    )
    trace_path = args.traces or args.report.with_suffix(".traces.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in traces),
        encoding="utf-8",
    )
    selected_ids_sha256 = hashlib.sha256("\n".join(row_ids).encode()).hexdigest()
    first_repetition = [record for record in traces if record["repetition"] == 0]
    runtime_quality = evaluate_records(first_repetition)
    frozen_quality = routed_report["test"]
    result = {
        "artifact_schema_version": 1,
        "purpose": "persistent BF16 routed-runtime reference control",
        "policy": {
            "routed_report": str(args.routed_report),
            "routed_report_sha256": file_sha256(args.routed_report),
            "margin_max": margin_max,
            "selection_split": routed_report["policy"]["selection_split"],
            "runtime_split": args.split,
        },
        "router": {
            "model_id": router.model_id,
            "checkpoint": str(args.router_checkpoint),
            "artifact_bytes": artifact_size(args.router_checkpoint),
        },
        "specialist": {
            "model_id": specialist.model_id,
            "calibration_report": str(args.specialist_report),
            "calibration_report_sha256": file_sha256(args.specialist_report),
            "memory_footprint_bytes": specialist.model.get_memory_footprint(),
            "quantization": "BF16 reference checkpoint",
            **adapter_record,
            "frozen_quality_scoring": {
                "message_batch_size": specialist_score_metadata["batch_size"],
                "candidate_batch_size": (
                    int(specialist_score_metadata["batch_size"])
                    * len(specialist_score_metadata["labels"])
                ),
                "sequence_bucket_size": specialist_score_metadata[
                    "sequence_bucket_size"
                ],
                "metadata_path": str(args.specialist_score_cache_metadata),
                "metadata_sha256": file_sha256(args.specialist_score_cache_metadata),
            },
            "runtime_scoring": {
                "message_batch_size": 1,
                "candidate_batch_size": len(specialist.labels),
                "sequence_bucket_size": specialist.sequence_bucket_size,
            },
        },
        "data": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
            "rows": len(rows),
            "repetitions": args.repetitions,
            "selected_ids_sha256": selected_ids_sha256,
            "contains_message_text": False,
        },
        "parity": parity,
        "runtime_quality": runtime_quality,
        "quality_delta_vs_frozen_ledger": {
            "scam_recall": runtime_quality["binary_safety"]["scam_recall"]
            - frozen_quality["binary_safety"]["scam_recall"],
            "false_positive_rate": runtime_quality["binary_safety"]["false_positive_rate"]
            - frozen_quality["binary_safety"]["false_positive_rate"],
            "macro_f1": runtime_quality["macro_f1"] - frozen_quality["macro_f1"],
        },
        "latency": latency_report(traces),
        "measurement_mode": "interleaved_persistent_per_request",
        "process_peak_rss_bytes": peak_rss_bytes(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
        },
        "trace_ledger": {
            "path": str(trace_path),
            "sha256": file_sha256(trace_path),
            "rows": len(traces),
            "contains_message_text": False,
        },
        "limitations": [
            "BF16 reference timing does not establish final GGUF or phone latency.",
            "The frozen specialist quality ledger and product runtime share an explicit "
            "message, candidate, and sequence-bucket shape; exact decision parity is required.",
            "Cold model load, evidence extraction, SDK result construction, and I/O are excluded.",
            "Overall p95 is accompanied by escalated p95, p99, and maximum to expose the tail.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["latency"], indent=2))


if __name__ == "__main__":
    main()
