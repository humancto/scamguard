#!/usr/bin/env python3
"""Measure the frozen encoder-to-GGUF route with both models persistently loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_routed_transformers_runtime import (
    latency_report,
    peak_rss_bytes,
    read_jsonl,
    trace_requests,
)
from scamguard.gguf_runtime import QwenGGUFVerdictBackend
from scamguard.metrics import file_sha256
from scamguard.model import TransformersBackend
from training.eval_routed import (
    evaluate_records,
    join_split,
    read_prediction_ledger,
    route_records,
)


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"cannot identify llama.cpp revision: {completed.stderr.strip()}")
    return completed.stdout.strip()


def resident_set_bytes(process_id: int) -> int:
    if process_id < 1:
        raise ValueError("process ID must be positive")
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(process_id)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode or not completed.stdout.strip():
            raise RuntimeError("could not read native runner resident memory")
        return int(completed.stdout.strip()) * 1024
    statm = Path(f"/proc/{process_id}/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    raise RuntimeError("resident-memory sampling is unsupported on this platform")


def validate_quantized_evidence(
    report: dict[str, Any],
    *,
    model: Path,
    model_sha256: str,
    predictions: Path,
    calibration: Path,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    n_gpu_layers: int,
) -> None:
    if report.get("model_sha256") != model_sha256:
        raise ValueError("quantized report identifies a different GGUF model")
    if Path(str(report.get("model", ""))).expanduser().resolve() != model.resolve():
        raise ValueError("quantized report path differs from the selected GGUF model")
    prediction_ledger = report.get("prediction_ledger", {})
    if prediction_ledger.get("sha256") != file_sha256(predictions):
        raise ValueError("quantized prediction ledger differs from its quality report")
    calibration_record = report.get("calibration", {})
    if calibration_record.get("sha256") != file_sha256(calibration):
        raise ValueError("quantized report reused a different calibration artifact")
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    for field in ("temperature", "scam_threshold", "safe_threshold"):
        if float(report.get(field, float("nan"))) != float(calibration_payload[field]):
            raise ValueError(f"quantized report calibration mismatch: {field}")
    if report.get("safe_threshold_semantics") != "minimum_safe_probability":
        raise ValueError("quantized report SAFE-threshold semantics are incompatible")
    parity = report.get("quantization_parity", {})
    if (
        parity.get("exact_calibrated_verdict_parity") is not True
        or parity.get("release_gate_passed") is not True
    ):
        raise ValueError("quantized quality report lacks exact frozen verdict parity")
    runtime = report.get("runtime_config", {})
    expected = {
        "ctx_size": ctx_size,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "n_gpu_layers": n_gpu_layers,
        "parallel": 1,
    }
    for field, value in expected.items():
        if runtime.get(field) != value:
            raise ValueError(f"quantized quality runtime mismatch: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--quantization", choices=("Q4_K_M", "Q5_K_M", "Q8_0"), required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bf16-report", type=Path, required=True)
    parser.add_argument("--specialist-report", type=Path, required=True)
    parser.add_argument("--specialist-predictions", type=Path, required=True)
    parser.add_argument("--routed-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--probability-tolerance", type=float, default=5e-3)
    parser.add_argument("--ctx-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.probability_tolerance <= 0.0:
        parser.error("--probability-tolerance must be positive")
    for path in (
        args.router_checkpoint,
        args.router_predictions,
        args.runner,
        args.llama_source,
        args.model,
        args.processor,
        args.calibration,
        args.bf16_report,
        args.specialist_report,
        args.specialist_predictions,
        args.routed_report,
        args.data,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if file_sha256(args.runner) != args.runner_sha256:
        raise ValueError("native runner SHA-256 differs from --runner-sha256")
    if file_sha256(args.model) != args.model_sha256:
        raise ValueError("GGUF model SHA-256 differs from --model-sha256")
    if git_revision(args.llama_source) != args.llama_revision:
        raise ValueError("llama.cpp source revision differs from --llama-revision")

    import torch

    if args.require_mps and not torch.backends.mps.is_available():
        raise RuntimeError("--require-mps was set but Metal is unavailable")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    routed_report = json.loads(args.routed_report.read_text(encoding="utf-8"))
    specialist_report = json.loads(args.specialist_report.read_text(encoding="utf-8"))
    bf16_report = json.loads(args.bf16_report.read_text(encoding="utf-8"))
    validate_quantized_evidence(
        specialist_report,
        model=args.model,
        model_sha256=args.model_sha256,
        predictions=args.specialist_predictions,
        calibration=args.calibration,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        n_gpu_layers=args.n_gpu_layers,
    )
    if routed_report["inputs"]["router_sha256"] != file_sha256(args.router_predictions):
        raise ValueError("router ledger differs from routed policy report")
    if routed_report["inputs"]["specialist_sha256"] != file_sha256(
        args.specialist_predictions
    ):
        raise ValueError("GGUF ledger differs from routed policy report")
    margin_max = float(routed_report["policy"]["margin_max"])

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
    specialist = QwenGGUFVerdictBackend(
        runner=args.runner,
        model=args.model,
        processor=args.processor,
        calibration=args.calibration,
        expected_model_sha256=args.model_sha256,
        expected_runner_sha256=args.runner_sha256,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    try:
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
        runtime_identity = specialist.runtime_identity()
        specialist_resident_bytes = resident_set_bytes(specialist.scorer.process_id)
        router_process_peak_bytes = peak_rss_bytes()
    finally:
        specialist.close()

    trace_path = args.traces or args.report.with_suffix(".traces.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in traces),
        encoding="utf-8",
    )
    first_repetition = [record for record in traces if record["repetition"] == 0]
    runtime_quality = evaluate_records(first_repetition)
    frozen_quality = routed_report["test"]
    latency = latency_report(traces)
    latency["scope"] = (
        "batch-one router tokenization/inference, frozen route decision, and selective specialist "
        "prompt rendering, local pipe round trip, native GGUF tokenization, three-candidate "
        "likelihood scoring, calibration, and verdict; both models persistently loaded; excludes "
        "cold load, evidence extraction, and SDK result construction"
    )
    selected_ids_sha256 = hashlib.sha256("\n".join(row_ids).encode()).hexdigest()
    result = {
        "artifact_schema_version": 1,
        "purpose": "persistent quantized GGUF routed-runtime release evidence",
        "measurement_mode": "interleaved_persistent_per_request",
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
        },
        "specialist": {
            "model_id": specialist.model_id,
            "quantization": args.quantization,
            "calibration_report": str(args.bf16_report),
            "calibration_report_sha256": file_sha256(args.bf16_report),
            "quantized_quality_report": str(args.specialist_report),
            "quantized_quality_report_sha256": file_sha256(args.specialist_report),
            "frozen_quality_scoring": {
                "message_batch_size": bf16_report["score_cache"]["message_batch_size"],
                "candidate_batch_size": bf16_report["score_cache"]["candidate_batch_size"],
                "sequence_bucket_size": bf16_report["score_cache"][
                    "sequence_bucket_size"
                ],
            },
            "runtime_scoring": {
                "message_batch_size": 1,
                "candidate_batch_size": 3,
                "sequence_bucket_size": 64,
            },
            "native_runtime": runtime_identity,
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
            "false_positive_rate": runtime_quality["binary_safety"][
                "false_positive_rate"
            ]
            - frozen_quality["binary_safety"]["false_positive_rate"],
            "macro_f1": runtime_quality["macro_f1"] - frozen_quality["macro_f1"],
        },
        "latency": latency,
        "memory": {
            "router_process_peak_rss_bytes": router_process_peak_bytes,
            "specialist_process_resident_bytes_after_trace": specialist_resident_bytes,
            "combined_conservative_bytes": (
                router_process_peak_bytes + specialist_resident_bytes
            ),
            "limitation": (
                "sum may double-count shared pages; child value is post-trace resident memory, "
                "not an independently sampled peak"
            ),
        },
        "process_peak_rss_bytes": router_process_peak_bytes + specialist_resident_bytes,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "router_device": device,
            "llama_cpp_revision": args.llama_revision,
            "runner_source_sha256": file_sha256(
                Path(__file__).resolve().parents[1] / "native" / "gguf_verdict_runner.cpp"
            ),
        },
        "trace_ledger": {
            "path": str(trace_path),
            "sha256": file_sha256(trace_path),
            "rows": len(traces),
            "contains_message_text": False,
        },
        "limitations": [
            "Desktop Metal timing does not establish physical-phone latency.",
            "Local pipe overhead is included so the native subprocess architecture is not hidden.",
            "Cold model load, evidence extraction, and SDK result construction are excluded.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["latency"], indent=2))


if __name__ == "__main__":
    main()
