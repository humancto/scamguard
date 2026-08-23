#!/usr/bin/env python3
"""Benchmark the hash-verified GGUF pack through ScamGuard's public Scanner API."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_routed_transformers_runtime import summarize
from scamguard.gguf_runtime import PACK_MANIFEST_NAME, QwenGGUFVerdictBackend
from scamguard.metrics import file_sha256
from scamguard.scanner import Scanner


def read_rows(path: Path, count: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][:count]
    if len(rows) != count:
        raise ValueError(f"requested {count} rows, found {len(rows)}")
    return rows


def benchmark(
    *, pack: Path, data: Path, split: str, rows: int, repetitions: int
) -> dict[str, Any]:
    if rows < 1 or repetitions < 1:
        raise ValueError("rows and repetitions must be positive")
    manifest_path = pack / PACK_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = data / f"{split}.jsonl"
    selected = read_rows(split_path, rows)
    latencies: list[float] = []
    native_latencies: list[float] = []
    verdict_records: list[str] = []
    prefix_reused = True
    transformers_was_loaded = "transformers" in sys.modules
    started = time.perf_counter()
    with Scanner(model_path=str(pack)) as scanner:
        backend = scanner.backend
        if not isinstance(backend, QwenGGUFVerdictBackend):
            raise TypeError("runtime pack did not load the native GGUF backend")
        scanner.scan(str(selected[0]["text"]))
        for repetition in range(repetitions):
            for row in selected:
                result = scanner.scan(str(row["text"]))
                latencies.append(backend.last_round_trip_ms)
                native_latencies.append(backend.last_native_elapsed_ms)
                prefix_reused = prefix_reused and backend.last_prefix_reused
                verdict_records.append(
                    f"{repetition}\t{row['id']}\t{result.verdict.value}"
                )
        runtime_identity = backend.runtime_identity()
    wall_seconds = time.perf_counter() - started
    round_trip = summarize(latencies)
    native = summarize(native_latencies)
    transformers_loaded_after = "transformers" in sys.modules
    transformers_imported_by_runtime = (
        not transformers_was_loaded and transformers_loaded_after
    )
    return {
        "artifact_schema_version": 1,
        "purpose": "public Scanner API GGUF runtime-pack benchmark",
        "pack": {
            "path": str(pack),
            "manifest_sha256": file_sha256(manifest_path),
            "purpose": manifest.get("purpose"),
            "publication_authorized": manifest.get("publication_authorized"),
        },
        "data": {
            "path": str(split_path),
            "sha256": file_sha256(split_path),
            "rows": rows,
            "repetitions": repetitions,
            "requests": len(latencies),
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(str(row["id"]) for row in selected).encode()
            ).hexdigest(),
            "contains_message_text": False,
        },
        "runtime": runtime_identity,
        "sdk_round_trip_latency": {**round_trip, "samples_ms": latencies},
        "native_elapsed_latency": {**native, "samples_ms": native_latencies},
        "verdict_ledger_sha256": hashlib.sha256(
            "\n".join(verdict_records).encode()
        ).hexdigest(),
        "wall_seconds_including_startup_and_warmup": wall_seconds,
        "transformers_was_loaded": transformers_was_loaded,
        "transformers_loaded_after": transformers_loaded_after,
        "transformers_imported_by_runtime": transformers_imported_by_runtime,
        "gates": {
            "all_requests_reused_prefix": prefix_reused,
            "sdk_p95_under_50_ms": bool(
                round_trip["p95_ms"] is not None
                and float(round_trip["p95_ms"]) < 50.0
            ),
            "runtime_does_not_import_transformers": not transformers_imported_by_runtime,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "limitations": [
            "An upstream base-model control establishes runtime plumbing, not ScamGuard quality.",
            "Desktop Metal timing is not physical-phone evidence.",
            "The final trained and quantized pack must repeat this benchmark and all "
            "quality gates.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(
        pack=args.pack,
        data=args.data,
        split=args.split,
        rows=args.rows,
        repetitions=args.repetitions,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pack": result["pack"],
                "sdk_round_trip_latency": result["sdk_round_trip_latency"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
