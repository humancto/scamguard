#!/usr/bin/env python3
"""Fail closed when a continual-encoder experiment no longer matches frozen inputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scamguard.metrics import file_sha256


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_sample_probability(
    rows: list[dict[str, object]], alpha: float
) -> dict[str, float]:
    counts = Counter(str(row["source"]) for row in rows)
    total_mass = sum(count ** (1.0 - alpha) for count in counts.values())
    return {
        source: count ** (1.0 - alpha) / total_mass
        for source, count in sorted(counts.items())
    }


def model_file(checkpoint: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing model weights in {checkpoint}")


def verify(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    data_dir = Path(data["directory"])
    teacher = config["teacher"]
    initialization = config["initialization"]
    expected_hashes = {
        data_dir / "manifest.json": data["manifest_sha256"],
        data_dir / "train.jsonl": data["train_sha256"],
        data_dir / "dev.jsonl": data["dev_sha256"],
        data_dir / "test.jsonl": data["test_sha256"],
        Path(teacher["ledger"]): teacher["ledger_sha256"],
        Path(teacher["manifest"]): teacher["manifest_sha256"],
        model_file(Path(initialization["checkpoint"])): initialization["model_sha256"],
    }
    mismatches = [
        f"{path}: expected {expected}, found {file_sha256(path)}"
        for path, expected in expected_hashes.items()
        if file_sha256(path) != expected
    ]
    train_rows = read_jsonl(data_dir / "train.jsonl")
    if len(train_rows) != data["train_rows"]:
        mismatches.append(
            f"train rows: expected {data['train_rows']}, found {len(train_rows)}"
        )
    source_counts = Counter(str(row["source"]) for row in train_rows)
    for source, expected in data["new_supervised_rows"].items():
        if source_counts[source] != expected:
            mismatches.append(
                f"source {source}: expected {expected}, found {source_counts[source]}"
            )
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != data["schema_version"]:
        mismatches.append("processed data schema version differs from config")
    if manifest.get("schema14_increment", {}).get("sealed_ood_opened") is not False:
        mismatches.append("YouTube-call OOD is not recorded as sealed")
    if manifest.get("schema15_increment", {}).get("apptek_ood_opened") is not False:
        mismatches.append("AppTek OOD is not recorded as sealed")

    teacher_manifest = json.loads(Path(teacher["manifest"]).read_text(encoding="utf-8"))
    if teacher_manifest.get("rows") != teacher["anchor_rows"]:
        mismatches.append("teacher anchor row count differs from config")
    if teacher_manifest.get("contains_text") is not False or teacher["contains_text"] is not False:
        mismatches.append("teacher cache is not explicitly text-free")
    if teacher_manifest.get("checkpoint_model_sha256") != initialization["model_sha256"]:
        mismatches.append("teacher checkpoint differs from initialization checkpoint")

    actual_probability = source_sample_probability(
        train_rows,
        float(config["training"]["source_balance_alpha"]),
    )
    expected_probability = config["expected_sampling_probability"]
    if set(actual_probability) != set(expected_probability) or any(
        not math.isclose(actual_probability[source], expected_probability[source], abs_tol=1e-12)
        for source in actual_probability
    ):
        mismatches.append("source sampling probabilities differ from config")
    if mismatches:
        raise SystemExit("continual experiment preflight failed:\n" + "\n".join(mismatches))
    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "train_rows": len(train_rows),
        "teacher_anchor_rows": teacher["anchor_rows"],
        "new_supervised_rows": data["new_supervised_rows"],
        "source_sampling_probability": actual_probability,
        "sealed_artifacts_opened": False,
        "status": "preflight_passed",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/encoder-schema16-retention-alpha05-w2.json"),
    )
    args = parser.parse_args()
    verify(args.config)


if __name__ == "__main__":
    main()
