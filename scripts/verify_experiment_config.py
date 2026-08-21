#!/usr/bin/env python3
"""Fail before training when an immutable experiment config no longer matches data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scamguard.metrics import file_sha256


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    expected = {
        args.data / "manifest.json": data["manifest_sha256"],
        args.data / "qwen_sft/train.jsonl": data["train_jsonl_sha256"],
        args.data / "qwen_sft/dev.jsonl": data["dev_jsonl_sha256"],
    }
    for key, digest in data.get("evaluation", {}).items():
        split = key.removesuffix("_sha256")
        expected[args.data / f"{split}.jsonl"] = digest

    mismatches = []
    for path, recorded in expected.items():
        actual = file_sha256(path)
        if actual != recorded:
            mismatches.append(f"{path}: recorded {recorded}, actual {actual}")
    counts = {
        "train_examples": line_count(args.data / "qwen_sft/train.jsonl"),
        "dev_examples": line_count(args.data / "qwen_sft/dev.jsonl"),
    }
    for key, actual in counts.items():
        if data[key] != actual:
            mismatches.append(f"{key}: recorded {data[key]}, actual {actual}")
    if mismatches:
        raise SystemExit("experiment preflight failed:\n" + "\n".join(mismatches))
    print(f"experiment preflight passed: {config['experiment_id']}")


if __name__ == "__main__":
    main()
