#!/usr/bin/env python3
"""Fetch and verify the pinned upstream Qwen3.5-0.8B Q4_0 control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from scamguard.metrics import file_sha256

REPO_ID = "ggml-org/Qwen3.5-0.8B-GGUF"
REVISION = "8fea620810c4afa23dd6443f999a48574c1611a3"
FILENAME = "Qwen3.5-0.8B-Q4_0.gguf"
EXPECTED_BYTES = 563_036_064
EXPECTED_SHA256 = "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"


def verify_artifact(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"GGUF byte count differs: expected {expected_bytes}, got {actual_bytes}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"GGUF SHA-256 differs: expected {expected_sha256}, got {actual_sha256}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gguf"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("reports/source-audits/qwen35-08b-upstream-q4-control.json"),
    )
    args = parser.parse_args()
    destination = args.output_dir / FILENAME
    if destination.is_file():
        verify_artifact(destination, EXPECTED_BYTES, EXPECTED_SHA256)
    else:
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                revision=REVISION,
                local_dir=args.output_dir,
            )
        )
        if downloaded.resolve() != destination.resolve():
            raise ValueError(f"download destination differs: {downloaded}")
        verify_artifact(destination, EXPECTED_BYTES, EXPECTED_SHA256)
    receipt = {
        "artifact_schema_version": 1,
        "role": "unmodified quantized base runtime control; never a training input",
        "repository": REPO_ID,
        "repository_revision": REVISION,
        "filename": FILENAME,
        "license": "Apache-2.0",
        "bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
        "local_path": str(destination),
        "verification_status": "passed",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
