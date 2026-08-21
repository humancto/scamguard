#!/usr/bin/env python3
"""Fetch the pinned CC0 YouTube scam-call transcript archive from Kaggle."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from scamguard.metrics import file_sha256

DATASET = "rivalcults/youtube-scam-phone-call-transcripts"
VERSION = 2
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    f"{DATASET}?datasetVersionNumber={VERSION}"
)
EXPECTED_SHA256 = "3f67497736e9421c2f6e59efc46c129006419d40fc752cbb981042940384cedd"
EXPECTED_BYTES = 149_701


def receipt(destination: Path) -> dict[str, object]:
    return {
        "dataset": DATASET,
        "version": VERSION,
        "download_url": DOWNLOAD_URL,
        "license_declared_by_publisher": "CC0-1.0",
        "publisher_description": (
            "243 manually corrected partial transcripts sourced from YouTube scam calls; "
            "mostly scammer/scambaiter calls, with some autodialer messages; publisher says "
            "names, addresses, and phone numbers were removed"
        ),
        "archive": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        },
    }


def fetch(destination: Path) -> dict[str, object]:
    if destination.is_file():
        if file_sha256(destination) != EXPECTED_SHA256:
            raise RuntimeError("existing YouTube scam-call archive differs from the pinned source")
        return receipt(destination)
    if destination.exists():
        raise RuntimeError(f"download destination exists and is not a file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "ScamGuard/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
    if len(payload) != EXPECTED_BYTES:
        raise RuntimeError(
            "YouTube scam-call archive size mismatch: "
            f"expected {EXPECTED_BYTES}, got {len(payload)}"
        )
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        if file_sha256(temporary) != EXPECTED_SHA256:
            raise RuntimeError("YouTube scam-call archive hash differs from pinned version 2")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/youtube_scam_calls_v2.zip"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/raw/youtube_scam_calls_v2.receipt.json"),
    )
    args = parser.parse_args()
    result = fetch(args.destination)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
