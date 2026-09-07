#!/usr/bin/env python3
"""Fetch the exact MIT-declared English phone-scam dialogue release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from scamguard.metrics import file_sha256

REPOSITORY = "shakeleoatmeal/phone-scam-detection-synthetic"
REVISION = "27a1f6d0cf21dda995130d221383900b060dbcc9"
LICENSE = "MIT"
FILES = {
    "README.md": {
        "destination": "README.md",
        "bytes": 1061,
        "sha256": "38171953e18bba0a66f5aa7f3bf11039ebbb4e033afbde95d38b46dd2a04d7ac",
    },
    "data/train-00000-of-00001.parquet": {
        "destination": "train.parquet",
        "bytes": 704685,
        "sha256": "628452503397e07261192fa361e1f9a0e02c6ac7eca61a2ea82406f54feb51e2",
    },
    "data/validation-00000-of-00001.parquet": {
        "destination": "validation.parquet",
        "bytes": 203848,
        "sha256": "034237e6f3a907c599d2b784c668f673e60fb9b943ed4d93561dadfaa886292c",
    },
    "data/test-00000-of-00001.parquet": {
        "destination": "test.parquet",
        "bytes": 101114,
        "sha256": "91687ddc04bb59b9865ed001782a8a542d9704491d11efe3bacbb7e15da124fb",
    },
}


def verify_files(destination: Path) -> dict[str, dict[str, object]]:
    verified: dict[str, dict[str, object]] = {}
    for source_name, contract in FILES.items():
        path = destination / str(contract["destination"])
        if not path.is_file():
            raise FileNotFoundError(f"missing phone-scam source file: {path}")
        size = path.stat().st_size
        digest = file_sha256(path)
        if size != contract["bytes"] or digest != contract["sha256"]:
            raise RuntimeError(f"phone-scam source file differs: {source_name}")
        verified[str(contract["destination"])] = {
            "source_path": source_name,
            "bytes": size,
            "sha256": digest,
        }
    return verified


def verify_receipt(destination: Path, receipt: dict[str, object]) -> bool:
    if (
        receipt.get("repository") != REPOSITORY
        or receipt.get("revision") != REVISION
        or receipt.get("license_declared_by_publisher") != LICENSE
    ):
        return False
    try:
        return receipt.get("files") == verify_files(destination)
    except (FileNotFoundError, RuntimeError):
        return False


def fetch(destination: Path) -> dict[str, object]:
    receipt_path = destination / "download_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if verify_receipt(destination, receipt):
            return receipt
        raise RuntimeError("existing phone-scam files do not match their download receipt")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("refusing to mix a new download with an unverified destination")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        cached = {
            source_name: Path(
                hf_hub_download(
                    repo_id=REPOSITORY,
                    repo_type="dataset",
                    revision=REVISION,
                    filename=source_name,
                )
            )
            for source_name in FILES
        }
    except HfHubHTTPError as error:
        raise RuntimeError(f"pinned phone-scam dataset download failed: {error}") from error
    for source_name, cached_path in cached.items():
        shutil.copy2(cached_path, destination / str(FILES[source_name]["destination"]))
    receipt: dict[str, object] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license_declared_by_publisher": LICENSE,
        "license_evidence": "README front matter and Hugging Face dataset metadata",
        "scope": "README plus publisher train, validation, and test text parquet files only",
        "synthetic": True,
        "files": verify_files(destination),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/phone_scam_synthetic"),
    )
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination), indent=2))


if __name__ == "__main__":
    main()
