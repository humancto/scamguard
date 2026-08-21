#!/usr/bin/env python3
"""Fetch only the gated, text-metadata portion of pinned TeleAntiFraud-28k."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from scamguard.metrics import file_sha256

REPOSITORY = "JimmyMa99/TeleAntiFraud"
REVISION = "0872e54b584b28d34e0911dffcf696f0b2e5e49a"
FILES = ("README.md", "dataset_manifest.json", "binary_classification.zip")


def verify_existing(destination: Path, receipt: dict[str, object]) -> bool:
    if receipt.get("repository") != REPOSITORY or receipt.get("revision") != REVISION:
        return False
    files = receipt.get("files")
    if not isinstance(files, dict):
        return False
    for filename in FILES:
        metadata = files.get(filename)
        path = destination / filename
        if not isinstance(metadata, dict) or not path.is_file():
            return False
        if file_sha256(path) != metadata.get("sha256"):
            return False
    return True


def fetch(destination: Path) -> dict[str, object]:
    receipt_path = destination / "download_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if verify_existing(destination, receipt):
            return receipt
        raise RuntimeError("existing TeleAntiFraud files do not match their download receipt")

    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("refusing to mix a new download with an unverified destination")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        cached = {
            filename: Path(
                hf_hub_download(
                    repo_id=REPOSITORY,
                    repo_type="dataset",
                    revision=REVISION,
                    filename=filename,
                )
            )
            for filename in FILES
        }
    except GatedRepoError as error:
        raise RuntimeError(
            "TeleAntiFraud access is gated. Accept its Hugging Face conditions, run `hf auth "
            "login`, and retry; never paste the token into a dataset file or chat."
        ) from error
    except HfHubHTTPError as error:
        raise RuntimeError(f"pinned TeleAntiFraud download failed: {error}") from error

    for filename, cached_path in cached.items():
        shutil.copy2(cached_path, destination / filename)
    receipt: dict[str, object] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license_declared_by_publisher": "Apache-2.0",
        "scope": "binary text metadata only; audio.zip and SFT archives intentionally excluded",
        "files": {
            filename: {
                "bytes": (destination / filename).stat().st_size,
                "sha256": file_sha256(destination / filename),
            }
            for filename in FILES
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/teleantifraud"),
    )
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination), indent=2))


if __name__ == "__main__":
    main()
