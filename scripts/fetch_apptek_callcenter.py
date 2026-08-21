#!/usr/bin/env python3
"""Fetch only pinned text metadata for the AppTek call-center benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from scamguard.metrics import file_sha256

DATASET = "apptek-com/apptek_callcenter_dialogues"
REVISION = "95a8c157e4fd6df2f3c77593160c83db79b75dc7"
LICENSE = "CC-BY-SA-4.0"
SOURCE_FILES: dict[str, dict[str, object]] = {
    "en-AU": {
        "bytes": 1_156_772,
        "sha256": "c3afc921b1b2a602dffb7ef8c77f751da3c9bbd84976fd75761ee2d7d26f61f4",
    },
    "en-CA": {
        "bytes": 1_171_827,
        "sha256": "e96d13b4d66774b1c3df08335c7599e1405965ffaf79d08c6545ab6fa3bccdc6",
    },
    "en-CN": {
        "bytes": 996_534,
        "sha256": "0465e1e80d6bd63821867c2152ccc10dbb7bb86a18d32402948cee1ac30376e7",
    },
    "en-GB": {
        "bytes": 1_412_857,
        "sha256": "9a2f19a4045d038330cd53a52ed4e8a81b4b7a782d87ce41f655a329364d8d83",
    },
    "en-GB_SCT": {
        "bytes": 1_524_634,
        "sha256": "2b79ca8945989ceab44d6e8b14db5ef5a0e62179393fc2c5635231c739f82c25",
    },
    "en-GB_WLS": {
        "bytes": 1_500_338,
        "sha256": "c067a862cabf69fbe5d8cb88a3c5753b8abe3f285c62fd728b7eed76d07c73b6",
    },
    "en-IE": {
        "bytes": 1_338_187,
        "sha256": "83f52b0b530bfb416493a68da0a48692f86a0818721091d78c8c1680500dacc7",
    },
    "en-IN": {
        "bytes": 1_199_680,
        "sha256": "640081da5996c547d083108a800f23939abee1dd6e9b243bf32fe1d30b7692dd",
    },
    "en-MX": {
        "bytes": 1_086_287,
        "sha256": "7d3f9e7920991eebdc6ea9877daabe42647f9dba0dce8e387e0db7bbd45ffce9",
    },
    "en-SG": {
        "bytes": 1_120_767,
        "sha256": "f1cc73e3cc5e8a39884b0296590089502eb4d5853bf11e0a6cf17cacbc965df1",
    },
    "en-US_Aave": {
        "bytes": 1_262_820,
        "sha256": "72370e6f98a8a73d987f9f8e9e242e44c34ca76ee8baf3e6800242429c66c01a",
    },
    "en-US_General": {
        "bytes": 1_266_982,
        "sha256": "2c37db51246dd8be8c3e1946afd8ee5ad15ef5857db4e1a39393abc711524d72",
    },
    "en-US_Southern": {
        "bytes": 1_280_115,
        "sha256": "0430583c0ce7de77dcf489ea2ef6c8202800edf9d7e8a2b2bacc46cd7ce4d495",
    },
    "en-ZA": {
        "bytes": 1_120_686,
        "sha256": "a1f001460b7e9730c6019328adc84b18a24a0a2e1b6abbb7fc9087d166555362",
    },
}


def source_url(accent: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
        f"diarization/{accent}/metadata.jsonl"
    )


def verify(path: Path, expected: dict[str, object]) -> None:
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"AppTek metadata byte count differs for {path.name}")
    if file_sha256(path) != expected["sha256"]:
        raise RuntimeError(f"AppTek metadata hash differs for {path.name}")


def fetch(
    destination: Path,
    *,
    source_files: dict[str, dict[str, object]] = SOURCE_FILES,
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {}
    for accent, expected in source_files.items():
        path = destination / f"{accent}.jsonl"
        if path.is_file():
            verify(path, expected)
        elif path.exists():
            raise RuntimeError(f"AppTek destination exists and is not a file: {path}")
        else:
            request = urllib.request.Request(
                source_url(accent), headers={"User-Agent": "ScamGuard/0.1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                payload = response.read()
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            try:
                verify(temporary, expected)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        artifacts[accent] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "license_declared_by_publisher": LICENSE,
        "data_use": "text-only evaluation and analysis; never fitting",
        "audio_downloaded": False,
        "total_bytes": sum(int(value["bytes"]) for value in artifacts.values()),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/apptek_callcenter_dialogues"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/raw/apptek_callcenter_dialogues.receipt.json"),
    )
    args = parser.parse_args()
    result = fetch(args.destination)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
