#!/usr/bin/env python3
"""Fetch the pinned HarperValleyBank transcript and metadata trees without audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from scamguard.metrics import file_sha256

REPOSITORY = "https://github.com/cricketclub/gridspace-stanford-harper-valley.git"
REVISION = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
LICENSE = "CC-BY-4.0"
LICENSE_SHA256 = "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"
TRANSCRIPT_TREE_SHA256 = "99f30d235cf79bcfbb3438ff472e3e4ed2dcdb671512cde63da60024ad75b807"
METADATA_TREE_SHA256 = "d527d581d8124167c9e6b838cd5e02c600bfc23f6aee242ebe589ae4dc1fb042"


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_repository(path: Path) -> dict[str, object]:
    revision = run("git", "rev-parse", "HEAD", cwd=path)
    if revision != REVISION:
        raise RuntimeError(f"HarperValleyBank revision differs: {revision}")
    license_path = path / "LICENSE"
    if file_sha256(license_path) != LICENSE_SHA256:
        raise RuntimeError("HarperValleyBank license differs from the pinned artifact")
    transcript_files = sorted((path / "data" / "transcript").glob("*.json"))
    metadata_files = sorted((path / "data" / "metadata").glob("*.json"))
    if len(transcript_files) != 1446 or len(metadata_files) != 1446:
        raise RuntimeError("HarperValleyBank source count differs from the pinned release")
    if {item.stem for item in transcript_files} != {item.stem for item in metadata_files}:
        raise RuntimeError("HarperValleyBank transcript and metadata IDs differ")
    if tree_sha256(transcript_files, path) != TRANSCRIPT_TREE_SHA256:
        raise RuntimeError("HarperValleyBank transcript tree differs from the pinned release")
    if tree_sha256(metadata_files, path) != METADATA_TREE_SHA256:
        raise RuntimeError("HarperValleyBank metadata tree differs from the pinned release")
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "transcript_files": len(transcript_files),
        "transcript_tree_sha256": TRANSCRIPT_TREE_SHA256,
        "metadata_files": len(metadata_files),
        "metadata_tree_sha256": METADATA_TREE_SHA256,
        "audio_downloaded": False,
        "citation": "https://arxiv.org/abs/2010.13929",
    }


def fetch(output: Path) -> dict[str, object]:
    if output.exists():
        return verify_repository(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="harper-valley-", dir=output.parent) as temporary:
        checkout = Path(temporary) / "repository"
        run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(checkout))
        run(
            "git",
            "sparse-checkout",
            "set",
            "--no-cone",
            "/LICENSE",
            "/README.md",
            "/data/transcript/",
            "/data/metadata/",
            cwd=checkout,
        )
        run("git", "checkout", "--detach", REVISION, cwd=checkout)
        manifest = verify_repository(checkout)
        os.replace(checkout, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/harper_valley/repository"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.force and args.output.exists():
        shutil.rmtree(args.output)
    manifest = fetch(args.output)
    manifest_path = args.output.parent / "source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
