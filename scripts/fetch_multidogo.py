#!/usr/bin/env python3
"""Fetch the pinned MultiDoGO human-dialogue text without unrelated artifacts."""

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

REPOSITORY = "https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset.git"
REVISION = "baa30639c4b271f394b81443c842193407cdf26d"
LICENSE = "CDLA-Permissive-1.0"
LICENSE_SHA256 = "8be8b09ba4230a6ab89a62439b45e3374e15870360d27c9a51131592b91a2f10"
README_SHA256 = "1a36fb24ceb44a75224ca4566b816024abc0b05ba13d27c959809b243440008b"
DIALOGUE_TREE_SHA256 = "0196e5ae82fc3b8c488b82d0a3cdf8dca74911a8bab5fa5ed5e1bf6ceee2ae97"
DOMAINS = ("airline", "fastfood", "finance", "insurance", "media", "software")
EXPECTED_FILE_SHA256 = {
    "airline": "d820e68f3199464700e6a5911ceb3f53f5cecc7606d65165874157567c038246",
    "fastfood": "1924c74a7c2a334205d8e9181f50b58b9a2d01391784687fc1c035ae4b00d00d",
    "finance": "52d36794c3c9c49b67444c193a6324ad32bfb46e3550b15e95b35ec31e9a9d3c",
    "insurance": "f2ef72d869becc16ac1f76d562dba1111feeefd2dbce72230aef374ab354a9f3",
    "media": "0b01a2a913f47e8d64a9a9bd18674148b32014cfd294f60cb82830c999a7a358",
    "software": "3f90adc3c6c4f81edad4227e020d669c36c8f963061216b845ab5f853132de47",
}


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
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
        raise RuntimeError(f"MultiDoGO revision differs: {revision}")
    if file_sha256(path / "LICENSE.txt") != LICENSE_SHA256:
        raise RuntimeError("MultiDoGO license differs from the pinned artifact")
    if file_sha256(path / "README.md") != README_SHA256:
        raise RuntimeError("MultiDoGO README differs from the pinned artifact")
    files = [path / "data" / "unannotated" / f"{domain}.tsv" for domain in DOMAINS]
    if not all(item.is_file() for item in files):
        raise RuntimeError("MultiDoGO dialogue tree is incomplete")
    for domain, file_path in zip(DOMAINS, files, strict=True):
        if file_sha256(file_path) != EXPECTED_FILE_SHA256[domain]:
            raise RuntimeError(f"MultiDoGO {domain} file differs from the pinned artifact")
    if tree_sha256(files, path) != DIALOGUE_TREE_SHA256:
        raise RuntimeError("MultiDoGO dialogue tree differs from the pinned release")
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "readme_sha256": README_SHA256,
        "domains": list(DOMAINS),
        "dialogue_files": len(files),
        "dialogue_tree_sha256": DIALOGUE_TREE_SHA256,
        "audio_downloaded": False,
        "citation": "https://aclanthology.org/D19-1460/",
    }


def fetch(output: Path) -> dict[str, object]:
    if output.exists():
        return verify_repository(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="multidogo-", dir=output.parent) as temporary:
        checkout = Path(temporary) / "repository"
        run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(checkout))
        run(
            "git",
            "sparse-checkout",
            "set",
            "--no-cone",
            "/LICENSE.txt",
            "/README.md",
            "/NOTICE",
            "/data/unannotated/",
            cwd=checkout,
        )
        run("git", "checkout", "--detach", REVISION, cwd=checkout)
        manifest = verify_repository(checkout)
        os.replace(checkout, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/multidogo/repository")
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
