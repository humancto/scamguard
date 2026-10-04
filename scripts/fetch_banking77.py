#!/usr/bin/env python3
"""Fetch the publisher-pinned BANKING77 source files."""

from __future__ import annotations

import argparse
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path

import certifi

try:
    from scripts.build_banking77_safe import SOURCE_FILES
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the repo on sys.path.
    from build_banking77_safe import SOURCE_FILES  # type: ignore[no-redef]

from scamguard.metrics import file_sha256


def fetch(destination: Path, *, force: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    for source in SOURCE_FILES:
        output = destination / source.filename
        if output.exists() and not force:
            if file_sha256(output) == source.sha256:
                print(f"verified BANKING77 source: {output}")
                continue
            raise RuntimeError(f"existing BANKING77 source has wrong hash: {output}")
        request = urllib.request.Request(
            source.url,
            headers={"User-Agent": "ScamGuard-dataset-fetcher/0.1 (+research; hash-pinned)"},
        )
        temporary: Path | None = None
        try:
            with urllib.request.urlopen(request, timeout=90, context=context) as response:
                with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                    temporary = Path(handle.name)
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            actual = file_sha256(temporary)
            if actual != source.sha256:
                raise RuntimeError(
                    f"hash mismatch for {source.filename}: expected {source.sha256}, got {actual}"
                )
            os.replace(temporary, output)
            temporary = None
            print(f"downloaded BANKING77 source: {output}")
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(args.destination, force=args.force)


if __name__ == "__main__":
    main()
