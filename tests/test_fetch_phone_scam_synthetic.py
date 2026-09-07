from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.fetch_phone_scam_synthetic as source
from scamguard.metrics import file_sha256


def materialize_contract(tmp_path: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for source_name, contract in source.FILES.items():
        path = tmp_path / str(contract["destination"])
        path.write_bytes(source_name.encode("utf-8"))
        files[path.name] = {
            "source_path": source_name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return {
        "repository": source.REPOSITORY,
        "revision": source.REVISION,
        "license_declared_by_publisher": source.LICENSE,
        "files": files,
    }


def test_verify_receipt_requires_exact_identity_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = materialize_contract(tmp_path)
    monkeypatch.setattr(
        source,
        "FILES",
        {
            metadata["source_path"]: {
                "destination": name,
                "bytes": metadata["bytes"],
                "sha256": metadata["sha256"],
            }
            for name, metadata in receipt["files"].items()
        },
    )

    assert source.verify_receipt(tmp_path, receipt)
    (tmp_path / "README.md").write_text("tampered", encoding="utf-8")
    assert not source.verify_receipt(tmp_path, receipt)


def test_fetch_rejects_unverified_nonempty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "source"
    destination.mkdir()
    (destination / "unknown").write_text("data", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to mix"):
        source.fetch(destination)


def test_fetch_rejects_tampered_existing_receipt(tmp_path: Path) -> None:
    destination = tmp_path / "source"
    destination.mkdir()
    (destination / "download_receipt.json").write_text(
        json.dumps(
            {
                "repository": source.REPOSITORY,
                "revision": "wrong",
                "license_declared_by_publisher": source.LICENSE,
                "files": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="do not match"):
        source.fetch(destination)
