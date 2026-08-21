from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.audit_teleantifraud import audit


def _record(index: int) -> dict[str, object]:
    return {
        "prompt": [
            {"role": "system", "content": "Classify the call."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": f"audio/call-{index}.wav"},
                    {"type": "text", "text": f"Unique transcript {index}"},
                ],
            },
        ],
        "answer": "fraud" if index % 2 else "normal",
    }


def _write_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("binary/train.json", json.dumps([_record(i) for i in range(4000)]))
        archive.writestr(
            "binary/test.json",
            json.dumps([_record(10_000 + i) for i in range(400)]),
        )


def test_teleantifraud_audit_is_text_free_and_receipt_bound(tmp_path: Path) -> None:
    source = tmp_path / "binary_classification.zip"
    _write_archive(source)
    receipt = tmp_path / "download_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "files": {
                    source.name: {
                        "sha256": file_sha256(source),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = audit(source, receipt)

    assert report["aggregate"]["rows"] == 4400
    assert report["aggregate"]["labels"] == {"normal": 2200, "fraud": 2200}
    assert report["source"]["receipt_verified"] is True
    assert report["admission"]["train_rows_admitted"] == 0
    assert "Unique transcript" not in json.dumps(report)


def test_teleantifraud_audit_rejects_archive_traversal(tmp_path: Path) -> None:
    source = tmp_path / "binary_classification.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../train.json", "[]")
        archive.writestr("test.json", "[]")

    with pytest.raises(ValueError, match="unsafe archive member"):
        audit(source)
