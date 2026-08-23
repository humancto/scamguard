from __future__ import annotations

import hashlib

import pytest

from scripts.fetch_qwen08_base_gguf import verify_artifact


def test_verify_artifact_requires_exact_size_and_hash(tmp_path) -> None:
    path = tmp_path / "control.gguf"
    path.write_bytes(b"verified-control")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    verify_artifact(path, len(b"verified-control"), digest)
    with pytest.raises(ValueError, match="byte count"):
        verify_artifact(path, 1, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(path, len(b"verified-control"), "0" * 64)
