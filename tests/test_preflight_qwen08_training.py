from __future__ import annotations

import json
from pathlib import Path

from scamguard.metrics import file_sha256
from scripts.preflight_qwen08_training import manifest_sha256, snapshot_manifest


def test_snapshot_manifest_is_sorted_and_content_bound(tmp_path: Path) -> None:
    (tmp_path / "z.json").write_text("z", encoding="utf-8")
    (tmp_path / "a.json").write_text("a", encoding="utf-8")

    records = snapshot_manifest(tmp_path)

    assert [record["name"] for record in records] == ["a.json", "z.json"]
    assert all(record["bytes"] == 1 for record in records)
    assert len(manifest_sha256(records)) == 64


def test_snapshot_manifest_changes_with_file_content(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"first")
    before = manifest_sha256(snapshot_manifest(tmp_path))
    artifact.write_bytes(b"second")

    after = manifest_sha256(snapshot_manifest(tmp_path))

    assert before != after


def test_tracked_preflight_is_bound_to_current_launch_sources() -> None:
    repository = Path(__file__).resolve().parents[1]
    report = json.loads(
        (repository / "reports" / "QWEN08_TRAINING_PREFLIGHT.json").read_text(
            encoding="utf-8"
        )
    )
    bindings = report["source_bindings"]

    assert report["passed"] is True
    assert report["parameter_update_performed"] is False
    assert report["contains_training_or_audit_rows"] is False
    assert report["base_model_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert report["transformers_revision"] == "0c92811846095910816a87aca50050d10c545270"
    assert bindings == {
        "preflight_script_sha256": file_sha256(
            repository / "scripts" / "preflight_qwen08_training.py"
        ),
        "training_launcher_sha256": file_sha256(
            repository / "training" / "train_qwen_lora.py"
        ),
        "pyproject_sha256": file_sha256(repository / "pyproject.toml"),
        "uv_lock_sha256": file_sha256(repository / "uv.lock"),
    }
