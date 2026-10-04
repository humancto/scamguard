from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_qwen_stage10_curriculum import (
    CONTRAST_LICENSE,
    CONTRAST_SOURCE,
    LONG_CALL_SOURCE,
    build,
    validate_contrasts,
)
from scripts.generate_government_evidence_contrasts import generate
from tests.test_qwen_ppone_curriculum import qwen_row, write_jsonl
from tests.test_qwen_stage9_curriculum import prepare_fixture


def add_long_call_replay(parent: Path) -> None:
    train_path = parent / "qwen_sft/train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    family = "synthetic:long_call_action_state:test-government-family:v1"
    rows.extend(
        [
            qwen_row(
                "long-government-safe",
                family,
                "SAFE",
                "End the call and use the official public directory",
                source=LONG_CALL_SOURCE,
            ),
            qwen_row(
                "long-government-uncertain",
                family,
                "UNCERTAIN",
                "A government record may need review",
                source=LONG_CALL_SOURCE,
                category="GOVERNMENT_LEGAL",
            ),
            qwen_row(
                "long-government-scam",
                family,
                "SCAM",
                "Pay now or legal action will begin",
                source=LONG_CALL_SOURCE,
                category="GOVERNMENT_LEGAL",
            ),
        ]
    )
    write_jsonl(train_path, rows)
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_manifest["splits"]["train"]["sha256"] = file_sha256(train_path)
    parent_manifest_path.write_text(json.dumps(parent_manifest), encoding="utf-8")
    (parent / "qwen_sft/manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent_manifest_path)}),
        encoding="utf-8",
    )


def contrast_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows_path = tmp_path / "government.jsonl"
    manifest_path = tmp_path / "government-manifest.json"
    rows = generate()
    write_jsonl(rows_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "source": CONTRAST_SOURCE,
                "license": CONTRAST_LICENSE,
                "used_for_fitting": True,
                "used_for_threshold": False,
                "held_rows_copied": 0,
                "evaluation_rows_read_by_generator": 0,
                "rows": len(rows),
                "families": len({str(row["family_id"]) for row in rows}),
                "labels": dict(
                    sorted(Counter(str(row["label"]) for row in rows).items())
                ),
                "sha256": file_sha256(rows_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, rows_path


def test_stage10_includes_matched_contrasts_and_preserves_boundaries(
    tmp_path: Path,
) -> None:
    base_inputs = prepare_fixture(tmp_path)
    add_long_call_replay(base_inputs[0])
    contrast_manifest, contrast_rows = contrast_fixture(tmp_path)
    output = tmp_path / "stage10"
    manifest = build(
        *base_inputs,
        contrast_manifest,
        contrast_rows,
        output,
        anchors_per_source_verdict=1,
        phone_scam_families_per_type=1,
        long_call_government_families=1,
    )

    rows = [
        json.loads(line)
        for line in (output / "qwen_sft/train.jsonl").read_text().splitlines()
    ]
    identifiers = {str(row["id"]) for row in rows}
    assert len([row for row in rows if row["source"] == CONTRAST_SOURCE]) == 96
    assert {
        "long-government-safe",
        "long-government-uncertain",
        "long-government-scam",
        "phone-refund-scam",
        "phone-ssn-scam",
        "phone-support-scam",
        "ppone-scam",
        "ppone-uncertain",
    } <= identifiers
    assert "phone-flagged-safe" not in identifiers
    assert manifest["selection"]["held_rows_used_for_fitting"] == 0
    assert manifest["selection"]["open_stage9_rows_copied_or_transformed"] == 0
    assert (output / "qwen_sft/dev.jsonl").read_bytes() == (
        base_inputs[0] / "qwen_sft/dev.jsonl"
    ).read_bytes()


def test_stage10_rejects_contrast_manifest_that_claims_held_rows(
    tmp_path: Path,
) -> None:
    manifest_path, rows_path = contrast_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["held_rows_copied"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen contract"):
        validate_contrasts(manifest_path, rows_path)
