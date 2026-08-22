from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_multidogo_dialogues import LICENSE, SOURCE
from scripts.build_schema19_call_windows import read_jsonl
from scripts.build_schema24_annotated_hard_negatives import build
from scripts.fetch_multidogo import REVISION


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def base_row(identifier: str, family: str, text: str, split: str) -> dict[str, object]:
    return {
        "id": identifier,
        "text": text,
        "label": "SAFE",
        "category": "NONE",
        "source": SOURCE,
        "source_label": "legitimate_human_service_dialogue:finance",
        "license": LICENSE,
        "split": split,
        "family_id": family,
        "is_synthetic": False,
        "source_domain": "finance",
        "source_record_id": identifier,
        "source_window": "highest_risk_agent_turn",
    }


def annotated_row(
    identifier: str, family: str, text: str, paper_split: str
) -> dict[str, object]:
    source = base_row(
        identifier,
        family,
        text,
        "train" if paper_split == "train" else "validation",
    )
    return source | {
        "publisher_annotation_granularity": "turn",
        "publisher_annotation_split": paper_split,
        "annotation_audit_sha256": "a" * 64,
        "annotation_label_scope": (
            "publisher intent and slot labels; SAFE remains a legitimate-domain weak label"
        ),
        "annotation_stratum": "sensitive_service",
    }


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_train = [
        base_row("parent-train", "family-parent", "A routine parent message.", "train")
    ]
    write_jsonl(parent / "train.jsonl", parent_train)
    write_jsonl(
        parent / "dev.jsonl",
        [base_row("parent-dev", "family-dev-parent", "A development message.", "dev")],
    )
    write_jsonl(
        parent / "test.jsonl",
        [base_row("parent-test", "family-test-parent", "A regression message.", "test")],
    )
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 23,
                "counts": {"train": 1, "dev": 1, "test": 1},
                "labels": {"SAFE": 3},
                "sources": {SOURCE: 3},
            }
        ),
        encoding="utf-8",
    )

    curriculum = tmp_path / "curriculum"
    curriculum.mkdir()
    rows = {
        "train": [
            annotated_row(
                "new-train",
                "family-new",
                "The verified airline desk can review this ordinary itinerary tomorrow.",
                "train",
            ),
            annotated_row(
                "collision-train",
                "family-parent",
                "A separate view of the parent family.",
                "train",
            ),
        ],
        "dev": [
            annotated_row(
                "annotation-dev",
                "family-annotation-dev",
                "The normal insurer can review the policy next week.",
                "dev",
            )
        ],
        "test": [
            annotated_row(
                "annotation-test",
                "family-annotation-test",
                "The software vendor documented a normal subscription renewal.",
                "test",
            )
        ],
    }
    artifacts = {}
    for split, split_rows in rows.items():
        path = curriculum / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        artifacts[split] = {
            "path": str(path),
            "rows": len(split_rows),
            "sha256": file_sha256(path),
        }
    (curriculum / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "source": SOURCE,
                "license": LICENSE,
                "revision": REVISION,
                "policy": {
                    "publisher_annotations_are_not_independent_scam_labels": True,
                    "paper_dev_test_rows_enter_fitting": False,
                    "existing_source_train_validation_boundary_preserved": True,
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return parent, curriculum


def test_schema24_admits_only_novel_publisher_train_families(tmp_path: Path) -> None:
    parent, curriculum = fixture(tmp_path)
    output = tmp_path / "schema24"

    manifest = build(parent, curriculum, output)

    train = read_jsonl(output / "train.jsonl")
    assert manifest["schema_version"] == 24
    assert {row["id"] for row in train} == {"parent-train", "new-train"}
    assert next(row for row in train if row["id"] == "new-train")["schema24_admitted"] is True
    increment = manifest["schema24_increment"]
    assert increment["collision_families_removed"] == 1  # type: ignore[index]
    assert increment["paper_dev_test_rows_used_for_fitting"] is False  # type: ignore[index]
    assert len(read_jsonl(output / "multidogo_annotation_dev.jsonl")) == 1
    assert len(read_jsonl(output / "multidogo_annotation_test.jsonl")) == 1


def test_schema24_refuses_wrong_publisher_split(tmp_path: Path) -> None:
    parent, curriculum = fixture(tmp_path)
    train_path = curriculum / "train.jsonl"
    rows = read_jsonl(train_path)
    rows[0]["publisher_annotation_split"] = "test"
    write_jsonl(train_path, rows)
    manifest_path = curriculum / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["train"]["sha256"] = file_sha256(train_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid annotation curriculum row"):
        build(parent, curriculum, tmp_path / "schema24")
