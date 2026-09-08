from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_qwen_phone_curriculum import (
    PHONE_LICENSE,
    PHONE_REPOSITORY,
    PHONE_REVISION,
    PHONE_SOURCE,
)
from scripts.build_qwen_ppone_curriculum import (
    PPONE_REPOSITORY,
    PPONE_REVISION,
    PPONE_SOURCE,
    build,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def qwen_row(
    identifier: str,
    family: str,
    label: str,
    text: str,
    *,
    source: str = "parent-source",
    category: str | None = None,
) -> dict[str, object]:
    resolved_category = category or ("NONE" if label == "SAFE" else "OTHER_SCAM")
    return {
        "id": identifier,
        "family_id": family,
        "source": source,
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{text}</message>",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "verdict": label,
                        "category": resolved_category,
                        "signals": [],
                        "evidence": [],
                        "recommended_action": "NO_ACTION",
                    }
                ),
            },
        ],
    }


def raw_row(
    identifier: str,
    family: str,
    label: str,
    text: str,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "id": identifier,
        "family_id": family,
        "source": source,
        "source_label": "fixture",
        "license": "Public-Domain" if source == PPONE_SOURCE else "MIT",
        "label": label,
        "category": "UNKNOWN" if label != "SAFE" else "NONE",
        "text": text,
        "split": "train",
        "is_synthetic": source == PHONE_SOURCE,
    }


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    parent = tmp_path / "parent"
    parent_sft = parent / "qwen_sft"
    parent_sft.mkdir(parents=True)
    parent_train = [
        qwen_row("safe", "safe-family", "SAFE", "Family dinner is at six"),
        qwen_row(
            "credential",
            "credential-family",
            "SCAM",
            "Send your password now",
            category="CREDENTIAL_MFA",
        ),
        qwen_row(
            "job",
            "job-family",
            "SCAM",
            "Pay a deposit for the job",
            category="JOB_OPPORTUNITY",
        ),
        qwen_row(
            "uncertain", "uncertain-family", "UNCERTAIN", "Recorded account notice"
        ),
    ]
    parent_dev = [qwen_row("dev", "dev-family", "SAFE", "Dentist reminder tomorrow")]
    write_jsonl(parent_sft / "train.jsonl", parent_train)
    write_jsonl(parent_sft / "dev.jsonl", parent_dev)
    parent_manifest = {
        "experiment_kind": "qwen_phone_generalization_stage7_curriculum",
        "release_eligible": False,
        "publication_authorized": False,
        "splits": {
            "train": {"sha256": file_sha256(parent_sft / "train.jsonl")},
            "dev": {"sha256": file_sha256(parent_sft / "dev.jsonl")},
        },
    }
    (parent / "manifest.json").write_text(json.dumps(parent_manifest), encoding="utf-8")
    (parent_sft / "manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent / "manifest.json")}),
        encoding="utf-8",
    )

    ppone_train = tmp_path / "ppone-train.jsonl"
    ppone_rows = [
        raw_row(
            "ppone-scam",
            "ppone-scam-family",
            "SCAM",
            "A court warrant requires immediate payment",
            source=PPONE_SOURCE,
        )
        | {"individual_scam_ground_truth": False},
        raw_row(
            "ppone-uncertain",
            "ppone-uncertain-family",
            "UNCERTAIN",
            "This is a recorded promotion for home services",
            source=PPONE_SOURCE,
        )
        | {"individual_scam_ground_truth": False},
    ]
    write_jsonl(ppone_train, ppone_rows)
    ppone_manifest = tmp_path / "ppone-manifest.json"
    ppone_manifest.write_text(
        json.dumps(
            {
                "source": {
                    "repository": PPONE_REPOSITORY,
                    "revision": PPONE_REVISION,
                    "rights": "Data is public domain",
                },
                "policy": {
                    "positive_or_suspicious_source_only": True,
                    "individual_scam_ground_truth": False,
                    "safe_rows_supplied_by_this_source": 0,
                    "validation_used_for_fitting_or_threshold": False,
                    "test_prediction_sealed_until_candidate_freeze": True,
                    "independent_human_label_review_complete": False,
                },
                "artifacts": {
                    "train": {"rows": 2, "sha256": file_sha256(ppone_train)}
                },
            }
        ),
        encoding="utf-8",
    )

    phone_train = tmp_path / "phone-train.jsonl"
    phone_rows = [
        raw_row(
            "phone-safe",
            "phone-safe-family",
            "SAFE",
            "The refund is visible in your official account",
            source=PHONE_SOURCE,
        )
        | {"source_dialogue_type": "refund", "source_length_category": "long"},
        raw_row(
            "phone-scam",
            "phone-scam-family",
            "SCAM",
            "Return the refund with gift cards immediately",
            source=PHONE_SOURCE,
        )
        | {"source_dialogue_type": "refund", "source_length_category": "long"},
    ]
    write_jsonl(phone_train, phone_rows)
    phone_manifest = tmp_path / "phone-manifest.json"
    phone_manifest.write_text(
        json.dumps(
            {
                "source": PHONE_SOURCE,
                "repository": PHONE_REPOSITORY,
                "revision": PHONE_REVISION,
                "license_declared_by_publisher": PHONE_LICENSE,
                "artifacts": {
                    "train": {"rows": 2, "sha256": file_sha256(phone_train)}
                },
                "policy": {
                    "synthetic": True,
                    "direct_reddit_scrape": False,
                    "source_metadata_used_as_model_input": False,
                    "test_prediction_sealed_until_candidate_freeze": True,
                    "variation_style_is_perfectly_label_confounded": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return parent, ppone_manifest, ppone_train, phone_manifest, phone_train


def test_build_uses_only_ppone_train_and_preserves_parent_dev(tmp_path: Path) -> None:
    parent, ppone_manifest, ppone_train, phone_manifest, phone_train = fixture(tmp_path)
    output = tmp_path / "output"
    manifest = build(
        parent,
        ppone_manifest,
        ppone_train,
        phone_manifest,
        phone_train,
        output,
        anchors_per_source_verdict=1,
        critical_category_families=1,
    )

    rows = read_rows(output / "qwen_sft/train.jsonl")
    ids = {row["id"] for row in rows}
    assert {"ppone-scam", "ppone-uncertain", "phone-safe", "phone-scam"} <= ids
    assert manifest["selection"]["ppone_validation_rows_read"] == 0
    assert manifest["selection"]["ppone_test_rows_read"] == 0
    assert manifest["selection"]["held_rows_used_for_fitting"] == 0
    assert manifest["parent"]["full_replay_rows"] == 0
    assert (output / "qwen_sft/dev.jsonl").read_bytes() == (
        parent / "qwen_sft/dev.jsonl"
    ).read_bytes()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_rejects_drifted_ppone_rights_contract(tmp_path: Path) -> None:
    parent, ppone_manifest, ppone_train, phone_manifest, phone_train = fixture(tmp_path)
    manifest = json.loads(ppone_manifest.read_text())
    manifest["source"]["rights"] = "unknown"
    ppone_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="PPoNE source differs"):
        build(
            parent,
            ppone_manifest,
            ppone_train,
            phone_manifest,
            phone_train,
            tmp_path / "output",
        )


def test_build_rejects_ppone_overlap_with_dev(tmp_path: Path) -> None:
    parent, ppone_manifest, ppone_train, phone_manifest, phone_train = fixture(tmp_path)
    dev = parent / "qwen_sft/dev.jsonl"
    rows = read_rows(dev)
    rows[0]["messages"][1]["content"] = (
        "Classify this message:\n<message>This is a recorded promotion for home services</message>"
    )
    write_jsonl(dev, rows)
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    parent_manifest["splits"]["dev"]["sha256"] = file_sha256(dev)
    (parent / "manifest.json").write_text(json.dumps(parent_manifest), encoding="utf-8")
    (parent / "qwen_sft/manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent / "manifest.json")}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="near-overlap"):
        build(
            parent,
            ppone_manifest,
            ppone_train,
            phone_manifest,
            phone_train,
            tmp_path / "output",
        )
