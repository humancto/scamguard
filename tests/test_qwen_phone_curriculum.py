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
    build,
)


def qwen_row(identifier: str, family: str, label: str, text: str) -> dict[str, object]:
    target = {
        "verdict": label,
        "category": "NONE" if label == "SAFE" else "FINANCIAL_IMPERSONATION",
        "signals": [] if label == "SAFE" else ["URGENCY_PRESSURE"],
        "evidence": [] if label == "SAFE" else ["urgent"],
        "recommended_action": "NO_ACTION" if label == "SAFE" else "VERIFY_OFFICIAL_CHANNEL",
    }
    return {
        "id": identifier,
        "family_id": family,
        "source": "parent",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Classify this message:\n<message>{text}</message>"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
    }


def raw_row(identifier: str, family: str, label: str, text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "family_id": family,
        "source": PHONE_SOURCE,
        "source_label": label,
        "label": label,
        "text": text,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent = tmp_path / "parent"
    sft = parent / "qwen_sft"
    sft.mkdir(parents=True)
    write_jsonl(sft / "train.jsonl", [qwen_row("parent", "p-family", "SAFE", "Dinner at six")])
    write_jsonl(sft / "dev.jsonl", [qwen_row("dev", "d-family", "SCAM", "urgent bank call")])
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_kind": "qwen_boundary_recovery_stage3_curriculum",
                "release_eligible": False,
                "publication_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (sft / "manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent / "manifest.json")}),
        encoding="utf-8",
    )
    phone_train = tmp_path / "phone-train.jsonl"
    write_jsonl(
        phone_train,
        [
            raw_row("phone-safe", "s-family", "SAFE", "Your appointment is tomorrow"),
            raw_row("phone-scam", "x-family", "SCAM", "Urgent transfer required now"),
        ],
    )
    phone_manifest = tmp_path / "phone-manifest.json"
    phone_manifest.write_text(
        json.dumps(
            {
                "source": PHONE_SOURCE,
                "repository": PHONE_REPOSITORY,
                "revision": PHONE_REVISION,
                "license_declared_by_publisher": PHONE_LICENSE,
                "artifacts": {
                    "train": {
                        "rows": 2,
                        "sha256": file_sha256(phone_train),
                    }
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
    return parent, phone_manifest, phone_train


def test_build_replays_parent_and_uses_only_phone_train(tmp_path: Path) -> None:
    parent, phone_manifest, phone_train = fixture(tmp_path)
    output = tmp_path / "output"
    manifest = build(parent, phone_manifest, phone_train, output)

    rows = [json.loads(line) for line in (output / "qwen_sft/train.jsonl").read_text().splitlines()]
    assert {row["id"] for row in rows} == {"parent", "phone-safe", "phone-scam"}
    assert manifest["phone_supplement"]["converted_rows"] == 2
    assert manifest["phone_supplement"]["test_rows_read"] == 0
    assert manifest["selection"]["held_rows_used_for_fitting"] == 0
    assert (output / "qwen_sft/dev.jsonl").read_bytes() == (
        parent / "qwen_sft/dev.jsonl"
    ).read_bytes()


def test_build_rejects_drifted_phone_revision(tmp_path: Path) -> None:
    parent, phone_manifest, phone_train = fixture(tmp_path)
    manifest = json.loads(phone_manifest.read_text())
    manifest["revision"] = "drifted"
    phone_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned rights"):
        build(parent, phone_manifest, phone_train, tmp_path / "output")


def test_build_rejects_near_overlap_with_held_reference(tmp_path: Path) -> None:
    parent, phone_manifest, phone_train = fixture(tmp_path)
    held = tmp_path / "held.jsonl"
    write_jsonl(held, [{"text": "Urgent transfer required now"}])

    with pytest.raises(ValueError, match="held evaluation"):
        build(
            parent,
            phone_manifest,
            phone_train,
            tmp_path / "output",
            overlap_references=(held,),
        )


def test_build_excludes_rows_over_the_frozen_token_budget(tmp_path: Path) -> None:
    parent, phone_manifest, phone_train = fixture(tmp_path)

    class FakeTokenizer:
        tokenizer = None

        def __init__(self) -> None:
            self.tokenizer = self

        def apply_chat_template(self, messages: object, **_: object) -> str:
            return str(messages)

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            size = 999 if "Urgent transfer required now" in text else 20
            return {"input_ids": list(range(size))}

    output = tmp_path / "output"
    manifest = build(
        parent,
        phone_manifest,
        phone_train,
        output,
        tokenizer=FakeTokenizer(),
        max_length=640,
    )

    assert manifest["phone_supplement"]["converted_rows"] == 1
    assert manifest["phone_supplement"]["token_length_filter"] == {
        "enforced": True,
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "max_length": 640,
        "over_length_rows_excluded": 1,
        "over_length_ids_sha256": (
            "2b3bcd231d6d21eba36050312666b8bf961db5979b9202b3fbe1158147431475"
        ),
    }
