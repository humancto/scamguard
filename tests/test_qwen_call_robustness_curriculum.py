from __future__ import annotations

import json
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_qwen_call_robustness_curriculum import build
from training.build_qwen_sft import convert


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def raw_row(
    identifier: str,
    label: str,
    family: str,
    *,
    source: str = "fixture",
    text: str = "ordinary message",
    source_window: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": identifier,
        "label": label,
        "family_id": family,
        "source": source,
        "text": text,
    }
    if source_window is not None:
        row["source_window"] = source_window
    return row


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "parent"
    sft = parent / "qwen_sft"
    sft.mkdir(parents=True)
    train = [
        raw_row("uncertain", "UNCERTAIN", "family-u"),
        raw_row("safe", "SAFE", "family-s"),
        raw_row(
            "scam",
            "SCAM",
            "family-x",
            text="Pay with a gift card now and send the code",
        ),
        raw_row(
            "dialogue",
            "SAFE",
            "family-d",
            source="scamguard_synthetic_dialogue_v2",
        ),
    ]
    dev = [raw_row("dev", "SAFE", "family-dev")]
    write_jsonl(parent / "train.jsonl", train)
    write_jsonl(parent / "dev.jsonl", dev)
    parent_manifest = {
        "schema_version": 24,
        "release_eligible": False,
        "publication_authorized": False,
    }
    (parent / "manifest.json").write_text(json.dumps(parent_manifest), encoding="utf-8")
    write_jsonl(sft / "train.jsonl", [convert(row) for row in train])
    write_jsonl(sft / "dev.jsonl", [convert(row) for row in dev])
    (sft / "manifest.json").write_text(
        json.dumps({"input_manifest_sha256": file_sha256(parent / "manifest.json")}),
        encoding="utf-8",
    )

    multidogo = tmp_path / "multidogo"
    multidogo.mkdir()
    real_train = [
        raw_row(
            "md-turn",
            "SAFE",
            "md-train",
            source="multidogo_human_service_dialogues",
            source_window="highest_risk_agent_turn",
        ),
        raw_row(
            "md-call",
            "SAFE",
            "md-train",
            source="multidogo_human_service_dialogues",
            text="AGENT: How may I help?\nCUSTOMER: Please review my booking.",
            source_window="recent_complete_turns",
        ),
    ]
    held = [
        raw_row(
            "md-held",
            "SAFE",
            "md-held-family",
            source="multidogo_human_service_dialogues",
            source_window="recent_complete_turns",
        )
    ]
    real_path = multidogo / "multidogo_real_train.jsonl"
    held_path = multidogo / "multidogo_call_validation.jsonl"
    write_jsonl(real_path, real_train)
    write_jsonl(held_path, held)
    (multidogo / "manifest.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "license": "CDLA-Permissive-1.0",
                "artifacts": {
                    "real_train": {
                        "rows": len(real_train),
                        "sha256": file_sha256(real_path),
                    },
                    "call_validation": {
                        "rows": len(held),
                        "sha256": file_sha256(held_path),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return parent, multidogo


def test_builds_split_safe_replay_and_complete_call_curriculum(tmp_path: Path) -> None:
    parent, multidogo = fixture(tmp_path)
    output = tmp_path / "output"

    manifest = build(
        parent,
        multidogo,
        output,
        multidogo_repetitions=2,
        core_per_label=1,
    )

    train = [
        json.loads(line)
        for line in (output / "qwen_sft/train.jsonl").read_text().splitlines()
    ]
    assert manifest["multidogo"]["held_validation_rows_used_for_fitting"] == 0
    assert manifest["multidogo"]["family_cross_split"] is False
    assert sum(str(row["id"]).startswith("stage2-md-") for row in train) == 2
    assert "md-held" not in {row["id"] for row in train}
    assert manifest["splits"]["dev"]["byte_identical_to_parent"] is True
    assert json.loads((output / "qwen_sft/manifest.json").read_text())[
        "input_manifest_sha256"
    ] == file_sha256(output / "manifest.json")


def test_rejects_multidogo_family_crossing_validation(tmp_path: Path) -> None:
    parent, multidogo = fixture(tmp_path)
    held_path = multidogo / "multidogo_call_validation.jsonl"
    held = [
        raw_row(
            "md-held",
            "SAFE",
            "md-train",
            source="multidogo_human_service_dialogues",
            source_window="recent_complete_turns",
        )
    ]
    write_jsonl(held_path, held)
    manifest_path = multidogo / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["call_validation"] = {
        "rows": 1,
        "sha256": file_sha256(held_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="crosses training and validation"):
        build(parent, multidogo, tmp_path / "output")
