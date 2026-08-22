from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scamguard.signals import extract_signal_matches
from scripts.build_schema17_call_minimal_pairs import build
from scripts.generate_call_minimal_pairs import (
    FORBIDDEN_SAFE_CUES,
    HOLDOUT_SCENARIOS,
    RISK_MECHANISMS,
    SOURCE,
    generate,
)
from scripts.generate_legitimate_call_openings import SCENARIOS, STRUCTURES


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def parent_row(identifier: str, label: str, split: str) -> dict[str, object]:
    return {
        "id": identifier,
        "text": f"fixture text for {identifier}",
        "label": label,
        "category": "NONE" if label == "SAFE" else "FINANCIAL",
        "source": "fixture",
        "source_label": "fixture",
        "license": "Apache-2.0",
        "split": split,
        "family_id": f"fixture-{identifier}",
        "is_synthetic": False,
    }


def test_generator_is_deterministic_balanced_and_minimally_paired() -> None:
    rows = generate()

    assert rows == generate()
    assert len(rows) == len(SCENARIOS) * len(STRUCTURES) * len(RISK_MECHANISMS) * 2
    assert Counter(str(row["label"]) for row in rows) == {"SAFE": 384, "SCAM": 384}
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)
    assert len(grouped) == 384
    for pair in grouped.values():
        assert {str(row["label"]) for row in pair} == {"SAFE", "SCAM"}
        assert len({str(row["shared_context_sha256"]) for row in pair}) == 1
        assert len({str(row["text"]).rsplit("\n", 1)[0] for row in pair}) == 1
        safe = next(row for row in pair if row["label"] == "SAFE")
        scam = next(row for row in pair if row["label"] == "SCAM")
        assert all(cue not in str(safe["text"]).casefold() for cue in FORBIDDEN_SAFE_CUES)
        assert extract_signal_matches(str(scam["text"]))
        assert safe["external_benchmark_text_copied"] is False


def test_schema17_holds_out_complete_scenarios_and_preserves_parent_bytes(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    output = tmp_path / "output"
    write_jsonl(parent / "train.jsonl", [parent_row("train", "SAFE", "train")])
    write_jsonl(parent / "dev.jsonl", [parent_row("dev", "SAFE", "dev")])
    write_jsonl(parent / "test.jsonl", [parent_row("test", "SCAM", "test")])
    (parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 14,
                "counts": {"train": 1, "dev": 1, "test": 1},
                "labels": {"SAFE": 2, "SCAM": 1},
                "sources": {"fixture": 3},
            }
        )
    )
    source_rows = generate()
    source_data = tmp_path / "pairs.jsonl"
    write_jsonl(source_data, source_rows)
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps({"source": SOURCE, "sha256": file_sha256(source_data)})
    )

    manifest = build(parent, source_data, source_manifest, output)

    expected_validation = len(HOLDOUT_SCENARIOS) * len(STRUCTURES) * len(RISK_MECHANISMS) * 2
    expected_train = len(source_rows) - expected_validation
    assert manifest["schema_version"] == 17
    assert manifest["schema17_increment"]["train_rows"] == expected_train
    assert manifest["schema17_increment"]["validation_rows"] == expected_validation
    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation_rows = [
        json.loads(line)
        for line in (output / "call_pair_validation.jsonl").read_text().splitlines()
    ]
    assert len(train_rows) == 1 + expected_train
    assert len(validation_rows) == expected_validation
    assert {row["scenario"] for row in validation_rows} == set(HOLDOUT_SCENARIOS)
    assert not {
        row.get("scenario") for row in train_rows if row.get("source") == SOURCE
    } & set(HOLDOUT_SCENARIOS)
    for filename in ("dev.jsonl", "test.jsonl"):
        assert (output / filename).read_bytes() == (parent / filename).read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(parent, source_data, source_manifest, output)
