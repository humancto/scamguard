from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.build_schema15_legitimate_openings import build
from scripts.generate_legitimate_call_openings import (
    FORBIDDEN_SAFETY_CUES,
    SCENARIOS,
    SOURCE,
    STRUCTURES,
    generate,
)


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


def test_generator_is_balanced_deterministic_and_has_no_explicit_safety_cues() -> None:
    rows = generate(4)

    assert rows == generate(4)
    assert len(rows) == len(SCENARIOS) * len(STRUCTURES)
    assert Counter(str(row["scenario"]) for row in rows) == {
        scenario: 4 for scenario in SCENARIOS
    }
    assert Counter(str(row["dialogue_structure"]) for row in rows) == {
        structure: len(SCENARIOS) for structure in STRUCTURES
    }
    assert all(str(row["text"]).count("\n") == 3 for row in rows)
    assert all(
        cue not in str(row["text"]).casefold()
        for row in rows
        for cue in FORBIDDEN_SAFETY_CUES
    )
    assert all(row["external_benchmark_text_copied"] is False for row in rows)


def test_schema15_balances_groups_and_preserves_parent_evaluation_bytes(
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
    source_rows = generate(4)
    source_data = tmp_path / "legitimate.jsonl"
    write_jsonl(source_data, source_rows)
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps({"source": SOURCE, "sha256": file_sha256(source_data)})
    )

    manifest = build(
        parent,
        source_data,
        source_manifest,
        output,
        dose_per_scenario=4,
    )

    assert manifest["schema_version"] == 15
    assert manifest["schema15_increment"]["rows"] == len(SCENARIOS) * 4
    assert manifest["schema15_increment"]["apptek_rows_used_for_fitting"] == 0
    assert manifest["schema15_increment"]["apptek_ood_opened"] is False
    output_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(output_rows) == 1 + len(SCENARIOS) * 4
    assert Counter(
        row["scenario"] for row in output_rows if row.get("source") == SOURCE
    ) == {scenario: 4 for scenario in SCENARIOS}
    for filename in ("dev.jsonl", "test.jsonl"):
        assert (output / filename).read_bytes() == (parent / filename).read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(parent, source_data, source_manifest, output, dose_per_scenario=4)
