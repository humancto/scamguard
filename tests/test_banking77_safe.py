from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_banking77_safe as banking77


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=banking77.EXPECTED_HEADER)
        writer.writeheader()
        for text, category in rows:
            writer.writerow({"text": text, "category": category})


def install_fixture_sources(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    categories = ["cancel_transfer", "compromised_card", "verify_my_identity"]
    train_rows = [
        ("Please cancel the transfer I made yesterday", categories[0]),
        ("I need to stop a bank transfer", categories[0]),
        ("Please cancel the transfer I made yesterday", categories[0]),
        ("Someone used my card at 212-555-0100", categories[1]),
        ("I do not recognize this card payment", categories[1]),
        ("How can I verify my identity", categories[2]),
        ("Where do I upload identity documents", categories[2]),
    ]
    test_rows = [
        ("Can a transfer be reversed", categories[0]),
        ("My card details may have been stolen", categories[1]),
        ("What documents prove my identity", categories[2]),
    ]
    write_csv(source / "train.csv", train_rows)
    write_csv(source / "test.csv", test_rows)
    (source / "categories.json").write_text(json.dumps(categories), encoding="utf-8")
    (source / "LICENSE").write_text("fixture CC-BY-4.0", encoding="utf-8")
    files = tuple(
        banking77.SourceFile(
            filename,
            f"https://example.invalid/{filename}",
            hashlib.sha256((source / filename).read_bytes()).hexdigest(),
        )
        for filename in ("train.csv", "test.csv", "categories.json", "LICENSE")
    )
    monkeypatch.setattr(banking77, "SOURCE_FILES", files)
    monkeypatch.setattr(banking77, "EXPECTED_ROWS", {"train": 7, "test": 3})
    monkeypatch.setattr(banking77, "EXPECTED_CATEGORIES", 3)
    return [text for text, _ in train_rows + test_rows]


def test_build_preserves_sealed_test_and_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    messages = install_fixture_sources(source, monkeypatch)
    references = tmp_path / "references"
    references.mkdir()
    output = tmp_path / "output"
    report_path = tmp_path / "report.json"

    report = banking77.build(
        source,
        output,
        report_path,
        references=(references,),
        selection_per_intent=1,
    )

    assert report["policy"]["fit_artifact_used_for_fitting"] is False
    assert report["policy"]["official_test_predictions_opened"] is False
    assert report["policy"]["individual_safe_ground_truth_claimed"] is False
    assert report["counts"]["train"]["exact_duplicate_rows_removed"] == 1
    assert report["counts"]["train"]["selection_rows"] == 3
    assert report["counts"]["test"]["sealed_rows"] == 3
    serialized = report_path.read_text(encoding="utf-8")
    assert all(message not in serialized for message in messages)
    validation = [
        json.loads(line)
        for line in (output / "banking77_validation.jsonl").read_text().splitlines()
    ]
    sealed = [
        json.loads(line)
        for line in (output / "banking77_test_sealed.jsonl").read_text().splitlines()
    ]
    assert {row["source_intent"] for row in validation} == {
        "cancel_transfer",
        "compromised_card",
        "verify_my_identity",
    }
    assert all(row["label"] == "SAFE" for row in validation + sealed)
    assert all("555" not in row["text"] for row in validation)
    assert not ({row["family_id"] for row in validation} & {row["family_id"] for row in sealed})


def test_rejects_unpinned_or_incomplete_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    install_fixture_sources(source, monkeypatch)
    (source / "train.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned revision"):
        banking77.verify_sources(source)

    source = tmp_path / "incomplete"
    source.mkdir()
    install_fixture_sources(source, monkeypatch)
    monkeypatch.setattr(banking77, "EXPECTED_ROWS", {"train": 8, "test": 3})
    with pytest.raises(ValueError, match="expected 8"):
        banking77.read_split(
            source / "train.csv",
            official_split="train",
            categories={"cancel_transfer", "compromised_card", "verify_my_identity"},
        )
