from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.audit_multidogo_annotations as audit_module
from scripts.audit_multidogo_annotations import (
    SENTENCE_HEADER,
    TURN_HEADER,
    cross_granularity_stats,
    read_annotation_file,
    read_unannotated_conversation_ids,
    slot_type,
)
from scripts.fetch_multidogo import (
    ANNOTATION_GRANULARITIES,
    ANNOTATION_SPLITS,
    DOMAINS,
    annotation_paths,
)


def write_annotation(path: Path, header: list[str], row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def test_annotation_path_contract_is_complete() -> None:
    paths = annotation_paths(Path("fixture"))
    assert len(paths) == len(ANNOTATION_GRANULARITIES) * len(DOMAINS) * len(ANNOTATION_SPLITS)
    assert len(set(paths)) == len(paths)


def test_turn_annotation_parser_preserves_intents_and_slots(tmp_path: Path) -> None:
    source = tmp_path / "train.tsv"
    write_annotation(
        source,
        TURN_HEADER,
        {
            "conversationId": "conversation-1",
            "turnNumber": "2",
            "utteranceId": "utterance-2",
            "utterance": "Please review my card payment.",
            "slot-labels": "O O O B-payment I-payment",
            "intent": "review_payment<div>dispute_charge",
        },
    )

    rows = read_annotation_file(
        source, "splits_annotated_at_turn_level", "finance", "train"
    )

    assert rows[0]["intents"] == ("review_payment", "dispute_charge")
    assert rows[0]["slot_labels"] == ("O", "O", "O", "B-payment", "I-payment")
    assert rows[0]["sentence_number"] is None


def test_turn_annotation_parser_accepts_lossless_decimal_index(tmp_path: Path) -> None:
    source = tmp_path / "train.tsv"
    write_annotation(
        source,
        TURN_HEADER,
        {
            "conversationId": "conversation-1",
            "turnNumber": "2.0",
            "utteranceId": "utterance-2",
            "utterance": "Please review my card payment.",
            "slot-labels": "O O O B-payment I-payment",
            "intent": "review_payment",
        },
    )

    rows = read_annotation_file(
        source, "splits_annotated_at_turn_level", "finance", "train"
    )

    assert rows[0]["turn_number"] == 2


def test_turn_annotation_parser_rejects_fractional_index(tmp_path: Path) -> None:
    source = tmp_path / "train.tsv"
    write_annotation(
        source,
        TURN_HEADER,
        {
            "conversationId": "conversation-1",
            "turnNumber": "2.5",
            "utteranceId": "utterance-2",
            "utterance": "Please review my card payment.",
            "slot-labels": "O O O B-payment I-payment",
            "intent": "review_payment",
        },
    )

    with pytest.raises(ValueError, match="non-integer"):
        read_annotation_file(
            source, "splits_annotated_at_turn_level", "finance", "train"
        )


def test_sentence_annotation_requires_sentence_index(tmp_path: Path) -> None:
    source = tmp_path / "dev.tsv"
    write_annotation(
        source,
        SENTENCE_HEADER,
        {
            "conversationId": "conversation-2",
            "turnNumber": "4",
            "sentenceNumber": "not-an-integer",
            "utteranceId": "utterance-4",
            "utterance": "I need help.",
            "slot-labels": "O O O",
            "intent": "request_help",
        },
    )

    with pytest.raises(ValueError, match="non-integer"):
        read_annotation_file(
            source, "splits_annotated_at_sentence_level", "software", "dev"
        )


def test_empty_publisher_utterance_is_retained_for_quarantine(tmp_path: Path) -> None:
    source = tmp_path / "train.tsv"
    write_annotation(
        source,
        SENTENCE_HEADER,
        {
            "conversationId": "conversation-empty",
            "turnNumber": "4",
            "sentenceNumber": "0",
            "utteranceId": "utterance-empty",
            "utterance": "",
            "slot-labels": "O",
            "intent": "contentonly",
        },
    )

    rows = read_annotation_file(
        source, "splits_annotated_at_sentence_level", "software", "train"
    )

    assert rows[0]["empty_utterance"] is True
    assert rows[0]["utterance"] == ""


def test_cross_granularity_stats_report_publisher_divergence() -> None:
    key = ("finance", "conversation-1", 2)
    turn_rows = {key: {"utterance": "Please review my card payment."}}
    sentence_rows = {
        key: [
            {"utterance": "Please review my card payment.", "empty_utterance": False},
            {"utterance": "Extra publisher text.", "empty_utterance": False},
        ]
    }

    stats = cross_granularity_stats(turn_rows, sentence_rows)

    assert stats["common_turn_keys"] == 1
    assert stats["common_keys_with_publisher_text_divergence"] == 1


def test_unannotated_id_reader_accepts_upstream_csv_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_module, "DOMAINS", ("finance",))
    source_dir = tmp_path / "data" / "unannotated"
    source_dir.mkdir(parents=True)
    source = source_dir / "finance.tsv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversationId",
                "turnNumber",
                "utteranceId",
                "utterance",
                "authorRole",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "conversationId": "conversation-1",
                "turnNumber": "2",
                "utteranceId": "utterance-2",
                "utterance": "Please review my card payment.",
                "authorRole": "customer",
            }
        )

    identifiers = read_unannotated_conversation_ids(tmp_path)

    assert identifiers == {("finance", "conversation-1")}


def test_slot_type_collapses_bio_prefixes() -> None:
    assert slot_type("O") is None
    assert slot_type("B-account_number") == "account_number"
    assert slot_type("I-account_number") == "account_number"
    assert slot_type("payment") == "payment"
