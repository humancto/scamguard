from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.audit_multidogo_annotations as audit_module
from scripts.audit_multidogo_annotations import (
    SENTENCE_HEADER,
    TURN_HEADER,
    alignment_failures,
    read_annotation_file,
    read_source_turn_index,
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


def test_alignment_is_role_and_text_fail_closed() -> None:
    rows = [
        {
            "domain": "finance",
            "conversation_id": "conversation-1",
            "turn_number": 2,
            "utterance": "Please review my card payment.",
        }
    ]
    aligned = {
        ("finance", "conversation-1", 2): {
            "role": "customer",
            "utterance": "Please review my card payment.",
        }
    }
    wrong_role = {
        ("finance", "conversation-1", 2): {
            "role": "agent",
            "utterance": "Please review my card payment.",
        }
    }

    assert not alignment_failures(rows, aligned, "splits_annotated_at_turn_level")
    assert alignment_failures(rows, wrong_role, "splits_annotated_at_turn_level") == {
        "annotated_non_customer_turn": 1
    }


def test_source_turn_index_accepts_upstream_csv_suffix(
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

    index = read_source_turn_index(tmp_path)

    assert index[("finance", "conversation-1", 2)]["role"] == "customer"


def test_slot_type_collapses_bio_prefixes() -> None:
    assert slot_type("O") is None
    assert slot_type("B-account_number") == "account_number"
    assert slot_type("I-account_number") == "account_number"
    assert slot_type("payment") == "payment"
