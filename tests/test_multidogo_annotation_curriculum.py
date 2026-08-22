from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.build_multidogo_annotation_curriculum as curriculum
from scripts.audit_multidogo_annotations import TURN_HEADER


def write_turn_annotations(
    path: Path,
    conversation_id: str,
    intent: str,
    slots: str,
    turn: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "conversationId": conversation_id,
                "turnNumber": str(turn),
                "utteranceId": f"utterance-{turn}",
                "utterance": "Please review my card payment.",
                "slot-labels": slots,
                "intent": intent,
            }
        )


def write_annotation_tree(root: Path) -> None:
    for domain in curriculum.DOMAINS:
        for split in curriculum.ANNOTATION_SPLITS:
            write_turn_annotations(
                root
                / "data"
                / "paper_splits"
                / curriculum.TURN_GRANULARITY
                / domain
                / f"{split}.tsv",
                f"{domain}-{split}",
                "review_payments" if domain == "finance" else "request_help",
                "O O B-card B-payment" if domain == "finance" else "O O O O",
            )


def source_row(conversation_id: str, domain: str = "finance") -> dict[str, object]:
    return {
        "id": f"row-{conversation_id}",
        "text": "AGENT: I can help review the payment through the normal service channel.",
        "label": "SAFE",
        "category": "NONE",
        "source": curriculum.SOURCE,
        "source_domain": domain,
        "source_record_id": conversation_id,
        "source_window": "highest_risk_agent_turn",
        "family_id": f"multidogo:{conversation_id}",
    }


def test_label_tokens_normalize_plural_ontology_names() -> None:
    assert curriculum.label_tokens("review_payments_and_accounts") >= {
        "payment",
        "account",
    }


def test_annotation_index_preserves_splits_and_surfaces_sensitive_concepts(
    tmp_path: Path,
) -> None:
    write_annotation_tree(tmp_path)

    index = curriculum.build_annotation_index(tmp_path)

    finance = index[("finance", "finance-train")]
    assert finance["paper_split"] == "train"
    assert finance["sensitive_concepts"] == ["card", "payment"]
    assert finance["hard_negative_score"] == 2
    assert finance["annotated_customer_turns"] == 1


def test_annotation_index_rejects_conversation_crossing_paper_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(curriculum, "DOMAINS", ("finance",))
    for split in curriculum.ANNOTATION_SPLITS:
        write_turn_annotations(
            tmp_path
            / "data"
            / "paper_splits"
            / curriculum.TURN_GRANULARITY
            / "finance"
            / f"{split}.tsv",
            "same-conversation" if split != "test" else "test-conversation",
            "review_payment",
            "O B-payment",
            turn=0 if split == "train" else 2,
        )

    with pytest.raises(ValueError, match="crosses paper splits"):
        curriculum.build_annotation_index(tmp_path)


def test_enrichment_admits_only_requested_paper_split() -> None:
    index = {
        ("finance", "train-conversation"): {
            "paper_split": "train",
            "annotated_customer_turns": 2,
            "intents": ["review_payment"],
            "slot_types": ["card"],
            "sensitive_concepts": ["card", "payment"],
            "hard_negative_score": 2,
        },
        ("finance", "test-conversation"): {
            "paper_split": "test",
            "annotated_customer_turns": 1,
            "intents": ["review_payment"],
            "slot_types": [],
            "sensitive_concepts": ["payment"],
            "hard_negative_score": 1,
        },
    }
    rows = [
        source_row("train-conversation"),
        source_row("test-conversation"),
        source_row("unannotated-conversation"),
    ]

    enriched = curriculum.enrich_rows(rows, index, {"train"}, "a" * 64)

    assert [row["source_record_id"] for row in enriched] == ["train-conversation"]
    assert enriched[0]["annotation_stratum"] == "sensitive_service"
    assert enriched[0]["publisher_annotation_split"] == "train"
    assert "legitimate-domain weak label" in str(enriched[0]["annotation_label_scope"])


def test_audit_validation_rejects_tampered_annotation(tmp_path: Path) -> None:
    monkeypatch_paths = []
    for path in curriculum.annotation_paths(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
        monkeypatch_paths.append(path)
    report = {
        "artifact_schema_version": 1,
        "revision": curriculum.REVISION,
        "annotation_tree_git_oid": curriculum.ANNOTATION_TREE_GIT_OID,
        "contains_source_text": False,
        "alignment": {
            "all_annotations_join_to_pinned_unannotated_turns": True,
            "annotated_rows_are_customer_turns": True,
            "turn_text_matches_source": True,
            "sentence_text_occurs_in_source_turn": True,
            "conversation_crosses_paper_splits": False,
        },
        "files": {
            str(path.relative_to(tmp_path)): {
                "sha256": curriculum.file_sha256(path),
            }
            for path in monkeypatch_paths
        },
    }
    monkeypatch_paths[0].write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its audit"):
        curriculum.validate_audit_report(tmp_path, report)
