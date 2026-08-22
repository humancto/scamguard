from __future__ import annotations

from pathlib import Path

import pytest

import scripts.build_multidogo_annotation_curriculum as curriculum


def annotation_row(
    *,
    conversation_id: str = "conversation-1",
    text: str = "Please transfer the payment through my normal bank account.",
    slots: tuple[str, ...] = ("O",),
    intents: tuple[str, ...] = ("transfermoney",),
    split: str = "train",
    domain: str = "finance",
) -> dict[str, object]:
    return {
        "domain": domain,
        "split": split,
        "conversation_id": conversation_id,
        "turn_number": 2,
        "sentence_number": None,
        "utterance_id": "utterance-2",
        "utterance": text,
        "intents": intents,
        "slot_labels": slots,
        "empty_utterance": not text,
    }


def test_label_tokens_find_concepts_inside_compound_ontology_names() -> None:
    assert curriculum.label_tokens("transfermoney account_number") >= {
        "transfer",
        "account",
    }


def test_candidate_uses_publisher_semantics_and_privacy_normalization() -> None:
    row = annotation_row(text="Please transfer 5000 tomorrow.", slots=("O", "O", "O"))

    candidate = curriculum.annotation_candidate(row, "a" * 64)

    assert candidate is not None
    assert candidate["text"] == "CUSTOMER: Please transfer <NUMBER> tomorrow."
    assert candidate["publisher_sensitive_concepts"] == ["transfer"]
    assert candidate["annotation_stratum"] == "sensitive_service"
    assert candidate["privacy_values_replaced"] is True


def test_candidate_excludes_rows_with_publisher_pii_slots() -> None:
    row = annotation_row(text="My name is Ana.", slots=("O", "O", "name", "name"))

    assert curriculum.annotation_candidate(row, "a" * 64) is None


def test_select_rows_reserves_held_text_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(curriculum, "DOMAINS", ("finance",))
    shared = curriculum.annotation_candidate(annotation_row(), "a" * 64)
    assert shared is not None
    train_unique = curriculum.annotation_candidate(
        annotation_row(conversation_id="train-unique", text="Review the normal balance tomorrow."),
        "a" * 64,
    )
    test_shared = dict(shared) | {
        "id": "test-shared",
        "family_id": "test-family",
        "publisher_annotation_split": "test",
        "split": "validation",
    }
    assert train_unique is not None
    pools = {
        "train": {"finance": [shared, train_unique]},
        "dev": {
            "finance": [
                dict(train_unique)
                | {
                    "id": "dev-unique",
                    "text": "CUSTOMER: The official account review is routine.",
                    "family_id": "dev-family",
                    "publisher_annotation_split": "dev",
                    "split": "validation",
                }
            ]
        },
        "test": {"finance": [test_shared]},
    }

    selected = curriculum.select_rows(pools, {"train": 1, "dev": 1, "test": 1})

    assert selected["test"][0]["id"] == "test-shared"
    assert selected["train"][0]["id"] == train_unique["id"]


def test_audit_validation_rejects_tampered_annotation(tmp_path: Path) -> None:
    paths = []
    for path in curriculum.annotation_paths(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
        paths.append(path)
    report = {
        "artifact_schema_version": 2,
        "revision": curriculum.REVISION,
        "annotation_tree_git_oid": curriculum.ANNOTATION_TREE_GIT_OID,
        "contains_source_text": False,
        "contracts": {
            "publisher_readme_describes_paper_splits_as_customer_turns": True,
            "annotation_identities_unique_within_granularity": True,
            "conversations_do_not_cross_splits_within_granularity": True,
            "empty_annotation_utterances_quarantined_from_curriculum": True,
            "turn_level_rows_are_selected_directly": True,
            "sentence_level_rows_are_audit_only": True,
            "annotated_and_unannotated_conversation_ids_are_separate_collections": True,
            "paper_dev_test_rows_enter_fitting": False,
        },
        "files": {
            str(path.relative_to(tmp_path)): {"sha256": curriculum.file_sha256(path)}
            for path in paths
        },
    }
    paths[0].write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its audit"):
        curriculum.validate_audit_report(tmp_path, report)
