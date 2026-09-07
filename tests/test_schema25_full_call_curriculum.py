from __future__ import annotations

from scripts.build_multidogo_dialogues import LICENSE as MULTIDOGO_LICENSE
from scripts.build_multidogo_dialogues import SOURCE as MULTIDOGO_SOURCE
from scripts.build_phone_scam_synthetic import SOURCE as PHONE_SOURCE
from scripts.build_schema25_full_call_curriculum import full_call_row, phone_row
from scripts.fetch_multidogo import REVISION as MULTIDOGO_REVISION
from scripts.fetch_phone_scam_synthetic import LICENSE as PHONE_LICENSE
from scripts.fetch_phone_scam_synthetic import REVISION as PHONE_REVISION


def test_full_call_removes_weak_action_supervision() -> None:
    row = {
        "id": "call-1",
        "family_id": "family-1",
        "text": "CUSTOMER: I need help.\nAGENT: I can help through the official app.",
        "source": MULTIDOGO_SOURCE,
        "license": MULTIDOGO_LICENSE,
        "source_revision": MULTIDOGO_REVISION,
        "source_window": "recent_complete_turns",
        "label": "SAFE",
        "is_synthetic": False,
        "split": "train",
        "action_targets": {"requested_disclosure_or_transfer": True},
        "action_label_method": "weak",
        "action_verdict_weight": 0.25,
    }

    shaped = full_call_row(row)

    assert "action_targets" not in shaped
    assert "action_label_method" not in shaped
    assert "action_verdict_weight" not in shaped
    assert shaped["schema25_action_supervision"] == "removed_weak_heuristic_targets"
    assert shaped["text"] == row["text"]


def test_phone_row_marks_validation_as_non_sota_diagnostic() -> None:
    row = {
        "id": "phone-1",
        "family_id": "family-2",
        "text": "caller: Please use the official number. receiver: Okay.",
        "source": PHONE_SOURCE,
        "license": PHONE_LICENSE,
        "source_revision": PHONE_REVISION,
        "is_synthetic": True,
        "metadata_used_as_model_input": False,
        "split": "validation",
    }

    shaped = phone_row(row, "validation")

    assert shaped["schema25_role"] == "confounded_development_diagnostic_only"
    assert shaped["schema25_admitted"] is True
