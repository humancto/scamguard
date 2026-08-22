from __future__ import annotations

from scamguard.signals import extract_signal_matches
from scripts.build_harper_valley_calls import (
    HARMFUL_ENDINGS,
    HOLDOUT_TASKS,
    action_targets,
    render_context,
    state_targets,
)
from scripts.generate_call_action_states import TARGET_KEYS


def test_harmful_endings_are_extractive_and_action_complete() -> None:
    for task, ending in HARMFUL_ENDINGS.items():
        assert extract_signal_matches(ending), task
        targets = state_targets("harmful_scam", {}, task)
        assert tuple(targets) == TARGET_KEYS
        assert targets["requested_disclosure_or_transfer"] is True
        assert targets["caller_controls_target"] is True
        assert targets["pressure_or_secrecy"] is True


def test_task_disjoint_holdout_covers_sensitive_and_routine_service() -> None:
    assert HOLDOUT_TASKS == {"get branch hours", "replace card"}


def test_context_prefers_human_correction_and_preserves_roles() -> None:
    context, selected = render_context(
        [
            {
                "speaker_role": "agent",
                "human_transcript": "How can I help?",
                "transcript": "how kin eye help",
                "dialog_acts": [],
            },
            {
                "speaker_role": "caller",
                "human_transcript": "Please check my balance.",
                "transcript": "please cheque",
                "dialog_acts": [],
            },
        ]
    )

    assert context == "AGENT: How can I help?\nCUSTOMER: Please check my balance."
    assert len(selected) == 2


def test_real_call_weak_targets_use_source_task_and_dialog_acts() -> None:
    targets = action_targets(
        "check balance",
        [
            {
                "speaker_role": "agent",
                "dialog_acts": ["gridspace_data_question"],
            }
        ],
    )

    assert tuple(targets) == TARGET_KEYS
    assert targets["sensitive_action_language"] is True
    assert targets["requested_disclosure_or_transfer"] is True
    assert targets["caller_controls_target"] is False
