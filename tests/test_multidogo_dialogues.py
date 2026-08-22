from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_multidogo_dialogues import (
    EXPECTED_HEADER,
    assign_unique_agent_turns,
    base_action_targets,
    read_domain,
    state_rows,
)


def write_dialogue(path: Path, conversation_id: str = "conversation-1") -> None:
    utterances = (
        ("customer", "Hello, I need help reviewing my account payment."),
        ("agent", "I can help with the payment and account details."),
        ("customer", "The charge appears twice on my statement."),
        ("agent", "Could you please provide the order number from the statement?"),
        ("customer", "Yes, I have the order number in front of me."),
        ("agent", "Thank you, I can review it without asking for your password."),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_HEADER)
        writer.writeheader()
        for index, (role, text) in enumerate(utterances):
            writer.writerow(
                {
                    "conversationId": conversation_id,
                    "turnNumber": index,
                    "utteranceId": f"utterance-{index}",
                    "utterance": text,
                    "authorRole": role,
                }
            )


def source_row(identifier: str, agent_text: str) -> dict[str, object]:
    return {
        "source_record_id": identifier,
        "source_domain": "finance",
        "family_id": f"near-{identifier}",
        "privacy_values_replaced": False,
        "turns": [
            {"role": "customer", "text": "I need help with a charge.", "turn": 0},
            {"role": "agent", "text": agent_text, "turn": 1},
            {
                "role": "agent",
                "text": f"A unique alternative response for {identifier} is available.",
                "turn": 2,
            },
            {"role": "customer", "text": "Please continue with the review.", "turn": 3},
        ],
        "selected_agent_index": 1,
        "selected_agent_text": agent_text,
    }


def test_read_domain_preserves_complete_human_roles(tmp_path: Path) -> None:
    source = tmp_path / "finance.tsv"
    write_dialogue(source)
    dialogues = read_domain(source, "finance")
    assert len(dialogues) == 1
    assert dialogues[0]["domain"] == "finance"
    assert [turn["role"] for turn in dialogues[0]["turns"]] == [
        "customer",
        "agent",
        "customer",
        "agent",
        "customer",
        "agent",
    ]


def test_unique_agent_turn_assignment_uses_an_alternative() -> None:
    repeated = "Please provide the payment account number for review."
    first = source_row("first", repeated)
    second = source_row("second", repeated)
    assign_unique_agent_turns([[first], [second]])
    assert first["selected_agent_text"] != second["selected_agent_text"]


def test_action_targets_distinguish_legitimate_request_from_caller_control() -> None:
    targets = base_action_targets(
        "Could you please provide the order number from your account statement?"
    )
    assert targets["sensitive_action_language"] is True
    assert targets["requested_disclosure_or_transfer"] is True
    assert targets["caller_controls_target"] is False
    assert targets["pressure_or_secrecy"] is False


def test_state_rows_keep_harmful_action_before_shared_delay() -> None:
    source = source_row("state", "Please provide the order number for this payment.")
    rows = state_rows([source], "train", ("finance",), 1)
    assert len(rows) == 4
    harmful = next(row for row in rows if row["contrast_state"] == "harmful_scam")
    assert harmful["label"] == "SCAM"
    assert harmful["decisive_action_precedes_shared_continuation"] is True
    assert str(harmful["text"]).index("safe account") < str(harmful["text"]).index(
        "pause and check"
    )
    assert len({row["shared_context_sha256"] for row in rows}) == 1
