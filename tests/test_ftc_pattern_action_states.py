from __future__ import annotations

from collections import Counter, defaultdict

from scamguard.preprocessing import compact_dialogue_evidence_recent
from scripts.generate_ftc_pattern_action_states import (
    CONTRAST_STATES,
    FTC_WEBSITE_POLICY,
    METHOD,
    PATTERNS,
    TARGET_KEYS,
    VALIDATION_PATTERNS,
    generate,
    validate,
)
from scripts.validate_dataset import SYNTHETIC_METHODS, SYNTHETIC_REFERENCE_PREFIXES


def test_ftc_pattern_rows_are_balanced_and_scenario_disjoint() -> None:
    train, validation = generate()
    validate(train, validation)
    assert len(train) == 384
    assert len(validation) == 96
    assert {row["scenario"] for row in train} == set(PATTERNS) - set(VALIDATION_PATTERNS)
    assert {row["scenario"] for row in validation} == set(VALIDATION_PATTERNS)
    assert Counter(row["label"] for row in train) == {
        "SAFE": 192,
        "UNCERTAIN": 96,
        "SCAM": 96,
    }


def test_ftc_pattern_families_change_only_the_decisive_turn() -> None:
    train, validation = generate()
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in train + validation:
        grouped[str(row["contrast_id"])].append(row)
    for family in grouped.values():
        assert {row["contrast_state"] for row in family} == set(CONTRAST_STATES)
        lines = [str(row["text"]).splitlines() for row in family]
        changing = [
            index
            for index in range(len(lines[0]))
            if len({variant[index] for variant in lines}) > 1
        ]
        assert changing == [4]


def test_ftc_pattern_provenance_and_targets_are_explicit() -> None:
    train, validation = generate()
    assert METHOD in SYNTHETIC_METHODS
    for row in train + validation:
        assert row["synthetic_method"] == METHOD
        assert str(row["pattern_reference"]).startswith(SYNTHETIC_REFERENCE_PREFIXES)
        assert row["rights_reference"] == FTC_WEBSITE_POLICY
        assert row["external_transcript_text_copied"] is False
        assert tuple(row["action_targets"]) == TARGET_KEYS


def test_ftc_decisive_action_survives_evidence_compaction() -> None:
    train, validation = generate()
    for row in train + validation:
        decisive = str(row["text"]).splitlines()[4].partition(": ")[2]
        compacted = compact_dialogue_evidence_recent(str(row["text"]))
        assert decisive in compacted
