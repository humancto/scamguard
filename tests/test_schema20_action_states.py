from __future__ import annotations

from collections import Counter, defaultdict

from scripts import build_schema20_action_states as schema20
from scripts.generate_call_action_states import (
    CONTRAST_STATES,
    HOLDOUT_SCENARIOS,
    RISK_MECHANISMS,
    TARGET_KEYS,
    generate,
)
from scripts.generate_call_evidence_pairs import CONTEXT_FRAMES
from scripts.generate_legitimate_call_openings import SCENARIOS, STRUCTURES


def test_action_state_generator_is_deterministic_and_context_matched() -> None:
    rows = generate()

    assert rows == generate()
    families = (
        len(SCENARIOS)
        * len(STRUCTURES)
        * len(CONTEXT_FRAMES)
        * len(RISK_MECHANISMS)
    )
    assert len(rows) == families * len(CONTRAST_STATES)
    assert Counter(str(row["label"]) for row in rows) == {
        "SAFE": families * 2,
        "UNCERTAIN": families,
        "SCAM": families,
    }
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["contrast_id"])].append(row)
    assert len(grouped) == families
    for contrast in grouped.values():
        assert {str(row["contrast_state"]) for row in contrast} == set(CONTRAST_STATES)
        assert len({str(row["shared_context_sha256"]) for row in contrast}) == 1
        assert len({str(row["text"]).rsplit("\n", 1)[0] for row in contrast}) == 1
        assert all(tuple(row["action_targets"]) == TARGET_KEYS for row in contrast)
        verified = next(
            row for row in contrast if row["contrast_state"] == "verified_safe"
        )
        harmful = next(
            row for row in contrast if row["contrast_state"] == "harmful_scam"
        )
        assert verified["label"] == "SAFE"
        assert verified["action_targets"]["independent_verification"] is True
        assert harmful["label"] == "SCAM"
        assert harmful["action_targets"]["caller_controls_target"] is True


def test_long_action_states_preserve_four_way_contrast_and_holdout_contract() -> None:
    rows = schema20.expand_long_action_states(generate())

    assert len(rows) == len(generate())
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["contrast_id"])].append(row)
    for contrast in grouped.values():
        assert {str(row["contrast_state"]) for row in contrast} == set(CONTRAST_STATES)
        contexts = {str(row["text"]).rsplit("\n", 1)[0] for row in contrast}
        assert len(contexts) == 1
        assert len(next(iter(contexts))) > 1_000
        assert all(row["external_benchmark_text_copied"] is False for row in contrast)

    train = [row for row in rows if row["scenario"] not in set(HOLDOUT_SCENARIOS)]
    validation = [row for row in rows if row["scenario"] in set(HOLDOUT_SCENARIOS)]
    assert not {str(row["contrast_id"]) for row in train} & {
        str(row["contrast_id"]) for row in validation
    }
    assert {str(row["scenario"]) for row in validation} == set(HOLDOUT_SCENARIOS)
