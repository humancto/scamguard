from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.build_schema23_evidence_compaction import (
    CALIBRATION_FAMILIES_PER_DOMAIN,
    MULTIDOGO_REAL_VERDICT_WEIGHT,
    partition_multidogo,
    remove_exact_duplicate_state_families,
    remove_reference_overlap_families,
    shape_multidogo_state_context,
)
from training.train_encoder import action_target_metrics, fit_action_thresholds


def state_row(domain: str, family: int, state: str) -> dict[str, object]:
    return {
        "id": f"{domain}-{family}-{state}",
        "family_id": f"{domain}-{family}",
        "source_domain": domain,
        "contrast_state": state,
    }


def real_row(domain: str, family: int, window: str) -> dict[str, object]:
    return {
        "id": f"real-{domain}-{family}-{window}",
        "family_id": f"{domain}-{family}",
        "source_domain": domain,
        "source_window": window,
        "action_verdict_weight": 0.5,
    }


def test_multidogo_action_calibration_is_family_and_domain_disjoint() -> None:
    domains = ("airline", "fastfood", "finance", "media")
    states = ("routine_safe", "verified_safe", "unresolved", "harmful_scam")
    state_rows = [
        state_row(domain, family, state)
        for domain in domains
        for family in range(25)
        for state in states
    ]
    real_rows = [
        real_row(domain, family, window)
        for domain in domains
        for family in range(25)
        for window in ("recent_complete_turns", "highest_risk_agent_turn")
    ]
    fit, calibration, calibration_states = partition_multidogo(real_rows, state_rows)
    fit_families = {row["family_id"] for row in fit}
    calibration_families = {row["family_id"] for row in calibration}
    assert not fit_families & calibration_families
    assert len(calibration_families) == len(domains) * CALIBRATION_FAMILIES_PER_DOMAIN
    assert len(calibration_states) == len(calibration_families) * len(states)
    assert all(
        row.get("action_verdict_weight") == MULTIDOGO_REAL_VERDICT_WEIGHT
        for row in fit
        if str(row["id"]).startswith("real-")
    )


def test_schema23_builder_is_tracked() -> None:
    assert Path("scripts/build_schema23_evidence_compaction.py").is_file()


def test_overlap_control_removes_whole_candidate_family() -> None:
    candidates = [
        {"id": "a1", "family_id": "a", "text": "send the verification code now"},
        {"id": "a2", "family_id": "a", "text": "use the official app yourself"},
        {"id": "b1", "family_id": "b", "text": "a wholly different ordinary sentence"},
    ]
    references = [{"text": "send the verification code now"}]
    kept, stats = remove_reference_overlap_families(candidates, references)
    assert [row["id"] for row in kept] == ["b1"]
    assert stats["exact_overlap_rows"] == 1
    assert stats["families_removed_for_any_near_overlap"] == 1
    assert stats["rows_removed_with_overlap_families"] == 2


def test_action_thresholds_are_fit_only_from_explicit_target_rows() -> None:
    names = ("target",)
    rows = [
        {"id": "negative", "action_targets": {"target": False}},
        {"id": "positive", "action_targets": {"target": True}},
    ]
    probabilities = np.array([0.2, 0.4])
    action_logits = np.log(probabilities / (1.0 - probabilities))
    logits = np.column_stack((np.zeros((2, 3)), action_logits))
    thresholds = fit_action_thresholds(rows, logits, names)
    assert thresholds == {"target": 0.4}
    metrics = action_target_metrics(rows, logits, names, thresholds)
    assert metrics["exact_match_at_0_5"] == 0.5
    assert metrics["exact_match_at_calibrated"] == 1.0


def test_multidogo_context_shape_keeps_decisive_and_two_delayed_turns() -> None:
    states = ("routine_safe", "verified_safe", "unresolved", "harmful_scam")
    rows = [
        {
            "id": state,
            "contrast_id": "family",
            "contrast_state": state,
            "text": (
                "CUSTOMER: shared prefix\n"
                f"AGENT: decisive {state}\n"
                "CUSTOMER: source continuation\n"
                "AGENT: more source continuation\n"
                "CUSTOMER: delayed check\n"
                "AGENT: delayed pause"
            ),
        }
        for state in states
    ]
    shaped = shape_multidogo_state_context(rows)
    assert len(shaped) == 4
    for row in shaped:
        assert str(row["text"]).splitlines() == [
            "CUSTOMER: shared prefix",
            f"AGENT: decisive {row['contrast_state']}",
            "CUSTOMER: delayed check",
            "AGENT: delayed pause",
        ]
        assert len(str(row["schema23_source_text_sha256"])) == 64


def test_shaped_state_dedup_removes_the_complete_later_family() -> None:
    rows = [
        {
            "id": f"{family}-{state}",
            "contrast_id": family,
            "contrast_state": state,
            "text": f"shared {state}",
        }
        for family in ("a", "b")
        for state in ("routine_safe", "verified_safe", "unresolved", "harmful_scam")
    ]
    kept, stats = remove_exact_duplicate_state_families(rows)
    assert {row["contrast_id"] for row in kept} == {"a"}
    assert stats["families_removed"] == 1
    assert stats["rows_removed"] == 4
