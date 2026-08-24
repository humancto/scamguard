from scripts.build_ai_internal_overlay import apply_decision


def test_apply_decision_relabels_without_mutating_source() -> None:
    row = {"id": "one", "text": "ordinary message", "label": "UNCERTAIN", "category": "OTHER"}

    revised, outcome = apply_decision(
        row,
        {"auditor_label": "SAFE", "contains_sensitive_data": "no"},
    )

    assert outcome == "relabelled"
    assert revised is not None
    assert revised["label"] == "SAFE"
    assert revised["category"] == "NONE"
    assert revised["schema24_ai_internal_original_label"] == "UNCERTAIN"
    assert row["label"] == "UNCERTAIN"


def test_apply_decision_quarantines_sensitive_row() -> None:
    revised, outcome = apply_decision(
        {"id": "one", "text": "code 123456", "label": "SAFE"},
        {"auditor_label": "SAFE", "contains_sensitive_data": "yes"},
    )

    assert revised is None
    assert outcome == "quarantined_sensitive"
