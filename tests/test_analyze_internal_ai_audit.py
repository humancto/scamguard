from pathlib import Path

from scripts.analyze_internal_ai_audit import internal_ai_result


def test_internal_ai_result_never_authorizes_release() -> None:
    result = internal_ai_result(
        {
            "path": "/tmp/joined.csv",
            "release_gate_passed": True,
            "agreement": 1.0,
            "imported_from_blind_bundle": True,
        },
        decisions_path=Path("internal.csv"),
        canonical_audit_path=Path("canonical.csv"),
    )

    assert result["agreement"] == 1.0
    assert result["verified_against_blind_bundle"] is True
    assert result["metric_gate_would_pass_if_review_were_independent_human"] is True
    assert result["independent_human_review"] is False
    assert result["imported_from_blind_bundle"] is False
    assert result["release_gate_passed"] is False
    assert result["publication_authorized"] is False
    assert "path" not in result
