#!/usr/bin/env python3
"""Compare a completed blind AI review without authorizing the human release gate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

try:
    from scripts.import_blind_audit import import_returned_audit
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from import_blind_audit import import_returned_audit  # type: ignore[no-redef]


def internal_ai_result(
    comparison: dict[str, object],
    *,
    decisions_path: Path,
    canonical_audit_path: Path,
) -> dict[str, object]:
    """Remove any implication that an internal AI review is independent human evidence."""

    metric_gate_would_pass = bool(comparison.get("release_gate_passed"))
    result = dict(comparison)
    result.pop("path", None)
    result.update(
        {
            "artifact_schema_version": 1,
            "review_kind": "ai_internal_blind",
            "reviewer_type": "ai_assistant",
            "independent_human_review": False,
            "decision_artifact_path": str(decisions_path),
            "canonical_comparison_path": str(canonical_audit_path),
            "verified_against_blind_bundle": True,
            "imported_from_blind_bundle": False,
            "joined_audit_artifact_persisted": False,
            "metric_gate_would_pass_if_review_were_independent_human": metric_gate_would_pass,
            "release_gate_passed": False,
            "publication_authorized": False,
            "limitations": [
                "The reviewer is the same AI system assisting with model development.",
                "This result is useful for label diagnostics and exploratory training only.",
                "A separate independent human review remains required for release publication.",
            ],
        }
    )
    return result


def analyze(
    decisions_path: Path,
    bundle_path: Path,
    canonical_audit_path: Path,
    canonical_manifest_path: Path,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="scamguard-ai-audit-") as directory:
        temporary = Path(directory)
        comparison = import_returned_audit(
            decisions_path,
            bundle_path,
            canonical_audit_path,
            canonical_manifest_path,
            temporary / "joined.csv",
            temporary / "comparison.json",
        )
    return internal_ai_result(
        comparison,
        decisions_path=decisions_path,
        canonical_audit_path=canonical_audit_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(
        args.decisions,
        args.bundle,
        args.canonical_audit,
        args.canonical_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
