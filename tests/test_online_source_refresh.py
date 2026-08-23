from __future__ import annotations

import json
import re
from pathlib import Path


def test_online_refresh_is_zero_admission_and_preserves_schema24() -> None:
    repository = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            repository
            / "reports/source-audits/online-refresh-2026-08-22.json"
        ).read_text(encoding="utf-8")
    )
    candidates = report["candidates"]

    assert report["current_schema_version"] == 24
    assert report["current_schema_changed"] is False
    assert report["downloaded_message_rows"] == 0
    assert report["direct_reddit_rows_admitted"] == 0
    assert report["decision"]["schema24_remains_frozen"] is True
    assert report["decision"]["new_training_rows"] == 0
    assert report["decision"]["new_evaluation_rows"] == 0
    assert len(candidates) == 5
    assert len({candidate["source_id"] for candidate in candidates}) == len(candidates)
    assert all(candidate["admitted_rows"] == 0 for candidate in candidates)
    assert all(candidate["reason_codes"] for candidate in candidates)
    assert all(candidate["primary_url"].startswith("https://") for candidate in candidates)
    assert all(candidate["metadata_url"].startswith("https://") for candidate in candidates)

    hub_candidates = candidates[:3]
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", candidate["publisher_revision"])
        for candidate in hub_candidates
    )
    assert candidates[3]["decision"] == "track_future_multilingual_external_diagnostic"
    assert candidates[4]["declared_wrapper_license"] is None
