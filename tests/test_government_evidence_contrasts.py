from __future__ import annotations

from collections import Counter, defaultdict

from scripts.generate_government_evidence_contrasts import SOURCE, generate
from training.build_qwen_sft import convert_supported_rows


def test_government_evidence_contrasts_are_balanced_matched_triads() -> None:
    rows = generate()
    assert len(rows) == 96
    assert Counter(str(row["label"]) for row in rows) == {
        "SAFE": 32,
        "SCAM": 32,
        "UNCERTAIN": 32,
    }
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        assert row["source"] == SOURCE
        assert row["split"] == "train"
        assert row["is_synthetic"] is True
        groups[str(row["contrast_group"])].append(row)
    assert len(groups) == 32
    assert all(
        {str(row["label"]) for row in group} == {"SAFE", "SCAM", "UNCERTAIN"}
        for group in groups.values()
    )
    assert all(
        len({str(row["common_frame_sha256"]) for row in group}) == 1
        for group in groups.values()
    )


def test_government_evidence_contrasts_satisfy_grounded_sft_contract() -> None:
    rows = generate()
    converted, excluded = convert_supported_rows(rows)
    assert excluded == []
    assert len(converted) == len(rows)
