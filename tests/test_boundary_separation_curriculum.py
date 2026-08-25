from __future__ import annotations

from collections import Counter, defaultdict

from scripts.generate_boundary_separation_curriculum import (
    NEUTRAL_CLOSINGS,
    ROWS_PER_LABEL_PER_SCENARIO,
    SCENARIOS,
    SOURCE,
    generate,
)
from training.build_qwen_sft import convert_supported_rows


def test_boundary_triads_are_grounded_balanced_and_endpoint_matched() -> None:
    rows = [row for row in generate() if row["source"] == SOURCE]
    assert Counter(str(row["label"]) for row in rows) == {
        "SAFE": len(SCENARIOS) * ROWS_PER_LABEL_PER_SCENARIO,
        "UNCERTAIN": len(SCENARIOS) * ROWS_PER_LABEL_PER_SCENARIO,
        "SCAM": len(SCENARIOS) * ROWS_PER_LABEL_PER_SCENARIO,
    }
    converted, excluded = convert_supported_rows(rows)
    assert len(converted) == len(rows)
    assert excluded == []

    endings: dict[tuple[str, int], set[str]] = defaultdict(set)
    allowed = {"\n".join(value) for value in NEUTRAL_CLOSINGS}
    for row in rows:
        text = str(row["text"])
        ending_lines = tuple(text.splitlines()[-2:])
        assert "\n".join(ending_lines) in allowed
        endings[(str(row["scenario"]), list(NEUTRAL_CLOSINGS).index(ending_lines))].add(
            str(row["label"])
        )
    assert all(labels == {"SAFE", "UNCERTAIN", "SCAM"} for labels in endings.values())


def test_boundary_curriculum_retains_stage3_rows_without_id_collisions() -> None:
    rows = generate()
    counts = Counter(str(row["source"]) for row in rows)
    assert counts["scamguard_synthetic_evidence_persistence_v1"] > 0
    assert counts[SOURCE] == len(SCENARIOS) * ROWS_PER_LABEL_PER_SCENARIO * 3
    assert len({str(row["id"]) for row in rows}) == len(rows)
    assert all(str(row["pattern_reference"]).startswith("https://") for row in rows)
