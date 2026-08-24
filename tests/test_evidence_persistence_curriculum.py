from __future__ import annotations

from collections import Counter, defaultdict

from scripts.generate_evidence_persistence_curriculum import (
    NEUTRAL_CLOSINGS,
    ROWS_PER_LABEL_PER_SCENARIO,
    generate,
)
from training.build_qwen_sft import convert_supported_rows


def test_persistence_curriculum_is_balanced_grounded_and_endpoint_matched() -> None:
    rows = generate()
    counts = Counter(str(row["label"]) for row in rows)
    assert counts["SAFE"] == counts["SCAM"]
    assert counts["SAFE"] > 0
    converted, excluded = convert_supported_rows(rows)
    assert len(converted) == len(rows)
    assert excluded == []

    endings: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        text = str(row["text"])
        ending = "\n".join(text.splitlines()[-3:])
        assert ending in {"\n".join(value) for value in NEUTRAL_CLOSINGS}
        key = (str(row["scenario"]), list(NEUTRAL_CLOSINGS).index(tuple(text.splitlines()[-3:])))
        endings[key].add(str(row["label"]))
    assert all(labels == {"SAFE", "SCAM"} for labels in endings.values())


def test_persistence_curriculum_has_expected_scenario_geometry() -> None:
    rows = generate()
    by_scenario_label = Counter((str(row["scenario"]), str(row["label"])) for row in rows)
    assert by_scenario_label
    assert set(by_scenario_label.values()) == {ROWS_PER_LABEL_PER_SCENARIO}
    assert len({str(row["id"]) for row in rows}) == len(rows)
