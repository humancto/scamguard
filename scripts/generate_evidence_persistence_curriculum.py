#!/usr/bin/env python3
"""Generate paired dialogues where earlier evidence must survive neutral closing turns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scamguard.metrics import file_sha256
from scripts.generate_dialogue_curriculum import SCENARIOS, render, short_hash
from training.build_qwen_sft import convert_supported_rows

GENERATOR_VERSION = 1
ROWS_PER_LABEL_PER_SCENARIO = 32
SOURCE = "scamguard_synthetic_evidence_persistence_v1"

# Identical closings are used for both labels so the endpoint cannot reveal the
# verdict. The earlier grounded behavior, not the last speaker or last phrase,
# must determine the label.
NEUTRAL_CLOSINGS = (
    (
        "PARTICIPANT: I want time to verify what was said.",
        "CONTACT: I understand; take the time you need.",
        "PARTICIPANT: I will review my own records before deciding.",
    ),
    (
        "PARTICIPANT: I am not ready to continue right now.",
        "CONTACT: All right, I will pause here.",
        "PARTICIPANT: I have written down the details from earlier.",
    ),
    (
        "PARTICIPANT: Let me think about the earlier request.",
        "CONTACT: Of course; I can wait.",
        "PARTICIPANT: I will make my decision after this conversation ends.",
    ),
    (
        "PARTICIPANT: I need to check this with someone I trust.",
        "CONTACT: That is fine; we can stop for now.",
        "PARTICIPANT: Thanks, I am ending the conversation.",
    ),
)


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, spec in SCENARIOS.items():
        for label_key, label in (("scam", "SCAM"), ("safe", "SAFE")):
            for index in range(ROWS_PER_LABEL_PER_SCENARIO):
                base = render(str(spec[label_key]), scenario, label, index)
                closing = NEUTRAL_CLOSINGS[index % len(NEUTRAL_CLOSINGS)]
                text = base + "\n" + "\n".join(closing)
                row_id = "persistence-" + short_hash(
                    f"v{GENERATOR_VERSION}:{scenario}:{label}:{index}:{text}"
                )
                rows.append(
                    {
                        "id": row_id,
                        "text": text,
                        "label": label,
                        "category": str(spec["category"]) if label == "SCAM" else "NONE",
                        "source": SOURCE,
                        "source_label": label.casefold(),
                        "license": "Apache-2.0",
                        "split": "train",
                        "family_id": (
                            f"synthetic:evidence-persistence:{scenario}:"
                            f"{label.casefold()}:v{GENERATOR_VERSION}"
                        ),
                        "is_synthetic": True,
                        "synthetic_method": (
                            "paired_original_advisory_grounded_dialogue_with_label_matched_"
                            "neutral_closing_turns"
                        ),
                        "pattern_reference": str(spec["reference"]),
                        "source_language": "English",
                        "scenario": scenario,
                        "generator_version": GENERATOR_VERSION,
                        "context_policy": (
                            "earlier grounded behavior remains authoritative after neutral ending"
                        ),
                    }
                )
    return sorted(rows, key=lambda row: str(row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/evidence_persistence_curriculum.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/evidence_persistence_curriculum_manifest.json"),
    )
    args = parser.parse_args()
    rows = generate()
    converted, excluded = convert_supported_rows(rows)
    if excluded or len(converted) != len(rows):
        raise ValueError("every persistence row must satisfy the grounded SFT contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "source": SOURCE,
        "license": "Apache-2.0",
        "method": (
            "paired deterministic advisory-grounded dialogues with identical neutral closing "
            "families across labels"
        ),
        "used_for_fitting": True,
        "used_for_threshold": False,
        "held_rows_copied": 0,
        "design_disclosure": (
            "The prior-open BothBosu validation aggregate and qualitative failure mode informed "
            "the evidence-persistence objective; no BothBosu row is included or transformed."
        ),
        "rows": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "scenarios": dict(Counter(str(row["scenario"]) for row in rows)),
        "sha256": file_sha256(args.output),
        "pattern_references": sorted({str(row["pattern_reference"]) for row in rows}),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
