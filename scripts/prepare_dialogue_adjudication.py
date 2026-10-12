#!/usr/bin/env python3
"""Prepare a frozen, blind, error-enriched dialogue-label adjudication packet.

This is not a training-data builder, a relabeler, or a model-quality benchmark.
Publisher-label disagreements are expected and must not fail a review-completion
gate. The existing 635-row training audit and its contracts are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_protocol import audit_protocol, audit_protocol_sha256
from scripts.blind_audit import blind_review_id, file_sha256
from scripts.build_blind_audit_bundle import build_bundle
from scripts.check_audit_completion import audit_ids_sha256, audit_input_sha256

ROLE = "benchmark_label_adjudication"
SEED = 20261011
TARGET_ROWS = 96
SCAM_THRESHOLD = 0.18446353822946548
SPLITS = ("phone_scam_validation", "scam_dialogue_validation")
EXPECTED_HASHES = {
    "report": "ac8a6d53939355a82d453103170bc6fb31ef7d07b92cdbaf56bbcd0f3f83e045",
    "ledger": "83eadf7b1615d6c5628984918e0b90a4baf0b4424402cb3b606609a15356e4aa",
    "phone_scam_validation": "96d06085c4b5423a33235e417e1abcfa428ef0dd5a1f4551ded6305ead79f9ec",
    "scam_dialogue_validation": "c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0",
}
EXPECTED_SOURCE_ROWS = {"phone_scam_validation": 344, "scam_dialogue_validation": 294}
EXPECTED_TARGETS = {
    ("phone_scam_validation", "SAFE"): 15,
    ("phone_scam_validation", "SCAM"): 3,
    ("scam_dialogue_validation", "SAFE"): 28,
    ("scam_dialogue_validation", "SCAM"): 1,
}
SOURCE_NAMES = {
    "phone_scam_validation": "shakeleoatmeal_phone_scam_synthetic",
    "scam_dialogue_validation": "bothbosu_scam_dialogue",
}
CANONICAL_FIELDS = (
    "id",
    "dataset_split",
    "source",
    "source_label",
    "label",
    "category",
    "text",
    "auditor_label",
    "label_correct",
    "contains_sensitive_data",
    "notes",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stratum(row: dict) -> tuple[str, str, str]:
    return row["dataset_split"], row["source_dialogue_type"], row["label"]


def validate_inputs(report_path: Path, ledger_path: Path, sources: dict[str, Path]) -> list[dict]:
    """Join only the two declared open sources; never load any other dataset."""
    paths = {"report": report_path, "ledger": ledger_path, **sources}
    if set(sources) != set(SPLITS):
        raise ValueError("only the two frozen open dialogue sources are allowed")
    for name, path in paths.items():
        if file_sha256(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "completed"
        or report.get("experiment_id") != "sg-ettin150m-recovery-v2"
    ):
        raise ValueError("the completed frozen Ettin v2 evaluation is required")
    if report.get("sealed_tests_opened") is not False:
        raise ValueError("adjudication may not bind sealed-test evaluation")
    binding = report["prediction_ledger"]
    if (
        binding["sha256"] != EXPECTED_HASHES["ledger"]
        or Path(binding["path"]).resolve() != ledger_path.resolve()
    ):
        raise ValueError("report/ledger binding mismatch")
    predictions: dict[tuple[str, str], dict] = {}
    total = 0
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("split") not in SPLITS:
                continue
            key = row["split"], row["id"]
            if key in predictions:
                raise ValueError("duplicate dialogue prediction ID")
            predictions[key] = row
    if binding["rows"] != total or binding.get("examples", total) != total:
        raise ValueError("ledger row count differs from report")
    joined = []
    canonical_ids: set[str] = set()
    for split in SPLITS:
        if report["data_sha256"][split] != EXPECTED_HASHES[split]:
            raise ValueError("report/source binding mismatch")
        rows = read_jsonl(sources[split])
        if len(rows) != EXPECTED_SOURCE_ROWS[split]:
            raise ValueError("source row count differs from the frozen population")
        identifiers = [row["id"] for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate source ID")
        if set(identifiers) != {key[1] for key in predictions if key[0] == split}:
            raise ValueError("missing or extra dialogue predictions")
        for row in rows:
            if (
                row.get("split") not in {"validation", split}
                or row.get("source") != SOURCE_NAMES[split]
                or row.get("label") not in {"SAFE", "SCAM"}
                or not isinstance(row.get("text"), str)
                or not row["text"].strip()
                or not isinstance(row.get("source_dialogue_type"), str)
                or not row["source_dialogue_type"]
            ):
                raise ValueError("invalid open dialogue source row")
            if row["id"] in canonical_ids:
                raise ValueError("canonical source IDs must be globally unique")
            canonical_ids.add(row["id"])
            prediction = predictions[split, row["id"]]
            if prediction["truth"] != row["label"] or prediction["text_sha256"] != text_sha256(
                row["text"]
            ):
                raise ValueError("source truth/text differs from the prediction binding")
            for field in ("source", "category", "source_language"):
                if prediction.get(field) != row.get(field):
                    raise ValueError("source metadata differs from prediction")
            probabilities = prediction["probabilities"]
            if (
                set(probabilities) != {"SAFE", "UNCERTAIN", "SCAM"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in probabilities.values()
                )
                or abs(sum(probabilities.values()) - 1) > 1e-5
            ):
                raise ValueError("invalid frozen probability vector")
            verdict = (
                "SCAM"
                if probabilities["SCAM"] >= SCAM_THRESHOLD
                else "SAFE"
                if probabilities["SAFE"] >= 0.8 and probabilities["SCAM"] <= 0.2
                else "UNCERTAIN"
            )
            if prediction["verdict"] != verdict or prediction["threshold_scam"] is not (
                verdict == "SCAM"
            ):
                raise ValueError("prediction differs from frozen serving decision")
            joined.append({**row, "dataset_split": split, "prediction": prediction})
    return joined


def select_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    targets = [
        row
        for row in rows
        if (row["label"], row["prediction"]["verdict"])
        in {
            ("SAFE", "SCAM"),
            ("SCAM", "SAFE"),
        }
    ]
    counts = Counter((row["dataset_split"], row["label"]) for row in targets)
    if counts != EXPECTED_TARGETS:
        raise ValueError("target inventory differs from frozen 43 false alerts and 4 misses")
    weights = Counter(stratum(row) for row in targets)
    pools: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["prediction"]["verdict"] == row["label"] and stratum(row) in weights:
            pools[stratum(row)].append(row)
    if any(not pools[key] for key in weights):
        raise ValueError("a target stratum has no concordant control")
    control_count = TARGET_ROWS - len(targets)
    if control_count < len(weights) or sum(len(pools[key]) for key in weights) < control_count:
        raise ValueError("insufficient controls for the exact 96-row packet")
    # One per target stratum, then sequential largest proportional deficit with
    # lexicographic tie breaking and capacity caps; no scores enter control ranking.
    allocated = dict.fromkeys(weights, 1)
    while sum(allocated.values()) < control_count:
        available = [key for key in weights if allocated[key] < len(pools[key])]
        key = min(
            available,
            key=lambda key: (
                -(control_count * weights[key] / len(targets) - allocated[key]),
                key,
            ),
        )
        allocated[key] += 1
    generator = random.Random(SEED)
    controls = []
    for key in sorted(weights):
        controls.extend(
            generator.sample(sorted(pools[key], key=lambda row: row["id"]), allocated[key])
        )
    selected = sorted(targets + controls, key=lambda row: blind_review_id(row["id"]))
    if len(selected) != TARGET_ROWS or len({row["id"] for row in selected}) != TARGET_ROWS:
        raise ValueError("selection is not exactly 96 unique source rows")
    target_ids = {row["id"] for row in targets}
    private_selection = {
        "seed": SEED,
        "rows": [
            {
                "id": row["id"],
                "blind_id": blind_review_id(row["id"]),
                "dataset_split": row["dataset_split"],
                "source": row["source"],
                "source_dialogue_type": row["source_dialogue_type"],
                "reference_label": row["label"],
                "model_verdict": row["prediction"]["verdict"],
                "selection_role": "target" if row["id"] in target_ids else "concordant_control",
                "text_sha256": text_sha256(row["text"]),
            }
            for row in selected
        ],
        "control_allocations": [
            {
                "split": key[0],
                "dialogue_type": key[1],
                "reference_label": key[2],
                "target_rows": weights[key],
                "eligible_controls": len(pools[key]),
                "selected_controls": allocated[key],
            }
            for key in sorted(weights)
        ],
    }
    return selected, private_selection


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare(
    report_path: Path,
    ledger_path: Path,
    sources: dict[str, Path],
    output: Path,
    public_report: Path,
    review_app: Path,
) -> dict:
    inputs = [report_path, ledger_path, *sources.values(), review_app]
    if output.exists() or public_report.exists():
        raise FileExistsError("refusing to overwrite an existing adjudication artifact")
    if public_report.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("public receipt must not overwrite an input")
    if public_report.resolve().is_relative_to(output.resolve()):
        raise ValueError("public receipt must be outside the private answer-key directory")
    rows = validate_inputs(report_path, ledger_path, sources)
    selected, selection = select_rows(rows)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    bindings = output / "private-input-bindings.json"
    write_json(bindings, {"role": ROLE, "sha256": EXPECTED_HASHES, "seed": SEED})
    selection_path = output / "private-selection.json"
    write_json(selection_path, selection)
    canonical = output / "private-canonical.csv"
    with canonical.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    field: str(row.get(field, "")) if field in CANONICAL_FIELDS[:7] else ""
                    for field in CANONICAL_FIELDS
                }
            )
    manifest_path = output / "private-canonical.manifest.json"
    write_json(
        manifest_path,
        {
            "artifact_schema_version": 2,
            "role": ROLE,
            "audit_protocol_version": audit_protocol()["version"],
            "audit_protocol_sha256": audit_protocol_sha256(),
            "selected_inputs_sha256": audit_input_sha256(canonical),
            "selected_ids_sha256": audit_ids_sha256(canonical),
            "selected_rows": len(selected),
            "data_manifest_path": str(bindings),
            "data_manifest_sha256": file_sha256(bindings),
            "selection_manifest_sha256": file_sha256(selection_path),
            "disagreements_are_expected_and_must_be_preserved": True,
            "zero_disagreement_release_gate_applicable": False,
        },
    )
    bundle_path = output / "dialogue-adjudication-blind.zip"
    bundle = build_bundle(canonical, manifest_path, review_app, bundle_path)
    requirements_path = output / "reviewer-requirements.json"
    write_json(
        requirements_path,
        {
            "role": ROLE,
            "bundle_sha256": bundle["bundle_sha256"],
            "audit_protocol_sha256": audit_protocol_sha256(),
            "qualified_independent_human_required": True,
            "qualification": (
                "Document experience reviewing fraud/phishing evidence and text-label ambiguity."
            ),
            "independence": (
                "Reviewer must not be the model/training author and must not inspect "
                "the answer key or model predictions."
            ),
            "ai_decisions_do_not_count_as_independent_human_review": True,
            "instructions": (
                "Review all messages using the frozen rubric; return the CSV with an attributed "
                "qualification/independence attestation. Preserve disagreements as adjudication "
                "findings; do not force agreement with source labels."
            ),
            "reviewer_identity": "",
            "reviewer_qualification": "",
            "reviewer_attestation": "",
            "review_complete": False,
            "zero_disagreement_release_gate_applicable": False,
        },
    )
    result = {
        "artifact_schema_version": 1,
        "role": ROLE,
        "status": "pending_qualified_independent_human_adjudication",
        "experiment_id": "sg-ettin150m-recovery-v2",
        "selected_rows": TARGET_ROWS,
        "selection": {
            "target_reference_safe_model_scam": 43,
            "target_reference_scam_model_safe": 4,
            "concordant_controls": 49,
            "seed": SEED,
            "method": (
                "All frozen target errors plus seeded controls matching source, dialogue type, "
                "and reference label; one per target stratum then capacity-capped "
                "proportional-deficit allocation."
            ),
            "controls_are_not_human_verified_truth": True,
            "error_enriched_diagnostic_not_unbiased_accuracy": True,
        },
        "source_sha256": EXPECTED_HASHES,
        "packet_sha256": bundle["bundle_sha256"],
        "blind_inputs_sha256": bundle["blind_inputs_sha256"],
        "audit_protocol_sha256": audit_protocol_sha256(),
        "private_canonical_inputs_sha256": audit_input_sha256(canonical),
        "private_selection_sha256": file_sha256(selection_path),
        "reviewer_requirements_sha256": file_sha256(requirements_path),
        "contains_text": False,
        "contains_row_identifiers_or_predictions": False,
        "reviewer_packet_contains_answer_key": False,
        "independent_human_review": False,
        "release_authorized": False,
        "benchmark_labels_changed": False,
        "training_use_authorized": False,
        "threshold_retuning_authorized": False,
        "sealed_inputs_opened": False,
        "zero_disagreement_release_gate_applicable": False,
        "disagreements_must_be_preserved_in_versioned_adjudication": True,
        "existing_635_row_training_audit_unchanged_and_still_required": True,
        "limitations": [
            "Only source labels are being adjudicated; no independent accuracy claim is licensed, "
            "even after completion.",
            "The reviewer ZIP excludes selection roles, source metadata, reference labels, "
            "and model predictions.",
            "The existing training-audit zero-disagreement release gate is not this "
            "adjudication's completion criterion.",
            "Preparation does not perform or impersonate human review and does not resolve "
            "source/model redistribution rights.",
        ],
    }
    public_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(public_report, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/runs/sg-ettin150m-recovery-v2.open.json")
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("reports/runs/sg-ettin150m-recovery-v2.open.predictions.jsonl"),
    )
    parser.add_argument(
        "--phone",
        type=Path,
        default=Path(
            "data/experiments/schema25-full-call-curriculum/processed/phone_scam_validation.jsonl"
        ),
    )
    parser.add_argument(
        "--bothbosu",
        type=Path,
        default=Path("data/external/scam_dialogue/scam_dialogue_validation.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/audit/ettin-dialogue-v3"))
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("reports/data/ettin-dialogue-v3-adjudication.json"),
    )
    args = parser.parse_args()
    result = prepare(
        args.report,
        args.ledger,
        dict(zip(SPLITS, (args.phone, args.bothbosu), strict=True)),
        args.output,
        args.public_report,
        Path(__file__).with_name("review_blind_audit.py"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
