#!/usr/bin/env python3
"""CPU-only text-transform regression; no model inference or sealed-test access.

Compare the original V2 transform at its frozen SDK revision with current V2 on
every hash-pinned training/development/open row from the completed V2 experiment.
Report V3 input changes separately: changed input is NOT improved model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from scamguard.preprocessing import DIALOGUE_POLICY_ACTION_PRESERVING_V3, prepare_model_text

ROOT = Path(__file__).resolve().parents[1]
REVISION = "bd96b81e05a15b69f64bf036ba922f9ce155013a"
SOURCE = "src/scamguard/preprocessing.py"
SOURCE_SHA = "97b7d7c324b80ed0a1c449baf2f31e2e1d2ca45282d1761464aff67dabdb6822"
CONFIG = "configs/ettin150m-recovery-v2.json"
CONFIG_SHA = "5c6a241a6994d0a64ea7d92c8052b9b4d1a76e917b553683ba1207babd7986b5"
V2 = "speaker-neutral-evidence-recent-v2"
DIAGNOSTIC_CASES = {
    "phone-scam-synthetic-c0f0ef4e46f08032": (
        "phone_scam_validation", "cloud_access_request",
        r"\bgive\b[^.!?]{0,100}\baccess\b[^.!?]{0,100}\bAWS\s+Management\s+Console\b",
    ),
    "bothbosu_scam_dialogue-76a78e6220e58480": (
        "scam_dialogue_validation", "account_number_request",
        r"\bconfirm\b[^.!?]{0,110}\baccount\s+number\b",
    ),
}


def retention(cases: dict[str, str], checkpoint: Path) -> dict:
    from transformers import AutoTokenizer

    expected = {
        "tokenizer.json": "e0a696f1fbf0f2c7d212e8b9e133250d19bd1c7e5bc34fc91c59d008894dc7b1",
        "tokenizer_config.json": "d1356b2fcea62ce13fa24529b5f99a9f95a62ef278d75ff0e9b0b2c32a325da4",
    }
    for filename, digest in expected.items():
        if sha((checkpoint / filename).read_bytes()) != digest:
            raise ValueError("frozen tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    if set(cases) != set(DIAGNOSTIC_CASES):
        raise ValueError("missing retention diagnostic")
    results = {}
    for identifier, text in cases.items():
        _, name, pattern = DIAGNOSTIC_CASES[identifier]
        if re.search(pattern, text, re.I) is None:
            raise ValueError("diagnostic raw text has no expected request")
        result = {}
        for policy in (V2, DIALOGUE_POLICY_ACTION_PRESERVING_V3):
            prepared = prepare_model_text(text, policy)
            encoded = tokenizer(prepared, max_length=256, truncation=True)["input_ids"]
            served = tokenizer.decode(
                encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            result[policy] = {
                "compacted_chars": len(prepared), "served_tokens": len(encoded),
                "request_in_compacted_text": bool(re.search(pattern, prepared, re.I)),
                "request_in_served_text": bool(re.search(pattern, served, re.I)),
            }
        results[name] = result
    return {"role": "selected_open_error_diagnostics_not_accuracy", "examples": results,
            "tokenizer_sha256": expected, "max_length": 256, "truncation_side": "right"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run() -> dict:
    original = subprocess.run(
        ["git", "show", f"{REVISION}:{SOURCE}"], cwd=ROOT, check=True, capture_output=True,
    ).stdout
    if sha(original) != SOURCE_SHA:
        raise ValueError("frozen original preprocessing source hash mismatch")
    namespace: dict = {"__name__": "frozen_v2_preprocessing"}
    # Only execute the known, hash-verified local project module (stdlib re only).
    exec(compile(original, SOURCE, "exec"), namespace)
    payload = (ROOT / CONFIG).read_bytes()
    if sha(payload) != CONFIG_SHA:
        raise ValueError("frozen experiment config hash mismatch")
    config = json.loads(payload)
    bindings = {**config["data"], **config["evaluation"]["splits"]}
    populations = {}
    cases = {}
    for name, binding in bindings.items():
        if name not in {"train", "dev"} and binding["role"] not in {
            "open_diagnostic", "regression",
        }:
            raise ValueError("unexpected input role; sealed inputs are forbidden")
        raw = (ROOT / binding["path"]).read_bytes()
        if sha(raw) != binding["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) != binding["rows"]:
            raise ValueError(f"input count mismatch: {name}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate input ID: {name}")
        unchanged_v2 = changed_v3 = 0
        for row in rows:
            text = row["text"]
            if row["id"] in DIAGNOSTIC_CASES and DIAGNOSTIC_CASES[row["id"]][0] == name:
                cases[row["id"]] = text
            old = namespace["prepare_model_text"](text, V2)
            current = prepare_model_text(text, V2)
            unchanged_v2 += old == current
            changed_v3 += current != prepare_model_text(text, DIALOGUE_POLICY_ACTION_PRESERVING_V3)
        populations[name] = {
            "rows": len(rows), "input_sha256": binding["sha256"],
            "v2_identical": unchanged_v2, "v2_changed": len(rows) - unchanged_v2,
            "v3_changed_inputs": changed_v3,
        }
    return {
        "schema_version": 1, "purpose": "text_transform_regression_only",
        "frozen_runtime_revision": REVISION, "frozen_source_sha256": SOURCE_SHA,
        "current_source_sha256": sha((ROOT / SOURCE).read_bytes()),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "config_sha256": CONFIG_SHA, "candidate_policy": DIALOGUE_POLICY_ACTION_PRESERVING_V3,
        "populations": populations,
        "rows": sum(item["rows"] for item in populations.values()),
        "v2_all_identical": all(item["v2_changed"] == 0 for item in populations.values()),
        "request_retention_diagnostics": retention(cases, ROOT / config["outputs"]["checkpoint"]),
        "model_inference_run": False, "sealed_tests_opened": False,
        "candidate_quality_validated": False, "release_authorized": False,
        "limitations": [
            "V3 is opt-in; no existing calibration manifest changed.",
            "Input retention is not evidence of improved classification or latency.",
            "Open diagnostics informed V3 and cannot become independent final evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create preserves prior evidence on accidental reruns.
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "rows", "v2_all_identical", "model_inference_run", "candidate_quality_validated",
    )}))
    if not report["v2_all_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
