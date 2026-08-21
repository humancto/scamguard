#!/usr/bin/env python3
"""Build privacy-normalized human-authored dialogue hard negatives.

Taskmaster-1's two-person Wizard-of-Oz dialogues are legitimate transactional
roleplay, not naturally occurring user communications and not scam labels.  We
therefore use them only as weakly labelled SAFE hard negatives and keep a
conversation-family-held selection slice outside fitting and calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_REVISION = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
SOURCE_URL = (
    "https://github.com/google-research-datasets/Taskmaster/tree/"
    f"{SOURCE_REVISION}/TM-1-2019"
)
EXPECTED_RAW_SHA256 = "cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f"
PARTITION_SALT = "scamguard-taskmaster1-dialogue-v1"
MAX_CHARS = 425

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\+?\d(?:[\d ()-]{5,}\d)(?![A-Za-z0-9])")

DOMAIN_PREFIXES = (
    "auto-repair",
    "coffee-ordering",
    "movie-tickets",
    "pizza-ordering",
    "restaurant-table",
    "uber-lyft",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def privacy_normalize(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = URL_RE.sub("<URL>", text)
    text = LONG_NUMBER_RE.sub("<NUMBER>", text)
    return " ".join(text.split())


def domain_for(instruction_id: str) -> str | None:
    normalized = instruction_id.casefold().replace("_", "-")
    return next((domain for domain in DOMAIN_PREFIXES if normalized.startswith(domain)), None)


def render_latest_context(utterances: list[dict[str, Any]], max_chars: int = MAX_CHARS) -> str:
    """Retain the most recent complete turns within the mobile context budget."""
    rendered: list[str] = []
    total = 0
    for utterance in reversed(utterances):
        raw_speaker = str(utterance.get("speaker", "UNKNOWN")).upper()
        speaker = "USER" if raw_speaker == "USER" else "ASSISTANT"
        body = privacy_normalize(str(utterance.get("text", "")))
        if not body:
            continue
        line = f"{speaker}: {body}"
        addition = len(line) + (1 if rendered else 0)
        if rendered and total + addition > max_chars:
            break
        if not rendered and len(line) > max_chars:
            line = line[: max_chars - 1].rstrip() + "…"
            addition = len(line)
        rendered.append(line)
        total += addition
    return "\n".join(reversed(rendered))


def partition(conversation_id: str) -> str:
    value = int(short_hash(f"{PARTITION_SALT}:{conversation_id}", 8), 16) % 100
    return "train" if value < 80 else "validation"


def row_for(dialogue: dict[str, Any], split: str) -> dict[str, object] | None:
    conversation_id = str(dialogue.get("conversation_id", "")).strip()
    instruction_id = str(dialogue.get("instruction_id", "")).strip()
    domain = domain_for(instruction_id)
    utterances = dialogue.get("utterances")
    if not conversation_id or domain is None or not isinstance(utterances, list):
        return None
    text = render_latest_context(utterances)
    if len(text) < 120 or text.count("\n") < 3:
        return None
    return {
        "id": "tm1-" + short_hash(conversation_id),
        "text": text,
        "label": "SAFE",
        "category": "NONE",
        "source": "taskmaster1_woz_dialogues",
        "source_label": "legitimate_task_dialogue",
        "license": "CC-BY-4.0",
        "split": split,
        "family_id": f"taskmaster1:{conversation_id}",
        "is_synthetic": False,
        "label_policy": "source_domain_legitimate_human_wizard_of_oz_roleplay",
        "source_language": "English",
        "source_domain": domain,
        "provenance_class": "human_crowdsourced_roleplay",
        "naturally_occurring_communication": False,
        "privacy_normalization": "email_url_and_phone_or_account_like_values_replaced",
        "context_policy": f"latest_complete_turns_capped_at_{MAX_CHARS}_characters",
    }


def deterministic_cap(
    rows: list[dict[str, object]], per_domain: int, seed: str
) -> list[dict[str, object]]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_domain"])].append(row)
    selected: list[dict[str, object]] = []
    for domain in DOMAIN_PREFIXES:
        candidates = sorted(
            grouped[domain],
            key=lambda row: short_hash(f"{seed}:{row['family_id']}", 64),
        )
        selected.extend(candidates[:per_domain])
    return sorted(selected, key=lambda row: str(row["id"]))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(
    raw_path: Path,
    train_path: Path,
    validation_path: Path,
    manifest_path: Path,
    *,
    train_per_domain: int = 100,
    validation_per_domain: int = 75,
) -> dict[str, object]:
    actual_hash = sha256(raw_path)
    if actual_hash != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"Taskmaster raw hash mismatch: expected {EXPECTED_RAW_SHA256}, got {actual_hash}"
        )
    dialogues = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(dialogues, list):
        raise ValueError("Taskmaster source is not a JSON array")

    pools: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    skipped = 0
    for dialogue in dialogues:
        if not isinstance(dialogue, dict):
            skipped += 1
            continue
        conversation_id = str(dialogue.get("conversation_id", ""))
        split = partition(conversation_id)
        row = row_for(dialogue, split)
        if row is None:
            skipped += 1
            continue
        pools[split].append(row)

    train_rows = deterministic_cap(pools["train"], train_per_domain, "tm1-train-v1")
    validation_rows = deterministic_cap(
        pools["validation"], validation_per_domain, "tm1-validation-v1"
    )
    train_families = {str(row["family_id"]) for row in train_rows}
    validation_families = {str(row["family_id"]) for row in validation_rows}
    if train_families & validation_families:
        raise AssertionError("Taskmaster conversation family crossed train and validation")

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    manifest: dict[str, object] = {
        "diagnostic_schema_version": 2,
        "source": {
            "repository": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license": "CC-BY-4.0",
            "raw_sha256": actual_hash,
            "collection": "two-person Wizard-of-Oz dialogues written by human participants",
        },
        "policy": {
            "label": "weak SAFE from legitimate task domain; not independently scam-labelled",
            "provenance_class": "human_crowdsourced_roleplay",
            "counted_as_naturally_occurring_communication": False,
            "used_for_fitting": True,
            "validation_used_for_fitting": False,
            "validation_used_for_threshold": False,
            "validation_may_inform_candidate_selection": True,
            "partition": "sha256 conversation-family 80/20 before capped sampling",
            "one_context_window_per_conversation": True,
            "max_context_characters": MAX_CHARS,
            "window_evidence": (
                "all 5,507 eligible source conversations measured at no more than 150 tokens "
                "after speaker-neutral-v1 with the pinned ModernBERT tokenizer"
            ),
            "privacy_normalization": (
                "email, URL, and phone/account-like values replaced before materialization"
            ),
        },
        "counts": {
            "source_dialogues": len(dialogues),
            "skipped_dialogues": skipped,
            "eligible_train": len(pools["train"]),
            "eligible_validation": len(pools["validation"]),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_domains": dict(Counter(str(row["source_domain"]) for row in train_rows)),
            "validation_domains": dict(
                Counter(str(row["source_domain"]) for row in validation_rows)
            ),
        },
        "artifacts": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": sha256(validation_path),
            },
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/taskmaster1_woz_dialogues.json")
    )
    parser.add_argument(
        "--train", type=Path, default=Path("data/generated/taskmaster_safe_train.jsonl")
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/external/taskmaster/taskmaster_validation.jsonl"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/external/taskmaster/manifest.json")
    )
    parser.add_argument("--train-per-domain", type=int, default=100)
    parser.add_argument("--validation-per-domain", type=int, default=75)
    args = parser.parse_args()
    manifest = build(
        args.raw,
        args.train,
        args.validation,
        args.manifest,
        train_per_domain=args.train_per_domain,
        validation_per_domain=args.validation_per_domain,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
