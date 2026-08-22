#!/usr/bin/env python3
"""Audit pinned MultiDoGO intent/slot annotations without emitting source text."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from scripts.build_multidogo_dialogues import EXPECTED_HEADER as UNANNOTATED_HEADER
    from scripts.fetch_multidogo import (
        ANNOTATION_GRANULARITIES,
        ANNOTATION_SPLITS,
        DOMAINS,
        annotation_paths,
        verify_repository,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_multidogo_dialogues import (  # type: ignore[no-redef]
        EXPECTED_HEADER as UNANNOTATED_HEADER,
    )
    from fetch_multidogo import (  # type: ignore[no-redef]
        ANNOTATION_GRANULARITIES,
        ANNOTATION_SPLITS,
        DOMAINS,
        annotation_paths,
        verify_repository,
    )

TURN_HEADER = [
    "conversationId",
    "turnNumber",
    "utteranceId",
    "utterance",
    "slot-labels",
    "intent",
]
SENTENCE_HEADER = [
    "conversationId",
    "turnNumber",
    "sentenceNumber",
    "utteranceId",
    "utterance",
    "slot-labels",
    "intent",
]
INTENT_SEPARATOR = "<div>"
BIO_PREFIX_RE = re.compile(r"^[BI][-_](.+)$")


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def annotation_header(granularity: str) -> list[str]:
    if granularity == "splits_annotated_at_turn_level":
        return TURN_HEADER
    if granularity == "splits_annotated_at_sentence_level":
        return SENTENCE_HEADER
    raise ValueError(f"unknown MultiDoGO annotation granularity: {granularity}")


def read_annotation_file(
    path: Path, granularity: str, domain: str, split: str
) -> list[dict[str, Any]]:
    expected = annotation_header(granularity)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError(
                f"unexpected MultiDoGO annotation header in {path}: {reader.fieldnames!r}"
            )
        for source in reader:
            conversation_id = source["conversationId"].strip()
            utterance_id = source["utteranceId"].strip()
            utterance = " ".join(source["utterance"].split())
            intents = tuple(
                value.strip()
                for value in source["intent"].split(INTENT_SEPARATOR)
                if value.strip()
            )
            slot_labels = tuple(value for value in source["slot-labels"].split() if value)
            if not conversation_id or not utterance_id or not utterance or not intents:
                raise ValueError(f"incomplete MultiDoGO annotation row in {path}")
            try:
                turn_number = int(source["turnNumber"])
                sentence_number = (
                    int(source["sentenceNumber"])
                    if granularity == "splits_annotated_at_sentence_level"
                    else None
                )
            except ValueError as error:
                raise ValueError(f"non-integer MultiDoGO annotation index in {path}") from error
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "conversation_id": conversation_id,
                    "turn_number": turn_number,
                    "sentence_number": sentence_number,
                    "utterance_id": utterance_id,
                    "utterance": utterance,
                    "intents": intents,
                    "slot_labels": slot_labels,
                }
            )
    if not rows:
        raise ValueError(f"empty MultiDoGO annotation file: {path}")
    return rows


def read_source_turn_index(repository: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    index: dict[tuple[str, str, int], dict[str, str]] = {}
    for domain in DOMAINS:
        path = repository / "data" / "unannotated" / f"{domain}.tsv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            # The upstream unannotated files use CSV despite their .tsv suffix.
            reader = csv.DictReader(handle)
            if reader.fieldnames != UNANNOTATED_HEADER:
                raise ValueError(f"unexpected MultiDoGO source header in {path}")
            for row in reader:
                key = (domain, row["conversationId"].strip(), int(row["turnNumber"]))
                if key in index:
                    raise ValueError(f"duplicate MultiDoGO source turn: {key!r}")
                index[key] = {
                    "role": row["authorRole"].strip(),
                    "utterance": " ".join(row["utterance"].split()),
                }
    return index


def alignment_failures(
    rows: list[dict[str, Any]],
    source_index: dict[tuple[str, str, int], dict[str, str]],
    granularity: str,
) -> Counter[str]:
    failures: Counter[str] = Counter()
    for row in rows:
        key = (str(row["domain"]), str(row["conversation_id"]), int(row["turn_number"]))
        source = source_index.get(key)
        if source is None:
            failures["missing_source_turn"] += 1
            continue
        if source["role"] != "customer":
            failures["annotated_non_customer_turn"] += 1
        annotated = normalized_text(str(row["utterance"]))
        original = normalized_text(source["utterance"])
        if granularity == "splits_annotated_at_turn_level":
            if annotated != original:
                failures["turn_text_mismatch"] += 1
        elif annotated not in original:
            failures["sentence_not_in_source_turn"] += 1
    return failures


def slot_type(label: str) -> str | None:
    if label == "O":
        return None
    matched = BIO_PREFIX_RE.match(label)
    return matched.group(1) if matched else label


def audit(repository: Path) -> dict[str, object]:
    source_manifest = verify_repository(repository, require_annotations=True)
    source_index = read_source_turn_index(repository)
    file_paths = annotation_paths(repository)
    expected_paths = {
        repository / "data" / "paper_splits" / granularity / domain / f"{split}.tsv"
        for granularity in ANNOTATION_GRANULARITIES
        for domain in DOMAINS
        for split in ANNOTATION_SPLITS
    }
    if set(file_paths) != expected_paths:
        raise RuntimeError("MultiDoGO annotation path enumeration changed")

    counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    conversation_splits: dict[tuple[str, str, str], str] = {}
    file_manifest: dict[str, dict[str, object]] = {}
    all_failures: Counter[str] = Counter()
    for granularity in ANNOTATION_GRANULARITIES:
        for domain in DOMAINS:
            for split in ANNOTATION_SPLITS:
                path = (
                    repository
                    / "data"
                    / "paper_splits"
                    / granularity
                    / domain
                    / f"{split}.tsv"
                )
                rows = read_annotation_file(path, granularity, domain, split)
                all_failures.update(alignment_failures(rows, source_index, granularity))
                conversations = {str(row["conversation_id"]) for row in rows}
                for conversation_id in conversations:
                    key = (granularity, domain, conversation_id)
                    previous = conversation_splits.setdefault(key, split)
                    if previous != split:
                        all_failures["conversation_crosses_paper_splits"] += 1
                for row in rows:
                    intent_counts.update(str(value) for value in row["intents"])
                    slot_counts.update(
                        value
                        for label in row["slot_labels"]
                        if (value := slot_type(str(label))) is not None
                    )
                counts[f"{granularity}:{domain}:{split}:rows"] = len(rows)
                counts[f"{granularity}:{domain}:{split}:conversations"] = len(conversations)
                relative = str(path.relative_to(repository))
                file_manifest[relative] = {
                    "rows": len(rows),
                    "conversations": len(conversations),
                    "sha256": file_sha256(path),
                }
    if all_failures:
        raise RuntimeError(f"MultiDoGO annotation audit failed: {dict(all_failures)}")
    return {
        "artifact_schema_version": 1,
        "source": "multidogo_human_service_dialogues",
        "repository": source_manifest["repository"],
        "revision": source_manifest["revision"],
        "license": source_manifest["license"],
        "annotation_tree_git_oid": source_manifest["annotation_tree_git_oid"],
        "contains_source_text": False,
        "role": "schema24 label-quality and participant-aware source audit only",
        "alignment": {
            "all_annotations_join_to_pinned_unannotated_turns": True,
            "annotated_rows_are_customer_turns": True,
            "turn_text_matches_source": True,
            "sentence_text_occurs_in_source_turn": True,
            "conversation_crosses_paper_splits": False,
        },
        "counts": dict(sorted(counts.items())),
        "intent_counts": dict(intent_counts.most_common()),
        "slot_type_counts": dict(slot_counts.most_common()),
        "files": file_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository", type=Path, default=Path("data/raw/multidogo/repository")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data/multidogo_annotation_audit.json")
    )
    args = parser.parse_args()
    report = audit(args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
