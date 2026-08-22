#!/usr/bin/env python3
"""Audit pinned MultiDoGO intent/slot annotations without emitting source text."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
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

TURN_GRANULARITY = "splits_annotated_at_turn_level"
SENTENCE_GRANULARITY = "splits_annotated_at_sentence_level"
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
INTEGER_INDEX_RE = re.compile(r"^(?P<integer>0|[1-9][0-9]*)(?:\.0+)?$")


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def annotation_header(granularity: str) -> list[str]:
    if granularity == TURN_GRANULARITY:
        return TURN_HEADER
    if granularity == SENTENCE_GRANULARITY:
        return SENTENCE_HEADER
    raise ValueError(f"unknown MultiDoGO annotation granularity: {granularity}")


def parse_annotation_index(value: str, path: Path) -> int:
    """Parse an upstream non-negative integer index without rounding."""
    matched = INTEGER_INDEX_RE.fullmatch(value.strip())
    if matched is None:
        raise ValueError(f"non-integer MultiDoGO annotation index in {path}")
    return int(matched.group("integer"))


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
            if not conversation_id or not utterance_id or not intents:
                raise ValueError(f"incomplete MultiDoGO annotation row in {path}")
            turn_number = parse_annotation_index(source["turnNumber"], path)
            sentence_number = (
                parse_annotation_index(source["sentenceNumber"], path)
                if granularity == SENTENCE_GRANULARITY
                else None
            )
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
                    "empty_utterance": not utterance,
                }
            )
    if not rows:
        raise ValueError(f"empty MultiDoGO annotation file: {path}")
    return rows


def slot_type(label: str) -> str | None:
    if label == "O":
        return None
    matched = BIO_PREFIX_RE.match(label)
    return matched.group(1) if matched else label


def annotation_identity(row: dict[str, Any], granularity: str) -> tuple[object, ...]:
    base: tuple[object, ...] = (
        str(row["domain"]),
        str(row["conversation_id"]),
        int(row["turn_number"]),
    )
    if granularity == SENTENCE_GRANULARITY:
        return (*base, int(row["sentence_number"]))
    return base


def read_unannotated_conversation_ids(repository: Path) -> set[tuple[str, str]]:
    identifiers: set[tuple[str, str]] = set()
    for domain in DOMAINS:
        path = repository / "data" / "unannotated" / f"{domain}.tsv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != UNANNOTATED_HEADER:
                raise ValueError(f"unexpected MultiDoGO source header in {path}")
            for row in reader:
                conversation_id = row["conversationId"].strip()
                if not conversation_id:
                    raise ValueError(f"empty MultiDoGO source conversation ID in {path}")
                identifiers.add((domain, conversation_id))
    return identifiers


def cross_granularity_stats(
    turn_rows: dict[tuple[object, ...], dict[str, Any]],
    sentence_rows: dict[tuple[object, ...], list[dict[str, Any]]],
) -> dict[str, int]:
    turn_keys = set(turn_rows)
    sentence_keys = set(sentence_rows)
    common = turn_keys & sentence_keys
    text_aligned = 0
    text_divergent = 0
    for key in common:
        turn_text = normalized_text(str(turn_rows[key]["utterance"]))
        nonempty_sentences = [
            normalized_text(str(row["utterance"]))
            for row in sentence_rows[key]
            if row.get("empty_utterance") is not True
        ]
        if all(sentence in turn_text for sentence in nonempty_sentences):
            text_aligned += 1
        else:
            text_divergent += 1
    return {
        "turn_keys": len(turn_keys),
        "sentence_turn_keys": len(sentence_keys),
        "common_turn_keys": len(common),
        "turn_only_keys": len(turn_keys - sentence_keys),
        "sentence_only_keys": len(sentence_keys - turn_keys),
        "common_keys_with_all_sentence_text_contained": text_aligned,
        "common_keys_with_publisher_text_divergence": text_divergent,
    }


def audit(repository: Path) -> dict[str, object]:
    source_manifest = verify_repository(repository, require_annotations=True)
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
    identities: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    turn_rows: dict[tuple[object, ...], dict[str, Any]] = {}
    sentence_rows: defaultdict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    annotation_conversations: dict[str, set[tuple[str, str]]] = defaultdict(set)
    file_manifest: dict[str, dict[str, object]] = {}
    failures: Counter[str] = Counter()

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
                usable_rows = [row for row in rows if row.get("empty_utterance") is not True]
                empty_rows = len(rows) - len(usable_rows)
                if not usable_rows:
                    raise RuntimeError(f"MultiDoGO annotation file has no usable rows: {path}")
                conversations = {str(row["conversation_id"]) for row in rows}
                token_mismatches = 0
                for row in rows:
                    identity = annotation_identity(row, granularity)
                    if identity in identities[granularity]:
                        failures["duplicate_annotation_identity"] += 1
                    identities[granularity].add(identity)
                    conversation_id = str(row["conversation_id"])
                    annotation_conversations[granularity].add((domain, conversation_id))
                    split_key = (granularity, domain, conversation_id)
                    previous = conversation_splits.setdefault(split_key, split)
                    if previous != split:
                        failures["conversation_crosses_paper_splits"] += 1
                    if row.get("empty_utterance") is True:
                        continue
                    intent_counts.update(str(value) for value in row["intents"])
                    slot_counts.update(
                        value
                        for label in row["slot_labels"]
                        if (value := slot_type(str(label))) is not None
                    )
                    if len(str(row["utterance"]).split()) != len(row["slot_labels"]):
                        token_mismatches += 1
                    turn_key = (
                        domain,
                        conversation_id,
                        int(row["turn_number"]),
                    )
                    if granularity == TURN_GRANULARITY:
                        turn_rows[turn_key] = row
                    else:
                        sentence_rows[turn_key].append(row)
                counts[f"{granularity}:{domain}:{split}:rows"] = len(rows)
                counts[f"{granularity}:{domain}:{split}:usable_rows"] = len(usable_rows)
                counts[f"{granularity}:{domain}:{split}:empty_utterance_rows"] = empty_rows
                counts[f"{granularity}:{domain}:{split}:slot_token_mismatch_rows"] = (
                    token_mismatches
                )
                counts[f"{granularity}:{domain}:{split}:conversations"] = len(conversations)
                relative = str(path.relative_to(repository))
                file_manifest[relative] = {
                    "rows": len(rows),
                    "usable_rows": len(usable_rows),
                    "empty_utterance_rows": empty_rows,
                    "slot_token_mismatch_rows": token_mismatches,
                    "conversations": len(conversations),
                    "sha256": file_sha256(path),
                }

    if failures:
        raise RuntimeError(f"MultiDoGO annotation audit failed: {dict(failures)}")
    unannotated_ids = read_unannotated_conversation_ids(repository)
    turn_unannotated_overlap = annotation_conversations[TURN_GRANULARITY] & unannotated_ids
    sentence_unannotated_overlap = (
        annotation_conversations[SENTENCE_GRANULARITY] & unannotated_ids
    )
    cross_stats = cross_granularity_stats(turn_rows, sentence_rows)
    return {
        "artifact_schema_version": 2,
        "source": "multidogo_human_service_dialogues",
        "repository": source_manifest["repository"],
        "revision": source_manifest["revision"],
        "license": source_manifest["license"],
        "readme_sha256": source_manifest["readme_sha256"],
        "annotation_tree_git_oid": source_manifest["annotation_tree_git_oid"],
        "contains_source_text": False,
        "role": "schema24 publisher-annotation integrity and selection audit",
        "contracts": {
            "publisher_readme_describes_paper_splits_as_customer_turns": True,
            "annotation_identities_unique_within_granularity": True,
            "conversations_do_not_cross_splits_within_granularity": True,
            "empty_annotation_utterances_quarantined_from_curriculum": True,
            "turn_level_rows_are_selected_directly": True,
            "sentence_level_rows_are_audit_only": True,
            "annotated_and_unannotated_conversation_ids_are_separate_collections": (
                not turn_unannotated_overlap and not sentence_unannotated_overlap
            ),
            "paper_dev_test_rows_enter_fitting": False,
        },
        "collection_relationships": {
            "turn_annotation_conversations": len(
                annotation_conversations[TURN_GRANULARITY]
            ),
            "sentence_annotation_conversations": len(
                annotation_conversations[SENTENCE_GRANULARITY]
            ),
            "unannotated_conversations": len(unannotated_ids),
            "turn_to_unannotated_conversation_id_overlap": len(turn_unannotated_overlap),
            "sentence_to_unannotated_conversation_id_overlap": len(
                sentence_unannotated_overlap
            ),
            **cross_stats,
        },
        "known_upstream_anomalies": {
            "empty_utterances_are_excluded_from_model_rows": True,
            "slot_token_count_mismatches_are_reported_not_used_for_token_alignment": True,
            "cross_granularity_text_divergence_is_reported_and_sentence_rows_are_not_selected": (
                True
            ),
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
