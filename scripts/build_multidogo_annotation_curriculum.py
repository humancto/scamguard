#!/usr/bin/env python3
"""Build a split-safe hard-negative curriculum from audited MultiDoGO annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from scripts.audit_multidogo_annotations import read_annotation_file, slot_type
    from scripts.build_multidogo_dialogues import LICENSE, SOURCE
    from scripts.build_schema19_call_windows import read_jsonl, write_jsonl
    from scripts.fetch_multidogo import (
        ANNOTATION_SPLITS,
        ANNOTATION_TREE_GIT_OID,
        DOMAINS,
        REVISION,
        annotation_paths,
        verify_repository,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from audit_multidogo_annotations import (  # type: ignore[no-redef]
        read_annotation_file,
        slot_type,
    )
    from build_multidogo_dialogues import LICENSE, SOURCE  # type: ignore[no-redef]
    from build_schema19_call_windows import (  # type: ignore[no-redef]
        read_jsonl,
        write_jsonl,
    )
    from fetch_multidogo import (  # type: ignore[no-redef]
        ANNOTATION_SPLITS,
        ANNOTATION_TREE_GIT_OID,
        DOMAINS,
        REVISION,
        annotation_paths,
        verify_repository,
    )

ARTIFACT_SCHEMA_VERSION = 1
TURN_GRANULARITY = "splits_annotated_at_turn_level"
LABEL_TOKEN_RE = re.compile(r"[a-z0-9]+")
SENSITIVE_CONCEPTS = {
    "account",
    "address",
    "bank",
    "billing",
    "card",
    "claim",
    "code",
    "credit",
    "credential",
    "deductible",
    "deposit",
    "email",
    "identity",
    "insurance",
    "license",
    "loan",
    "order",
    "password",
    "payment",
    "phone",
    "pin",
    "policy",
    "purchase",
    "refund",
    "routing",
    "security",
    "subscription",
    "transfer",
    "verification",
    "verify",
}
TOKEN_ALIASES = {
    "accounts": "account",
    "cards": "card",
    "claims": "claim",
    "credentials": "credential",
    "loans": "loan",
    "orders": "order",
    "payments": "payment",
    "policies": "policy",
    "purchases": "purchase",
    "refunds": "refund",
    "subscriptions": "subscription",
    "transfers": "transfer",
}


def label_tokens(value: str) -> set[str]:
    """Normalize publisher ontology labels without inspecting message text."""

    return {
        TOKEN_ALIASES.get(token, token)
        for token in LABEL_TOKEN_RE.findall(value.casefold())
    }


def build_annotation_index(repository: Path) -> dict[tuple[str, str], dict[str, object]]:
    """Aggregate turn-level labels by conversation while retaining paper splits."""

    mutable: dict[tuple[str, str], dict[str, Any]] = {}
    for domain in DOMAINS:
        for split in ANNOTATION_SPLITS:
            path = repository / "data" / "paper_splits" / TURN_GRANULARITY / domain / f"{split}.tsv"
            rows = read_annotation_file(path, TURN_GRANULARITY, domain, split)
            for row in rows:
                key = (domain, str(row["conversation_id"]))
                record = mutable.setdefault(
                    key,
                    {
                        "paper_split": split,
                        "turn_keys": set(),
                        "intents": set(),
                        "slot_types": set(),
                    },
                )
                if record["paper_split"] != split:
                    raise ValueError(f"MultiDoGO conversation crosses paper splits: {key!r}")
                turn_key = (int(row["turn_number"]), str(row["utterance_id"]))
                if turn_key in record["turn_keys"]:
                    raise ValueError(f"duplicate MultiDoGO annotated turn: {key!r} {turn_key!r}")
                record["turn_keys"].add(turn_key)
                record["intents"].update(str(value) for value in row["intents"])
                record["slot_types"].update(
                    value
                    for label in row["slot_labels"]
                    if (value := slot_type(str(label))) is not None
                )

    index: dict[tuple[str, str], dict[str, object]] = {}
    for key, record in mutable.items():
        intents = sorted(str(value) for value in record["intents"])
        slot_types = sorted(str(value) for value in record["slot_types"])
        ontology_tokens = set().union(*(label_tokens(value) for value in intents + slot_types))
        sensitive = sorted(ontology_tokens & SENSITIVE_CONCEPTS)
        index[key] = {
            "paper_split": record["paper_split"],
            "annotated_customer_turns": len(record["turn_keys"]),
            "intents": intents,
            "slot_types": slot_types,
            "sensitive_concepts": sensitive,
            "hard_negative_score": len(sensitive),
        }
    return index


def validate_audit_report(repository: Path, report: dict[str, Any]) -> None:
    if (
        report.get("artifact_schema_version") != 1
        or report.get("revision") != REVISION
        or report.get("annotation_tree_git_oid") != ANNOTATION_TREE_GIT_OID
        or report.get("contains_source_text") is not False
    ):
        raise ValueError("MultiDoGO annotation audit does not match the curriculum contract")
    alignment = report.get("alignment")
    if not isinstance(alignment, dict) or alignment != {
        "all_annotations_join_to_pinned_unannotated_turns": True,
        "annotated_rows_are_customer_turns": True,
        "turn_text_matches_source": True,
        "sentence_text_occurs_in_source_turn": True,
        "conversation_crosses_paper_splits": False,
    }:
        raise ValueError("MultiDoGO annotation alignment audit did not pass exactly")
    files = report.get("files")
    if not isinstance(files, dict):
        raise ValueError("MultiDoGO annotation audit lacks file identities")
    expected = {str(path.relative_to(repository)) for path in annotation_paths(repository)}
    if set(files) != expected:
        raise ValueError("MultiDoGO annotation audit file set is incomplete")
    for relative in sorted(expected):
        metadata = files[relative]
        path = repository / relative
        if not isinstance(metadata, dict) or metadata.get("sha256") != file_sha256(path):
            raise ValueError(f"MultiDoGO annotation differs from its audit: {relative}")


def artifact_rows(
    manifest: dict[str, Any], artifact: str, source_directory: Path
) -> list[dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(artifact), dict):
        raise ValueError(f"MultiDoGO source manifest lacks {artifact}")
    metadata = artifacts[artifact]
    path = source_directory / Path(str(metadata["path"])).name
    if file_sha256(path) != metadata.get("sha256"):
        raise ValueError(f"MultiDoGO {artifact} differs from its source manifest")
    rows = read_jsonl(path)
    if len(rows) != metadata.get("rows"):
        raise ValueError(f"MultiDoGO {artifact} row count differs from its source manifest")
    return rows


def enrich_rows(
    rows: list[dict[str, object]],
    annotation_index: dict[tuple[str, str], dict[str, object]],
    paper_splits: set[str],
    audit_sha256: str,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["source_domain"]), str(row["source_record_id"]))
        annotation = annotation_index.get(key)
        if annotation is None or annotation.get("paper_split") not in paper_splits:
            continue
        enriched.append(
            row
            | {
                "publisher_annotation_granularity": "turn",
                "publisher_annotation_split": annotation["paper_split"],
                "publisher_annotated_customer_turns": annotation["annotated_customer_turns"],
                "publisher_intents": annotation["intents"],
                "publisher_slot_types": annotation["slot_types"],
                "publisher_sensitive_concepts": annotation["sensitive_concepts"],
                "annotation_hard_negative_score": annotation["hard_negative_score"],
                "annotation_stratum": (
                    "sensitive_service"
                    if int(annotation["hard_negative_score"]) > 0
                    else "routine_service"
                ),
                "annotation_label_scope": (
                    "publisher intent and slot labels; SAFE remains a legitimate-domain weak label"
                ),
                "annotation_audit_sha256": audit_sha256,
            }
        )
    return sorted(enriched, key=lambda row: str(row["id"]))


def build(
    repository: Path,
    source_directory: Path,
    audit_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite annotation curriculum: {output}")
    source_identity = verify_repository(repository, require_annotations=True)
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_audit_report(repository, audit_report)
    audit_sha256 = file_sha256(audit_path)

    source_manifest_path = source_directory / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("source") != SOURCE
        or source_manifest.get("license") != LICENSE
        or source_manifest.get("revision") != REVISION
    ):
        raise ValueError("MultiDoGO source derivative differs from the curriculum contract")
    real_train = artifact_rows(source_manifest, "real_train", source_directory)
    call_validation = artifact_rows(source_manifest, "call_validation", source_directory)
    annotation_index = build_annotation_index(repository)
    train = enrich_rows(real_train, annotation_index, {"train"}, audit_sha256)
    dev = enrich_rows(call_validation, annotation_index, {"dev"}, audit_sha256)
    test = enrich_rows(call_validation, annotation_index, {"test"}, audit_sha256)
    if not train or not dev or not test:
        raise ValueError(
            "annotation curriculum requires non-empty train, dev, and test intersections"
        )

    split_rows = {"train": train, "dev": dev, "test": test}
    split_families = {
        split: {str(row["family_id"]) for row in rows}
        for split, rows in split_rows.items()
    }
    if (
        split_families["train"] & split_families["dev"]
        or split_families["train"] & split_families["test"]
        or split_families["dev"] & split_families["test"]
    ):
        raise ValueError("annotation curriculum conversation family crosses output splits")
    ids = [str(row["id"]) for rows in split_rows.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("annotation curriculum row ID crosses output splits")

    output.mkdir(parents=True)
    paths = {split: output / f"{split}.jsonl" for split in split_rows}
    for split, rows in split_rows.items():
        write_jsonl(paths[split], rows)
    manifest: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source": SOURCE,
        "repository": source_identity["repository"],
        "revision": REVISION,
        "license": LICENSE,
        "annotation_tree_git_oid": ANNOTATION_TREE_GIT_OID,
        "annotation_audit_path": str(audit_path),
        "annotation_audit_sha256": audit_sha256,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "policy": {
            "publisher_annotations_are_intent_and_slot_labels_only": True,
            "publisher_annotations_are_not_independent_scam_labels": True,
            "safe_label_remains_weak_legitimate_service_domain_label": True,
            "only_turn_level_annotations_select_model_rows": True,
            "sentence_level_annotations_are_alignment_audit_only": True,
            "existing_source_train_validation_boundary_preserved": True,
            "paper_train_rows_enter_fitting": True,
            "paper_dev_test_rows_enter_fitting": False,
            "raw_text_added_beyond_existing_derivative": False,
            "direct_reddit_scrape": False,
            "model_rows_redistributed": False,
        },
        "counts": {
            split: {
                "rows": len(rows),
                "families": len(split_families[split]),
                "by_domain": dict(Counter(str(row["source_domain"]) for row in rows)),
                "by_annotation_stratum": dict(
                    Counter(str(row["annotation_stratum"]) for row in rows)
                ),
                "by_source_window": dict(Counter(str(row["source_window"]) for row in rows)),
            }
            for split, rows in split_rows.items()
        },
        "artifacts": {
            split: {
                "path": str(path),
                "rows": len(split_rows[split]),
                "sha256": file_sha256(path),
            }
            for split, path in paths.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository", type=Path, default=Path("data/raw/multidogo/repository")
    )
    parser.add_argument(
        "--source-directory", type=Path, default=Path("data/external/multidogo")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("reports/data/multidogo_annotation_audit.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/multidogo_annotated")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.repository, args.source_directory, args.audit, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
