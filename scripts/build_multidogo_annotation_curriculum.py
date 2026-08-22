#!/usr/bin/env python3
"""Build a split-safe hard-negative curriculum from audited MultiDoGO annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scamguard.metrics import file_sha256

try:
    from scripts.audit_multidogo_annotations import (
        TURN_GRANULARITY,
        read_annotation_file,
        slot_type,
    )
    from scripts.build_multidogo_dialogues import LICENSE, SOURCE
    from scripts.build_schema19_call_windows import write_jsonl
    from scripts.build_taskmaster_hard_negatives import privacy_normalize
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
        TURN_GRANULARITY,
        read_annotation_file,
        slot_type,
    )
    from build_multidogo_dialogues import LICENSE, SOURCE  # type: ignore[no-redef]
    from build_schema19_call_windows import write_jsonl  # type: ignore[no-redef]
    from build_taskmaster_hard_negatives import (  # type: ignore[no-redef]
        privacy_normalize,
    )
    from fetch_multidogo import (  # type: ignore[no-redef]
        ANNOTATION_SPLITS,
        ANNOTATION_TREE_GIT_OID,
        DOMAINS,
        REVISION,
        annotation_paths,
        verify_repository,
    )

ARTIFACT_SCHEMA_VERSION = 2
SELECTION_SALT = "scamguard-multidogo-publisher-annotations-v2"
SPLIT_CAPS_PER_DOMAIN = {"train": 200, "dev": 90, "test": 200}
MIN_TEXT_CHARACTERS = 8
MAX_TEXT_CHARACTERS = 500
LABEL_TOKEN_RE = re.compile(r"[a-z0-9]+")
SHORT_NUMBER_RE = re.compile(r"\b\d{3,}\b")
SENSITIVE_CONCEPTS = {
    "account",
    "address",
    "bank",
    "balance",
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
    "ssn",
    "subscription",
    "transfer",
    "username",
    "verification",
    "verify",
}
PII_SLOT_TYPES = {
    "account_id",
    "account_number",
    "address",
    "approver_name",
    "booking_confirmation_number",
    "card_number",
    "claimid",
    "company_name",
    "email_address",
    "name",
    "password",
    "phone_number",
    "phonenumber",
    "policyid",
    "ssn",
    "target_account_number",
    "username",
}


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def label_tokens(value: str) -> set[str]:
    """Extract both lexical tokens and known concepts from publisher ontology labels."""
    flattened = "".join(LABEL_TOKEN_RE.findall(value.casefold()))
    tokens = set(LABEL_TOKEN_RE.findall(value.casefold()))
    return tokens | {concept for concept in SENSITIVE_CONCEPTS if concept in flattened}


def normalize_annotation_text(value: str) -> tuple[str, bool]:
    normalized = privacy_normalize(value)
    normalized = SHORT_NUMBER_RE.sub("<NUMBER>", normalized)
    normalized = " ".join(normalized.split())
    return normalized, normalized != " ".join(value.split())


def annotation_candidate(row: dict[str, Any], audit_sha256: str) -> dict[str, object] | None:
    if row.get("empty_utterance") is True:
        return None
    intents = sorted(str(value) for value in row["intents"])
    slot_types = sorted(
        {
            value
            for label in row["slot_labels"]
            if (value := slot_type(str(label))) is not None
        }
    )
    if set(slot_types) & PII_SLOT_TYPES:
        return None
    text, privacy_changed = normalize_annotation_text(str(row["utterance"]))
    if not MIN_TEXT_CHARACTERS <= len(text) <= MAX_TEXT_CHARACTERS:
        return None
    ontology_tokens = set().union(*(label_tokens(value) for value in intents + slot_types))
    sensitive = sorted(ontology_tokens & SENSITIVE_CONCEPTS)
    conversation_id = str(row["conversation_id"])
    domain = str(row["domain"])
    turn_number = int(row["turn_number"])
    identity = f"{TURN_GRANULARITY}:{domain}:{conversation_id}:{turn_number}"
    return {
        "id": "multidogo-annotated-turn-" + short_hash(identity),
        "text": f"CUSTOMER: {text}",
        "label": "SAFE",
        "category": "NONE",
        "source": SOURCE,
        "source_label": f"publisher_customer_turn:{domain}",
        "license": LICENSE,
        "split": "train" if row["split"] == "train" else "validation",
        "family_id": f"multidogo-annotated:{domain}:{conversation_id}",
        "is_synthetic": False,
        "label_policy": "publisher_legitimate_service_domain_weak_safe_label",
        "source_language": "English",
        "source_record_id": conversation_id,
        "source_domain": domain,
        "source_revision": REVISION,
        "source_turn_number": turn_number,
        "source_window": "publisher_annotated_customer_turn",
        "context_policy": "one_highest_risk_eligible_customer_turn_per_conversation",
        "provenance_class": "human_customer_and_trained_agent_roleplay",
        "naturally_occurring_communication": False,
        "privacy_normalization": (
            "email_url_phone_account_and_three_plus_digit_values_replaced; "
            "rows with publisher PII slot types excluded"
        ),
        "privacy_values_replaced": privacy_changed,
        "publisher_annotation_granularity": "turn",
        "publisher_annotation_split": row["split"],
        "publisher_intents": intents,
        "publisher_slot_types": slot_types,
        "publisher_sensitive_concepts": sensitive,
        "annotation_hard_negative_score": len(sensitive),
        "annotation_stratum": "sensitive_service" if sensitive else "routine_service",
        "annotation_label_scope": (
            "publisher intent and slot labels; SAFE remains a legitimate-domain weak label"
        ),
        "annotation_audit_sha256": audit_sha256,
    }


def candidate_rank(row: dict[str, object], split: str, domain: str) -> tuple[object, ...]:
    return (
        -int(row["annotation_hard_negative_score"]),
        short_hash(f"{SELECTION_SALT}:{split}:{domain}:{row['id']}", 64),
    )


def build_candidate_pools(
    repository: Path, audit_sha256: str
) -> tuple[dict[str, dict[str, list[dict[str, object]]]], dict[str, object]]:
    pools: dict[str, dict[str, list[dict[str, object]]]] = {
        split: {domain: [] for domain in DOMAINS} for split in ANNOTATION_SPLITS
    }
    stats: Counter[str] = Counter()
    for split in ANNOTATION_SPLITS:
        for domain in DOMAINS:
            path = (
                repository
                / "data"
                / "paper_splits"
                / TURN_GRANULARITY
                / domain
                / f"{split}.tsv"
            )
            per_conversation: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
            for source_row in read_annotation_file(path, TURN_GRANULARITY, domain, split):
                stats[f"{split}:{domain}:source_rows"] += 1
                candidate = annotation_candidate(source_row, audit_sha256)
                if candidate is None:
                    stats[f"{split}:{domain}:ineligible_rows"] += 1
                    continue
                per_conversation[str(source_row["conversation_id"])].append(candidate)
            for candidates in per_conversation.values():
                candidates.sort(key=lambda row: candidate_rank(row, split, domain))
                pools[split][domain].append(candidates[0])
            pools[split][domain].sort(key=lambda row: candidate_rank(row, split, domain))
            stats[f"{split}:{domain}:eligible_families"] = len(pools[split][domain])
    return pools, dict(sorted(stats.items()))


def select_rows(
    pools: dict[str, dict[str, list[dict[str, object]]]],
    caps: dict[str, int] | None = None,
) -> dict[str, list[dict[str, object]]]:
    caps = caps or SPLIT_CAPS_PER_DOMAIN
    selected: dict[str, list[dict[str, object]]] = {split: [] for split in ANNOTATION_SPLITS}
    used_texts: set[str] = set()
    # Reserve evaluation text first, then ensure fitting rows cannot exactly overlap it.
    for split in ("test", "dev", "train"):
        for domain in DOMAINS:
            admitted: list[dict[str, object]] = []
            for row in pools[split][domain]:
                normalized = " ".join(str(row["text"]).casefold().split())
                if normalized in used_texts:
                    continue
                used_texts.add(normalized)
                admitted.append(row)
                if len(admitted) == caps[split]:
                    break
            if len(admitted) != caps[split]:
                raise ValueError(
                    f"MultiDoGO {split}/{domain} has {len(admitted)} unique eligible "
                    f"families; required {caps[split]}"
                )
            selected[split].extend(admitted)
    return {
        split: sorted(rows, key=lambda row: str(row["id"]))
        for split, rows in selected.items()
    }


def validate_audit_report(repository: Path, report: dict[str, Any]) -> None:
    if (
        report.get("artifact_schema_version") != 2
        or report.get("revision") != REVISION
        or report.get("annotation_tree_git_oid") != ANNOTATION_TREE_GIT_OID
        or report.get("contains_source_text") is not False
    ):
        raise ValueError("MultiDoGO annotation audit does not match the curriculum contract")
    contracts = report.get("contracts")
    if not isinstance(contracts, dict) or contracts != {
        "publisher_readme_describes_paper_splits_as_customer_turns": True,
        "annotation_identities_unique_within_granularity": True,
        "conversations_do_not_cross_splits_within_granularity": True,
        "empty_annotation_utterances_quarantined_from_curriculum": True,
        "turn_level_rows_are_selected_directly": True,
        "sentence_level_rows_are_audit_only": True,
        "annotated_and_unannotated_conversation_ids_are_separate_collections": True,
        "paper_dev_test_rows_enter_fitting": False,
    }:
        raise ValueError("MultiDoGO annotation integrity audit did not pass exactly")
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


def build(
    repository: Path,
    audit_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite annotation curriculum: {output}")
    source_identity = verify_repository(repository, require_annotations=True)
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_audit_report(repository, audit_report)
    audit_sha256 = file_sha256(audit_path)
    pools, pool_stats = build_candidate_pools(repository, audit_sha256)
    split_rows = select_rows(pools)
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
        "selection_salt": SELECTION_SALT,
        "selection_caps_per_domain": SPLIT_CAPS_PER_DOMAIN,
        "policy": {
            "publisher_annotations_are_intent_and_slot_labels_only": True,
            "publisher_annotations_are_not_independent_scam_labels": True,
            "safe_label_remains_weak_legitimate_service_domain_label": True,
            "only_turn_level_annotations_select_model_rows": True,
            "sentence_level_annotations_are_audit_only": True,
            "publisher_paper_split_boundary_preserved": True,
            "paper_train_rows_enter_fitting": True,
            "paper_dev_test_rows_enter_fitting": False,
            "one_row_per_conversation_family": True,
            "publisher_pii_slot_rows_excluded": True,
            "privacy_normalization_applied_before_materialization": True,
            "direct_reddit_scrape": False,
            "model_rows_redistributed": False,
        },
        "pool_counts": pool_stats,
        "counts": {
            split: {
                "rows": len(rows),
                "families": len(split_families[split]),
                "by_domain": dict(Counter(str(row["source_domain"]) for row in rows)),
                "by_annotation_stratum": dict(
                    Counter(str(row["annotation_stratum"]) for row in rows)
                ),
                "privacy_values_replaced": sum(
                    row["privacy_values_replaced"] is True for row in rows
                ),
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
        "--audit", type=Path, default=Path("reports/data/multidogo_annotation_audit.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/multidogo_annotated")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.repository, args.audit, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
