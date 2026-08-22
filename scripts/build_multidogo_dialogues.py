#!/usr/bin/env python3
"""Build bounded, dialogue-disjoint MultiDoGO service-call evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256

try:
    from scripts.audit_source_overlap import near_overlap_signatures
    from scripts.build_dataset import (
        cluster_near_duplicates,
        family_skeleton,
        normalized,
        simhash64,
    )
    from scripts.build_schema19_call_windows import write_jsonl
    from scripts.build_taskmaster_hard_negatives import privacy_normalize
    from scripts.fetch_multidogo import DOMAINS, LICENSE_SHA256, REVISION, verify_repository
    from scripts.generate_call_action_states import CONTRAST_STATES, TARGET_KEYS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from audit_source_overlap import near_overlap_signatures  # type: ignore[no-redef]
    from build_dataset import (  # type: ignore[no-redef]
        cluster_near_duplicates,
        family_skeleton,
        normalized,
        simhash64,
    )
    from build_schema19_call_windows import write_jsonl  # type: ignore[no-redef]
    from build_taskmaster_hard_negatives import privacy_normalize  # type: ignore[no-redef]
    from fetch_multidogo import (  # type: ignore[no-redef]
        DOMAINS,
        LICENSE_SHA256,
        REVISION,
        verify_repository,
    )
    from generate_call_action_states import (  # type: ignore[no-redef]
        CONTRAST_STATES,
        TARGET_KEYS,
    )

SOURCE = "multidogo_human_service_dialogues"
STATE_SOURCE = "scamguard_multidogo_grounded_evidence_states_v1"
LICENSE = "CDLA-Permissive-1.0"
EXPECTED_HEADER = ["conversationId", "turnNumber", "utteranceId", "utterance", "authorRole"]
TRAIN_CONVERSATIONS_PER_DOMAIN = 150
VALIDATION_CONVERSATIONS_PER_DOMAIN = 75
TRAIN_STATE_DOMAINS = ("airline", "fastfood", "finance", "media")
VALIDATION_STATE_DOMAINS = ("insurance", "software")
TRAIN_STATE_FAMILIES_PER_DOMAIN = 75
VALIDATION_STATE_FAMILIES_PER_DOMAIN = 75
CANDIDATE_POOL_PER_DOMAIN = 3000
MAX_CONTEXT_CHARS = 1400
MAX_STATE_CONTEXT_CHARS = 650
MIN_TURNS = 6
MAX_TURNS = 32
PARTITION_SALT = "scamguard-multidogo-evidence-v1"

SENSITIVE_RE = re.compile(
    r"\b(account|address|bank|balance|billing|card|claim|code|credit|deductible|deposit|"
    r"download|email|identity|install|insurance|license|loan|order|password|payment|phone|"
    r"policy|purchase|refund|routing|security|social security|subscription|transfer|verify)\b",
    re.IGNORECASE,
)
REQUEST_RE = re.compile(
    r"\b(can you|could you|do you have|give me|may i have|need you to|please|provide|read me|"
    r"send|share|tell me|what is|what's|would you)\b",
    re.IGNORECASE,
)
IRREVERSIBLE_RE = re.compile(
    r"\b(book|buy|cancel|charge|confirm the order|deposit|pay|place the order|purchase|"
    r"submit|transfer)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MIXED_ACCOUNT_RE = re.compile(
    r"\b(?=[A-Za-z0-9]{10,}\b)(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)"
    r"[A-Za-z0-9]+\b"
)

VERIFIED_ENDINGS = {
    "airline": (
        "AGENT: Review the itinerary in the official airline app you open yourself; "
        "I will not ask for a password or one-time code."
    ),
    "fastfood": (
        "AGENT: You can confirm the order in the restaurant's official app or at the "
        "counter; do not use a payment link from this chat."
    ),
    "finance": (
        "AGENT: End this conversation and open your normal bank app or call the number "
        "on your card to verify; never read a one-time code to me."
    ),
    "insurance": (
        "AGENT: Verify the claim through the insurer's official app or the number on your "
        "policy; I will not direct you to an outside payment account."
    ),
    "media": (
        "AGENT: Open the service's official app yourself to review the subscription; "
        "I will not ask for a password or security code."
    ),
    "software": (
        "AGENT: Use the vendor's official support page that you navigate to yourself; "
        "do not install remote-control software from a caller."
    ),
}
HARMFUL_ENDINGS = {
    "airline": (
        "AGENT: The booking will be cancelled unless you buy gift cards now and read me "
        "every code while I keep you on the line."
    ),
    "fastfood": (
        "AGENT: To release the order, pay the fee to the personal account I provide and "
        "send me the transfer confirmation right now."
    ),
    "finance": (
        "AGENT: Move the balance to the safe account number I give you and read me the "
        "one-time code immediately; do not call the bank."
    ),
    "insurance": (
        "AGENT: Your claim closes today unless you wire the refundable fee to my account "
        "now and give me the confirmation code."
    ),
    "media": (
        "AGENT: Keep this confidential, buy a gift card for the subscription charge, "
        "and read me the code before we disconnect."
    ),
    "software": (
        "AGENT: Install the remote-control app I name, give me screen access, and read me "
        "the security code right now."
    ),
}
UNRESOLVED_ENDING = (
    "AGENT: I see an unverified service note, but I cannot confirm what action it requires. "
    "Do not send money or disclose a code until it is independently checked."
)
SHARED_CONTINUATION = (
    "CUSTOMER: I am going to pause and check my own records before deciding.\n"
    "AGENT: Of course. Take your time and do not act until you are comfortable."
)


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def normalize_source_text(value: str) -> str:
    return MIXED_ACCOUNT_RE.sub("<ACCOUNT_ID>", privacy_normalize(value))


def read_domain(path: Path, domain: str) -> list[dict[str, object]]:
    conversations: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"unexpected MultiDoGO {domain} header: {reader.fieldnames!r}")
        for row in reader:
            if row["authorRole"] not in {"agent", "customer"}:
                raise ValueError(f"unexpected MultiDoGO role: {row['authorRole']!r}")
            conversation_id = row["conversationId"].strip()
            if not conversation_id:
                raise ValueError("MultiDoGO row lacks a conversation ID")
            conversations[conversation_id].append(row)
    output = []
    for conversation_id, rows in conversations.items():
        rows.sort(key=lambda row: int(row["turnNumber"]))
        turn_numbers = [int(row["turnNumber"]) for row in rows]
        if len(turn_numbers) != len(set(turn_numbers)):
            continue
        turns = [
            {
                "role": row["authorRole"],
                "text": " ".join(row["utterance"].split()),
                "turn": int(row["turnNumber"]),
            }
            for row in rows
            if row["utterance"].strip()
        ]
        if not quality_dialogue(turns):
            continue
        output.append({"id": conversation_id, "domain": domain, "turns": turns})
    return output


def quality_dialogue(turns: list[dict[str, object]]) -> bool:
    if not MIN_TURNS <= len(turns) <= MAX_TURNS:
        return False
    roles = Counter(str(turn["role"]) for turn in turns)
    if roles["agent"] < 2 or roles["customer"] < 2:
        return False
    utterances = [normalized(str(turn["text"])) for turn in turns]
    nonempty = [value for value in utterances if value]
    if len(nonempty) < MIN_TURNS or len(set(nonempty)) / len(nonempty) < 0.6:
        return False
    words = TOKEN_RE.findall(" ".join(str(turn["text"]) for turn in turns))
    if len(words) < 35:
        return False
    adjacent_role_changes = sum(
        turns[index]["role"] != turns[index - 1]["role"] for index in range(1, len(turns))
    )
    return adjacent_role_changes / (len(turns) - 1) >= 0.65


def render_lines(turns: list[dict[str, object]]) -> list[str]:
    return [
        f"{'AGENT' if turn['role'] == 'agent' else 'CUSTOMER'}: "
        f"{normalize_source_text(str(turn['text']))}"
        for turn in turns
        if str(turn["text"]).strip()
    ]


def latest_lines(lines: list[str], max_chars: int) -> str:
    selected: list[str] = []
    total = 0
    for line in reversed(lines):
        addition = len(line) + int(bool(selected))
        if selected and total + addition > max_chars:
            break
        if not selected and len(line) > max_chars:
            line = line[: max_chars - 1].rstrip() + "…"
            addition = len(line)
        selected.append(line)
        total += addition
    return "\n".join(reversed(selected))


def earliest_lines(lines: list[str], max_chars: int) -> str:
    selected: list[str] = []
    total = 0
    for line in lines:
        addition = len(line) + int(bool(selected))
        if selected and total + addition > max_chars:
            break
        if not selected and len(line) > max_chars:
            line = line[: max_chars - 1].rstrip() + "…"
            addition = len(line)
        selected.append(line)
        total += addition
    return "\n".join(selected)


def agent_turn_score(text: str) -> tuple[int, int]:
    sensitive = len(SENSITIVE_RE.findall(text))
    requests = len(REQUEST_RE.findall(text))
    irreversible = len(IRREVERSIBLE_RE.findall(text))
    return sensitive * 5 + requests * 3 + irreversible * 4, len(text)


def select_agent_turn(turns: list[dict[str, object]]) -> tuple[int, str]:
    candidates = [
        (index, normalize_source_text(str(turn["text"])))
        for index, turn in enumerate(turns)
        if turn["role"] == "agent" and len(TOKEN_RE.findall(str(turn["text"]))) >= 4
    ]
    if not candidates:
        raise ValueError("eligible MultiDoGO dialogue lacks a usable agent turn")
    return max(candidates, key=lambda item: agent_turn_score(item[1]))


def assign_unique_agent_turns(groups: list[list[dict[str, object]]]) -> None:
    """Choose an informative agent turn without exact repetition across admitted splits."""
    used: set[str] = set()
    selected = sorted(
        (row for group in groups for row in group),
        key=lambda row: short_hash(
            f"{PARTITION_SALT}:unique-agent-turn:{row['source_record_id']}", length=64
        ),
    )
    for row in selected:
        candidates = [
            (index, normalize_source_text(str(turn["text"])))
            for index, turn in enumerate(list(row["turns"]))
            if turn["role"] == "agent" and len(TOKEN_RE.findall(str(turn["text"]))) >= 4
        ]
        candidates.sort(key=lambda item: agent_turn_score(item[1]), reverse=True)
        choice = next((item for item in candidates if normalized(item[1]) not in used), None)
        if choice is None:
            raise RuntimeError("MultiDoGO conversation lacks a globally unique agent turn")
        row["selected_agent_index"], row["selected_agent_text"] = choice
        used.add(normalized(choice[1]))


def base_action_targets(agent_text: str) -> dict[str, bool]:
    sensitive = bool(SENSITIVE_RE.search(agent_text))
    targets = {
        "sensitive_action_language": sensitive,
        "requested_disclosure_or_transfer": sensitive and bool(REQUEST_RE.search(agent_text)),
        "caller_controls_target": False,
        "official_self_navigation": False,
        "independent_verification": False,
        "pressure_or_secrecy": False,
        "irreversible_action": bool(IRREVERSIBLE_RE.search(agent_text)),
    }
    if tuple(targets) != TARGET_KEYS:
        raise RuntimeError("MultiDoGO action target schema changed")
    return targets


def state_action_targets(state: str, routine: dict[str, bool], domain: str) -> dict[str, bool]:
    if state == "routine_safe":
        return dict(routine)
    if state == "verified_safe":
        return {
            "sensitive_action_language": True,
            "requested_disclosure_or_transfer": False,
            "caller_controls_target": False,
            "official_self_navigation": True,
            "independent_verification": True,
            "pressure_or_secrecy": False,
            "irreversible_action": False,
        }
    if state == "unresolved":
        return {
            "sensitive_action_language": True,
            "requested_disclosure_or_transfer": False,
            "caller_controls_target": False,
            "official_self_navigation": False,
            "independent_verification": False,
            "pressure_or_secrecy": False,
            "irreversible_action": False,
        }
    if state == "harmful_scam":
        return {
            "sensitive_action_language": True,
            "requested_disclosure_or_transfer": True,
            "caller_controls_target": True,
            "official_self_navigation": False,
            "independent_verification": False,
            "pressure_or_secrecy": True,
            "irreversible_action": domain != "software",
        }
    raise ValueError(f"unknown state: {state}")


def conversation_candidate(dialogue: dict[str, object]) -> dict[str, object]:
    turns = list(dialogue["turns"])
    lines = render_lines(turns)
    text = latest_lines(lines, MAX_CONTEXT_CHARS)
    agent_index, agent_text = select_agent_turn(turns)
    privacy_changed = any(
        normalize_source_text(str(turn["text"])) != " ".join(str(turn["text"]).split())
        for turn in turns
    )
    source_id = str(dialogue["id"])
    domain = str(dialogue["domain"])
    return {
        "id": "multidogo-candidate-" + short_hash(source_id),
        "text": text,
        "label": "SAFE",
        "category": "NONE",
        "source": SOURCE,
        "source_label": f"legitimate_human_service_dialogue:{domain}",
        "license": LICENSE,
        "split": "candidate",
        "family_id": f"multidogo:{source_id}",
        "is_synthetic": False,
        "source_language": "English",
        "source_record_id": source_id,
        "source_domain": domain,
        "turns": turns,
        "selected_agent_index": agent_index,
        "selected_agent_text": agent_text,
        "privacy_values_replaced": privacy_changed,
    }


def select_conversations(
    candidates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    candidate_domains: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        candidate_domains[str(row["source_domain"])].append(row)
    pooled: list[dict[str, object]] = []
    for domain in DOMAINS:
        ranked = sorted(
            candidate_domains[domain],
            key=lambda row: short_hash(
                f"{PARTITION_SALT}:audit-pool:{domain}:{row['source_record_id']}", length=64
            ),
        )
        if len(ranked) < CANDIDATE_POOL_PER_DOMAIN:
            raise RuntimeError(f"MultiDoGO {domain} lacks the frozen audit-pool size")
        pooled.extend(ranked[:CANDIDATE_POOL_PER_DOMAIN])
    exact: dict[str, dict[str, object]] = {}
    for row in pooled:
        key = normalized(str(row["text"]))
        incumbent = exact.get(key)
        if incumbent is None or str(row["source_record_id"]) < str(incumbent["source_record_id"]):
            exact[key] = row
    exact_removed = len(pooled) - len(exact)
    clustered, conflicts, stats = cluster_near_duplicates(list(exact.values()))
    if conflicts:
        raise RuntimeError("all-SAFE MultiDoGO near-template clusters cannot conflict")
    representatives: dict[str, dict[str, object]] = {}
    for row in clustered:
        family = str(row["family_id"])
        incumbent = representatives.get(family)
        if incumbent is None or str(row["source_record_id"]) < str(
            incumbent["source_record_id"]
        ):
            representatives[family] = row

    train: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    by_domain: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in representatives.values():
        by_domain[str(row["source_domain"])].append(row)
    for domain in DOMAINS:
        ranked = sorted(
            by_domain[domain],
            key=lambda row: short_hash(
                f"{PARTITION_SALT}:{domain}:{row['family_id']}", length=64
            ),
        )
        required = TRAIN_CONVERSATIONS_PER_DOMAIN + VALIDATION_CONVERSATIONS_PER_DOMAIN
        if len(ranked) < required:
            raise RuntimeError(
                f"MultiDoGO {domain} has {len(ranked)} clean families; requires {required}"
            )
        train.extend(ranked[:TRAIN_CONVERSATIONS_PER_DOMAIN])
        validation.extend(ranked[TRAIN_CONVERSATIONS_PER_DOMAIN:required])
    stats = dict(stats)
    stats.update(
        {
            "deterministic_candidate_pool": len(pooled),
            "candidate_pool_per_domain": CANDIDATE_POOL_PER_DOMAIN,
            "exact_duplicate_conversations_removed": exact_removed,
            "same_label_near_template_rows_collapsed": len(clustered) - len(representatives),
            "near_template_representatives": len(representatives),
        }
    )
    return train, validation, stats


def real_rows(selected: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in selected:
        source_id = str(source["source_record_id"])
        domain = str(source["source_domain"])
        family_id = f"multidogo:{source_id}"
        actions = base_action_targets(str(source["selected_agent_text"]))
        common = {
            "label": "SAFE",
            "category": "NONE",
            "source": SOURCE,
            "source_label": f"legitimate_human_service_dialogue:{domain}",
            "license": LICENSE,
            "split": split,
            "family_id": family_id,
            "is_synthetic": False,
            "source_language": "English",
            "source_record_id": source_id,
            "source_domain": domain,
            "source_revision": REVISION,
            "near_template_family_id": source["family_id"],
            "provenance_class": "human_customer_and_trained_agent_roleplay",
            "naturally_occurring_communication": False,
            "privacy_normalization": "email_url_and_phone_or_account_like_values_replaced",
            "privacy_values_replaced": bool(source["privacy_values_replaced"]),
            "action_targets": actions,
            "action_label_method": "high_precision_lexical_weak_supervision_v1",
            "action_verdict_weight": 0.5,
        }
        output.append(
            common
            | {
                "id": "multidogo-context-" + short_hash(source_id),
                "text": source["text"],
                "source_window": "recent_complete_turns",
                "context_policy": f"latest_complete_turns_capped_at_{MAX_CONTEXT_CHARS}_characters",
            }
        )
        output.append(
            common
            | {
                "id": "multidogo-agent-turn-" + short_hash(source_id),
                "text": f"AGENT: {source['selected_agent_text']}",
                "source_window": "highest_risk_agent_turn",
                "context_policy": "single_agent_turn_ranked_by_frozen_sensitive_action_lexicon",
            }
        )
    return sorted(output, key=lambda row: str(row["id"]))


def state_rows(
    selected: list[dict[str, object]], split: str, domains: tuple[str, ...], per_domain: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_domain: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_domain[str(row["source_domain"])].append(row)
    for domain in domains:
        ranked = sorted(
            by_domain[domain],
            key=lambda row: short_hash(
                f"{PARTITION_SALT}:states:{domain}:{row['source_record_id']}", length=64
            ),
        )[:per_domain]
        for source in ranked:
            source_id = str(source["source_record_id"])
            turns = list(source["turns"])
            selected_index = int(source["selected_agent_index"])
            prefix = latest_lines(render_lines(turns[:selected_index]), MAX_STATE_CONTEXT_CHARS)
            source_continuation = earliest_lines(
                render_lines(turns[selected_index + 1 :]), MAX_STATE_CONTEXT_CHARS
            )
            routine = f"AGENT: {source['selected_agent_text']}"
            endings = {
                "routine_safe": routine,
                "verified_safe": VERIFIED_ENDINGS[domain],
                "unresolved": UNRESOLVED_ENDING,
                "harmful_scam": HARMFUL_ENDINGS[domain],
            }
            family_id = f"multidogo:{source_id}"
            contrast_id = "multidogo-evidence-state-" + short_hash(f"v1:{source_id}")
            routine_targets = base_action_targets(str(source["selected_agent_text"]))
            shared_frame = "\n".join(
                part
                for part in (
                    prefix,
                    "<DECISIVE_AGENT_ACTION>",
                    source_continuation,
                    SHARED_CONTINUATION,
                )
                if part
            )
            shared_prefix_hash = hashlib.sha256(shared_frame.encode()).hexdigest()
            for state in CONTRAST_STATES:
                label = {
                    "routine_safe": "SAFE",
                    "verified_safe": "SAFE",
                    "unresolved": "UNCERTAIN",
                    "harmful_scam": "SCAM",
                }[state]
                text = "\n".join(
                    part
                    for part in (
                        prefix,
                        endings[state],
                        source_continuation,
                        SHARED_CONTINUATION,
                    )
                    if part
                )
                output.append(
                    {
                        "id": f"{contrast_id}-{state}",
                        "text": text,
                        "label": label,
                        "category": (
                            "CREDENTIAL_THEFT"
                            if state == "harmful_scam" and domain == "software"
                            else ("FINANCIAL" if state == "harmful_scam" else "NONE")
                        ),
                        "source": STATE_SOURCE,
                        "source_label": f"human_grounded_evidence_state:{state}:{domain}",
                        "license": LICENSE,
                        "split": split,
                        "family_id": family_id,
                        "contrast_id": contrast_id,
                        "contrast_state": state,
                        "is_synthetic": True,
                        "synthetic_method": (
                            "decisive_agent_turn_transformation_with_shared_delayed_continuation_v1"
                        ),
                        "source_language": "English",
                        "source_record_id": source_id,
                        "source_domain": domain,
                        "source_revision": REVISION,
                        "near_template_family_id": source["family_id"],
                        "action_targets": state_action_targets(state, routine_targets, domain),
                        "shared_context_sha256": shared_prefix_hash,
                        "decisive_action_precedes_shared_continuation": True,
                        "human_grounded": True,
                        "external_benchmark_text_copied": False,
                        "pattern_reference": "https://consumer.ftc.gov/articles/how-avoid-scam",
                    }
                )
    return sorted(output, key=lambda row: str(row["id"]))


def reference_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        if "quarantine" in path.name:
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"MultiDoGO reference directory is empty: {directory}")
    return rows


def remove_reference_overlap_families(
    real: list[dict[str, object]],
    states: list[dict[str, object]],
    reference_signatures: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    combined = real + states
    candidate_signatures = [
        simhash64(family_skeleton(str(row["text"]))) for row in combined
    ]
    overlap_indexes = near_overlap_signatures(candidate_signatures, reference_signatures, 6)
    dropped_families = {
        str(combined[index]["family_id"]) for index in overlap_indexes
    }
    filtered_real = [row for row in real if str(row["family_id"]) not in dropped_families]
    filtered_states = [row for row in states if str(row["family_id"]) not in dropped_families]
    return filtered_real, filtered_states, {
        "near_overlap_rows": len(overlap_indexes),
        "conversation_families_removed": len(dropped_families),
        "real_rows_removed_with_families": len(real) - len(filtered_real),
        "state_rows_removed_with_families": len(states) - len(filtered_states),
    }


def build(repository: Path, output: Path, reference_directory: Path) -> dict[str, object]:
    source_manifest = verify_repository(repository)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MultiDoGO output: {output}")
    candidates: list[dict[str, object]] = []
    eligible_counts: dict[str, int] = {}
    for domain in DOMAINS:
        path = repository / "data" / "unannotated" / f"{domain}.tsv"
        dialogues = read_domain(path, domain)
        eligible_counts[domain] = len(dialogues)
        candidates.extend(conversation_candidate(dialogue) for dialogue in dialogues)
    train_selected, validation_selected, near_stats = select_conversations(candidates)
    assign_unique_agent_turns([train_selected, validation_selected])
    real_train = real_rows(train_selected, "train")
    real_validation = real_rows(validation_selected, "validation")
    states_train = state_rows(
        train_selected,
        "train",
        TRAIN_STATE_DOMAINS,
        TRAIN_STATE_FAMILIES_PER_DOMAIN,
    )
    states_validation = state_rows(
        validation_selected,
        "validation",
        VALIDATION_STATE_DOMAINS,
        VALIDATION_STATE_FAMILIES_PER_DOMAIN,
    )
    references = reference_rows(reference_directory)
    reference_signatures = [
        simhash64(family_skeleton(str(row["text"]))) for row in references
    ]
    real_train, states_train, train_overlap = remove_reference_overlap_families(
        real_train, states_train, reference_signatures
    )
    real_validation, states_validation, validation_overlap = remove_reference_overlap_families(
        real_validation, states_validation, reference_signatures
    )
    if {str(row["family_id"]) for row in real_train + states_train} & {
        str(row["family_id"]) for row in real_validation + states_validation
    }:
        raise RuntimeError("MultiDoGO conversation family crosses train and validation")
    output.mkdir(parents=True)
    artifacts = {
        "real_train": (output / "multidogo_real_train.jsonl", real_train),
        "state_train": (output / "multidogo_state_train.jsonl", states_train),
        "call_validation": (output / "multidogo_call_validation.jsonl", real_validation),
        "state_validation": (output / "multidogo_state_validation.jsonl", states_validation),
    }
    for _, (path, rows) in artifacts.items():
        write_jsonl(path, rows)
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "source": SOURCE,
        "state_source": STATE_SOURCE,
        "repository": source_manifest["repository"],
        "revision": REVISION,
        "license": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "dialogue_tree_sha256": source_manifest["dialogue_tree_sha256"],
        "citation": source_manifest["citation"],
        "domains": list(DOMAINS),
        "eligible_conversations_by_domain": eligible_counts,
        "candidate_conversations": len(candidates),
        "near_template_stats": near_stats,
        "selected_conversations_with_privacy_values_replaced": sum(
            1
            for source_id in {
                str(row["source_record_id"])
                for row in real_train + real_validation
                if bool(row["privacy_values_replaced"])
            }
        ),
        "partition": (
            f"deterministic {CANDIDATE_POOL_PER_DOMAIN}-conversation audit pool per domain; "
            f"one representative per exact/near-template family; "
            f"{TRAIN_CONVERSATIONS_PER_DOMAIN} train and "
            f"{VALIDATION_CONVERSATIONS_PER_DOMAIN} validation conversations per domain"
        ),
        "reference_overlap_control": {
            "reference_directory": str(reference_directory),
            "reference_rows": len(references),
            "near_hamming_max": 6,
            "training": train_overlap,
            "validation": validation_overlap,
        },
        "state_domain_partition": {
            "train": list(TRAIN_STATE_DOMAINS),
            "validation": list(VALIDATION_STATE_DOMAINS),
        },
        "policy": {
            "source_conversations_are_human_human_roleplay": True,
            "source_agent_is_trained_annotator": True,
            "weak_safe_label_from_legitimate_service_domain": True,
            "real_rows_counted_separately_from_synthetic_derivatives": True,
            "conversation_family_cross_split": False,
            "one_conversation_per_near_template_family": True,
            "audio_downloaded": False,
            "direct_reddit_scrape": False,
            "action_labels_on_real_rows": "high-precision lexical weak supervision",
            "state_variants": "decisive action followed by identical delay continuation",
            "model_rows_redistributed": False,
        },
        "counts": {
            "raw_domain_files": len(DOMAINS),
            "selected_train_conversations": len(
                {str(row["family_id"]) for row in real_train}
            ),
            "selected_validation_conversations": len(
                {str(row["family_id"]) for row in real_validation}
            ),
            "real_train_rows": len(real_train),
            "real_validation_rows": len(real_validation),
            "state_train_rows": len(states_train),
            "state_train_families": len({str(row["contrast_id"]) for row in states_train}),
            "state_validation_rows": len(states_validation),
            "state_validation_families": len(
                {str(row["contrast_id"]) for row in states_validation}
            ),
        },
        "artifacts": {
            name: {"path": str(path), "rows": len(rows), "sha256": file_sha256(path)}
            for name, (path, rows) in artifacts.items()
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
    parser.add_argument("--output", type=Path, default=Path("data/external/multidogo"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/experiments/schema20-action-states/processed"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.repository, args.output, args.reference_dir), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
