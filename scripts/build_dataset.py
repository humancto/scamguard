#!/usr/bin/env python3
"""Build leak-resistant ScamBench splits from pinned source archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from scamguard.signals import extract_signal_matches

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b\w+\[\.\]\w+(?:/\S*)?", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
MIXED_TOKEN_RE = re.compile(
    r"\b(?=[a-z0-9]{4,20}\b)(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]+\b", re.I
)
TRACKING_SUFFIX_RE = re.compile(r"(?:[~=_|]{2,}|[&*#]{2,})\s*[\w-]{3,32}\s*$", re.I)
PHONE_LIKE_RE = re.compile(r"(?<![A-Za-z0-9])\+?\d(?:[\d ()-]{5,}\d)(?![A-Za-z0-9])")
LONG_DIGIT_RE = re.compile(r"\d{10,}")
SPACE_RE = re.compile(r"\s+")
MARKETING_RE = re.compile(
    r"\b(?:casino|free spins?|gambling|betting|bet now|opt[ -]?out|unsubscribe|stop msg|"
    r"bonus code|deposit bonus)\b",
    re.I,
)
FRAUD_RE = re.compile(
    r"\b(?:account|bank|card|credential|delivery|identity|login|parcel|password|payment|"
    r"refund|security|suspended|verify|verification|wallet)\b",
    re.I,
)
DEFENSIVE_GUIDANCE_RE = re.compile(
    r"\b(?:beware of (?:fraud|scams?)|scam alert|fraud awareness|avoid scams?|"
    r"protect yourself|do not share|don['’]t share|never share|never asks?|will never ask|"
    r"will never call|does not ask|report (?:it|them|scams?)|"
    r"no compart(?:as|a)|nunca (?:te |le )?pedir[áa]|cuidado con (?:las )?estafas|"
    r"ne partagez jamais|ne (?:vous )?demandera jamais|attention aux arnaques|"
    r"niemals weitergeben|nicht weitergeben|wird (?:sie )?niemals fragen|vorsicht vor betrug|"
    r"deel\w* nooit|nooit delen|zal nooit vragen|"
    r"non condividere|non chieder[àa] mai|attenzione alle truffe|"
    r"n[ãa]o compartilhe|nunca pedir[áa]|cuidado com golpes|"
    r"jangan bagikan|tidak akan pernah meminta|waspada penipuan|"
    r"if (?:this|it|that) (?:is|was) not you)\b",
    re.I,
)
AUTH_CODE_RE = re.compile(
    r"\b(?:otp|one[- ]time (?:pin|password|code)|authentication code|verification code|"
    r"login code|sign[- ]in code|security code|\btac\b)\b",
    re.I,
)
STRONG_SCAM_SIGNALS = {
    "advance_fee",
    "contact_diversion",
    "credential_request",
    "guaranteed_return",
    "off_platform_request",
    "otp_request",
    "remote_access_request",
    "secrecy_isolation",
    "shortened_url",
    "suspicious_link",
    "threat_or_coercion",
    "too_good_to_be_true",
    "trust_acceleration",
    "unusual_payment_method",
}
EXTERNAL_ACTION_RE = re.compile(
    r"\b(?:click|visit|open|follow|tap|update|verify|confirm|activate|restore|log ?in|"
    r"sign ?in)\b.{0,100}(?:<URL>|https?://|www\.|\[\.\]|\b(?:link|here)\b)",
    re.I,
)
LOW_RISK_MARKETING_RE = re.compile(
    r"\b(?:advert|offer|promotion|sale|discount|subscription|free trial|mobile plan|"
    r"holiday|vacation|property|plot allotment|homeowners?|tenants?|loan for|"
    r"compensation up to|study participants?|upgrade|latest (?:phone|mobile)|"
    r"line ?rental|shop(?:ping)?|tickets? for sale)\b",
    re.I,
)


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    return SPACE_RE.sub(" ", value).strip()


def privacy_normalize_real_text(value: str) -> str:
    """Remove contact/account-like values that a detector does not need to memorize."""

    value = EMAIL_RE.sub("<EMAIL>", value)
    value = PHONE_LIKE_RE.sub("<PHONE_NUMBER>", value)
    return LONG_DIGIT_RE.sub("<ACCOUNT_NUMBER>", value)


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", clean_text(value).casefold())


def family_skeleton(value: str) -> str:
    text = TRACKING_SUFFIX_RE.sub("", clean_text(value)).casefold()
    text = URL_RE.sub("<url>", text)
    text = EMAIL_RE.sub("<email>", text)
    text = NUMBER_RE.sub("<number>", text)
    text = MIXED_TOKEN_RE.sub("<token>", text)
    text = re.sub(r"[^a-z<> ]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def simhash64(value: str) -> int:
    """Return a deterministic character-4-gram SimHash for near-template clustering."""

    padded = f"  {value}  "
    grams = [padded[index : index + 4] for index in range(max(1, len(padded) - 3))]
    weights = [0] * 64
    for gram in grams:
        hashed = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if hashed & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def simhash_bands(signature: int, *, max_hamming: int = 6) -> tuple[tuple[int, int], ...]:
    """Return a complete candidate index for the requested Hamming radius.

    Splitting 64 bits into ``max_hamming + 1`` disjoint bands guarantees that
    signatures at distance at most ``max_hamming`` share at least one exact
    band (pigeonhole principle). Four 16-bit bands do not provide that
    guarantee for radius six because differences can touch every band.
    """

    band_count = max_hamming + 1
    if not 0 <= max_hamming < 64:
        raise ValueError("max_hamming must be in [0, 63]")
    base_width, wider_bands = divmod(64, band_count)
    offset = 0
    result = []
    for band in range(band_count):
        width = base_width + int(band < wider_bands)
        mask = (1 << width) - 1
        result.append((band, (signature >> offset) & mask))
        offset += width
    return tuple(result)


def cluster_near_duplicates(
    rows: list[dict[str, object]], *, max_hamming: int = 6
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Cluster real rows by near-template similarity and quarantine mixed-label clusters."""

    real_indices = [index for index, row in enumerate(rows) if not row["is_synthetic"]]
    parent = {index: index for index in real_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    signatures: dict[int, int] = {}
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index in real_indices:
        signature = simhash64(family_skeleton(str(rows[index]["text"])))
        signatures[index] = signature
        candidates: set[int] = set()
        for key in simhash_bands(signature, max_hamming=max_hamming):
            candidates.update(buckets[key])
        for candidate in candidates:
            if (signature ^ signatures[candidate]).bit_count() <= max_hamming:
                union(index, candidate)
        for key in simhash_bands(signature, max_hamming=max_hamming):
            buckets[key].append(index)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index in real_indices:
        groups[find(index)].append(index)

    quarantined_indices: set[int] = set()
    conflicts: list[dict[str, object]] = []
    largest_cluster = 0
    for members in groups.values():
        largest_cluster = max(largest_cluster, len(members))
        labels = {str(rows[index]["label"]) for index in members}
        if len(labels) > 1:
            quarantined_indices.update(members)
            conflicts.append(
                {
                    "type": "near_template_label_conflict",
                    "labels": sorted(labels),
                    "candidates": [rows[index] for index in members],
                }
            )
            continue
        family_id = "near-" + min(
            short_hash(family_skeleton(str(rows[index]["text"]))) for index in members
        )
        for index in members:
            rows[index] = rows[index] | {
                "family_id": family_id,
                "split": split_for_family(family_id),
            }

    kept = [row for index, row in enumerate(rows) if index not in quarantined_indices]
    stats = {
        "real_rows_clustered": len(real_indices),
        "near_template_clusters": len(groups),
        "largest_near_template_cluster": largest_cluster,
        "near_template_rows_quarantined": len(quarantined_indices),
    }
    return kept, conflicts, stats


def remove_near_overlaps(
    candidates: list[dict[str, object]],
    references: list[dict[str, object]],
    *,
    max_hamming: int = 6,
) -> tuple[list[dict[str, object]], int]:
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    reference_signatures = [simhash64(family_skeleton(str(row["text"]))) for row in references]
    for index, signature in enumerate(reference_signatures):
        for key in simhash_bands(signature, max_hamming=max_hamming):
            buckets[key].append(index)

    kept = []
    dropped = 0
    for row in candidates:
        signature = simhash64(family_skeleton(str(row["text"])))
        possible: set[int] = set()
        for key in simhash_bands(signature, max_hamming=max_hamming):
            possible.update(buckets[key])
        if any(
            (signature ^ reference_signatures[index]).bit_count() <= max_hamming
            for index in possible
        ):
            dropped += 1
        else:
            kept.append(row)
    return kept, dropped


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def split_for_family(family_id: str) -> str:
    bucket = int(hashlib.sha256(family_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def category_for(text: str) -> str:
    lowered = text.casefold()
    if any(token in lowered for token in ("password", "otp", "verification code", "sign-in")):
        return "CREDENTIAL_THEFT"
    if any(token in lowered for token in ("parcel", "delivery", "toll", "shipping")):
        return "DELIVERY_TOLL"
    if any(token in lowered for token in ("job", "hired", "earn", "work from home")):
        return "OPPORTUNITY"
    if any(token in lowered for token in ("bank", "payment", "loan", "investment", "refund")):
        return "FINANCIAL"
    return "UNKNOWN"


def protective_message_label(text: str) -> tuple[str, str] | None:
    """Recognize content that warns the recipient rather than manipulating them."""

    signals = {match.signal.value for match in extract_signal_matches(text)}
    if DEFENSIVE_GUIDANCE_RE.search(text):
        if EXTERNAL_ACTION_RE.search(text) or "contact_diversion" in signals:
            return "UNCERTAIN", "defensive_guidance_with_external_action"
        return "SAFE", "defensive_guidance"
    if AUTH_CODE_RE.search(text) and signals <= {
        "authority_impersonation",
        "credential_request",
        "otp_request",
    }:
        return "SAFE", "standalone_authentication_notification"
    return None


def has_strong_scam_evidence(text: str) -> bool:
    signals = {match.signal.value for match in extract_signal_matches(text)}
    return bool(signals & STRONG_SCAM_SIGNALS)


def mendeley_message_label(text: str, source_label: str) -> tuple[str, str] | None:
    if source_label == "ham":
        return "SAFE", "source_ham"
    if source_label == "spam":
        return "UNCERTAIN", "source_generic_spam"
    if source_label != "smishing":
        return None
    protective = protective_message_label(text)
    if protective:
        return protective
    signals = {match.signal.value for match in extract_signal_matches(text)}
    if LOW_RISK_MARKETING_RE.search(text) and signals <= {
        "artificial_urgency",
        "contact_diversion",
        "payment_request",
        "suspicious_link",
    }:
        return "UNCERTAIN", "commercial_offer_without_clear_fraud"
    if has_strong_scam_evidence(text):
        return "SCAM", "source_smishing_with_strong_text_evidence"
    return "UNCERTAIN", "source_smishing_without_strong_text_evidence"


def forum_message_label(text: str, source_label: str) -> tuple[str, str]:
    protective = protective_message_label(text)
    if protective:
        return protective
    if source_label == "spam":
        return "UNCERTAIN", "source_generic_spam"
    if source_label == "wrong number" and not has_strong_scam_evidence(text):
        return "UNCERTAIN", "wrong_number_without_message_evidence"
    if has_strong_scam_evidence(text):
        return "SCAM", "source_reported_smishing_with_strong_text_evidence"
    return "UNCERTAIN", "source_reported_without_strong_text_evidence"


def make_row(
    *, text: str, label: str, source: str, source_label: str, license_name: str
) -> dict[str, object] | None:
    text = privacy_normalize_real_text(clean_text(text))
    if len(text) < 4:
        return None
    family_id = "family-" + short_hash(family_skeleton(text))
    return {
        "id": source + "-" + short_hash(normalized(text)),
        "text": text,
        "label": label,
        "category": category_for(text) if label == "SCAM" else "NONE",
        "source": source,
        "source_label": source_label,
        "license": license_name,
        "split": split_for_family(family_id),
        "family_id": family_id,
        "is_synthetic": False,
    }


def read_sms_phishing(path: Path) -> Iterable[dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("Dataset_5971.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace"))
            for source_row in reader:
                source_label = source_row["LABEL"].strip().casefold()
                text = privacy_normalize_real_text(clean_text(source_row["TEXT"]))
                label_decision = mendeley_message_label(text, source_label)
                if label_decision:
                    label, label_policy = label_decision
                    row = make_row(
                        text=text,
                        label=label,
                        source="mendeley_sms_phishing",
                        source_label=source_label,
                        license_name="CC-BY-4.0",
                    )
                    if row:
                        row["label_policy"] = label_policy
                        yield row


def read_uci(path: Path) -> Iterable[dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("SMSSpamCollection") as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                source_label, separator, text = line.partition("\t")
                if not separator:
                    continue
                label = "SAFE" if source_label == "ham" else "UNCERTAIN"
                row = make_row(
                    text=text,
                    label=label,
                    source="uci_sms_spam",
                    source_label=source_label,
                    license_name="CC-BY-4.0",
                )
                if row:
                    yield row


def read_financial_holdout(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for source_row in csv.DictReader(handle):
            source_label = source_row["label"].strip().casefold()
            label = {"ham": "SAFE", "scam": "SCAM"}.get(source_label)
            if not label:
                continue
            row = make_row(
                text=source_row["message"],
                label=label,
                source="mendeley_financial_scam",
                source_label=source_label,
                license_name="CC-BY-4.0",
            )
            if row:
                row["split"] = "ood"
                yield row


def read_wspr(path: Path) -> Iterable[dict[str, object]]:
    """Yield one reproducible example per WSPR campaign-like template family.

    The source is positive-only and was confirmed by VirusTotal/APWG. Obvious
    gambling/marketing messages without fraud evidence are conservatively mapped
    to UNCERTAIN so they do not teach ScamGuard that ordinary advertising is fraud.
    Sender and destination-number columns are deliberately discarded.
    """

    representatives: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for source_row in csv.DictReader(handle):
            text = privacy_normalize_real_text(clean_text(source_row.get("message", "")))
            if not text:
                continue
            protective = protective_message_label(text)
            marketing_only = bool(MARKETING_RE.search(text)) and not bool(FRAUD_RE.search(text))
            non_fraud_content = protective is not None
            strong_evidence = has_strong_scam_evidence(text)
            label = (
                "SCAM"
                if strong_evidence and not marketing_only and not non_fraud_content
                else "UNCERTAIN"
            )
            if protective:
                label_policy = "reported_phishing_with_" + protective[1]
            elif marketing_only:
                label_policy = "reported_phishing_marketing_review"
            elif not strong_evidence:
                label_policy = "reported_phishing_without_strong_text_evidence"
            else:
                label_policy = "source_reported_phishing_with_strong_text_evidence"
            row = make_row(
                text=text,
                label=label,
                source="wspr_sms_phishing",
                source_label=(
                    "reported_phishing_non_fraud_content_review"
                    if non_fraud_content
                    else "reported_phishing_marketing_review"
                    if marketing_only
                    else "reported_phishing"
                ),
                license_name="MIT",
            )
            if row is None:
                continue
            row["label_policy"] = label_policy
            family_id = str(row["family_id"])
            current = representatives.get(family_id)
            if current is None or str(row["id"]) < str(current["id"]):
                representatives[family_id] = row
    yield from representatives.values()


def read_imc25_forum(path: Path) -> Iterable[dict[str, object]]:
    """Read the CC-BY public-forum artifact while discarding identifying metadata."""

    category_map = {
        "banking": "FINANCIAL",
        "delivery": "DELIVERY_TOLL",
        "government": "IDENTITY_IMPERSONATION",
        "telecom": "IDENTITY_IMPERSONATION",
        "wrong number": "RELATIONSHIP",
        "hey mum/dad": "IDENTITY_IMPERSONATION",
    }
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for source_row in csv.DictReader(handle):
            source_label = source_row.get("scam_type", "").strip().casefold() or "unknown"
            # The artifact already contains placeholders. Mask any residual long
            # telephone-like sequence and email before it enters a model split.
            text = EMAIL_RE.sub("<EMAIL>", source_row.get("text", ""))
            text = PHONE_LIKE_RE.sub("<PHONE_NUMBER>", text)
            # The artifact records user-reported scam types, but some rows are
            # ordinary authentication notices or anti-fraud education. Product
            # labels describe the observable message, not surrounding context.
            label, label_policy = forum_message_label(text, source_label)
            row = make_row(
                text=text,
                label=label,
                source="imc25_public_forum_smishing",
                source_label=source_label,
                license_name="CC-BY-4.0",
            )
            if row is None:
                continue
            row["label_policy"] = label_policy
            if label == "SCAM" and source_label in category_map:
                row["category"] = category_map[source_label]
            row["source_language"] = source_row.get("language", "").strip() or "UNKNOWN"
            yield row


def read_azsc(path: Path) -> Iterable[dict[str, object]]:
    """Read AZ-SC as a mixed-provenance OOD set, never as training data.

    The source paper reports consented user messages, translated UCI messages,
    and self-generated messages but does not expose provenance per row. We
    therefore preserve that limitation and require text-level evidence before
    mapping a source smishing label to the product's SCAM class.
    """

    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for source_row in csv.DictReader(handle):
            source_label = source_row.get("label", "").strip().casefold()
            text = privacy_normalize_real_text(clean_text(source_row.get("message", "")))
            if source_label == "ham":
                label, label_policy = "SAFE", "source_ham"
            elif source_label == "spam":
                label, label_policy = "UNCERTAIN", "source_generic_spam"
            elif source_label == "smishing":
                protective = protective_message_label(text)
                if protective:
                    label, label_policy = protective
                elif has_strong_scam_evidence(text):
                    label = "SCAM"
                    label_policy = "source_smishing_with_strong_text_evidence"
                else:
                    label = "UNCERTAIN"
                    label_policy = "source_smishing_without_strong_text_evidence"
            else:
                continue
            row = make_row(
                text=text,
                label=label,
                source="azsc_azerbaijani_sms",
                source_label=source_label,
                license_name="CC-BY-4.0",
            )
            if row:
                row["label_policy"] = label_policy
                row["source_language"] = "Azerbaijani"
                row["source_provenance"] = "mixed_unidentified_real_translated_self_generated"
                row["split"] = "ood"
                yield row


def deterministic_diverse_sample(
    rows: list[dict[str, object]], limit: int, *, seed: str
) -> list[dict[str, object]]:
    """Reserve 25% for type/language breadth, then fill proportionally by hash rank."""

    if len(rows) <= limit:
        return sorted(rows, key=lambda row: str(row["id"]))
    groups: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    def rank(row: dict[str, object]) -> str:
        return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()

    for row in rows:
        groups[(str(row["source_label"]), str(row.get("source_language", "UNKNOWN")))].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=rank)

    diversity_target = min(limit, max(len(groups), limit // 4))
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    depth = 0
    while len(selected) < diversity_target:
        added = False
        for key in sorted(groups):
            if depth < len(groups[key]):
                row = groups[key][depth]
                selected.append(row)
                selected_ids.add(str(row["id"]))
                added = True
                if len(selected) == diversity_target:
                    break
        if not added:
            break
        depth += 1
    remainder = sorted((row for row in rows if str(row["id"]) not in selected_ids), key=rank)
    return selected + remainder[: limit - len(selected)]


def forum_partition(family_id: str) -> str:
    bucket = int(hashlib.sha256(f"forum-ood-v1:{family_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 55:
        return "train"
    if bucket < 65:
        return "validation"
    return "ood"


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def deduplicate(
    rows: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], int, list[dict[str, object]]]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[normalized(str(row["text"]))].append(row)

    kept: list[dict[str, object]] = []
    dropped = 0
    conflicts: list[dict[str, object]] = []
    for key, candidates in grouped.items():
        labels = {str(row["label"]) for row in candidates}
        if len(labels) > 1:
            conflicts.append(
                {
                    "normalized_hash": short_hash(key),
                    "labels": sorted(labels),
                    "candidates": candidates,
                }
            )
            dropped += len(candidates)
            continue
        kept.append(candidates[0])
        dropped += len(candidates) - 1
    return kept, dropped, conflicts


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    materialized = list(rows)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in materialized),
        encoding="utf-8",
    )
    return len(materialized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--synthetic", type=Path, default=Path("data/generated/synthetic.jsonl"))
    parser.add_argument(
        "--dialogue-synthetic",
        type=Path,
        default=Path("data/generated/dialogue_curriculum.jsonl"),
    )
    parser.add_argument(
        "--taskmaster-safe",
        type=Path,
        default=Path("data/generated/taskmaster_safe_train.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--schema-version",
        type=int,
        default=12,
        help="dataset schema identity recorded in the output manifest",
    )
    parser.add_argument("--forum-train-scam-limit", type=int, default=1000)
    parser.add_argument("--forum-train-uncertain-limit", type=int, default=100)
    parser.add_argument("--forum-train-safe-limit", type=int, default=300)
    parser.add_argument("--forum-validation-scam-limit", type=int, default=1000)
    parser.add_argument("--forum-validation-uncertain-limit", type=int, default=100)
    parser.add_argument("--forum-validation-safe-limit", type=int, default=50)
    parser.add_argument("--forum-ood-scam-limit", type=int, default=2000)
    parser.add_argument("--forum-ood-uncertain-limit", type=int, default=200)
    parser.add_argument("--forum-ood-safe-limit", type=int, default=100)
    args = parser.parse_args()
    limits = {
        "forum_train_scam": args.forum_train_scam_limit,
        "forum_train_uncertain": args.forum_train_uncertain_limit,
        "forum_train_safe": args.forum_train_safe_limit,
        "forum_validation_scam": args.forum_validation_scam_limit,
        "forum_validation_uncertain": args.forum_validation_uncertain_limit,
        "forum_validation_safe": args.forum_validation_safe_limit,
        "forum_ood_scam": args.forum_ood_scam_limit,
        "forum_ood_uncertain": args.forum_ood_uncertain_limit,
        "forum_ood_safe": args.forum_ood_safe_limit,
    }
    if any(value < 0 for value in limits.values()):
        raise ValueError(f"forum sample limits must be non-negative: {limits}")

    required = {
        "SMS phishing": args.raw / "sms_phishing_5971.zip",
        "UCI SMS": args.raw / "uci_sms_spam.zip",
        "financial holdout": args.raw / "financial_scam.csv",
        "WSPR SMS phishing": args.raw / "wspr_phishing_messages.csv",
        "IMC 2025 forum smishing": args.raw / "imc25_public_forum_smishing.csv",
        "AZ-SC Azerbaijani SMS": args.raw / "azsc_azerbaijani_sms.csv",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    for name, path in {
        "synthetic curriculum": args.synthetic,
        "synthetic dialogue curriculum": args.dialogue_synthetic,
        "Taskmaster SAFE curriculum": args.taskmaster_safe,
    }.items():
        if not path.is_file():
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError(
            "missing inputs; run scripts/fetch_datasets.py:\n" + "\n".join(missing)
        )

    # Prefer the phishing-specific source when an exact UCI-derived record appears in both.
    candidate_rows = itertools_chain(
        read_sms_phishing(required["SMS phishing"]),
        read_uci(required["UCI SMS"]),
        read_jsonl(args.synthetic),
        read_jsonl(args.dialogue_synthetic),
        read_jsonl(args.taskmaster_safe),
    )
    core_rows, dropped, conflicts = deduplicate(candidate_rows)
    taskmaster_rows = [
        row for row in core_rows if row["source"] == "taskmaster1_woz_dialogues"
    ]
    core_rows = [
        row for row in core_rows if row["source"] != "taskmaster1_woz_dialogues"
    ]
    holdout, holdout_dropped, holdout_conflicts = deduplicate(
        read_financial_holdout(required["financial holdout"])
    )
    wspr_rows, wspr_dropped, wspr_conflicts = deduplicate(read_wspr(required["WSPR SMS phishing"]))
    forum_rows, forum_dropped, forum_conflicts = deduplicate(
        read_imc25_forum(required["IMC 2025 forum smishing"])
    )
    azsc_rows, azsc_dropped, azsc_conflicts = deduplicate(
        read_azsc(required["AZ-SC Azerbaijani SMS"])
    )
    combined_rows, cross_source_dropped, cross_source_conflicts = deduplicate(
        core_rows + wspr_rows + forum_rows
    )

    # SimHash clusters catch near-identical templates whose tracking tokens or
    # small wording changes survive deterministic field masking. Mixed-label
    # clusters are ambiguous by construction and are quarantined in full.
    clustered, near_conflicts, near_stats = cluster_near_duplicates(combined_rows)
    external_sources = {"wspr_sms_phishing", "imc25_public_forum_smishing"}
    core_rows = [row for row in clustered if row["source"] not in external_sources]
    clustered_wspr = [row for row in clustered if row["source"] == "wspr_sms_phishing"]
    clustered_forum = [row for row in clustered if row["source"] == "imc25_public_forum_smishing"]

    # Taskmaster is pre-partitioned by conversation before sampling. It supplies
    # fitting-only weak SAFE hard negatives, while its disjoint selection slice
    # lives under data/external. Cluster it independently, then remove any
    # cross-source near match and force every surviving family back to train.
    clustered_taskmaster, taskmaster_near_conflicts, taskmaster_near_stats = (
        cluster_near_duplicates(taskmaster_rows)
    )
    clustered_taskmaster, taskmaster_near_overlaps_removed = remove_near_overlaps(
        clustered_taskmaster, clustered
    )
    taskmaster_train = [row | {"split": "train"} for row in clustered_taskmaster]

    # WSPR is highly repetitive. Retain one deterministic representative per
    # near-template cluster so its source cannot dominate the training loss.
    wspr_representatives: dict[str, dict[str, object]] = {}
    for row in clustered_wspr:
        family_id = str(row["family_id"])
        current = wspr_representatives.get(family_id)
        if current is None or str(row["id"]) < str(current["id"]):
            wspr_representatives[family_id] = row
    wspr_rows = list(wspr_representatives.values())
    wspr_near_templates_removed = len(clustered_wspr) - len(wspr_rows)
    wspr_train = [row | {"split": "train"} for row in wspr_rows if row["split"] == "train"]
    wspr_holdout = [row | {"split": "ood"} for row in wspr_rows if row["split"] != "train"]

    # Exclude forum families that are near matches to another source, retain one
    # representative per remaining family, and cap training so positives do not
    # overwhelm real SAFE/hard-negative examples. Only evidence-grounded SCAM
    # rows enter generative supervision; harder rows remain eligible for OOD.
    non_forum_families = {
        str(row["family_id"]) for row in clustered if row["source"] != "imc25_public_forum_smishing"
    }
    forum_cross_source_near_removed = sum(
        str(row["family_id"]) in non_forum_families for row in clustered_forum
    )
    forum_representatives: dict[str, dict[str, object]] = {}
    for row in clustered_forum:
        family_id = str(row["family_id"])
        if family_id in non_forum_families:
            continue
        current = forum_representatives.get(family_id)
        if current is None or str(row["id"]) < str(current["id"]):
            forum_representatives[family_id] = row
    forum_rows = list(forum_representatives.values())
    forum_train_pool = [
        row
        for row in forum_rows
        if forum_partition(str(row["family_id"])) == "train"
        and (row["label"] != "SCAM" or extract_signal_matches(str(row["text"])))
    ]
    forum_validation_pool = [
        row for row in forum_rows if forum_partition(str(row["family_id"])) == "validation"
    ]
    forum_holdout_pool = [
        row for row in forum_rows if forum_partition(str(row["family_id"])) == "ood"
    ]
    forum_train_scam = deterministic_diverse_sample(
        [row for row in forum_train_pool if row["label"] == "SCAM"],
        args.forum_train_scam_limit,
        seed="imc25-forum-train-scam-v1",
    )
    forum_train_uncertain = deterministic_diverse_sample(
        [row for row in forum_train_pool if row["label"] == "UNCERTAIN"],
        args.forum_train_uncertain_limit,
        seed="imc25-forum-train-uncertain-v1",
    )
    forum_train_safe = deterministic_diverse_sample(
        [row for row in forum_train_pool if row["label"] == "SAFE"],
        args.forum_train_safe_limit,
        seed="imc25-forum-train-safe-v1",
    )
    forum_validation_scam = deterministic_diverse_sample(
        [row for row in forum_validation_pool if row["label"] == "SCAM"],
        args.forum_validation_scam_limit,
        seed="imc25-forum-validation-scam-v1",
    )
    forum_validation_uncertain = deterministic_diverse_sample(
        [row for row in forum_validation_pool if row["label"] == "UNCERTAIN"],
        args.forum_validation_uncertain_limit,
        seed="imc25-forum-validation-uncertain-v1",
    )
    forum_validation_safe = deterministic_diverse_sample(
        [row for row in forum_validation_pool if row["label"] == "SAFE"],
        args.forum_validation_safe_limit,
        seed="imc25-forum-validation-safe-v1",
    )
    forum_holdout_scam = deterministic_diverse_sample(
        [row for row in forum_holdout_pool if row["label"] == "SCAM"],
        args.forum_ood_scam_limit,
        seed="imc25-forum-ood-scam-v1",
    )
    forum_holdout_uncertain = deterministic_diverse_sample(
        [row for row in forum_holdout_pool if row["label"] == "UNCERTAIN"],
        args.forum_ood_uncertain_limit,
        seed="imc25-forum-ood-uncertain-v1",
    )
    forum_holdout_safe = deterministic_diverse_sample(
        [row for row in forum_holdout_pool if row["label"] == "SAFE"],
        args.forum_ood_safe_limit,
        seed="imc25-forum-ood-safe-v1",
    )
    forum_train = [
        row | {"split": "train"}
        for row in forum_train_scam + forum_train_uncertain + forum_train_safe
    ]
    forum_validation = [
        row | {"split": "validation"}
        for row in forum_validation_scam + forum_validation_uncertain + forum_validation_safe
    ]
    forum_holdout = [
        row | {"split": "ood"}
        for row in forum_holdout_scam + forum_holdout_uncertain + forum_holdout_safe
    ]
    rows = core_rows + wspr_train + forum_train + taskmaster_train

    # External financial rows remain untouched by fitting but near-template
    # overlaps with any development row are removed to keep the OOD claim honest.
    development_keys = {normalized(str(row["text"])) for row in rows}
    holdout = [row for row in holdout if normalized(str(row["text"])) not in development_keys]
    holdout, holdout_near_overlaps_removed = remove_near_overlaps(holdout, rows)

    # AZ-SC is evaluated independently because its paper does not expose which
    # individual rows are real, translated, or self-generated. Collapse its
    # near-template families and remove development overlap before evaluation.
    clustered_azsc, azsc_near_conflicts, azsc_near_stats = cluster_near_duplicates(azsc_rows)
    azsc_representatives: dict[str, dict[str, object]] = {}
    for row in clustered_azsc:
        family_id = str(row["family_id"])
        current = azsc_representatives.get(family_id)
        if current is None or str(row["id"]) < str(current["id"]):
            azsc_representatives[family_id] = row | {"split": "ood"}
    azsc_holdout = list(azsc_representatives.values())
    development_keys = {normalized(str(row["text"])) for row in rows}
    azsc_holdout = [
        row for row in azsc_holdout if normalized(str(row["text"])) not in development_keys
    ]
    azsc_holdout, azsc_near_overlaps_removed = remove_near_overlaps(azsc_holdout, rows)

    args.output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        counts[split] = write_jsonl(
            args.output / f"{split}.jsonl", (row for row in rows if row["split"] == split)
        )
    counts["ood_financial"] = write_jsonl(args.output / "ood_financial.jsonl", holdout)
    counts["ood_wspr"] = write_jsonl(args.output / "ood_wspr.jsonl", wspr_holdout)
    counts["forum_validation"] = write_jsonl(
        args.output / "forum_validation.jsonl", forum_validation
    )
    counts["ood_forum"] = write_jsonl(args.output / "ood_forum.jsonl", forum_holdout)
    counts["ood_azsc"] = write_jsonl(args.output / "ood_azsc.jsonl", azsc_holdout)
    write_jsonl(
        args.output / "quarantine_label_conflicts.jsonl",
        conflicts
        + holdout_conflicts
        + wspr_conflicts
        + forum_conflicts
        + azsc_conflicts
        + cross_source_conflicts
        + near_conflicts
        + taskmaster_near_conflicts
        + azsc_near_conflicts,
    )

    manifest = {
        "schema_version": args.schema_version,
        "policy": {
            "split_unit": "masked template plus 64-bit character SimHash cluster",
            "near_template_hamming_max": 6,
            "near_template_candidate_bands": 7,
            "near_template_candidate_index_complete": True,
            "generic_spam_label": "UNCERTAIN",
            "ood_financial_used_for_fitting": False,
            "ood_wspr_used_for_fitting": False,
            "ood_wspr_binary_subset_is_positive_only": True,
            "direct_reddit_scrape": False,
            "forum_artifact": "IMC 2025 CC-BY-4.0 research release",
            "forum_validation_used_for_fitting": False,
            "forum_validation_used_for_threshold": False,
            "forum_validation_used_for_training_size_selection": True,
            "forum_ood_used_for_fitting_or_selection": False,
            "azsc_ood_used_for_fitting_or_selection": False,
            "azsc_source_provenance": (
                "mixed consented user SMS, translated UCI, and self-generated rows; "
                "per-row provenance unavailable; excluded from licensed-real totals"
            ),
            "forum_wrong_number_without_message_evidence": "UNCERTAIN",
            "defensive_guidance_without_external_action": "SAFE",
            "standalone_authentication_notification": "SAFE",
            "source_reported_non_fraud_content": "UNCERTAIN",
            "source_reported_scam_requires_strong_text_evidence": True,
            "bare_domain_recognition": "curated web TLD allowlist; schemes/placeholders unchanged",
            "real_source_privacy_normalization": (
                "emails and long phone/account-like digit sequences replaced with placeholders"
            ),
            "synthetic_generation": (
                "v5 short-message grammars, targeted counterfactual v1 train-only correction "
                "families, plus v2 paired five-turn dialogue grammars across 12 scenarios, all "
                "grounded in official scam advisories; no source message text copied"
            ),
            "taskmaster_hard_negative_provenance": (
                "human-authored Wizard-of-Oz roleplay, weak SAFE label from legitimate task "
                "domain; counted separately from naturally occurring communications"
            ),
            "taskmaster_selection_used_for_fitting_or_threshold": False,
            "taskmaster_context_policy": (
                "latest complete turns capped at 425 characters; verified at no more than 150 "
                "tokens after speaker-neutral-v1 with the pinned ModernBERT tokenizer"
            ),
        },
        "forum_sample_limits": limits,
        "counts": counts,
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "sources": dict(Counter(str(row["source"]) for row in rows)),
        "exact_duplicates_removed": dropped,
        "cross_source_duplicates_removed": cross_source_dropped,
        "cross_source_conflicting_label_groups_quarantined": len(cross_source_conflicts),
        "conflicting_label_groups_quarantined": len(conflicts),
        "holdout_duplicates_removed": holdout_dropped,
        "holdout_conflicting_label_groups_quarantined": len(holdout_conflicts),
        "holdout_near_overlaps_removed": holdout_near_overlaps_removed,
        "near_template_label_conflicts_quarantined": len(near_conflicts),
        "near_template_stats": near_stats,
        "taskmaster_train_rows": len(taskmaster_train),
        "taskmaster_near_template_stats": taskmaster_near_stats,
        "taskmaster_near_template_label_conflicts_quarantined": len(
            taskmaster_near_conflicts
        ),
        "taskmaster_near_overlaps_with_other_sources_removed": (
            taskmaster_near_overlaps_removed
        ),
        "wspr_template_representatives": len(wspr_rows),
        "wspr_train_enrichment": len(wspr_train),
        "wspr_positive_holdout": len(wspr_holdout),
        "wspr_duplicates_removed": wspr_dropped,
        "wspr_near_templates_removed": wspr_near_templates_removed,
        "wspr_conflicting_label_groups_quarantined": len(wspr_conflicts),
        "forum_exact_duplicates_removed": forum_dropped,
        "forum_cross_source_near_rows_removed": forum_cross_source_near_removed,
        "forum_template_representatives": len(forum_rows),
        "forum_train_scam": len(forum_train_scam),
        "forum_train_uncertain": len(forum_train_uncertain),
        "forum_train_safe": len(forum_train_safe),
        "forum_validation_scam": len(forum_validation_scam),
        "forum_validation_uncertain": len(forum_validation_uncertain),
        "forum_validation_safe": len(forum_validation_safe),
        "forum_ood_scam": len(forum_holdout_scam),
        "forum_ood_uncertain": len(forum_holdout_uncertain),
        "forum_ood_safe": len(forum_holdout_safe),
        "forum_conflicting_label_groups_quarantined": len(forum_conflicts),
        "azsc_exact_duplicates_removed": azsc_dropped,
        "azsc_conflicting_label_groups_quarantined": len(azsc_conflicts),
        "azsc_near_template_label_conflicts_quarantined": len(azsc_near_conflicts),
        "azsc_near_template_stats": azsc_near_stats,
        "azsc_template_representatives": len(azsc_representatives),
        "azsc_near_overlaps_with_development_removed": azsc_near_overlaps_removed,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def itertools_chain(*iterables: Iterable[dict[str, object]]) -> Iterable[dict[str, object]]:
    for iterable in iterables:
        yield from iterable


if __name__ == "__main__":
    main()
