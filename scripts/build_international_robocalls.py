#!/usr/bin/env python3
"""Build a text-only external diagnostic from the International Robocalls Dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        URL_RE,
        clean_text,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the repo on sys.path.
    from build_dataset import (  # type: ignore[no-redef]
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        URL_RE,
        clean_text,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        privacy_normalize_real_text,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )

from scamguard.metrics import file_sha256

SOURCE = "international_robocalls_zenodo_21066049"
SOURCE_DOI = "10.5281/zenodo.21066049"
SOURCE_RECORD = 21066049
SOURCE_METADATA_REVISION = 4
SOURCE_MODIFIED = "2026-07-01T07:00:08.494820+00:00"
SOURCE_SHA256 = "2a5b09a9ea260001f67465bed329dbf51f9e1d82fbe3d11aa1656105ef3a7995"
SOURCE_MD5 = "318c485aa38e1760fe7ac3338f20526c"
LICENSE_NAME = "CC-BY-NC-4.0"
EXPECTED_MEMBERS = {
    "text/",
    "text/text/",
    "text/text/int.csv",
    "text/text/us.csv",
}
CSV_MEMBERS = {
    "international": "text/text/int.csv",
    "us": "text/text/us.csv",
}
EXPECTED_HEADER = ["cluster_id", "campaign", "transcript"]
# The immutable archive contains one more record in each CSV than its landing-page prose reports.
EXPECTED_ROWS = {"international": 116, "us": 725}

SCAM_CAMPAIGNS = {
    "ROBOCALL: Bank Fraud Alert Scam",
    "ROBOCALL: Benefits/Grants Scam",
    "ROBOCALL: Education/Student Loan Scam",
    "ROBOCALL: Google Business Listing Scam",
    "ROBOCALL: Health Insurance Scam",
    "ROBOCALL: IRS/Tax Scam",
    "ROBOCALL: Legal Threat Scam",
    "ROBOCALL: Loan/Credit Scam",
    "ROBOCALL: Loan/Debt Scam (Hindi)",
    "ROBOCALL: Vehicle Warranty Scam",
}
UNCERTAIN_CAMPAIGNS = {
    "ROBOCALL: Financial Spam (Generic)",
    "ROBOCALL: Insurance Quote Spam",
    "ROBOCALL: Political Campaign",
    "ROBOCALL: Telemarketing (Generic)",
    "ROBOCALL: Unclassified",
}


def safe_members(archive: zipfile.ZipFile) -> set[str]:
    members = {info.filename for info in archive.infolist()}
    for name in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe international robocall ZIP member: {name!r}")
    if members != EXPECTED_MEMBERS:
        raise ValueError(f"unexpected international robocall ZIP members: {sorted(members)!r}")
    return members


def label_for_campaign(campaign: str) -> str:
    if campaign in SCAM_CAMPAIGNS:
        return "SCAM"
    if campaign in UNCERTAIN_CAMPAIGNS:
        return "UNCERTAIN"
    raise ValueError(f"unknown international robocall campaign: {campaign!r}")


def reference_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in paths:
        candidates = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file() or "quarantine" in path.name:
                continue
            rows.extend(row for row in read_jsonl(path) if isinstance(row.get("text"), str))
    return rows


def read_source(source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("international robocall source differs from the pinned Zenodo artifact")

    rows: list[dict[str, object]] = []
    region_counts = Counter[str]()
    campaign_counts = Counter[str]()
    privacy = Counter[str]()
    with zipfile.ZipFile(source) as archive:
        safe_members(archive)
        for region, member in CSV_MEMBERS.items():
            with archive.open(member) as raw_handle:
                handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(handle)
                if reader.fieldnames != EXPECTED_HEADER:
                    raise ValueError(f"unexpected {member} header: {reader.fieldnames!r}")
                source_count = 0
                for source_count, source_row in enumerate(reader, start=1):
                    raw_text = str(source_row["transcript"]).strip()
                    campaign = str(source_row["campaign"]).strip()
                    cluster_id = str(source_row["cluster_id"]).strip()
                    if not raw_text or not campaign:
                        raise ValueError(f"missing required value in {member} row {source_count}")
                    label = label_for_campaign(campaign)
                    privacy["rows_with_email"] += bool(EMAIL_RE.search(raw_text))
                    privacy["rows_with_phone_like"] += bool(PHONE_LIKE_RE.search(raw_text))
                    privacy["rows_with_long_digit_sequence"] += bool(
                        LONG_DIGIT_RE.search(raw_text)
                    )
                    privacy["rows_with_url"] += bool(URL_RE.search(raw_text))
                    text = URL_RE.sub(
                        "<URL>", privacy_normalize_real_text(clean_text(raw_text))
                    )
                    row = make_row(
                        text=text,
                        label=label,
                        source=SOURCE,
                        source_label=campaign,
                        license_name=LICENSE_NAME,
                    )
                    if row is None:
                        raise ValueError(f"transcript became empty in {member} row {source_count}")
                    source_identity = f"{region}:{source_count}:{cluster_id}:{raw_text}"
                    row.update(
                        {
                            "split": "international_robocalls",
                            "source_region": region,
                            "source_cluster_id": cluster_id or None,
                            "source_record_sha256": hashlib.sha256(
                                source_identity.encode("utf-8")
                            ).hexdigest(),
                            "provenance_class": "real_world_honeypot_robocall_transcript",
                            "individual_robocall_human_verified": True,
                            "individual_scam_ground_truth": False,
                            "label_policy": (
                                "publisher_campaign_scam_silver_label"
                                if label == "SCAM"
                                else "publisher_spam_telemarketing_or_unclassified_robocall"
                            ),
                            "privacy_normalization": (
                                "url_email_phone_and_long_digit_placeholders_v1"
                            ),
                        }
                    )
                    rows.append(row)
                    region_counts[region] += 1
                    campaign_counts[campaign] += 1
                if source_count != EXPECTED_ROWS[region]:
                    raise ValueError(
                        f"expected {EXPECTED_ROWS[region]} rows in {member}, found {source_count}"
                    )
    return rows, {
        "source_rows": len(rows),
        "region_rows": dict(sorted(region_counts.items())),
        "campaign_rows": dict(sorted(campaign_counts.items())),
        "privacy_findings_before_normalization": dict(privacy),
    }


def one_per_family(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family_id"])].append(row)
    kept = [min(group, key=lambda row: str(row["id"])) for group in groups.values()]
    return kept, len(rows) - len(kept)


def build(
    source: Path,
    output: Path,
    report_path: Path | None = None,
    *,
    references: tuple[Path, ...] = (),
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite international diagnostic: {output}")

    source_rows, source_stats = read_source(source)
    exact_rows, exact_dropped, exact_conflicts = deduplicate(source_rows)
    held_references = reference_rows(references)
    nonoverlap_rows, near_overlap = remove_near_overlaps(exact_rows, held_references)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(nonoverlap_rows)
    conflict_groups = list(exact_conflicts) + list(near_conflicts)
    conflict_rows = [
        row
        for conflict in conflict_groups
        for row in conflict["candidates"]  # type: ignore[index]
    ]
    representatives, repeated_template_rows = one_per_family(clustered)
    representatives.sort(key=lambda row: str(row["id"]))

    if any(
        EMAIL_RE.search(str(row["text"]))
        or PHONE_LIKE_RE.search(str(row["text"]))
        or LONG_DIGIT_RE.search(str(row["text"]))
        or URL_RE.search(str(row["text"]))
        for row in representatives + conflict_rows
    ):
        raise ValueError("privacy-like value survived international robocall normalization")
    if len({str(row["id"]) for row in representatives}) != len(representatives):
        raise ValueError("international diagnostic IDs are not unique")
    if {str(row["label"]) for row in representatives} != {"SCAM", "UNCERTAIN"}:
        raise ValueError("international diagnostic must retain SCAM and UNCERTAIN rows")

    output.mkdir(parents=True)
    artifact = output / "international_robocalls_diagnostic.jsonl"
    quarantine = output / "label_conflicts_quarantine.jsonl"
    write_jsonl(artifact, representatives)
    write_jsonl(quarantine, conflict_groups)
    manifest: dict[str, object] = {
        "diagnostic_schema_version": 1,
        "purpose": (
            "fresh noncommercial real-robocall diagnostic; excluded from fitting, threshold "
            "selection, promotion tuning, commercial artifacts, and public row redistribution"
        ),
        "source": {
            "doi": f"https://doi.org/{SOURCE_DOI}",
            "zenodo_record": SOURCE_RECORD,
            "metadata_revision": SOURCE_METADATA_REVISION,
            "metadata_modified": SOURCE_MODIFIED,
            "license": LICENSE_NAME,
            "raw_sha256": SOURCE_SHA256,
            "publisher_md5": SOURCE_MD5,
            "publisher_reported_rows": {"international": 115, "us": 724},
            "archive_observed_rows": EXPECTED_ROWS,
            "row_count_discrepancy": (
                "the pinned CSVs each contain one more data row than the Zenodo description"
            ),
            "collection": "real inbound calls to an international honeypot",
            "annotation": (
                "each released recording was human-verified as a robocall; campaign and scam "
                "labels are publisher silver labels rather than adjudicated scam truth"
            ),
        },
        "policy": {
            "used_for_fitting": False,
            "used_for_threshold": False,
            "used_for_promotion_tuning": False,
            "commercial_use_allowed": False,
            "rows_redistributed": False,
            "safe_rows_supplied_by_source": 0,
            "individual_scam_ground_truth": False,
            "scam_campaigns_mapped_to_scam": sorted(SCAM_CAMPAIGNS),
            "spam_telemarketing_and_unclassified_mapped_to_uncertain": sorted(
                UNCERTAIN_CAMPAIGNS
            ),
            "one_representative_per_near_template_family": True,
            "raw_text_written_to_manifest": False,
        },
        "counts": source_stats
        | {
            "exact_duplicate_rows_removed": exact_dropped,
            "exact_conflict_groups_quarantined": len(exact_conflicts),
            "near_overlaps_with_references_removed": near_overlap,
            "near_conflict_groups_quarantined": len(near_conflicts),
            "same_label_near_template_rows_removed": repeated_template_rows,
            "reference_rows": len(held_references),
            "final_rows": len(representatives),
            "final_labels": dict(
                Counter(str(row["label"]) for row in representatives)
            ),
        },
        "near_template_stats": near_stats,
        "artifacts": {
            "diagnostic": {
                "path": str(artifact),
                "rows": len(representatives),
                "sha256": file_sha256(artifact),
            },
            "quarantine": {
                "path": str(quarantine),
                "conflict_groups": len(conflict_groups),
                "candidate_rows": len(conflict_rows),
                "sha256": file_sha256(quarantine),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/international_robocalls_text.zip"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/international_robocalls")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-audits/international-robocalls.json"),
    )
    parser.add_argument("--reference", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(
            f"missing international robocall source; run scripts/fetch_datasets.py: {args.source}"
        )
    print(
        json.dumps(
            build(
                args.source,
                args.output,
                args.report,
                references=tuple(args.reference),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
