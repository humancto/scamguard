#!/usr/bin/env python3
"""Build a frozen SAFE-call diagnostic from the English Vystadial corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

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

SOURCE = "vystadial_openslr6_en"
SOURCE_REVISION = "b7454512a3547c829be201d9afceb64af8c28e61"
SOURCE_REPOSITORY = "BrunoHays/vystadial-telephony-en"
PRIMARY_SOURCE = "https://www.openslr.org/6/"
LICENSE_NAME = "CC-BY-SA-3.0"
EXPECTED_COLUMNS = (
    "audio_id",
    "license",
    "original_transcription",
    "processed_transcription",
    "transcription",
    "split",
    "dur",
    "sample_rate",
    "lang",
    "src",
    "channels",
    "n_segments",
    "build",
    "call_id",
    "spk",
)
DEFAULT_SAMPLE_SIZE = 1024
SAMPLE_SEED = "scamguard-vystadial-safe-diagnostic-v1"


@dataclass(frozen=True)
class ParquetSource:
    filename: str
    url: str
    sha256: str
    rows: int


SOURCES = (
    ParquetSource(
        filename="test-00000-of-00002.parquet",
        url=(
            "https://huggingface.co/datasets/BrunoHays/vystadial-telephony-en/"
            f"resolve/{SOURCE_REVISION}/data/test-00000-of-00002.parquet"
        ),
        sha256="399e6e1f13846b1bbad1ddb60df552c443137db56b86f7153d1a0b6e76b5844c",
        rows=24_127,
    ),
    ParquetSource(
        filename="test-00001-of-00002.parquet",
        url=(
            "https://huggingface.co/datasets/BrunoHays/vystadial-telephony-en/"
            f"resolve/{SOURCE_REVISION}/data/test-00001-of-00002.parquet"
        ),
        sha256="3c7108edc251d5f7d95c2bbab53c1716b7d9c2ebe6d6670142602cec2b24fb92",
        rows=24_127,
    ),
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reference_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in paths:
        candidates = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file() or "quarantine" in path.name:
                continue
            rows.extend(row for row in read_jsonl(path) if isinstance(row.get("text"), str))
    return rows


def read_source(source_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    privacy = Counter[str]()
    speaker_counts = Counter[str]()
    durations: list[float] = []
    seen_audio_ids: set[str] = set()
    seen_call_ids: set[str] = set()

    for spec in SOURCES:
        path = source_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned Vystadial shard: {path}")
        if file_sha256(path) != spec.sha256:
            raise ValueError(f"Vystadial shard differs from pinned revision: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != spec.rows:
            raise ValueError(
                f"expected {spec.rows} rows in {spec.filename}, found {parquet.metadata.num_rows}"
            )
        available = set(parquet.schema_arrow.names)
        if missing := sorted(set(EXPECTED_COLUMNS) - available):
            raise ValueError(f"missing Vystadial columns in {spec.filename}: {missing}")
        for source_row in parquet.read(columns=list(EXPECTED_COLUMNS)).to_pylist():
            if source_row["license"] != LICENSE_NAME:
                raise ValueError("unexpected Vystadial row license")
            if not (
                source_row["split"] == "test"
                and source_row["lang"] == "en"
                and source_row["src"] == "vystadial-openslr6-en"
                and source_row["sample_rate"] == 16_000
                and source_row["channels"] == 1
                and source_row["n_segments"] == 1
                and source_row["build"] == "utt_full"
            ):
                raise ValueError("unexpected Vystadial processing metadata")
            raw_text = str(source_row["original_transcription"] or "").strip()
            if raw_text != str(source_row["transcription"] or "").strip():
                raise ValueError("processed Vystadial transcript differs from upstream transcript")
            if str(source_row["processed_transcription"] or "").strip():
                raise ValueError(
                    "processed Vystadial transcript unexpectedly contains rewritten text"
                )
            audio_id = str(source_row["audio_id"] or "").strip()
            call_id = str(source_row["call_id"] or "").strip()
            speaker = str(source_row["spk"] or "").strip()
            if not raw_text or not audio_id or not call_id or not speaker:
                raise ValueError("missing Vystadial transcript identity")
            if audio_id in seen_audio_ids or call_id in seen_call_ids:
                raise ValueError("Vystadial source identities are not unique")
            seen_audio_ids.add(audio_id)
            seen_call_ids.add(call_id)
            privacy["rows_with_email"] += bool(EMAIL_RE.search(raw_text))
            privacy["rows_with_phone_like"] += bool(PHONE_LIKE_RE.search(raw_text))
            privacy["rows_with_long_digit_sequence"] += bool(LONG_DIGIT_RE.search(raw_text))
            privacy["rows_with_url"] += bool(URL_RE.search(raw_text))
            text = URL_RE.sub("<URL>", privacy_normalize_real_text(clean_text(raw_text)))
            row = make_row(
                text=text,
                label="SAFE",
                source=SOURCE,
                source_label="benign_restaurant_dialogue_system_call",
                license_name=LICENSE_NAME,
            )
            if row is None:
                privacy["rows_empty_after_normalization"] += 1
                continue
            speaker_hash = stable_hash(f"{SOURCE}:{speaker}")[:16]
            source_identity = f"{audio_id}:{call_id}:{raw_text}"
            row.update(
                {
                    "split": "vystadial_safe",
                    "source_record_sha256": stable_hash(source_identity),
                    "source_speaker_hash": speaker_hash,
                    "provenance_class": "real_human_telephone_dialogue_utterance",
                    "individual_safe_ground_truth": True,
                    "label_policy": "benign_restaurant_dialogue_collection_context",
                    "privacy_normalization": "url_email_phone_and_long_digit_placeholders_v1",
                }
            )
            rows.append(row)
            speaker_counts[speaker_hash] += 1
            durations.append(float(source_row["dur"]))

    return rows, {
        "source_rows": len(seen_audio_ids),
        "candidate_rows": len(rows),
        "unique_speakers": len(speaker_counts),
        "source_rows_per_speaker_min": min(speaker_counts.values()),
        "source_rows_per_speaker_max": max(speaker_counts.values()),
        "duration_seconds": round(sum(durations), 6),
        "duration_min_seconds": min(durations),
        "duration_max_seconds": max(durations),
        "privacy_findings_before_normalization": dict(privacy),
    }


def balanced_family_representatives(
    rows: list[dict[str, object]], *, sample_size: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family_id"])].append(row)

    selected: list[dict[str, object]] = []
    speaker_load = Counter[str]()
    family_order = sorted(groups, key=lambda family: stable_hash(f"{SAMPLE_SEED}:{family}"))
    for family in family_order:
        candidates = sorted(
            groups[family],
            key=lambda row: (
                speaker_load[str(row["source_speaker_hash"])],
                stable_hash(f"{SAMPLE_SEED}:{row['id']}"),
            ),
        )
        chosen = candidates[0]
        selected.append(chosen)
        speaker_load[str(chosen["source_speaker_hash"])] += 1

    queues: dict[str, deque[dict[str, object]]] = {}
    for speaker in sorted({str(row["source_speaker_hash"]) for row in selected}):
        speaker_rows = sorted(
            (row for row in selected if row["source_speaker_hash"] == speaker),
            key=lambda row: stable_hash(f"{SAMPLE_SEED}:sample:{row['id']}"),
        )
        queues[speaker] = deque(speaker_rows)
    speaker_order = sorted(queues, key=lambda speaker: stable_hash(f"{SAMPLE_SEED}:{speaker}"))
    capped: list[dict[str, object]] = []
    while len(capped) < min(sample_size, len(selected)):
        added = False
        for speaker in speaker_order:
            if queues[speaker] and len(capped) < sample_size:
                row = queues[speaker].popleft()
                capped.append(row | {"split": "vystadial_safe"})
                added = True
        if not added:
            break
    capped.sort(key=lambda row: str(row["id"]))
    final_speakers = Counter(str(row["source_speaker_hash"]) for row in capped)
    return capped, {
        "near_template_representatives": len(selected),
        "same_family_rows_removed": len(rows) - len(selected),
        "sample_cap_rows_removed": len(selected) - len(capped),
        "final_unique_speakers": len(final_speakers),
        "final_rows_per_speaker_min": min(final_speakers.values()),
        "final_rows_per_speaker_max": max(final_speakers.values()),
    }


def build(
    source_dir: Path,
    output: Path,
    report_path: Path | None = None,
    *,
    references: tuple[Path, ...] = (),
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Vystadial diagnostic: {output}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    source_rows, source_stats = read_source(source_dir)
    exact_rows, exact_dropped, exact_conflicts = deduplicate(source_rows)
    if exact_conflicts:
        raise ValueError("SAFE-only Vystadial source produced an exact label conflict")
    held_references = reference_rows(references)
    nonoverlap_rows, near_overlap = remove_near_overlaps(exact_rows, held_references)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(nonoverlap_rows)
    if near_conflicts:
        raise ValueError("SAFE-only Vystadial source produced a near-template label conflict")
    representatives, selection_stats = balanced_family_representatives(
        clustered, sample_size=sample_size
    )

    if any(
        EMAIL_RE.search(str(row["text"]))
        or PHONE_LIKE_RE.search(str(row["text"]))
        or LONG_DIGIT_RE.search(str(row["text"]))
        or URL_RE.search(str(row["text"]))
        for row in representatives
    ):
        raise ValueError("privacy-like value survived Vystadial normalization")
    if len({str(row["id"]) for row in representatives}) != len(representatives):
        raise ValueError("Vystadial diagnostic IDs are not unique")
    if {str(row["label"]) for row in representatives} != {"SAFE"}:
        raise ValueError("Vystadial diagnostic must contain SAFE rows only")

    output.mkdir(parents=True)
    artifact = output / "vystadial_safe_diagnostic.jsonl"
    write_jsonl(artifact, representatives)
    manifest: dict[str, object] = {
        "diagnostic_schema_version": 1,
        "purpose": (
            "fresh real-human legitimate telephone-utterance SAFE diagnostic; excluded from "
            "fitting, calibration, and promotion tuning until the frozen Stage 7 result is recorded"
        ),
        "source": {
            "primary": PRIMARY_SOURCE,
            "derivative_repository": SOURCE_REPOSITORY,
            "derivative_revision": SOURCE_REVISION,
            "license": LICENSE_NAME,
            "shards": [
                {
                    "filename": spec.filename,
                    "rows": spec.rows,
                    "sha256": spec.sha256,
                }
                for spec in SOURCES
            ],
            "collection": (
                "crowdworkers called a statistical telephone dialogue system to find a dining venue"
            ),
            "annotation": "upstream human orthographic transcript, unchanged by the derivative",
        },
        "policy": {
            "used_for_fitting": False,
            "used_for_threshold": False,
            "used_for_promotion_tuning": False,
            "commercial_use_allowed": True,
            "attribution_required": True,
            "share_alike_required": True,
            "audio_redistributed": False,
            "rows_redistributed": False,
            "individual_safe_ground_truth": True,
            "one_representative_per_near_template_family": True,
            "speaker_balanced_deterministic_cap": sample_size,
            "sample_seed": SAMPLE_SEED,
            "raw_text_written_to_manifest": False,
        },
        "counts": source_stats
        | {
            "exact_duplicate_rows_removed": exact_dropped,
            "near_overlaps_with_references_removed": near_overlap,
            "reference_rows": len(held_references),
            "final_rows": len(representatives),
            "final_labels": {"SAFE": len(representatives)},
        }
        | selection_stats,
        "near_template_stats": near_stats,
        "artifacts": {
            "diagnostic": {
                "path": str(artifact),
                "rows": len(representatives),
                "sha256": file_sha256(artifact),
            }
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
    parser.add_argument("--source", type=Path, default=Path("data/raw/vystadial"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/vystadial_safe")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-audits/vystadial-safe.json"),
    )
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source,
                args.output,
                args.report,
                references=tuple(args.reference),
                sample_size=args.sample_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
