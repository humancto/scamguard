#!/usr/bin/env python3
"""Build an evaluation-only SAFE-call benchmark from pinned AppTek text metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.build_dataset import (
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )
    from scripts.fetch_apptek_callcenter import LICENSE, REVISION, SOURCE_FILES, verify
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_dataset import (  # type: ignore[no-redef]
        EMAIL_RE,
        LONG_DIGIT_RE,
        PHONE_LIKE_RE,
        cluster_near_duplicates,
        deduplicate,
        make_row,
        read_jsonl,
        remove_near_overlaps,
        write_jsonl,
    )
    from fetch_apptek_callcenter import (  # type: ignore[no-redef]
        LICENSE,
        REVISION,
        SOURCE_FILES,
        verify,
    )

from scamguard.metrics import file_sha256

EXPECTED_TOP_KEYS = {"accent", "domain", "duration", "file_name", "segments"}
EXPECTED_SEGMENT_KEYS = {"end", "gender", "role", "speaker_id", "start", "text"}
EXPECTED_ROLES = {"agent", "customer"}
PARTITION_SALT = "scamguard-apptek-safe-calls-v1"
MAX_WINDOW_CHARS = 425
SELECTION_FRACTION = 0.20


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def clip_words(text: str, *, recent: bool, limit: int = MAX_WINDOW_CHARS) -> str:
    if len(text) <= limit:
        return text
    candidate = text[-limit:] if recent else text[:limit]
    if recent:
        _, separator, remainder = candidate.partition(" ")
        return (remainder if separator else candidate).strip()
    return candidate.rsplit(" ", 1)[0].strip()


def render_window(segments: list[dict[str, Any]], *, recent: bool) -> str:
    ordered = list(reversed(segments)) if recent else segments
    lines: list[str] = []
    total = 0
    for segment in ordered:
        role = str(segment["role"]).upper()
        body = " ".join(str(segment["text"]).split())
        if not body:
            continue
        prefix = f"{role}: "
        line = prefix + body
        addition = len(line) + int(bool(lines))
        if lines and total + addition > MAX_WINDOW_CHARS:
            break
        if not lines and addition > MAX_WINDOW_CHARS:
            line = prefix + clip_words(
                body,
                recent=recent,
                limit=MAX_WINDOW_CHARS - len(prefix),
            )
            addition = len(line)
        lines.append(line)
        total += addition
    if recent:
        lines.reverse()
    return "\n".join(lines)


def reference_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for path in paths:
        candidates = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen_paths or not candidate.is_file() or "quarantine" in candidate.name:
                continue
            seen_paths.add(resolved)
            rows.extend(read_jsonl(candidate))
    return rows


def read_source(
    raw: Path,
    *,
    source_files: dict[str, dict[str, object]] = SOURCE_FILES,
    expected_calls: int | None = 873,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    calls = 0
    seen_files: set[str] = set()
    privacy_counts: Counter[str] = Counter()
    accent_calls: Counter[str] = Counter()
    domain_calls: Counter[str] = Counter()
    segment_counts: list[int] = []
    durations: list[float] = []
    for expected_accent, expected in source_files.items():
        path = raw / f"{expected_accent}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing AppTek metadata: {path}")
        verify(path, expected)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            call = json.loads(line)
            if set(call) != EXPECTED_TOP_KEYS:
                raise ValueError(f"unexpected AppTek call schema: {path.name}:{line_number}")
            accent = str(call["accent"])
            domain = str(call["domain"])
            file_name = str(call["file_name"])
            segments = call["segments"]
            if accent != expected_accent or not domain or not file_name or file_name in seen_files:
                raise ValueError(f"invalid AppTek call identity: {path.name}:{line_number}")
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"invalid AppTek segments: {path.name}:{line_number}")
            seen_files.add(file_name)
            calls += 1
            accent_calls[accent] += 1
            domain_calls[domain] += 1
            segment_counts.append(len(segments))
            durations.append(float(call["duration"]))
            speaker_hashes: set[str] = set()
            previous_start = -1.0
            for segment in segments:
                if set(segment) != EXPECTED_SEGMENT_KEYS:
                    raise ValueError(f"unexpected AppTek segment schema: {file_name}")
                if segment["role"] not in EXPECTED_ROLES:
                    raise ValueError(f"unexpected AppTek role: {segment['role']!r}")
                start, end = float(segment["start"]), float(segment["end"])
                if start < previous_start or end < start:
                    raise ValueError(f"invalid AppTek segment timing: {file_name}")
                previous_start = start
                text = str(segment["text"])
                privacy_counts["email_like_segments"] += bool(EMAIL_RE.search(text))
                privacy_counts["phone_like_segments"] += bool(PHONE_LIKE_RE.search(text))
                privacy_counts["long_digit_segments"] += bool(LONG_DIGIT_RE.search(text))
                speaker_id = str(segment["speaker_id"]).strip()
                if not speaker_id:
                    raise ValueError(f"missing AppTek speaker ID: {file_name}")
                speaker_hashes.add(short_hash(f"{accent}:{speaker_id}"))

            call_hash = short_hash(f"{accent}:{file_name}")
            for window_kind, text in (
                ("early", render_window(segments, recent=False)),
                ("recent", render_window(segments, recent=True)),
            ):
                if len(text) < 80:
                    continue
                row = make_row(
                    text=text,
                    label="SAFE",
                    source="apptek_callcenter_dialogues",
                    source_label="legitimate_service_roleplay",
                    license_name=LICENSE,
                )
                if row is None:
                    continue
                row.update(
                    {
                        "source_call_hash": call_hash,
                        "source_speaker_hashes": sorted(speaker_hashes),
                        "source_window": window_kind,
                        "source_language": "English",
                        "source_accent": accent,
                        "source_domain": domain,
                        "provenance_class": "human_spontaneous_roleplay",
                        "naturally_occurring_communication": False,
                        "label_policy": "weak_safe_from_legitimate_service_roleplay_domain",
                        "privacy_normalization": (
                            "publisher roleplay plus ScamGuard email and phone normalization"
                        ),
                    }
                )
                rows.append(row)
    if expected_calls is not None and calls != expected_calls:
        raise ValueError(f"expected {expected_calls} AppTek calls, found {calls}")
    return rows, {
        "source_calls": calls,
        "candidate_windows": len(rows),
        "accent_calls": dict(accent_calls),
        "domain_calls": dict(domain_calls),
        "duration_seconds_min": min(durations),
        "duration_seconds_max": max(durations),
        "segments_per_call_min": min(segment_counts),
        "segments_per_call_max": max(segment_counts),
        "privacy_like_counts_before_normalization": dict(privacy_counts),
    }


def speaker_components(rows: list[dict[str, object]]) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_call: defaultdict[str, list[int]] = defaultdict(list)
    by_speaker: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_call[str(row["source_call_hash"])].append(index)
        for speaker_hash in row["source_speaker_hashes"]:  # type: ignore[union-attr]
            by_speaker[str(speaker_hash)].append(index)
    for group in (*by_call.values(), *by_speaker.values()):
        for index in group[1:]:
            union(group[0], index)
    components: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    return list(components.values())


def choose_selection_components(
    components: list[list[int]], rows: list[dict[str, object]]
) -> set[int]:
    if len(components) < 2:
        raise ValueError("AppTek source cannot be split into independent speaker components")
    def component_identity(members: list[int]) -> str:
        call_hashes = sorted({str(rows[index]["source_call_hash"]) for index in members})
        return ":".join(call_hashes)

    ranked = sorted(
        enumerate(components),
        key=lambda item: short_hash(PARTITION_SALT + ":" + component_identity(item[1]), 64),
    )
    total_rows = len(rows)
    target = round(total_rows * SELECTION_FRACTION)
    minimum = 3 if len(ranked) >= 5 else 1
    maximum = len(ranked) - 2 if len(ranked) >= 4 else len(ranked) - 1
    best: tuple[int, str, set[int]] | None = None
    if len(ranked) <= 20:
        for mask in range(1, 1 << len(ranked)):
            selected_positions = {
                position for position in range(len(ranked)) if mask & (1 << position)
            }
            if not minimum <= len(selected_positions) <= maximum:
                continue
            count = sum(len(ranked[position][1]) for position in selected_positions)
            identity = ":".join(str(position) for position in sorted(selected_positions))
            candidate = (abs(count - target), identity, selected_positions)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    else:
        selected_positions: set[int] = set()
        count = 0
        for position, (_component_index, members) in enumerate(ranked):
            if len(selected_positions) < minimum or abs(count + len(members) - target) < abs(
                count - target
            ):
                selected_positions.add(position)
                count += len(members)
        best = (abs(count - target), "greedy", selected_positions)
    if best is None:
        raise ValueError("unable to construct AppTek component partition")
    return {ranked[position][0] for position in best[2]}


def build(
    raw: Path,
    references: list[Path],
    output: Path,
    report_path: Path | None = None,
    *,
    source_files: dict[str, dict[str, object]] = SOURCE_FILES,
    expected_calls: int | None = 873,
) -> dict[str, object]:
    source_rows, source_stats = read_source(
        raw,
        source_files=source_files,
        expected_calls=expected_calls,
    )
    exact_rows, exact_dropped, conflicts = deduplicate(source_rows)
    if conflicts:
        raise ValueError("SAFE-only AppTek source unexpectedly produced an exact label conflict")
    existing_rows = reference_rows(references)
    nonoverlap_rows, near_overlap = remove_near_overlaps(exact_rows, existing_rows)
    clustered, near_conflicts, near_stats = cluster_near_duplicates(nonoverlap_rows)
    if near_conflicts:
        raise ValueError("SAFE-only AppTek source unexpectedly produced a near label conflict")

    representatives: dict[str, dict[str, object]] = {}
    for row in clustered:
        family = str(row["family_id"])
        current = representatives.get(family)
        if current is None or str(row["id"]) < str(current["id"]):
            representatives[family] = row
    rows = list(representatives.values())
    near_templates_removed = len(clustered) - len(rows)
    components = speaker_components(rows)
    selection_component_indexes = choose_selection_components(components, rows)
    finalized: list[dict[str, object]] = []
    for component_index, members in enumerate(components):
        component_keys = sorted(str(rows[index]["source_call_hash"]) for index in members)
        family_id = "apptek-component-" + short_hash("|".join(component_keys))
        split = "selection" if component_index in selection_component_indexes else "ood"
        for index in members:
            finalized.append(rows[index] | {"family_id": family_id, "split": split})
    finalized.sort(key=lambda row: str(row["id"]))
    split_rows = {
        split: [row for row in finalized if row["split"] == split]
        for split in ("selection", "ood")
    }
    if not all(split_rows.values()):
        raise ValueError("AppTek selection or OOD partition is empty")
    selection_speakers = {
        speaker
        for row in split_rows["selection"]
        for speaker in row["source_speaker_hashes"]  # type: ignore[union-attr]
    }
    ood_speakers = {
        speaker
        for row in split_rows["ood"]
        for speaker in row["source_speaker_hashes"]  # type: ignore[union-attr]
    }
    if selection_speakers & ood_speakers:
        raise ValueError("AppTek speaker identity crossed selection and OOD")
    if any(
        EMAIL_RE.search(str(row["text"]))
        or PHONE_LIKE_RE.search(str(row["text"]))
        or LONG_DIGIT_RE.search(str(row["text"]))
        for row in finalized
    ):
        raise ValueError("privacy-like value survived AppTek normalization")

    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "apptek_call_selection.jsonl"
    ood_path = output / "apptek_call_ood.jsonl"
    write_jsonl(selection_path, split_rows["selection"])
    write_jsonl(ood_path, split_rows["ood"])
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "source": {
            "dataset": "apptek-com/apptek_callcenter_dialogues",
            "revision": REVISION,
            "license": LICENSE,
            "collection": (
                "newly collected spontaneous English service-call roleplay; not real customer data"
            ),
            "publisher_designation": "evaluation and analysis only; training out of scope",
            "metadata_files": {
                accent: {
                    "bytes": expected["bytes"],
                    "sha256": expected["sha256"],
                }
                for accent, expected in source_files.items()
            },
        },
        "policy": {
            "label": "weak SAFE from legitimate service-roleplay domain",
            "used_for_fitting": False,
            "used_for_threshold": False,
            "selection_may_inform_candidate_selection": True,
            "ood_prediction_sealed_until_candidate_freeze": True,
            "partition": (
                "one representative per near-template cluster; shared-speaker/call components; "
                "deterministic closest-to-20-percent component subset"
            ),
            "audio_downloaded": False,
            "raw_text_written_to_manifest": False,
            "independent_row_label_review_complete": False,
        },
        "counts": source_stats
        | {
            "exact_duplicate_windows_removed": exact_dropped,
            "near_overlaps_with_existing_data_removed": near_overlap,
            "near_template_representatives": len(rows),
            "near_template_rows_removed": near_templates_removed,
            "speaker_components": len(components),
            "selection_components": len(selection_component_indexes),
            "ood_components": len(components) - len(selection_component_indexes),
            "selection_windows": len(split_rows["selection"]),
            "ood_windows": len(split_rows["ood"]),
            "selection_calls": len(
                {str(row["source_call_hash"]) for row in split_rows["selection"]}
            ),
            "ood_calls": len({str(row["source_call_hash"]) for row in split_rows["ood"]}),
            "selection_accents": dict(
                Counter(str(row["source_accent"]) for row in split_rows["selection"])
            ),
            "ood_accents": dict(Counter(str(row["source_accent"]) for row in split_rows["ood"])),
            "selection_domains": dict(
                Counter(str(row["source_domain"]) for row in split_rows["selection"])
            ),
            "ood_domains": dict(Counter(str(row["source_domain"]) for row in split_rows["ood"])),
        },
        "near_template_stats": near_stats,
        "artifacts": {
            "selection": {
                "path": str(selection_path),
                "sha256": file_sha256(selection_path),
            },
            "ood": {"path": str(ood_path), "sha256": file_sha256(ood_path)},
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/apptek_callcenter_dialogues")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[
            Path("data/processed"),
            Path("data/experiments/schema14-natural-dialogue/processed"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/apptek_callcenter")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/source-audits/apptek-callcenter.json"),
    )
    args = parser.parse_args()
    build(args.raw, args.reference, args.output, args.report)


if __name__ == "__main__":
    main()
