#!/usr/bin/env python3
"""Build schema v19 with long, source-separated call windows and action pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from scamguard.metrics import file_sha256
from scamguard.signals import extract_signal_matches

try:
    from scripts.build_dataset import make_row, normalized
    from scripts.build_schema18_call_evidence_pairs import validate_pair_rows
    from scripts.build_taskmaster_hard_negatives import (
        EXPECTED_RAW_SHA256 as TASKMASTER_RAW_SHA256,
    )
    from scripts.build_taskmaster_hard_negatives import (
        privacy_normalize,
        render_latest_context,
    )
    from scripts.build_youtube_scam_calls import (
        EXPECTED_HEADER as YOUTUBE_EXPECTED_HEADER,
    )
    from scripts.build_youtube_scam_calls import (
        SOURCE_SHA256 as YOUTUBE_RAW_SHA256,
    )
    from scripts.build_youtube_scam_calls import archive_members
    from scripts.generate_call_evidence_pairs import (
        FORBIDDEN_SAFE_ENDING_CUES,
        HOLDOUT_SCENARIOS,
    )
    from scripts.generate_legitimate_call_openings import SCENARIOS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from build_dataset import make_row, normalized  # type: ignore[no-redef]
    from build_schema18_call_evidence_pairs import (  # type: ignore[no-redef]
        validate_pair_rows,
    )
    from build_taskmaster_hard_negatives import (  # type: ignore[no-redef]
        EXPECTED_RAW_SHA256 as TASKMASTER_RAW_SHA256,
    )
    from build_taskmaster_hard_negatives import (
        privacy_normalize,
        render_latest_context,
    )
    from build_youtube_scam_calls import (  # type: ignore[no-redef]
        EXPECTED_HEADER as YOUTUBE_EXPECTED_HEADER,
    )
    from build_youtube_scam_calls import (  # type: ignore[no-redef]
        SOURCE_SHA256 as YOUTUBE_RAW_SHA256,
    )
    from build_youtube_scam_calls import archive_members  # type: ignore[no-redef]
    from generate_call_evidence_pairs import (  # type: ignore[no-redef]
        FORBIDDEN_SAFE_ENDING_CUES,
        HOLDOUT_SCENARIOS,
    )
    from generate_legitimate_call_openings import SCENARIOS  # type: ignore[no-redef]

SCHEMA_VERSION = 19
LONG_PAIR_SOURCE = "scamguard_synthetic_call_window_pairs_v1"
LONG_PAIR_GENERATOR_VERSION = 1
TASKMASTER_LONG_MAX_CHARS = 1000
YOUTUBE_LONG_MAX_CHARS = 1000
YOUTUBE_SOURCE = "youtube_scam_calls_cc0"
TASKMASTER_SOURCE = "taskmaster1_woz_dialogues"

PRESERVED_FILES = (
    "dev.jsonl",
    "test.jsonl",
    "ood_financial.jsonl",
    "ood_wspr.jsonl",
    "forum_validation.jsonl",
    "ood_forum.jsonl",
    "ood_azsc.jsonl",
    "quarantine_label_conflicts.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def shared_long_extension(row: dict[str, object]) -> str:
    """Create label-invariant, scenario-specific history before the final action."""
    scenario = str(row["scenario"])
    spec = SCENARIOS[scenario]
    variant = int(short_hash(str(row["pair_id"]), 8), 16)
    desk = spec["desk"][variant % len(spec["desk"])]
    topic = spec["topic"][(variant // 3) % len(spec["topic"])]
    detail = spec["detail"][(variant // 7) % len(spec["detail"])]
    request = spec["request"][(variant // 11) % len(spec["request"])]
    return "\n".join(
        (
            (
                "AGENT: Before describing any next step, I am reviewing the service "
                f"history for {topic}."
            ),
            (
                f"CUSTOMER: That is fine. The request was opened through the {desk}, "
                "and I have time to review it carefully."
            ),
            f"AGENT: The current note says that {detail.casefold()}.",
            (
                "CUSTOMER: Yes, that matches what I was told earlier. I do not want to "
                "skip any ordinary review step."
            ),
            (
                "AGENT: I am checking the date, status, and contact note already attached "
                "to the request."
            ),
            (
                "CUSTOMER: Please explain what the record shows before asking me to do "
                "anything outside this conversation."
            ),
            f"AGENT: The record also includes your earlier request: {request}",
            (
                "CUSTOMER: Correct. I am listening, but I will decide what to do after I "
                "understand the proposed action."
            ),
            (
                "AGENT: There is no need to rush while I finish reading the existing "
                "notes and summarize the options."
            ),
            (
                "CUSTOMER: Good. I prefer a clear explanation and a normal way to verify "
                "the request independently."
            ),
            (
                "AGENT: I have finished the review and can now describe the specific next "
                "step associated with this case."
            ),
            (
                "CUSTOMER: All right. Tell me exactly what action you are proposing and "
                "why it is needed."
            ),
        )
    )


def expand_long_pairs(source_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Length-match pairs while leaving the final action as the only label difference."""
    validate_pair_rows(source_rows)
    rows: list[dict[str, object]] = []
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["pair_id"])].append(row)
    for parent_pair_id, pair in sorted(grouped.items()):
        extension = shared_long_extension(pair[0])
        pair_id = "call-window-pair-" + short_hash(
            f"v{LONG_PAIR_GENERATOR_VERSION}:{parent_pair_id}:{extension}"
        )
        contexts: set[str] = set()
        for source_row in pair:
            original_context, ending = str(source_row["text"]).rsplit("\n", 1)
            context = f"{original_context}\n{extension}"
            contexts.add(context)
            label = str(source_row["label"])
            if label == "SAFE" and any(
                cue in ending.casefold() for cue in FORBIDDEN_SAFE_ENDING_CUES
            ):
                raise ValueError(f"risk action leaked into long SAFE ending: {parent_pair_id}")
            text = f"{context}\n{ending}"
            if label == "SCAM" and not extract_signal_matches(text):
                raise ValueError(f"long SCAM pair lacks a safety signal: {parent_pair_id}")
            row = dict(source_row)
            row.update(
                {
                    "id": f"{pair_id}-{'safe' if label == 'SAFE' else 'scam'}",
                    "text": text,
                    "source": LONG_PAIR_SOURCE,
                    "source_label": (
                        "synthetic_legitimate_long_call_pair"
                        if label == "SAFE"
                        else "synthetic_scam_long_call_pair"
                    ),
                    "family_id": (
                        f"synthetic:call_window_pair:{source_row['scenario']}:"
                        f"{source_row['dialogue_structure']}:{source_row['context_frame']}:"
                        f"{source_row['risk_mechanism']}:v{LONG_PAIR_GENERATOR_VERSION}"
                    ),
                    "pair_id": pair_id,
                    "parent_pair_id": parent_pair_id,
                    "generator_version": LONG_PAIR_GENERATOR_VERSION,
                    "shared_context_sha256": hashlib.sha256(
                        context.encode("utf-8")
                    ).hexdigest(),
                    "context_window_curriculum": "long_shared_history_before_final_action",
                    "selection_signal": (
                        "schema18 perfect short-pair separation but failed long-dialogue "
                        "retention; no external benchmark text copied"
                    ),
                }
            )
            rows.append(row)
        if len(contexts) != 1:
            raise ValueError(f"long pair context differs: {parent_pair_id}")
    return sorted(rows, key=lambda row: str(row["id"]))


def build_taskmaster_long_rows(
    raw_path: Path,
    seed_rows: list[dict[str, object]],
    split: str,
) -> list[dict[str, object]]:
    if file_sha256(raw_path) != TASKMASTER_RAW_SHA256:
        raise ValueError("Taskmaster raw artifact differs from the pinned revision")
    dialogues = json.loads(raw_path.read_text(encoding="utf-8"))
    by_id = {
        str(dialogue.get("conversation_id")): dialogue
        for dialogue in dialogues
        if isinstance(dialogue, dict)
    }
    output: list[dict[str, object]] = []
    for seed in seed_rows:
        family_id = str(seed.get("family_id", ""))
        if not family_id.startswith("taskmaster1:"):
            raise ValueError(f"unexpected Taskmaster family: {family_id}")
        conversation_id = family_id.removeprefix("taskmaster1:")
        dialogue = by_id.get(conversation_id)
        if dialogue is None or not isinstance(dialogue.get("utterances"), list):
            raise ValueError(f"Taskmaster conversation is missing: {conversation_id}")
        text = render_latest_context(
            dialogue["utterances"], max_chars=TASKMASTER_LONG_MAX_CHARS
        )
        if len(text) < len(str(seed["text"])):
            raise ValueError(f"Taskmaster long window shrank: {conversation_id}")
        if normalized(text) == normalized(str(seed["text"])):
            continue
        row = dict(seed)
        row.update(
            {
                "id": "tm1-long-" + short_hash(conversation_id),
                "text": privacy_normalize(text),
                "split": split,
                "source_window": "recent_complete_turns_long",
                "context_policy": (
                    f"latest_complete_turns_capped_at_{TASKMASTER_LONG_MAX_CHARS}_characters"
                ),
            }
        )
        output.append(row)
    if len({str(row["id"]) for row in output}) != len(output):
        raise ValueError("Taskmaster long windows contain duplicate IDs")
    return sorted(output, key=lambda row: str(row["id"]))


def validate_youtube_recent(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if (
            row.get("source") != YOUTUBE_SOURCE
            or row.get("license") != "CC0-1.0"
            or row.get("label") != "SCAM"
            or row.get("split") != "train"
            or row.get("source_window") != "recent"
            or row.get("is_synthetic") is not False
        ):
            raise ValueError(f"unexpected YouTube recent-window row: {row.get('id')}")


def clip_recent(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    candidate = text[-max_chars:]
    _, separator, remainder = candidate.partition(" ")
    return (remainder if separator else candidate).strip()


def build_youtube_long_rows(
    raw_path: Path,
    seed_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Extend only source records already assigned to the publisher-safe train partition."""
    if file_sha256(raw_path) != YOUTUBE_RAW_SHA256:
        raise ValueError("YouTube scam-call archive differs from the pinned version")
    seed_by_record: dict[str, dict[str, object]] = {}
    existing_keys: set[str] = set()
    for seed in seed_rows:
        record_id = str(seed.get("source_record_id", ""))
        if not record_id:
            raise ValueError(f"YouTube train row lacks a source record: {seed.get('id')}")
        seed_by_record.setdefault(record_id, seed)
        existing_keys.add(normalized(str(seed["text"])))

    contents: dict[str, str] = {}
    with zipfile.ZipFile(raw_path) as archive:
        members = archive_members(archive)
        raw = archive.read(members["FullTranscriptData.csv"])
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != YOUTUBE_EXPECTED_HEADER:
            raise ValueError(f"unexpected YouTube source header: {reader.fieldnames!r}")
        for source_row in reader:
            contents[str(source_row["ID"]).strip()] = str(source_row["Content"]).strip()

    output: list[dict[str, object]] = []
    for record_id, seed in sorted(seed_by_record.items()):
        if record_id not in contents:
            raise ValueError(f"YouTube source record is missing: {record_id}")
        candidate = clip_recent(contents[record_id], YOUTUBE_LONG_MAX_CHARS)
        built = make_row(
            text=candidate,
            label="SCAM",
            source=YOUTUBE_SOURCE,
            source_label="publisher_scam_call",
            license_name="CC0-1.0",
        )
        if built is None or normalized(str(built["text"])) in existing_keys:
            continue
        row = dict(seed)
        row.update(
            {
                "id": built["id"],
                "text": built["text"],
                "source_window": "recent_long",
                "context_policy": (
                    f"recent_whitespace_complete_{YOUTUBE_LONG_MAX_CHARS}_character_window"
                ),
            }
        )
        output.append(row)
        existing_keys.add(normalized(str(row["text"])))
    if len({str(row["id"]) for row in output}) != len(output):
        raise ValueError("YouTube long windows contain duplicate IDs")
    return sorted(output, key=lambda row: str(row["id"]))


def build(
    parent: Path,
    pair_data: Path,
    pair_manifest_path: Path,
    taskmaster_raw: Path,
    taskmaster_train: Path,
    taskmaster_validation: Path,
    taskmaster_manifest_path: Path,
    youtube_train: Path,
    youtube_raw: Path,
    youtube_manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite schema-v19 output: {output}")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("schema_version") != 14:
        raise ValueError("schema-v19 parent must be schema version 14")

    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(pair_data) != pair_manifest.get("sha256"):
        raise ValueError("pair artifact differs from its generator manifest")
    long_pairs = expand_long_pairs(read_jsonl(pair_data))
    holdouts = set(HOLDOUT_SCENARIOS)
    pair_train = [row | {"split": "train"} for row in long_pairs if row["scenario"] not in holdouts]
    pair_validation = [
        row | {"split": "validation"} for row in long_pairs if row["scenario"] in holdouts
    ]

    taskmaster_manifest = json.loads(taskmaster_manifest_path.read_text(encoding="utf-8"))
    taskmaster_artifacts = taskmaster_manifest.get("artifacts", {})
    if file_sha256(taskmaster_train) != taskmaster_artifacts.get("train", {}).get("sha256"):
        raise ValueError("Taskmaster train artifact differs from its manifest")
    if (
        file_sha256(taskmaster_validation)
        != taskmaster_artifacts.get("validation", {}).get("sha256")
    ):
        raise ValueError("Taskmaster validation artifact differs from its manifest")
    taskmaster_train_long = build_taskmaster_long_rows(
        taskmaster_raw, read_jsonl(taskmaster_train), "train"
    )
    taskmaster_validation_long = build_taskmaster_long_rows(
        taskmaster_raw, read_jsonl(taskmaster_validation), "validation"
    )
    if {str(row["family_id"]) for row in taskmaster_train_long} & {
        str(row["family_id"]) for row in taskmaster_validation_long
    }:
        raise ValueError("Taskmaster family crosses train and long-window validation")

    youtube_manifest = json.loads(youtube_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(youtube_train) != youtube_manifest.get("artifacts", {}).get("train", {}).get(
        "sha256"
    ):
        raise ValueError("YouTube train artifact differs from its manifest")
    youtube_recent = [
        row for row in read_jsonl(youtube_train) if row.get("source_window") == "recent"
    ]
    validate_youtube_recent(youtube_recent)
    youtube_long = build_youtube_long_rows(youtube_raw, read_jsonl(youtube_train))

    parent_train = read_jsonl(parent / "train.jsonl")
    increments = taskmaster_train_long + youtube_recent + youtube_long + pair_train
    all_ids = [str(row["id"]) for row in parent_train + increments]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("schema-v19 increment has a duplicate or parent-colliding ID")

    output.mkdir(parents=True)
    combined_train = parent_train + sorted(increments, key=lambda row: str(row["id"]))
    write_jsonl(output / "train.jsonl", combined_train)
    write_jsonl(output / "call_pair_validation.jsonl", pair_validation)
    write_jsonl(output / "call_window_validation.jsonl", taskmaster_validation_long)
    for filename in PRESERVED_FILES:
        source_path = parent / filename
        if source_path.is_file():
            shutil.copy2(source_path, output / filename)

    development_rows = list(combined_train)
    for split in ("dev", "test"):
        development_rows.extend(read_jsonl(output / f"{split}.jsonl"))
    counts = dict(parent_manifest["counts"])
    counts.update(
        {
            "train": len(combined_train),
            "call_pair_validation": len(pair_validation),
            "call_window_validation": len(taskmaster_validation_long),
        }
    )
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "labels": dict(Counter(str(row["label"]) for row in development_rows)),
            "sources": dict(Counter(str(row["source"]) for row in development_rows)),
            "parent": {
                "schema_version": 14,
                "manifest_sha256": file_sha256(parent_manifest_path),
                "train_sha256": file_sha256(parent / "train.jsonl"),
            },
            "schema19_increment": {
                "long_pair_source": LONG_PAIR_SOURCE,
                "pair_source_data_sha256": file_sha256(pair_data),
                "pair_source_manifest_sha256": file_sha256(pair_manifest_path),
                "pair_train_rows": len(pair_train),
                "pair_train_families": len({str(row["pair_id"]) for row in pair_train}),
                "pair_validation_rows": len(pair_validation),
                "pair_validation_families": len(
                    {str(row["pair_id"]) for row in pair_validation}
                ),
                "taskmaster_long_train_rows": len(taskmaster_train_long),
                "taskmaster_long_validation_rows": len(taskmaster_validation_long),
                "taskmaster_raw_sha256": file_sha256(taskmaster_raw),
                "taskmaster_manifest_sha256": file_sha256(taskmaster_manifest_path),
                "youtube_recent_train_rows": len(youtube_recent),
                "youtube_long_train_rows": len(youtube_long),
                "youtube_raw_sha256": file_sha256(youtube_raw),
                "youtube_manifest_sha256": file_sha256(youtube_manifest_path),
                "pair_exposure": "one long-context copy per semantic pair family",
                "taskmaster_window_policy": (
                    f"latest complete turns capped at {TASKMASTER_LONG_MAX_CHARS} characters"
                ),
                "licenses": {
                    LONG_PAIR_SOURCE: "Apache-2.0",
                    TASKMASTER_SOURCE: "CC-BY-4.0",
                    YOUTUBE_SOURCE: "CC0-1.0",
                },
                "apptek_rows_used_for_fitting": 0,
                "bothbosu_rows_used_for_fitting": 0,
                "apptek_ood_opened": False,
                "bothbosu_ood_opened": False,
                "youtube_ood_opened": False,
                "moz_holdout_opened": False,
            },
            "preserved_parent_artifacts": {
                filename: {
                    "sha256": file_sha256(output / filename),
                    "byte_identical_to_parent": file_sha256(output / filename)
                    == file_sha256(parent / filename),
                }
                for filename in PRESERVED_FILES
                if (output / filename).is_file()
            },
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("data/experiments/schema14-natural-dialogue/processed"),
    )
    parser.add_argument(
        "--pair-data", type=Path, default=Path("data/generated/call_evidence_pairs_v2.jsonl")
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("data/generated/call_evidence_pairs_v2_manifest.json"),
    )
    parser.add_argument(
        "--taskmaster-raw",
        type=Path,
        default=Path("data/raw/taskmaster1_woz_dialogues.json"),
    )
    parser.add_argument(
        "--taskmaster-train",
        type=Path,
        default=Path("data/generated/taskmaster_safe_train.jsonl"),
    )
    parser.add_argument(
        "--taskmaster-validation",
        type=Path,
        default=Path("data/external/taskmaster/taskmaster_validation.jsonl"),
    )
    parser.add_argument(
        "--taskmaster-manifest",
        type=Path,
        default=Path("data/external/taskmaster/manifest.json"),
    )
    parser.add_argument(
        "--youtube-train",
        type=Path,
        default=Path("data/external/youtube_scam_calls/youtube_scam_train.jsonl"),
    )
    parser.add_argument(
        "--youtube-raw",
        type=Path,
        default=Path("data/raw/youtube_scam_calls_v2.zip"),
    )
    parser.add_argument(
        "--youtube-manifest",
        type=Path,
        default=Path("data/external/youtube_scam_calls/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/schema19-call-windows/processed"),
    )
    args = parser.parse_args()
    build(
        args.parent,
        args.pair_data,
        args.pair_manifest,
        args.taskmaster_raw,
        args.taskmaster_train,
        args.taskmaster_validation,
        args.taskmaster_manifest,
        args.youtube_train,
        args.youtube_raw,
        args.youtube_manifest,
        args.output,
    )


if __name__ == "__main__":
    main()
