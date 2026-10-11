from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from scamguard.metrics import file_sha256
from scripts.import_blind_audit import import_returned_audit, load_and_verify_bundle
from scripts.prepare_ettin_review import (
    LABEL_TARGETS,
    PHONE_SOURCE,
    PINNED_HASHES,
    prepare_package,
    select_rows,
    stratum,
)
from scripts.verify_imported_blind_audit import verify_imported_audit


def frame_fixture() -> list[dict[str, object]]:
    rows = []
    for label in LABEL_TARGETS:
        for index in range(300):
            identifier = f"fixture-{label}-{index}"
            rows.append(
                {
                    "id": identifier,
                    "text": identifier,
                    "source": "fixture-source",
                    "label": label,
                    "dataset_split": "train" if index % 2 else "dev",
                }
            )
    for label in ("SAFE", "SCAM"):
        for index in range(50):
            identifier = f"fixture-phone-{label}-{index}"
            rows.append(
                {
                    "id": identifier,
                    "text": identifier,
                    "source": PHONE_SOURCE,
                    "label": label,
                    "dataset_split": "train",
                }
            )
    for domain in ("airline", "fastfood", "finance", "insurance", "media", "software"):
        for index in range(12):
            identifier = f"fixture-multidogo-human-{domain}-{index}"
            rows.append(
                {
                    "id": identifier,
                    "text": identifier,
                    "source": "multidogo_human_service_dialogues",
                    "source_domain": domain,
                    "label": "SAFE",
                    "dataset_split": "train",
                }
            )
    for domain in ("airline", "fastfood", "finance", "media"):
        for label in LABEL_TARGETS:
            for index in range(12):
                identifier = f"fixture-multidogo-grounded-{domain}-{label}-{index}"
                rows.append(
                    {
                        "id": identifier,
                        "text": identifier,
                        "source": "scamguard_multidogo_grounded_evidence_states_v1",
                        "source_domain": domain,
                        "label": label,
                        "dataset_split": "train",
                    }
                )
    return rows


def data_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"schema_version":25}\n')
    rows = frame_fixture()
    for split in ("train", "dev"):
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["dataset_split"] == split)
        )
    # Any accidental attempt to read an evaluation split must fail JSON parsing.
    (data / "test.jsonl").write_text("DO NOT READ HELD-OUT LABELS")
    return data, {name: file_sha256(data / name) for name in PINNED_HASHES}


def test_selection_is_deterministic_exact_and_covers_phone_and_domains() -> None:
    rows = frame_fixture()
    chosen = select_rows(rows)
    assert chosen == select_rows(list(reversed(rows)))
    assert len(chosen) == len({row["id"] for row in chosen}) == 635
    assert Counter(row["label"] for row in chosen) == LABEL_TARGETS
    available = Counter(stratum(row) for row in rows)
    sampled = Counter(stratum(row) for row in chosen)
    assert set(available) == set(sampled)
    assert all(sampled[key] >= min(6, count) for key, count in available.items())
    phone = Counter(row["label"] for row in chosen if row["source"] == PHONE_SOURCE)
    assert phone["SAFE"] >= 24
    assert phone["SCAM"] >= 24


@pytest.mark.parametrize("change", ["duplicate", "heldout", "invalid_label", "insufficient"])
def test_rejects_invalid_sampling_frames(change: str) -> None:
    rows = frame_fixture()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "heldout":
        rows[0]["dataset_split"] = "test"
    elif change == "invalid_label":
        rows[0]["label"] = "spam"
    else:
        rows = [row for row in rows if row["label"] != "UNCERTAIN"]
    with pytest.raises(ValueError):
        select_rows(rows)


def test_new_schema25_bundle_round_trips_through_existing_importer(tmp_path: Path) -> None:
    data, hashes = data_fixture(tmp_path)
    output = tmp_path / "review"
    preparation = tmp_path / "preparation.json"
    result = prepare_package(data, output, preparation, expected_hashes=hashes)
    assert result["experiment_id"] == "sg-ettin150m-bounded-v1"
    assert result["independent_human_complete_rows"] == 0
    assert result["release_gate_passed"] is False
    assert set(result["source_files"]) == {"train.jsonl", "dev.jsonl"}
    bundle = Path(result["bundle"]["bundle"])
    assert load_and_verify_bundle(bundle)["selected_rows"] == 635
    canonical = output / "label-audit.csv"
    manifest = output / "label-audit.manifest.json"
    with canonical.open(newline="") as handle:
        canonical_rows = list(csv.DictReader(handle))
    assert all(
        not row[field]
        for row in canonical_rows
        for field in ("auditor_label", "label_correct", "contains_sensitive_data", "notes")
    )
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "review.py",
            "scamguard_blind_audit.csv",
            "scamguard_blind_audit.manifest.json",
        }
        blind = list(
            csv.DictReader(archive.read("scamguard_blind_audit.csv").decode().splitlines())
        )
    assert set(blind[0]) == {"id", "text", "auditor_label", "contains_sensitive_data", "notes"}
    assert all(not row["auditor_label"] and not row["contains_sensitive_data"] for row in blind)
    # Disposable synthetic fixture only: never create a claimed human review of real data.
    fixture_labels = {row["text"]: row["label"] for row in canonical_rows}
    returned = tmp_path / "fixture-return.csv"
    with returned.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind[0]))
        writer.writeheader()
        writer.writerows(
            row | {"auditor_label": fixture_labels[row["text"]], "contains_sensitive_data": "no"}
            for row in blind
        )
    reviewed = tmp_path / "fixture-reviewed.csv"
    import_report = tmp_path / "fixture-report.json"
    imported = import_returned_audit(returned, bundle, canonical, manifest, reviewed, import_report)
    assert imported["release_gate_passed"] is True
    verified = verify_imported_audit(
        returned_csv=returned,
        bundle_path=bundle,
        canonical_audit=canonical,
        canonical_manifest=manifest,
        reviewed_audit=reviewed,
        report_path=import_report,
    )
    assert verified["passed"] is True
    with pytest.raises(ValueError, match="overwrite"):
        prepare_package(data, output, preparation, expected_hashes=hashes)


def test_rejects_changed_pinned_frame_before_any_artifact_write(tmp_path: Path) -> None:
    data, hashes = data_fixture(tmp_path)
    (data / "train.jsonl").write_text((data / "train.jsonl").read_text() + "\n")
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="frozen schema-25 frame"):
        prepare_package(data, output, tmp_path / "report.json", expected_hashes=hashes)
    assert not output.exists()
