from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import build_ettin_v3_corpus as corpus


def make_row(source: str, index: int, split: str = "train") -> dict[str, object]:
    return {
        "id": f"row-{split}-{index}",
        "text": f"Example {split} message number {index} about routine support.",
        "family_id": f"family-{split}-{index}",
        "label": "SAFE",
        "category": "NONE",
        "source": source,
        "license": (corpus.ALLOWED_SOURCES | corpus.EXCLUDED_SOURCES)[source],
        "split": split,
    }


@pytest.fixture
def frozen_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sources = corpus.ALLOWED_SOURCES | corpus.EXCLUDED_SOURCES
    train = [make_row(source, index) for index, source in enumerate(sources)]
    dev = [make_row("scamguard_synthetic_v5", 100, "dev")]

    def write_inputs(train_rows=None, dev_rows=None, inventory_change=None):
        rows = {"train": train if train_rows is None else train_rows,
                "dev": dev if dev_rows is None else dev_rows}
        payloads = {}
        for split, values in rows.items():
            payloads[split] = b"".join(
                (json.dumps(row, ensure_ascii=False, separators=(",", ": ")) + "\n").encode()
                for row in values
            )
        payloads["parent_manifest"] = corpus.json_bytes({"fixture": True})
        payloads["source_registry"] = corpus.json_bytes([])
        pins = {
            key: (corpus.INPUT_PINS[key][0], corpus.digest(payload))
            for key, payload in payloads.items()
        }
        inventory = {
            "data": {
                "train": pins["train"][0], "dev": pins["dev"][0],
                "train_rows": len(rows["train"]), "dev_rows": len(rows["dev"]),
            },
            "data_sha256": {
                "train": pins["train"][1], "dev": pins["dev"][1],
                "manifest": pins["parent_manifest"][1],
            },
            "source_registry": {"path": pins["source_registry"][0],
                                "sha256": pins["source_registry"][1]},
            "sources": [
                {
                    "id": source, "license_declared": license_name,
                    "rows": sum(row["source"] == source for row in rows["train"]),
                    "dev_rows": sum(row["source"] == source for row in rows["dev"]),
                    "selected_train_ids_sha256": corpus.ids_digest(
                        [row for row in rows["train"] if row["source"] == source]
                    ),
                }
                for source, license_name in sources.items()
            ],
        }
        if inventory_change:
            inventory_change(inventory)
        payloads["source_inventory"] = corpus.json_bytes(inventory)
        pins["source_inventory"] = (corpus.INPUT_PINS["source_inventory"][0],
                                    corpus.digest(payloads["source_inventory"]))
        for key, payload in payloads.items():
            path = tmp_path / pins[key][0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        monkeypatch.setattr(corpus, "INPUT_PINS", pins)
        monkeypatch.setattr(corpus, "PRIVACY_CODE_PINS", {})
        return rows, payloads

    write_inputs()
    return tmp_path, train, dev, write_inputs


def test_filter_is_byte_preserving_deterministic_and_never_release_clear(frozen_inputs):
    root, train, dev, _ = frozen_inputs
    before = {name: (root / ref[0]).read_bytes() for name, ref in corpus.INPUT_PINS.items()}
    report = corpus.build(root)
    output = root / corpus.OUTPUT
    kept_lines = [
        line for row, line in zip(train, before["train"].splitlines(keepends=True), strict=True)
        if row["source"] in corpus.ALLOWED_SOURCES
    ]
    assert (output / "train.jsonl").read_bytes() == b"".join(kept_lines)
    assert (output / "dev.jsonl").read_bytes() == before["dev"]
    assert report["splits"]["train"]["rows"] == 12
    assert report["excluded"]["train"]["rows"] == 2
    assert report["splits"]["dev"]["rows"] == len(dev)
    assert report["research_eligible"] is True
    assert report["data_integrity_preflight"] == "passed"
    assert report["release_authorized"] is report["production_ready"] is False
    assert report["human_review_status"] == "pending_new_corpus_bound_audit"
    assert report["requires_fresh_base_weights"] is True
    assert report["filter"]["new_rows_added"] == 0
    assert report["privacy_audit"]["splits"]["train"]["text_modified"] is False
    for name, ref in corpus.INPUT_PINS.items():
        assert (root / ref[0]).read_bytes() == before[name]
    serialized = json.dumps(report)
    for row in train + dev:
        assert row["id"] not in serialized
        assert row["text"] not in serialized
        assert row["family_id"] not in serialized
    # Alternate output is a new artifact; row bytes and all data hashes remain identical.
    other = corpus.build(root, output="data/experiments/repeat/processed",
                         report="reports/data/repeat.json")
    assert other["splits"]["train"]["sha256"] == report["splits"]["train"]["sha256"]
    assert other["splits"]["dev"]["sha256"] == report["splits"]["dev"]["sha256"]


@pytest.mark.parametrize("key", list(corpus.INPUT_PINS))
def test_input_hash_mismatch_fails_before_any_write(frozen_inputs, key):
    root, _, _, _ = frozen_inputs
    with (root / corpus.INPUT_PINS[key][0]).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="pinned input hash mismatch"):
        corpus.build(root)
    assert not (root / corpus.OUTPUT).exists()
    assert not (root / corpus.REPORT).exists()


@pytest.mark.parametrize("target", [corpus.OUTPUT, corpus.REPORT])
def test_existing_outputs_are_not_overwritten(frozen_inputs, target):
    root, _, _, _ = frozen_inputs
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        corpus.build(root)
    assert path.read_bytes() == b"preserve me"


@pytest.mark.parametrize("field,value,error", [
    ("source", "new_unreviewed_source", "unknown source"),
    ("license", "CC0-1.0", "source license"),
    ("label", "SPAM", "invalid label"),
    ("split", "test", "source split"),
    ("text", "", "missing/non-string"),
    ("family_id", None, "missing/non-string"),
])
def test_unadmitted_or_malformed_rows_fail(frozen_inputs, field, value, error):
    root, train, _, write = frozen_inputs
    train[0][field] = value
    write(train)
    with pytest.raises(ValueError, match=error):
        corpus.build(root)
    assert not (root / corpus.OUTPUT).exists()


@pytest.mark.parametrize("kind", ["id", "text", "normalized_text"])
def test_duplicate_rows_fail(frozen_inputs, kind):
    root, train, _, write = frozen_inputs
    if kind == "normalized_text":
        train[1]["text"] = "  " + train[0]["text"].upper() + "\t"
    else:
        train[1][kind] = train[0][kind]
    write(train)
    with pytest.raises(ValueError, match="duplicate"):
        corpus.build(root)


@pytest.mark.parametrize("kind", ["id", "text", "family_id", "near_template_family_id"])
def test_split_overlap_fails(frozen_inputs, kind):
    root, train, dev, write = frozen_inputs
    dev[0][kind] = train[0][kind if kind != "near_template_family_id" else "family_id"]
    write(train, dev)
    with pytest.raises(ValueError, match="train/dev overlap"):
        corpus.build(root)


@pytest.mark.parametrize("ancestry", [
    {"parent_source": "shakeleoatmeal_phone_scam_synthetic"},
    {"ancestor": {"repository": "shakeleoatmeal/phone-scam-detection-synthetic"}},
    {"sources": ["youtube_scam_calls_cc0"]},
    {"parent_family": "phone-scam-near-example"},
    {"derivation": "youtube-call-example"},
])
def test_embedded_excluded_provenance_is_rejected(frozen_inputs, ancestry):
    root, train, _, write = frozen_inputs
    train[0].update(ancestry)
    write(train)
    with pytest.raises(ValueError, match="embeds excluded provenance"):
        corpus.build(root)


@pytest.mark.parametrize("kind", ["family_id", "near_template_family_id", "parent_id",
                                  "schema23_source_text_sha256"])
def test_known_excluded_derivative_ancestry_is_rejected(frozen_inputs, kind):
    root, train, _, write = frozen_inputs
    excluded = train[-1]
    if kind == "parent_id":
        value = excluded["id"]
    elif kind == "schema23_source_text_sha256":
        value = corpus.digest(excluded["text"].encode())
    else:
        value = excluded["family_id"]
    train[0][kind] = value
    write(train)
    with pytest.raises(ValueError, match="shares excluded ancestry"):
        corpus.build(root)


@pytest.mark.parametrize("field,value", [("rows", 999), ("license_declared", "unknown"),
                                       ("selected_train_ids_sha256", "0" * 64)])
def test_source_inventory_inconsistency_is_rejected(frozen_inputs, field, value):
    root, _, _, write = frozen_inputs
    write(inventory_change=lambda inventory: inventory["sources"][0].update({field: value}))
    with pytest.raises(ValueError, match="inventory"):
        corpus.build(root)


def test_deliberate_within_split_contrast_families_are_preserved(frozen_inputs):
    root, train, _, write = frozen_inputs
    train[1]["family_id"] = train[0]["family_id"]
    write(train)
    report = corpus.build(root)
    assert report["splits"]["train"]["families"] == 11


@pytest.mark.parametrize("text", [
    "The verification code is 543219.",
    "My password is secret942!", "Email actual-person@service.com to verify.",
    "The account number is 12345678901234.", "My phone is 415-555-9281.",
])
def test_privacy_check_detects_values_without_modifying_inputs(frozen_inputs, text):
    root, train, _, write = frozen_inputs
    train[0]["text"] = text
    write(train)
    original = copy.deepcopy(train)
    with pytest.raises(ValueError, match="privacy-like value"):
        corpus.build(root)
    assert train == original
    assert not (root / corpus.OUTPUT).exists()


def test_symlink_input_rejected(frozen_inputs):
    root, _, _, _ = frozen_inputs
    path = root / corpus.INPUT_PINS["train"][0]
    moved = path.with_name("moved.jsonl")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        corpus.build(root)


def test_scanner_code_drift_rejected(frozen_inputs, monkeypatch):
    root, _, _, _ = frozen_inputs
    path = root / "scanner.py"
    path.write_text("# changed scanner\n")
    monkeypatch.setattr(corpus, "PRIVACY_CODE_PINS", {"scanner.py": "0" * 64})
    with pytest.raises(ValueError, match="privacy scanner code differs"):
        corpus.build(root)


@pytest.mark.parametrize("path", ["../outside", "/absolute/outside"])
def test_output_path_escape_rejected(frozen_inputs, path):
    root, _, _, _ = frozen_inputs
    with pytest.raises(ValueError, match="repo-relative"):
        corpus.build(root, output=path)
