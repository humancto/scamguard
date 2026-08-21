import csv
from pathlib import Path

import pytest

from scripts.build_scam_dialogue_holdout import diagnostic_partition, read_dialogues


def write_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dialogue", "type", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_dialogue_reader_normalizes_values_and_preserves_upstream_labels(tmp_path: Path) -> None:
    source = tmp_path / "dialogues.csv"
    write_fixture(
        source,
        [
            {
                "dialogue": (
                    "caller: Give me 123-456-7890 and visit https://bad.example now. "
                    "receiver: No."
                ),
                "type": "ssn",
                "label": "1",
            },
            {
                "dialogue": "caller: Your delivery is at reception. receiver: Thank you.",
                "type": "delivery",
                "label": "0",
            },
        ],
    )

    rows = list(read_dialogues(source))

    assert [row["label"] for row in rows] == ["SCAM", "SAFE"]
    assert [row["category"] for row in rows] == ["IDENTITY_IMPERSONATION", "NONE"]
    assert all(row["source_language"] == "English" for row in rows)
    assert "123-456-7890" not in str(rows[0]["text"])
    assert "bad.example" not in str(rows[0]["text"])
    assert "<PHONE_NUMBER>" in str(rows[0]["text"])
    assert "<URL>" in str(rows[0]["text"])


def test_dialogue_reader_rejects_type_label_conflicts(tmp_path: Path) -> None:
    source = tmp_path / "dialogues.csv"
    write_fixture(
        source,
        [{"dialogue": "A legitimate delivery call.", "type": "delivery", "label": "1"}],
    )

    with pytest.raises(ValueError, match="type/label conflict"):
        list(read_dialogues(source))


def test_dialogue_partition_is_deterministic_and_two_way() -> None:
    first = [diagnostic_partition(f"family-{index}") for index in range(100)]
    second = [diagnostic_partition(f"family-{index}") for index in range(100)]

    assert first == second
    assert set(first) == {"validation", "ood"}
