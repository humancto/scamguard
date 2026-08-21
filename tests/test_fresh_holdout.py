from __future__ import annotations

from pathlib import Path

from scripts.build_fresh_holdout import read_moz_smishing


def test_moz_reader_normalizes_every_label_and_private_value(tmp_path: Path) -> None:
    source = tmp_path / "moz.csv"
    source.write_text(
        "id,source,text,label\n"
        '1,sms,"Send to 841234567 or a@example.com",Smishing\n'
        '2,sms,"Your code is 1234567890",Legitimate\n',
        encoding="utf-8",
    )

    rows, privacy = read_moz_smishing(source)

    assert [row["label"] for row in rows] == ["SCAM", "SAFE"]
    assert rows[0]["category"] == "FINANCIAL"
    assert rows[1]["category"] == "NONE"
    assert "841234567" not in str(rows[0]["text"])
    assert "a@example.com" not in str(rows[0]["text"])
    assert "1234567890" not in str(rows[1]["text"])
    assert privacy == {
        "rows_with_email_before_normalization": 1,
        "rows_with_phone_like_before_normalization": 2,
        "rows_with_long_digits_before_normalization": 1,
    }


def test_moz_reader_rejects_unknown_labels(tmp_path: Path) -> None:
    source = tmp_path / "moz.csv"
    source.write_text(
        "id,source,text,label\n1,sms,Maybe something else,Unknown\n",
        encoding="utf-8",
    )

    rows, privacy = read_moz_smishing(source)

    assert rows == []
    assert privacy == {}
