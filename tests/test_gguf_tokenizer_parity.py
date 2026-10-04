import pytest

from scripts.check_gguf_tokenizer_parity import compare_records


def test_parity_checks_entire_token_sequence_not_only_count() -> None:
    rows = compare_records({"a": [1, 2], "b": [3]}, "READY\t4\nTOKENS\ta\t2,1\nTOKENS\tb\t3\n")
    assert rows[0]["hf_tokens"] == rows[0]["gguf_tokens"] == 2
    assert rows[0]["exact_match"] is False
    assert rows[1]["exact_match"] is True


@pytest.mark.parametrize(
    "output",
    [
        "READY\t3\nTOKENS\ta\t1\n",
        "READY\t4\n",
        "READY\t4\nTOKENS\ta\t1\nTOKENS\ta\t1\n",
        "READY\t4\nTOKENS\tb\t1\n",
        "READY\t4\nTOKENS\ta\tnot-a-token\n",
    ],
)
def test_parity_rejects_stale_missing_duplicate_or_invalid_records(output: str) -> None:
    with pytest.raises(ValueError):
        compare_records({"a": [1]}, output)
