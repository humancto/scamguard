from __future__ import annotations

import pytest

from training.train_qwen_lora import completion_token_start


def test_completion_token_start_at_exact_token_boundary() -> None:
    assert completion_token_start(
        "abc",
        "abcdef",
        [(0, 2), (2, 3), (3, 5), (5, 6)],
    ) == 2


def test_completion_token_start_masks_bpe_token_crossing_boundary() -> None:
    assert completion_token_start(
        "abc",
        "abcdef",
        [(0, 2), (2, 4), (4, 6)],
    ) == 2


def test_completion_token_start_rejects_non_prefix_or_truncated_completion() -> None:
    with pytest.raises(ValueError, match="exact string prefix"):
        completion_token_start("abc", "abZdef", [(0, 2), (2, 4), (4, 6)])
    with pytest.raises(ValueError, match="no token wholly inside"):
        completion_token_start("abc", "abcd", [(0, 2), (2, 4)])
