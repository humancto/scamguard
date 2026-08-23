"""Exact tokenization helpers shared by Qwen evaluation and local inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def bucketed_sequence_length(sequences: Sequence[Sequence[int]], bucket_size: int) -> int:
    """Return the smallest bucket that fits every sequence, or the raw maximum for zero."""

    if not sequences:
        raise ValueError("cannot pad an empty candidate batch")
    if bucket_size < 0:
        raise ValueError("sequence bucket size cannot be negative")
    maximum = max(len(sequence) for sequence in sequences)
    if bucket_size == 0:
        return maximum
    return ((maximum + bucket_size - 1) // bucket_size) * bucket_size


def candidate_token_sequences(
    tokenizer: Any, prompt: str, labels: Sequence[str]
) -> tuple[list[list[int]], int]:
    """Tokenize complete continuations and return their shared token prefix.

    BPE tokenization is not compositional at arbitrary string boundaries. Tokenizing
    ``prompt`` and a verdict independently can therefore produce a different token
    sequence from tokenizing their concatenation. Full-sequence tokenization keeps
    benchmark and product inference on the exact sequence the model sees.
    """

    sequences = [
        tokenizer(prompt + label + '"', add_special_tokens=False)["input_ids"]
        for label in labels
    ]
    common_prefix = 0
    for tokens in zip(*sequences, strict=False):
        if len(set(tokens)) != 1:
            break
        common_prefix += 1
    if common_prefix == 0 or any(common_prefix >= len(sequence) for sequence in sequences):
        raise ValueError("verdict candidates do not have a usable shared token prefix")
    return sequences, common_prefix
