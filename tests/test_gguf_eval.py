"""Binary task serialization and patched llama.cpp score parsing are deterministic."""

import struct

import numpy as np

from training.eval_gguf import parse_scores, write_tasks


def test_multiple_choice_task_header_contains_absolute_offsets(tmp_path) -> None:
    path = tmp_path / "tasks.bin"
    write_tasks(path, ["prompt-a", "prompt-b"], ["SAFE", "SCAM"])
    data = path.read_bytes()

    task_count, first, second = struct.unpack("<III", data[:12])

    assert task_count == 2
    assert first == 12
    assert first < second < len(data)


def test_parse_llama_scores() -> None:
    output = "noise\n1\t100.00000000\t-0.1\t-2.0\t-3.0\n2\t50.00000000\t-3\t-2\t-0.1\n"

    scores = parse_scores(output, expected=2)

    np.testing.assert_allclose(scores, [[-0.1, -2.0, -3.0], [-3.0, -2.0, -0.1]])
