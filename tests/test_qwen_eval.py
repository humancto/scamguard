"""The batched Qwen scorer must match its simple reference implementation."""

from types import SimpleNamespace

import numpy as np
import torch

from training.eval_qwen import (
    LABELS,
    choose_safe_threshold,
    load_score_cache,
    predict_with_abstention,
    save_score_cache,
    score_cache_identity,
    score_message_unbatched,
    score_messages,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert not add_special_tokens
        # The scorer must tokenize each complete prompt+candidate string. Bare
        # candidate tokenization intentionally fails to catch boundary concatenation.
        if text in {'SAFE"', 'UNCERTAIN"', 'SCAM"'}:
            raise AssertionError("candidate was tokenized independently of its prompt")
        candidates = [('SAFE"', [2, 9]), ('UNCERTAIN"', [3, 4, 9]), ('SCAM"', [5, 9])]
        for suffix, tokens in candidates:
            if text.endswith(suffix):
                prefix = text[: -len(suffix)]
                return {"input_ids": [7] * (3 + len(prefix) % 4) + tokens}
        return {"input_ids": [7] * (3 + len(text) % 4)}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize and add_generation_prompt
        return "prompt:" + messages[-1]["content"]


class FakeModel:
    def __init__(self) -> None:
        self.kept_values: list[int] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int = 0,
    ) -> object:
        assert input_ids.shape == attention_mask.shape
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32).view(1, -1, 1)
        tokens = torch.arange(16, dtype=torch.float32).view(1, 1, -1)
        logits = positions * 0.03 + tokens * 0.07
        logits = logits.expand(input_ids.shape[0], -1, -1)
        self.kept_values.append(logits_to_keep)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_batched_qwen_scores_match_unbatched_reference() -> None:
    texts = ["short", "a somewhat longer message", "x"]
    model = FakeModel()
    processor = FakeProcessor()
    device = torch.device("cpu")

    batched = score_messages(model, processor, texts, device, batch_size=2)
    reference = np.stack(
        [score_message_unbatched(model, processor, text, device) for text in texts]
    )

    # Padding changes the fake model's floating-point reduction order by about
    # one float32 ULP; the scored candidate suffixes must remain numerically
    # equivalent, not bit-identical.
    np.testing.assert_allclose(batched, reference, rtol=1e-6, atol=2e-7)
    assert model.kept_values[:2] == [4, 4]
    assert model.kept_values[2:] == [0, 0, 0]


def test_score_cache_requires_exact_experiment_identity(tmp_path) -> None:
    identity = score_cache_identity(
        model="Qwen/example",
        revision="abc123",
        adapter_sha256="adapter-digest",
        data_sha256="data-digest",
        examples=2,
        batch_size=4,
    )
    scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    save_score_cache(tmp_path, "dev", scores, identity)

    loaded = load_score_cache(tmp_path, "dev", identity)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, scores)
    changed_batch = identity | {"batch_size": 8}
    assert load_score_cache(tmp_path, "dev", changed_batch) is None


def test_score_cache_rejects_partial_or_invalid_arrays(tmp_path) -> None:
    identity = score_cache_identity(
        model="Qwen/example",
        revision="abc123",
        adapter_sha256=None,
        data_sha256="data-digest",
        examples=2,
        batch_size=4,
    )
    save_score_cache(tmp_path, "test", np.zeros((2, 3)), identity)
    with (tmp_path / "test.npy").open("wb") as handle:
        np.save(handle, np.zeros((1, 3)), allow_pickle=False)

    assert load_score_cache(tmp_path, "test", identity) is None


def test_abstention_rule_prioritizes_scam_then_safe() -> None:
    probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.60, 0.05, 0.35],
            [0.40, 0.45, 0.15],
        ]
    )

    predicted = predict_with_abstention(probabilities, scam_threshold=0.30, safe_threshold=0.70)

    assert predicted.tolist() == [
        LABELS.index("SAFE"),
        LABELS.index("SCAM"),
        LABELS.index("UNCERTAIN"),
    ]


def test_safe_threshold_is_fit_after_scam_threshold_is_frozen() -> None:
    truth = np.array(
        [
            LABELS.index("SAFE"),
            LABELS.index("SAFE"),
            LABELS.index("UNCERTAIN"),
            LABELS.index("SCAM"),
        ]
    )
    probabilities = np.array(
        [
            [0.90, 0.08, 0.02],
            [0.75, 0.20, 0.05],
            [0.40, 0.55, 0.05],
            [0.20, 0.10, 0.70],
        ]
    )

    threshold = choose_safe_threshold(truth, probabilities, scam_threshold=0.60)
    predicted = predict_with_abstention(
        probabilities, scam_threshold=0.60, safe_threshold=threshold
    )

    assert threshold == 0.75
    assert predicted.tolist() == truth.tolist()
