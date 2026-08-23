from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import scamguard.model as model_module


def test_load_backend_routes_calibrated_qwen_adapter(tmp_path, monkeypatch) -> None:
    calibration = tmp_path / "scamguard_calibration.json"
    calibration.write_text(
        json.dumps({"backend_type": "qwen_verdict_likelihood"}), encoding="utf-8"
    )
    sentinel = object()
    monkeypatch.setattr(model_module, "QwenVerdictBackend", lambda path: sentinel)

    assert model_module.load_backend(tmp_path) is sentinel


class BoundarySensitiveTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert not add_special_tokens
        if text in {'SAFE"', 'UNCERTAIN"', 'SCAM"'}:
            raise AssertionError("candidate was tokenized independently of its prompt")
        for suffix, tokens in (
            ('SAFE"', [2, 9]),
            ('UNCERTAIN"', [3, 4, 9]),
            ('SCAM"', [5, 9]),
        ):
            if text.endswith(suffix):
                return {"input_ids": [7, 7, 7] + tokens}
        return {"input_ids": [7, 7, 7]}


class FakeProcessor:
    tokenizer = BoundarySensitiveTokenizer()

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert messages and not tokenize and add_generation_prompt
        return "prompt:"


class SuffixOnlyFakeModel:
    def __init__(self) -> None:
        self.logits_to_keep: int | None = None
        self.sequence_length: int | None = None

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> object:
        assert input_ids.shape == attention_mask.shape
        self.logits_to_keep = logits_to_keep
        self.sequence_length = input_ids.shape[1]
        logits = torch.arange(16, dtype=torch.float32).view(1, 1, -1)
        return SimpleNamespace(logits=logits.expand(len(input_ids), logits_to_keep, -1))


def test_qwen_backend_uses_exact_low_memory_candidate_scoring() -> None:
    backend = model_module.QwenVerdictBackend.__new__(model_module.QwenVerdictBackend)
    backend.labels = ("SAFE", "UNCERTAIN", "SCAM")
    backend.processor = FakeProcessor()
    backend.model = SuffixOnlyFakeModel()
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.temperature = 1.0
    backend.sequence_bucket_size = 64

    scores = backend.predict("message")

    assert abs(scores.safe + scores.uncertain + scores.scam - 1.0) < 1e-6
    assert backend.model.logits_to_keep == 4
    assert backend.model.sequence_length == 64


def test_qwen_backend_matches_evaluator_probability_math() -> None:
    from training.eval_qwen import score_message, softmax

    backend = model_module.QwenVerdictBackend.__new__(model_module.QwenVerdictBackend)
    backend.labels = ("SAFE", "UNCERTAIN", "SCAM")
    backend.processor = FakeProcessor()
    backend.model = SuffixOnlyFakeModel()
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.temperature = 1.7
    backend.sequence_bucket_size = 64

    runtime = backend.predict("message")
    expected = softmax(
        score_message(
            backend.model,
            backend.processor,
            "message",
            backend.device,
            sequence_bucket_size=64,
        )[None, :],
        backend.temperature,
    )[0]

    np.testing.assert_allclose(
        [runtime.safe, runtime.uncertain, runtime.scam], expected, rtol=0.0, atol=0.0
    )


class FakeONNXSessionOptions:
    pass


class FakeONNXSession:
    def __init__(self, path: str, *, sess_options: object, providers: list[str]) -> None:
        assert Path(path).name == "candidate-fp32.onnx"
        assert providers == ["CPUExecutionProvider"]
        assert sess_options.intra_op_num_threads == 4

    @staticmethod
    def get_inputs() -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    @staticmethod
    def run(outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert outputs == ["logits"]
        assert inputs["input_ids"].shape == inputs["attention_mask"].shape
        return [np.array([[0.0, 1.0, 2.0]], dtype=np.float32)]


class FakeONNXRuntime:
    SessionOptions = FakeONNXSessionOptions
    InferenceSession = FakeONNXSession
    ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")


class FakeONNXTokenizer:
    @classmethod
    def from_pretrained(cls, path: Path, *, local_files_only: bool) -> FakeONNXTokenizer:
        assert path.is_dir() and local_files_only
        return cls()

    def __call__(self, text: str, **kwargs: object) -> dict[str, np.ndarray]:
        assert text and kwargs["return_tensors"] == "np"
        return {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }


def _write_fake_onnx_pack(tmp_path: Path) -> Path:
    model_path = tmp_path / "candidate-fp32.onnx"
    model_path.write_bytes(b"fake onnx")
    digest = model_module.file_sha256(model_path)
    calibration = tmp_path / "scamguard_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "labels": ["SAFE", "UNCERTAIN", "SCAM"],
                "temperature": 2.0,
                "scam_threshold": 0.7,
                "safe_threshold": 0.2,
                "model_id": "test-onnx",
            }
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text("{}", encoding="utf-8")
    (tmp_path / "candidate.manifest.json").write_text(
        json.dumps(
            {
                "sequence_length": 256,
                "input_transform": {"dialogue_policy": "speaker-neutral-v1"},
                "input_contract": {"input_ids": [1, "dynamic"]},
                "fp32": {"sha256": digest},
                "runtime_files": {
                    path.name: {"sha256": model_module.file_sha256(path)}
                    for path in (calibration, tokenizer, tokenizer_config)
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path


def test_load_backend_routes_hash_verified_onnx_pack(tmp_path, monkeypatch) -> None:
    model_path = _write_fake_onnx_pack(tmp_path)
    monkeypatch.setattr(
        model_module,
        "_onnx_dependencies",
        lambda: (FakeONNXRuntime, FakeONNXTokenizer),
    )

    backend = model_module.load_backend(model_path)
    scores = backend.predict("A message")

    assert isinstance(backend, model_module.ONNXBackend)
    assert backend.model_id == "test-onnx:fp32"
    assert scores.scam > scores.uncertain > scores.safe


def test_onnx_backend_rejects_tampered_model(tmp_path, monkeypatch) -> None:
    model_path = _write_fake_onnx_pack(tmp_path)
    model_path.write_bytes(b"tampered")
    monkeypatch.setattr(
        model_module,
        "_onnx_dependencies",
        lambda: (FakeONNXRuntime, FakeONNXTokenizer),
    )

    try:
        model_module.ONNXBackend(model_path)
    except ValueError as error:
        assert "hash differs" in str(error)
    else:
        raise AssertionError("tampered ONNX model should fail closed")
