"""Model backends kept behind a small runtime interface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .metrics import file_sha256
from .preprocessing import DIALOGUE_POLICIES, prepare_model_text
from .prompts import SYSTEM_PROMPT
from .qwen_scoring import candidate_token_sequences


@dataclass(frozen=True, slots=True)
class ModelScores:
    safe: float
    uncertain: float
    scam: float

    def __post_init__(self) -> None:
        values = (self.safe, self.uncertain, self.scam)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("model scores must be probabilities")
        if abs(sum(values) - 1.0) > 1e-4:
            raise ValueError("model scores must sum to 1")


class ModelBackend(Protocol):
    model_id: str
    scam_threshold: float
    safe_threshold: float

    def predict(self, text: str) -> ModelScores: ...


class SklearnBackend:
    """Loads only a locally produced, explicitly supplied joblib artifact."""

    def __init__(self, path: str | Path) -> None:
        import joblib

        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        payload = joblib.load(artifact_path)
        required = {"pipeline", "labels", "model_id", "scam_threshold", "safe_threshold"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"invalid model artifact; missing {sorted(missing)}")
        self.pipeline = payload["pipeline"]
        self.labels = tuple(payload["labels"])
        if set(self.labels) != {"SAFE", "UNCERTAIN", "SCAM"}:
            raise ValueError("model artifact has an incompatible label set")
        self.model_id = str(payload["model_id"])
        self.scam_threshold = float(payload["scam_threshold"])
        self.safe_threshold = float(payload["safe_threshold"])

    def predict(self, text: str) -> ModelScores:
        raw = self.pipeline.predict_proba([text])[0]
        probabilities = dict(zip(self.labels, (float(value) for value in raw), strict=True))
        return ModelScores(
            safe=probabilities["SAFE"],
            uncertain=probabilities["UNCERTAIN"],
            scam=probabilities["SCAM"],
        )


class TransformersBackend:
    """Runs an exported local sequence classifier; no network fallback is allowed."""

    def __init__(self, path: str | Path, *, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = Path(path).expanduser().resolve()
        calibration_path = model_path / "scamguard_calibration.json"
        if not model_path.is_dir() or not calibration_path.is_file():
            raise FileNotFoundError(f"missing local calibrated model: {model_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        required = {"temperature", "scam_threshold", "safe_threshold", "labels"}
        missing = required - calibration.keys()
        if missing:
            raise ValueError(f"invalid calibration; missing {sorted(missing)}")
        if set(calibration["labels"]) != {"SAFE", "UNCERTAIN", "SCAM"}:
            raise ValueError("model calibration has an incompatible label set")

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.labels = tuple(calibration["labels"])
        self.temperature = float(calibration["temperature"])
        self.scam_threshold = float(calibration["scam_threshold"])
        self.safe_threshold = float(calibration["safe_threshold"])
        self.model_id = str(calibration.get("model_id", model_path.name))

    def predict(self, text: str) -> ModelScores:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, :3] / self.temperature
            values = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()
        probabilities = dict(zip(self.labels, (float(value) for value in values), strict=True))
        return ModelScores(
            safe=probabilities["SAFE"],
            uncertain=probabilities["UNCERTAIN"],
            scam=probabilities["SCAM"],
        )


def _onnx_dependencies() -> tuple[object, object]:
    try:
        import onnxruntime
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ONNX inference requires `uv sync --extra onnx --extra neural`"
        ) from error
    return onnxruntime, AutoTokenizer


class ONNXBackend:
    """Run a self-contained, hash-verified local ONNX encoder pack."""

    def __init__(self, path: str | Path, *, threads: int = 4) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_file() or model_path.suffix != ".onnx":
            raise FileNotFoundError(f"missing local ONNX model: {model_path}")
        if model_path.name.endswith("-fp32.onnx"):
            stem = model_path.name.removesuffix("-fp32.onnx")
            manifest_key = "fp32"
        elif model_path.name.endswith("-int8.onnx"):
            stem = model_path.name.removesuffix("-int8.onnx")
            manifest_key = "int8_dynamic"
        else:
            raise ValueError("ONNX filename must end in -fp32.onnx or -int8.onnx")

        pack = model_path.parent
        manifest_path = pack / f"{stem}.manifest.json"
        calibration_path = pack / "scamguard_calibration.json"
        if not manifest_path.is_file() or not calibration_path.is_file():
            raise FileNotFoundError("ONNX pack is missing its manifest or calibration")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = str(manifest[manifest_key]["sha256"])
        if file_sha256(model_path) != expected_hash:
            raise ValueError("ONNX model hash differs from its export manifest")
        runtime_files = manifest.get("runtime_files")
        if not isinstance(runtime_files, dict):
            raise ValueError("ONNX manifest is missing runtime-file hashes")
        for filename, metadata in runtime_files.items():
            runtime_path = pack / filename
            if not isinstance(metadata, dict) or not runtime_path.is_file():
                raise FileNotFoundError(f"ONNX pack runtime file is missing: {filename}")
            if file_sha256(runtime_path) != metadata.get("sha256"):
                raise ValueError(f"ONNX runtime file hash differs from manifest: {filename}")

        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if tuple(calibration.get("labels", ())) != ("SAFE", "UNCERTAIN", "SCAM"):
            raise ValueError("ONNX calibration has an incompatible ordered label set")

        dialogue_policy = str(manifest["input_transform"]["dialogue_policy"])
        if dialogue_policy not in DIALOGUE_POLICIES:
            raise ValueError(f"unsupported ONNX dialogue policy: {dialogue_policy}")
        input_length = manifest["input_contract"]["input_ids"]
        if not isinstance(input_length, list) or len(input_length) != 2:
            raise ValueError("invalid ONNX input-shape manifest")
        dynamic_sequence = input_length[1] == "dynamic"
        sequence_length = int(manifest["sequence_length"])

        onnxruntime, auto_tokenizer = _onnx_dependencies()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if {item.name for item in self.session.get_inputs()} != {"input_ids", "attention_mask"}:
            raise ValueError("ONNX runtime input names differ from the model contract")
        self.tokenizer = auto_tokenizer.from_pretrained(pack, local_files_only=True)
        self.labels = tuple(calibration["labels"])
        self.temperature = float(calibration["temperature"])
        self.scam_threshold = float(calibration["scam_threshold"])
        self.safe_threshold = float(calibration["safe_threshold"])
        self.model_id = f"{calibration.get('model_id', stem)}:{manifest_key}"
        self.sequence_length = sequence_length
        self.dynamic_sequence = dynamic_sequence
        self.dialogue_policy = dialogue_policy

    def predict(self, text: str) -> ModelScores:
        prepared = prepare_model_text(text, self.dialogue_policy)
        encoded = self.tokenizer(
            prepared,
            return_tensors="np",
            truncation=True,
            max_length=self.sequence_length,
            padding=False if self.dynamic_sequence else "max_length",
        )
        logits = self.session.run(
            ["logits"],
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]},
        )[0][0]
        adjusted = logits / self.temperature
        exponentials = np.exp(adjusted - adjusted.max())
        values = exponentials / exponentials.sum()
        probabilities = dict(zip(self.labels, (float(value) for value in values), strict=True))
        return ModelScores(
            safe=probabilities["SAFE"],
            uncertain=probabilities["UNCERTAIN"],
            scam=probabilities["SCAM"],
        )


class QwenVerdictBackend:
    """Runs a local Qwen LoRA adapter using calibrated verdict-token likelihoods."""

    def __init__(self, path: str | Path, *, device: str | None = None) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        adapter_path = Path(path).expanduser().resolve()
        calibration_path = adapter_path / "scamguard_calibration.json"
        if not adapter_path.is_dir() or not calibration_path.is_file():
            raise FileNotFoundError(f"missing local calibrated Qwen adapter: {adapter_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        required = {
            "base_model",
            "labels",
            "temperature",
            "scam_threshold",
            "safe_threshold",
            "system_prompt_sha256",
        }
        missing = required - calibration.keys()
        if missing:
            raise ValueError(f"invalid Qwen calibration; missing {sorted(missing)}")
        prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
        if calibration["system_prompt_sha256"] != prompt_digest:
            raise ValueError("Qwen calibration was fitted with a different system prompt")
        self.labels = tuple(str(label) for label in calibration["labels"])
        if self.labels != ("SAFE", "UNCERTAIN", "SCAM"):
            raise ValueError("Qwen calibration has an incompatible ordered label set")

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            calibration["base_model"],
            revision=calibration.get("base_model_revision"),
            local_files_only=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            calibration["base_model"],
            revision=calibration.get("base_model_revision"),
            local_files_only=True,
            dtype=torch.bfloat16 if self.device.type == "mps" else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path).to(self.device).eval()
        self.temperature = float(calibration["temperature"])
        self.scam_threshold = float(calibration["scam_threshold"])
        self.safe_threshold = float(calibration["safe_threshold"])
        self.model_id = str(calibration.get("model_id", adapter_path.name))

    def predict(self, text: str) -> ModelScores:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this message:\n<message>{text}</message>"},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += '{"verdict":"'
        sequences, common_prefix = candidate_token_sequences(
            self.processor.tokenizer, prompt, self.labels
        )
        maximum = max(map(len, sequences))
        candidates = [sequence[common_prefix:] for sequence in sequences]
        kept_logits = max(len(candidate) + 1 for candidate in candidates)
        pad = self.processor.tokenizer.pad_token_id
        input_ids = self.torch.tensor(
            [[pad] * (maximum - len(sequence)) + sequence for sequence in sequences],
            device=self.device,
        )
        attention = self.torch.tensor(
            [[0] * (maximum - len(sequence)) + [1] * len(sequence) for sequence in sequences],
            device=self.device,
        )
        with self.torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=kept_logits,
            ).logits
            log_probabilities = self.torch.log_softmax(logits.float(), dim=-1)
        scores = []
        for row, candidate in enumerate(candidates):
            token_scores = [
                log_probabilities[row, kept_logits - len(candidate) + offset - 1, token]
                for offset, token in enumerate(candidate)
            ]
            scores.append(self.torch.stack(token_scores).mean())
        probabilities = (
            self.torch.softmax(self.torch.stack(scores) / self.temperature, dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        values = dict(zip(self.labels, (float(value) for value in probabilities), strict=True))
        return ModelScores(safe=values["SAFE"], uncertain=values["UNCERTAIN"], scam=values["SCAM"])


def load_backend(path: str | Path) -> ModelBackend:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix == ".onnx":
        return ONNXBackend(resolved)
    if resolved.is_dir():
        calibration_path = resolved / "scamguard_calibration.json"
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration.get("backend_type") == "qwen_verdict_likelihood":
                return QwenVerdictBackend(resolved)
        return TransformersBackend(resolved)
    return SklearnBackend(resolved)


class ConservativeHeuristicBackend:
    """A non-benchmark fallback for SDK smoke tests and the model-free demo."""

    model_id = "heuristic-unbenchmarked-v0"
    scam_threshold = 0.80
    safe_threshold = 0.20

    def predict(self, text: str) -> ModelScores:
        from .signals import extract_signal_matches

        signals = {match.signal.value for match in extract_signal_matches(text)}
        high_specificity = {
            "otp_request",
            "unusual_payment_method",
            "remote_access_request",
            "guaranteed_return",
            "advance_fee",
            "secrecy_isolation",
        }
        score = 0.05
        score += min(0.30, 0.08 * len(signals))
        score += min(0.54, 0.27 * len(signals & high_specificity))
        if "suspicious_link" in signals and "artificial_urgency" in signals:
            score += 0.18
        scam = min(score, 0.98)
        uncertain = min(0.65, 0.08 + 0.08 * len(signals))
        safe = max(0.0, 1.0 - scam - uncertain)
        total = safe + uncertain + scam
        return ModelScores(safe / total, uncertain / total, scam / total)
