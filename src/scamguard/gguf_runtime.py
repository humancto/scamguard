"""Persistent native GGUF verdict scoring for the routed product runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
import selectors
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .metrics import file_sha256
from .model import ModelScores
from .prompts import SYSTEM_PROMPT

LABELS = ("SAFE", "UNCERTAIN", "SCAM")
READY = re.compile(r"^READY\t(\d+)\t(\d+)\t(\d+)\t(\d+)$")
RESULT = re.compile(
    r"^RESULT\t([^\t]+)\t(-?[0-9.eE+]+)\t(-?[0-9.eE+]+)\t"
    r"(-?[0-9.eE+]+)\t(\d+)\t(\d+)\t([01])\t(\d+)$"
)
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+")


@dataclass(frozen=True, slots=True)
class NativeScore:
    raw_scores: tuple[float, float, float]
    native_elapsed_ms: float
    round_trip_ms: float
    maximum_sequence_tokens: int
    prefix_reused: bool
    prefix_tokens: int


def calibrated_probabilities(
    raw_scores: tuple[float, float, float], temperature: float
) -> tuple[float, float, float]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    scaled = [value / temperature for value in raw_scores]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("GGUF scores must be finite")
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)  # type: ignore[return-value]


class PersistentGGUFScorer:
    """Own one native scorer process and exchange one hex-framed request at a time."""

    def __init__(
        self,
        *,
        runner: Path,
        model: Path,
        ctx_size: int = 640,
        batch_size: int = 640,
        ubatch_size: int = 128,
        threads: int = 4,
        n_gpu_layers: int = 99,
        prefix: str | None = None,
        startup_timeout_seconds: float = 60.0,
    ) -> None:
        self.runner = runner.expanduser().resolve()
        self.model = model.expanduser().resolve()
        for path in (self.runner, self.model):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not self.runner.stat().st_mode & 0o111:
            raise PermissionError(f"GGUF runner is not executable: {self.runner}")
        for name, value in {
            "ctx_size": ctx_size,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "threads": threads,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if n_gpu_layers < 0:
            raise ValueError("n_gpu_layers must be non-negative")
        if ubatch_size > batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        self.ctx_size = ctx_size
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.n_gpu_layers = n_gpu_layers
        self._stderr: TextIO = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        command = [
            str(self.runner),
            "--model",
            str(self.model),
            "--ctx-size",
            str(ctx_size),
            "--batch-size",
            str(batch_size),
            "--ubatch-size",
            str(ubatch_size),
            "--threads",
            str(threads),
            "--n-gpu-layers",
            str(n_gpu_layers),
        ]
        if prefix is not None:
            if not prefix:
                raise ValueError("GGUF prefix must be non-empty when supplied")
            command.extend(("--prefix-hex", prefix.encode("utf-8").hex()))
        self.prefix = prefix
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        try:
            ready_line = self._readline(startup_timeout_seconds)
            match = READY.fullmatch(ready_line.rstrip("\n"))
            if match is None or int(match.group(1)) != 2:
                raise RuntimeError(f"invalid GGUF runner readiness record: {ready_line!r}")
            self.protocol_version = int(match.group(1))
            self.loaded_model_bytes = int(match.group(2))
            self.loaded_ctx_size = int(match.group(3))
            self.loaded_prefix_tokens = int(match.group(4))
            if self.loaded_ctx_size != ctx_size:
                raise RuntimeError("GGUF runner created a different context size")
            if (prefix is None) != (self.loaded_prefix_tokens == 0):
                raise RuntimeError("GGUF runner created an unexpected prefix cache")
        except Exception:
            self.close(force=True)
            raise

    @property
    def process_id(self) -> int:
        return self._process.pid

    def diagnostics(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        return self._stderr.read()

    def _readline(self, timeout_seconds: float) -> str:
        if self._process.stdout is None:
            raise RuntimeError("GGUF runner stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout_seconds):
                raise TimeoutError("timed out waiting for GGUF runner output")
            line = self._process.stdout.readline()
        finally:
            selector.close()
        if not line:
            status = self._process.poll()
            tail = self.diagnostics()[-4000:]
            raise RuntimeError(f"GGUF runner exited with status {status}: {tail}")
        return line

    def score(
        self, identifier: str, question: str, *, timeout_seconds: float = 30.0
    ) -> NativeScore:
        if IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError("GGUF request identifier has unsupported characters")
        if not question:
            raise ValueError("GGUF question must not be empty")
        if self._process.stdin is None:
            raise RuntimeError("GGUF runner stdin is unavailable")
        request = f"{identifier}\t{question.encode('utf-8').hex()}\n"
        started = time.perf_counter_ns()
        self._process.stdin.write(request)
        self._process.stdin.flush()
        response = self._readline(timeout_seconds).rstrip("\n")
        finished = time.perf_counter_ns()
        if response.startswith("ERROR\t"):
            raise RuntimeError(f"GGUF runner rejected request: {response}")
        match = RESULT.fullmatch(response)
        if match is None or match.group(1) != identifier:
            raise RuntimeError(f"invalid GGUF runner result record: {response!r}")
        scores = tuple(float(match.group(index)) for index in (2, 3, 4))
        if not all(math.isfinite(value) for value in scores):
            raise RuntimeError("GGUF runner returned non-finite scores")
        elapsed_microseconds = int(match.group(5))
        maximum_sequence_tokens = int(match.group(6))
        prefix_reused = match.group(7) == "1"
        prefix_tokens = int(match.group(8))
        if elapsed_microseconds < 1 or maximum_sequence_tokens < 1:
            raise RuntimeError("GGUF runner returned invalid timing or token count")
        if prefix_reused != (self.prefix is not None):
            raise RuntimeError("GGUF runner did not apply the configured prefix cache")
        if prefix_tokens != self.loaded_prefix_tokens:
            raise RuntimeError("GGUF runner prefix token count changed")
        return NativeScore(
            raw_scores=scores,  # type: ignore[arg-type]
            native_elapsed_ms=elapsed_microseconds / 1_000,
            round_trip_ms=(finished - started) / 1_000_000,
            maximum_sequence_tokens=maximum_sequence_tokens,
            prefix_reused=prefix_reused,
            prefix_tokens=prefix_tokens,
        )

    def close(self, *, force: bool = False) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            if hasattr(self, "_stderr") and not self._stderr.closed:
                self._stderr.close()
            return
        if not force and process.stdin is not None:
            try:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                force = True
        if force and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if not self._stderr.closed:
            self._stderr.close()

    def __enter__(self) -> PersistentGGUFScorer:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class QwenGGUFVerdictBackend:
    """Apply frozen Qwen calibration to the persistent native three-candidate scorer."""

    def __init__(
        self,
        *,
        runner: Path,
        model: Path,
        processor: Path,
        calibration: Path,
        expected_model_sha256: str,
        expected_runner_sha256: str,
        ctx_size: int = 640,
        batch_size: int = 640,
        ubatch_size: int = 128,
        threads: int = 4,
        n_gpu_layers: int = 99,
    ) -> None:
        from transformers import AutoProcessor

        if file_sha256(model) != expected_model_sha256:
            raise ValueError("GGUF model SHA-256 differs from the selected artifact")
        if file_sha256(runner) != expected_runner_sha256:
            raise ValueError("GGUF runner SHA-256 differs from the pinned runtime")
        record: dict[str, Any] = json.loads(calibration.read_text(encoding="utf-8"))
        if tuple(record.get("labels", ())) != LABELS:
            raise ValueError("GGUF calibration label order is incompatible")
        if record.get("safe_threshold_semantics") != "minimum_safe_probability":
            raise ValueError("GGUF calibration SAFE-threshold semantics are incompatible")
        if record.get("sequence_bucket_size") != 64:
            raise ValueError("GGUF calibration must bind the frozen 64-token bucket")
        if record.get("system_prompt_sha256") != hashlib.sha256(
            SYSTEM_PROMPT.encode()
        ).hexdigest():
            raise ValueError("GGUF calibration system prompt differs from the runtime prompt")
        self.temperature = float(record["temperature"])
        self.scam_threshold = float(record["scam_threshold"])
        self.safe_threshold = float(record["safe_threshold"])
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("GGUF calibration temperature must be finite and positive")
        if not 0.0 <= self.scam_threshold <= 1.0:
            raise ValueError("GGUF calibration scam threshold must be in [0, 1]")
        if not 0.0 <= self.safe_threshold <= 1.0:
            raise ValueError("GGUF calibration safe threshold must be in [0, 1]")
        self.safe_probability_threshold = self.safe_threshold
        self.safe_max_scam_probability = None
        self.sequence_bucket_size = 64
        self.labels = LABELS
        self.processor_path = processor.expanduser().resolve()
        self.processor = AutoProcessor.from_pretrained(
            self.processor_path, local_files_only=True
        )
        sentinel = "SCAMGUARD_RUNTIME_MESSAGE_SENTINEL"
        sentinel_prompt = self.processor.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Classify this message:\n"
                        f"<message>{sentinel}</message>"
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        marker = f"<message>{sentinel}"
        if sentinel_prompt.count(marker) != 1:
            raise ValueError("Qwen chat template does not preserve the runtime message marker")
        self.cached_prefix = sentinel_prompt.split(marker, maxsplit=1)[0]
        self.scorer = PersistentGGUFScorer(
            runner=runner,
            model=model,
            ctx_size=ctx_size,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            threads=threads,
            n_gpu_layers=n_gpu_layers,
            prefix=self.cached_prefix,
        )
        self.model_id = f"{model.name}@{expected_model_sha256}"
        self.model_sha256 = expected_model_sha256
        self.runner_sha256 = expected_runner_sha256
        self.calibration_sha256 = file_sha256(calibration)
        self._request_count = 0
        self.last_native_elapsed_ms = 0.0
        self.last_round_trip_ms = 0.0
        self.last_maximum_sequence_tokens = 0
        self.last_prefix_reused = False
        self.last_prefix_tokens = 0

    def predict(self, text: str) -> ModelScores:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify this message:\n<message>{text}</message>",
            },
        ]
        question = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        question += '{"verdict":"'
        identifier = f"request-{self._request_count}"
        self._request_count += 1
        result = self.scorer.score(identifier, question)
        probabilities = calibrated_probabilities(result.raw_scores, self.temperature)
        self.last_native_elapsed_ms = result.native_elapsed_ms
        self.last_round_trip_ms = result.round_trip_ms
        self.last_maximum_sequence_tokens = result.maximum_sequence_tokens
        self.last_prefix_reused = result.prefix_reused
        self.last_prefix_tokens = result.prefix_tokens
        return ModelScores(
            safe=probabilities[0],
            uncertain=probabilities[1],
            scam=probabilities[2],
        )

    def close(self) -> None:
        self.scorer.close()

    def runtime_identity(self) -> dict[str, Any]:
        return {
            "runner": str(self.scorer.runner),
            "runner_sha256": self.runner_sha256,
            "protocol_version": self.scorer.protocol_version,
            "model": str(self.scorer.model),
            "model_sha256": self.model_sha256,
            "artifact_bytes": self.scorer.model.stat().st_size,
            "loaded_model_tensor_bytes": self.scorer.loaded_model_bytes,
            "processor": str(self.processor_path),
            "calibration_sha256": self.calibration_sha256,
            "ctx_size_per_sequence": self.scorer.ctx_size,
            "batch_size": self.scorer.batch_size,
            "ubatch_size": self.scorer.ubatch_size,
            "threads": self.scorer.threads,
            "n_gpu_layers": self.scorer.n_gpu_layers,
            "message_batch_size": 1,
            "candidate_batch_size": 3,
            "sequence_bucket_size": self.sequence_bucket_size,
            "prefix_cache_enabled": self.scorer.prefix is not None,
            "prefix_tokens": self.scorer.loaded_prefix_tokens,
            "prefix_sha256": hashlib.sha256(self.cached_prefix.encode()).hexdigest(),
        }
