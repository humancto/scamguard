"""Persistent native GGUF verdict scoring for the routed product runtime."""

from __future__ import annotations

import hashlib
import json
import math
import platform
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
PACK_MANIFEST_NAME = "scamguard_gguf_pack.json"
GGUF_BACKEND_TYPE = "qwen_gguf_verdict_branch_token"
GGUF_SCORING_VERSION = "qwen-verdict-branch-token-v1"
QWEN35_08B_PROCESSOR = "Qwen/Qwen3.5-0.8B"
QWEN35_08B_PROCESSOR_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
FROZEN_PROMPT_PREFIX = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\nClassify this message:\n"
)
FROZEN_PROMPT_SUFFIX = (
    '</message><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{"verdict":"'
)
FROZEN_PROMPT_PREFIX_SHA256 = (
    "7066a03d24a7859f93c168ef17f19570ddd2fecf031e5273bfd3c30e40380510"
)
FROZEN_PROMPT_SUFFIX_SHA256 = (
    "ae2e9c4a85d503db999767cb61481bca25528d4026e857db041d086fd7ded47f"
)
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
            if match is None or int(match.group(1)) != 3:
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
        processor: Path | None = None,
        prompt_prefix: str | None = None,
        prompt_suffix: str | None = None,
        calibration: Path,
        expected_model_sha256: str,
        expected_runner_sha256: str,
        ctx_size: int = 640,
        batch_size: int = 640,
        ubatch_size: int = 128,
        threads: int = 4,
        n_gpu_layers: int = 99,
    ) -> None:
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
        if (
            record.get("scoring_mode") != "branch_token"
            or record.get("scoring_version") != GGUF_SCORING_VERSION
        ):
            raise ValueError("GGUF calibration must bind the branch-token scorer")
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
        if processor is not None and (prompt_prefix is not None or prompt_suffix is not None):
            raise ValueError("provide a processor or frozen prompt fragments, not both")
        if processor is None and (not prompt_prefix or not prompt_suffix):
            raise ValueError("GGUF runtime requires a processor or frozen prompt fragments")
        self.processor_path: Path | None = None
        self.processor: Any | None = None
        if processor is not None:
            from transformers import AutoProcessor

            self.processor_path = processor.expanduser().resolve()
            self.processor = AutoProcessor.from_pretrained(
                self.processor_path, local_files_only=True
            )
            sentinel = "SCAMGUARD_RUNTIME_MESSAGE_SENTINEL"
            rendered = self.processor.apply_chat_template(
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
            question = rendered + '{"verdict":"'
            marker = f"<message>{sentinel}"
            if question.count(marker) != 1:
                raise ValueError(
                    "Qwen chat template does not preserve the runtime message marker"
                )
            prompt_prefix, prompt_suffix = question.split(marker, maxsplit=1)
        assert prompt_prefix is not None and prompt_suffix is not None
        if not prompt_prefix or not prompt_suffix.startswith("</message>"):
            raise ValueError("GGUF frozen prompt fragments have incompatible message boundaries")
        self.cached_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
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
        question = self.cached_prefix + "<message>" + text + self.prompt_suffix
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
            "prompt_source": (
                str(self.processor_path)
                if self.processor_path is not None
                else "hash-bound runtime pack fragments"
            ),
            "calibration_sha256": self.calibration_sha256,
            "ctx_size_per_sequence": self.scorer.ctx_size,
            "batch_size": self.scorer.batch_size,
            "ubatch_size": self.scorer.ubatch_size,
            "threads": self.scorer.threads,
            "n_gpu_layers": self.scorer.n_gpu_layers,
            "message_batch_size": 1,
            "candidate_batch_size": 3,
            "scoring_mode": "branch_token",
            "scoring_version": GGUF_SCORING_VERSION,
            "sequence_bucket_size": self.sequence_bucket_size,
            "prefix_cache_enabled": self.scorer.prefix is not None,
            "prefix_tokens": self.scorer.loaded_prefix_tokens,
            "prefix_sha256": hashlib.sha256(self.cached_prefix.encode()).hexdigest(),
        }


def _pack_member(root: Path, section: dict[str, Any], role: str) -> Path:
    relative = Path(str(section.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"GGUF pack {role} path must stay inside the pack")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"GGUF pack {role} is missing or escapes the pack: {relative}")
    expected_hash = str(section.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"GGUF pack {role} SHA-256 is invalid")
    if file_sha256(resolved) != expected_hash:
        raise ValueError(f"GGUF pack {role} hash differs from its manifest")
    expected_bytes = section.get("bytes")
    if expected_bytes is not None and expected_bytes != resolved.stat().st_size:
        raise ValueError(f"GGUF pack {role} byte count differs from its manifest")
    return resolved


def load_gguf_runtime_pack(path: str | Path) -> QwenGGUFVerdictBackend:
    """Load a self-contained GGUF pack without a Transformers runtime dependency."""

    requested = Path(path).expanduser().resolve()
    manifest_path = requested / PACK_MANIFEST_NAME if requested.is_dir() else requested
    if manifest_path.name != PACK_MANIFEST_NAME or not manifest_path.is_file():
        raise FileNotFoundError(f"missing GGUF runtime-pack manifest: {manifest_path}")
    root = manifest_path.parent.resolve()
    record: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        record.get("artifact_schema_version") != 1
        or record.get("backend_type") != GGUF_BACKEND_TYPE
    ):
        raise ValueError("GGUF runtime-pack manifest has an incompatible schema or backend")
    if record.get("publication_authorized") is not False:
        raise ValueError(
            "runtime packs cannot self-authorize publication; use the release verifier"
        )
    if record.get("purpose") not in {"release_candidate", "upstream_base_control"}:
        raise ValueError("GGUF runtime-pack purpose is invalid")
    model_record = record.get("model")
    runner_record = record.get("runner")
    calibration_record = record.get("calibration")
    prompt_record = record.get("prompt")
    runtime = record.get("runtime")
    if not all(
        isinstance(item, dict)
        for item in (
            model_record,
            runner_record,
            calibration_record,
            prompt_record,
            runtime,
        )
    ):
        raise ValueError("GGUF runtime-pack manifest is missing required sections")
    assert isinstance(model_record, dict)
    assert isinstance(runner_record, dict)
    assert isinstance(calibration_record, dict)
    assert isinstance(prompt_record, dict)
    assert isinstance(runtime, dict)
    if (
        runner_record.get("portable_system_dependencies_only") is not True
        or runner_record.get("system") != platform.system()
        or runner_record.get("machine") != platform.machine()
    ):
        raise ValueError("GGUF runtime-pack runner is not portable for this machine")
    expected_runtime = {
        "protocol_version": 3,
        "message_batch_size": 1,
        "candidate_batch_size": 3,
        "scoring_mode": "branch_token",
        "scoring_version": GGUF_SCORING_VERSION,
        "sequence_bucket_size": 64,
        "prefix_cache_enabled": True,
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise ValueError(f"GGUF runtime-pack contract mismatch: {field}")
    integer_runtime = {
        field: runtime.get(field)
        for field in (
            "ctx_size",
            "batch_size",
            "ubatch_size",
            "threads",
            "n_gpu_layers",
        )
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in integer_runtime.values()
    ):
        raise ValueError("GGUF runtime-pack numeric settings must be integers")

    prefix = str(prompt_record.get("prefix", ""))
    suffix = str(prompt_record.get("suffix", ""))
    prompt_hashes = {
        "prefix_sha256": FROZEN_PROMPT_PREFIX_SHA256,
        "suffix_sha256": FROZEN_PROMPT_SUFFIX_SHA256,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    if (
        prefix != FROZEN_PROMPT_PREFIX
        or suffix != FROZEN_PROMPT_SUFFIX
        or hashlib.sha256(prefix.encode()).hexdigest() != FROZEN_PROMPT_PREFIX_SHA256
        or hashlib.sha256(suffix.encode()).hexdigest() != FROZEN_PROMPT_SUFFIX_SHA256
        or prompt_record.get("message_open") != "<message>"
        or prompt_record.get("processor_repository") != QWEN35_08B_PROCESSOR
        or prompt_record.get("processor_revision") != QWEN35_08B_PROCESSOR_REVISION
    ):
        raise ValueError("GGUF runtime-pack prompt differs from the frozen Qwen template")
    for field, expected in prompt_hashes.items():
        if prompt_record.get(field) != expected:
            raise ValueError(f"GGUF runtime-pack prompt binding mismatch: {field}")

    model = _pack_member(root, model_record, "model")
    if model.suffix.casefold() != ".gguf":
        raise ValueError("GGUF runtime-pack model must use the .gguf extension")
    runner = _pack_member(root, runner_record, "runner")
    calibration = _pack_member(root, calibration_record, "calibration")
    backend = QwenGGUFVerdictBackend(
        runner=runner,
        model=model,
        prompt_prefix=prefix,
        prompt_suffix=suffix,
        calibration=calibration,
        expected_model_sha256=str(model_record["sha256"]),
        expected_runner_sha256=str(runner_record["sha256"]),
        ctx_size=integer_runtime["ctx_size"],
        batch_size=integer_runtime["batch_size"],
        ubatch_size=integer_runtime["ubatch_size"],
        threads=integer_runtime["threads"],
        n_gpu_layers=integer_runtime["n_gpu_layers"],
    )
    backend.pack_manifest_path = manifest_path
    backend.pack_manifest_sha256 = file_sha256(manifest_path)
    backend.pack_purpose = str(record.get("purpose", ""))
    return backend
