"""Explicit prospective tokenizer contract; never patch historical AutoProcessor runs.

The generic fast-tokenizer wrapper preserves the published serialized backend instead
of reconstructing it with the historical Qwen2Tokenizer constructor's older regex.
This text-only loader is opt-in and is not wired into existing training recipes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gguf_runtime import QWEN35_08B_PROCESSOR, QWEN35_08B_PROCESSOR_REVISION
from .metrics import file_sha256

PUBLISHED_TOKENIZER_CONTRACT = "qwen35-08b-published-json-nfc-v1"
PUBLISHED_FILES = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
}


def assert_serialized_backend(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for field in (
        "model",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "added_tokens",
    ):
        if field not in expected or expected[field] != actual.get(field):
            raise ValueError(f"published tokenizer backend changed: {field}")


def load_serialized_tokenizer(tokenizer_file: Path, config_file: Path, template_file: Path) -> Any:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    config = json.loads(config_file.read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token=config.get("bos_token"),
        eos_token=config.get("eos_token"),
        pad_token=config.get("pad_token"),
        unk_token=config.get("unk_token"),
        model_max_length=config["model_max_length"],
        clean_up_tokenization_spaces=False,
        chat_template=template_file.read_text(encoding="utf-8"),
    )
    assert_serialized_backend(
        json.loads(Tokenizer.from_file(str(tokenizer_file)).to_str()),
        json.loads(tokenizer.backend_tokenizer.to_str()),
    )
    return tokenizer


@dataclass(frozen=True)
class PublishedQwenTextProcessor:
    tokenizer: Any
    identity: dict[str, Any]

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(messages, **kwargs)


def load_published_qwen_text_processor() -> PublishedQwenTextProcessor:
    """Load only the pinned local 0.8B tokenizer; no model weights or network access."""
    import tokenizers
    import transformers
    from transformers.utils.hub import cached_file

    paths = {}
    for name, expected_hash in PUBLISHED_FILES.items():
        path = Path(
            cached_file(
                QWEN35_08B_PROCESSOR,
                name,
                revision=QWEN35_08B_PROCESSOR_REVISION,
                local_files_only=True,
            )
        )
        if file_sha256(path) != expected_hash:
            raise ValueError(f"published tokenizer artifact differs: {name}")
        paths[name] = path
    tokenizer = load_serialized_tokenizer(
        paths["tokenizer.json"], paths["tokenizer_config.json"], paths["chat_template.jinja"]
    )
    return PublishedQwenTextProcessor(
        tokenizer,
        {
            "contract": PUBLISHED_TOKENIZER_CONTRACT,
            "repository": QWEN35_08B_PROCESSOR,
            "revision": QWEN35_08B_PROCESSOR_REVISION,
            "files": dict(PUBLISHED_FILES),
            "loader": "PreTrainedTokenizerFast from serialized tokenizer.json",
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "backend_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode()
            ).hexdigest(),
            "historical_adapters_compatible": False,
        },
    )
