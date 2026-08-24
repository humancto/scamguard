from __future__ import annotations

import json
import os
import platform
import textwrap
from pathlib import Path

import pytest

from benchmarks.benchmark_gguf_runtime_pack import benchmark
from scamguard.gguf_runtime import (
    FROZEN_PROMPT_PREFIX,
    FROZEN_PROMPT_SUFFIX,
    PACK_MANIFEST_NAME,
    QWEN35_08B_PROCESSOR,
    QWEN35_08B_PROCESSOR_REVISION,
    load_gguf_runtime_pack,
)
from scamguard.scanner import Scanner, scan
from scamguard.taxonomy import Verdict
from scripts.build_gguf_runtime_pack import build_pack


def fake_runner(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys

            prefix_tokens = 141 if "--prefix-hex" in sys.argv else 0
            print(f"READY\t3\t563036064\t640\t{prefix_tokens}", flush=True)
            for line in sys.stdin:
                line = line.rstrip("\\n")
                if line == "QUIT":
                    break
                identifier, question_hex = line.split("\\t", 1)
                question = bytes.fromhex(question_hex).decode()
                if "<message>" not in question or not question.endswith('{"verdict":"'):
                    print(f"ERROR\\t{identifier}\\tbad-prompt", flush=True)
                    continue
                print(
                    f"RESULT\\t{identifier}\\t-0.1\\t-3.0\\t-5.0"
                    f"\\t1200\\t48\\t1\\t{prefix_tokens}",
                    flush=True,
                )
            """
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


def runtime_pack(tmp_path: Path) -> Path:
    model = tmp_path / "source.gguf"
    runner = tmp_path / "source-runner"
    calibration = tmp_path / "source-calibration.json"
    model.write_bytes(b"test GGUF model")
    fake_runner(runner)
    calibration.write_text(
        json.dumps(
            {
                "temperature": 1.0,
                "scam_threshold": 0.8,
                "safe_threshold": 0.6,
                "safe_threshold_semantics": "minimum_safe_probability",
                "scoring_mode": "branch_token",
                "score_cache": {
                    "sequence_bucket_size": 64,
                    "scoring_version": "qwen-verdict-branch-token-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pack"
    build_pack(
        model=model,
        runner=runner,
        calibration_source=calibration,
        output=output,
        prompt_prefix=FROZEN_PROMPT_PREFIX,
        prompt_suffix=FROZEN_PROMPT_SUFFIX,
        processor_repository=QWEN35_08B_PROCESSOR,
        processor_revision=QWEN35_08B_PROCESSOR_REVISION,
        purpose="upstream_base_control",
        dependency_report={
            "inspection": "test",
            "dependencies": [],
            "portable_system_dependencies_only": True,
        },
    )
    return output


def test_pack_runs_through_persistent_scanner_and_one_call_helper(tmp_path: Path) -> None:
    pack = runtime_pack(tmp_path)

    with Scanner(model_path=str(pack)) as scanner:
        first_process = scanner.backend.scorer.process_id  # type: ignore[attr-defined]
        first = scanner.scan("Your normal appointment is tomorrow.")
        second = scanner.scan("Your normal appointment is still tomorrow.")

        assert scanner.backend.scorer.process_id == first_process  # type: ignore[attr-defined]
        assert first.verdict is Verdict.SAFE
        assert second.verdict is Verdict.SAFE
        assert scanner.backend.last_prefix_reused is True  # type: ignore[attr-defined]

    one_shot = scan("Your normal appointment is tomorrow.", model_path=str(pack))
    assert one_shot.verdict is Verdict.SAFE


def test_pack_loader_rejects_tampered_model(tmp_path: Path) -> None:
    pack = runtime_pack(tmp_path)
    (pack / "source.gguf").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="model hash differs"):
        load_gguf_runtime_pack(pack)


def test_pack_loader_rejects_path_escape_and_self_authorization(tmp_path: Path) -> None:
    pack = runtime_pack(tmp_path)
    manifest_path = pack / PACK_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["path"] = "../source.gguf"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside"):
        load_gguf_runtime_pack(pack)

    manifest["model"]["path"] = "source.gguf"
    manifest["publication_authorized"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot self-authorize"):
        load_gguf_runtime_pack(pack)

    manifest["publication_authorized"] = False
    manifest["purpose"] = "unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="purpose is invalid"):
        load_gguf_runtime_pack(pack)


def test_pack_loader_rejects_wrong_machine(tmp_path: Path) -> None:
    pack = runtime_pack(tmp_path)
    manifest_path = pack / PACK_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runner"]["system"] == platform.system()
    manifest["runner"]["machine"] = "not-this-machine"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not portable for this machine"):
        load_gguf_runtime_pack(pack)


def test_public_sdk_pack_benchmark_is_text_free(tmp_path: Path) -> None:
    pack = runtime_pack(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.jsonl").write_text(
        "\n".join(
            json.dumps({"id": identifier, "text": text})
            for identifier, text in (
                ("one", "A routine appointment reminder."),
                ("two", "A routine shipping update."),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = benchmark(pack=pack, data=data, split="test", rows=2, repetitions=2)

    assert report["data"]["requests"] == 4  # type: ignore[index]
    assert report["data"]["contains_message_text"] is False  # type: ignore[index]
    assert report["gates"]["all_requests_reused_prefix"] is True  # type: ignore[index]
    assert report["gates"]["runtime_does_not_import_transformers"] is True  # type: ignore[index]
