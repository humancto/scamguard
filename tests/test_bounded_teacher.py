from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import pytest

from benchmarks import run_bounded_teacher as teacher


def make_gguf(path: Path, template: str = "official template") -> None:
    def string(value: str) -> bytes:
        content = value.encode()
        return struct.pack("<Q", len(content)) + content

    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 3)
        + string("test.array")
        + struct.pack("<IIQ", 9, 8, 2)
        + string("a")
        + string("b")
        + string("test.number")
        + struct.pack("<II", 4, 7)
        + string("tokenizer.chat_template")
        + struct.pack("<I", 8)
        + string(template)
    )


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    source, model, server = (tmp_path / name for name in ("test.jsonl", "model.gguf", "server"))
    rows = []
    for label, count in (("SAFE", 350), ("UNCERTAIN", 50), ("SCAM", 200)):
        rows.extend(
            {
                "id": f"{label}-{i}",
                "label": label,
                "text": f"private message {label} {i}",
                "source": "synthetic-fixture",
                "category": "none",
            }
            for i in range(count)
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    make_gguf(model)
    server.write_bytes(b"mock-server")
    source_sha = teacher.file_sha256(source)
    monkeypatch.setattr(teacher, "SOURCE_SHA256", source_sha)
    manifest = teacher.prepare(source, model, server, expected_sha=source_sha)
    path = tmp_path / "manifest.json"
    teacher.write_new(path, manifest)
    return path, manifest


class MockServer:
    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.requests = []
        self.verdict = "SAFE"
        self.finish_reason = "stop"
        self.reasoning = None
        self.token_count = 100
        self.template = "official template"
        self.raise_after = None
        self.generations = 0

    def request(self, endpoint: str, payload: dict | None = None) -> dict:
        self.requests.append((endpoint, payload))
        if endpoint == "/props":
            return {
                "model_path": self.manifest["teacher"]["model_path"],
                "chat_template": self.template,
                "total_slots": 1,
                "default_generation_settings": {"n_ctx": 2048},
                "build_info": "mock",
            }
        if endpoint == "/apply-template":
            return {"prompt": "formatted official prompt"}
        if endpoint == "/tokenize":
            return {"tokens": list(range(self.token_count))}
        if endpoint == "/v1/chat/completions":
            self.generations += 1
            if self.raise_after is not None and self.generations > self.raise_after:
                raise TimeoutError("mock deadline")
            return {
                "choices": [
                    {
                        "finish_reason": self.finish_reason,
                        "message": {
                            "content": json.dumps({"verdict": self.verdict}),
                            "reasoning_content": self.reasoning,
                        },
                    }
                ]
            }
        raise AssertionError(endpoint)


def mock_run(frozen, tmp_path, monkeypatch, mock=None):
    path, manifest = frozen
    mock = mock or MockServer(manifest)
    monkeypatch.setattr(teacher, "LocalServer", lambda url, deadline: mock)
    output = tmp_path / "teacher.json"
    result = teacher.run(path, output, "http://127.0.0.1:8097", time.time())
    return output, result, mock


def test_prepare_is_deterministic_text_free_and_proportional(frozen) -> None:
    _, manifest = frozen
    teacher.validate_manifest(manifest)
    assert manifest["counts"]["selected"] == {"SAFE": 171, "UNCERTAIN": 29, "SCAM": 100}
    assert len(manifest["selection"]) == 300
    assert "private message" not in json.dumps(manifest)
    source = Path(manifest["source"]["path"])
    rows = teacher.read_rows(source)
    selected, counts = teacher.select_rows(list(reversed(rows)))
    assert [row["id"] for row in selected] == [row["id"] for row in manifest["selection"]]
    assert counts == manifest["counts"]
    assert manifest["eligibility_authorization"] is False


def test_selection_sparse_label_minimum_and_unique_ids() -> None:
    rows = [
        {"id": f"{label}-{i}", "label": label, "text": "fixture"}
        for label, count in (("SAFE", 400), ("UNCERTAIN", 3), ("SCAM", 400))
        for i in range(count)
    ]
    _, counts = teacher.select_rows(rows)
    assert counts["selected"]["UNCERTAIN"] == 3
    assert sum(counts["selected"].values()) == 300
    with pytest.raises(ValueError, match="duplicate"):
        teacher.select_rows(rows + [rows[0]])


@pytest.mark.parametrize("field", ["protocol", "selection", "source"])
def test_manifest_tamper_is_rejected(frozen, field) -> None:
    _, manifest = frozen
    manifest[field] = {}
    with pytest.raises(ValueError, match="binding"):
        teacher.validate_manifest(manifest)


def test_wrong_source_hash_rejected(frozen) -> None:
    _, manifest = frozen
    with pytest.raises(ValueError, match="source differs"):
        teacher.prepare(
            Path(manifest["source"]["path"]),
            Path(manifest["teacher"]["model_path"]),
            Path(manifest["teacher"]["server_path"]),
            expected_sha="0" * 64,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1",
        "http://example.org",
        "http://127.0.0.1@evil.test",
        "http://127.0.0.1/proxy",
        "http://localhost?target=evil",
    ],
)
def test_only_loopback_server_allowed(url) -> None:
    with pytest.raises(ValueError, match="loopback"):
        teacher.LocalServer(url, time.time() + 2)


def test_expired_budget_makes_no_http_request() -> None:
    client = teacher.LocalServer("http://127.0.0.1:8097", time.time() - 1)
    with pytest.raises(TimeoutError, match="budget"):
        client.request("/props")


def test_redirect_is_not_followed() -> None:
    with pytest.raises(ValueError, match="redirects"):
        teacher.NoRedirect().redirect_request(None, None, 302, "", {}, "http://evil.test")


def test_completed_run_has_all_rows_bound_and_frozen_generation(
    frozen, tmp_path, monkeypatch
) -> None:
    output, result, mock = mock_run(frozen, tmp_path, monkeypatch)
    assert result["complete"] is True
    assert result["completed_examples"] == 300
    assert mock.generations == 303
    assert result["prediction_ledger"]["sha256"] == teacher.file_sha256(
        output.with_suffix(".predictions.jsonl")
    )
    assert "private message" not in output.read_text()
    assert "private message" not in output.with_suffix(".predictions.jsonl").read_text()
    generations = [
        payload for endpoint, payload in mock.requests if endpoint == "/v1/chat/completions"
    ]
    assert all(
        payload["max_tokens"] == 24 and payload["temperature"] == 0 for payload in generations
    )
    assert all(payload["reasoning_effort"] == "none" for payload in generations)
    with pytest.raises(FileExistsError):
        teacher.run(frozen[0], output, "http://127.0.0.1:8097", time.time())


def test_partial_failure_is_terminal_not_resampled(frozen, tmp_path, monkeypatch) -> None:
    mock = MockServer(frozen[1])
    mock.raise_after = 8  # Three smoke calls followed by five measured examples.
    output, result, _ = mock_run(frozen, tmp_path, monkeypatch, mock)
    assert result["status"] == "terminal_incomplete"
    assert result["completed_examples"] == 5
    assert result["complete"] is False
    assert len(teacher.read_rows(output.with_suffix(".predictions.jsonl"))) == 5


@pytest.mark.parametrize("bad", ["template", "finish", "reasoning", "context", "verdict"])
def test_incompatible_teacher_fails_before_benchmark(frozen, tmp_path, monkeypatch, bad) -> None:
    mock = MockServer(frozen[1])
    if bad == "template":
        mock.template = "wrong template"
    elif bad == "finish":
        mock.finish_reason = "length"
    elif bad == "reasoning":
        mock.reasoning = "unexpected reasoning"
    elif bad == "context":
        mock.token_count = 2040
    else:
        mock.verdict = "MAYBE"
    _, result, _ = mock_run(frozen, tmp_path, monkeypatch, mock)
    assert result["complete"] is False
    assert result["completed_examples"] == 0


def bound_candidate(tmp_path, manifest, rows=None):
    if rows is None:
        rows = [
            row
            | {
                "verdict": row["truth"],
                "threshold_scam": row["truth"] == "SCAM",
                "probabilities": {label: float(label == row["truth"]) for label in teacher.LABELS},
            }
            for row in manifest["selection"]
        ]
    ledger = tmp_path / "candidate.predictions.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = tmp_path / "candidate.json"
    report.write_text(
        json.dumps(
            {
                "data_sha256": {"test": manifest["source"]["sha256"]},
                "prediction_ledger": {"sha256": teacher.file_sha256(ledger), "examples": len(rows)},
            }
        )
    )
    return report, ledger


def test_compare_complete_bound_exact_pairs_all_labels(frozen, tmp_path, monkeypatch) -> None:
    output, _, _ = mock_run(frozen, tmp_path, monkeypatch)
    report, ledger = bound_candidate(tmp_path, frozen[1])
    result = teacher.compare(
        frozen[0], output, output.with_suffix(".predictions.jsonl"), report, ledger
    )
    assert result["candidate"]["macro_f1"] == 1
    assert result["candidate"]["binary"]["recall"] == 1
    assert result["teacher"]["binary"]["recall"] == 0
    assert result["candidate"]["confusion"] == [[171, 0, 0], [0, 29, 0], [0, 0, 100]]
    assert result["teacher"]["alert_composition"]["alerts"] == 0
    assert result["candidate_minus_teacher"]["recall"]["ci95_paired_bootstrap"] == [1, 1]
    assert result["eligibility_authorization"] is False
    assert result["diagnostic_only"] is True


@pytest.mark.parametrize(
    "bad",
    [
        "duplicate",
        "missing",
        "truth",
        "text_sha256",
        "verdict",
        "decision",
        "probability",
        "tamper",
        "count",
        "source",
        "teacher_incomplete",
    ],
)
def test_compare_rejects_invalid_pairs_or_binding(frozen, tmp_path, monkeypatch, bad) -> None:
    output, _, _ = mock_run(frozen, tmp_path, monkeypatch)
    rows = [
        row | {"verdict": row["truth"], "threshold_scam": row["truth"] == "SCAM"}
        for row in frozen[1]["selection"]
    ]
    if bad == "duplicate":
        rows.append(rows[0])
    elif bad == "missing":
        rows.pop()
    elif bad in {"truth", "text_sha256", "verdict"}:
        rows[0][bad] = "invalid"
    elif bad == "decision":
        rows[0]["threshold_scam"] = not rows[0]["threshold_scam"]
    elif bad == "probability":
        rows[0]["probabilities"] = [float("nan"), 0, 1]
    report, ledger = bound_candidate(tmp_path, frozen[1], rows)
    if bad == "tamper":
        ledger.write_text(ledger.read_text() + "\n")
    elif bad in {"count", "source"}:
        receipt = json.loads(report.read_text())
        if bad == "count":
            receipt["prediction_ledger"]["examples"] += 1
        else:
            receipt["data_sha256"]["test"] = "wrong"
        report.write_text(json.dumps(receipt))
    elif bad == "teacher_incomplete":
        receipt = json.loads(output.read_text())
        receipt["complete"] = False
        output.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        teacher.compare(frozen[0], output, output.with_suffix(".predictions.jsonl"), report, ledger)


def test_no_output_overwrite(tmp_path) -> None:
    output = tmp_path / "receipt.json"
    teacher.write_new(output, {"historical": True})
    with pytest.raises(FileExistsError):
        teacher.write_new(output, {"historical": False})
    assert json.loads(output.read_text())["historical"] is True
