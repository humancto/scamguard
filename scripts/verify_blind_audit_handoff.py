#!/usr/bin/env python3
"""Black-box preflight an extracted blind-audit handoff without retaining decisions."""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Final
from urllib.request import Request, urlopen

try:
    from scamguard.metrics import file_sha256
    from scripts.import_blind_audit import load_and_verify_bundle
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from import_blind_audit import load_and_verify_bundle  # type: ignore[no-redef]

    from scamguard.metrics import file_sha256

SCHEMA_VERSION: Final[int] = 1
STARTUP_TIMEOUT_SECONDS: Final[float] = 10.0
REQUEST_TIMEOUT_SECONDS: Final[float] = 5.0
STATE_FIELDS: Final[frozenset[str]] = frozenset(
    {"row", "total", "complete", "remaining", "review_finished"}
)
ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "index",
        "text",
        "auditor_label",
        "contains_sensitive_data",
        "notes",
        "complete",
    }
)
URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^Independent blind-audit UI: (http://127\.0\.0\.1:\d+/)$"
)
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'const token=("(?:[^"\\]|\\.)*")'
)


def _run_check(python: str, reviewer: Path, root: Path) -> dict[str, Any]:
    checked = subprocess.run(
        [python, "-I", str(reviewer), "--check"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=STARTUP_TIMEOUT_SECONDS,
    )
    value = json.loads(checked.stdout)
    if not isinstance(value, dict):
        raise ValueError("isolated reviewer check did not return a JSON object")
    return value


def _read_startup_line(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        raise ValueError("reviewer process has no stdout pipe")
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(STARTUP_TIMEOUT_SECONDS):
            raise TimeoutError("reviewer server did not announce a loopback URL")
        return process.stdout.readline().strip()
    finally:
        selector.close()


def _load_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError("reviewer API did not return a JSON object")
    return value


def _validate_state(value: dict[str, Any], expected_rows: int) -> None:
    if set(value) != STATE_FIELDS:
        raise ValueError("reviewer state exposes fields outside the frozen blind API")
    row = value.get("row")
    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
        raise ValueError("reviewer row exposes fields outside the frozen blind API")
    if value.get("total") != expected_rows:
        raise ValueError("reviewer API row count differs from the bundle manifest")
    if not isinstance(row.get("id"), str) or not re.fullmatch(
        r"sg-[0-9a-f]{32}", row["id"]
    ):
        raise ValueError("reviewer API did not expose an opaque row ID")
    if not isinstance(row.get("text"), str) or not row["text"]:
        raise ValueError("reviewer API message is empty")


def _exercise_server(
    python: str, reviewer: Path, root: Path, expected_rows: int
) -> dict[str, object]:
    process = subprocess.Popen(
        [python, "-I", "-u", str(reviewer), "--port", "0"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _read_startup_line(process)
        match = URL_PATTERN.fullmatch(line)
        if match is None:
            stderr = process.stderr.read() if process.poll() is not None and process.stderr else ""
            raise ValueError(f"reviewer did not bind to an ephemeral IPv4 loopback port: {stderr}")
        url = match.group(1)
        with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            html = response.read().decode("utf-8")
        token_match = TOKEN_PATTERN.search(html)
        if token_match is None:
            raise ValueError("review page did not contain its ephemeral write token")
        token = json.loads(token_match.group(1))
        initial = _load_json(f"{url}api/state")
        _validate_state(initial, expected_rows)
        if initial.get("complete") != 0 or initial.get("remaining") != expected_rows:
            raise ValueError("production handoff is not an untouched blank review")
        row = initial["row"]
        assert isinstance(row, dict)
        request = Request(
            f"{url}api/row",
            data=json.dumps(
                {
                    "id": row["id"],
                    "auditor_label": "UNCERTAIN",
                    "contains_sensitive_data": False,
                    "notes": "disposable handoff preflight; not a human decision",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-ScamGuard-Audit-Token": token,
            },
            method="POST",
        )
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            saved = json.loads(response.read())
        if not isinstance(saved, dict):
            raise ValueError("reviewer save endpoint did not return a JSON object")
        _validate_state(saved, expected_rows)
        if saved.get("complete") != 1 or saved.get("remaining") != expected_rows - 1:
            raise ValueError("reviewer did not persist exactly one disposable decision")
        return {
            "loopback_host": "127.0.0.1",
            "ephemeral_port": True,
            "state_schema_passed": True,
            "write_token_required": True,
            "disposable_save_passed": True,
        }
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=REQUEST_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=REQUEST_TIMEOUT_SECONDS)


def verify_handoff(bundle_path: Path, *, python: str = sys.executable) -> dict[str, object]:
    before_sha256 = file_sha256(bundle_path)
    manifest = load_and_verify_bundle(bundle_path)
    selected_rows = manifest.get("selected_rows")
    if not isinstance(selected_rows, int) or isinstance(selected_rows, bool) or selected_rows < 1:
        raise ValueError("bundle manifest has an invalid selected-row count")
    with tempfile.TemporaryDirectory(prefix="scamguard-audit-handoff-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(root)
        reviewer = root / "review.py"
        initial = _run_check(python, reviewer, root)
        expected_initial = {
            "valid": True,
            "rows": selected_rows,
            "complete_rows": 0,
            "remaining_rows": selected_rows,
            "contains_answer_key": False,
        }
        if initial != expected_initial:
            raise ValueError("isolated reviewer check differs from the blank handoff contract")
        server = _exercise_server(python, reviewer, root, selected_rows)
        resumed = _run_check(python, reviewer, root)
        if resumed.get("complete_rows") != 1 or resumed.get("remaining_rows") != selected_rows - 1:
            raise ValueError("isolated reviewer did not resume the disposable decision")
    after_sha256 = file_sha256(bundle_path)
    if after_sha256 != before_sha256:
        raise ValueError("handoff preflight modified the source bundle")
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "measurement_kind": "blind_audit_production_handoff_preflight",
        "passed": True,
        "contains_message_text": False,
        "contains_answer_key": False,
        "source_bundle_untouched": True,
        "rows": selected_rows,
        "initial_complete_rows": 0,
        "initial_remaining_rows": selected_rows,
        "isolated_python_check_passed": True,
        "save_resume_smoke_passed": True,
        "server": server,
        "bindings": {
            "bundle_path": str(bundle_path),
            "bundle_sha256": before_sha256,
            "bundle_bytes": bundle_path.stat().st_size,
            "blind_inputs_sha256": manifest["blind_inputs_sha256"],
            "canonical_audit_manifest_sha256": manifest[
                "canonical_audit_manifest_sha256"
            ],
            "review_app_sha256": manifest["review_app_sha256"],
            "review_csv_template_sha256": manifest["review_csv_template_sha256"],
            "audit_protocol_sha256": manifest["audit_protocol_sha256"],
            "verifier_sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def _write_report(path: Path, result: dict[str, object], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_handoff(args.bundle)
        if args.output is not None:
            _write_report(args.output, result, replace=args.replace)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.SubprocessError,
        TimeoutError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        print(f"blind-audit handoff preflight failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
