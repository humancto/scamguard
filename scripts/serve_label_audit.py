#!/usr/bin/env python3
"""Serve a local, blind review UI for a manifest-bound label-audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import tempfile
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from scripts.check_audit_completion import LABELS, audit_summary, validate_audit_binding
except ModuleNotFoundError:  # Direct execution places scripts/ rather than repo on sys.path.
    from check_audit_completion import (  # type: ignore[no-redef]
        LABELS,
        audit_summary,
        validate_audit_binding,
    )

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
MUTABLE_FIELDS = {"auditor_label", "label_correct", "contains_sensitive_data", "notes"}
MAX_REQUEST_BYTES = 16_384
MAX_NOTES_LENGTH = 2_000


def read_audit(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = MUTABLE_FIELDS - set(fieldnames)
    if missing:
        raise ValueError(f"audit is missing mutable fields: {', '.join(sorted(missing))}")
    identifiers = [row.get("id", "") for row in rows]
    if not rows:
        raise ValueError("audit contains no rows")
    if any(not identifier for identifier in identifiers):
        raise ValueError("audit contains an empty ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("audit contains duplicate IDs")
    return fieldnames, rows


def assert_binding(audit_path: Path, manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise ValueError(f"audit manifest is missing: {manifest_path}")
    _, errors = validate_audit_binding(audit_path, manifest_path)
    if errors:
        raise ValueError("; ".join(errors))


def is_complete(row: dict[str, str]) -> bool:
    return all(
        str(row.get(field, "")).strip()
        for field in ("auditor_label", "label_correct", "contains_sensitive_data")
    )


def safe_row(row: dict[str, str], index: int) -> dict[str, Any]:
    """Expose no project label, source metadata, category, or model output to the reviewer."""
    return {
        "id": row["id"],
        "index": index,
        "text": row.get("text", ""),
        "auditor_label": row.get("auditor_label", ""),
        "contains_sensitive_data": row.get("contains_sensitive_data", ""),
        "notes": row.get("notes", ""),
        "complete": is_complete(row),
    }


def audit_state(audit_path: Path, manifest_path: Path, index: int | None) -> dict[str, Any]:
    assert_binding(audit_path, manifest_path)
    _, rows = read_audit(audit_path)
    summary, errors = audit_summary(audit_path)
    if errors:
        raise ValueError("; ".join(errors))
    if index is None:
        index = next((i for i, row in enumerate(rows) if not is_complete(row)), 0)
    index = max(0, min(index, len(rows) - 1))
    return {
        "row": safe_row(rows[index], index),
        "total": len(rows),
        "complete": summary["complete_rows"],
        "remaining": summary["incomplete_rows"],
        "review_finished": summary["incomplete_rows"] == 0,
    }


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    original_mode = path.stat().st_mode
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, original_mode)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def update_audit_row(
    audit_path: Path,
    manifest_path: Path,
    identifier: str,
    auditor_label: str,
    contains_sensitive_data: bool,
    notes: str,
) -> dict[str, Any]:
    assert_binding(audit_path, manifest_path)
    fieldnames, rows = read_audit(audit_path)
    label = auditor_label.strip().upper()
    if label not in LABELS:
        raise ValueError("auditor_label must be SAFE, UNCERTAIN, or SCAM")
    if not isinstance(contains_sensitive_data, bool):
        raise ValueError("contains_sensitive_data must be a boolean")
    if not isinstance(notes, str):
        raise ValueError("notes must be text")
    notes = notes.strip()
    if len(notes) > MAX_NOTES_LENGTH:
        raise ValueError(f"notes must be at most {MAX_NOTES_LENGTH} characters")

    matching = [index for index, row in enumerate(rows) if row["id"] == identifier]
    if len(matching) != 1:
        raise ValueError("audit row ID was not found exactly once")
    index = matching[0]
    expected = rows[index].get("label", "").strip().upper()
    if expected not in LABELS:
        raise ValueError("project label is invalid")
    rows[index]["auditor_label"] = label
    rows[index]["label_correct"] = "yes" if label == expected else "no"
    rows[index]["contains_sensitive_data"] = "yes" if contains_sensitive_data else "no"
    rows[index]["notes"] = notes
    atomic_write_csv(audit_path, fieldnames, rows)
    assert_binding(audit_path, manifest_path)

    next_index = next(
        (offset for offset in range(index + 1, len(rows)) if not is_complete(rows[offset])),
        next((offset for offset, row in enumerate(rows) if not is_complete(row)), index),
    )
    return audit_state(audit_path, manifest_path, next_index)


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScamGuard independent label review</title>
<style>
:root{color-scheme:light;--navy:#101a2e;--ink:#26344d;--muted:#66758e;--paper:#f6f9fd;--panel:#fff;--line:#d9e2ef;--blue:#1769aa;--amber:#9a6100;--red:#a9284a;--green:#16714d;--shadow:0 22px 65px rgba(39,63,99,.13)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:linear-gradient(90deg,transparent 31px,rgba(23,105,170,.08) 32px,transparent 33px),linear-gradient(rgba(217,226,239,.38) 1px,transparent 1px),var(--paper);background-size:100% 100%,100% 28px,auto;color:var(--ink);font-family:"Avenir Next",Avenir,system-ui,sans-serif}button,textarea{font:inherit}.shell{width:min(1120px,calc(100% - 32px));margin:auto;padding:34px 0 52px}header{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:18px}.eyebrow{margin:0 0 4px;color:var(--blue);font:700 12px/1.2 ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase}h1{margin:0;color:var(--navy);font:600 clamp(28px,5vw,48px)/1.02 "Iowan Old Style",Georgia,serif;letter-spacing:-.025em}.blind{max-width:410px;margin:0;color:var(--muted);font-size:14px;line-height:1.5}.progress{height:7px;overflow:hidden;background:var(--line);border-radius:99px}.progress>div{width:0;height:100%;background:var(--blue);transition:width .25s}.meter{display:flex;justify-content:space-between;margin:9px 0 23px;color:var(--muted);font:650 12px/1.3 ui-monospace,monospace}main{display:grid;grid-template-columns:minmax(0,1fr) 315px;gap:22px}.card{background:rgba(255,255,255,.96);border:1px solid var(--line);box-shadow:var(--shadow)}.message-card{min-height:410px;padding:clamp(25px,5vw,54px);display:flex;flex-direction:column}.sample-number{color:var(--muted);font:650 12px/1.3 ui-monospace,monospace;letter-spacing:.06em}blockquote{flex:1;display:grid;place-items:center;margin:24px 0;color:var(--navy);font:500 clamp(21px,3.5vw,34px)/1.42 "Iowan Old Style",Georgia,serif;overflow-wrap:anywhere}.privacy{display:flex;align-items:center;gap:9px;color:var(--muted);font-size:12px}.privacy:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(22,113,77,.1)}.controls{padding:22px}.controls h2{margin:0 0 5px;color:var(--navy);font:650 18px/1.2 "Iowan Old Style",Georgia,serif}.hint{margin:0 0 19px;color:var(--muted);font-size:12px}.labels{display:grid;gap:9px}.label{width:100%;display:grid;grid-template-columns:28px 1fr;align-items:center;gap:10px;padding:13px 14px;border:1px solid var(--line);background:var(--panel);color:var(--ink);text-align:left;cursor:pointer;transition:transform .12s,border-color .12s,background .12s}.label:hover{transform:translateX(3px);border-color:var(--blue)}.label.selected[data-label=SAFE]{background:#e3f5ed;border-color:var(--green)}.label.selected[data-label=UNCERTAIN]{background:#fff1ca;border-color:var(--amber)}.label.selected[data-label=SCAM]{background:#ffe3eb;border-color:var(--red)}kbd{display:inline-grid;place-items:center;width:25px;height:25px;border:1px solid currentColor;border-radius:50%;font:700 11px/1 ui-monospace,monospace}.check{display:flex;gap:10px;align-items:flex-start;margin:19px 0 13px;font-size:13px;line-height:1.4}input[type=checkbox]{width:18px;height:18px;accent-color:var(--red)}textarea{width:100%;min-height:84px;resize:vertical;padding:11px;border:1px solid var(--line);background:var(--paper);color:var(--ink)}.save{width:100%;margin-top:12px;padding:13px;border:0;background:var(--navy);color:#fff;font-weight:700;cursor:pointer}.save:disabled{cursor:not-allowed;opacity:.42}.nav{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px}.nav button{padding:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}button:focus-visible,textarea:focus-visible,input:focus-visible{outline:3px solid var(--blue);outline-offset:3px}.status{min-height:20px;margin:12px 0 0;color:var(--muted);font-size:12px}.status.error{color:var(--red);font-weight:650}.finished{display:none;margin-top:22px;padding:18px 20px;border-left:5px solid var(--green);background:#e3f5ed;color:#10583c}.finished strong{display:block;color:var(--navy)}@media(max-width:760px){header{display:block}.blind{margin-top:12px}main{grid-template-columns:1fr}.message-card{min-height:330px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><div class="shell">
<header><div><p class="eyebrow">ScamGuard / independent review</p><h1>Read the evidence.<br>Call it yourself.</h1></div><p class="blind">Project labels and model predictions are hidden. Judge only the message in front of you.</p></header>
<div class="progress" aria-label="Review progress"><div id="progress"></div></div><div class="meter"><span id="count">Loading…</span><span id="remaining"></span></div>
<main><section class="card message-card" aria-labelledby="sample"><span class="sample-number" id="sample">Sample</span><blockquote id="message">Loading audit row…</blockquote><span class="privacy">Localhost only · no message text leaves this computer</span></section>
<aside class="card controls"><h2>Your verdict</h2><p class="hint">Keys 1–3 select. Enter saves and advances.</p><div class="labels" role="group" aria-label="Auditor label"><button class="label" data-label="SAFE"><kbd>1</kbd><span>Safe</span></button><button class="label" data-label="UNCERTAIN"><kbd>2</kbd><span>Uncertain</span></button><button class="label" data-label="SCAM"><kbd>3</kbd><span>Scam</span></button></div><label class="check"><input id="sensitive" type="checkbox"><span>Contains personal or sensitive data that should be quarantined</span></label><textarea id="notes" maxlength="2000" placeholder="Optional reviewer note"></textarea><button class="save" id="save" disabled>Save &amp; next <kbd>↵</kbd></button><div class="nav"><button id="previous">← Previous</button><button id="next">Next →</button></div><p class="status" id="status" role="status" aria-live="polite"></p></aside></main>
<div class="finished" id="finished"><strong>All rows have a decision.</strong>Return to the terminal and run the audit check. Any disagreement or sensitive-data finding still fails the release gate.</div></div>
<script>
const token=__TOKEN__;let state=null,selected="";const $=id=>document.getElementById(id),buttons=[...document.querySelectorAll(".label")];
function status(message,error=false){$("status").textContent=message;$("status").classList.toggle("error",error)}
function select(label){selected=label;buttons.forEach(b=>b.classList.toggle("selected",b.dataset.label===label));$("save").disabled=!selected}
function render(s){state=s;const r=s.row;$("sample").textContent=`Sample ${r.index+1} of ${s.total}`;$("message").textContent=r.text;$("count").textContent=`${s.complete} reviewed`;$("remaining").textContent=`${s.remaining} remaining`;$("progress").style.width=`${s.complete/s.total*100}%`;$("sensitive").checked=r.contains_sensitive_data.toLowerCase()==="yes";$("notes").value=r.notes;select(r.auditor_label);$("finished").style.display=s.review_finished?"block":"none";status(r.complete?"Saved decision loaded.":"Not yet reviewed.")}
async function load(index){try{const suffix=Number.isInteger(index)?`?index=${index}`:"",response=await fetch(`/api/state${suffix}`,{cache:"no-store"}),payload=await response.json();if(!response.ok)throw Error(payload.error||"Unable to load audit.");render(payload)}catch(error){status(error.message,true)}}
async function save(){if(!selected||!state)return;$("save").disabled=true;status("Saving locally…");try{const response=await fetch("/api/row",{method:"POST",headers:{"Content-Type":"application/json","X-ScamGuard-Audit-Token":token},body:JSON.stringify({id:state.row.id,auditor_label:selected,contains_sensitive_data:$("sensitive").checked,notes:$("notes").value})}),payload=await response.json();if(!response.ok)throw Error(payload.error||"Unable to save decision.");render(payload);status("Saved. Next incomplete sample loaded.")}catch(error){status(error.message,true);$("save").disabled=!selected}}
buttons.forEach(b=>b.addEventListener("click",()=>select(b.dataset.label)));$("save").addEventListener("click",save);$("previous").addEventListener("click",()=>load(Math.max(0,state.row.index-1)));$("next").addEventListener("click",()=>load(Math.min(state.total-1,state.row.index+1)));document.addEventListener("keydown",event=>{if(event.target.tagName==="TEXTAREA")return;if(event.key==="1")select("SAFE");if(event.key==="2")select("UNCERTAIN");if(event.key==="3")select("SCAM");if(event.key==="Enter"){event.preventDefault();save()}if(event.key==="ArrowLeft")load(Math.max(0,state.row.index-1));if(event.key==="ArrowRight")load(Math.min(state.total-1,state.row.index+1))});load();
</script></body></html>"""


class AuditHTTPServer(HTTPServer):
    audit_path: Path
    manifest_path: Path
    write_token: str


class AuditHandler(BaseHTTPRequestHandler):
    server: AuditHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"audit-ui: {format % args}")

    def _host_is_loopback(self) -> bool:
        raw = self.headers.get("Host", "")
        host = raw.split("]", 1)[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
        return host.casefold() in LOOPBACK_HOSTS

    def _origin_is_local(self) -> bool:
        origin = self.headers.get("Origin")
        return not origin or (urlparse(origin).hostname or "").casefold() in LOOPBACK_HOSTS

    def _headers(self, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.end_headers()

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._headers(status, "application/json; charset=utf-8")
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())

    def _reject_nonlocal(self) -> bool:
        if self._host_is_loopback() and self._origin_is_local():
            return False
        self._json(HTTPStatus.FORBIDDEN, {"error": "localhost request required"})
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self._reject_nonlocal():
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = HTML.replace("__TOKEN__", json.dumps(self.server.write_token)).encode()
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8")
                self.wfile.write(body)
            elif parsed.path == "/api/state":
                raw_index = parse_qs(parsed.query).get("index", [None])[0]
                index = int(raw_index) if raw_index is not None else None
                self._json(
                    HTTPStatus.OK,
                    audit_state(self.server.audit_path, self.server.manifest_path, index),
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        if self._reject_nonlocal():
            return
        if not secrets.compare_digest(
            self.headers.get("X-ScamGuard-Audit-Token", ""), self.server.write_token
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "write token is missing or invalid"})
            return
        if urlparse(self.path).path != "/api/row":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            result = update_audit_row(
                self.server.audit_path,
                self.server.manifest_path,
                str(payload.get("id", "")),
                str(payload.get("auditor_label", "")),
                payload.get("contains_sensitive_data"),
                payload.get("notes", ""),
            )
            self._json(HTTPStatus.OK, result)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("data/audit/schema24-label-audit.csv"))
    parser.add_argument(
        "--audit-manifest", type=Path, default=Path("data/audit/schema24-label-audit.manifest.json")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the review UI in a browser.")
    args = parser.parse_args()
    if args.host.casefold() not in LOOPBACK_HOSTS:
        raise SystemExit("refusing non-loopback bind; use 127.0.0.1, localhost, or ::1")
    assert_binding(args.audit, args.audit_manifest)
    read_audit(args.audit)
    server = AuditHTTPServer((args.host, args.port), AuditHandler)
    server.audit_path = args.audit
    server.manifest_path = args.audit_manifest
    server.write_token = secrets.token_urlsafe(32)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Independent audit UI: {url}")
    print("Project labels and model outputs are hidden. Press Ctrl-C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAudit UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
