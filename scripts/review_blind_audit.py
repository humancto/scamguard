#!/usr/bin/env python3
"""Dependency-free localhost reviewer for a ScamGuard blind-audit bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import tempfile
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "scamguard_blind_audit.csv"
MANIFEST_PATH = ROOT / "scamguard_blind_audit.manifest.json"
FIELDS = ("id", "text", "auditor_label", "contains_sensitive_data", "notes")
LABELS = {"SAFE", "UNCERTAIN", "SCAM"}
BOOLEANS = {"yes", "no"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
EXPECTED_PROTOCOL_SHA256 = "d9dcc931447ce5229ca5e07398b20944759dfe0f0224369c14c2729be10cbb59"
MAX_REQUEST_BYTES = 16_384
MAX_NOTES_LENGTH = 2_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def ids_sha256(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(row.get("id", "")) for row in rows)).encode()
    ).hexdigest()


def inputs_sha256(rows: list[dict[str, str]]) -> str:
    canonical = [
        {"id": str(row.get("id", "")), "text": str(row.get("text", ""))}
        for row in sorted(rows, key=lambda item: str(item.get("id", "")))
    ]
    return canonical_sha256(canonical)


def load_bundle() -> tuple[dict[str, object], list[dict[str, str]]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    errors: list[str] = []
    if manifest.get("artifact_schema_version") != 1:
        errors.append("unsupported bundle schema")
    if manifest.get("artifact_type") != "scamguard_blind_label_audit_bundle":
        errors.append("invalid bundle artifact type")
    if manifest.get("review_id_scheme") != "sha256-domain-separated-128-v1":
        errors.append("invalid opaque review-ID scheme")
    if manifest.get("review_order") != "opaque-review-id-lexicographic-v1":
        errors.append("invalid blind review order")
    if fieldnames != FIELDS or manifest.get("blind_fields") != list(FIELDS):
        errors.append("blind CSV fields differ from the frozen schema")
    if manifest.get("review_app_sha256") != file_sha256(Path(__file__).resolve()):
        errors.append("review application differs from the bundle manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("frozen audit protocol is missing")
    elif canonical_sha256(protocol) != EXPECTED_PROTOCOL_SHA256:
        errors.append("frozen audit protocol differs from this reviewed application")
    if manifest.get("audit_protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        errors.append("audit protocol hash differs from the reviewed application")
    if not rows or manifest.get("selected_rows") != len(rows):
        errors.append("blind CSV row count differs from the bundle manifest")
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(rows):
        errors.append("blind CSV IDs are empty or duplicated")
    if manifest.get("selected_ids_sha256") != ids_sha256(rows):
        errors.append("blind CSV IDs differ from the bundle manifest")
    if manifest.get("blind_inputs_sha256") != inputs_sha256(rows):
        errors.append("blind CSV message inputs differ from the bundle manifest")
    for index, row in enumerate(rows, start=2):
        label = str(row.get("auditor_label", "")).strip().upper()
        sensitive = str(row.get("contains_sensitive_data", "")).strip().casefold()
        notes = str(row.get("notes", ""))
        populated = bool(label or sensitive or notes.strip())
        if populated and (label not in LABELS or sensitive not in BOOLEANS):
            errors.append(f"row {index} has an incomplete or invalid decision")
        if len(notes) > MAX_NOTES_LENGTH:
            errors.append(f"row {index} notes exceed {MAX_NOTES_LENGTH} characters")
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return manifest, rows


def is_complete(row: dict[str, str]) -> bool:
    return (
        str(row.get("auditor_label", "")).strip().upper() in LABELS
        and str(row.get("contains_sensitive_data", "")).strip().casefold() in BOOLEANS
    )


def safe_row(row: dict[str, str], index: int) -> dict[str, object]:
    return {
        "id": row["id"],
        "index": index,
        "text": row["text"],
        "auditor_label": row["auditor_label"],
        "contains_sensitive_data": row["contains_sensitive_data"],
        "notes": row["notes"],
        "complete": is_complete(row),
    }


def state(index: int | None = None) -> dict[str, object]:
    _, rows = load_bundle()
    complete = sum(is_complete(row) for row in rows)
    if index is None:
        index = next((offset for offset, row in enumerate(rows) if not is_complete(row)), 0)
    index = max(0, min(index, len(rows) - 1))
    return {
        "row": safe_row(rows[index], index),
        "total": len(rows),
        "complete": complete,
        "remaining": len(rows) - complete,
        "review_finished": complete == len(rows),
    }


def atomic_write(rows: list[dict[str, str]]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=ROOT,
            prefix=".audit.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, CSV_PATH)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def update(identifier: str, label: str, sensitive: bool, notes: str) -> dict[str, object]:
    _, rows = load_bundle()
    label = label.strip().upper()
    if label not in LABELS:
        raise ValueError("auditor_label must be SAFE, UNCERTAIN, or SCAM")
    if not isinstance(sensitive, bool):
        raise ValueError("contains_sensitive_data must be a boolean")
    if not isinstance(notes, str) or len(notes.strip()) > MAX_NOTES_LENGTH:
        raise ValueError(f"notes must be text of at most {MAX_NOTES_LENGTH} characters")
    matching = [offset for offset, row in enumerate(rows) if row["id"] == identifier]
    if len(matching) != 1:
        raise ValueError("audit row ID was not found exactly once")
    index = matching[0]
    rows[index]["auditor_label"] = label
    rows[index]["contains_sensitive_data"] = "yes" if sensitive else "no"
    rows[index]["notes"] = notes.strip()
    atomic_write(rows)
    next_index = next(
        (offset for offset in range(index + 1, len(rows)) if not is_complete(rows[offset])),
        next((offset for offset, row in enumerate(rows) if not is_complete(row)), index),
    )
    return state(next_index)


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScamGuard blind review</title><style>
:root{color-scheme:light;--navy:#101a2e;--ink:#26344d;--muted:#66758e;--paper:#f5f8fc;--panel:#fff;--line:#d8e1ee;--blue:#1769aa;--red:#a9284a;--green:#16714d}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,sans-serif}.shell{width:min(1050px,calc(100% - 30px));margin:34px auto}h1{margin:.15em 0;color:var(--navy);font:600 clamp(30px,5vw,48px)/1.05 Georgia,serif}.eyebrow{color:var(--blue);font:700 12px monospace;letter-spacing:.12em;text-transform:uppercase}.blind{color:var(--muted);max-width:650px}.progress{height:7px;background:var(--line);border-radius:10px;overflow:hidden}.progress div{height:100%;background:var(--blue)}.meter{display:flex;justify-content:space-between;margin:8px 0 20px;color:var(--muted);font:12px monospace}main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:20px}.card{background:var(--panel);border:1px solid var(--line);box-shadow:0 20px 55px #273f631c}.message{min-height:430px;padding:42px;display:flex;flex-direction:column}.sample{color:var(--muted);font:12px monospace}.message blockquote{flex:1;display:grid;place-items:center;margin:24px 0;color:var(--navy);font:500 clamp(21px,3.5vw,34px)/1.42 Georgia,serif;overflow-wrap:anywhere}.local{color:var(--green);font-size:12px}.controls{padding:22px}.rubric{font-size:12px;color:var(--muted);line-height:1.4}.labels{display:grid;gap:8px}.label,.save,.nav button{padding:12px;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer;text-align:left}.label.selected{border:2px solid var(--blue);background:#eaf4fc}.label small{display:block;color:var(--muted);margin-top:3px}.check{display:flex;gap:9px;margin:16px 0;font-size:13px}textarea{width:100%;min-height:75px;padding:10px;border:1px solid var(--line)}.save{width:100%;margin-top:9px;background:var(--navy);color:#fff;text-align:center;font-weight:700}.save:disabled{opacity:.45}.nav{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.nav button{text-align:center}.status{min-height:18px;color:var(--muted);font-size:12px}.error{color:var(--red)}.done{display:none;margin-top:18px;padding:15px;border-left:5px solid var(--green);background:#e3f5ed}@media(max-width:760px){main{grid-template-columns:1fr}.message{min-height:320px;padding:28px}}</style></head><body><div class="shell"><p class="eyebrow">ScamGuard / independent review</p><h1>Read the evidence. Call it yourself.</h1><p class="blind">No project label, source, category, split, or model prediction exists in this package. Judge only the displayed message.</p><div class="progress"><div id="bar"></div></div><div class="meter"><span id="count">Loading</span><span id="remaining"></span></div><main><section class="card message"><span class="sample" id="sample"></span><blockquote id="message">Loading…</blockquote><span class="local">Localhost only · no message leaves this computer</span></section><aside class="card controls"><details open><summary>Frozen rubric</summary><div class="rubric"><ul id="principles"></ul></div></details><div class="labels" id="labels"></div><label class="check"><input id="sensitive" type="checkbox"><span id="sensitiveText"></span></label><textarea id="notes" maxlength="2000" placeholder="Optional note"></textarea><button class="save" id="save" disabled>Save &amp; next</button><div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div><p class="status" id="status"></p></aside></main><div class="done" id="done"><strong>All decisions are saved.</strong> Stop the server and return scamguard_blind_audit.csv.</div></div><script>
const token=__TOKEN__,protocol=__PROTOCOL__;let state=null,selected="";const $=id=>document.getElementById(id);function status(t,e=false){$("status").textContent=t;$("status").classList.toggle("error",e)}for(const [label,description] of Object.entries(protocol.labels)){const b=document.createElement("button");b.className="label";b.dataset.label=label;b.innerHTML=`<strong>${label}</strong><small></small>`;b.querySelector("small").textContent=description;b.onclick=()=>select(label);$("labels").appendChild(b)}for(const text of protocol.principles){const li=document.createElement("li");li.textContent=text;$("principles").appendChild(li)}$("sensitiveText").textContent=protocol.sensitive_data;function select(label){selected=label;document.querySelectorAll(".label").forEach(b=>b.classList.toggle("selected",b.dataset.label===label));$("save").disabled=!selected}function render(s){state=s;const r=s.row;$("sample").textContent=`Sample ${r.index+1} of ${s.total}`;$("message").textContent=r.text;$("count").textContent=`${s.complete} reviewed`;$("remaining").textContent=`${s.remaining} remaining`;$("bar").style.width=`${s.complete/s.total*100}%`;$("sensitive").checked=r.contains_sensitive_data.toLowerCase()==="yes";$("notes").value=r.notes;select(r.auditor_label);$("done").style.display=s.review_finished?"block":"none";status(r.complete?"Saved decision loaded.":"Not yet reviewed.")}async function load(index){try{const q=Number.isInteger(index)?`?index=${index}`:"";const response=await fetch(`/api/state${q}`,{cache:"no-store"});const payload=await response.json();if(!response.ok)throw Error(payload.error);render(payload)}catch(e){status(e.message,true)}}async function save(){if(!state||!selected)return;$("save").disabled=true;try{const response=await fetch("/api/row",{method:"POST",headers:{"Content-Type":"application/json","X-ScamGuard-Audit-Token":token},body:JSON.stringify({id:state.row.id,auditor_label:selected,contains_sensitive_data:$("sensitive").checked,notes:$("notes").value})});const payload=await response.json();if(!response.ok)throw Error(payload.error);render(payload)}catch(e){status(e.message,true);$("save").disabled=!selected}}$("save").onclick=save;$("prev").onclick=()=>load(Math.max(0,state.row.index-1));$("next").onclick=()=>load(Math.min(state.total-1,state.row.index+1));document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;if(e.key==="1")select("SAFE");if(e.key==="2")select("UNCERTAIN");if(e.key==="3")select("SCAM");if(e.key==="Enter")save()});load();</script></body></html>"""


def json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


class Server(HTTPServer):
    write_token: str
    protocol: dict[str, object]


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, format: str, *args: object) -> None:
        print(f"blind-audit-ui: {format % args}")

    def _local(self) -> bool:
        raw = self.headers.get("Host", "")
        host = raw.split("]", 1)[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
        origin = self.headers.get("Origin")
        return host.casefold() in LOOPBACK_HOSTS and (
            not origin or (urlparse(origin).hostname or "").casefold() in LOOPBACK_HOSTS
        )

    def _headers(self, status: HTTPStatus, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        self._headers(status, "application/json; charset=utf-8")
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())

    def do_GET(self) -> None:  # noqa: N802
        if not self._local():
            self._json(HTTPStatus.FORBIDDEN, {"error": "localhost request required"})
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                html = HTML.replace("__TOKEN__", json_for_script(self.server.write_token)).replace(
                    "__PROTOCOL__", json_for_script(self.server.protocol)
                )
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8")
                self.wfile.write(html.encode())
            elif parsed.path == "/api/state":
                raw = parse_qs(parsed.query).get("index", [None])[0]
                self._json(HTTPStatus.OK, state(int(raw) if raw is not None else None))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        if not self._local():
            self._json(HTTPStatus.FORBIDDEN, {"error": "localhost request required"})
            return
        if not secrets.compare_digest(
            self.headers.get("X-ScamGuard-Audit-Token", ""), self.server.write_token
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "write token is missing or invalid"})
            return
        try:
            if urlparse(self.path).path != "/api/row":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            result = update(
                str(payload.get("id", "")),
                str(payload.get("auditor_label", "")),
                payload.get("contains_sensitive_data"),
                payload.get("notes", ""),
            )  # type: ignore[arg-type]
            self._json(HTTPStatus.OK, result)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Validate the extracted bundle and exit."
    )
    args = parser.parse_args()
    if args.host.casefold() not in LOOPBACK_HOSTS:
        raise SystemExit("refusing non-loopback bind")
    manifest, rows = load_bundle()
    complete = sum(is_complete(row) for row in rows)
    if args.check:
        print(
            json.dumps(
                {
                    "valid": True,
                    "rows": len(rows),
                    "complete_rows": complete,
                    "remaining_rows": len(rows) - complete,
                    "contains_answer_key": False,
                },
                indent=2,
            )
        )
        return
    server = Server((args.host, args.port), Handler)
    server.write_token = secrets.token_urlsafe(32)
    server.protocol = manifest["protocol"]  # type: ignore[assignment]
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Independent blind-audit UI: {url}")
    print("No project labels or source metadata are present. Press Ctrl-C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBlind-audit UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
