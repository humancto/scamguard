"""Dependency-free, localhost-only demo server."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .scanner import Scanner

_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ScamGuard local demo</title>
<style>body{max-width:760px;margin:3rem auto;padding:0 1rem;font:17px system-ui;background:#08111f;color:#eef5ff}textarea{width:100%;min-height:160px;padding:1rem;box-sizing:border-box}button{margin-top:1rem;padding:.8rem 1.2rem}pre{white-space:pre-wrap;background:#10223a;padding:1rem;border-radius:10px}.note{color:#abc8e7}</style>
<h1>ScamGuard</h1><p class="note">Runs on this computer. The message is sent only to localhost.</p>
<textarea id="message" placeholder="Paste a suspicious message"></textarea><br>
<button id="scan">Scan locally</button><pre id="result">Waiting for a message.</pre>
<script>document.querySelector('#scan').onclick=async()=>{const message=document.querySelector('#message').value;const r=await fetch('/scan',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message})});document.querySelector('#result').textContent=JSON.stringify(await r.json(),null,2)}</script>
"""


def serve(scanner: Scanner, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            self._send(HTTPStatus.OK, _HTML.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/scan":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 120_000:
                    raise ValueError("request too large")
                payload = json.loads(self.rfile.read(length))
                result = scanner.scan(str(payload["message"]))
                body = json.dumps(result.to_dict(), ensure_ascii=False, indent=2).encode()
                self._send(HTTPStatus.OK, body, "application/json")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self._send(HTTPStatus.BAD_REQUEST, body, "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return

    print(f"ScamGuard local demo: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
