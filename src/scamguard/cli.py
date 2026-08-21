"""ScamGuard CLI."""

from __future__ import annotations

import argparse
import json
import sys

from .demo import serve
from .scanner import Scanner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scamguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan one message")
    scan_parser.add_argument("message", nargs="?", help="message; stdin is used when omitted")
    scan_parser.add_argument("--model", help="trusted local artifact file or model directory")

    demo_parser = subparsers.add_parser("demo", help="start the localhost-only demo")
    demo_parser.add_argument("--model", help="trusted local artifact file or model directory")
    demo_parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = _parser().parse_args()
    scanner = Scanner(model_path=args.model)
    if args.command == "scan":
        message = args.message if args.message is not None else sys.stdin.read()
        print(json.dumps(scanner.scan(message).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "demo":
        serve(scanner, port=args.port)


if __name__ == "__main__":
    main()
