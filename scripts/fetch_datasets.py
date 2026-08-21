#!/usr/bin/env python3
"""Fetch only approved, redistributable source datasets with pinned hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import certifi


@dataclass(frozen=True)
class Source:
    key: str
    filename: str
    url: str
    sha256: str
    license: str
    citation: str
    use: str


SOURCES = (
    Source(
        key="uci_sms_spam",
        filename="uci_sms_spam.zip",
        url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        sha256="1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
        license="CC-BY-4.0",
        citation="https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        use="SAFE ham and UNCERTAIN generic spam; spam is not relabeled as scam",
    ),
    Source(
        key="mendeley_sms_phishing",
        filename="sms_phishing_5971.zip",
        url=(
            "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
            "edb361de-918d-469f-9106-e84823830665/file_downloaded"
        ),
        sha256="9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3",
        license="CC-BY-4.0",
        citation="https://doi.org/10.17632/f45bkkt8pr.1",
        use="training/dev/test after cross-source deduplication and family splitting",
    ),
    Source(
        key="mendeley_financial_scam",
        filename="financial_scam.csv",
        url=(
            "https://data.mendeley.com/public-files/datasets/znsk27yk3h/files/"
            "d8cdbed5-0c31-4010-82e7-ed432c4bc313/file_downloaded"
        ),
        sha256="58ec802e4de8768643e3be0badbaedb3ae5017a273e8f3898e72f6a29cbc96ad",
        license="CC-BY-4.0",
        citation="https://doi.org/10.17632/znsk27yk3h.2",
        use="out-of-domain holdout only; never used for model or threshold fitting",
    ),
    Source(
        key="wspr_sms_phishing",
        filename="wspr_phishing_messages.csv",
        url=(
            "https://raw.githubusercontent.com/wspr-ncsu/sms-phishing/"
            "ef7da01dfc145ce283a2af380e5dd8b817191ee7/phishing_messages.csv"
        ),
        sha256="d125c394af792faeb2b71b3f7100cad3b4cece02aff78cb4ec1fb9e7db90f230",
        license="MIT",
        citation=(
            "https://github.com/wspr-ncsu/sms-phishing/commit/"
            "ef7da01dfc145ce283a2af380e5dd8b817191ee7"
        ),
        use=(
            "one deterministic representative per campaign-like template family; "
            "80% training enrichment and 20% no-SAFE family holdout"
        ),
    ),
    Source(
        key="imc25_public_forum_smishing",
        filename="imc25_public_forum_smishing.csv",
        url=(
            "https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/"
            "a6175560b57387199871e51fbef6bc523d2516b4/dataset/final_dataset_output.csv"
        ),
        sha256="1bbd1e9e82c3ea023112207b80da268a5c4a07d2353c2b0898360ab037fa9a64",
        license="CC-BY-4.0",
        citation="https://doi.org/10.1145/3730567.3764431",
        use=(
            "privacy-normalized, near-clustered public-forum reports; deterministic capped "
            "training enrichment, selection-only validation, and unseen-family OOD holdout"
        ),
    ),
    Source(
        key="azsc_azerbaijani_sms",
        filename="azsc_azerbaijani_sms.csv",
        url=(
            "https://raw.githubusercontent.com/vusalshahbaz/"
            "sms-classification-dataset-azerbaijan/"
            "f3ebfa36103fb71731cc984a00f1e648c4a5dc8d/dataset.csv"
        ),
        sha256="3ffaf4d38daa7e9fd1dcf0b292ae12a8c73eca8261b8ced89788241f9216acbf",
        license="CC-BY-4.0",
        citation="https://doi.org/10.25045/jpit.v17.i1.04",
        use=(
            "multilingual OOD diagnostic only; the paper describes a mixture of consented "
            "user SMS, translated UCI rows, and self-generated rows without per-row provenance"
        ),
    ),
    Source(
        key="chichewa_sms_fraud",
        filename="chichewa_sms_fraud.xlsx",
        url=(
            "https://zenodo.org/api/records/14607454/files/"
            "SMS_Fraud_Chichewa_Dataset_SM.xlsx/content"
        ),
        sha256="4f83cfaab196f8fab3bdbf9c89e15313ddaa889da066335fcc2f35cc6b3f487a",
        license="CC-BY-4.0",
        citation="https://doi.org/10.5281/zenodo.14607454",
        use=(
            "privacy-normalized Chichewa-only external diagnostic; one representative per "
            "near-template family; translations excluded; never used for fitting or thresholding"
        ),
    ),
    Source(
        key="scam_dialogue",
        filename="scam_dialogue_all.csv",
        url=(
            "https://huggingface.co/datasets/BothBosu/scam-dialogue/resolve/"
            "321b961b5ae353e19ed479b960658dcd223d5e06/scam-dialogue_all.csv"
        ),
        sha256="fe8a8fa0aa2b8afb0b0a672fb7f9739b323cb6dd12064f786a68c2a1f49a4e0b",
        license="Apache-2.0",
        citation="https://huggingface.co/datasets/BothBosu/scam-dialogue",
        use=(
            "family-collapsed synthetic multi-turn external diagnostic only; excluded from "
            "fitting and thresholding and never counted as real data"
        ),
    ),
    Source(
        key="taskmaster1_woz_dialogues",
        filename="taskmaster1_woz_dialogues.json",
        url=(
            "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/"
            "d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/woz-dialogs.json"
        ),
        sha256="cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f",
        license="CC-BY-4.0",
        citation=(
            "https://github.com/google-research-datasets/Taskmaster/tree/"
            "d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019"
        ),
        use=(
            "privacy-normalized human-authored Wizard-of-Oz transactional dialogue hard "
            "negatives; deterministic conversation-family train/selection partition; "
            "reported separately from naturally occurring communications"
        ),
    ),
    Source(
        key="moz_smishing",
        filename="moz_smishing.csv",
        url=(
            "https://huggingface.co/datasets/MOZNLP/MOZ-Smishing/resolve/"
            "1092f9d9a545b29ae6be030ee9713b615fc2d987/test.csv"
        ),
        sha256="814a11d9b05741c4b47eb0d0784b1fd12a2a076f83a714a9908bdda594986ab8",
        license="CreativeML-OpenRAIL-M (publisher-declared; dataset-specific scope unclear)",
        citation="https://doi.org/10.18653/v1/2025.africanlp-1.23",
        use=(
            "local, privacy-normalized, newly sourced evaluation holdout only; excluded from "
            "training and public data redistribution pending dataset-specific license clarification"
        ),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source: Source, destination: Path, *, force: bool = False) -> None:
    output = destination / source.filename
    if output.exists() and not force:
        if sha256(output) == source.sha256:
            print(f"verified {source.key}: {output}")
            return
        raise RuntimeError(f"existing file has wrong hash: {output}; use --force")

    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "ScamGuard-dataset-fetcher/0.1 (+research; hash-pinned)"},
    )
    temporary: Path | None = None
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=90, context=context) as response:
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                temporary = Path(handle.name)
                while block := response.read(1024 * 1024):
                    handle.write(block)
        actual = sha256(temporary)
        if actual != source.sha256:
            raise RuntimeError(
                f"hash mismatch for {source.key}: expected {source.sha256}, got {actual}"
            )
        os.replace(temporary, output)
        print(f"downloaded {source.key}: {output}")
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for source in SOURCES:
        fetch(source, args.destination, force=args.force)
    manifest = args.destination / "sources.json"
    manifest.write_text(
        json.dumps([asdict(source) for source in SOURCES], indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote provenance manifest: {manifest}")


if __name__ == "__main__":
    main()
