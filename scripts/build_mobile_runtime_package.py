#!/usr/bin/env python3
"""Build a deterministic, provenance-bound iOS or Android runtime package."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from scamguard.metrics import file_sha256

PACKAGE_SCHEMA_VERSION = 1
ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
PINNED_LLAMA_CPP_REVISION = "521a64cd01979bb5b1a466152c576a9d809b068d"


def verify_mobile_package(
    package: Path, *, expected_platform: str | None = None
) -> dict[str, object]:
    if not package.is_file():
        raise FileNotFoundError(package)
    with zipfile.ZipFile(package) as archive:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe mobile runtime package member: {info.filename}")
            if info.filename in members:
                raise ValueError(f"duplicate mobile runtime package member: {info.filename}")
            if info.date_time != ZIP_TIMESTAMP:
                raise ValueError(f"non-deterministic ZIP timestamp: {info.filename}")
            members[info.filename] = info
        manifest_name = "scamguard_mobile_runtime.manifest.json"
        if manifest_name not in members:
            raise ValueError("mobile runtime package manifest is missing")
        manifest = json.loads(archive.read(manifest_name))
        if not isinstance(manifest, dict):
            raise ValueError("mobile runtime package manifest must be an object")
        platform = manifest.get("platform")
        if platform not in {"ios", "android"} or (
            expected_platform is not None and platform != expected_platform
        ):
            raise ValueError("mobile runtime package platform is incompatible")
        required = {
            "artifact_schema_version": PACKAGE_SCHEMA_VERSION,
            "artifact_type": "scamguard_mobile_runtime_package",
            "publication_authorized": False,
            "architecture": "arm64",
            "native_abi_version": 1,
            "protocol_version": 3,
            "scoring_mode": "branch_token",
            "scoring_version": "qwen-verdict-branch-token-v1",
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError(f"mobile runtime manifest has incompatible {key}")
        source = manifest.get("source")
        if (
            not isinstance(source, dict)
            or source.get("llama_cpp_revision") != PINNED_LLAMA_CPP_REVISION
        ):
            raise ValueError("mobile runtime package has incompatible llama.cpp provenance")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("mobile runtime package has no file records")
        recorded_names: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("mobile runtime file record must be an object")
            name = record.get("path")
            if not isinstance(name, str) or name == manifest_name or name in recorded_names:
                raise ValueError("mobile runtime file record has an invalid path")
            if name not in members:
                raise ValueError(f"mobile runtime package member is missing: {name}")
            payload = archive.read(name)
            if record.get("bytes") != len(payload):
                raise ValueError(f"mobile runtime package byte count differs: {name}")
            if record.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"mobile runtime package hash differs: {name}")
            recorded_names.add(name)
        if set(members) != recorded_names | {manifest_name}:
            raise ValueError("mobile runtime package contains an unrecorded member")
        if platform == "android":
            required_names = {
                "runtime/jni/arm64-v8a/libscamguard-jni.so",
                "runtime/kotlin/com/scamguard/runtime/ScamGuardNative.kt",
                "LICENSE",
            }
        else:
            required_names = {
                "runtime/ScamGuardGGUF.xcframework/Info.plist",
                "runtime/ScamGuardRuntime.swift",
                "LICENSE",
            }
            if not any(name.endswith("/libScamGuardGGUF-ios-device.a") for name in recorded_names):
                raise ValueError("iOS runtime package lacks the physical-device library")
            if not any(
                name.endswith("/libScamGuardGGUF-ios-simulator.a")
                for name in recorded_names
            ):
                raise ValueError("iOS runtime package lacks the simulator library")
        if not required_names.issubset(recorded_names):
            raise ValueError(f"{platform} runtime package is incomplete")
    return manifest


def git_revision(repository: Path, *, require_clean: bool) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean:
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ValueError(f"runtime package source repository is dirty: {repository}")
    return revision


def _runtime_files(platform: str, runtime: Path, wrapper: Path) -> list[tuple[Path, str, int]]:
    if platform == "android":
        if not runtime.is_file() or runtime.name != "libscamguard-jni.so":
            raise ValueError("Android runtime must be libscamguard-jni.so")
        if not wrapper.is_file() or wrapper.suffix != ".kt":
            raise ValueError("Android wrapper must be a Kotlin source file")
        return [
            (runtime, "runtime/jni/arm64-v8a/libscamguard-jni.so", 0o755),
            (wrapper, "runtime/kotlin/com/scamguard/runtime/ScamGuardNative.kt", 0o644),
        ]
    if platform == "ios":
        if not runtime.is_dir() or runtime.suffix != ".xcframework":
            raise ValueError("iOS runtime must be an XCFramework directory")
        if not wrapper.is_file() or wrapper.suffix != ".swift":
            raise ValueError("iOS wrapper must be a Swift source file")
        files: list[tuple[Path, str, int]] = []
        for path in sorted(runtime.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"runtime package cannot contain symlinks: {path}")
            if path.is_file():
                relative = path.relative_to(runtime).as_posix()
                mode = 0o755 if path.suffix == ".a" else 0o644
                files.append((path, f"runtime/ScamGuardGGUF.xcframework/{relative}", mode))
        if not files:
            raise ValueError("iOS XCFramework contains no files")
        files.append((wrapper, "runtime/ScamGuardRuntime.swift", 0o644))
        return files
    raise ValueError(f"unsupported mobile platform: {platform}")


def _zip_entry(name: str, data: bytes, mode: int) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info, data


def build_mobile_package(
    *,
    platform: str,
    runtime: Path,
    wrapper: Path,
    license_path: Path,
    output: Path,
    scamguard_revision: str,
    llama_cpp_revision: str,
    toolchain_name: str,
    toolchain_version: str,
    minimum_os_version: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mobile runtime package: {output}")
    if llama_cpp_revision != PINNED_LLAMA_CPP_REVISION:
        raise ValueError("mobile runtime differs from the pinned llama.cpp revision")
    if not license_path.is_file():
        raise FileNotFoundError(license_path)
    if not all((scamguard_revision, toolchain_name, toolchain_version, minimum_os_version)):
        raise ValueError("runtime package provenance fields must be non-empty")

    inputs = _runtime_files(platform, runtime, wrapper)
    inputs.append((license_path, "LICENSE", 0o644))
    records = [
        {
            "path": archive_path,
            "sha256": file_sha256(source),
            "bytes": source.stat().st_size,
        }
        for source, archive_path, _mode in inputs
    ]
    manifest: dict[str, object] = {
        "artifact_schema_version": PACKAGE_SCHEMA_VERSION,
        "artifact_type": "scamguard_mobile_runtime_package",
        "publication_authorized": False,
        "platform": platform,
        "architecture": "arm64",
        "minimum_os_version": minimum_os_version,
        "native_abi_version": 1,
        "protocol_version": 3,
        "scoring_mode": "branch_token",
        "scoring_version": "qwen-verdict-branch-token-v1",
        "source": {
            "scamguard_revision": scamguard_revision,
            "llama_cpp_revision": llama_cpp_revision,
        },
        "toolchain": {"name": toolchain_name, "version": toolchain_version},
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, archive_path, mode in sorted(inputs, key=lambda item: item[1]):
            info, data = _zip_entry(archive_path, source.read_bytes(), mode)
            archive.writestr(info, data)
        info, data = _zip_entry("scamguard_mobile_runtime.manifest.json", manifest_bytes, 0o644)
        archive.writestr(info, data)
    verify_mobile_package(output, expected_platform=platform)
    return {
        **manifest,
        "package_path": str(output),
        "package_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "package_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("ios", "android"), required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--license", dest="license_path", type=Path, default=Path("LICENSE"))
    parser.add_argument("--scamguard-source", type=Path, default=Path("."))
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--toolchain-name", required=True)
    parser.add_argument("--toolchain-version", required=True)
    parser.add_argument("--minimum-os-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scamguard_revision = git_revision(args.scamguard_source, require_clean=True)
    llama_cpp_revision = git_revision(args.llama_source, require_clean=False)
    manifest = build_mobile_package(
        platform=args.platform,
        runtime=args.runtime,
        wrapper=args.wrapper,
        license_path=args.license_path,
        output=args.output,
        scamguard_revision=scamguard_revision,
        llama_cpp_revision=llama_cpp_revision,
        toolchain_name=args.toolchain_name,
        toolchain_version=args.toolchain_version,
        minimum_os_version=args.minimum_os_version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
