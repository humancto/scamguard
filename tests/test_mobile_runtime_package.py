from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scamguard.gguf_runtime import UTF8PROC_LICENSE_MEMBER, UTF8PROC_LICENSE_SHA256
from scamguard.metrics import file_sha256
from scripts.build_mobile_runtime_package import (
    PINNED_LLAMA_CPP_REVISION,
    build_mobile_package,
    verify_mobile_package,
)


def build_android_fixture(tmp_path: Path, name: str) -> tuple[Path, dict[str, object]]:
    runtime = tmp_path / "libscamguard-jni.so"
    wrapper = tmp_path / "ScamGuardNative.kt"
    license_path = tmp_path / "LICENSE"
    runtime.write_bytes(b"android-runtime")
    wrapper.write_text("class ScamGuardNative\n", encoding="utf-8")
    license_path.write_text("Apache-2.0\n", encoding="utf-8")
    output = tmp_path / name
    manifest = build_mobile_package(
        platform="android",
        runtime=runtime,
        wrapper=wrapper,
        license_path=license_path,
        output=output,
        scamguard_revision="a" * 40,
        llama_cpp_revision=PINNED_LLAMA_CPP_REVISION,
        toolchain_name="Android NDK",
        toolchain_version="27.3.13750724",
        minimum_os_version="28",
    )
    return output, manifest


def test_android_package_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    first, manifest = build_android_fixture(tmp_path, "first.zip")
    second, _ = build_android_fixture(tmp_path, "second.zip")

    assert file_sha256(first) == file_sha256(second) == manifest["package_sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "LICENSE",
            UTF8PROC_LICENSE_MEMBER,
            "runtime/jni/arm64-v8a/libscamguard-jni.so",
            "runtime/kotlin/com/scamguard/runtime/ScamGuardNative.kt",
            "scamguard_mobile_runtime.manifest.json",
        ]
        embedded = json.loads(archive.read("scamguard_mobile_runtime.manifest.json"))
    assert embedded["publication_authorized"] is False
    assert embedded["platform"] == "android"
    assert embedded["protocol_version"] == 5
    license_record = next(
        record for record in embedded["files"] if record["path"] == UTF8PROC_LICENSE_MEMBER
    )
    assert license_record["sha256"] == UTF8PROC_LICENSE_SHA256
    assert embedded["toolchain"]["version"] == "27.3.13750724"
    assert embedded["files"][1]["sha256"] == file_sha256(tmp_path / "ScamGuardNative.kt")
    assert verify_mobile_package(first, expected_platform="android")["platform"] == "android"


def test_ios_package_walks_xcframework_without_symlinks(tmp_path: Path) -> None:
    runtime = tmp_path / "ScamGuardGGUF.xcframework"
    slice_dir = runtime / "ios-arm64"
    slice_dir.mkdir(parents=True)
    (runtime / "Info.plist").write_text("plist\n", encoding="utf-8")
    (slice_dir / "libScamGuardGGUF-ios-device.a").write_bytes(b"ios-runtime")
    simulator = runtime / "ios-arm64-simulator"
    simulator.mkdir()
    (simulator / "libScamGuardGGUF-ios-simulator.a").write_bytes(b"simulator-runtime")
    wrapper = tmp_path / "ScamGuardRuntime.swift"
    wrapper.write_text("public final class ScamGuardRuntime {}\n", encoding="utf-8")
    license_path = tmp_path / "LICENSE"
    license_path.write_text("Apache-2.0\n", encoding="utf-8")

    output = tmp_path / "ios.zip"
    manifest = build_mobile_package(
        platform="ios",
        runtime=runtime,
        wrapper=wrapper,
        license_path=license_path,
        output=output,
        scamguard_revision="b" * 40,
        llama_cpp_revision=PINNED_LLAMA_CPP_REVISION,
        toolchain_name="Xcode",
        toolchain_version="26.6",
        minimum_os_version="16.4",
    )

    assert manifest["platform"] == "ios"
    with zipfile.ZipFile(output) as archive:
        assert (
            "runtime/ScamGuardGGUF.xcframework/ios-arm64/libScamGuardGGUF-ios-device.a"
            in archive.namelist()
        )
        assert UTF8PROC_LICENSE_MEMBER in archive.namelist()


@pytest.mark.parametrize("protocol", [None, 3, 4])
def test_mobile_package_rejects_stale_protocol(
    tmp_path: Path, protocol: int | None
) -> None:
    package, _ = build_android_fixture(tmp_path, "runtime.zip")
    stale_package = tmp_path / "stale-runtime.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(stale_package, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "scamguard_mobile_runtime.manifest.json":
                record = json.loads(payload)
                if protocol is None:
                    del record["protocol_version"]
                else:
                    record["protocol_version"] = protocol
                payload = json.dumps(record).encode()
            target.writestr(info, payload)

    with pytest.raises(ValueError, match="incompatible protocol_version"):
        verify_mobile_package(stale_package)


@pytest.mark.parametrize("change", ["missing", "tampered", "rehashed"])
def test_mobile_package_requires_the_pinned_utf8proc_license(
    tmp_path: Path, change: str
) -> None:
    package, _ = build_android_fixture(tmp_path, "runtime.zip")
    stale_package = tmp_path / "bad-license.zip"
    replacement = b"different license"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(stale_package, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == UTF8PROC_LICENSE_MEMBER:
                if change == "missing":
                    continue
                payload = replacement
            elif info.filename == "scamguard_mobile_runtime.manifest.json":
                manifest = json.loads(payload)
                if change == "missing":
                    manifest["files"] = [
                        record for record in manifest["files"]
                        if record["path"] != UTF8PROC_LICENSE_MEMBER
                    ]
                elif change == "rehashed":
                    for record in manifest["files"]:
                        if record["path"] == UTF8PROC_LICENSE_MEMBER:
                            record.update(
                                sha256=hashlib.sha256(replacement).hexdigest(),
                                bytes=len(replacement),
                            )
                payload = json.dumps(manifest).encode()
            target.writestr(info, payload)

    with pytest.raises(ValueError, match="utf8proc"):
        verify_mobile_package(stale_package)


def test_package_refuses_wrong_llama_revision_and_overwrite(tmp_path: Path) -> None:
    output, _ = build_android_fixture(tmp_path, "runtime.zip")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_android_fixture(tmp_path, "runtime.zip")

    runtime = tmp_path / "libscamguard-jni.so"
    wrapper = tmp_path / "ScamGuardNative.kt"
    license_path = tmp_path / "LICENSE"
    with pytest.raises(ValueError, match="pinned llama.cpp"):
        build_mobile_package(
            platform="android",
            runtime=runtime,
            wrapper=wrapper,
            license_path=license_path,
            output=output.with_name("wrong.zip"),
            scamguard_revision="a" * 40,
            llama_cpp_revision="0" * 40,
            toolchain_name="Android NDK",
            toolchain_version="27.3.13750724",
            minimum_os_version="28",
        )


def test_verifier_rejects_unrecorded_and_duplicate_members(tmp_path: Path) -> None:
    package, _ = build_android_fixture(tmp_path, "runtime.zip")
    with zipfile.ZipFile(package, "a") as archive:
        info = zipfile.ZipInfo("unrecorded.txt", (2020, 1, 1, 0, 0, 0))
        archive.writestr(info, "not in manifest")
    with pytest.raises(ValueError, match="unrecorded member"):
        verify_mobile_package(package)

    duplicate, _ = build_android_fixture(tmp_path, "duplicate.zip")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "a") as archive:
            info = zipfile.ZipInfo("LICENSE", (2020, 1, 1, 0, 0, 0))
            archive.writestr(info, "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        verify_mobile_package(duplicate)
