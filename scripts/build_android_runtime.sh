#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 LLAMA_CPP_DIR ANDROID_NDK_DIR BUILD_DIR" >&2
  exit 2
fi

llama_cpp_dir=$(cd "$1" && pwd)
android_ndk_dir=$(cd "$2" && pwd)
build_dir=$3
toolchain="$android_ndk_dir/build/cmake/android.toolchain.cmake"
expected_ndk_revision=27.3.13750724
if [[ ! -f "$toolchain" ]]; then
  echo "Android NDK toolchain not found: $toolchain" >&2
  exit 1
fi
source_properties="$android_ndk_dir/source.properties"
if [[ ! -f "$source_properties" ]]; then
  echo "Android NDK source.properties not found: $source_properties" >&2
  exit 1
fi
actual_ndk_revision=$(sed -n 's/^Pkg\.Revision[[:space:]]*=[[:space:]]*//p' "$source_properties")
if [[ "$actual_ndk_revision" != "$expected_ndk_revision" ]]; then
  echo "Android NDK revision mismatch: expected $expected_ndk_revision, got $actual_ndk_revision" >&2
  exit 1
fi

uv run --no-project --with cmake cmake -S native -B "$build_dir" \
  -DLLAMA_CPP_DIR="$llama_cpp_dir" \
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  -DSCAMGUARD_BUILD_JNI=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_METAL=OFF
uv run --no-project --with cmake cmake \
  --build "$build_dir" --config Release --parallel 8 --target scamguard-jni

strip_tools=("$android_ndk_dir"/toolchains/llvm/prebuilt/*/bin/llvm-strip)
if [[ ${#strip_tools[@]} -ne 1 || ! -x "${strip_tools[0]}" ]]; then
  echo "expected exactly one executable llvm-strip in the pinned Android NDK" >&2
  exit 1
fi
runtime_library="$build_dir/libscamguard-jni.so"
if [[ ! -f "$runtime_library" ]]; then
  echo "Android runtime library was not produced: $runtime_library" >&2
  exit 1
fi
"${strip_tools[0]}" --strip-unneeded "$runtime_library"
