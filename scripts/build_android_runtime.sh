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
if [[ ! -f "$toolchain" ]]; then
  echo "Android NDK toolchain not found: $toolchain" >&2
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
