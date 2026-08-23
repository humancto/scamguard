#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LLAMA_CPP_DIR OUTPUT.xcframework" >&2
  exit 2
fi

llama_cpp_dir=$(cd "$1" && pwd)
output=$2
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/scamguard-ios.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

build_slice() {
  local name=$1
  local sdk=$2
  local build_dir="$work_dir/$name"
  uv run --no-project --with cmake cmake -S native -B "$build_dir" -G Xcode \
    -DLLAMA_CPP_DIR="$llama_cpp_dir" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sdk" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.4 \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON
  uv run --no-project --with cmake cmake \
    --build "$build_dir" --config Release --parallel 8 --target scamguard-gguf

  local -a archives=()
  while IFS= read -r -d '' archive; do
    archives+=("$archive")
  done < <(find "$build_dir" -type f -name '*.a' -print0)
  if [[ ${#archives[@]} -lt 2 ]]; then
    echo "expected ScamGuard and llama.cpp static archives in $build_dir" >&2
    exit 1
  fi
  xcrun libtool -static -o "$work_dir/libScamGuardGGUF-$name.a" "${archives[@]}"
}

build_slice ios-device iphoneos
build_slice ios-simulator iphonesimulator

mkdir -p "$(dirname "$output")"
xcrun xcodebuild -create-xcframework \
  -library "$work_dir/libScamGuardGGUF-ios-device.a" -headers native/include \
  -library "$work_dir/libScamGuardGGUF-ios-simulator.a" -headers native/include \
  -output "$output"
