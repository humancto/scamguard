#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LLAMA_CPP_DIR BUILD_DIR" >&2
  exit 2
fi

llama_cpp_dir=$(cd "$1" && pwd)
build_dir=$2
cmake_args=(
  -DLLAMA_CPP_DIR="$llama_cpp_dir"
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=OFF
  -DLLAMA_OPENSSL=OFF
)
if [[ $(uname -s) == Darwin ]]; then
  cmake_args+=(
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DGGML_METAL=ON
  )
else
  cmake_args+=(-DGGML_METAL=OFF)
fi

uv run --no-project --with cmake cmake -S native -B "$build_dir" \
  "${cmake_args[@]}"
uv run --no-project --with cmake cmake \
  --build "$build_dir" --config Release --parallel 8 --target scamguard-gguf-verdict
