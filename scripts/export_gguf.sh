#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LLAMA_CPP_DIR MERGED_MODEL_DIR OUTPUT_PREFIX PYTHON_BIN" >&2
  exit 2
fi

llama_cpp_dir=$1
merged_model_dir=$2
output_prefix=$3
python_bin=$4
converter="$llama_cpp_dir/convert_hf_to_gguf.py"
quantizer="$llama_cpp_dir/build-scamguard-arm64/bin/llama-quantize"
if [[ ! -x "$quantizer" ]]; then
  quantizer="$llama_cpp_dir/build-arm64/bin/llama-quantize"
fi

if [[ ! -f "$converter" || ! -x "$quantizer" || ! -x "$python_bin" ]]; then
  echo "llama.cpp converter, arm64 quantizer, or Python interpreter is missing" >&2
  exit 2
fi

mkdir -p "$(dirname "$output_prefix")"

"$python_bin" "$converter" "$merged_model_dir" \
  --outfile "${output_prefix}-bf16.gguf" \
  --outtype bf16 \
  --no-mtp
"$quantizer" "${output_prefix}-bf16.gguf" "${output_prefix}-q4_k_m.gguf" Q4_K_M
"$quantizer" "${output_prefix}-bf16.gguf" "${output_prefix}-q5_k_m.gguf" Q5_K_M
