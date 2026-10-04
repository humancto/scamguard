# Protocol 4: chat-token correction and unresolved Unicode parity

## Decision

The native chat-boundary bug is corrected, but complete training-to-deployment tokenization
parity **fails**. This is a runtime diagnostic, not an accepted model, accuracy/latency benchmark,
or authorization to reopen the closed Qwen training line.

Protocol 3 called `llama_tokenize` with `add_special=true, parse_special=false` on an already
rendered ChatML prompt. Protocol 4 uses `false, true`: no implicit BOS/EOS insertion, and chat
control tokens are recognized. Cached prefixes and all verdict candidates use the same production
helper. The C ABI layout remains version 1. Literal special-token matching is compatibility
evidence, not a prompt-injection defense.

Current clients and validators reject protocol-3 or missing identity in native caches, calibration,
packs, mobile receipts, and release evidence. Historical reports and model artifacts remain
unchanged. Old native scores and timings describe the old input stream, not a validated correction.

## Real model-backed evidence

The vocabulary-only probe links the production helper. Ten synthetic messages cover empty input,
ordinary/scam/ambiguous text, Hindi, Chinese, decomposed accents, combined Unicode, literal control
tokens, and whitespace. Each contributes a question and three completed verdict candidates; the
fixed cached prefix supplies the 41st comparison. No dataset or sealed benchmark rows are read.

The receipt is
[`runs/qwen35-08b-q4-protocol4-tokenizer-investigation-v2.json`](runs/qwen35-08b-q4-protocol4-tokenizer-investigation-v2.json),
SHA-256 `2512c64b3ab68c7db1c817c19e5de891861e9d2f0ddd3ccd04d9bc5b525bf03c`.
It binds model, probe, runner, tokenizer files, script, and native-source hashes.

| Comparison | Exact matches | Meaning |
|---|---:|---|
| Native versus historical live AutoProcessor | 29/41 token sequences | Full historical training/runtime parity fails. |
| Native versus pinned serialized tokenizer.json | 33/41 token sequences | Native differs on decomposed accents. |
| NFC-normalized diagnostic input versus serialized reference | 41/41 token sequences | Isolates normalization; production inputs are unchanged. |
| Cached versus uncached native inference | 10/10 argmax and token-count matches | Functional consistency, not classification accuracy or calibrated parity. |

The largest cached/uncached raw-score difference was 0.000375899. The cached prefix contains
127 tokens. No calibrated decision policy or latency target was assessed.

## Two isolated causes

1. The pinned Transformers environment's `AutoProcessor` instantiates `Qwen2Tokenizer`. Its live
   constructor replaces the serialized Qwen3.5 pretokenizer with the older Qwen2 regex, omitting
   combining marks from letter runs. The receipt records both regex definitions. Native matches
   the serialized reference on the isolated Hindi fixture, not the historical live loader.
2. Both HF references apply NFC normalization. Native BPE does not. For `cafe` followed by U+0301,
   HF tokenizes the composed form while native retains the decomposed form. NFC applied only to
   ephemeral diagnostic requests resolves every serialized-reference fixture mismatch.

Do not silently replace the tokenizer used to train/evaluate historical adapters: it changes their
input identity. Their BF16 quality failures remain recorded failures; this diagnostic does not
demonstrate that a tokenizer change would repair them. A future candidate must freeze tokenizer,
loader behavior, normalization, prompt, and runtime together before training. Shared-native NFC
handling and the historical-versus-serialized tokenizer contract remain unresolved.

## Verification and reproduction

- Arm64 runner and vocabulary probe built; C ABI smoke passed. No external llama.cpp source edits.
- Full Python suite: 506 passed in 35.23 seconds; changed Python code passed Ruff.
- Swift wrapper/smoke passed offline simulator-target type checking. Kotlin wrapper/smoke compiled
  against the installed Android 30 SDK. Neither establishes device execution.
- Real native inference ran all ten synthetic fixtures, cached and uncached. Full tokenizer audit
  intentionally exits 1. No training, weight merge, quantization, or HF upload was performed.

After `make gguf-verdict-runner`:

```bash
uv run --offline --no-project --with cmake cmake --build build/native-gguf-verdict \
  --target scamguard-gguf-tokenize scamguard-gguf-c-abi-smoke
build/native-gguf-verdict/scamguard-gguf-c-abi-smoke
.venv/bin/python scripts/check_gguf_tokenizer_parity.py \
  --probe build/native-gguf-verdict/scamguard-gguf-tokenize \
  --model artifacts/gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
  --runner build/native-gguf-verdict/scamguard-gguf-verdict \
  --report reports/runs/qwen35-08b-q4-protocol4-tokenizer-reproduction.json
```

Existing report paths are refused, including failed reports. Keep the nonzero exit and mismatch
details; do not remove Unicode fixtures or count NFC-only diagnostic input as production parity.
