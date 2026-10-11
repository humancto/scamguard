# Protocol 5: verified native NFC correction

## Decision

The native Unicode bug is fixed on the expanded model-backed regression probe: **81/81 exact
token sequences match the pinned published Qwen3.5 tokenizer**, compared with 53/81 before NFC.
This is a tokenizer/runtime result, not a model-quality result, exhaustive Unicode proof, latency
benchmark, release authorization, or permission to restart the closed Qwen adapter line.

The debugging sequence retained a failing before-fix receipt, isolated normalization from the
historical loader mismatch, and reran the same expanded fixtures after the correction. No dataset
or sealed benchmark rows were read. No training, weight merge, quantization, or HF upload occurred.

## Implementation and identity

- Protocol **5**, unchanged C ABI **1**. Both cached prefixes and full verdict candidates call the
  same production helper, which applies NFC before tokenization and retains the protocol-4
  `add_special=false, parse_special=true` ChatML handling.
- utf8proc **2.10.0 / Unicode 16.0.0**, upstream commit
  `a1b99daa2a3393884220264c927a48ba1251a9c6`, is vendored unmodified and statically linked.
  Explicit byte lengths preserve embedded NULs; malformed UTF-8 is rejected; allocated buffers
  have RAII ownership. No external llama.cpp sources were edited.
- [Vendor provenance](../native/vendor/utf8proc/README.scamguard.md) records the upstream files,
  revision, checksums, and MIT/Unicode notices. Runtime packs retain the exact pinned license at
  `licenses/utf8proc/LICENSE.md`; desktop/mobile/release validators verify its identity and bytes.
- Python, Swift, and Kotlin clients require protocol 5. Old calibration, caches, packs, and mobile
  evidence must not be relabeled or reused.

The published reference is `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. The explicit text-only opt-in loader in
[`published_qwen_tokenizer.py`](../src/scamguard/published_qwen_tokenizer.py) verifies the pinned
tokenizer/config/template hashes and all six canonical backend components after loading. Its
contract is `qwen35-08b-published-json-nfc-v1`.

**Historical training/evaluation loaders are deliberately unchanged.** Their `Qwen2Tokenizer`
constructor overrides the serialized combining-mark regex. Protocol 5 matches that historical
reference on only 69/81 sequences, with Hindi, combined Unicode, and Arabic still differing. The
new loader is reachable only through the explicit parity-audit opt-in; it is not silently wired
into training. A future experiment must freeze this complete contract before fitting any weights.
The old adapters' BF16 quality failures remain failures.

## Real-model evidence

Both receipts use the verified 563,036,064-byte upstream Q4_0 control, SHA-256
`57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.

| Check | Result | Scope |
|---|---:|---|
| Protocol 4, expanded published-reference probe | 53/81 exact sequences | Before-fix failing baseline |
| Protocol 5, published-reference probe | 81/81 exact sequences | 20 question-only sequences, 60 verdict completions, and one fixed prefix |
| Protocol 5, historical live HF reference | 69/81 exact sequences | Historical input incompatibility remains explicit |
| Cached versus uncached native inference | 20/20 top-verdict and token-count matches | Functional smoke, not calibrated accuracy |

The probe includes decomposed accents, Hindi, Chinese, Hangul Jamo, Arabic, Vietnamese, Greek,
Cyrillic, reordered combining marks, emoji joiners, compatibility characters that NFC must not
fold, embedded NULs, mixed newlines, literal special tokens, whitespace, and ordinary/scam/ambiguous
messages. Literal special-token parity is not a prompt-injection defense.

The 127-token prefix was reused for every cached request. Maximum absolute cached/uncached raw
score difference was **0.00114589652**; raw scores are not bitwise identical. The smoke asserts
finite scores, cache use, token-count agreement, and top-verdict agreement, not calibrated
threshold equivalence or a numerical error budget for deployment.

- [Before-fix receipt](runs/qwen35-08b-q4-protocol4-published-before-nfc.json), SHA-256
  `148538e99bf32ad3068ecdcecb043fa0c04ab98148535eb67bddb4fb7aeda5ff`.
- [After-fix receipt](runs/qwen35-08b-q4-protocol5-published-tokenizer-parity.json), SHA-256
  `addec772958f19bb38d5072119d6cad099aa916bbf33fd2f6d501f3871532e2e`.
- The earlier protocol-4 investigation receipt retains SHA-256
  `2512c64b3ab68c7db1c817c19e5de891861e9d2f0ddd3ccd04d9bc5b525bf03c`.

The after-fix receipt binds the probe/runner/model binaries, audit script, native normalization
sources, vendored dependency files, and both selected/historical tokenizer identities.

## Verification and reproduction

- Arm64 runner, production-linked vocabulary probe, and C ABI smoke built successfully.
- CTest: **2/2 passed**, including 17 no-model normalization/error cases and C ABI compatibility.
- Python suite: **542 passed in 34.31 seconds** on the final rerun; changed Python files passed Ruff.
  First-party diff whitespace checks pass. The unmodified upstream `utf8proc_data.c` retains its
  generated trailing whitespace/blank EOF so its recorded upstream checksum stays exact.
- Swift wrapper/smoke passed offline simulator-target type checking. Kotlin wrapper/smoke compiled
  offline against Android 30. Neither proves device execution. Platform native binaries still
  require rebuilding and physical-device verification before any mobile claim.
- `otool -L` shows only system libraries/frameworks for the arm64 runner; utf8proc is static.
- Independent read-only review found no actionable defect in normalization, linkage, explicit
  loader isolation, protocol migration, or license preservation.

After `make gguf-verdict-runner`:

```bash
uv run --offline --no-project --with cmake cmake --build build/native-gguf-verdict \
  --parallel 4 --target scamguard-gguf-tokenize scamguard-gguf-c-abi-smoke \
  scamguard-unicode-nfc-test
uv run --offline --no-project --with cmake ctest \
  --test-dir build/native-gguf-verdict --output-on-failure
.venv/bin/python scripts/check_gguf_tokenizer_parity.py \
  --probe build/native-gguf-verdict/scamguard-gguf-tokenize \
  --model artifacts/gguf/Qwen3.5-0.8B-Q4_0.gguf \
  --model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
  --reference published \
  --runner build/native-gguf-verdict/scamguard-gguf-verdict \
  --report reports/runs/qwen35-08b-q4-protocol5-reproduction.json
```

Existing report paths are refused. `--reference historical` remains the default and intentionally
fails against the corrected native tokenizer. Use `--reference published` only for the explicit
prospective contract, not to erase historical incompatibility.

The original ScamGuard quality/under-20-ms/SOTA/publication goal is still incomplete. This change
removes the native NFC blocker and establishes a reproducible starting contract for a new,
prospectively frozen experiment; it does not promote any rejected model.
