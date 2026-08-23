# Reproducibility

All commands assume a native arm64 Python 3.11 interpreter on Apple Silicon. The lockfile pins the
Python environment; experiment JSON files pin model, framework, dataset, and export revisions.

## Data and tests

```bash
uv sync --extra train --extra dev --extra qwen
make data
make audit
make test
make lint
make forum-learning-curve
make qwen-token-audit
```

The pinned DeBERTa reference runs on CPU (`make reference`) because the current Apple-MPS
DeBERTa path aborts in an MPS matrix-multiplication datatype assertion. Its report labels the
device and batched-forward latency explicitly; those timings are not a product latency claim.
The public scaler was serialized by scikit-learn 1.8, so the harness validates its 23 stored
means/scales and applies the exact StandardScaler formula directly. The checkpoint also mixes an
FP16 encoder with FP32 custom heads; the harness normalizes the complete scorer to FP32 on CPU.
The measured result and artifact hashes are in `reports/DEBERTA_REFERENCE.md`.

Downloaded data is hash-checked before extraction. Generated and processed rows are ignored by Git
and rebuilt locally. Compare their hashes with `data/processed/manifest.json` and the immutable
experiment configuration before training.

## Schema-v12 encoder

```bash
# 149M rejected counterfactual control
make encoder-schema12
```

This command trains for three epochs on the 14,446-row schema-v12 split while preserving the byte-identical
schema-v9 development and regression rows. It applies the versioned `speaker-neutral-v1` transform
to eligible multi-turn inputs. Post-training evaluation also scores the 450-row Taskmaster and
294-row BothBosu selection slices with development-only calibration. It does not open the 1,049-row
BothBosu OOD partition or the sealed MOZ primary test. The only schema-v12 data increment is a
balanced 512-row, eight-family counterfactual repair curriculum derived from schema-v11 error
categories without copying evaluation text.

Schema v12 is a frozen rejected control, not the current promotion candidate. Its failure analysis
is in `reports/ENCODER_SCHEMA12.md`.

## Schema-v13 dose-16 experiment

```bash
make schema13-dose16
make encoder-schema13-dose16
```

These targets materialize and train the isolated 14,062-row dose ablation under
`data/experiments/schema13-dose16/`. They do not overwrite canonical schema-v12 processed data.
Validation requires schema version 13 explicitly and confirms that the unchanged development and
regression artifact hashes still match schema v9/v12. The model-only result is rejected; the
post-hoc `trusted-channel-v1` result and all caveats are recorded in
`reports/ENCODER_SCHEMA13.md`.

## Schema-v13 ONNX deployment pack

```bash
uv sync --extra train --extra dev --extra neural --extra onnx
make encoder-onnx-export
make encoder-onnx-eval-fp32
make encoder-onnx-eval-int8
```

The exporter creates fixed-hash FP32 and dynamic-INT8 graphs plus tokenizer, calibration, and
manifest files under `artifacts/onnx/schema13-dynamic-pack/`. It refuses to overwrite an existing
pack. Both evaluators read only the already-open development and unchanged-regression splits. The
FP32 graph is the accepted fidelity reference; the first INT8 graph is retained as a measured
quality rejection. Neither command opens the sealed dialogue or MOZ sources.

## Schema-v13 Core ML deployment packs

```bash
uv sync --all-extras
make encoder-coreml-export
make encoder-coreml-eval
```

Core ML compilation and prediction require native macOS compiler access. The exporter uses a fixed
128-token iOS 17 ML Program, refuses to overwrite an existing pack, verifies fixed-mask and rotary
rewrites against PyTorch, and binds the package plus runtime assets in its manifest. The default
target builds the quality-preserving FP32 artifact under
`artifacts/coreml/schema13-seq128-fp32/`. FP16 is a separate compression experiment and is rejected
by the recorded full-set parity test. Full results and hashes are in `reports/COREML_SCHEMA13.md`.
Neither path reads sealed sources.

The gated dialogue-source workflow is separate:

```bash
# First accept the publisher's Hugging Face gate and authenticate locally.
make teleantifraud-fetch
make teleantifraud-audit
```

The fetcher pins the exact repository revision and downloads only the 60.3 kB binary metadata
archive, manifest, and dataset card—not the 12.7 GB audio archive or SFT set. The audit emits only
counts, schema, duplication, privacy-risk, and provenance-field evidence. It admits zero rows until
manual Chinese-language, privacy, and construction-provenance review is complete.

## Historical schema-v9 encoders

```bash
# 149M fast-path candidate
make encoder

# 395M sub-1B quality/teacher candidate
make encoder-large
```

Both commands select checkpoints using development scam recall under the 2% FPR cap and then apply
one frozen, temperature-calibrated development threshold to regression and OOD slices. The local
JSON report is written under `reports/runs/`. The current 149M result and hashes are summarized in
`reports/ENCODER_SCHEMA9.md`.

## Qwen candidates

```bash
make qwen-2b
make qwen-batch-benchmark
make qwen-eval
make qwen-generation

uv run python scripts/create_error_audit.py \
  --predictions reports/runs/qwen35-2b-schema6-lora.predictions.jsonl

uv run python benchmarks/compare_paired.py \
  --candidate reports/runs/qwen35-2b-schema6-lora.predictions.jsonl \
  --reference reports/runs/deberta-v022.predictions.jsonl

# Run only after the 2B quality result is known.
make qwen-08b

# Required escalation when the 2B result misses any frozen gate.
make qwen-4b
make qwen-4b-core-eval
make qwen-4b-eval

# Current schema-v9 4B training configuration, after the exact tokenizer audit passes.
make qwen-token-audit
make qwen-4b-schema9
```

The core 4B command scores `dev` and `test` first. The full command resumes from the same
integrity-keyed cache and adds every OOD/adversarial slice without rescoring completed rows.

The trainer asserts that every trainable LoRA tensor belongs to the language tower. Evaluation
fits temperature and the scam threshold on development SAFE/SCAM rows only, then freezes both for
test, financial OOD, WSPR OOD, forum OOD, realistic-placeholder forum derivatives, and adversarial
slices. `benchmarks/forum_learning_curve.py` is the only consumer allowed to use
`forum_validation`; its source deliberately never opens a test or OOD file.
The 2B target trains for exactly one epoch and therefore skips Trainer's redundant end-of-epoch
development-loss pass: there is only one checkpoint to choose from. `make qwen-eval` performs the
required development calibration and every frozen-slice evaluation after the adapter is saved.
The batch benchmark selects the fastest evaluation-only batch whose raw scores remain within 0.02
of batch-one scoring and whose argmax labels all match. The schema-v6 2B run selected batch one;
larger MPS batches shifted raw likelihoods by up to 0.104 on the deterministic probe. The evaluator
therefore uses batch one and commits an integrity-keyed raw-score cache after every completed split,
so an interrupted reference run resumes without accepting partial or stale scores. Product latency
is still timed separately with single messages after all slices are scored.
The evaluator also writes an ignored, text-free prediction ledger. The post-fit audit command joins
it back to the ignored local corpus and selects deterministic false-negative, false-positive, and
highest-loss samples for independent review; it must not be used to tune against the test set.
The paired comparison requires identical binary example IDs and publishes paired-bootstrap
confidence intervals plus an exact McNemar test against the pinned public DeBERTa reference.
The evaluator and `QwenVerdictBackend` call the same full-continuation tokenization helper. This is
required because independently tokenizing a JSON prompt and verdict can change BPE tokens at the
string boundary. Both paths request only the final candidate-suffix logits to avoid allocating a
full sequence-by-vocabulary output tensor.

The Apple-Silicon Make targets pass `--require-mps`. This is a fail-fast guard: a restricted
sandbox can expose an arm64 PyTorch build while hiding Metal, which otherwise makes a 2B run fall
back silently to float32 CPU training. Use a separate, explicitly documented CPU/CUDA command on
non-Apple hardware rather than removing the guard without recording the accelerator.

Create the schema-v23 router ledger and evaluate a specialist with the frozen development-only
routing policy:

```bash
make encoder-schema23-ledger
make routed-eval \
  SPECIALIST_PREDICTIONS=reports/runs/qwen35-08b-schema24-full.predictions.jsonl \
  ROUTED_REPORT=reports/runs/sg-modernbert-schema23-qwen08-schema24-routed.json
```

The encoder step is checkpoint-only evaluation. Both input ledgers and the routed test ledger omit
message text. `training/eval_routed.py` rejects duplicate or mismatched `(split, id)` joins and
never derives end-to-end p95 from aggregate component percentiles.

For the pinned untouched 0.8B negative control, measure the actual persistent route three times:

```bash
make qwen-08b-base-routed-runtime
```

This command first creates or verifies batch-one, three-candidate, bucket-64 base scores, then
verifies the frozen ledgers and specialist score-cache identity, keeps both models loaded, and
emits text-free request traces under `reports/runs/`. It requires exact route
and calibrated-verdict parity while separately disclosing probability drift, runtime quality,
fast/escalated/full p50/p95/p99/maximum, and process peak RSS. The tracked result is
`reports/ROUTED_BASE_RUNTIME.md`.

Score the family-collapsed multi-turn diagnostic with the already-frozen encoder calibration:

```bash
make encoder-dialogue-base
make encoder-dialogue-large
```

These commands score only the 294-row selection slice and do not refit temperature or thresholds.
Their reports include the untruncated token-length distribution and the fraction clipped by the
256-token short-message window, so a poor conversation result cannot be mistaken for an in-domain
SMS failure. The 1,049-row OOD partition remains prediction-sealed until the model and routing
policy are frozen.

Score the source-family-held legitimate Taskmaster dialogue slice:

```bash
make encoder-taskmaster-base
make encoder-taskmaster-large
```

These historical schema-v9 baselines use their original frozen thresholds. Taskmaster validation
cannot fit a threshold or update weights; the corresponding 1,800 fitting families exist only in
schema v10.

## Native arm64 GGUF toolchain

Use the exact llama.cpp revision and apply the small score-output patch before building:

```bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git ../llama.cpp
git -C ../llama.cpp checkout 521a64cd01979bb5b1a466152c576a9d809b068d
git -C ../llama.cpp apply "$PWD/scripts/llama_cpp_multiple_choice_scores.patch"

uv run --no-project --python /opt/homebrew/bin/python3.11 --with cmake \
  cmake -S ../llama.cpp -B ../llama.cpp/build-arm64 \
  -DGGML_METAL=ON -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64

uv run --no-project --python /opt/homebrew/bin/python3.11 --with cmake \
  cmake --build ../llama.cpp/build-arm64 --config Release --parallel 8 \
  --target llama-quantize llama-perplexity llama-cli
```

Merge, quantize, and rerun the calibrated benchmark:

```bash
# These targets export the selected 4B quality winner.
make qwen-merge
make qwen-gguf LLAMA_CPP_DIR=../llama.cpp PYTHON_BIN="$PWD/.venv/bin/python"
make qwen-gguf-eval LLAMA_CPP_DIR=../llama.cpp
```

The custom scorer changes no model math. It removes the generic multiple-choice helper's inserted
space so candidates exactly continue the ScamGuard JSON prefix, and prints the already-computed
length-normalized answer log-probabilities. `training/eval_gguf.py` applies the same frozen
temperature and threshold used by the reference checkpoint.

Build and rerun the pinned upstream 0.8B Q4 runtime controls independently of model training:

```bash
uv run --no-project --with cmake cmake \
  -S ../llama.cpp -B ../llama.cpp/build-scamguard-arm64 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
uv run --no-project --with cmake cmake \
  --build ../llama.cpp/build-scamguard-arm64 --config Release --parallel 8 \
  --target llama-bench llama-perplexity

make qwen-08b-base-gguf
make qwen-08b-base-gguf-benchmark LLAMA_CPP_DIR=../llama.cpp
make qwen-08b-base-gguf-verdict-benchmark LLAMA_CPP_DIR=../llama.cpp
```

The fetch step pins the Hugging Face repository revision, byte count, SHA-256, and Apache-2.0
license before the artifact is used. The generic benchmark records Metal and CPU kernel floors.
The exact scorer records contexts 256 and 640 separately; its reports contain IDs and hashes but no
message text. See `reports/QWEN08_Q4_RUNTIME_FLOOR.md` for the measured scope and limitations.

The exporter passes `--no-mtp`. Qwen3.5-4B's text config advertises one MTP layer, while the
selected Hugging Face checkpoint contains no MTP tensors; including that metadata produces a GGUF
that asks the runtime for a nonexistent final block. ScamGuard uses only the main 32-layer model.
