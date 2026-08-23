# Qwen3.5-0.8B training and Hugging Face release

The correct ScamGuard training source is the official
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) Transformers checkpoint at revision
`2fc06364715b967f1860aea9cf38778875588b17`. Train a LoRA adapter against those safetensors, merge
the selected adapter, and only then convert and quantize it. The
[`ggml-org/Qwen3.5-0.8B-GGUF`](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) repository is an
Apache-2.0 deployment conversion of that base—not the object to fine-tune.

The upstream Q4_0 control is now hash-verified at 563,036,064 bytes and benchmarked through native
arm64 llama.cpp. Its exact verdict scorer misses both the strict 20 ms fast path and, narrowly, the
formal 50 ms laptop boundary; this freezes Qwen as a routed specialist. Those control measurements
do not establish trained ScamGuard accuracy, post-merge memory, or mobile performance. The release
will test `Q4_K_M` and `Q5_K_M` and publish the smallest quantization that retains the frozen
model's decisions and gates. Fast-path, specialist, and routed end-to-end latency remain separate
published measurements.

## Current evidence, not a release candidate

The repository has a pinned 0.8B LoRA configuration and a native MPS training path. The untouched
base has now been scored on all 6,713 open core and publisher-held examples: it reaches only 29.98%
core test scam recall and fails every core gate, although its publisher-test SAFE FPR is 0.08%.
That control is documented in `reports/QWEN08_BASE_SCHEMA24_BASELINE.md`. Historical adapter runs
remain five-row smoke tests; they prove that loading, adapter training, and scoring execute but are
explicitly rejected by the publication validator. The full adapted 0.8B challenger starts only
after schema v24 data and independent label audits are frozen. The separate upstream Q4 runtime
control and routed deployment decision are documented in `reports/QWEN08_Q4_RUNTIME_FLOOR.md`.

## Release sequence

1. Finish the schema-v24 real-dialogue annotation audit, rebuild immutable family-disjoint data,
   and record every source license, hash, admission decision, and PII transformation.
2. Create a new immutable 0.8B experiment config with the schema-v24 hashes. Run the full LoRA
   experiment against the pinned official checkpoint; do not overwrite the historical schema-v6
   config or smoke reports.

   ```bash
   make schema24-annotated-hard-negatives
   make schema24-audit
   # An independent reviewer uses the local blind-review UI. It edits only the ignored CSV.
   make schema24-audit-review
   make schema24-audit-check
   make qwen-08b-full-data
   make qwen-08b-full-token-audit
   make qwen-08b-full-freeze
   # Review and commit the generated immutable config, then start the measured run.
   make qwen-08b-full
   make qwen-08b-full-eval
   make qwen-08b-full-gates
   ```

   The review UI binds to `127.0.0.1`, loads no remote assets, hides project labels, source labels,
   fraud categories, and model outputs, and derives agreement only after a reviewer saves their own
   verdict. It checks the audit and dataset manifest binding before every write and atomically
   persists progress, so the review can be stopped and resumed. Do not have a person who authored
   the labels perform the independent review.

   The freeze step requires schema version 24, a completed independent human-label audit, non-empty
   annotation train/dev/test strata, zero publisher dev/test rows in fitting, complete verbatim
   evidence, and zero examples over the frozen 640-token limit. The 640-token ceiling preserves
   decisive long-dialogue actions; it is not the under-20-ms fast-path latency budget. The freeze
   step refuses to overwrite an existing
   config. The trainer independently rechecks the config, model revision, hyperparameters, LoRA
   module allowlist, data/report hashes, counts, and output path before loading model weights.
3. Freeze calibration, routing, and thresholds on development/selection data. Pass every internal
   and independent external-selection gate before opening any prediction-sealed test.

   ```bash
   make encoder-schema23-ledger
   make routed-eval \
     SPECIALIST_PREDICTIONS=reports/runs/qwen35-08b-schema24-full.predictions.jsonl \
     ROUTED_REPORT=reports/runs/sg-modernbert-schema23-qwen08-schema24-routed.json
   ```

   The evaluator joins text-free ledgers exactly, chooses the uncertainty margin on development
   only, and publishes component baselines plus escalation. The untouched Qwen base fails this
   role; `reports/ROUTED_BASE_DIAGNOSTIC.md` is the frozen negative control.
4. Merge the adapter with `training/merge_qwen_adapter.py`. Its equivalence audit must show that
   adapter and merged verdict scores remain within the configured tolerance.
5. Convert the merged directory with `scripts/export_gguf.sh`. Evaluate at least `Q4_K_M` and
   `Q5_K_M`; the exporter currently emits `Q4_K_M`, so create the Q5 artifact with the same pinned
   llama.cpp quantizer when Q4 loses a gate.
6. Rerun the exact frozen quality protocol on the selected GGUF. Quantization is rejected if any
   safety decision or required gate regresses outside the published tolerance.
7. Measure tokenizer-to-verdict p50/p95, peak memory, and artifact bytes on the reference desktop
   and a physical phone. If Qwen is routed, also publish escalation rate and full routed p50/p95.
8. Complete the label, multilingual-claim, redistribution, PII, and secrets audits. Build a model
   card that includes failures, dataset provenance tiers, frozen thresholds, hardware, runtime,
   limitations, hashes, the Apache-2.0 license, and upstream attribution.
9. Copy `configs/huggingface-release-qwen35-08b.template.json` to an ignored release workspace,
   fill it with the final artifact/report paths, exact sizes and SHA-256 hashes, and change
   `publication_status` to `approved` only after every fact is evidenced.
10. Run the fail-closed authorization check:

    ```bash
    make huggingface-release-check \
      HF_RELEASE_MANIFEST=/absolute/path/to/qwen35-08b-release.json
    ```

    Upload only when it returns `"publication_authorized": true`.

## Hugging Face package boundary

Use two public repositories after authorization:

- `humancto/scamguard-qwen3.5-0.8b` for the merged Transformers/safetensors model, tokenizer,
  calibration, model card, license, and compact evaluation reports;
- `humancto/scamguard-qwen3.5-0.8b-GGUF` for the selected GGUF quantization, tokenizer/config,
  hashes, llama.cpp revision, usage example, and post-quantization/mobile reports.

Do not upload raw or processed training rows as part of either model release. Synthetic examples
may become a separate dataset only after their provenance and privacy audit. Licensed research
artifacts still require a source-by-source redistribution review. Direct Reddit content remains
excluded from training and release because public visibility is not a model-training license.

Repository creation and upload are intentionally manual after the validator succeeds: authenticate
with the Hugging Face CLI, create both model repositories, and upload from clean staging
directories containing only the manifest-authorized files. Record the resulting commit IDs back in
the release report. This prevents credentials, local corpora, prediction ledgers, or an unselected
quantization from being swept into an upload.
