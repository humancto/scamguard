# Physical-mobile benchmark protocol

ScamGuard treats mobile execution as measured evidence, not an inference from parameter count,
GGUF size, a simulator, or desktop Apple Silicon. Hugging Face authorization requires one physical
iOS run and one physical Android run over the same text-free sample identity set.

## Measurement contract

Each platform run must:

- use the exact selected GGUF and frozen calibration;
- bind the platform runtime package by SHA-256;
- run offline on a physical device with `simulator: false`;
- record manufacturer, phone form factor, model, hardware identifier, architecture, OS/version,
  runtime revision, accelerator, threads, and thermal state before and after the run;
- warm the runtime with at least five requests;
- measure at least 100 unique messages and at least 100 requests using a monotonic clock;
- time the complete local tokenization-to-verdict path, not model kernels alone;
- retain one raw, text-free record for every `(sample ID, repetition)` pair;
- record the runtime and frozen reference verdict for every request;
- record prefix-cache reuse on every routed specialist request; and
- record startup latency and peak memory independently of request latency.

The iOS and Android traces must contain the same selected IDs. The verifier derives stable opaque
`sgm-…` IDs from the quantized ledger, then checks every device reference verdict against that
ledger. A trace row may contain only an opaque sample ID, repetition, elapsed milliseconds,
runtime/reference verdicts, routing state, and prefix-cache state—never the message text. The
report binds the full quantized prediction-ledger SHA-256 and a separately recomputed hash of the
sampled reference verdicts.

## Recomputed evidence

`scripts/verify_mobile_benchmark.py` rejects copied summaries. It recomputes each device's p50,
p95, p99, maximum, request count, selected-ID hash, sampled-reference hash, and exact verdict
parity from the raw samples. The release-level summary is deliberately conservative: each latency
percentile and peak memory value is the worse of the two physical devices. It is then required to
match `runtime.mobile` in the Hugging Face release manifest.

Validate a completed report with:

```bash
make mobile-benchmark-check \
  MOBILE_BENCHMARK_REPORT=/absolute/path/mobile-benchmark.json \
  MOBILE_GGUF_MODEL=/absolute/path/scamguard.gguf \
  MOBILE_RUNTIME_CALIBRATION=/absolute/path/scamguard_calibration.json \
  MOBILE_QUANTIZED_QUALITY=/absolute/path/quantized-quality.json \
  MOBILE_PREDICTION_LEDGER=/absolute/path/quantized-quality.predictions.jsonl \
  IOS_RUNTIME_PACKAGE=/absolute/path/scamguard-ios-runtime.zip \
  ANDROID_RUNTIME_PACKAGE=/absolute/path/scamguard-android-runtime.zip
```

Passing this verifier proves internal consistency and artifact binding. It cannot cryptographically
prove that a device description is truthful, so the final release review must retain the original
device logs and package build receipts. No mobile claim is allowed from a simulator, desktop host,
upstream base-model control, summary-only JSON, or fewer than both required platforms.
