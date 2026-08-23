# Encoder + Qwen3.5-0.8B base persistent runtime control

## Decision

The untouched Qwen3.5-0.8B base remains rejected as the routed specialist. With both models
persistently loaded on Apple Silicon Metal, the encoder fast path and complete routed distribution
clear 20 ms at p95. Exact product-runtime probabilities, routes, and decisions match the frozen
quality ledger. The base is still rejected because its quality is poor and its specialist tail is
far beyond budget: full-distribution p99 is 190.79 ms and escalated p95 is 216.66 ms.

This is a negative control for the eventual schema-24 LoRA candidate, not a projection of its
quality or latency.

## Measured end-to-end distribution

The frozen development-selected policy escalates only calibrated encoder `UNCERTAIN` verdicts
(`margin_max = -1.0`). The test set contains 2,374 requests, repeated three times for 7,122 timed
requests. It escalates 111 unique requests, or 4.676%. Both BF16 models were loaded once and warmed
before timing. Timing starts at tokenizer entry and ends after the final calibrated verdict; it
excludes cold load, file I/O, evidence extraction, and SDK result construction.

| Path | Samples | Mean | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Full routed distribution | 7,122 | 16.74 ms | 8.60 ms | 17.26 ms | 190.79 ms | 238.98 ms |
| Encoder-only fast path | 6,789 | 8.83 ms | 8.55 ms | 10.71 ms | 13.58 ms | 42.84 ms |
| Escalated path | 333 | 177.92 ms | 169.73 ms | 216.66 ms | 233.26 ms | 238.98 ms |
| Qwen specialist component | 333 | 168.35 ms | 160.28 ms | 205.44 ms | 222.59 ms | 228.05 ms |

The full-distribution p95 passes the at-most-20-ms desktop gate because escalation remains below
5%, but the disclosed p99 shows the cost immediately above that percentile. The escalated-path p95
fails the under-50-ms laptop boundary. Process peak RSS was 1,134,280,704 bytes. These macOS/MPS BF16
measurements neither establish GGUF performance nor substitute for a physical-phone run.

## Scoring-contract parity

Frozen specialist quality and product runtime both use one message, three verdict candidates, and
a 64-token left-padding bucket per forward pass. The cache identity includes all three values, and
both paths share the same float64 temperature-softmax arithmetic.

The maximum specialist probability difference is exactly zero. All specialist verdicts, routes,
and final routed decisions match on every request across all three repetitions. The router's
maximum probability difference is 0.00000173, with no decision or route change.

The runtime trace measures 100% scam recall, 1.3746% SAFE FPR, and 0.7730 three-way macro F1,
exactly matching the frozen product-shaped ledger. The base remains far below the 0.94 macro-F1
gate despite its parity and full-distribution p95 passes.

## Reproduction

```bash
make qwen-08b-base-routed-runtime
```

The target requires Apple Metal, verifies the pinned router and specialist ledgers against the
frozen routed report, verifies the specialist score-cache identity, loads both models from local
artifacts, and emits a text-free per-request trace. The default is three repetitions. For a quick
diagnostic only, set `ROUTED_RUNTIME_REPETITIONS=1`.

The trained candidate must be calibrated and evaluated with a frozen scoring batch contract and
must repeat this product-batch-one parity test. It must then repeat the trace after merge,
quantization, and on a physical phone.

## Evidence identities

- Test corpus SHA-256: `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554`
- Selected-ID order SHA-256: `67d51198e0c621d2c977989c2800b678ad504501eaa7e5a83e9b9edeaaf42d33`
- Frozen routed report SHA-256: `11ac4c30f4ce1122681d96f5348c0815ab42cd769464af5868118da971ce80b2`
- Specialist score-cache metadata SHA-256: `90d13ae8c02349feffae7f0628c8fefec1cf213154eb6180de3b8d74051cbbfe`
- Runtime JSON SHA-256: `6827927909fc8258e936472a453559dcc1601f381da3603c35914d1b555c3d88`
- Text-free trace ledger SHA-256: `df7dd0ab01d22f4040f3118c2a6b56912094e092e5889acf6005f84620b718d6`

The generated JSON and trace stay under ignored `reports/runs/`; this tracked report freezes their
identities and interpretation. The independent 635-row human audit remains 0/635, so full LoRA
training and any Hugging Face publication remain blocked by design.
