# Encoder + Qwen3.5-0.8B base persistent runtime control

## Decision

The untouched Qwen3.5-0.8B base remains rejected as the routed specialist. With both models
persistently loaded on Apple Silicon Metal, the encoder fast path clears 20 ms at p95, but the
complete routed distribution does not. Exact product-runtime decisions also differ from the frozen
quality ledger on ten of 2,374 test IDs. Similar aggregate quality is not a substitute for exact
decision parity.

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
| Full routed distribution | 7,122 | 17.07 ms | 8.41 ms | 36.28 ms | 187.72 ms | 315.43 ms |
| Encoder-only fast path | 6,789 | 9.27 ms | 8.34 ms | 14.85 ms | 24.93 ms | 85.60 ms |
| Escalated path | 333 | 176.13 ms | 165.63 ms | 235.89 ms | 271.52 ms | 315.43 ms |
| Qwen specialist component | 333 | 166.12 ms | 155.90 ms | 222.84 ms | 252.68 ms | 297.02 ms |

The full-distribution p95 fails the at-most-20-ms desktop gate. The escalated-path p95 also fails
the under-50-ms laptop boundary. Process peak RSS was 1,254,113,280 bytes. These macOS/MPS BF16
measurements neither establish GGUF performance nor substitute for a physical-phone run.

## Scoring-contract parity

The frozen specialist quality ledger was generated with 16 messages per accelerator batch and
three verdict candidates per message: 48 candidate sequences per forward pass. Product runtime is
one message and three candidate sequences per forward pass. Both paths now share the same float64
temperature-softmax arithmetic, but accelerator batch shape still changes BF16 logits slightly.

The maximum specialist probability difference was 0.0030255. No probability crossed the disclosed
0.005 diagnostic tolerance, yet ten specialist decisions and ten final routed decisions changed;
router decisions and route selection matched exactly. The transitions on the first repetition
were three `SAFE -> UNCERTAIN`, two `SCAM -> UNCERTAIN`, three `UNCERTAIN -> SAFE`, and two
`UNCERTAIN -> SCAM`. Exact decision parity therefore fails.

The runtime trace still measures 100% scam recall, 1.2600% SAFE FPR, and 0.7871 three-way macro F1.
Relative to the frozen batch-16 ledger, those aggregate deltas are 0 recall, -0.0573 percentage
points FPR, and +0.00019 macro F1. Those small aggregate changes do not waive per-example parity or
the existing 0.94 macro-F1 gate.

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
- Frozen routed report SHA-256: `db3fbee9a42e9b9ebd55cbc4c3047c2d3d27d2ef2b78bce7b8c8f6bbc728c164`
- Specialist score-cache metadata SHA-256: `5d7e573739b2adb4cde8bdee686aa1e33553025b8c59f905848e14f0c3cfd41d`
- Runtime JSON SHA-256: `6382dd32679983cd7ab1be7df8c934636b78d1638de24f2144d921f90b25c01f`
- Text-free trace ledger SHA-256: `8193a95c4d30a2651925a24b59a389cd31f64511e4df00cba9118af16a89457d`

The generated JSON and trace stay under ignored `reports/runs/`; this tracked report freezes their
identities and interpretation. The independent 635-row human audit remains 0/635, so full LoRA
training and any Hugging Face publication remain blocked by design.
