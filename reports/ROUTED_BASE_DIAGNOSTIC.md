# Encoder + Qwen3.5-0.8B base routed diagnostic

## Decision

The routed evaluation contract is ready, but the untouched Qwen3.5-0.8B base is rejected as the
specialist. This is a control result, not a prediction about the schema-24 LoRA candidate.

The evaluator joined the schema-23 ModernBERT router and Qwen base ledgers one-to-one on
`(split, id)`: 2,634 development examples and 2,374 untouched test examples. It rejected duplicate
keys and required `truth`, `source`, `source_language`, and `category` to match. Neither input nor
the routed output ledger contains message text.

The encoder ledger uses the production scanner's calibration contract: its historical
`safe_threshold = 0.20` is a maximum SAFE-path risk, so SAFE requires `p_safe >= 0.80` and
`p_scam < 0.20`. Qwen calibration instead stores a direct minimum SAFE probability. Runtime
backends now expose those normalized boundaries separately, and calibration files record the
threshold semantics explicitly. This report supersedes the earlier pre-fix local diagnostic.

## Frozen routing result

Routing candidates were selected using development data only, under a 25% escalation cap and 2%
SAFE false-positive cap. Every calibrated router `UNCERTAIN` is mandatory specialist traffic;
additional rows may be selected by a frozen top-two probability-margin boundary.

The selected boundary is `margin_max = -1.0`, meaning that no extra confidence-band traffic was
worth sending to the base model. Development escalation was 5.505%. On test, only the 111 mandatory
router-uncertain rows were escalated, or 4.676%.

| Test policy | Scam recall | SAFE FPR | Three-way macro F1 | Escalation |
|---|---:|---:|---:|---:|
| Encoder only | 100.00% | 1.031% | 0.7519 | 0% |
| Qwen base only | 29.98% | 2.119% | 0.4306 | 100% |
| Frozen routed control | 100.00% | 1.317% | 0.7869 | 4.676% |

The routed control preserves recall and improves macro F1 by 0.0350 versus the calibrated encoder,
but it adds five SAFE false positives, increases FPR by 0.286 percentage points, and remains far
below the 0.94 macro-F1 gate. The Qwen base is therefore rejected and cannot justify a production
hybrid quality claim. The trained schema-24 adapter must produce a new specialist ledger and win
this exact paired comparison.

## Latency discipline

This diagnostic does not publish a routed p95. Aggregate encoder and Qwen percentiles cannot be
combined into a request percentile, and the existing Q4 exact-scorer p95 is a percentile of run
means rather than per-request samples. After the specialist and policy are frozen, the complete
router-to-specialist path must be timed per request on desktop and a physical phone.

## Reproduction

```bash
make encoder-schema23-ledger
make qwen-08b-base-routed-diagnostic

# Final trained specialist:
make routed-eval \
  SPECIALIST_PREDICTIONS=reports/runs/qwen35-08b-schema24-full.predictions.jsonl \
  ROUTED_REPORT=reports/runs/sg-modernbert-schema23-qwen08-schema24-routed.json
```

The first command performs checkpoint-only encoder scoring; it does not train or modify the
checkpoint. `routed-eval` requires both ledgers to exist and never reads benchmark message text.

## Evidence identities

- Encoder ledger SHA-256:
  `4eac0c156041c2611ae8276ac1af6cd8f1348687b7b1825aec1465b0dff227b0`
- Qwen base ledger SHA-256:
  `78386a94e0996848a24637694172210ad2d1c378bb12ff52c9c9ed52c7a33ded`
- Routed diagnostic report SHA-256:
  `db3fbee9a42e9b9ebd55cbc4c3047c2d3d27d2ef2b78bce7b8c8f6bbc728c164`
- Routed test ledger SHA-256:
  `9ae8ebc9aaa23bebf82adcd24fa9d87c2489695d2605dbc9d1c1f70079b60d2d`

The raw ledgers and JSON report are ignored local artifacts. This tracked report freezes their
identities and interpretation. Human audit, full LoRA training, merged/quantized parity, routed
end-to-end timing, physical-mobile measurement, and release authorization remain open gates.
