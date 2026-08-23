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
| Qwen base only | 30.32% | 2.520% | 0.4295 | 100% |
| Frozen routed control | 100.00% | 1.375% | 0.7730 | 4.676% |

The routed control preserves recall and improves macro F1 by 0.0210 versus the calibrated encoder,
but it adds six SAFE false positives, increases FPR by 0.344 percentage points, and remains far
below the 0.94 macro-F1 gate. The Qwen base is therefore rejected and cannot justify a production
hybrid quality claim. The trained schema-24 adapter must produce a new specialist ledger and win
this exact paired comparison.

## Latency discipline

The follow-up persistent runtime control now publishes a routed p95 from 7,122 actual request
traces; it is documented in `reports/ROUTED_BASE_RUNTIME.md`. Aggregate encoder and Qwen
percentiles still cannot be combined into a request percentile. With matching batch-one,
three-candidate, bucket-64 scoring, the route now has exact probability and decision parity and
measures 17.26 ms full-distribution p95. Its 190.79 ms p99 and 216.66 ms escalated p95 still expose
a severe specialist tail. A physical-phone result remains required for any mobile claim.

## Reproduction

```bash
make encoder-schema23-ledger
make qwen-08b-base-routed-diagnostic
make qwen-08b-base-routed-runtime

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
  `ccf383d7713a3d5c9347036a3d7d62b410df2e58996e9751ad2a3ffd8f45c2b4`
- Routed diagnostic report SHA-256:
  `11ac4c30f4ce1122681d96f5348c0815ab42cd769464af5868118da971ce80b2`
- Routed test ledger SHA-256:
  `ff2267abee409bac0231c29ce60c2eaf3b5ad64221f6fc372c406b96642fb959`

The raw ledgers and JSON report are ignored local artifacts. This tracked report freezes their
identities and interpretation. Human audit, full LoRA training, merged/quantized parity, routed
end-to-end timing, physical-mobile measurement, and release authorization remain open gates.
