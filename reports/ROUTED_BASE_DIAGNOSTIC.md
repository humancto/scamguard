# Encoder + Qwen3.5-0.8B base routed diagnostic

## Decision

The routed evaluation contract is ready, but the untouched Qwen3.5-0.8B base is rejected as the
specialist. This is a control result, not a prediction about the schema-24 LoRA candidate.

The evaluator joined the schema-23 ModernBERT router and Qwen base ledgers one-to-one on
`(split, id)`: 2,634 development examples and 2,374 untouched test examples. It rejected duplicate
keys and required `truth`, `source`, `source_language`, and `category` to match. Neither input nor
the routed output ledger contains message text.

## Frozen routing result

Routing candidates were selected using development data only, under a 25% escalation cap and 2%
SAFE false-positive cap. Every calibrated router `UNCERTAIN` is mandatory specialist traffic;
additional rows may be selected by a frozen top-two probability-margin boundary.

The selected boundary is `margin_max = -1.0`, meaning that no extra confidence-band traffic was
worth sending to the base model. Development escalation was 1.063%. On test, only the 18 mandatory
router-uncertain rows were escalated, or 0.758%.

| Test policy | Scam recall | SAFE FPR | Three-way macro F1 | Escalation |
|---|---:|---:|---:|---:|
| Encoder only | 100.00% | 1.031% | 0.8581 | 0% |
| Qwen base only | 29.98% | 2.119% | 0.4306 | 100% |
| Frozen routed control | 100.00% | 1.031% | 0.8140 | 0.758% |

The routed control preserves the binary boundary only because the encoder already owns it. It
reduces macro F1 by 0.0441 versus the calibrated encoder. The Qwen base therefore earns no routing
role and cannot be used to justify a hybrid quality claim. The trained schema-24 adapter must
produce a new specialist ledger and win this exact paired comparison.

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
  `00ca1bbd314c8908dde0a4017d59c115589fb5a40a5508eef128b1350e0bacbe`
- Qwen base ledger SHA-256:
  `78386a94e0996848a24637694172210ad2d1c378bb12ff52c9c9ed52c7a33ded`
- Routed diagnostic report SHA-256:
  `87ed65578854ae0b3b847d4cb320ce973af351c8a474299120b7d140044fcb95`
- Routed test ledger SHA-256:
  `61886a3bce9f840ff710701b4d85d8aa21f1805991e79ec72bc0f223b3e1ba99`

The raw ledgers and JSON report are ignored local artifacts. This tracked report freezes their
identities and interpretation. Human audit, full LoRA training, merged/quantized parity, routed
end-to-end timing, physical-mobile measurement, and release authorization remain open gates.
