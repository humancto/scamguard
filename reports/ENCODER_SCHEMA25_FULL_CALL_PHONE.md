# Encoder schema 25: full-call and phone-source result

## Decision

**Rejected. Do not distill, export, quantize, open a sealed benchmark, or publish this
checkpoint.**

Schema 25 tested whether complete licensed service calls plus the pinned MIT
`shakeleoatmeal/phone-scam-detection-synthetic` training partition could repair the schema-23
ModernBERT boundary. It improved the two main diagnostics but passed only **20 of 38** frozen gates.
The remaining false-positive rates are incompatible with a safety product: 16.96% on 896 held
MultiDoGO service calls, 9.15% on 153 prior-open BothBosu SAFE dialogues, and 77.65% on 170
publisher phone-synthetic SAFE examples.

The publisher phone validation split is a source-specific diagnostic, not SOTA evidence. Its
`variation_style` metadata perfectly predicts the label in the publisher artifact, so ScamGuard
does not expose that metadata to the model and does not treat in-source validation as independent
generalization.

## Frozen data and recipe

- Parent: privacy-normalized schema 24, excluding all 1,095 internally AI-reviewed rows because
  the independent human audit is incomplete.
- Training rows: 24,436, including 812 full MultiDoGO training calls and 1,190 phone-synthetic
  training rows. Phone validation (344 rows) remained evaluation-only; phone test (172 rows)
  remained prediction-sealed.
- Model: `answerdotai/ModernBERT-base`, 149M parameters, one epoch, batch 16, learning rate `1e-6`,
  verdict-retention weight 4, action loss weight 0.5, and action-row verdict weight 0.25.
- Processed manifest SHA-256:
  `6c1096f1017edb10b690fde8968d9de892d2bff8ae49b9183c5da15e0380ad57`.
- Frozen config SHA-256:
  `b35feeca4a1aa1715a26f0905f1aef09e1b0977116be69909e6ad6adbecddf65`.

## Result

| Gate slice | Required | Schema 23 | Schema 25 | Pass |
|---|---:|---:|---:|:---:|
| Development scam recall | >=97% | 99.61% | 99.61% | Yes |
| Development SAFE FPR | <=2% | 1.44% | 1.79% | Yes |
| Unchanged test scam recall | >=97% | 100.00% | 100.00% | Yes |
| Unchanged test SAFE FPR | <=2% | 1.03% | 1.20% | Yes |
| MultiDoGO full-call SAFE FPR | <=2% | 23.10% | 16.96% | **No** |
| Prior-open BothBosu scam recall | >=97% | 77.30% | 85.82% | **No** |
| Prior-open BothBosu SAFE FPR | <=2% | 13.73% | 9.15% | **No** |
| Phone-synthetic scam recall | >=90% | 91.38% pre-run baseline | 93.10% | Yes |
| Phone-synthetic SAFE FPR | <=10% | 89.41% pre-run baseline | 77.65% | **No** |

The candidate also failed the verified/unresolved participant-control contrasts, action exact-match
gates, and all six MultiDoGO domain FPR gates. Improvements therefore remain diagnostic evidence,
not a promotable checkpoint.

Training took 733.0 seconds on Apple-silicon MPS. The complete FP32 checkpoint is 602,057,514
bytes. Batch-one end-to-end PyTorch/MPS latency was 9.57 ms median and 15.39 ms p95 over 250
samples. That clears the desktop timing target but is not a physical-phone measurement.

## Evidence identities

- Model weights SHA-256:
  `b2882ab93099d9b9cdb3e0712d41b5bbf6cd7f2063686d6df380d4c6fb71a347`
- Calibration SHA-256:
  `53ddfe13d309a79ad873797189cdc7463ccf0d5dd3d5dfd2c8a6bb8aa9b3e005`
- Full run report SHA-256:
  `458c03465ad9fe29a43a07565f82cb441b83326fc0aa88200601a0acde6daca0`
- Gate report SHA-256:
  `f26e1e703205ada5dd88b67d9e9494a517c820b234b1e96627a2a40199246ead`

## Next frozen experiment

Qwen Stage 7 continues from the strongest Stage 3 Qwen3.5-0.8B adapter instead of spending more
compute on the rejected encoder boundary. Its immutable curriculum replays all 25,098 Stage 3
rows and adds 1,052 supported phone-training rows once. It excludes 36 scam rows without a
verbatim runtime evidence span and 102 rows longer than the frozen 640-token contract. Independent
SimHash checks found zero overlap with Stage 3 fitting data or held evaluation references.

The resulting 26,150-row curriculum has manifest SHA-256
`198be06ef6480f7226e550d9a99b048922f574d9f1eb6497f36e7c3ba880fe19`; the token audit has
SHA-256 `0b0a3c654d0822dc8e47a01be71fa4f66ef662454c0c03f9e161a7772fcc8691` and reports zero
examples above 640 tokens. The frozen Stage 7 config has SHA-256
`0df6e1d014d9a493b037065fdf655e76e063f0e3744c9a5212aa48150e422967`, one epoch, learning
rate `1e-6`, seed `20260906`, and no publication authority.
