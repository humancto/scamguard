# ModernBERT schema-v12 counterfactual ablation

Run date: 2026-08-21. Decision: **rejected**.

This run tests whether 512 balanced, training-only counterfactual messages can repair schema v11's
known-contact and official-channel false alarms without changing any evaluation row. It succeeds on
part of the regression problem and stays below the desktop latency target, but fails the safety
gate by erasing one complete held-out identity-scam family.

## Frozen candidate

| Field | Value |
|---|---|
| model | ModernBERT-base, 149M parameters |
| data schema | v12: 14,446 train / 2,634 dev / 2,374 unchanged regression |
| training | three epochs, speaker-neutral-v1 dialogue transform |
| calibrated temperature | `3.6192039273` |
| frozen scam threshold | `0.1198302284` |
| artifact | 602,034,248 bytes, FP32 |
| model-forward latency | 8.38 ms median / 13.72 ms p95, batch one on MPS |
| end-to-end latency | 8.86 ms median / 14.27 ms p95, tokenizer through probability |

Latency excludes SDK event handling and is not a physical-phone measurement.

## Main results

| Slice | Rows | Scam recall | SAFE FPR | Macro F1 | Gate result |
|---|---:|---:|---:|---:|---|
| development | 2,634 | 85.60% | 0.35% | 0.7563 | fail: recall and macro F1 |
| unchanged regression | 2,374 | 99.49% | 4.18% | 0.7842 | fail: FPR and macro F1 |
| financial OOD | source-defined | 80.77% | 37.43% | source-defined | fail |
| adversarial | source-defined | 98.13% | 0.63% | source-defined | diagnostic |
| Azerbaijani OOD | 4,327 | 92.65% | 5.84% | source-defined | diagnostic |
| Chichewa OOD | 677 | 58.73% | 11.88% | source-defined | fail |
| BothBosu selection | 294 | 61.70% | 34.64% | source-defined | fail |
| Taskmaster selection | 450 SAFE | not applicable | 0.00% | source-defined | pass |

The model clears the declared end-to-end desktop p95 target but does not clear the required quality
gates. No size or latency advantage can compensate for the held-out family failure.

## Failure localization

Text-free prediction ledgers were joined to frozen row metadata after evaluation. They contain IDs,
labels, family IDs, decisions, and probabilities, but no message content.

- 74 development SCAM rows are missed. Exactly 72 are the untouched
  `synthetic:identity_case_callback:v5` family; the other two are Mendeley rows.
- All 72 identity-family rows were recalled by the schema-v11 baseline, so the schema-v12 data
  increment caused a regression rather than exposing a pre-existing blind spot.
- 73 regression SAFE rows are false alarms, down from 202 under schema v11. Of these, 59 are
  `synthetic:family_transfer_verified:v5`, 12 are multilingual official-app alerts, and two are
  Mendeley rows.
- The 512-row increment is balanced overall, but it places 128 SAFE transfer/family-status examples
  and 128 scam counterparts into a single full retrain. The learned boundary generalized beyond the
  intended trust cues and suppressed callback-style impersonation scams.

This evidence rejects both "more synthetic data is automatically better" and "a larger generative
model will automatically fix dialogue." The next useful experiment is a controlled dose/coverage
ablation with fixed evaluation rows, not a larger undifferentiated corpus.

## Next experiment boundary

1. Treat schema v11 as the training baseline and schema v12 as a rejected ablation.
2. Build schema v13 variants that vary the new counterfactual dose and explicitly cover
   callback-style impersonation as a paired scam boundary without copying evaluation text.
3. Select only on development plus the already-open Taskmaster and BothBosu selection slices.
4. Require non-zero recall for every development scam family before aggregate metrics are
   considered; the intended gate remains at least 98% recall and at most 2% SAFE FPR.
5. Keep the 1,049-row BothBosu OOD partition and 1,820-row MOZ primary test prediction-sealed until a
   candidate and its policy are frozen.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| complete run report | `95758cc03034131f7165bd38ca21b3ac1b59aeb31e5bb7cb63a23d96488ca638` |
| model weights | `2d84ad8e38cbc648af60827cc4a75125265044516f27e339356b1332f900a36e` |
| calibration artifact | `3dd3b92540850d0ded90f72bbc6261052f304918a7497417db1e18068f4b9081` |
| development ledger report | `315063197e5a2512c7bfe555c4c932c3b92e8cb570cabbabd0cfdd4a43de6711` |
| development predictions | `e7aa60f081bda1972456315a4abcb69c3626215287ceb549574a3e7546b16209` |
| regression ledger report | `11195ec82bbcf1445ca340273819b9bc95299056b953d2c01bc3ea711ad21631` |
| regression predictions | `423487573c7eea2b16fc0b5d71b34def27e27a3c881c5f9af6fe7ec00af24060` |

The prediction-sealed BothBosu OOD and MOZ artifacts were not scored during this run.
