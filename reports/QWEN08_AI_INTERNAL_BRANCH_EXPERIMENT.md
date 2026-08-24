# Qwen3.5-0.8B schema-24 AI-internal experiment

## Decision

The first trained 0.8B specialist is promising but rejected. It clears the binary safety contract
on the unchanged primary regression set and all six core scam-category recall gates, but it passes
only 29 of 39 complete release gates. Quantization and Hugging Face publication remain unauthorized.

This is an AI-internal, non-release experiment. The 635-row blind human audit remains incomplete;
the assistant review cannot substitute for an independent reviewer.

## Frozen identities

- Base: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Training: 23,435 train and 2,634 development examples, one epoch, effective batch size 16.
- Adapter: 41.3 MB; SHA-256
  `d0cb140ba70849cc4462012518bd52d713f475f5c0c30fa9fb75b1947f25111e`.
- Scorer: `qwen-verdict-branch-token-v1`, frozen before branch-score test metrics were read.
- Temperature: 1.023658, fit on development only.
- SCAM threshold: 0.0787215, the highest development threshold satisfying recall at least 97% and
  FPR at most 2%.
- SAFE threshold: 0.500983, fit on development after the SCAM policy was frozen.
- Prediction ledger SHA-256:
  `2454903dde6c93a0fcf1c00d0aa08f2e19581441f122ed8eb13ea3b3a6361c16`.

## Measured result

| Slice | Scam recall | SAFE FPR | Calibrated macro F1 | Decision |
|---|---:|---:|---:|---|
| Development | 97.08% | 0.694% | 0.7399 | binary gate passes |
| Primary regression test | 99.66% | 0.115% | 0.7407 | binary gate passes; macro stretch fails |
| MultiDoGO complete calls | n/a | 5.692% | n/a | fails overall and every domain gate |
| BothBosu prior-open dialogue | 65.96% | 3.268% | n/a | fails both gates |
| Publisher annotation dev/test | n/a | 0% / 0% | n/a | all domain gates pass |
| Taskmaster | n/a | 0.222% | n/a | passes |

The primary test has 585 true positives, two false negatives, two false positives, and 1,744 true
negatives at the frozen SCAM threshold. Its post-temperature multiclass ECE is 0.02051, unscaled
multiclass Brier score is 0.06910, and NLL is 0.10656. The BF16 adapter path measures 92.58 ms
median and 96.15 ms p95 on the local MPS benchmark at product batch size one. That is a reference
quality measurement, not the final quantized mobile latency.

## Scoring correction

The historical evaluator length-normalized the complete spellings of `SAFE`, `UNCERTAIN`, and
`SCAM`. Because the labels tokenize to different lengths, that rule rewarded the longer predictable
suffix of `UNCERTAIN` and selected `UNCERTAIN` for every development example. That report is kept
as rejected historical evidence.

The corrected scorer performs one model forward pass and compares the first token at which the
three verdict strings diverge. Its configuration and the conservative development threshold rule
were committed before branch-score test metrics were inspected. Raw argmax remains diagnostic;
release decisions use the calibrated frozen thresholds.

## Failure diagnosis and next experiment

The publisher-annotated turn-level MultiDoGO slices have zero false positives, while complete
MultiDoGO calls reach 5.69%. The licensed training partition supplied short highest-risk agent turns
for the relevant schema-23 families, not enough complete service-call windows. The next frozen
curriculum therefore adds only publisher-training complete calls, verifies family disjointness from
the held call split, and replays existing SCAM, UNCERTAIN, long-call, and dialogue controls. No
primary-test or BothBosu row enters fitting.

Because this training decision follows observed regression results, the already opened primary test,
MultiDoGO validation, and BothBosu selection results become regression evidence for the continuation.
The 1,820-row `primary_test_v8` remains unopened until the continuation adapter, scorer, and threshold
policy are frozen.

## Release boundary

Do not merge, quantize, package, or publish this adapter. A later candidate must pass every frozen
quality slice, post-quantization parity, desktop/mobile latency and memory checks, provenance and
license review, the independent human audit, and the Hugging Face release checker.
