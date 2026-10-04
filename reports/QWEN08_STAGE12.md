# Qwen3.5-0.8B Stage 12 result

## Result

**Reject Stage 12 before the full regression. Keep Stage 7 as the current reference and stop the
interpolation line of experiments.**

Exact delta interpolation recovered Stage 7's 11/12 PPoNE scam recall and increased phone scam
recall to 140/174. It did not recover the intended abstention behavior: PPoNE macro F1 improved by
only 0.0062 instead of the required 0.03, and PPoNE UNCERTAIN recall stayed at 1/21 instead of 3/21.
It also added three ordinary-phone SAFE alarms, reaching 29/152 versus Stage 7's 26/152. Five of
eight gates passed; all three failures are binding.

| Metric | Stage 7 | Stage 12 | Frozen requirement | Result |
|---|---:|---:|---:|---|
| Dev scam recall | 97.08% | 97.08% | at least 97% | pass |
| Dev SAFE FPR | 0.25% | 0.25% | at most 2% | pass |
| Dev macro F1 | 0.7856 | 0.7880 | no regression | pass |
| Phone scam recall | 78.16% (136/174) | 80.46% (140/174) | no regression | pass |
| Phone ordinary-SAFE FPR | 17.11% (26/152) | 19.08% (29/152) | no regression | **fail** |
| PPoNE scam recall | 91.67% (11/12) | 91.67% (11/12) | no regression | pass |
| PPoNE macro F1 | 0.2398 | 0.2460 | at least +0.03 | **fail** |
| PPoNE UNCERTAIN recall | 4.76% (1/21) | 4.76% (1/21) | at least 3/21 | **fail** |

- Materialized adapter SHA-256:
  `d2647c4b34d956ac6017cc3c070c167f00b0f269a8ef4ee152f6ff5a0a1b3f7c`.
- Adapter size: 86,640,085 bytes.
- Interpolation manifest SHA-256:
  `5e82c271813015888540e304f7dbf2163314e3642c4062144556d8afee1d2ecb`.
- Development report SHA-256:
  `62703faa30429d46be1020e09a5b34aeef38550682cc787e7ebcf3531c3559b2`.
- Open selection report SHA-256:
  `67b61bec86f3028b8b20e773defa9ee8d7db3dd278a2fc8b8a3fa15896653956`.
- Promotion-gate report SHA-256:
  `4edbd2b74754a366bea436b42b0ef6094b404b7aba04dc71159aa713aabd0710`.

The exact-delta construction worked as designed; the quality hypothesis did not. Further weights
would be direct tuning to repeatedly opened PPoNE and phone validation rows. The next experiment
must use new rights-clear training evidence and a new independently held development slice, not
another interpolation search.

## Decision boundary

Stage 12 is the final interpolation experiment. It combines the actual Stage 7 and Stage 10 LoRA
weight deltas at fixed 0.5 weight, represented exactly as one rank-32 LoRA adapter. It does not run
two models or two adapters at inference time.

Stage 11 directly averaged the parent A and B factor tensors. Because LoRA applies their product,
that introduced cross terms and did not preserve the intended delta mixture. Stage 12 instead
concatenates the parent A factors and weighted B factors. With unchanged LoRA scaling, the resulting
product is exactly `0.5 * B7*A7 + 0.5 * B10*A10` for every adapted module. A unit test verifies this
identity numerically.

## Research limitations

The 0.5 hypothesis is post-hoc and informed by development, PPoNE validation, and phone validation.
Those sources are no longer independent confirmation. The value of Stage 12 is architectural: it
tests the single-adapter analogue of an encouraging score path with a mathematically correct delta
construction. PPoNE test, publisher-phone test, and primary test v8 remain sealed.

The rank-32 adapter is expected to remain below 90 MB and adds no base-model parameters. It may be
roughly twice the adapter compute of rank 16, but adapter work is small relative to the 0.8B base;
latency claims remain prohibited until a candidate passes quality gates and is measured on device.

## Frozen policy

- Method: `linear_lora_delta_concat_v1`.
- Frozen experiment config:
  `ed31ead9e43f881aa067c6e5e6bf2329d7133de2a8e8aa63bb1462bb438f0802`.
- Parents: Stage 7 and Stage 10, each rank 16, fixed weights 0.5 and 0.5.
- Output: one rank-32 adapter with doubled alpha so parent and output scaling are identical.
- Development-only threshold calibration: at least 97% scam recall and at most 2% SAFE FPR.
- Promotion: every existing development, quality-stratified phone, and PPoNE gate must pass.
- Failure: reject before the full regression; do not inspect source tests or sealed primary data.
- Success: run the full frozen regression, then proceed to quantization and physical-device work only
  if every downstream gate passes.

Hugging Face publication remains unauthorized.
