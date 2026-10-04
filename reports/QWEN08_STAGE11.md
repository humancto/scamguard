# Qwen3.5-0.8B Stage 11 result

## Result

**Reject Stage 11 before the full regression. Keep Stage 7 as the current reference.**

The one-adapter interpolation improved development macro F1 and phone recall, but failed three of
the eight frozen open gates. In particular, factor-space interpolation did not reproduce the
score-space diagnostic: PPoNE scam recall remained at Stage 10's 10/12 rather than Stage 7's 11/12,
PPoNE macro F1 regressed below Stage 7, and PPoNE UNCERTAIN recall returned to 1/21. The full
regression and every sealed test remain unopened.

| Metric | Stage 7 | Stage 11 | Frozen requirement | Result |
|---|---:|---:|---:|---|
| Dev scam recall | 97.08% | 97.08% | at least 97% | pass |
| Dev SAFE FPR | 0.25% | 0.20% | at most 2% | pass |
| Dev macro F1 | 0.7856 | 0.8064 | no regression | pass |
| Phone scam recall | 78.16% (136/174) | 79.31% (138/174) | no regression | pass |
| Phone ordinary-SAFE FPR | 17.11% (26/152) | 17.11% (26/152) | no regression | pass |
| PPoNE scam recall | 91.67% (11/12) | 83.33% (10/12) | no regression | **fail** |
| PPoNE macro F1 | 0.2398 | 0.2323 | at least +0.03 | **fail** |
| PPoNE UNCERTAIN recall | 4.76% (1/21) | 4.76% (1/21) | at least 3/21 | **fail** |

- Materialized adapter SHA-256:
  `201e9473a65796667be787b0751d14e93109e8ecfd35e02568f1d01f7fe40cd7`.
- Interpolation manifest SHA-256:
  `7ccbbb5095b2aade810a9c020f62441ab074723e23de733cfb4d88fbac19bd4c`.
- Development report SHA-256:
  `df1a1c10b81538c7ef5c220d71435a7fc7d4fa3e3c2283ad9b6f5eae44a0ce99`.
- Open selection report SHA-256:
  `6926bcb4306f56f27416661d1cef92c7f7d785d50533dda5223c74d7669163f2`.
- Promotion-gate report SHA-256:
  `114161156b79662467d2e78e3ee483fdb1f40caa6bc732658ec8fa55c8cb1103`.

## Decision boundary

Stage 11 tests a single-adapter checkpoint interpolation between Stage 7 and Stage 10. It is not a
two-adapter ensemble and adds no inference-time model. The fixed interpolation weight is 0.5 toward
Stage 10, with Stage 7 as the other 0.5.

This is explicitly a post-hoc hypothesis after repeated inspection of the development, PPoNE
validation, and phone validation sets. Those sets are useful for engineering selection but are not
fresh confirmation. PPoNE test, publisher-phone test, and primary test v8 remain sealed.

## Why this experiment

Stage 10 improved seven of eight frozen open gates but lost one of twelve PPoNE SCAM examples that
Stage 7 detected. A text-free Stage 7/Stage 10 score-path diagnostic showed that the 0.5 arithmetic
position retained 11/12 PPoNE SCAM recall, improved PPoNE macro F1 from 0.2398 to 0.2737, and raised
phone recall from 136/174 to 138/174. This diagnostic cannot itself ship because it evaluates two
adapters. It only motivates testing whether one factor-aligned LoRA checkpoint interpolation keeps
the same useful geometry.

The parent adapters share an identical LoRA contract, and Stage 10 was trained by continuing from
Stage 7. That makes factor-space interpolation a defensible early-stopping/retention hypothesis,
though not a mathematical guarantee of score interpolation.

## Frozen identities and policy

- Stage 7 adapter:
  `14f1d2bf121e76fa158cea722416994ec9cbfbc545956d24b9693b7808441357`.
- Stage 10 adapter:
  `3eb2974d7db70d2f859a10d1876a9f25e290bbe487abae27f837dfc949b59cf0`.
- Phone score-path diagnostic:
  `2a87fe8aac928436adab09e85717053d525267fafaad389cd2063d70addb4a9a`.
- PPoNE score-path diagnostic:
  `1e616c8efbcd7c1a8d7e96e9ce6583762f4fc32e924a16c5f6d14b2aad18d543`.
- Interpolation method: `linear_lora_weight_space_v1`; right/Stage 10 weight: `0.5`.
- Frozen experiment config:
  `bf87174085254bb76b8f6ca601562fc6e10729ecb8a92fb02e9149510f2948b7`.
- Runtime adapter count: one.

Stage 11 must calibrate thresholds from development only, then pass all eight frozen Stage 7
non-regression and PPoNE/phone promotion gates. A single failure rejects it before the full
regression. Quantization, packaging, physical-device timing, source tests, independent human review,
and Hugging Face publication remain prohibited until the complete downstream gate chain passes.
