# Qwen3.5-0.8B Stage 11 preregistration

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
