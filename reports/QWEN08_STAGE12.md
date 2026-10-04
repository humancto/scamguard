# Qwen3.5-0.8B Stage 12 preregistration

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
