# Qwen3.5-4B dialogue selection diagnostic

Run date: 2026-08-21.

This experiment asks a narrow architecture question: does the larger historical Qwen3.5-4B LoRA
already understand the multi-turn scam/legitimate boundary that defeated the encoders? It does
better on scam dialogue than the rejected schema-v10 encoder, but it does not meet the release
gates and does not justify replacing the tiny fast path.

## Frozen subject

- Base: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Adapter SHA-256:
  `093f67a1903ea5738a851c44b6b0872a2eced96426226b7eea87cd5c639e1823`.
- Adapter size: 149,947,932 bytes.
- Calibration is unchanged from the historical schema-v6 development run: temperature
  `9.9999373017`, scam threshold `0.34428313997`, SAFE threshold `0.28497121964`.
- Scoring uses length-normalized teacher-forced verdict likelihood. Neither selection slice fits
  weights, temperature, or thresholds.

## Results

| Selection slice | Rows | Scam recall | SAFE FPR | Binary counts |
|---|---:|---:|---:|---|
| BothBosu telephone dialogue | 294 (141 SCAM, 153 SAFE) | 92.91% (131/141) | 24.84% (38/153) | TP 131, FN 10, TN 115, FP 38 |
| Taskmaster transactional dialogue | 450 SAFE | not estimable | 4.44% (20/450) | TN 430, FP 20 |

The 95% Wilson intervals are 87.44–96.10% for BothBosu recall, 18.66–32.24% for BothBosu FPR,
and 2.90–6.76% for Taskmaster FPR. Both FPR intervals exceed the 2% release boundary, and BothBosu
recall is below 97%.

Sampled current allocated memory peaked at 9,243,752,960 bytes for BothBosu and 9,232,176,128 bytes
for Taskmaster on Apple MPS. This is an instantaneous lower-bound measurement, not a quantized
mobile package estimate.

## Interpretation

Model scale helps retain scam sensitivity relative to the 149M schema-v10 dialogue ablation, but it
does not provide the calibrated trust-boundary discrimination ScamGuard needs. Qwen3.5-4B is
rejected as the sole detector under this historical adapter. It can remain a teacher or an
uncertainty-band explainer after the fast candidate is stable. These results cannot be labelled a
schema-v12 head-to-head because the adapter and calibration predate schema v12.

The 1,049-row BothBosu OOD partition and 1,820-row MOZ primary test were not scored.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| BothBosu selection data | `c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0` |
| BothBosu result | `3b92474776532e6fd34d24b01bfd12bda10d9dc0c56543b43f18b8687e8a3e1c` |
| BothBosu text-free prediction ledger | `9ca02b798a5753cedf155408bfb229c9748bbc932c99d8a6dc715ae9b92b6ef0` |
| Taskmaster selection data | `539b81a06328b2914407565c1bb7fac54a486333cea45a0853b4e2160df79760` |
| Taskmaster result | `5d6fc2022fb0ceccef0c4032420f50488e0af4f9ee5dc149a404c6788ea7dbf3` |
| Taskmaster text-free prediction ledger | `703e1fb137bca87017833d2e4db60479cb65e6474524a4098378099a42e82630` |
