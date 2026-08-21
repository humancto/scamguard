# ModernBERT schema-v13 dose and policy experiment

Run date: 2026-08-21. Model-only decision: **rejected**. Policy status: **promising open-set
candidate; not release-ready**.

Schema v13 reduces schema v12's counterfactual dose from 64 to 16 rows per family while holding the
base model, seed, optimization, calibration, transform, and evaluation artifacts fixed. It then
evaluates a separately versioned deterministic policy over text-free prediction ledgers.

## Frozen model

| Field | Value |
|---|---|
| model | ModernBERT-base, 149M parameters |
| data | schema-v13 dose-16: 14,062 train / 2,634 dev / 2,374 unchanged regression |
| selected checkpoint | epoch 2 of 3 by development-only safety recall |
| calibrated temperature | `3.3447986819` |
| frozen scam threshold | `0.2717509866` |
| artifact | 602,034,174 bytes, FP32 |
| model-forward latency | 8.31 ms median / 13.48 ms p95, batch one on MPS |
| tokenizer-to-probability latency | 8.80 ms median / 14.01 ms p95 |

The deterministic signal extraction plus policy layer separately measures 0.09 ms median / 0.11 ms
p95 over all 2,634 development messages. These measurements are from the development Mac, not a
physical phone.

## Model-only result

| Slice | Scam recall | SAFE FPR | Macro F1 | Decision |
|---|---:|---:|---:|---|
| development | 95.91% | 1.94% | 0.7303 | fail: recall and macro F1 |
| unchanged regression | 99.32% | 4.18% | 0.8817 | fail: FPR and macro F1 |
| financial OOD | 89.62% | 53.80% | 0.4808 | fail |
| WSPR positive-heavy OOD | 99.76% | not estimable | 0.6352 | diagnostic |
| forum validation | 99.00% | 20.00% | 0.8015 | fail |
| forum OOD | 99.35% | 18.00% | 0.7848 | diagnostic |
| materialized forum OOD | 94.68% | 14.10% | 0.6647 | diagnostic |
| adversarial | 100.00% | 2.50% | 0.6573 | near miss |
| Azerbaijani OOD | 91.67% | 4.29% | 0.5638 | fail |
| Chichewa OOD | 59.05% | 7.46% | 0.4754 | fail |
| BothBosu dialogue selection | 51.06% | 18.30% | 0.3032 | fail |
| Taskmaster SAFE selection | not applicable | 0.00% | 0.3333 | pass |

The smaller dose partially repairs schema v12's catastrophic identity-family regression: 53/72
held development messages are detected instead of 0/72. It does not reduce unchanged-regression
false positives: both schemas v12 and v13 raise 73 alarms, though the family mix shifts. Schema v13
is therefore rejected as a sole detector.

## Trusted-channel policy v1

Two narrow rules were frozen after the model error audit:

1. blocking independent verification while requesting credentials forces `SCAM`;
2. explicitly directing the user to an already-known contact, an independently opened official
   app, or the number on a physical card can force `SAFE` only when no link, credential, OTP,
   unusual-payment, urgency, secrecy, remote-access, or other high-risk signal is present.

The policy does not inspect source IDs, labels, model probabilities, or template-family IDs at
runtime. It is deterministic, versioned as `trusted-channel-v1`, and reports model-only and
policy-adjusted scores separately.

| Slice | Model recall / FPR | Policy recall / FPR | Rule-label audit |
|---|---:|---:|---|
| development | 95.91% / 1.94% | **99.61% / 0.10%** | 72 SCAM risk overrides; 396 SAFE trust overrides |
| unchanged regression | 99.32% / 4.18% | **99.32% / 0.11%** | 396 SAFE trust overrides |
| adversarial | 100.00% / 2.50% | **100.00% / 1.25%** | 13 SAFE trust overrides |
| materialized forum OOD | 94.68% / 14.10% | unchanged | no rule fires |
| BothBosu dialogue selection | 51.06% / 18.30% | unchanged | one SCAM rule match was already detected |

Across 18,729 open rows in the coverage audit, every trusted-channel SAFE override lands on a SAFE
label and every blocked-verification SCAM override lands on a SCAM label. This is encouraging, but
the rules were designed using already-open development/regression errors. They are candidate-policy
evidence, not independent proof. The unchanged-regression result is now post-hoc design evidence,
and neither sealed source may be opened while dialogue selection still fails.

The policy improves three-class macro F1 only to 0.7709 on development and 0.8959 on regression, so
it does not clear the macro-F1 stretch target. More importantly, it does not solve realistic
multi-turn dialogue or multilingual OOD performance.

## Next decision

- Keep schema v13 plus `trusted-channel-v1` as the current fast-path research candidate.
- Do not run the 1,049-row BothBosu OOD or 1,820-row MOZ primary test yet.
- Prioritize licensed, realistic dialogue. TeleAntiFraud-28k is the strongest newly identified
  candidate, but its Hugging Face repository is gated and requires authenticated access before its
  4,000/400 binary split can be inspected.
- Reject KorCCVi for current use because its transcript rights stay with third parties and its
  source is perfectly correlated with the class label.
- The subsequent deployment-feasibility experiment exported this rejected neural candidate without
  opening a sealed source. Dynamic FP32 ONNX preserved every open frozen verdict at 13.92 ms CPU
  p95; naive dynamic INT8 was rejected for quality shifts. This does not promote the underlying
  model. See [`ONNX_SCHEMA13.md`](ONNX_SCHEMA13.md).

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| complete model run | `f41b62a4f9406aa9002bbd712a6a01ee3879f8a13d3523d22e3a1b4d47619fe5` |
| model weights | `31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99` |
| calibration | `6aa357171cd21b3af036c0c0e37b2b08306810a26556673f2d4650e3cb56204c` |
| policy implementation | `2c50408bbea043ffdcb1219fd17272923ba73eed848b934e9e03a042859739b7` |
| development policy report | `b62749856478de7ba16bcff0deddbd32ae87262e4d87d96eba10c2d45b06bcca` |
| regression policy report | `b5bb4965a799c089ff633d09954c1781c1594e7b10c8240518f82de3b02fb043` |
| adversarial policy report | `f2aee8f02b9c0f1bbf696067603e6ae9a637874c9cf5fa968f35697104e1d154` |
| materialized-forum policy report | `38034ff044d959bd4208c346fc57dee9c5d6c0a0a3082d3f2b7955d0a911e269` |
| text-free open coverage audit | `a61cc86ddb4c38f951a553ad3ce29788497c5adfbe2e4007758fb7309f0c1fdb` |

Prediction-sealed artifact state is unchanged.
