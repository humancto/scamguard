# Schema v16 continual-distillation ablation

Experiment date: 2026-08-21. Decision: **reject without export or sealed evaluation**.

## Question

Schema v14 learned the real scam-call positives but created a broad call-opening shortcut. Schema
v15 added matched legitimate openings, yet destabilized the unchanged regression set. Schema v16
tests a different learning formulation rather than adding another dose:

- initialize from the schema-v13 checkpoint;
- train for one epoch on the fixed 14,479-row schema-v15 corpus at `5e-6`;
- retain schema-v13 behavior on all 14,062 inherited rows with frozen-logit KL loss, weight 2 and
  temperature 2;
- apply square-root source balancing (`alpha=0.5`) so the 161 real scam-call and 256 synthetic
  legitimate-opening rows receive 8.83% of expected samples;
- leave the 417 new rows unanchored so their hard labels can correct the old blind spot.

The experiment was predeclared in
`configs/encoder-schema16-retention-alpha05-w2.json`. A verifier checks the initialization model,
every data and teacher hash, row counts, source counts, sampling probabilities, hyperparameters,
and sealed-source policy before training.

## Retention contract

The frozen teacher cache contains only a row ID and three logits. It contains no message text and
cannot act as a second dataset. Its manifest binds the ledger to the exact schema-v13 training file,
schema-v13 model, speaker-neutral transform, label order, and 256-token window. Cache creation is
idempotent and refuses partial or mutated artifacts.

Retention loss is training-only. Evaluation rows intentionally carry no teacher fields, keeping
all reported metrics on the ordinary supervised product contract. Unit tests verify both sides:
teacher-free evaluation succeeds, while a retention-training batch missing its anchor fields fails
closed.

## Frozen-threshold result

All rows below use each checkpoint's development-frozen temperature and SCAM threshold. The
schema-v13 column is the strongest compact control, not the rejected schema-v15 checkpoint.

| Open benchmark | Schema 13 | Schema 16 | Gate | Result |
|---|---:|---:|---:|---|
| development recall | 95.91% | **99.42%** (511/514) | >=97% | pass |
| development SAFE FPR | 1.94% | **1.89%** (38/2,008) | <=2% | pass |
| development macro F1 | 0.7303 | **0.7371** | >=0.90 stretch | fail |
| unchanged regression recall | 99.32% | **100%** (587/587) | >=97% | pass |
| unchanged regression SAFE FPR | 4.18% | **3.15%** (55/1,746) | <=2% | fail |
| BothBosu recall | 51.06% | **99.29%** (140/141) | >=97% | pass |
| BothBosu SAFE FPR | 18.30% | **69.93%** (107/153) | <=2% | fail |
| Taskmaster SAFE FPR | 0% | 0.22% (1/450) | <=2% | pass |
| AppTek SAFE FPR | 8.91% | **30.17%** (105/348) | <8.91% | fail |
| YouTube scam-call recall | 34.29% | **98.57%** (69/70) | >=97% | pass |
| desktop CPU end-to-end p95 | 14.01 ms | **32.31 ms** | <=20 ms | fail |

The AppTek result is sharply localized: 105/174 early openings are false alarms, while 0/174
recent windows are false alarms. This is not a threshold-only miss. The same candidate obtains
99.29% BothBosu scam recall by flagging 69.93% of its legitimate dialogue, so its apparent
dialogue-recall gain is unsafe.

The 602,034,360-byte value is the FP32 research checkpoint, not a deployment package. The CPU
latency run measures tokenizer, device transfer, model forward, and probability conversion for
batch one on the development Mac. It is not a phone measurement. Because quality gates fail, no
Core ML, ONNX, quantization, mobile benchmark, or sealed-source prediction is authorized.

## Conclusion and next dataset specification

Frozen-logit retention improves the broad regression set relative to schema v15 and preserves high
short-message recall. Source balancing also learns the positive call source. Neither mechanism
teaches the missing causal boundary: introductory call structure is still treated as scam evidence.

The next data increment must therefore be **paired, structure-matched minimal contrasts**, not more
independent positive and negative rows. Each family should hold topic, opening structure, speaking
style, length, and neutral business vocabulary constant. Only the risk mechanism should change:
untrusted channel, secrecy or urgency, credential request, remote access, off-platform payment, or
irreversible transfer. SAFE counterparts must avoid explicit anti-scam phrases. Families must be
split as units and independently reviewed before fitting.

AppTek is now selection evidence and may localize errors but contributes no text to fitting. Its
1,396-window OOD partition, the 80-window YouTube OOD partition, BothBosu OOD, and the 1,820-row MOZ
holdout remain sealed. A smaller student is premature until a teacher clears the balanced
legitimate-call gates.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| predeclared experiment config | `1ca78c91f6cbcfa635c9c724e1f28e13999d69d5e0fb970db0222f912316ef9c` |
| schema-v15 training manifest | `aa24e34b48c58d1fbdeb1854c6dd261828891e764302381ebf5c59f1786ac9bf` |
| schema-v15 training rows | `4d0f03a946d407f81a354911a8da68f1c2b8e84c2fc00943da454317f3057214` |
| schema-v13 teacher model | `31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99` |
| teacher manifest | `c8699f153567fd062c1fbd927dd9cdcf49013b0861448fefe467706e6dc5a36a` |
| text-free teacher ledger | `5a718418debb2897f1078e43724d2ee9a072bb9725bb66aff53d3fa191b591a3` |
| checkpoint model | `4a8b7d4c651f09122fd2986cd6758cd4103d7cff3a55b4b0a52eba7d94db5ced` |
| checkpoint calibration | `649d834809c404bc4caa7f01719e806de909e5130fcc110a96e15ef9198870ff` |
| complete model report | `341d58347cfa3ce3d1d5fd1da2e6c9e77e872719e8c5b7150b2c7d063d80c193` |
| AppTek open report | `75cec1a58a52006f6bd44d13e9f1fc2ea32bfce6f1af52cf13ead9fb4392df53` |
| YouTube-call open report | `013e52090fbf6896c6995a9d9a27841bc002f74bc4151a419c0da7207899a016` |

Reproduce with `make encoder-schema16-preflight`, `make encoder-schema16-retention`,
`make apptek-eval-schema16`, and `make youtube-eval-schema16`.
