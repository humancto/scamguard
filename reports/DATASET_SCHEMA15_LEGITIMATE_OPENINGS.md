# Schema v15 legitimate-call opening correction

Experiment date: 2026-08-21. Decision: **reject checkpoint and training increment**.

## Hypothesis

The open AppTek SAFE-call benchmark localized every schema-v13 and schema-v14 false positive to
early service-call windows. Schema v15 tested whether a small matched SAFE increment could remove
that shortcut without compromising scam recall or the unchanged regression set.

The experiment changed data only. It retained ModernBERT-base, revision, seed, optimizer,
three-epoch schedule, 256-token window, speaker-neutral transform, loss, calibration procedure, and
all prior evaluation files.

## Increment

The deterministic generator emits 1,024 original four-turn SAFE dialogues:

- 16 legitimate service scenarios;
- four structures: inbound call, requested callback, transfer, and outbound requested update;
- 64 scenario/structure families;
- no copied AppTek or regression text;
- no explicit safety phrases such as “never ask,” “official app,” “gift card,” or “security code.”

The predeclared dose selects four rows from every family: 256 fitting rows total, 16 per scenario
and 64 per structure. Schema v15 therefore has 14,479 train rows and 28,158 unique processed rows.
Development, regression, and every inherited diagnostic are byte-identical to schema v14. AppTek
contributes zero fitting and threshold rows; its 1,396-window OOD partition remains unopened.

## Result

All numbers use each checkpoint's development-frozen temperature and SCAM threshold.

| Candidate | Regression recall | Regression SAFE FPR | BothBosu recall / SAFE FPR | AppTek SAFE FPR | YouTube-call recall |
|---|---:|---:|---:|---:|---:|
| schema 13 dose-16 | 99.32% | 4.18% | 51.06% / 18.30% | 8.91% (31/348) | 34.29% |
| schema 14 real dialogue | 99.83% | 8.48% | 100% / 73.20% | 22.13% (77/348) | 100% |
| schema 15 legitimate openings | 99.83% | **18.84%** | 69.50% / 35.29% | **15.52%** (54/348) | 100% |

Schema v15 improves AppTek FPR relative to schema v14 but remains substantially worse than schema
v13 and far above the 2% product cap. Its intended early-window slice records 53/174 false
positives; recent windows record 1/174. More importantly, the unchanged regression set records 329
SAFE false positives, including 324 in the synthetic-v5 source. The run's regression macro F1 is
0.6189 and development macro F1 is 0.7270. The unchanged BothBosu selection slice also fails both
binary gates.

Taskmaster remains 0/450 false positives and the positive-only YouTube-call slice remains 70/70
detected. Those successes cannot offset the much larger regression and balanced-dialogue failures.

The checkpoint is fast on the development Mac—12.12 ms end-to-end p95 over 250 batch-one examples,
with a 602,034,257-byte FP32 training artifact—but speed does not rescue a failed safety model. No
Core ML, ONNX, GGUF, mobile, or sealed-source evaluation is authorized for this checkpoint.

## Interpretation

The result rejects the simple “add more SAFE call openings” hypothesis. The correction learned
some intended structure but did not produce a stable scam boundary; it shifted errors into existing
synthetic SAFE families and still generalized poorly to unseen legitimate openings. Increasing the
same dose would be an unjustified post-hoc escalation.

The next candidate should change the learning formulation, not merely add rows. A defensible next
experiment is a paired contrastive or distillation objective built from diverse, independently
generated scam/legitimate call-opening pairs, with source-balanced sampling and regression-logit
retention against schema v13. AppTek may select among those candidates, but its text remains
evaluation-only and its OOD partition sealed.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| schema-v15 manifest | `aa24e34b48c58d1fbdeb1854c6dd261828891e764302381ebf5c59f1786ac9bf` |
| schema-v15 train | `4d0f03a946d407f81a354911a8da68f1c2b8e84c2fc00943da454317f3057214` |
| generator output | `2ca1f1cfeb4f283d29e1652839f4277e1fe3ac94bc5d5f7fc1781dfaf148ba94` |
| generator manifest | `f97005bda95750ae0ea5b90518ec6a9b159c8b8a10fa9d9e44695190ee056a2c` |
| AppTek selection | `e0e7ad4de8d378061159df18a8fa6c39fedd69d0cdd0dffe3f72579158290b62` |
| AppTek sealed OOD | `1e6d1176936324f073ca7dd5746bcce5b7849d5be5c4973782479e595f3c3ade` |
| checkpoint model | `770445b488702815576846fdafc0f4de0dda48ff3cfc1a667929b2415db6715b` |
| checkpoint calibration | `1ee133c75f11d19897864d7da7d13b6d03bc1cb5790c86fa63794bff9476e58f` |
| complete model report | `0823cc1681e28eb898de77549d621e7738829d435099e08a7d0b7f2064397e6d` |
| AppTek open report | `96c9580d6352b450ad500cfdc07c1e8b5641496ee15ff9b573177e491dd41258` |
| YouTube-call open report | `6e09da4862c0af56dc0bf036df4715847cf8e2b290ec99c9f594cad70602c555` |

Reproduce with `make schema15-legitimate-openings`,
`make encoder-schema15-legitimate-openings`, and `make apptek-eval-schema15`.
