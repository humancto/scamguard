# Schema-v9 fast encoder

Status: **rejected as the sole detector; retained as a fast-router candidate.**

This run fine-tunes the exact `answerdotai/ModernBERT-base` revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8` (149M parameters) on ScamBench schema v9. The
training objective combines class-weighted three-class cross entropy with an auxiliary
`SCAM`-versus-`SAFE or UNCERTAIN` boundary loss. Checkpoints are selected by development recall at
the 2% maximum false-positive rate, rather than by aggregate F1.

The development-selected threshold is `0.0839591`; it is applied unchanged to every other split.
The best checkpoint was step 1,480 (epoch 2 of 4), with development recall `85.80%` at `1.64%`
false-positive rate after post-training temperature calibration.

## Results

| Split | Scam recall | False-positive rate | Macro F1 (argmax) | Decision |
|---|---:|---:|---:|---|
| Development | 85.80% | 1.64% | 0.726 | recall failed |
| Regression test | 99.83% | 6.30% | 0.787 | FPR failed |
| Financial OOD | 96.15% | 63.16% | 0.473 | recall and FPR failed |
| WSPR OOD | 99.76% | N/A (no SAFE rows) | 0.634 | recall passed; FPR not measurable |
| Forum validation | 99.00% | 20.00% | 0.724 | FPR failed; only 25 SAFE controls |
| Forum OOD | 99.25% | 18.00% | 0.769 | FPR failed; 100 SAFE controls |
| Adversarial | 98.75% | 6.88% | 0.543 | FPR failed |
| Azerbaijani diagnostic | 93.63% | 5.23% | 0.379 | recall and FPR failed |
| Chichewa diagnostic | not run | not run | not run | added after this checkpoint report |
| BothBosu dialogue selection | 99.29% | 90.20% | 0.287 | FPR failed |
| Taskmaster SAFE selection | N/A | 95.78% | 0.285 | FPR failed |

The regression test clears aggregate and every core category recall gate, but that does not rescue
the model: 110 of 1,746 SAFE regression examples cross the frozen scam threshold. The model also
does not generalize evenly. All 72 development `IDENTITY_IMPERSONATION` messages in the held-out
`identity_case_callback` family are missed, despite perfect regression recall on different identity
families. This is evidence of family/template shortcut learning, not evidence that the category is
solved.

## Runtime and artifact

On the local Apple-silicon MPS environment, batch-one model-forward latency after tokenization is
`9.32 ms` median and `16.67 ms` p95 over 250 regression messages. This passes the desktop fast-path
stretch target, but it is not end-to-end SDK latency and is not a physical-mobile measurement.

- Inference artifact: 602,034,281 bytes (574 MiB, unquantized)
- `model.safetensors` SHA-256:
  `75568fdafe974db93ffb4eb6fee395be8d6648ad9642cb31f609370f4aea534b`
- calibration SHA-256:
  `82826f72c4760839274990a7ec3dc81770122668e028f2b0836ba133ef684b85`
- full local run report SHA-256:
  `1e4066bce2a1fc148ca03501a34909840688f664e1efb45fbf4054aeaa631cad`

The detailed JSON remains local under `reports/runs/sg-modernbert-schema9-safety.json`; large model
weights and generated reports are ignored by Git.

## 395M scaling result

ModernBERT-large revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` was trained for the same
four epochs. Epoch 2 was selected. Scaling to 395M did not repair the held-out
`identity_case_callback` family: development recall remained 85.80%, including 0/72 identity
examples, though development FPR fell to 0.05%. Regression recall was 99.83% at 0.23% FPR.

External generalization remained below release quality: 91.92% recall / 63.16% FPR on financial
OOD, 97.50% / 8.00% on forum validation, 98.35% / 12.00% on forum OOD, 90.69% / 3.44% on
Azerbaijani, and 40.63% / 3.59% on Chichewa. On the dialogue selection slices it produced 90.78%
recall / 54.90% FPR on BothBosu and 20.22% FPR on Taskmaster SAFE.

The unquantized artifact is 1,586,947,269 bytes. Batch-one model-forward after tokenization measured
13.35 ms median and 26.78 ms p95, failing the desktop 20 ms stretch gate. Immutable identities:

- model SHA-256: `dd7eeccba53caf96166ce455ee59328f7ceec62b87db314497b90be7c7bfe33e`;
- calibration SHA-256: `1ef663e12e859a4ee345d2344c5bedac6d7d3351ad527f99461a36a22ecbfa33`;
- full report SHA-256: `1c5258aea1b6712e1bf7a8d561671c19c60a29eef002f5d1ab5c8d7ddce0f25a`.

## Decision

Neither schema-v9 encoder is eligible as the sole detector, and 395M is rejected as the mobile
product model because its quality gains do not justify its size or latency. The subsequent
schema-v10 dialogue correction repaired Taskmaster false positives but overfit source format and is
also rejected. Schema v11 is the controlled follow-up; development, regression, and sealed OOD rows
remain untouched. The 395M checkpoint remains only a historical capacity control.
