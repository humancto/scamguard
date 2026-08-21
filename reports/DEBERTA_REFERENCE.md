# DeBERTa v0.2.2 reference — same-row rerun

The pinned public
[`notd5a/deberta-v3-malicious-sms-mms-detector-v0.2.2`](https://huggingface.co/notd5a/deberta-v3-malicious-sms-mms-detector-v0.2.2)
reference was rerun on the frozen privacy-normalized schema-v6 ScamBench rows. Its CC-BY-NC-4.0 license permits this
research comparison but does not make it a redistributable product dependency.

| Operating point | Development recall / FPR | Untouched test recall / FPR |
|---|---:|---:|
| Published threshold `0.7229` | 98.64% / 49.95% | 68.82% / 60.42% |
| ScamBench development threshold `0.9187091` | 11.48% / 1.89% | 8.18% / 1.32% |

The threshold is selected from development SAFE/SCAM rows only. The test result shows why the
model card's reported score cannot be treated as a head-to-head result: on these exact rows, the
published threshold predicts half or more of clean messages as scams, while satisfying the 2% FPR
ceiling collapses recall. At the published threshold it also measures 65.52% recall / 30.23% FPR
on financial OOD, 94.87% recall on positive-only WSPR, 81.20% / 72.00% on forum OOD, 96.29% /
87.18% on materialized forum variations, and 53.12% / 17.50% on adversarial derivatives. At the
ScamBench threshold those results fall to 8.43% / 0.00%, 3.18% recall, 12.20% / 6.00%, 15.95% /
10.26%, and 7.50% / 0.00%, respectively.

The model occupies 739,944,456 parameter bytes in FP32. An isolated batched CPU forward-only rerun
at batch 32 measured 10.39 ms median and 36.94 ms p95 per message. This excludes tokenization and is not a
batch-one product latency claim, so it fails the 20 ms fast-path evidence requirement as measured.

The published checkpoint mixes FP16 encoder tensors with FP32 custom-head tensors. The harness
normalizes the entire scorer to FP32 for CPU execution. It also validates the 23-feature public
scaler and applies its stored `(x - mean_) / scale_` formula directly because the artifact was
serialized with a different scikit-learn version. Neither repair changes weights, features, or
thresholds.

- Report SHA-256: `7657972be2337311b2c9f31471750260c7b8e4ab457212366d0734b9d8db1cd8`
- Text-free 10,057-row ledger SHA-256:
  `1f451c1e6659c3d0f91320305c0436cc4c77b92afd53387558337d4106d4b722`
- Pinned weight SHA-256: `0fdbcd779f2d6b10db0cef5cfb1656b65a6ded1529cf3dbfc5178b87e4029868`
