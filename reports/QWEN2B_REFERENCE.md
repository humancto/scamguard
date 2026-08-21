# Qwen3.5-2B schema-v6 reference

The pinned `Qwen/Qwen3.5-2B` base at revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` was trained for exactly one epoch with a
language-tower-only rank-16 LoRA adapter on all 10,111 frozen schema-v6 training examples. The
adapter contains 16,819,200 trainable parameters and occupies 87,345,480 bytes including its
tokenizer and run metadata.

Temperature `9.1981973779` and scam threshold `0.2816018693` were fit on the 2,634-row development
split. The threshold was then frozen for every result below.

| Slice | SCAM / SAFE denominator | Recall | False-positive rate |
|---|---:|---:|---:|
| Development | 514 / 2,008 | 100.00% | 1.29% |
| Untouched test | 587 / 1,746 | 100.00% | 4.52% |
| Financial OOD | 261 / 172 | 93.87% | 25.00% |
| WSPR OOD | 409 / 0 | 100.00% | not measurable |
| Forum validation | 1,000 / 25 | 99.90% | 64.00% |
| Forum OOD | 2,000 / 100 | 99.85% | 75.00% |
| Materialized forum OOD | 1,862 / 78 | 99.41% | 66.67% |
| Adversarial | 160 / 160 | 100.00% | 22.50% |

The candidate passes test recall and all core-category recall gates. It fails the test FPR gate
(`79/1,746`, 95% CI 3.65–5.60%) and the three-class macro-F1 stretch gate. Raw argmax predictions
collapse almost entirely to `UNCERTAIN`, producing a test macro-F1 of `0.01697`; thresholded scam
probabilities remain highly sensitive but do not separate source-style SAFE controls reliably.
The forum and financial OOD results show that this is a specificity/source-style shortcut, not a
redaction-placeholder artifact.

Against the pinned public DeBERTa reference on the identical 2,333 binary test rows, 2B improves
recall from 8.18% to 100.00% and binary accuracy from 75.91% to 96.61%, but worsens FPR from 1.32%
to 4.52%. The recall improvement is +91.82 percentage points (paired-bootstrap 95% CI +89.59 to
+93.94); the FPR regression is +3.21 points (95% CI +2.09 to +4.30). Exact McNemar counts are 562
candidate-only correct versus 79 reference-only correct (`p=1.04e-90`). This is a major recall
gain, not a release-gate pass.

Apple-MPS BF16 reference scoring measured 354.79 ms median and 579.94 ms p95 over 50 true
single-message calls, including prompt formatting, tensor construction, and model forward. The
model reported 4,493,760,448 parameter bytes; sampled MPS driver allocation peaked at a lower-bound
6,183,665,664 bytes. This reference path fails the 20 ms product gate before quantization. A
separate throughput probe retained batch one because batches 2–8 shifted raw likelihoods by up to
0.104 despite matching the 12 probe argmax labels.

The failed FPR and macro-F1 gates trigger the predeclared Qwen3.5-4B escalation on the unchanged
frozen corpus. Any later data redesign informed by these OOD outcomes must retire the affected OOD
sets and introduce newly sourced untouched replacements.

- Report SHA-256: `3cb00091b8361b84d8ffa5041352d1a0df432c486ae471e898ae5db3adaa1c76`
- Text-free 11,753-row ledger SHA-256:
  `43e96a3ad34d9992b38d9a5a81a94c725d291009be864f3075faabc2c2bbcc50`
- Adapter weights SHA-256:
  `c96bffc7cc9fcc39d907eead4a50477bbe09e1ecea6a20514d6bdb7fd6c05342`
- Calibration SHA-256:
  `5d2f4c5798f0c5b867fcd7025b07cf478289070e0a1d299b73002a996ef23508`
- Paired comparison SHA-256:
  `f417cd8db06cbbd17bb9599a00abf29318c553351d1fe0f6513a4e445a8a3b71`
- Evaluation batch benchmark SHA-256:
  `1d4fe70ff732fae4a812ed18d8722df46d11576c4df69e0160f0559b6169d3a6`

