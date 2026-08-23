# Qwen3.5-0.8B schema-24 base baseline

## Decision

The untouched Qwen3.5-0.8B base is a useful initialization, not a scam detector. With thresholds
fit only on product-shaped development scoring, it reaches 30.32% scam recall, 2.52% SAFE
false-positive rate, and 0.4295 calibrated three-class macro F1 on the unchanged core test. It
fails every core quality gate and runs far above the 20 ms complete fast-path budget in BF16.

This result supersedes the earlier batch-16 quality ledger. Frozen quality and product inference
now both use one message, three verdict candidates, and a 64-token left-padding bucket per forward
pass. The old ledger was numerically different and produced an optimistic FPR/routed result.

The same frozen base has only 1/1,199 false positives on the publisher-held schema-24 test slice,
versus 101/1,199 for the rejected schema-23 encoder. This is promising language understanding, but
not a standalone success: it abstains as `UNCERTAIN` on 339/1,199 publisher test messages,
including 181/200 media examples. Fine-tuning must preserve the low false-positive behavior while
recovering scam recall, decisive SAFE classification, and action/evidence semantics.

## Frozen setup

- Base: `Qwen/Qwen3.5-0.8B`
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Adapter: none
- Precision/runtime: official BF16 Transformers checkpoint on Apple MPS
- Scoring: length-normalized teacher-forced likelihood over `SAFE`, `UNCERTAIN`, and `SCAM`
- Scoring shape: one message, three candidates, 64-token sequence buckets
- Development-fit temperature: `9.999955957053254`
- Development-fit scam threshold: `0.3432537504762315`
- Development-fit SAFE threshold: `0.2859953328065229`
- Publisher dev/test slices were diagnostics only; neither influenced fitting or thresholds.
- The prediction-sealed 1,820-row primary holdout was not opened.

## Core result

| Slice | SCAM / SAFE | Recall | SAFE FPR | Scam precision | Calibrated macro F1 |
|---|---:|---:|---:|---:|---:|
| Development | 514 / 2,008 | 32.30% | 1.99% | 80.58% | 0.4599 |
| Unchanged test | 587 / 1,746 | 30.32% | 2.52% | 80.18% | 0.4295 |

The test recall 95% Wilson interval is 26.74%-34.16%; the test SAFE FPR interval is
1.88%-3.37%. The candidate fails the 97% recall, 2% FPR, 0.94 macro-F1, and every eligible
category-recall gate.

| Test scam category | Examples | Detected | Recall |
|---|---:|---:|---:|
| Credential theft | 144 | 63 | 43.75% |
| Delivery/toll | 72 | 0 | 0.00% |
| Financial | 72 | 0 | 0.00% |
| Identity impersonation | 144 | 82 | 56.94% |
| Opportunity | 72 | 4 | 5.56% |
| Relationship | 72 | 25 | 34.72% |

## Publisher-held SAFE diagnostic

| Slice | SAFE rows | False positives | SAFE FPR | `UNCERTAIN` decisions |
|---|---:|---:|---:|---:|
| Publisher dev | 506 | 2 | 0.40% | 123 |
| Publisher test | 1,199 | 1 | 0.08% | 330 |

Publisher-test FPR by domain is 0% for airline, fast-food, insurance, media, and software, and
0.50% for finance. Abstention is the remaining problem: airline has 24/200 uncertain decisions,
fast-food 12/200, finance 36/200, insurance 23/200, media 176/200, and software 59/199.

## Runtime boundary

On the local Apple-MPS reference path, 50 warmed single-message calls measured 165.82 ms median
and 210.61 ms p95, including prompt formatting, 64-token bucket padding, tensor construction, and
the three-candidate model forward. Input length was 167 tokens p50, 194 p95, and 196 maximum. The BF16 model reports a
1,705,972,160-byte memory footprint (1.59 GiB). Peak allocator telemetry is deliberately marked
unavailable because this final report replayed the exact cached benchmark scores; it must be
remeasured on the uncached trained and quantized candidates.

These numbers do not predict GGUF mobile speed. The under-20-ms contract remains attached to the
complete fast path; the Qwen specialist may satisfy a separate routed-latency budget only if the
end-to-end system gates pass.

## Artifact identities

- Report SHA-256: `1ae1ae7eceb9a502c293215f6422bcbe78c1ab72690bc6677e5e59beca4c8567`
- Text-free 6,713-row prediction ledger SHA-256:
  `ccf383d7713a3d5c9347036a3d7d62b410df2e58996e9751ad2a3ffd8f45c2b4`
- Development JSONL SHA-256:
  `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03`
- Test JSONL SHA-256:
  `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554`
- Publisher dev JSONL SHA-256:
  `fa68cfd458070f4b5d29dce116d18d35c9b894fc79a2296d088886fdb72e0490`
- Publisher test JSONL SHA-256:
  `a654e2915a0ca4534209e7c8c127b3458b53751603b1c65ba855d527df381026`

The JSON report, score cache, and text-free ledger remain ignored local run artifacts. This tracked
record is the immutable control for the audited full-data LoRA experiment. It does not authorize
training before independent label review, opening sealed data, quantization, or Hugging Face
publication.
