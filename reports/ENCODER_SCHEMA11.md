# ModernBERT-base schema-v11 result

Status: **rejected as a release detector; retained as a fast experimental router**.

Schema v11 is the controlled correction to the schema-v10 dialogue-format shortcut. It uses the
unchanged schema-v9 development and regression rows, 600 family-disjoint Taskmaster SAFE roleplays,
1,495 admitted paired synthetic dialogues, and the versioned `speaker-neutral-v1` input transform.
The 1,049-row BothBosu OOD partition and 1,820-row MOZ primary test remain prediction-sealed.

## Three-epoch result

The third epoch was retained because development recall improved at the fixed 2% FPR cap. The
temperature and threshold were fitted on development SAFE/SCAM rows only, then frozen.

| Slice | Rows | Scam recall | SAFE FPR | Argmax macro F1 |
|---|---:|---:|---:|---:|
| development | 2,634 | 97.86% | 1.99% | 0.7516 |
| regression | 2,374 | 99.83% | 11.57% | 0.8859 |
| financial OOD | 431 | 93.08% | 67.25% | 0.4529 |
| WSPR OOD | 488 | 100.00% | n/a; no SAFE rows | 0.6327 |
| forum selection | 1,125 | 99.80% | 28.00% | 0.7640 |
| forum OOD | 2,300 | 99.65% | 25.00% | 0.7780 |
| realistic-placeholder forum OOD | 2,079 | 96.83% | 29.49% | 0.5962 |
| adversarial | 320 | 100.00% | 18.13% | 0.6345 |
| Azerbaijani diagnostic | 4,327 | 99.51% | 12.93% | 0.6357 |
| Chichewa diagnostic | 677 | 76.51% | 31.49% | 0.4834 |
| BothBosu dialogue selection | 294 | 77.30% | 45.75% | 0.2385 |
| Taskmaster selection | 450 SAFE | n/a | 0.00% | 0.3333 |

The model clears development recall/FPR and all sufficiently populated regression scam-category
recall gates, but fails regression FPR, macro-F1, multilingual, financial, forum, adversarial, and
cross-source dialogue gates. It therefore cannot be presented as a sole detector or SOTA result.

## Epoch and dialogue-policy ablations

Epoch two achieved 95.14% development recall / 1.89% FPR, 99.83% regression recall / 11.11% FPR,
and 42.55% BothBosu recall / 30.07% FPR. Epoch three increased recall but also increased collateral
alarms: BothBosu moved to 77.30% recall / 45.75% FPR. A fourth epoch was not run because the failure
was no longer a missing-recall problem.

For the two-epoch checkpoint, retaining the most recent 256 tokens reduced BothBosu recall to
17.02% with 9.15% FPR, showing that early dialogue context carries more signal than the closing
turns. For the final checkpoint, scanning every first-speaker turn independently and taking the
maximum achieved 100% BothBosu recall only by flagging 93.46% of its SAFE dialogues; it also
flagged 16.44% of Taskmaster SAFE dialogues. That incremental max policy is rejected.

The full-dialogue selection set is deliberately difficult for a 256-token fast path: 215/294
(73.13%) inputs are truncated, with untruncated p50/p95/max lengths of 319/481/520 tokens. The
product contract should continue to measure individual incoming messages and bounded recent
context separately from whole-call stress tests.

## False-positive audit and next data decision

At the frozen threshold, regression contains 202 SAFE false positives. Of these, 197 are held
synthetic-v5 counterfactuals and only five are licensed-source Mendeley ham. Three semantic families
account for 177 of the synthetic failures:

| Held synthetic SAFE family | False positives |
|---|---:|
| verified family transfer | 72 |
| known-contact relationship check | 72 |
| marketplace platform-only handling | 33 |

This is useful diagnostic evidence, not permission to copy regression text into training. Schema
v12 should add independently worded, train-only counterfactual families for known-channel family
verification, ordinary transfer coordination, marketplace platform protections, and legitimate
multilingual official-app alerts. It should also add a family-disjoint dialogue development slice
before fitting any separate long-context calibration. Dataset size should grow only by the rows
needed to cover those behaviors.

## Performance and immutable artifacts

On Apple MPS, batch-one model-forward latency measured 8.74 ms median / 14.91 ms p95. The required
tokenizer-entry-through-probability path measured 9.38 ms median / 15.55 ms p95 across 250 samples.
This clears the desktop 20 ms experiment gate for the unquantized 602,034,248-byte checkpoint, but
does not establish physical-mobile latency or include SDK evidence extraction and I/O.

| Artifact | SHA-256 |
|---|---|
| schema-v11 data manifest | `42449fd733d82179a7bcc47614c176931af3378a3ab0d505b7b443acf546e4ac` |
| final model weights | `744698e322198b977c912d73bb21c5a69d9a7b8394c59e2aac89a451719e6522` |
| final calibration | `be7d8dc96fcd28f9977de7c8bc477d0d9e30608d7902dd1fc7442e88093ab199` |
| final report | `59e6fa564e9562bcad33ff018b219cb619b996afb2c1266e7e24b375aeea8241` |
| epoch-two report | `9eba5ab278c48638e3bca249b095aa6f44c1105e172162b7d37a5e38f8374ada` |

The model and raw prediction ledgers remain ignored local artifacts. Public documentation contains
metrics and hashes, not redistributed source rows or a release checkpoint.
