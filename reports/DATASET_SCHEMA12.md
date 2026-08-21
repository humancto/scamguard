# ScamGuard dataset schema v12

Freeze date: 2026-08-21.

Schema v12 is a narrow counterfactual repair to rejected schema v11. It adds 512 balanced,
training-only examples in eight paired families while keeping development, regression, Taskmaster
selection, BothBosu selection, and both prediction-sealed sources unchanged. No evaluation message
was copied into the new curriculum.

## Why this increment exists

The schema-v11 149M encoder reached 97.86% recall at 1.99% FPR on development and stayed below the
20 ms desktop target, but failed the unchanged regression set at 11.57% FPR. A text-free prediction
ledger found 202 SAFE false positives. Of those, 197 belonged to four synthetic-v5 error families:
verified transfers, known contacts, in-platform marketplace actions, and official-app alerts. Five
were naturally occurring Mendeley ham rows. This localized error pattern justified a small paired
repair rather than more parameters or bulk internet data.

Schema v12 adds 64 rows to each of these independently worded families:

- `known_channel_transfer_safe` and `new_number_transfer_scam`;
- `family_status_safe` and `private_emergency_payment_scam`;
- `marketplace_in_app_safe` and `marketplace_refund_scam`;
- `official_app_review_safe` and `bank_review_link_scam`.

The 256 SAFE examples preserve a legitimate trust boundary. The 256 SCAM counterparts change the
sender, channel, secrecy, link, or payment action. Copy is original deterministic slot filling and
records its generation method and official pattern source. The first generated draft failed closed
because 19 SCAM rows lacked an extractive risk span; those marketplace-refund templates were
corrected and the entire corpus rebuilt before this freeze.

## Frozen counts

| Split or tier | Rows | SAFE | UNCERTAIN | SCAM | Use |
|---|---:|---:|---:|---:|---|
| train | 14,446 | 8,197 | 855 | 5,394 | fitting |
| dev | 2,634 | 2,008 | 112 | 514 | calibration and checkpoint selection |
| regression test | 2,374 | 1,746 | 41 | 587 | historical regression only |
| named processed diagnostics | 4,344 | source-specific | source-specific | source-specific | no fitting |
| Azerbaijani OOD | 4,327 | 2,963 | 1,160 | 204 | no fitting or selection |
| sealed MOZ primary test | 1,820 | 1,294 | 0 | 526 | prediction-sealed |

The independent validator covers 28,125 unique open processed rows before the separately sealed
MOZ source and finds no exact or family leakage. The 23,798-row core plus named diagnostics contains
13,127 naturally occurring licensed-source rows, 600 human-authored crowdsourced roleplays reported
as a separate tier, and 10,071 controlled synthetic rows. The fitting split itself contains 7,699
licensed-source rows, 600 roleplays, and 6,147 synthetic rows.

## Isolation and use boundaries

- Development and regression are byte-identical to schemas v9 through v11.
- The 450 Taskmaster SAFE conversations and 294 BothBosu selection conversations may inform
  candidate selection but never fitting or threshold calibration.
- The 1,049-row BothBosu OOD partition and 1,820-row MOZ primary test remain prediction-sealed.
- The targeted generator uses aggregate error categories and original templates. It does not read
  or reproduce evaluation-row text.
- Source-wide privacy normalization, extractive-evidence admission, exact deduplication, SimHash
  near-template clustering, and family-isolation validation all run again after generation.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| schema-v12 manifest | `e4dd86b5ceb753803979561d67aa49123605c491c7a1d007fb57d3b67e2b1f14` |
| train | `a5c964838d742352897c388cfa42142f498ffd369d046515e97038b521105d24` |
| unchanged dev | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| unchanged regression | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| complete synthetic generator output | `3fdbdaa3719bb4ed316d77966106a7f58df17ca2e94069cbc96b6733bf16c7ec` |
| synthetic generator manifest | `beea78dce364e7ba100c66d2024c79c0fae8a9bcb66cef8c366da31bdd002f49` |
| synthetic dialogue v2 | `7f963f8557d9307e2ec605638ef4cccbc798a3414d774d42e31e9d1917fdceed` |
| Taskmaster train | `c756886b2d97dc41d98956eb7dc468a7c43622e1861229b605768cedc78007c3` |
| Taskmaster selection | `539b81a06328b2914407565c1bb7fac54a486333cea45a0853b4e2160df79760` |
| speaker-neutral-v1 implementation | `2077db829188313088b1ba02b02763ab85ce2fcfffc9bc8abf8319d1cab46bd4` |
| BothBosu selection | `c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0` |
| BothBosu sealed OOD | `33d480aa505f16014e18a7193f379b618e7a9feeb90262c93e77433c022c1193` |
| sealed MOZ primary test | `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` |
| 240-row independent audit sample | `f7ea70711429d1b68f1676271c11d5d54b134ccb627b0644197017bbdd10f8ea` |

## Release boundary

The first schema-v12 encoder is rejected. Although the repair reduced regression false positives,
it caused a complete 72-row miss on an unchanged development identity-scam family. This freeze is
therefore retained as an auditable ablation and did not replace schema v11 without a controlled
dose/coverage experiment. The subsequent schema-v13 dose-16 neural run is also rejected; its
separately reported deterministic policy remains an open-set research candidate. Full results are
in [`ENCODER_SCHEMA12.md`](ENCODER_SCHEMA12.md) and
[`ENCODER_SCHEMA13.md`](ENCODER_SCHEMA13.md).

The 240-row audit sample has zero independent decisions, native review remains incomplete for
multilingual data, and no physical-device latency has been measured. This freeze supports a serious
experiment, not a production or SOTA claim. Prediction-sealed sources stay closed until a candidate
and its dialogue policy clear the declared open selection gates and are frozen.
