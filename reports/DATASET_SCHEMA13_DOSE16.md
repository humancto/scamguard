# ScamGuard dataset schema-v13 dose-16 ablation

Freeze date: 2026-08-21.

Schema v13 is a controlled data-volume experiment, not a new source expansion. It starts from the
same schema-v11 training baseline and admits 16 deterministic examples from each of schema v12's
eight paired counterfactual families: 64 SAFE and 64 SCAM rows in total. Development, regression,
open dialogue-selection, OOD, and prediction-sealed artifacts remain unchanged.

## Hypothesis

The 512-row schema-v12 increment reduced unchanged-regression FPR from 11.57% to 4.18%, but caused a
complete 72-row miss on the untouched `identity_case_callback` development family. A text-free
ledger localized 59 of the remaining 73 regression false positives to
`family_transfer_verified`. Schema v13 asks whether one quarter of the corrective dose retains some
of the specificity gain without learning the broad identity-scam veto.

This is the only intended training-data variable. The base model, revision, seed, optimizer,
epochs, input transform, threshold-calibration procedure, and evaluation rows are held constant.

## Frozen counts

| Split or tier | Rows | SAFE | UNCERTAIN | SCAM | Use |
|---|---:|---:|---:|---:|---|
| train | 14,062 | 8,005 | 855 | 5,202 | fitting |
| dev | 2,634 | 2,008 | 112 | 514 | calibration and checkpoint selection |
| regression test | 2,374 | 1,746 | 41 | 587 | historical regression only |
| named processed diagnostics | 4,344 | source-specific | source-specific | source-specific | no fitting |
| Azerbaijani OOD | 4,327 | 2,963 | 1,160 | 204 | no fitting or selection |
| sealed MOZ primary test | 1,820 | 1,294 | 0 | 526 | prediction-sealed |

The independent validator covers 27,741 unique open processed rows before the separately sealed
MOZ source and finds no exact or family leakage. The 23,414-row core plus named diagnostics contains
13,127 naturally occurring licensed-source rows, 600 human-authored roleplays, and 9,687 controlled
synthetic rows.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| schema-v13 dose-16 manifest | `68c481a889ce135eff4a472b71daeb65a00ee886daec4c415549f0726586c6bf` |
| train | `499e2cabc3e55859a46c0e363c46a29aa0f0aa8a72fa0d3c30c29b27fd28e325` |
| unchanged dev | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| unchanged regression | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| dose-16 synthetic generator output | `fc410f821a6b74e0fbe7d7615fe7b6bb7f5efc8f3830c9419ec1af1b96f65f51` |
| dose-16 synthetic manifest | `a1c1ef1e2a030ea1dc10d599f4615d032573e182cf3d2aee7315493f0e49aafd` |
| sealed MOZ primary test | `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` |

The 1,049-row BothBosu OOD partition and 1,820-row MOZ primary test remain prediction-sealed.
