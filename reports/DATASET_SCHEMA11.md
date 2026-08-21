# ScamGuard dataset schema v11

Freeze date: 2026-08-21.

Schema v11 is the mobile-window and dialogue-balance correction to rejected schema v10. It keeps
development and regression byte-identical, does not open either prediction-sealed source, and
changes only fitting dialogue composition plus the declared model-input transform.

## Why v10 was rejected

The schema-v10 149M checkpoint reduced Taskmaster SAFE FPR from 95.78% to 0% and repaired the held
identity family from 0/72 to 72/72. It also collapsed BothBosu scam-dialogue recall to 13.48%, raised
regression FPR to 11.23%, and right-truncated 79.17% of its Taskmaster fitting rows. This is direct
evidence of a source-format and dialogue-class shortcut, not a reason to scale the model.

## Frozen counts

| Split or tier | Rows | SAFE | UNCERTAIN | SCAM | Use |
|---|---:|---:|---:|---:|---|
| train | 13,934 | 7,941 | 855 | 5,138 | fitting |
| dev | 2,634 | 2,008 | 112 | 514 | calibration and checkpoint selection |
| regression test | 2,374 | 1,746 | 41 | 587 | historical regression only |
| named processed diagnostics | 4,344 | source-specific | source-specific | source-specific | no fitting |
| Azerbaijani OOD | 4,327 | 2,963 | 1,160 | 204 | no fitting or selection |
| sealed MOZ primary test | 1,820 | 1,294 | 0 | 526 | prediction-sealed |

The validator covers 27,613 unique open processed rows before the separately sealed MOZ source and
finds no exact or family leakage. The 23,286-row core plus named diagnostics contains 13,127
naturally occurring licensed-source rows, 600 human-authored crowdsourced roleplays reported as a
separate tier, and 9,559 controlled synthetic rows. The fitting split itself contains 7,699
licensed-source rows, 600 roleplays, and 5,635 synthetic rows.

## Dialogue correction

- Taskmaster fitting is capped at 600 disjoint families, exactly 100 per domain; selection remains
  450 disjoint families, exactly 75 per domain.
- Each Taskmaster row retains the newest complete turns within 425 characters. Across all 5,507
  eligible source conversations, the pinned ModernBERT tokenizer observes at most 150 tokens after
  the input transform; selected fitting and validation maxima are 123 and 126 tokens.
- Synthetic dialogue v2 generates 1,536 balanced rows across 12 paired scenarios. Exact/near
  cleanup admits 1,495 fitting rows: 732 SAFE and 763 SCAM.
- The admitted dialogue-shaped fitting subset is therefore 1,332 SAFE versus 763 SCAM, rather than
  schema v10's 2,184 SAFE versus 384 SCAM.
- `speaker-neutral-v1` strips synthetic-only headers, maps corpus-specific role names to compact
  first-appearance labels such as `A:` and `B:`, and normalizes whitespace before tokenization.
  It activates only for inputs with at least four turns and two known speaker roles, leaving normal
  short messages unchanged.

Eleven long licensed public-forum fitting rows still exceed 256 tokens (0.08% of training) and are
right-truncated under the declared short-text policy. The Taskmaster and synthetic fitting curricula
have zero truncation. The BothBosu selection set remains intentionally difficult: 215/294 full
conversations exceed 256 tokens after the compact transform, so its report must disclose that the
score is a first-window diagnostic rather than full-transcript latency.

## New synthetic scenarios

Version 2 retains remote support, government cases, refunds, bank fraud, delivery, jobs,
wrong-number grooming, and insurance, and adds tax collection, family emergencies, investment
grooming, and marketplace overpayment. Every scenario has a legitimate counterfactual and cites an
FTC, FBI/IC3, IRS, or USPIS pattern source. The copy is original deterministic slot filling; it does
not reproduce source messages or the external diagnostic.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| schema-v11 manifest | `42449fd733d82179a7bcc47614c176931af3378a3ab0d505b7b443acf546e4ac` |
| train | `3b390564890091be8aae1c4f1f977c3048dd5b43d130730f249141049c11a544` |
| unchanged dev | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| unchanged regression | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| Taskmaster train | `c756886b2d97dc41d98956eb7dc468a7c43622e1861229b605768cedc78007c3` |
| Taskmaster selection | `539b81a06328b2914407565c1bb7fac54a486333cea45a0853b4e2160df79760` |
| generated synthetic dialogue v2 | `7f963f8557d9307e2ec605638ef4cccbc798a3414d774d42e31e9d1917fdceed` |
| speaker-neutral-v1 implementation | `db66f2b829e4f9c50dfad54035a0e10f7617f8f168b9d337731fa355d78de965` |
| BothBosu selection | `c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0` |
| BothBosu sealed OOD | `33d480aa505f16014e18a7193f379b618e7a9feeb90262c93e77433c022c1193` |
| sealed MOZ primary test | `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` |

## Release boundary

The 216-row independent audit workbook remains unlabeled, native review is incomplete for
multilingual data, and no physical-device latency has been measured. Candidate selection may use
the open Taskmaster and BothBosu selection slices. The 1,049-row BothBosu OOD and 1,820-row MOZ
primary test stay prediction-sealed until the model and long-dialogue policy are frozen.
