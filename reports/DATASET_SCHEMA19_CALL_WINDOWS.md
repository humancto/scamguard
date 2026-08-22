# Dataset schema 19: licensed recent-call window curriculum

## Decision

Schema 19 replaces schema 18's repeated short synthetic-pair exposure with a single
length-matched pair curriculum and more licensed call windows. It is accepted for one frozen
training experiment. It is not a benchmark result and does not open any sealed partition.

The processed manifest SHA-256 is
`a87c1cfdf6e13c7e070499c9fbe8f2fe8d29f3ee0ccf4535c1ffbbd987b3147a`.
The predeclared training configuration SHA-256 is
`2d880a113b656ab9af3b8ab88bc96bca1695e46554bf415d3b5249a7c776ad4a`.

## Training composition

Schema 19 starts from schema 14, preserving the schema-13 corpus plus the 161 CC0 early
scam-call windows. The final training artifact contains 18,162 rows:

| Call component | Rows | Independent train families | Provenance | License |
|---|---:|---:|---|---|
| Taskmaster short + long recent windows | 1,193 | 600 conversations | Human-authored Wizard-of-Oz service-call roleplay | CC-BY-4.0 |
| YouTube early + recent + long-recent windows | 435 | 145 connected call/source families | Real scam-call or autodialer transcript-derived, positive-only | CC0-1.0 |
| Long evidence/action minimal pairs | 3,072 | 1,536 semantic pairs | Original deterministic advisory-grounded counterfactuals | Apache-2.0 |

The Taskmaster increment adds 593 non-duplicate long windows to the 600 existing short windows.
The YouTube increment adds 137 existing recent windows and 137 new 1,000-character recent
windows only for source records already assigned to the train partition. No publisher validation
or OOD source record is expanded or fitted.

Taskmaster is realistic human-authored roleplay, not naturally occurring private calls. The
YouTube source is counted as real scam-call-derived language, but not as ordinary victim calls or
independently relabelled ground truth. Multiple windows from one call are never counted as
independent calls.

## Synthetic construction

The 1,536 schema-18 train pair families are regenerated as long histories. Each SAFE/SCAM pair
has byte-identical preceding turns and differs only in the final proposed action. Shared history
contains an ordinary service review, requests for explanation and independent verification, and
no copied external benchmark text. The final SCAM action covers one of eight mechanisms:

- credential or one-time-code request
- remote access
- protection transfer
- cryptocurrency payment
- secrecy or isolation
- login link
- gift-card payment
- advance fee

All 3,072 train pair rows are 325–357 ModernBERT tokens before truncation (median 342), so the
frozen latest-256-token policy is exercised during fitting. The 1,024-row pair validation slice
contains 512 families from four scenario-held-out domains; all are 326–360 tokens and remain
outside fitting and threshold selection.

This is one pair exposure per epoch. Schema 18 used every short pair twice and overfit the new
task while moving the established real-dialogue boundary. Schema 19 therefore cuts pair sampling
in half and pair-loss weight from 2 to 1 while preserving the margin target of 3.

## Real and human-authored window controls

- Taskmaster is pinned to revision `d92cb6af3005f1dc09c39e75e7daf4a04905e00b` and raw
  SHA-256 `cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f`.
- Taskmaster is partitioned by conversation ID before sampling. The new SAFE-only long-window
  diagnostic contains 447 non-duplicate held-out conversations, has a 239-token median and
  261-token p95, and is never used for fitting or threshold selection.
- The YouTube archive is pinned at SHA-256
  `3f67497736e9421c2f6e59efc46c129006419d40fc752cbb981042940384cedd`.
- YouTube connected source/template families were partitioned before any schema-19 expansion.
  The existing 70-window/35-family open validation and 80-window/40-family sealed OOD artifacts
  are unchanged.
- Emails, URLs, and phone/account-like values are privacy-normalized before materialization.
- Seven Taskmaster train windows and three validation windows whose long form was identical to
  the existing short form were removed rather than double-counted.

The long YouTube train windows are 87–238 tokens (median 180, p95 230). The Taskmaster long train
windows are 117–275 tokens (median 239 among the long subset). These distributions cover the
actual 256-token inference region without pretending that short messages are long calls.

## Exclusions

- Direct Reddit scraping remains excluded under current Reddit API model-training rights terms.
  Public reports may guide the taxonomy, but individual posts are not copied.
- AppTek and BothBosu remain evaluation-only and contribute zero training rows.
- The GrandgemMa composite is excluded because its published rows include BothBosu train and test
  examples; fitting it would contaminate ScamGuard's dialogue benchmark.
- TeleAntiFraud remains gated and unavailable on this machine. No derivative mirror is used to
  bypass publisher access controls.
- Arabic, Thai, Vietnamese, and other synthetic dialogue releases remain diagnostics or research
  leads until native-language review and provenance checks are complete.

## Validation and frozen experiment

The general dataset validator passed 33,312 unique rows across the open processed artifacts with
no family leakage. The primary 1,820-row evaluation remains sealed. Schema-19 preflight also
verified all hashes, teacher ID coverage, pair structure, scenario holdouts, exact token-length
contracts, SAFE-only call-window labels, and sealed-state flags.

The model experiment initializes from schema 13, uses 14,062 text-free teacher-logit anchors,
trains every one of the 18,162 rows once for 1,136 optimizer steps, and applies retention weight
4 plus pair weight 1. It must pass development, regression, long pair, long SAFE-call,
Taskmaster, and BothBosu gates before AppTek/YouTube selection; only a complete quality pass can
advance to Core ML export and physical-device latency measurement.

Dataset size does not determine the under-20-ms runtime target. Runtime is governed by model
architecture, token budget, quantization, tokenizer, and device. This corpus is sized to improve
the decision boundary first; compression begins only after the teacher passes every quality gate.
