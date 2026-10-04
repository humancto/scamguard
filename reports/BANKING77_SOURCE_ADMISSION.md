# BANKING77 financial-service SAFE boundary source

## Pre-prediction decision

BANKING77 is admitted as a licensed, human-authored financial-service boundary source with a
strictly separated selection slice and sealed official test. It contains 13,083 English online
banking customer-service queries over 77 intents, including compromised or stolen cards, identity
verification, unrecognized payments, cash withdrawals, refunds, transfer cancellation, and source
of funds. These are legitimate support requests that share high-risk vocabulary with scams.

- Publisher repository: <https://github.com/PolyAI-LDN/task-specific-datasets>
- Pinned revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Paper: <https://arxiv.org/abs/2003.04807>
- License: CC-BY-4.0
- Published rows: 10,003 official train and 3,080 official test
- Declared intents: 77

The publisher describes the rows as customer-service queries but does not document whether they
were naturally occurring, crowdsourced, or authored internally. ScamGuard therefore treats them as
licensed human-authored queries, not verified real customer communications. The legitimate banking
domain supplies a weak SAFE label; no claim of individually adjudicated safety is made.

## Pinned source identities

| File | SHA-256 |
|---|---|
| `train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |
| `categories.json` | `53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63` |
| `LICENSE` | `7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661` |

All source and derived rows remain local and Git-ignored. The repository contains the hash-pinned
fetcher, fail-closed builder, source audit, and attribution rather than redistributing the corpus.

## Split and contamination contract

1. Validate every source hash, CSV schema, row count, declared intent, and license artifact.
2. Replace URLs, email addresses, phone-like values, and long digit sequences before IDs,
   clustering, or output.
3. Deduplicate and near-cluster by normalized character SimHash, retaining one row per family.
4. Give the official test split precedence. Remove every official-train row with exact or near
   overlap against any official-test row.
5. Remove overlap against all existing canonical, call, forum, PPoNE, international-robocall,
   Vystadial, and synthetic-dialogue references.
6. From the remaining official-train families, select exactly ten deterministic representatives
   per intent for the open selection diagnostic. The rest form an eligible fitting pool but remain
   unused until the frozen baseline is recorded and a separate curriculum is declared.
7. Keep the publisher's official test local and prediction-sealed. No model, threshold, prompt, or
   training decision may inspect its predictions before a candidate passes all open gates.

The audit removes eight normalized duplicate train rows and one duplicate test row. It removes 123
official-train rows that are near an official-test family and one official-test row overlapping an
existing ScamGuard reference. Near clustering removes another 144 train and 18 test variants.

| Artifact | Rows | SHA-256 | Status |
|---|---:|---|---|
| Eligible fitting pool | 8,958 | `e6baa68cb39cdb005dbf6e0ed27d81645ebdf580ccf83db1a76d5f1cdc7fe48f` | 388 rows used by Stage 13/14; official test still sealed |
| Open balanced selection | 770 | `efab4f6e90c9c84b6a44cf9546ee3036bf03ec59b49922b2275c9957f5b8ccf7` | May score once |
| Official test | 3,060 | `f7b9e005a01d78be64843f0d2dc249889500179f165b44a1ecda7a5dfeb9f5f9` | Prediction-sealed |

The selection and sealed artifacts each retain all 77 intents. The full audit compares against
80,092 prior reference rows. The source contains no email, phone-like, long-digit, or URL findings
under the current privacy recognizers.

## Predeclared baseline measurement

Evaluate the frozen Stage 7 Qwen3.5-0.8B adapter once on `banking77_validation` using the existing
Stage 7 development temperature, SCAM threshold, SAFE threshold, prompt, branch-token scorer, and
sequence bucket. Do not recalibrate.

Report:

- thresholded SAFE false-positive rate and 95% Wilson interval;
- calibrated SAFE, UNCERTAIN, and SCAM counts;
- raw argmax counts;
- per-intent false-positive counts, explicitly noting that each intent has only ten examples;
- text-free prediction and report hashes.

This open result can decide whether BANKING77 offers a useful fitting boundary. It cannot authorize
publication, quantization, sealed-test evaluation, or a SOTA claim. If Stage 7 already clears the
source, the fitting pool remains unused. If it fails, a separately committed curriculum must use
only `banking77_fit`, preserve the Stage 7 development calibration contract, and pass every prior
open regression gate before the official test can be considered.

Reproduce the source audit with:

```bash
make banking77-safe
```

After this declaration is committed, run the one frozen open diagnostic with:

```bash
make qwen-08b-stage7-banking77-selection
```

## Frozen Stage 7 result

The source declaration, split identities, metrics, and sealed-test prohibition above were committed
and pushed as `d1283eaaaec88c2d8695b324ed32fbd8930a6dc7` before any model prediction. The one permitted
selection run then scored all 770 rows on Apple MPS using the identity-checked Stage 7 development
cache and frozen calibration. No official-test prediction was generated.

| Metric | Frozen result |
|---|---:|
| Thresholded SAFE false positives | **8/770 (1.039%)** |
| SAFE FPR 95% Wilson interval | 0.527–2.037% |
| Calibrated verdicts | 762 SAFE, 0 UNCERTAIN, 8 SCAM |
| Raw argmax verdicts | 759 SAFE, 4 UNCERTAIN, 7 SCAM |
| Binary Brier score | 0.00399 |

The eight false positives are concentrated in five action-bearing intents:

| Intent | False positives / 10 |
|---|---:|
| `pin_blocked` | 2 |
| `request_refund` | 2 |
| `top_up_reverted` | 2 |
| `cash_withdrawal_not_recognised` | 1 |
| `cancel_transfer` | 1 |

Each per-intent denominator is only ten, so these are error-localization counts rather than stable
intent-level rate estimates. Seven of the eight raw argmaxes were already SCAM. The eighth had a
SAFE argmax but crossed the frozen low SCAM threshold, so threshold changes would trade away the
development recall contract and are not an acceptable repair.

- Evaluation report SHA-256:
  `23129c972fd5fa5ac4f049010f764cbe432181c55b2bb76e20eb0549633dd237`
- Text-free prediction ledger SHA-256:
  `d4ae7e49a892cb1057e39cd6c25a5ccb5d329ebd45f8e6e4c6153b3583e03dfc`
- Adapter SHA-256:
  `14f1d2bf121e76fa158cea722416994ec9cbfbc545956d24b9693b7808441357`
- Frozen calibration SHA-256:
  `a2309cc5479a85efa51ab74219411593a1e8b372888b4b8f97aaa1bf741b078b`

The BF16-plus-LoRA reference measured 93.89 ms median and 108.80 ms p95 on the standard
50-example development latency loop, with a sampled MPS driver-allocation peak of 2.51 GB. This is
not the final mobile runtime path.

The point estimate clears the 2% SAFE-FPR target, but the 95% upper interval narrowly exceeds 2%,
and the failures identify a coherent legitimate financial-action boundary. BANKING77 therefore
earns a small, split-safe training experiment. That experiment must use only `banking77_fit`, mix
in scam and uncertainty retention examples, retain the exact Stage 7 development calibration
contract, and pass the open BANKING77 selection plus every prior regression gate before the sealed
official test can be considered.

## Frozen Stage 13 experiment recipe

Stage 13 is a targeted continuation from the Stage 7 adapter, not a replacement model and not a
release candidate. The deterministic curriculum contains 1,400 rows in 993 families:

- 388 BANKING77 fit-only SAFE rows: four per intent across all 77 intents, increased to twenty for
  each of the five open-error intents above;
- 385 family-diverse Stage 7 source/verdict anchors;
- all 356 three-way FTC-pattern action-state examples;
- 192 SCAM/UNCERTAIN examples across credential, financial-impersonation, government, and payment
  categories;
- all 152 licensed CC0 real-call examples; and
- 48 SAFE plus 48 SCAM phone-boundary families.

After overlap removal between selection reasons, the final label mix is 744 SAFE, 420 SCAM, and
236 UNCERTAIN examples. The Stage 7 development split is copied byte-for-byte. The builder reads
the open BANKING77 validation text only for an overlap audit; it copies or transforms zero such
rows. It does not read the official test text.

The frozen optimization recipe is one epoch from the Stage 7 LoRA, learning rate `1e-6`, effective
batch size 16, maximum length 640, and seed `20261004`. The token audit covers 4,034 train-plus-dev
examples, finds zero examples over 640 tokens, and reports a maximum of 638 tokens. Apple MPS
preflight passed before training.

Promotion is fail-closed. The open BANKING77 slice must fall to at most 3/770 false positives
(0.5% FPR with Wilson upper bound at most 1.25%), while Stage 7 development macro-F1, phone scam
recall, audited ordinary-phone SAFE FPR, PPoNE recall/macro-F1/UNCERTAIN recall, Vystadial SAFE FPR,
and international-robocall recall must not regress. Only a joint pass permits the full regression;
the official BANKING77 test remains prediction-sealed.

Reproduce the frozen data, token audit, and preflight with:

```bash
make qwen-08b-stage13-preflight
```

## Stage 13 development rejection and Stage 14 freeze

Stage 13 completed one epoch on Apple MPS, but it failed the development screen before any
candidate prediction was made on BANKING77 validation. Scam recall stayed at 499/514 (97.08%),
while SAFE false positives increased from 5 to 6 and calibrated development macro-F1 fell from
0.7856 to 0.7517. The direct continuation is therefore rejected; no Stage 13 BANKING77, phone,
PPoNE, Vystadial, international-robocall, full-regression, or sealed-test result exists.

- Stage 13 adapter SHA-256:
  `2862864817351ae09d2b3a4cf9832b612731bafb36d934a067c8a0d7ae2cfb93`
- Training receipt SHA-256:
  `bd20209f713b10f54eca6213dbe5dbeafd1ffb73785e35b423e83df3e5e9194b`
- Development report SHA-256:
  `d16b1f9bb75afaf7dbcb5583937958d397d7afa57021eef12f324eefc4ed6b20`
- Frozen config SHA-256:
  `0fce65aeeb5aba053965005d1a1cb7b8e276563693663a850523d53f08ea6bce`

Stage 14 is one final reduced-step continuation using the identical curriculum, seed, batch shape,
and Stage 7 parent; only the learning rate changes from `1e-6` to `5e-7`. This is a direct test of
the evidence-supported hypothesis that Stage 13 moved too far from the retained Stage 7 boundary.
The Stage 14 recipe is frozen before training and must pass the development-only gate before any
open selection split can be scored. If it fails, the BANKING77 line stops rather than searching
additional learning rates against the repeatedly opened development set.

- Stage 14 frozen config SHA-256:
  `12dda7733c70c650ea33743e9165294f43a4f4f6aafac84af95bdf7864b0355e`
