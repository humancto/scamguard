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
| Eligible fitting pool | 8,958 | `e6baa68cb39cdb005dbf6e0ed27d81645ebdf580ccf83db1a76d5f1cdc7fe48f` | Unused |
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
