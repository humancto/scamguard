# ScamGuard source research — 2026-09-05

## Decision

Admit `shakeleoatmeal/phone-scam-detection-synthetic` as a bounded synthetic
curriculum source at revision
`27a1f6d0cf21dda995130d221383900b060dbcc9`. Admit its publisher validation split
only as a development diagnostic. Keep its publisher test split prediction-sealed
until a candidate is frozen. Do not use any of its results as state-of-the-art
evidence.

Continue using the pinned MultiDoGO role-play calls as the licensed, human-spoken
safe-call source. Add full conversation windows, but remove their weak heuristic
action targets before training. The single-turn targets confuse legitimate requests
for information with attacker-controlled disclosure; full context is the evidence
needed to tell those situations apart.

Do not scrape Reddit or other public forums directly. Public visibility is not a
license for model training or redistribution, and posts can contain personal or
sensitive information. Forum data can enter the project only through an explicit,
reviewed license and a privacy/provenance pipeline.
## Admitted sources

| Source | Status | Evidence | Role and limits |
|---|---|---|---|
| [Phone Scam Detection Dataset (Synthetic)](https://huggingface.co/datasets/shakeleoatmeal/phone-scam-detection-synthetic) | Admitted at exact revision | Publisher card declares MIT; 1,259 train, 361 validation, 180 test records; exact bytes and SHA-256 values are stored in `download_receipt.json` | Synthetic curriculum and development diagnostic only. The `variation_style` field perfectly predicts the label in the source, so its held splits are not credible final-quality evidence. Metadata is never model input. |
| [MultiDoGO](https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset) | Previously admitted at exact revision `baa30639c4b271f394b81443c842193407cdf26d` | Publisher repository and locally hash-pinned manifest; CDLA-Permissive-1.0 | Human-spoken role-play service calls. Legitimate-domain labels are weak SAFE evidence, not independent scam annotations. Publisher partitions and conversation families remain disjoint. |

## Investigated but not admitted

| Source | Decision | Reason |
|---|---|---|
| [TeleAntiFraud](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud), [publisher code](https://github.com/JimmyMa99/TeleAntiFraud) | Blocked pending human access decision | The sanitized release is Apache-2.0 and pinned by the fetcher at revision `0872e54b584b28d34e0911dffcf696f0b2e5e49a`, but Hugging Face returns `401 GatedRepoError` and asks the account holder to accept access terms and share contact information. ScamGuard does not accept those terms on the user's behalf. |
| [menaattia/phone-scam-dataset](https://huggingface.co/datasets/menaattia/phone-scam-dataset) | Excluded | No usable license declaration was found in the dataset card or repository files reviewed. The data also appears synthetic, so it adds no defensible real-world evidence. |
| [COVA-X / ScamLingua](https://scamlingua.org/) | Excluded | Access is request-based and the published terms prohibit redistribution and restrict use to non-commercial research. Those terms conflict with a reproducible public model pipeline. |
| Sting9 | Excluded | The published corpus uses ODC-BY-NC. The non-commercial restriction is incompatible with the intended broadly usable release. |
| [VISHGUARD](https://pubmed.ncbi.nlm.nih.gov/42373678/) | Research lead, not admitted | The 2026 data descriptor reports 3,000 multilingual synthetic phishing/scam examples. Exact artifact license, immutable revision, source files, and label-confound audit were not yet established, so no data was downloaded or used. |
| [BothBosu scam-dialogue](https://huggingface.co/datasets/BothBosu/scam-dialogue) | Prior-open regression only | Useful as an external dialogue diagnostic, but never used for fitting or threshold selection in this experiment. |

## Local admission evidence

The pinned synthetic source contains 1,800 publisher rows. The deterministic build:

- quarantined one generation-prompt artifact;
- clustered 1,799 remaining records into 1,739 near-duplicate families after privacy redaction;
- quarantined 11 mixed-label families;
- collapsed 33 same-label duplicates;
- removed cross-corpus overlaps against the existing schema and BothBosu references;
- retained 1,190 train, 344 validation, and 172 sealed-test records.
The source has a major construction confound: SAFE records use only
`detailed_helpful`, `efficient_professional`, and `standard_professional`, while
SCAM records use only `direct`, `somewhat_subtle`, and `very_subtle`. The model sees
only dialogue text, not that metadata, but the writing styles can still leak the
label. This is why the source is useful for coverage but cannot validate broad
generalization.

## Experiment boundary

Schema 25 is an exploratory data-semantics experiment initialized from the frozen
schema-23 ModernBERT checkpoint. Its purpose is to test whether full legitimate
conversation context reduces false alarms without losing scam recall. A candidate
must pass the unchanged development and safety gates before any new sealed
regression, quantization, physical-device claim, or Hugging Face publication.

No Reddit text, gated TeleAntiFraud record, phone-scam publisher test prediction, or
sealed primary holdout was opened for this decision.

## Claim-source ledger

| Claim | Source | Confidence |
|---|---|---|
| The admitted phone corpus declares MIT and provides fixed train/validation/test parquet files. | Publisher Hugging Face dataset card plus locally stored exact-revision receipt | High |
| TeleAntiFraud has an Apache-2.0 sanitized release but currently requires gated account approval. | Publisher Hugging Face dataset page, publisher GitHub repository, and observed `401 GatedRepoError` | High |
| COVA-X cannot be redistributed and is restricted to non-commercial research. | ScamLingua publisher access page | High |
| VISHGUARD is multilingual and synthetic with 3,000 examples. | 2026 Scientific Data descriptor indexed by PubMed | High for publication facts; not sufficient for artifact admission |
| The admitted phone corpus has a perfect style/label metadata confound. | Deterministic local audit of all 1,800 publisher rows | High |

## Scope and stopping rule

This pass prioritized sources that could legally and reproducibly enter a public
training pipeline. It stopped after finding one admissible synthetic dialogue source,
confirming the existing real-call source, and documenting why the most relevant
alternatives were blocked or unsuitable. More search breadth would not justify
training on unlicensed text; the next useful evidence comes from the frozen model
experiment and independent human label audit.
