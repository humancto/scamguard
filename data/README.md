# Data sources

Raw, generated, and processed data are intentionally ignored by Git. Reproduce them with
`make data`. `data/raw/sources.json` records exact URLs, SHA-256 digests, license names, and allowed
uses for the downloaded files.

## Accepted sources

| Key | Rows described by source | License | ScamGuard use |
|---|---:|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | 5,574 | CC-BY-4.0 | ham → SAFE; generic spam → UNCERTAIN |
| [Mendeley SMS Phishing Dataset](https://doi.org/10.17632/f45bkkt8pr.1) | 5,971 | CC-BY-4.0 | ham/spam/smishing with conservative mapping |
| [Financial Scam Classification](https://doi.org/10.17632/znsk27yk3h.2) | source-defined | CC-BY-4.0 | external holdout only |
| [NCSU WSPR SMS Phishing](https://github.com/wspr-ncsu/sms-phishing/commit/ef7da01dfc145ce283a2af380e5dd8b817191ee7) | 68,029 reports | MIT | one example per near-template campaign cluster; 1,913 training rows and a 488-row no-SAFE holdout |
| [IMC 2025 Public-Forum Smishing](https://github.com/reportsmishing/Smishing-Dataset-IMC25/tree/a6175560b57387199871e51fbef6bc523d2516b4) | 33,869 labeled reports | CC-BY-4.0 | 1,248 train, 1,125 selection-only validation, and 2,300 unseen-family OOD rows after filtering |
| [Google Taskmaster-1 WOz dialogues](https://github.com/google-research-datasets/Taskmaster/tree/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019) | 5,507 two-person dialogues | CC-BY-4.0 | 600 privacy-normalized, human-authored weak SAFE training windows and 450 disjoint selection windows; not counted as naturally occurring communications |
| [AWS MultiDoGO](https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/tree/baa30639c4b271f394b81443c842193407cdf26d) | 86,719 human-human service roleplays | CDLA-Permissive-1.0 | schema-v22 adds 1,790 weak-SAFE training views from 895 families and 1,184 controlled state derivatives; insurance/software states remain validation-only |
| [YouTube Scam Phone Call Transcripts v2](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts) | 243 partial transcripts / 222 source URLs | CC0-1.0 | 161 early windows in rejected schema-v14 ablation, 70 family-held selection windows, and 80 prediction-sealed OOD windows; counted only as real scam-call-derived language |
| [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | 873 spontaneous service-call roleplays | CC-BY-SA-4.0 | evaluation only: 348 open and 1,396 prediction-sealed SAFE windows split by shared-speaker/call components; no audio and zero fitting rows |
| [AZ-SC Azerbaijani SMS Collection](https://doi.org/10.25045/jpit.v17.i1.04) | 4,538 | CC-BY-4.0 | new multilingual OOD diagnostic only; never fitting or selection |
| ScamGuard synthetic v5 | 8,064 | Apache-2.0 | 70 base families plus 84 multilingual benign-lookalike families; original copy grounded in official advisories |
| ScamGuard synthetic dialogue v2 | 1,536 generated; 1,495 admitted | Apache-2.0 | 12 balanced five-turn scam/legitimate scenarios, training only |
| ScamGuard synthetic counterfactual v1 | 512 canonical; 128 in schema-v13 ablation | Apache-2.0 | eight balanced, paired error-driven families; training only |
| ScamGuard synthetic legitimate-call openings v1 | 1,024 generated; 256 in schema-v15 dose-16 | Apache-2.0 | 16 service scenarios × four opening structures; original SAFE dialogue without explicit anti-scam cues; training only |

## Additional external diagnostic

The CC0 YouTube scam-call corpus is positive-only and largely consists of scammer/scambaiter
interactions, with some autodialer messages. ScamGuard privacy-normalizes the publisher text,
removes existing-corpus exact/near overlap, connects source and near-template families, and then
partitions before materialization. The resulting 448 windows span 220 families: 298 train, 70 open
validation, and 80 sealed OOD. Schema v14 uses only the 161 early train windows so a long source call
is not double-weighted. That model raises open validation recall from 34.29% to 100% but is rejected
because unchanged-regression SAFE FPR reaches 8.48% and balanced-dialogue SAFE FPR reaches 73.20%.
Positive-only recall cannot establish precision or FPR. See
[`reports/DATASET_SCHEMA14_REAL_DIALOGUE.md`](../reports/DATASET_SCHEMA14_REAL_DIALOGUE.md).

The AppTek benchmark uses only the publisher's pinned diarization metadata and downloads no audio.
After schema checks, privacy normalization, overlap removal, and one-per-near-template collapse,
1,744 SAFE early/recent windows remain. Shared-call and shared-speaker edges form 77 components;
18 components (348 windows from 174 calls) are open for candidate selection and 59 components
(1,396 windows from 699 calls) remain prediction-sealed. AppTek is roleplay rather than customer
data, supplies no scam labels, and is explicitly evaluation-only. Schema v13 records 8.91% open
FPR and schema v14 records 22.13%; every false alarm is in an early-call window. See
[`reports/APPTEK_CALL_BENCHMARK.md`](../reports/APPTEK_CALL_BENCHMARK.md).

The CC-BY-4.0 [Chichewa SMS Fraud Classification dataset](https://doi.org/10.5281/zenodo.14607454)
contains 676 balanced Chichewa messages plus a 148-row legitimate telecom sheet. ScamGuard uses
only the original-language sheets, replaces live URLs and phone/account-like values, removes exact
duplicates, and keeps one representative per near-template family. The resulting named diagnostic
has 677 rows (315 SCAM and 362 SAFE) with zero exact or near overlap against existing processed
benchmarks. Its artifact SHA-256 is
`b429621d3508b9cb3f1a7c7f079603f10966cdcec81b339d5d176aa8c77b0b38`.

The source mixes collected and augmented messages without row-level provenance, so these rows are
not added to training and are not counted as 677 independently collected real messages. The human
and machine English translations are excluded because they would duplicate the same source IDs.
Native Chichewa review remains incomplete. The full local manifest is generated at
`data/external/chichewa/manifest.json`.

The Apache-2.0
[Synthetic Multi-Turn Scam and Non-Scam Phone Dialogue dataset](https://huggingface.co/datasets/BothBosu/scam-dialogue)
adds a deliberately different stress test: full caller/receiver conversations rather than isolated
SMS. The upstream 1,600 balanced, Llama-3-70B-generated dialogues become 1,343 rows (684 SCAM and
659 SAFE) after privacy normalization, conflict quarantine, near-template clustering, one-per-family
collapse, and overlap removal. A salted family hash assigns 294 rows to candidate selection and
keeps 1,049 rows prediction-sealed. Their artifact SHA-256 values are
`c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0` and
`33d480aa505f16014e18a7193f379b618e7a9feeb90262c93e77433c022c1193`, respectively.
It is external synthetic evidence only: it cannot increase the licensed-real count, fit a threshold,
or enter training. Earlier model-error analysis opened both partitions, so schema v22 treats the
artifact as a prior-open regression diagnostic rather than untouched evidence. Its local manifest
is generated at `data/external/scam_dialogue/manifest.json`.

Taskmaster-1 supplies the legitimate conversation shape missing from the original short-message
curriculum. Only its two-person Wizard-of-Oz subset is admitted. Each context is capped at complete
recent turns, and email, URL, and phone/account-like values are replaced before materialization. A
conversation-family hash is applied before sampling: 600 rows enter training and 450 rows remain
selection-only. The fitting and selection slices each preserve equal per-domain allocation. These
are human-authored transactional roleplays, not naturally occurring inbox or
call records and not independently scam-labelled, so manifests and size claims report them as a
separate provenance tier. The train and validation artifact hashes are
`c756886b2d97dc41d98956eb7dc468a7c43622e1861229b605768cedc78007c3` and
`539b81a06328b2914407565c1bb7fac54a486333cea45a0853b4e2160df79760`.

At model input, multi-party dialogue roles are mapped by first appearance to compact neutral labels
(`A:`, `B:`, and so on). The transform activates only for at least four recognized turns and two
distinct speakers, so ordinary short messages remain byte-for-byte unchanged. All 5,507 eligible
Taskmaster dialogues fit below 150 tokens after the complete-recent-turn cap.

MultiDoGO supplies licensed human-authored service dialogue across airline, fast food, finance,
insurance, media, and software. ScamGuard first filters source structure, then exact/near-template
clusters a deterministic 3,000-conversation audit pool per domain and retains one representative
per family. Each admitted conversation contributes a recent-context view and a frozen-lexicon
highest-risk agent-turn view. The original rows are weak SAFE roleplay, not naturally occurring
customer records. Four-state derivatives remain synthetic; their action-state train domains are
disjoint from held-out insurance and software. Whole families are removed if any view is near a
schema-v20 artifact. See
[`reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md`](../reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md).

## Sealed evaluation-only source

The newly sourced [MOZ-Smishing](https://doi.org/10.18653/v1/2025.africanlp-1.23) file contains
2,561 crowd-sourced Portuguese/Mozambican mobile-money messages. After source-wide privacy
normalization, exact/near conflict quarantine, one-per-family collapse, and overlap removal against
all prior processed benchmarks, `primary_test_v8` contains 1,820 still-unobserved rows: 526 SCAM and
1,294 SAFE. The source revision, hashes, denominators, and conflict counts are recorded in its local
manifest.

Hugging Face declares `creativeml-openrail-m`, but the repository has no dataset-specific license
file. ScamGuard therefore permits only local evaluation: this source cannot enter training and its
rows cannot be redistributed pending clarification. The raw and processed files remain Git-ignored;
the public project provides only the hash-pinned fetch and deterministic build path.

Mendeley `ham` maps to SAFE and `spam` to UNCERTAIN. Source-labeled smishing maps to SCAM only
when the text itself has a fraud pattern; low-risk ordinary commercial offers remain UNCERTAIN.
The external financial set is never used to train the model or select a threshold.
The WSPR source contains messages marked as phishing by VirusTotal/APWG. ScamGuard discards sender
and destination columns, collapses 68,029 reports using masked fields plus 64-bit character
SimHash near-template clustering, and maps obvious marketing/gambling copy without fraud evidence
to UNCERTAIN.
Families assigned outside training form a no-SAFE recall stress slice (409 SCAM and 79 UNCERTAIN)
and are never used for threshold selection or FPR claims.

The IMC 2025 artifact was created by researchers from reports on five public forums, including
Reddit and Twitter, and released as a CC-BY-4.0 research dataset. ScamGuard uses that published,
licensed artifact rather than scraping platform users directly. Only privacy-normalized message
text, language, and the paper's coarse scam label are considered; sender IDs, phone/network
metadata, timestamps, and named-entity metadata are discarded. Training admission is capped and
requires extractive evidence; harder remaining families are reserved for validation and OOD.
Generic spam, weak source-reported positives, and wrong-number openers without strong message-local
fraud evidence are `UNCERTAIN`.
Defensive scam education and standalone authentication-code notifications are SAFE unless the
message itself adds risky external routing; ambiguous defensive messages with that routing are
UNCERTAIN. The 1,125-row forum validation slice is used only to select training volume. A
deterministic learning curve reads no test or OOD file. The frozen 2,300-row forum OOD slice is
accompanied by 2,079 safely materialized derivatives that replace research placeholders with realistic `.example`
URLs and generic entities; neither derivative nor parent is used for fitting or selection.

The AZ-SC paper documents a hybrid collection of consented messages from 20 Azerbaijani users,
translated UCI messages, and self-generated instances. Its public rows do not identify which
provenance applies to each message. ScamGuard therefore does not train on AZ-SC and does not count
it toward licensed-real totals. It is admitted only as a hash-pinned Azerbaijani OOD diagnostic;
sender metadata is discarded, message text is privacy-normalized, generic spam and weak smishing
labels become UNCERTAIN, and near-template families are collapsed before scoring.

Synthetic v5 adds 12 scam scenario families and 12 paired legitimate lookalikes after the first
encoder exposed a government/identity-impersonation generalization gap. Each row records its
official FTC, FBI/IC3, IRS, or USPIS pattern reference and the deterministic generation method.
The messages are original slot-filled copy, not scraped or reproduced advisory text. All 1,728 new
rows are training-only; development, regression, and the sealed schema-v8 source supply independent
families.

Synthetic dialogue v2 generates 768 SCAM and 768 SAFE five-turn contexts across remote support,
government cases, refunds, banking, delivery, jobs, wrong-number grooming, insurance, tax
collection, family emergencies, investments, and marketplace overpayment. Every scenario is paired,
training-only, original copy, and grounded in an FTC, FBI/IC3, IRS, or USPIS advisory. Fail-closed
evidence, deduplication, and family-isolation checks admit 1,495 rows to the processed corpus.
The generator artifact SHA-256 is
`7f963f8557d9307e2ec605638ef4cccbc798a3414d774d42e31e9d1917fdceed`.

Synthetic counterfactual v1 contributes 256 SAFE and 256 SCAM messages across eight paired
families. It was designed from aggregate schema-v11 error clusters—not copied evaluation text—and
separates known-channel transfers from new-number payment requests, ordinary family updates from
private emergency-payment pressure, in-platform marketplace actions from refund manipulation, and
official-app review instructions from bank-review links. All rows are original deterministic slot
filling, training-only, carry extractive evidence when labelled SCAM, and pass the same family
isolation checks. The complete synthetic generator output (v5 plus this increment) has SHA-256
`3fdbdaa3719bb4ed316d77966106a7f58df17ca2e94069cbc96b6733bf16c7ec`; its manifest hash is
`beea78dce364e7ba100c66d2024c79c0fae8a9bcb66cef8c366da31bdd002f49`.

The isolated schema-v13 dose experiment keeps 16 examples per family instead: 64 SAFE plus 64 SCAM
rows. Its complete synthetic output hash is
`fc410f6250b50b1cc62c1500e4357817ed4a801c77c0e0a13e6132d2f51d5f51`. This ablation lives under
`data/experiments/schema13-dose16/`; it does not replace the canonical schema-v12 build or change
any development, regression, or sealed-source row.

Synthetic legitimate-call openings v1 contains 1,024 original SAFE four-turn dialogues in 64
scenario/structure families. It was designed from AppTek's open aggregate and metadata error slices,
not from copied benchmark text. It deliberately excludes explicit anti-scam phrases so the label
cannot be inferred from words such as “never ask” or “official app.” Schema v15 selects a balanced
256-row dose: 16 rows per service scenario and 64 per opening structure. AppTek contributes zero
fitting or threshold rows, and its OOD partition stays sealed.
The completed schema-v15 checkpoint is rejected: AppTek FPR remains 15.52%, unchanged-regression
FPR rises to 18.84%, and BothBosu records 69.50% recall / 35.29% FPR. Do not increase this generator
dose without changing the learning formulation and predeclaring a new experiment.

The full source-to-pattern and counterfactual methodology is in
[`reports/SYNTHETIC_DATA_V5.md`](../reports/SYNTHETIC_DATA_V5.md).

## Rejected or deferred sources

- TeleAntiFraud-28k: highest-priority real-call-derived dialogue candidate. Its Apache-2.0 dataset
  card describes 4,000/400 Chinese binary call records plus audio-grounded SFT data, but access is
  gated and the corpus mixes real-call ASR with augmented and synthetic construction. The pinned
  `make teleantifraud-fetch` and text-free `make teleantifraud-audit` workflow is ready; zero rows
  are admitted until authorized access, row-level provenance inspection, privacy normalization,
  family deduplication, and native review are complete.
- Direct Reddit scraping or API content: prohibited for model training without express
  rightsholder permission under the current
  [Reddit Data API Terms](https://redditinc.com/policies/data-api-terms). The separately published,
  CC-BY-4.0 IMC 2025 research artifact above is the only admitted forum-derived source.
- ES-Port: candidate authentic Spanish technical-support-call source under CC-BY-SA-3.0 Spain.
  It remains unadmitted pending residual-privacy, weak-label, language-shortcut, and share-alike
  review.
- Sting9: deferred until the downloadable artifact and license are both independently verifiable;
  its marketing page says CC0 while the governing legal page uses ODC-BY-NC and prohibits
  startup/product development and commercial model training.
- Unlicensed Kaggle mirrors, scraped inboxes, leaked messages, and vendor datasets without a
  redistribution/training grant.

Measured decisions for SmishX, SMISH_DT, SpamHunter, large generated GitHub files, Bengali data,
DIFrauD, COVA-X, and external ScamGuardBench are documented in
[`reports/ONLINE_SOURCE_RESEARCH.md`](../reports/ONLINE_SOURCE_RESEARCH.md).

## Build guarantees

`scripts/validate_dataset.py` fails on duplicate IDs, exact or SimHash-near cross-split text,
template-family leakage, missing provenance, missing SAFE/SCAM coverage, or PII-like values in
synthetic examples. Before family IDs or splits are assigned, every real source has email addresses,
long phone-like values, and long account-like digit sequences replaced with typed placeholders. The
validator independently rejects any real-source parent row that retains those patterns. The
manifest records exact/near duplicates, quarantined label conflicts,
source counts, and the largest discovered template cluster.
