# Data audit status

Current schema-v12 update: automated validation passes 28,125 unique processed examples with no
family leakage. Training has 14,446 rows; development (2,634) and regression (2,374) remain
byte-identical to schema v9. The 23,798-row core and named processed diagnostics now distinguish
13,127 naturally occurring licensed-source rows, 600 human-authored Taskmaster roleplays, and
10,071 controlled synthetic rows. See `reports/DATASET_SCHEMA12.md` for the increment and hashes;
`reports/DATASET_SCHEMA11.md` retains the preceding dialogue correction and token audit. Independent
label review is still incomplete, so this automated pass is not a production or SOTA
certification.
The regenerated source/label-stratified workbook contains 240 rows, SHA-256
`f7ea70711429d1b68f1676271c11d5d54b134ccb627b0644197017bbdd10f8ea`; 0/240 independent
decisions are complete.

## Historical schema-v9 audit snapshot

Snapshot: 2026-08-21. ScamBench schema v9 contains 25,518 unique parent examples. The 21,191-row
core and named diagnostics contain 13,127 licensed real-source rows and 8,064 synthetic rows; a
4,327-row Azerbaijani external diagnostic is separate. Frozen split sizes are: train 11,839,
dev 2,634, regression test 2,374, financial OOD 431, WSPR OOD 488, forum validation 1,125, and forum
OOD 2,300. The Azerbaijani slice contains 204 SCAM, 2,963 SAFE, and 1,160 UNCERTAIN rows and is
never used for training, calibration, or model selection. ScamGuard synthetic v5 contributes
5,040 base scam/lookalike cases and 3,024 benign lookalikes across Spanish, Dutch, French, German,
Italian, Indonesian, and Portuguese.

## Automated validation

The fail-closed validator independently checks required/non-empty fields, label/category
consistency, allowed licenses, boolean synthetic flags, exact overlap, family isolation, 64-bit
SimHash near-template overlap through a radius-complete seven-band candidate index among
development, forum validation, and OOD, synthetic and real-source PII-like values, core-category
coverage, and manifest count/label/source reconciliation. The frozen snapshot passes all checks
across 25,518 unique parent examples with no family leakage and zero unmasked email, long
phone-like, or 10-plus-digit account-like values.

The adversarial slice contains 320 deterministic held-out derivatives, including 64 explicit
prompt-injection attacks as well as zero-width, Unicode-homoglyph, SMS-noise, and punctuation
transformations. A separate 2,079-row forum derivative replaces research placeholders with safe,
realistic-looking values to expose placeholder shortcuts. Parent/output hashes are recorded in
sidecar manifests; neither derivative is used for training or selection.

Every SCAM row in train, dev, and test produces at least one verbatim deterministic evidence span.
This is a build gate: Qwen supervision rejects a SCAM target with empty evidence. Evidence coverage
is intentionally lower on noisy/non-generative slices: financial OOD 77.4% (202/261) and
materialized forum OOD 89.4% (1,665/1,862). Admitted forum validation/OOD SCAM parents now have
100% message-local evidence by construction; weak source-reported cases are UNCERTAIN.

The reproducible audit command selects 12 rows from every available source/label stratum. The
current source set produces 240 rows in the ignored local workbook `data/audit/label_audit.csv`.
It includes blank fields for an independent auditor label, correctness decision,
sensitive-data flag, and notes.
Its schema-v9 SHA-256 is
`cd1d101c2f57e568acc43427cfed44dafad809eceae908fa9c5534c72bb644b7`.

## Findings applied during build

- Schema v9 adds 1,728 training-only synthetic examples in 24 paired families after the first
  encoder exposed a development identity-impersonation gap. The new original copy covers official
  FTC, FBI/IC3, IRS, and USPIS scam mechanics and legitimate-channel controls. Every row records
  its pattern reference and generation method. No development, regression, or sealed-source message
  text was added to training. The validator initially rejected 86 rows without an extractable risk
  signal; the copy was corrected to make the harmful secrecy or verification-code request explicit,
  and the full corpus then passed without weakening the gate.

- Schema v8 adds a separate, newly sourced and still-sealed MOZ-Smishing robustness test. Its 2,561
  crowd-sourced Portuguese/Mozambican mobile-money messages yield 1,820 family representatives:
  526 SCAM and 1,294 SAFE. The build normalizes 1,103 rows with phone-like strings, 94 with long
  digit sequences, and two with email addresses before IDs or comparison; removes 34 exact
  duplicate/conflict rows; quarantines five exact and two near-template mixed-label groups;
  collapses 699 same-label near-template repeats; and removes four representative families with
  near overlap against every previously processed benchmark. Its SHA-256 is
  `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` and no model predictions
  have been run. The publisher's `creativeml-openrail-m` tag lacks a dataset-specific license file,
  so project policy permits local evaluation only: no training and no row redistribution pending
  clarification.
- Schema v7 adds the AZ-SC Azerbaijani collection only as external OOD. The official paper describes
  a mixture of translated UCI messages, consented messages from 20 users, and author-generated
  examples, but the published CSV has no per-row provenance. ScamGuard therefore excludes every
  AZ-SC row from licensed-real counts and fitting. Sender fields are discarded; message text is
  privacy-normalized; ham maps to SAFE; generic spam maps to UNCERTAIN; and only smishing with
  message-local evidence maps to SCAM. The build removes 36 exact duplicates, quarantines one exact
  label conflict and 79 rows in 10 near-template conflict clusters, retains 4,334 representatives,
  then removes five dev near-overlaps. Source commit
  `f3ebfa36103fb71731cc984a00f1e648c4a5dc8d` and raw CSV SHA-256
  `3ffaf4d38daa7e9fd1dcf0b292ae12a8c73eca8261b8ced89788241f9216acbf` are pinned.

- A pre-release v0.3 draft used four exact 16-bit SimHash lookup bands. Code review showed that
  this candidate index could miss a Hamming-distance-six pair when differences touched every band.
  No Qwen result from that draft was accepted: its runs were stopped at 30 of 973 updates or
  earlier. The corpus was rebuilt with seven disjoint bands, which is complete for radius six by
  the pigeonhole principle. This quarantined 109 additional conflict rows, reduced WSPR to 2,422
  independent representatives, and produced the hashes frozen in `configs/qwen35-2b-lora.json`.
  The 0/1k/3k/6k cap selection and all dependent audits were rerun before the now-rejected
  schema-v3 Qwen launch.
- A later schema-v3 Qwen3.5-2B run was stopped at update 60 of 968 after the 180-row stratified
  audit exposed false-positive supervision: anti-scam education, ordinary one-time-code alerts,
  and benign commercial offers had inherited source-level SCAM labels. No checkpoint or score from
  that run is accepted. Schema v4 applies a conservative text-local policy: defensive guidance and
  standalone authentication notifications are SAFE unless the same text adds a risky external
  action; defensive guidance with external routing and low-evidence commercial/spam cases are
  UNCERTAIN. After the schema-v6 rebuild, fitting and evaluation slices retain 186
  defensive-guidance and 87 authentication-notification SAFE controls. Zero processed SCAM rows
  carry those corrected label policies. All dependent datasets, cap curves, Qwen supervision,
  token audits, baselines, hashes, and immutable experiment configurations were rebuilt.
- A schema-v4 supervision audit then exposed a subtler shortcut: the signal extractor treated any
  dotted suffix as a URL, so a property-ad measurement such as `400sq.mtr` became false
  `suspicious_link` evidence. Tightening URL recognition to a curated web-TLD allowlist revealed
  source-reported SCAM rows with no strong message-local fraud cue. The schema-v4 ModernBERT run was
  stopped before completion and no score/artifact is accepted; schema-v4 Qwen training never
  started. Schema v5 requires strong text evidence for Mendeley smishing, WSPR phishing, and IMC
  forum SCAM labels; weak source-only positives are UNCERTAIN. It also recognizes rare real scam
  domains, defanged URLs, multilingual link instructions, APK-download requests, and locked-account
  coercion without reopening the arbitrary-suffix shortcut.
- A source-wide privacy scan then found public email addresses and hundreds of long phone/account-like
  values in Mendeley, UCI, and WSPR rows outside the already-normalized forum path. The schema-v5
  ModernBERT rerun was stopped at 59% and no score or artifact is accepted. Schema v6 replaces email,
  long phone-like, and 10-plus-digit account-like values with typed placeholders before IDs,
  deduplication, family clustering, or splitting. An independent validator now rejects any such
  surviving pattern in every real-source parent row. All data, curves, supervision, configurations,
  baselines, and references were invalidated and rebuilt again.
- A WSPR sample exposed repeated phishing campaigns with random trailing tracking tokens. Exact
  masked-template hashing did not join every variant, so ScamBench v0.2 added 64-bit character
  SimHash clustering and an independent cross-split validator.
- Obvious gambling, property-event, and opt-out marketing messages in the no-SAFE WSPR source
  are mapped to UNCERTAIN unless the text also contains fraud evidence.
- WSPR sender and destination-number fields are discarded; only the message is admitted.
- The IMC artifact contributes licensed reports derived from five public forums without a direct
  platform scrape. Sender/network/time/named-entity metadata is discarded and residual contact
  values are masked. Generic spam and evidence-free wrong-number openers map to UNCERTAIN. The
  fitting slice admits 148 real forum SAFE hard negatives; selection-only forum validation has 25
  SAFE rows and forum OOD has 100, so their FPR denominators are reported rather than hidden.
- The current build quarantines 3,742 rows in 634 near-template label-conflict clusters. It starts
  from 33,200 real rows in 26,386 near-template clusters; the largest cluster has 260 rows. The
  forum path removes 9,895 exact duplicates, retains 17,918 representatives, and removes 140
  cross-source near-overlaps. WSPR retains 2,401 representatives after removing 916 near-template
  repeats. Two
  near-template overlaps are removed from the financial OOD slice.
- A development-only 0/1k/3k/all-5,672 learning curve never opens test/OOD files. The lexical proxy
  cannot clear the release gate; its quality-first diagnostic recommends 1,000 forum SCAM and 100
  forum UNCERTAIN rows because this candidate gives the best forum-validation recall (91.9%) while
  larger 3k/all candidates fall to 90.6%/90.9%. The 148 forum SAFE rows stay fixed in every
  candidate. The validation FPR result (1/25) is too small a denominator for a release claim. The
  final Qwen family must confirm this choice.
- The corpus is multilingual and intentionally preserves some noisy encodings and truncated gateway
  captures. These need source-stratified error analysis rather than silent cleaning.
- The financial OOD source labels some semantically ambiguous messages as scams, including ordinary
  promotions and urgent personal repayment requests in the deterministic audit sample. This slice
  remains source-faithful and diagnostic only; it is not a product release gate.

## Remaining release gates

The current 240-row workbook has not yet received independent human labels. ScamGuard may be published as an
experiment with this limitation, but a production/SOTA model card must not call the dataset
human-audited until those fields are completed and agreement/error rates are reported. A second
audit sample should be drawn from the highest-loss training and evaluation rows after model fitting.
The multilingual synthetic hard-negative families also require native-speaker review before a
production or SOTA dataset claim. `make audit-check` currently fails as intended: 0 of 240
independent audit decisions are complete.

The schema-v6 test was opened repeatedly during Q4/Q5 deployment-quantization escalation. It
remains valid regression evidence for those frozen artifacts, but it is no longer an untouched
final-selection set. Schema v8 now supplies a newly sourced, family-isolated and model-unobserved
local holdout. It must remain sealed until the next candidate set and decision rule are frozen;
opening it and then changing the model retires it to regression-only status as well. Production and
SOTA claims still require the unfinished independent label audit and license clarification.
