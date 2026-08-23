# Data governance

## Admission checklist

A source enters training only when all answers are documented:

1. Is the artifact directly obtainable from its claimed publisher?
2. Does its license explicitly permit the planned training and redistribution?
3. Are the download URL, version, SHA-256 digest, citation, and label mapping pinned?
4. Was consent or public-release provenance documented, and was sensitive PII removed?
5. Can exact and near-template duplicates be isolated before splitting?
6. Is there an audit sample for label ambiguity and shortcut artifacts?

Failure or ambiguity means quarantine, not optimistic inclusion.

## Conservative label mapping

Source labels are evidence, not automatic product truth. ScamGuard requires the message text to
support a fraud conclusion. Generic spam, ordinary commercial offers, and evidence-free
wrong-number openers are `UNCERTAIN`. Defensive scam education and standalone OTP/authentication
notifications are `SAFE` when they contain no risky external action. If defensive wording also
routes the reader to a link, phone number, off-platform contact, credential entry, or payment, the
row is `UNCERTAIN` unless the fraud evidence is unambiguous. The processed row records the applied
`label_policy`; no SCAM row may retain a defensive/authentication/commercial correction policy.

This rule was added after a stratified audit caught source-level false positives. The affected
schema-v3 Qwen run was stopped at update 60 and is not a valid result. A later schema-v4 audit
caught an overbroad bare-domain detector (`400sq.mtr` was incorrectly treated as a URL) and
source-reported positives that lacked message-local fraud evidence. The schema-v4 ModernBERT run
was stopped before completion; schema-v4 Qwen training never started. Schema v5 uses a curated TLD
allowlist, recognizes defanged and scheme-qualified URLs, and requires strong message-local fraud
evidence for every SCAM mapping from Mendeley, WSPR, and forum sources. All data hashes, learning
curves, baselines, supervision, and experiment configurations were rebuilt from scratch.

Schema v6 followed when a source-wide privacy scan found public email and long phone/account-like
values outside the already-normalized forum artifact. The schema-v5 ModernBERT rerun was stopped
and rejected. Schema v6 replaces those values with typed placeholders before IDs,
deduplication, clustering, or splitting, then independently fails validation if a real-source row
retains one of the prohibited patterns. This also reduces contact-value memorization as a shortcut.

Schema v12 introduced explicit generation methods and authoritative pattern-reference URLs on
synthetic rows. The frozen schema-v22 training experiment contains 8,134 naturally occurring
licensed-source or real-scam-call-derived rows, 2,983 human-authored roleplays, and 13,091
controlled synthetic rows. The roleplay tier is never described as naturally occurring
communication or independently scam-labelled. Taskmaster and MultiDoGO original dialogues have
weak legitimate-domain SAFE labels; every matched state transformation remains synthetic in all
accounting. Schema v22 starts from the safer schema-v20 parent after schema v21's full-weight
HarperValleyBank dose was rejected for regression and call false positives.

HarperValleyBank is pinned by Git revision, license hash, complete transcript-tree hash, and
complete metadata-tree hash. Only transcripts and metadata are acquired. Six entire banking tasks
and 1,069 calls enter training; two other tasks and 377 calls remain validation-only. Original and
transformed versions stay in the same call family. See
`reports/DATASET_SCHEMA21_HUMAN_CALLS.md` for the frozen hashes, counts, and experiment gates.

MultiDoGO is pinned by Git revision, license hash, and the combined SHA-256 of six raw domain
files. A deterministic per-domain audit pool is exact/near-template clustered before sampling;
one representative per family survives. The schema-v22 state curriculum trains on four service
domains while insurance and software remain validation-only. Original rows and derived states
stay in the same conversation family, and any family with a near match to a schema-v20 artifact is
removed in full. See `reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md` for the frozen contract.

A newly sourced 1,820-row schema-v8 holdout remains prediction-sealed and excluded from training
and public redistribution. The 677-row Chichewa and 1,343-row BothBosu multi-turn artifacts are
external diagnostics. The latter is entirely synthetic and was opened during earlier model
diagnosis; schema v22 therefore records it as a prior-open regression diagnostic, not untouched
evidence. Neither increases the licensed-real total.

## Near-template isolation

Exact normalized-text deduplication runs first. Real messages are then masked for URLs, emails,
numbers, mixed alphanumeric tokens, and known campaign-tracking suffixes. ScamGuard computes a
64-bit character-4-gram SimHash and joins candidates at Hamming distance 6 or less. Every joined
cluster receives one split. A cluster containing conflicting source labels is quarantined in full;
WSPR contributes only one deterministic row per final cluster.

Candidate retrieval uses seven disjoint bit bands. Because a pair with at most six differing bits
must share at least one exact band, this index is complete for the declared radius; the final
Hamming calculation still decides whether to join the pair.

`scripts/validate_dataset.py` independently recomputes this relation across development splits and
the WSPR/forum holdouts and forum validation slice. This is deliberately stricter than trusting the
family IDs written by the build.

## Synthetic data

Synthetic rows must use fictional entities, `.example` or visibly defanged URLs, no functional
credentials, and a generator version/template-family ID. Paraphrases of one template stay in one
split. Hard negatives deliberately include legitimate OTP alerts, payment notices, deadlines,
gift-card mentions, and security education so keyword matching is punished.

Synthetic v5 contains 5,040 original English scenario variants and 3,024 benign lookalikes in
Spanish, Dutch, French, German, Italian, Indonesian, and
Portuguese. Each translated template has curated language-specific wording and values; related
variants remain in one split. The purpose is counterfactual balance, not inflating a claim with
machine-translated copies of evaluation messages. Native-speaker review is still required before a
production release. Version 5 adds paired train-only families derived from FTC, FBI IC3, IRS, and
USPIS pattern advisories; it does not copy advisory prose or inject synthetic rows into evaluation.

Synthetic counterfactual v1 is a separate 512-row, balanced train-only repair set. Its eight paired
families were designed from aggregate schema-v11 error categories; no development, regression, or
external-diagnostic message is copied. SCAM rows must retain extractive risk evidence, SAFE rows
must preserve a legitimate trust boundary, and every family remains isolated from evaluation.

Synthetic dialogue v2 generates 768 SCAM and 768 SAFE five-turn examples across 12 paired
scenarios; 1,495 survive the ordinary evidence, deduplication, and family-isolation gates. It was
introduced only after independent dialogue selection slices exposed both high false positives and
a later provenance-format shortcut. All rows are original training-only copy, use fictional
`.example` endpoints, and share neutral discourse variants across both labels. A versioned
speaker-neutral input transform removes corpus-specific role labels before tokenization.

## Review and release

Before a model release, audit random and high-loss samples from every source/category, document
class-conditional label error, run PII/secrets scans, and publish a model card. Dataset licenses do
not automatically grant rights to every downstream model distribution; review that separately.
For schema v24, give the reviewer only the answer-key-free artifact produced by
`make schema24-audit-bundle`; never give them the repository or canonical workbook. The returned
CSV contains the independent label and sensitive-data decision but no project correctness field.
`make schema24-audit-import` verifies its immutable inputs and derives correctness only after
joining it to the sealed canonical workbook. `make schema24-audit-check` fails closed until every
stratified row has a valid decision, with zero disagreements and zero sensitive-data findings.
