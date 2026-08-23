# Schema 24 publisher-annotation baseline

## Decision

The pinned MultiDoGO paper splits are useful as a new, family-disjoint SAFE hard-negative
diagnostic, but they are not independently reviewed scam labels. Schema 24 admits only publisher
train families after privacy, exact-overlap, and SimHash controls. Publisher dev/test families stay
outside fitting and threshold selection. The full Qwen3.5-0.8B run remains blocked on the
independent human audit.

The initial implementation assumption that paper-split conversation IDs join to `data/unannotated`
was rejected by the real source audit: both turn- and sentence-level ID intersections are exactly
zero. Model rows are therefore selected directly from the publisher's turn-level customer splits;
sentence-level files are integrity-audit inputs only.

## Pinned source audit

- Repository: `awslabs/multi-domain-goal-oriented-dialogues-dataset`
- Revision: `baa30639c4b271f394b81443c842193407cdf26d`
- License: CDLA-Permissive-1.0
- Annotation tree Git object: `5add780ac12b5aaceab9b746786d64ae8d7b0c8e`
- Annotation files: 36 across six domains, three publisher splits, and two granularities
- Turn annotation conversations: 14,147
- Sentence annotation conversations: 14,215
- Usable turn rows: 116,520; quarantined empty turn rows: 77
- Common non-empty turn keys across granularities: 109,823
- Common keys with publisher text divergence: 117; sentence rows are never selected
- No conversation crosses train/dev/test within either granularity

The machine-readable, text-free source audit is
`reports/data/multidogo_annotation_audit.json`.

## Bounded curriculum and schema 24

The curriculum chooses at most one eligible turn per conversation, ranks publisher intent/slot
concepts before routine controls, balances domains, excludes rows carrying publisher PII slot
types, and normalizes email, URL, phone/account-like, and three-plus-digit values before IDs or
materialization.

| Publisher split | Pre-overlap rows | Schema 24 rows | Role |
|---|---:|---:|---|
| Train | 1,200 | 1,095 | fitting only after parent and held-reference overlap removal |
| Dev | 540 | 506 | selection diagnostic only |
| Test | 1,200 | 1,199 | held diagnostic only |

The training increment contains 1,095 unique families. Contextual privacy normalization masks
access codes, account fragments, postal codes, and credential-like values before contamination
control. It changed 201 of 44,512 processed rows (220 typed replacements) and removed four parent
training families/20 rows whose redacted forms created new held-reference collisions. Schema 24
therefore contains 23,740 fitting rows in total. The repository validator passed 43,591 unique
model/evaluation rows with no family leakage; the existing 1,820-row primary holdout remained
prediction-sealed.

## Frozen schema-23 baseline on the new held slices

The rejected schema-23 ModernBERT checkpoint was scored only to establish the pre-experiment
baseline. Its frozen temperature and 0.249011 scam threshold were unchanged. No schema-24 row was
used for fitting or threshold selection.

| Slice | SAFE rows | False positives | SAFE FPR | 95% Wilson interval |
|---|---:|---:|---:|---:|
| Publisher dev after overlap control | 506 | 43 | 8.50% | 6.37%-11.25% |
| Publisher test after overlap control | 1,199 | 101 | 8.42% | 6.98%-10.13% |

Test SAFE FPR by domain was airline 0.50%, fast-food 0.50%, finance 10.50%, insurance 6.50%,
media 28.00%, and software 4.52%. This localizes the false-positive gap instead of averaging it
away. The schema-24 challenger must achieve at most 2% overall SAFE FPR and at most 3% in every
domain on these publisher-held slices while retaining the existing scam-recall and external
dialogue gates.

Artifact identities:

- Publisher dev JSONL SHA-256:
  `da4057458dd04225720962675c951be246e667ab8761dcf71e17f7879fd3fdd4`
- Publisher test JSONL SHA-256:
  `92bb0e8df4b1a9ef76a2b0c88afd76068f954ce31ac5f7fe2ebbb9e9dc6c9a70`
- Schema-23 model SHA-256:
  `20f9287fb5c0fff238d0a64710d6bb1557a94ce75afc9b9d7f02ca2b29febc57`

## Qwen SFT preflight

The full Qwen SFT builder retains 23,450 train and 2,634 dev examples. It excludes 290 inherited
positive-only YouTube scam-call windows whose text does not support the runtime's required verbatim
evidence; they are not relabelled. The 640-token audit covers all 26,084 retained examples with p95
545, p99 554, maximum 572, and zero truncation. The larger ceiling preserves decisive
long-dialogue actions and does not change the separate under-20-ms fast-path requirement.

The deterministic independent audit workbook contains 635 rows and is currently 0/635 complete.
Its immutable sample IDs and privacy-normalized inputs are hash-bound under audit-manifest schema 2.
Review protocol v1 defines SAFE, UNCERTAIN, SCAM, and sensitive-data decisions in the localhost-only
blind UI; the protocol SHA-256 is
`d9dcc931447ce5229ca5e07398b20944759dfe0f0224369c14c2729be10cbb59`. The completion report will
record percent agreement, a 95% Wilson lower bound, Cohen's kappa, a three-class confusion matrix,
and per-source/per-project-label agreement without copying message text. The experiment freezer and
Hugging Face release checker both reject a missing, incomplete, or differently versioned protocol.
Training, external selection, quantization, and Hugging Face publication remain unauthorized until
that audit passes and all downstream quality/release gates succeed.
