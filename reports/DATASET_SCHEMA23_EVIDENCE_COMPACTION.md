# ScamGuard dataset schema v23: evidence compaction

## Decision

Schema v23 is **frozen for one predeclared quality-teacher experiment; no model result is included
in this document**. It starts from schema v20 rather than either rejected full-dose service
ablation. The change is deliberately small: a participant-balanced, bounded dialogue input
contract; 216 licensed human MultiDoGO agent turns; 864 human-grounded action-state rows; and 356
original FTC-pattern-grounded action-state rows.

The training set has **22,670 unique rows**. Row count does not determine inference latency. The
under-20-ms target is gated separately on the exported runtime, sequence length, quantization, and
hardware after every quality gate passes.

## Why schema v23 exists

Schema v22 showed that adding broad service dialogue could suppress false alarms while collapsing
scam recall. It passed 20 of 29 gates, reached 0.63% regression FPR and 1.96% prior-open BothBosu
FPR, but only 41.13% BothBosu recall. Schema v23 therefore does not add another bulk dose. It
changes what the 256-token encoder sees and adds bounded action supervision around the exact
distinctions that previously disappeared outside the token window.

The `speaker-neutral-evidence-recent-v2` transform activates only for dialogues with at least four
recognized turns and two recognized speakers. It emits at most 1,400 characters in one encoder
pass:

- up to three participant-balanced service-action, scam-action, or explicit safety-boundary turns;
- up to three non-duplicate recent turns;
- neutral `A:`, `B:` speaker identities assigned by first appearance;
- no verdict, dataset identifier, scenario name, or benchmark label in model input.

Preflight verifies that the changing action remains present both after compaction and after the
actual ModernBERT tokenizer applies the frozen 256-token right-truncation policy.

## Final composition

| Data class | Rows | Accounting rule |
|---|---:|---|
| Naturally occurring licensed messages or real-scam-call-derived language | 8,134 | Preserved schema-v20 tier |
| Human-authored Taskmaster roleplay | 1,193 | Preserved licensed weak SAFE roleplay |
| Human-authored MultiDoGO agent turns | 216 | One action-focused view per admitted fitting family; weak verdict weight 0.25 |
| Controlled synthetic rows | 13,127 | Preserved generators plus grounded state increments |
| **Total** | **22,670** | **9,543 licensed non-synthetic + 13,127 controlled synthetic** |

There are 7,580 action-supervised fitting rows. Synthetic action-state rows use the frozen default
verdict weight 0.25. The 216 original MultiDoGO turns also receive verdict weight 0.25 so their
licensed action language improves the auxiliary representation without becoming a dominant SAFE
shortcut.

## MultiDoGO fitting and calibration

The upstream source remains pinned to AWS MultiDoGO revision
`baa30639c4b271f394b81443c842193407cdf26d` under CDLA-Permissive-1.0. Schema v23 uses only the
existing privacy-normalized, family-clustered source artifact:

| Partition | State rows | Original human turns | Families | Role |
|---|---:|---:|---:|---|
| Fitting | 864 | 216 | 216 | model weights |
| Action calibration | 320 | 80 | 80 | seven auxiliary thresholds only |
| Held-domain state validation | 592 | 0 | 148 | insurance/software quality gates |
| Original-call validation | 0 | 896 | 448 | domain SAFE-FPR gates |

The 80 calibration families are disjoint from fitting and validation. They fit only the seven
action thresholds; the three-way verdict threshold remains fitted exclusively on the unchanged
development split.

Each schema-v23 MultiDoGO state retains the complete common prefix, the changing agent action, and
two identical later pause/check turns. This is a deterministic transformation of the synthetic
state derivative, not the licensed source artifact. Every output row stores the SHA-256 of its
longer source-state text. It prevents an auxiliary action label from supervising words that are no
longer observable at the mobile token limit while still testing delayed evidence.

## FTC-pattern-grounded original data

The generator uses the FTC's [Robocall Scam Examples](https://consumer.ftc.gov/features/robocall-scam-examples)
and individual FTC pattern pages to choose ten mechanisms: marketplace charge, legal-enforcement,
border-package, technical-support, tax-debt, Social Security suspension, interest-reduction,
auto-warranty, student-loan-relief, and utility-rebate.

No robocall audio, transcript, or third-party wording is copied. The FTC's
[Website Policy](https://www.ftc.gov/policy-notices/website-policy) says most FTC-authored work is
public domain while identifying possible third-party material. ScamGuard therefore treats the
pages only as pattern evidence and releases every generated line as original Apache-2.0 project
copy.

The source generator creates 480 rows: ten patterns, three dialogue structures, four variants,
and four action states. Student-loan-relief and utility-rebate are scenario-held from fitting.
Before admission, every candidate is checked for exact and SimHash-near overlap against all 37,408
schema-v20 artifacts and all 1,343 prior-open BothBosu artifacts. Any match removes the complete
four-state family.

| FTC artifact | Rows | Families | Role |
|---|---:|---:|---|
| Generated source | 480 | 120 | pre-control source |
| Removed by whole-family overlap control | 40 | 10 | never admitted |
| Fitting | 356 | 89 | eight pattern types |
| Held pattern validation | 84 | 21 | student-loan and utility only |

The final recheck finds zero exact or Hamming-radius-six near overlaps across the 440 admitted rows.
The pre-control audit is retained in `reports/data/schema23_ftc_pattern_overlap.json`; it must not be
misread as the final admitted overlap result.

## Internet-source decision

Schema v23 directly scrapes **zero Reddit rows**. The parent corpus already contains a bounded,
privacy-normalized slice of the CC-BY-4.0 IMC 2025 research release derived from public forums,
including Reddit. That licensed research artifact—not raw platform posts—is the forum-derived
source.

TeleAntiFraud remains the highest-priority real-call-derived candidate, but its Apache-2.0 Hub
repository returned the publisher's gated 401 response on this machine. No mirror or access bypass
is permitted and zero rows are used. Sting9 is excluded because its governing ODC-BY-NC terms
conflict with its CC0 marketing claim and prohibit product/commercial model use. Repositories of
scammer emails without an explicit content license remain excluded.

## Frozen teacher and gates

The teacher is the rejected-but-informative schema-v20 ModernBERT-base checkpoint, used only as a
starting point and text-free verdict-retention anchor. The cache contains 21,234 IDs and the first
three verdict logits only under the exact v2 input contract. It contains no source text and no
auxiliary target logits.

The frozen run uses one epoch, batch size 16, 1,417 optimizer steps, learning rate 2e-6, right
truncation at 256 tokens, retention weight 4.0 at temperature 2.0, action loss weight 0.5, default
action verdict weight 0.25, and seed 20260821. No threshold is fitted on BothBosu, FTC holdout,
MultiDoGO held domains, external selection, or sealed data.

The candidate must pass every predeclared development, unchanged-regression, original-state,
FTC-pattern, MultiDoGO-call, MultiDoGO-state, Taskmaster, long-call, and prior-open BothBosu gate.
Only then may AppTek/YouTube selection, distillation, quantization, export, or sealed evaluation run.
Core ML desktop end-to-end p95 must be at most 20 ms; a physical-device measurement is mandatory
before any mobile claim.

## Preflight result and identities

Preflight passed with no failures:

- 22,670 fitting rows, including 9,543 licensed non-synthetic and 13,127 synthetic rows;
- 3,944 action-state rows checked through compaction, tokenization, and truncation;
- zero omitted or out-of-window decisive turns;
- zero exact or radius-six near overlaps for all 440 admitted FTC rows against 38,751 references;
- 21,234 text-free teacher anchors bound to the exact parent file, checkpoint, and v2 policy;
- all family partitions, source counts, hashes, frozen hyperparameters, and sealed-data declarations
  intact.

Frozen SHA-256 identities:

- FTC source manifest: `540971802435f94e152cebb38da54d1002f9e47135027fb797003f2e8d586a29`
- processed schema-v23 manifest: `d29700e51f974dbe4d066247e7422c2a7e0a29c8ea5de895f806acc0ce8f220f`
- training JSONL: `69421ca073dd2238073bf761e04f295c4d60dc7cb7de2a2360cbd795d36602d2`
- action-calibration JSONL: `fdcd2207b5e70335db7e09ed5e2b20e4bae6462a97c285828d2d73ab6eb33b48`
- held MultiDoGO states: `cf22e9c86de686389b13a9a2e3667fb9d20ab8aab4bc05a042c400d58b5b9a3a`
- teacher ledger: `c4fd11481072d8cf42375c2c33d9e993822128c6d6ebbcf3e76d27ffc7f1b726`
- teacher manifest: `ac28494d34e7c05c0cbc75d6d7353be85ed4451962b368553e4b7bf397c25010`
- experiment configuration: `835cb138eafd15db68a15bd4ff942f5bfaef27cf5bcdee699e15f470f162dc30`

The frozen configuration is
`configs/encoder-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right.json`. The machine-readable
preflight result is `reports/data/schema23_preflight.json`.
