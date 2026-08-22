# ScamGuard dataset schema v22: service evidence

## Decision

Schema v22 was frozen for one quality-teacher experiment and has now been **rejected**. It returned
to the safer schema-v20 parent and adds a bounded amount of licensed human-authored service
dialogue plus human-grounded four-state action evidence. It does not inherit schema v21's
over-weighted bank-call dose.

The training set has **24,208 unique rows**. This is the right scale for the experiment: enough to
add 895 service-conversation families and 296 grounded state families without drowning the
14,062-row retained teacher ledger or changing the established development and regression sets.
Training-row count does not determine inference latency. The under-20-ms deployment target is a
separate gate on sequence length, architecture, runtime, quantization, and hardware after quality
passes.

The trained checkpoint passed 20 of 29 predeclared gates. It restored unchanged-regression FPR to
0.63% and cut prior-open BothBosu FPR to 1.96%, but BothBosu recall fell to 41.13%; held-domain
harmful recall, routine-state FPR, action exact match, and several original-call domain FPR gates
also failed. No distillation, export, external selection, or sealed evaluation was authorized. See
`reports/ENCODER_SCHEMA22_SERVICE_EVIDENCE.md` for the complete measured result.

## New licensed source

- Publisher: <https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset>
- Paper: <https://aclanthology.org/D19-1460/>
- License: CDLA-Permissive-1.0
- Pinned revision: `baa30639c4b271f394b81443c842193407cdf26d`
- Pinned dialogue-tree SHA-256:
  `0196e5ae82fc3b8c488b82d0a3cdf8dca74911a8bab5fa5ed5e1bf6ceee2ae97`
- Pinned license SHA-256:
  `8be8b09ba4230a6ab89a62439b45e3374e15870360d27c9a51131592b91a2f10`
- Domains: airline, fast food, finance, insurance, media, and software
- Acquisition: six unannotated text files, README, notice, and license only; no audio

MultiDoGO contains human-customer and trained-human-agent roleplay. The selected rows are real
human-authored dialogue, but not naturally occurring customer calls and not independently
scam-labelled truth. Original rows therefore receive a weak SAFE label from the legitimate service
domain. This provenance tier remains separate from naturally occurring messages and real
scam-call-derived transcripts.

The source contains 86,719 conversations; 84,129 pass the frozen turn-count, speaker, repetition,
word-count, and alternation filters. Sampling does not start from all rows. It first creates a
deterministic 3,000-conversation audit pool per domain, removes 810 exact duplicates, clusters
near-template families at SimHash Hamming distance six, and retains one conversation per family.
This collapses 17,190 candidates to 3,485 representatives before train/validation selection.

## Final composition

| Data class | Rows | Accounting rule |
|---|---:|---|
| Naturally occurring licensed messages or real-scam-call-derived text | 8,134 | Preserved schema-v20 tier |
| Taskmaster human-authored roleplay | 1,193 | Licensed weak SAFE roleplay |
| MultiDoGO human-authored roleplay | 1,790 | New licensed weak SAFE roleplay; two views from 895 families |
| Existing controlled synthetic rows | 11,907 | Preserved schema-v20 generators and long action states |
| New MultiDoGO-grounded state rows | 1,184 | Synthetic derivatives from 296 source families |
| **Total** | **24,208** | **11,117 licensed non-synthetic + 13,091 controlled synthetic** |

There are 9,118 action-supervised training rows. Original MultiDoGO rows use a 0.5 verdict weight;
generated action-state rows use the frozen 0.25 default. This is a deliberate middle dose between
schema v20's auxiliary state supervision and schema v21's full-weight original-call treatment.

Each selected conversation contributes two distinct real views: recent complete turns and the
highest-risk agent turn chosen by a frozen lexical rule. Email, URL, phone/account-like, and mixed
long alphanumeric values are replaced before materialization. Exact text is unique across all new
train and validation artifacts.

## Independent partitions

| Split | Real SAFE rows | Grounded state rows | Real families | State families |
|---|---:|---:|---:|---:|
| Training | 1,790 | 1,184 | 895 | 296 |
| Validation | 896 | 592 | 448 | 148 |

Conversation families never cross the split. State-domain generalization is stricter: action
states train only on airline, fast food, finance, and media. All insurance and software states are
held out for validation. The validation states contain 148 harmful SCAM, 148 unresolved UNCERTAIN,
and 296 SAFE rows.

Every state family changes one decisive agent action while retaining the human-authored context
and an identical later hesitation/verification continuation:

- routine service action: SAFE;
- independent official-channel verification: SAFE;
- warning without a verified required action: UNCERTAIN;
- caller-controlled transfer, credential, gift-card, remote-access, or pressure action: SCAM.

Generated language is original and grounded in the FTC scam-avoidance pattern; no external
benchmark text is copied. The decisive action precedes shared later dialogue so a latest-window
model must retain relevant harm instead of treating the final polite or hesitant turn as a veto.

## Contamination and source controls

The final MultiDoGO builder compares every candidate view to all 37,408 schema-v20 rows across
train, development, regression, validation, and OOD files. If any view falls within SimHash
Hamming distance six, its entire conversation family and every derived state are removed. Five
training and two validation families were removed by this rule. The final dataset validator checks
41,870 unique examples and reports no exact duplicates, near-template cross-split leakage, residual
prohibited PII pattern, or sealed-test overlap.

Direct Reddit scraping remains excluded. Reddit-like reports are represented through the licensed
IMC 2025 research artifact already in the parent dataset. Future Reddit material may inform a
reviewable threat-taxonomy or adversarial benchmark only when rights, privacy, and contamination
are independently resolved; raw user posts are not copied into training.

BothBosu is explicitly marked **prior-open** because its earlier errors informed this dataset
design. It may serve as a regression diagnostic, but it is no longer untouched evidence. The
primary test and named OOD partitions remain sealed.

## Frozen teacher and gates

The quality teacher is ModernBERT-base (149M parameters), initialized from the schema-v13
checkpoint. It trains for one epoch, 1,513 optimizer steps, with a 256-token left-truncated window,
5e-6 learning rate, schema-v13 retention weight 4.0, action loss weight 0.5, and seven auxiliary
action heads. Qwen remains useful as a later explanation specialist or capacity control; a
generative 0.8B-4B model is not the cheapest route to a sub-20-ms always-on verdict.

The run is rejected if any earlier gate fails:

| Gate | Requirement |
|---|---:|
| Development and unchanged-regression recall | at least 97% |
| Development and unchanged-regression SAFE FPR | at most 2% |
| Original and held-domain harmful-state recall | at least 97% |
| Original and held-domain routine/verified SAFE FPR | at most 2% |
| Original and held-domain state ordering | at least 95% |
| MultiDoGO original-call SAFE FPR | at most 2% overall, 3% per domain |
| Original and held-domain action macro AUC / exact match | at least 97% / 90% |
| Taskmaster and long-call SAFE FPR | at most 2% |
| Prior-open BothBosu recall / SAFE FPR | at least 97% / at most 2% |
| Core ML desktop end-to-end p95 after quality passes | at most 20 ms |
| Physical mobile measurement | required before a mobile claim |

PyTorch latency is diagnostic only. Dataset construction cannot prove the runtime target, and a Mac
measurement cannot prove physical-phone behavior.

## Frozen identities

- MultiDoGO derivative manifest SHA-256:
  `475c19324a16e4adea5dab75aba2e5c209e7f59ccb78cb821d19bbfcab6dff63`
- Processed schema-v22 manifest SHA-256:
  `814414895d7cb808a5a28fa31675c23e068d3e7f0bf642cf41daef805088d2ec`
- Training JSONL SHA-256:
  `1c076d0f6d98d39178fdc503345d4a0c85dc5ed4a8d410789cb84f86d2afbfcc`
- Frozen experiment configuration SHA-256:
  `36aadf8df9878f2b841e2342d30cf81705f4bf68340dde0fa4c3fd693203dcbd`
- Frozen configuration:
  `configs/encoder-schema22-service-evidence-actionheads-ret4-aw05-vw025-left.json`

The preflight rechecks artifact hashes, license and source revision, roleplay accounting, family and
domain isolation, state completeness, action targets, tokenizer-length summaries, teacher anchors,
the prior-open BothBosu disclosure, and byte-preservation of the original verdict head before
training.
