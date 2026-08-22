# ScamGuard dataset schema v21: human-call action states

## Decision

Schema v21 is admitted for one frozen teacher experiment. It adds licensed human-spoken call
structure and conversation-grounded action contrasts without changing the unchanged development,
regression, or external benchmark text. This is a dataset decision, not a production or SOTA
claim; the checkpoint must still clear every predeclared quality gate before export or sealed
evaluation.

The final training set has **26,579 unique rows**. That is the deliberate scale for this stage:
large enough to expose the encoder to 1,069 new human call families and 4,276 matched action-state
variants, but small enough for controlled ablation and error attribution. Dataset row count does
not determine inference latency. The under-20-ms target depends on the deployed student's depth,
width, sequence length, runtime, and quantization; the 149M-parameter model remains a quality
teacher until a candidate passes the dialogue gates.

## Training composition

| Data class | Rows | Accounting rule |
|---|---:|---|
| Naturally occurring licensed messages or real-scam-call-derived text | 8,134 | Existing source rows; no new claim of independent victim calls |
| Taskmaster human-authored transactional roleplay | 1,193 | Licensed non-synthetic rows, reported separately from naturally occurring data |
| HarperValleyBank human-human spoken roleplay | 1,069 | New CC-BY-4.0 SAFE call rows, reported separately from naturally occurring data |
| Controlled synthetic rows before schema v21 | 11,907 | Existing original generators and human-call action-state curriculum |
| New Harper-grounded final-turn variants | 4,276 | Synthetic derivatives; never counted as human calls |
| **Total** | **26,579** | 10,396 licensed non-synthetic + 16,183 controlled synthetic |

The training labels are 14,877 SAFE, 8,242 SCAM, and 3,460 UNCERTAIN. There are 11,489 rows with
seven explicit action targets. The 1,069 original Harper SAFE calls keep full verdict weight while
their weak action labels train the auxiliary heads; generated action-state rows retain the frozen
0.25 verdict weight so they cannot overwhelm the licensed call boundary.

## New licensed source

- Publisher repository: <https://github.com/cricketclub/gridspace-stanford-harper-valley>
- Paper: <https://arxiv.org/abs/2010.13929>
- License: CC-BY-4.0
- Pinned Git revision: `0bd721e877c4a85d8c13ff837e68661ea6200a98`
- Calls: 1,446 human-human simulated telephone-banking calls across 59 speakers and eight tasks
- Source transcript segments: 25,730; 25,381 have human corrections
- Download policy: transcript, metadata, README, and license only; no audio
- Transcript tree SHA-256:
  `99f30d235cf79bcfbb3438ff472e3e4ed2dcdb671512cde63da60024ad75b807`
- Metadata tree SHA-256:
  `d527d581d8124167c9e6b838cd5e02c600bfc23f6aee242ebe589ae4dc1fb042`

The model text prefers the human-corrected transcript and uses source ASR only when the corrected
field is empty. There are 349 such fallbacks in the selected windows. Speaker identities are not
included, and phone/account-like sequences are privacy-normalized. These are human-spoken
roleplays, not naturally occurring bank calls, and that distinction remains explicit in every
count and report.

## Task-disjoint partition

Training uses balance checks, check orders, bill payments, password resets, appointment scheduling,
and money transfers. Validation holds out branch-hours and replacement-card tasks completely:

| Split | Original calls | Grounded state rows | Families |
|---|---:|---:|---:|
| Training | 1,069 | 4,276 | 1,069 |
| Validation | 377 | 1,508 | 377 |

Every call family stays in exactly one split. Every state family contains the same preceding human
call and four final agent states: routine SAFE, independently verified SAFE, unresolved, and
harmful SCAM. Only the final turn changes. Each harmful turn contains extractive scam evidence and
is grounded in the FTC's scam-avoidance patterns; no external benchmark text is copied.

This construction tests the behavior schema 20 failed to transfer: a legitimate call may discuss
accounts, transfers, passwords, cards, or payments without becoming a scam, while caller-controlled
destinations, credential disclosure, remote access, gift-card payment, pressure, and irreversible
action should change the verdict.

## Frozen quality gates

The teacher starts from the schema-13 ModernBERT checkpoint and trains for one epoch at 256 latest
tokens. Schema-13 teacher logits anchor 14,062 unchanged rows; 12,517 rows are new or unanchored.
The experiment is rejected before export if any earlier gate fails.

| Gate | Requirement |
|---|---:|
| Development and unchanged regression recall | at least 97% |
| Development and unchanged regression SAFE FPR | at most 2% |
| Original and Harper harmful-state recall | at least 97% |
| Original and Harper routine/verified SAFE FPR | at most 2% |
| Original and Harper four-state ordering | at least 95% |
| Harper original-call SAFE FPR | at most 2% |
| Taskmaster and long-call SAFE FPR | at most 2% |
| BothBosu latest-window recall | at least 97% |
| BothBosu latest-window SAFE FPR | at most 2% |
| Core ML desktop end-to-end p95, after quality passes | at most 20 ms |

The schema-20 teacher failed BothBosu at 93.62% recall and 42.48% SAFE FPR. Schema v21 must beat
both numbers by a large margin and satisfy the absolute 97%/2% gates; merely improving one metric
does not pass.

## Internet-source boundary

Direct Reddit scraping remains excluded. Reddit-like public reports already enter through the
licensed IMC 2025 research artifact, with privacy normalization and source-wide deduplication.
TeleAntiFraud-28k remains a high-value candidate, but its gated archive was not accessible in the
authorized environment, so zero rows were ingested. Schema-Guided Dialogue remains a potential
CC-BY-SA-4.0 expansion only if this task-disjoint human-call experiment identifies a residual domain
gap; bulk ingestion before measuring schema v21 would make the result harder to diagnose.

## Frozen identities

- Processed manifest SHA-256:
  `1172b5045a943f00f923319807a95ea1c762eafcee439e9efe28c8c8cc000edf`
- Training JSONL SHA-256:
  `00f1dcb2fd20b2e6d3390979e46ad76f69422e2056deafd8b75244cf06581e03`
- Harper derivative manifest SHA-256:
  `302c870084bf54e4a401155cbc72bdc1f120553e1bbebdb0f7396e1cc2d6930e`
- Frozen experiment configuration:
  `configs/encoder-schema21-human-calls-actionheads-ret4-aw05-vw025-left.json`

The independent validator reports 44,638 unique examples across development, diagnostics, and the
sealed primary test, with no family leakage. The schema-21 preflight verifies hashes, rights,
counts, task isolation, four-way state completeness, action targets, token windows, teacher
anchors, and preservation of the original three verdict-head rows before training.
