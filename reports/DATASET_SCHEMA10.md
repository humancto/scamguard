# ScamGuard dataset schema v10

Freeze date: 2026-08-21.

Status: **rejected after the first 149M ablation.** The data release is retained as failure evidence,
not as the current training recipe.

Schema v10 is a targeted dialogue-hardening release. It preserves every schema-v9 development and
regression row byte-for-byte and adds exactly 2,568 fitting rows. It does not open the 1,049-row
BothBosu OOD partition or the 1,820-row MOZ primary holdout.

## Why this increment exists

The schema-v9 encoders passed the familiar short-message regression set but failed source-family-held
conversation negatives:

| Frozen schema-v9 candidate | Parameters | Taskmaster SAFE FPR (n=450) | BothBosu SAFE FPR (n=153) | BothBosu scam recall (n=141) |
|---|---:|---:|---:|---:|
| ModernBERT-base | 149M | 95.78% | 90.20% | 99.29% |
| ModernBERT-large | 395M | 20.22% | 54.90% | 90.78% |

All values use each checkpoint's development-fitted threshold without external retuning. The larger
model did not solve the problem. The data lacked long legitimate transactional conversations and
paired multi-turn scam/legitimate contrasts.

## Frozen counts

| Split or tier | Rows | SAFE | UNCERTAIN | SCAM | Use |
|---|---:|---:|---:|---:|---|
| train | 14,407 | 8,793 | 855 | 4,759 | fitting |
| dev | 2,634 | 2,008 | 112 | 514 | calibration and checkpoint selection |
| regression test | 2,374 | 1,746 | 41 | 587 | historical regression only |
| named processed diagnostics | 4,344 | source-specific | source-specific | source-specific | no fitting |
| Azerbaijani OOD | 4,327 | 2,963 | 1,160 | 204 | no fitting or selection |
| sealed MOZ primary test | 1,820 | 1,294 | 0 | 526 | prediction-sealed |

The validator covers 28,086 unique processed rows before the separately sealed MOZ source and finds
no exact or family leakage. The 23,759-row core plus named processed diagnostics has three provenance
tiers:

- 13,127 naturally occurring licensed-source rows;
- 1,800 CC-BY-4.0 human-authored crowdsourced roleplay rows;
- 8,832 controlled Apache-2.0 synthetic rows.

The roleplay tier is intentionally not folded into a “real messages” headline count.

## Post-fit result and failure analysis

The best checkpoint was epoch 1. At its unchanged development-fitted threshold it achieved 99.81%
development recall at 0.30% FPR and repaired the held-out identity family from 0/72 to 72/72.
That local repair did not generalize into a viable product:

| Split | Scam recall | SAFE FPR | Decision |
|---|---:|---:|---|
| unchanged regression | 100.00% | 11.23% | reject |
| Taskmaster SAFE selection | N/A | 0.00% | hard-negative goal passed |
| BothBosu dialogue selection | 13.48% | 5.23% | severe dialogue-SCAM collapse |
| financial OOD | 86.92% | 57.89% | reject |

Taskmaster supplied 1,800 SAFE conversations while paired synthetic dialogue supplied only 384 SCAM
and 384 SAFE conversations. The model learned a broad “conversation means safe” shortcut. It also
saw 79.17% of Taskmaster fitting rows through right truncation because their 1,100-character cap
did not guarantee the 256-token contract. Schema v11 therefore reduces Taskmaster fitting to 600
source-family-disjoint rows, expands paired synthetic dialogue to 768 examples per label across 12
scenarios, caps complete recent turns at 600 characters, and neutralizes corpus-specific speaker
labels before tokenization.

## New fitting data

### Taskmaster-1 hard negatives

The pinned Google Taskmaster-1 two-person Wizard-of-Oz source contains 5,507 human-authored
transactional dialogues. ScamGuard applies privacy replacement and retains one latest-context window
per selected conversation. A salted family hash is applied before sampling.

- fitting: 1,800 SAFE rows, exactly 300 per source domain;
- selection: 450 SAFE rows, exactly 75 per domain;
- domains: auto repair, coffee ordering, movie tickets, pizza ordering, restaurant reservations,
  and ride booking;
- weak label: legitimate task-domain roleplay, not independently scam-labelled;
- fitting/selection family overlap: zero;
- cross-source near overlap admitted to fitting: zero.

### Paired synthetic dialogue v1

The deterministic generator creates 384 SCAM and 384 SAFE five-turn contexts. Every scenario has a
legitimate counterfactual and records the relevant FTC, FBI/IC3, or USPIS advisory. The eight
scenario pairs are remote support, government cases, refunds, banking, delivery, jobs, wrong-number
grooming, and insurance. All rows are training-only; no external diagnostic text is copied.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| schema-v10 manifest | `25a79b555d6cd0009d00a1abee14b73fef271e0236babc9750588007b8864eac` |
| train | `71b12b5d6f02e030026d7947044858b32add3df9dd02dd4e21d189274e4b3e88` |
| dev | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| regression test | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| Taskmaster train | `8a3da9e570cf6e4e08876493624e965fdb570d931250be622346f85217d2fa13` |
| Taskmaster selection | `47ff5e12866876a1b6f555f67b8b519d7c71b979467b700073c90fd0aec63b2c` |
| synthetic dialogue v1 | `fa7b7b0035d1d732074f49d0fd55bfc07b489dda6433dddb4dfe14fc7158b908` |
| BothBosu selection | `c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0` |
| BothBosu sealed OOD | `33d480aa505f16014e18a7193f379b618e7a9feeb90262c93e77433c022c1193` |
| sealed MOZ primary test | `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` |

## Latency relationship

Training-set size does not add a corpus lookup during inference. The under-20-ms product target is
controlled by parameter count, sequence length, quantization, runtime, hardware, and routing. The
synthetic examples fit inside the existing 256-token training window. The original Taskmaster cap
did not: 1,425/1,800 rows exceeded 256 tokens. Long-chat product inference must be incremental or
windowed and measured separately from the PRD's short-text latency target.

## Remaining release gates

- The 216-row stratified audit workbook still has zero independent label decisions.
- Taskmaster supplies weak legitimate-domain labels, not naturally occurring call ground truth.
- The synthetic dialogue curriculum needs independent review for realism and counterfactual balance.
- Native-speaker review remains incomplete for multilingual synthetic hard negatives and Chichewa.
- No schema-v10 model may claim SOTA until it is compared on identical frozen rows and the sealed
  primary test is opened once, after candidate and routing decisions are frozen.
