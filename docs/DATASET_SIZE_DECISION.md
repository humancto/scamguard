# Dataset-size decision

ScamGuard keeps schema v12 as the reproducible canonical control: **14,446 training examples** and
**28,125 unique processed examples** before the separately sealed MOZ source. The isolated schema
v13 dose experiment uses **14,062 training examples** and **27,741 unique processed examples** by
reducing one error-driven synthetic curriculum from 512 to 128 rows. Development and regression
remain byte-identical. Neither version is a release dataset: schema v12 is rejected for a severe
identity-family regression, while the schema-v13 neural model still misses the open recall, FPR,
macro-F1, and dialogue gates.

Schema v14 is a rejected 161-row real-dialogue ablation built on schema v13: **14,223 training
examples** and **27,902 unique processed examples**. It adds one early window from each of 145
CC0 scam-call source families, with 16 families contributing a second exact-unique early window.
The family-held 70-window open validation recall rises from 34.29% to 100%, but unchanged-regression
SAFE FPR rises from 4.18% to 8.48% and the balanced dialogue SAFE FPR rises from 18.30% to 73.20%.
The source is useful; this positive-only dose and checkpoint are not.

Schema v15 is a 256-row matched-negative ablation built on schema v14: **14,479 training examples**
and **28,158 unique processed examples**. Its increment is deliberately small and balanced: 16
legitimate service scenarios × four opening structures × four selected variants. All copy is
original synthetic SAFE dialogue. It uses only aggregate and metadata-level error localization from
the open AppTek slice, copies no AppTek text, fits no AppTek row, and leaves AppTek's 1,396-window
OOD partition sealed. This is an experiment size, not an accepted final corpus size.
The completed run is rejected: AppTek FPR is 15.52%, unchanged-regression FPR is 18.84%, and the
BothBosu selection result is 69.50% recall / 35.29% FPR. The smaller 8.91% schema-v13 AppTek FPR
remains the stronger baseline, though it also fails the 2% gate.

Schema v12's 23,798-row core and named diagnostics contain three explicit provenance tiers: 13,127
naturally occurring licensed-source rows, 600 human-authored crowdsourced roleplay dialogues, and
10,071 controlled synthetic rows. Schema v13 has the same 13,127 licensed-natural and 600
human-roleplay rows but 9,687 controlled synthetic rows, for a 23,414-row core-plus-named total.
Both retain the 4,327-row Azerbaijani mixed-provenance OOD diagnostic. These are targeted ablations,
not claims that either training-row count is universally optimal.

## Why these are the current experiment sizes

The real-data pipeline begins with far more messages, then removes exact duplicates, clusters
campaign-like near-templates, quarantines label conflicts, isolates families, and keeps noisy or
ambiguous sources out of fitting. More rows from the same phishing blast would inflate the corpus
without adding independent behavior.

The 14,446 training examples include 5,394 SCAM, 8,197 SAFE, and 855 UNCERTAIN cases. They combine
licensed SMS collections, one representative per WSPR campaign-like cluster, 1,248 licensed
public-forum examples (1,000 SCAM, 100 UNCERTAIN, and 148 SAFE), and synthetic counterfactuals. The
forum contribution is capped because it is the largest and noisiest source. The SAFE rows are held
constant in every cap candidate so adding positive reports cannot remove hard-negative coverage.

Synthetic v5 adds 1,728 training-only rows after the first encoder exposed a development-set
identity-impersonation failure. Twelve original scam families and twelve paired legitimate
lookalikes cover distinct mechanics in official FTC, FBI/IC3, IRS, and USPIS advisories. They add
behavioral breadth without changing any development, regression, or sealed-source message text.

Schema v10 added exactly 2,568 fitting rows after two source-family-held dialogue diagnostics exposed
a severe missing negative class. The schema-v9 149M encoder falsely flagged 95.78% of 450 unseen
Taskmaster SAFE dialogues and 90.20% of the 153 SAFE rows in the BothBosu selection slice. The 395M
encoder still produced 20.22% and 54.90% FPR, respectively. Parameter scaling did not solve the
problem. The increment therefore contains 1,800 privacy-normalized Taskmaster-1 two-person
transactional dialogues and 768 balanced, paired five-turn dialogue grammars. Taskmaster is
human-authored Wizard-of-Oz roleplay with a weak legitimate-domain SAFE label; it is neither counted
as naturally occurring communication nor presented as independently scam-labelled. Its disjoint
450-dialogue selection slice is excluded from fitting and calibration.

That first dialogue correction overfit the provenance format: it drove Taskmaster false positives
to zero but collapsed BothBosu scam recall to 13.48% and raised regression FPR to 11.23%. Schema
v11 therefore reduces Taskmaster fitting to 600 family-disjoint rows, expands the paired curriculum
to 1,536 generated conversations across 12 domains, and admits 1,495 after fail-closed evidence and
deduplication checks. A versioned speaker-neutral transform maps multi-party source roles to compact
`A:`/`B:` labels, while preserving short-message text. All 5,507 eligible Taskmaster dialogues fit
within 150 tokens after the complete-recent-turn cap; only 11 fitting rows exceed the encoder's
256-token window. The development, regression, Taskmaster selection, and BothBosu selection rows
remain outside fitting.

Schema v11 then met the development boundary but raised regression FPR to 11.57%. Its text-free
prediction ledger localized 197 of 202 SAFE false positives to four synthetic-v5 families involving
verified transfers, known contacts, in-platform marketplace actions, or official-app alerts. Schema
v12 adds only 512 training rows: 256 SAFE controls for those trust boundaries and 256 scam
counterparts that alter the sender, channel, secrecy, link, or payment action. Eight independently
worded paired families avoid copying regression text. The development and regression artifacts
remain byte-identical, so any improvement or damage is directly measurable.

The 512-row dose overcorrected. Schema v12 lowered unchanged-regression FPR to 4.18% but reduced
development recall to 85.60% and missed all 72 rows in the unchanged identity-callback family.
Schema v13 is the predeclared dose-16 ablation: 16 rows in each of the eight paired families, or
128 total. It recovers development recall to 95.91% and 53/72 identity cases, but still records
4.18% regression FPR and only 51.06% recall / 18.30% FPR on the open BothBosu dialogue selection.
The model-only result is rejected. A separately versioned deterministic policy clears the open
short-message binary gates, but it was designed after those errors were visible and does not repair
dialogue or macro F1. It is a research candidate requiring independent validation.

The next source-family-held SAFE-call benchmark then exposed a narrower failure. Schema v13 falsely
flagged 31/348 AppTek selection windows and schema v14 flagged 77/348; all false positives were in
the 174 early-call windows and none were in the 174 recent windows. A second bulk Taskmaster dose
would repeat the schema-v10 mistake, while more positive-only scam calls would reinforce schema
v14's shortcut. Schema v15 therefore adds 256 original early-opening controls: four examples in
each of 64 scenario/structure families. The full deterministic generator emits 1,024 rows so later
dose escalation can be isolated, but only the predeclared dose-16 subset enters this run. Explicit
anti-scam safety phrases are forbidden, preventing the SAFE label from being learned through words
such as “never ask,” “official app,” or “gift card.”
The result rejects dose escalation: although YouTube-call recall stays 100% and Taskmaster FPR stays
0%, the correction produces 329 unchanged-regression false positives. Future work must change the
learning formulation or source diversity rather than add more rows from this generator unchanged.

Before schema-v6 test or OOD outcomes were opened, a lexical proxy compared four forum-training caps using only
the fixed development and forum-validation slices:

| Forum SCAM / UNCERTAIN cap | Total train | Core-dev recall at <=2% FPR | Forum-validation recall | Forum-validation FPR |
|---:|---:|---:|---:|---:|
| 0 / 0 | 9,011 | 44.94% | 31.50% | 0/25 (0%) |
| 1,000 / 100 | 10,111 | 48.05% | 91.90% | 1/25 (4.00%) |
| 3,000 / 300 | 12,311 | 54.86% | 90.60% | 2/25 (8.00%) |
| 5,672 / 600 | 15,283 | 52.53% | 90.90% | 1/25 (4.00%) |

No lexical candidate passed the 97% release recall gate. The 1,000-row candidate produces the best
forum-validation recall while remaining the smallest strong-forum candidate; 3,000 and all-5,672
reduce held-out forum recall despite improving core-development recall. It remains the quality-first
cap for the Qwen experiment. The forum-validation FPR denominator is only 25 SAFE
messages, so its confidence interval is wide and it is not a release-quality FPR estimate. This is
an exploratory size selection, not a model quality result.

## Frozen schema-v12 identity

| Artifact | SHA-256 |
|---|---|
| manifest | `e4dd86b5ceb753803979561d67aa49123605c491c7a1d007fb57d3b67e2b1f14` |
| processed train | `a5c964838d742352897c388cfa42142f498ffd369d046515e97038b521105d24` |
| development, unchanged from v9 | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| regression test, unchanged from v9 | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| Taskmaster train | `c756886b2d97dc41d98956eb7dc468a7c43622e1861229b605768cedc78007c3` |
| Taskmaster selection | `539b81a06328b2914407565c1bb7fac54a486333cea45a0853b4e2160df79760` |
| BothBosu dialogue selection | `c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0` |
| synthetic dialogue generator output | `7f963f8557d9307e2ec605638ef4cccbc798a3414d774d42e31e9d1917fdceed` |
| complete synthetic generator output | `3fdbdaa3719bb4ed316d77966106a7f58df17ca2e94069cbc96b6733bf16c7ec` |
| synthetic generator manifest | `beea78dce364e7ba100c66d2024c79c0fae8a9bcb66cef8c366da31bdd002f49` |
| speaker-neutral transform implementation | `2077db829188313088b1ba02b02763ab85ce2fcfffc9bc8abf8319d1cab46bd4` |
| sealed schema-v8 MOZ test | `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` |

The MOZ artifact hash and 1,820-row denominator did not change when its overlap-reference manifest
was rebuilt against schema v12. Its state remains `SEALED_MODEL_PREDICTIONS_NOT_RUN`.

## Isolated schema-v13 dose-16 identity

| Artifact | SHA-256 |
|---|---|
| manifest | `68c481a889ce135eff4a472b71daeb65a00ee886daec4c415549f0726586c6bf` |
| processed train | `499e2c4ab4c191c58ac64bcc1027ecce17c0394a76677dde508b1d51ff53e325` |
| development, unchanged from v9/v12 | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| regression, unchanged from v9/v12 | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| complete synthetic generator output | `fc410f6250b50b1cc62c1500e4357817ed4a801c77c0e0a13e6132d2f51d5f51` |
| synthetic generator manifest | `a1c1ef5e97e6711f9e203f4888873f39fe4c16f00f38d3e89811be8cc569aafd` |

Build this experiment with `make schema13-dose16`; its files live under
`data/experiments/schema13-dose16/` and do not replace canonical `data/processed/`.

## Rejected schema-v14 real-dialogue identity

| Artifact | SHA-256 |
|---|---|
| manifest | `83ee137212eb99bac46f81a8ce265e7bad003c58cef35f5f951a9024a6ecc09b` |
| processed train | `93fa3d3ddea2c51b8093bc8fa486d26e345c08011da5de27e6d6dc1b42e6da97` |
| development, unchanged from v13 | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| regression, unchanged from v13 | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| real-call source audit | `a394be257adc881bf9d0de24e10df2e4180dc207fd3792eb1b71869556b4be22` |
| real-call open validation | `108daf448b01e64f1f6f58228b295119b536c44cede1e6b512abc1eaf9262b1e` |
| real-call sealed OOD | `69b0d22ee2a9d7b0d44df80fec30e1032bd16d24e5f078f105354c3eb6890cf8` |

Build the fresh source with `make youtube-scam-calls`, then build/validate the isolated dataset with
`make schema14-natural-dialogue`. The schema builder refuses to overwrite an existing experiment.
See [`reports/DATASET_SCHEMA14_REAL_DIALOGUE.md`](../reports/DATASET_SCHEMA14_REAL_DIALOGUE.md) for
the full admission and rejection record.

## Schema-v15 legitimate-opening dose-16 identity

| Artifact | SHA-256 |
|---|---|
| manifest | `aa24e34b48c58d1fbdeb1854c6dd261828891e764302381ebf5c59f1786ac9bf` |
| processed train | `4d0f03a946d407f81a354911a8da68f1c2b8e84c2fc00943da454317f3057214` |
| development, unchanged from v13/v14 | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| regression, unchanged from v13/v14 | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| complete legitimate-opening generator output | `2ca1f1cfeb4f283d29e1652839f4277e1fe3ac94bc5d5f7fc1781dfaf148ba94` |
| legitimate-opening generator manifest | `f97005bda95750ae0ea5b90518ec6a9b159c8b8a10fa9d9e44695190ee056a2c` |
| AppTek text-free source audit | `18cd3cce69a6cb4ab502b8c56324919b959cd7af3c58a7092e5285cb29edadf6` |
| AppTek open selection | `e0e7ad4de8d378061159df18a8fa6c39fedd69d0cdd0dffe3f72579158290b62` |
| AppTek sealed OOD | `1e6d1176936324f073ca7dd5746bcce5b7849d5be5c4973782479e595f3c3ade` |

Build and validate with `make schema15-legitimate-openings`. The builder refuses to overwrite the
isolated experiment. Reproduce the AppTek source and baseline selection reports with
`make apptek-callcenter`, `make apptek-eval-schema13`, and `make apptek-eval-schema14`.
The rejected checkpoint is documented in
[`reports/DATASET_SCHEMA15_LEGITIMATE_OPENINGS.md`](../reports/DATASET_SCHEMA15_LEGITIMATE_OPENINGS.md).

## Historical schema-v9 identity

| Artifact | SHA-256 |
|---|---|
| manifest | `c4c2f728e3bfbb0a5be67fde24f1c08c96edd1b0539c419f95320cc568fe0e50` |
| processed train | `bf7926fbc15d8b61debd7f52f665e85706b5d8f9665bfdd95d80b78f58ca615a` |
| development | `5b1bbf05c56d917d150a93147cf4e6913e6389f6a6e1a734b235cafc882c6b03` |
| regression test | `c55b396197575d32936a44c5432e6adc85b21cdb6f442fb663c760cd026dc554` |
| forum validation | `a31af23130a58cd79b5077841a5fc4488d53c35757743f4506f267eae3453220` |
| forum OOD | `b61cda974766d1c0dcf143b8bf8ce0ca100eebd77c0501d164471afdc9293332` |
| Azerbaijani OOD | `b29befed9bd55093e42a7d435f8883eec19b5e64f1fbdd0132a502ac0da84372` |
| sealed schema-v8 MOZ test | `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092` |

The schema-v9 Qwen SFT projection wraps each row in the frozen JSON contract. Its SHA-256 hashes are
`92ad679aabef095c6cacc67012ede1ec2c451907d062707648901f6857df7efe` for train and
`c6de2838597854ac497812f7c5467172d4bc27c0f7a24a35601be67deaf3d241` for development. The prior
Qwen reports and forum-cap curve remain explicitly schema-v6 historical artifacts.

## Relationship to the 20 ms target

Training-row count does not add a lookup over all rows during inference. Runtime latency is driven
by the exported architecture, parameter count, quantization, tokenizer, input length, hardware,
and any routing policy. The v0.3 lexical artifact demonstrates that distinction: it was trained on
all 10,111 schema-v6 rows yet measured 1.16 ms p95 batch-one inference on the development Mac. It fails the
quality gates, so speed alone does not make it the product model.

The release plan therefore separates the constraints:

1. choose the smallest model that clears the frozen quality gates;
2. quantize and measure the same checkpoint on native desktop and physical mobile hardware;
3. if the full Qwen path exceeds 20 ms, use a measured fast classifier plus an explicitly reported
   escalation rate rather than hiding slower routed cases.

## When to add more data

Do not enlarge the corpus merely to advertise a bigger number. Add a new version only when the
post-fit error audit identifies an independently sourced category, language, phrasing style, or
hard-negative gap. New rows must have compatible training rights, provenance, privacy treatment,
near-template clustering, and a family split. Because deployment-quantization escalation opened the
schema-v6 test, schema v8 creates a newly sourced primary test rather than reshuffling or relabeling
an observed row. Its 1,820 family representatives remain prediction-sealed. Because the publisher's
OpenRAIL tag does not include a dataset-specific license file, it is local-evaluation-only and cannot
become training data or a redistributed corpus without clarification.

Production release still requires independent human review of the stratified label workbook and
native-speaker review of multilingual synthetic hard negatives. Until then, v0.3 is a reproducible
research corpus—not a human-audited or SOTA dataset claim.

The full online source admission and rejection record is in
[`reports/ONLINE_SOURCE_RESEARCH.md`](../reports/ONLINE_SOURCE_RESEARCH.md).
