# ScamBench protocol v0.3

## Primary claim

A model passes only when a threshold selected on `dev` achieves both of the following on an eligible,
unobserved primary test:

1. scam recall ≥ 0.97;
2. clean-message false-positive rate ≤ 0.02.

Macro F1 ≥ 0.94 is the stretch target. The binary safety gate uses only clearly labeled SAFE and
SCAM rows. UNCERTAIN rows remain in the three-way macro-F1 evaluation and abstention analysis.
For Qwen/GGUF candidates, the SCAM threshold is the highest threshold on the binary development
subset that jointly achieves recall at least 0.97 and FPR at most 0.02. If that joint contract is
infeasible, the evaluator falls back to maximum recall under the FPR constraint and records the
failure. With the SCAM threshold frozen, a SAFE threshold is selected on the full development split
to maximize three-way macro F1; anything meeting neither threshold abstains as UNCERTAIN. Reports
retain raw argmax as a diagnostic, but the stretch gate uses this frozen product decision rule.

Qwen verdict scores compare the first tokenizer position where `SAFE`, `UNCERTAIN`, and `SCAM`
diverge, using one prompt forward pass and three class logits. Historical reports that
length-normalized each complete label spelling are retained as rejected evidence: unequal token
lengths made the longer `UNCERTAIN` suffix artificially favorable. A GGUF candidate must implement
the same branch-token rule and demonstrate score, verdict, and threshold parity before its quality
result can authorize packaging.
Every scam category represented by at least 20 untouched test examples must independently achieve
recall ≥ 0.97; tiny categories remain reported without being promoted into a statistically brittle
pass/fail rule.

## Leakage controls

- Normalize Unicode and whitespace before exact deduplication.
- Replace real-source emails, long phone-like values, and long account-like digit sequences before
  IDs, family clustering, splitting, or fitting; fail validation if a prohibited value survives.
- Replace URLs, emails, and numbers before hashing a template family.
- Cluster real-message templates with 64-bit character SimHash at Hamming distance ≤ 6; quarantine
  a complete cluster when its source labels conflict.
- Assign an entire family to exactly one development split.
- Fit thresholds and calibration only on `dev`.
- Never use `test`, `ood_financial`, `ood_wspr`, `ood_forum`, or
  `ood_forum_materialized` for early stopping, prompt selection, or hyperparameters.
  `forum_validation` may select forum training volume only; it cannot fit a threshold, update
  weights, or select prompts. A slice with no SAFE rows (currently WSPR) measures unseen-family
  recall and abstention behavior but cannot measure FPR. Forum validation and forum OOD include
  SAFE controls, so they report FPR with their exact denominators and wide intervals.
- `taskmaster_validation`, `scam_dialogue_validation`, and `apptek_call_selection` may inform
  candidate selection only. They cannot fit calibration or thresholds. Their source-family or
  shared-speaker/call-component partitions are fixed before scoring.
- `multidogo_annotation_dev` and `multidogo_annotation_test` are publisher-split, family-disjoint
  SAFE diagnostics selected from audited turn-level intent/slot annotations. Neither fits weights,
  calibration, thresholds, prompts, or routing. A schema-24 candidate must hold overall SAFE FPR to
  at most 2% and every one of the six domains to at most 3% on both slices.
- `ood_scam_dialogue` and `primary_test_v8` remain prediction-sealed until model, quantization, and
  routing decisions are frozen.
- `apptek_call_ood` remains prediction-sealed until model, calibration, routing, and export choices
  are frozen. AppTek rows never enter fitting; the publisher designates the corpus for evaluation
  and analysis only.
- Report source-stratified scores so a large easy source cannot hide a weak scam category.
- A test outcome may not silently become a hyperparameter. If a test result triggers another
  training, prompt, architecture, routing, or quantization choice, that split is retired to
  regression-only status for all subsequent primary claims. The schema-v6 test was observed during
  Q4/Q5 quantization escalation and therefore cannot support a final schema-v8 winner claim.

These controls prevent leakage created by ScamBench itself. They cannot establish that public
source messages were absent from a pretrained model's upstream corpus. Treat the independent OOD,
unseen-family, materialized-placeholder, and synthetic-family results as stronger evidence than an
in-distribution point estimate, while still reporting the unresolved pretraining-contamination
risk.

For schema v8, a newly sourced and family-isolated primary test is mandatory. It must be frozen
before any model trained or selected using schema-v6 test feedback is evaluated. The existing
schema-v6 test remains useful, fully reported regression evidence; it is not discarded or renamed
as unseen. `primary_test_v8` now supplies 1,820 sealed Portuguese/Mozambican real-message family
representatives (526 SCAM, 1,294 SAFE). No candidate predictions were used to build it. It is local
evaluation only pending a dataset-specific license clarification and cannot enter training.

## OOD interpretation

`ood_financial` preserves its source labels, including visibly ambiguous cases such as ordinary
promotions and urgent repayment requests marked as scams. It is a noisy-domain stress test, not a
release gate and not ground truth for the product taxonomy. Report its scores and representative
errors, but do not tune to them or treat disagreement as an automatic model failure.

`ood_wspr` contains real phishing-campaign messages gathered through the source project's
VirusTotal/APWG workflow. ScamBench retains one representative per near-template cluster after
discarding sender and destination fields. Because the holdout is positive-only after uncertain
marketing/gambling cases are separated, it supports an unseen-campaign recall diagnostic but no
false-positive claim.

`ood_forum` contains disjoint near-template families from the CC-BY-4.0 IMC 2025 public-report
artifact. It intentionally includes multilingual and low-cue messages excluded from generative
training, so it measures whether the detector generalizes beyond explicit URL/credential patterns.
It includes 100 real forum SAFE controls alongside 2,000 SCAM and 200 UNCERTAIN rows. It is not used
for threshold selection, prompt selection, or training-size decisions.

`ood_forum_materialized` is a deterministic derivative of the frozen forum OOD slice. It replaces
research placeholders such as `<URL>` and `<NAMED_ENTITY>` with safe realistic-looking values and
retains only changed rows. It measures placeholder shortcut sensitivity and is never a training or
selection source. The parent forum labels remain source-reported and may be noisy; neither slice is
promoted into a product release gate without human relabeling.

`ood_azsc` is an Azerbaijani diagnostic added after schema-v6 training was frozen. The source paper
describes consented user SMS, translated UCI rows, and self-generated examples but provides no
per-row provenance. ScamGuard excludes it from fitting, selection, and licensed-real counts; maps
only source smishing rows with message-local evidence to SCAM; and reports it strictly as a mixed-
provenance multilingual stress test.

The Apache-2.0 BothBosu telephone-dialogue release supplies a one-per-family external synthetic
diagnostic. `scam_dialogue_validation` has 294 rows and may inform candidate selection;
`ood_scam_dialogue` keeps 1,049 disjoint family representatives prediction-sealed until model and
routing decisions are frozen. Both measure whether a short-message detector can recognize
manipulation inside longer caller/receiver exchanges. They are excluded from training and threshold
fitting, and their upstream-generated rows never count as independently collected real messages or
as a primary release denominator.

`taskmaster_validation` contains 450 CC-BY-4.0, human-authored Wizard-of-Oz transactional
dialogues, with 75 source-family-held rows per domain. It measures false positives on legitimate
conversation form. Taskmaster does not supply scam annotations, and these roleplays are not
naturally occurring calls; the slice is a hard-negative selection diagnostic rather than a primary
real-world denominator. Schemas v11 and v12 fit on 600 disjoint Taskmaster conversation families
and keep the same 450-family selection slice outside fitting and threshold calibration.

`youtube_scam_validation` is a positive-only CC0 real-scam-call-derived selection diagnostic: 70
early/recent windows from 35 source-and-near-template-connected families. It is excluded from
fitting and threshold calibration but may inform candidate selection. Its 80-window, 40-family OOD
partition remains prediction-sealed. Because the source is primarily scammer/scambaiter video
transcripts and contains no SAFE denominator, report it as recall evidence only—never precision,
FPR, or ordinary-victim performance.

`apptek_call_selection` contains 348 SAFE early/recent windows from 174 spontaneous English
service-call roleplays. One representative per near-template family is retained, and shared-call
and shared-speaker components remain whole. The disjoint 1,396-window `apptek_call_ood` partition
is prediction-sealed. AppTek is not real customer data and has no scam labels; its weak SAFE label
comes from the legitimate service-roleplay domain. The open slice may localize false positives and
select candidates, but it cannot fit weights, thresholds, or calibration. Once its result changes
a training design, it is selection evidence rather than an independent final claim.

`primary_test_v8` is a new-source robustness gate, not a replacement English in-domain denominator.
Its mobile-money focus and Portuguese/Mozambican language make it especially valuable for detecting
false positives on legitimate money-transfer conversations. Report it separately from schema-v6
regression results; do not pool the two into a single headline percentage.

## Latency contract

The desktop verdict fast path must achieve p95 single-message latency at most 20 ms with batch size
one, measured from tokenizer entry through probability output. Model-forward-only timing is
reported as a diagnostic and cannot satisfy this gate. Reports must state CPU/accelerator,
input-length distribution, warmup, artifact bytes, and quantization. Qwen likelihood scoring or
generation is measured separately. A routed hybrid must publish its escalation rate and end-to-end
p50/p95/p99/maximum. Release requires routed p95 at most 20 ms and escalated-path p95 under 50 ms;
the p99 and maximum prevent a sub-5% escalation rate from hiding the specialist tail. Mobile
latency requires the same benchmark on a physical target device and cannot be inferred from desktop
results.

Routing must be frozen on development data from text-free per-example ledgers. Join router and
specialist rows exactly on `(split, id)`, reject duplicate IDs or truth/source/category drift, and
report router-only, specialist-only, and routed metrics on the same untouched test rows. Aggregate
component percentiles may support a conservative bound, but they cannot be algebraically converted
into routed p95; record the frozen policy end to end per request.

Every neural quality ledger must also freeze its accelerator scoring shape: messages per forward
pass, candidate sequences per message, and the padding/bucketing rule. Schema-24 Qwen scoring uses
one message, three candidates, and 64-token left-padding buckets. Before release, rerun the frozen
policy at that product shape and require exact route and calibrated-verdict parity per example. A
probability tolerance may diagnose numerical drift but cannot excuse a threshold crossing. Report
runtime quality separately when parity fails; do not substitute a similar aggregate metric for the
failed per-example contract.

Every calibration artifact must state its SAFE-threshold semantics. Historical encoder artifacts
store a maximum SAFE-path risk (`p_safe >= 1 - threshold` and `p_scam < threshold`); Qwen verdict
calibration stores a direct minimum SAFE probability. Evaluation ledgers and production runtime
must share the normalized decision function rather than interpreting the field name ad hoc.

## Required report

Each run must publish the immutable configuration and data manifest plus:

- confusion counts, precision, recall, F1, FPR, and 95% Wilson intervals for binomial rates;
- language-stratified results whenever source language is available;
- three-way macro F1;
- Brier score and 15-bin expected calibration error;
- results for test, out-of-domain, adversarial, and hard-negative slices;
- p50/p95/p99/maximum batch-one latency, peak memory, artifact bytes, quantization, and routing rate;
- seed, runtime, hardware, tokenizer, model revision, and source revisions.

## Comparison discipline

External numbers are not direct comparisons unless the examples, split, labels, threshold policy,
and deduplication are identical. The current public reference points are:

- the 89M DeBERTa malicious-SMS model reports scam recall 0.9023 and FPR about 0.0144 on its own
  test set;
- a 2026 scam-detection study reported major BERT out-of-domain degradation and substantially
  better—but still imperfect—large-model performance.

ScamGuard must rerun eligible baselines on ScamBench before using the phrase “beats SOTA.”
For a same-row reference comparison, publish paired-bootstrap 95% intervals for recall, FPR, and
accuracy differences plus the exact McNemar discordance test. A different-dataset model-card score
remains context only, regardless of its point estimate.
