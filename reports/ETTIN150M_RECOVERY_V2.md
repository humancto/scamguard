# Ettin-150M: one bounded recovery experiment

Protocol frozen on 2026-10-11 before training; the completed results follow.
The user explicitly authorized one recovery experiment after v1's terminal rejection.
This does not reopen any previous run or authorize a sweep.

## Completed result: two gates fixed, two still failing

The single authorized fit completed at exactly **1,853 steps**. All 21 declared open slices
plus development were evaluated through the saved checkpoint's actual SDK, followed by
the fixed 300-timing latency benchmark and the cached 300-row teacher comparison.
**16/18 promotion gates pass, up from 14/18. This candidate is rejected for promotion/HF release.**
No further fit, export, threshold change or sealed-test evaluation was run.

| Originally failed gate | V1 | V2 | Unchanged requirement | Result |
| --- | ---: | ---: | --- | --- |
| Development SCAM recall | 479/514 = 93.19% | 508/514 = 98.83% | >=97% | Pass |
| MultiDoGO SAFE false alarms | 32/896 = 3.57% | 16/896 = 1.79% | <=2% | Pass |
| Phone-dialogue SAFE false alarms | 23/170 = 13.53% | 15/170 = 8.82% | <=2% | Fail |
| BothBosu SAFE false alarms | 60/153 = 39.22% | 28/153 = 18.30% | <=2% | Fail |

Both remaining populations allow at most **3** false SCAM alerts at the frozen 2% cap.
Phone has 15; BothBosu has 28. Human review or HF login alone cannot remove these failures.

All 14 previously passing gates remain passing. Important trade-offs still exist:

- Development SAFE FPR: 38/2,008 (1.89%) -> 35/2,008 (1.74%).
- Open regression SCAM recall: unchanged 586/587 (99.83%); SAFE FPR:
  6/1,746 (0.34%) -> 23/1,746 (1.32%). Three-class macro F1 falls
  from 0.8141 to 0.7919, below the 0.94 stretch target.
- Phone SCAM recall: 173/174 (99.43%) -> 171/174 (98.28%).
  BothBosu SCAM recall: 141/141 (100%) -> 140/141 (99.29%).
- Taskmaster SAFE false alarms: 1/450 -> 0/450.
- The development identity-impersonation category is now 68/72 (94.44%).
  Its remaining four scams are marked SAFE. Aggregate development recall passing does not
  mean every development category reaches 97%.
- Development UNCERTAIN recall remains weak: 31/112 (27.68%); 81/112 are marked SCAM.
  All-label alert precision is 508/624 (81.41%), not the binary-subset 93.55%.
- Financial-diagnostic recall falls 97.31% -> 93.08%; adversarial recall falls
  100% -> 96.88%. These are transparent diagnostics, not newly invented promotion gates.

All 479 v1 development SCAM detections are retained; 29 former misses become SCAM.
This is the result of the combined continuation/calibration recipe, not a causal ablation
proving that replay alone produced the gain. The diagnostic sets have been inspected and
informed this recovery, so these are open-screen results, not fresh final-generalization evidence.

### Latency, size and completed fit

- 149,607,171 parameters; complete FP32 checkpoint **602,036,052 bytes
  (574.15 MiB)**. No quantized/mobile package produced.
- Training runtime 1,826.49 seconds; full train/save/calibration receipt 1,871.48 seconds
  (31.19 minutes). One fresh optimizer, one terminal checkpoint, no retry.
- All-open evaluation runtime 308.73 seconds; 19,823 text-free prediction records:
  2,634 development + 17,189 across the 21 open slices.
- Same latency selection as v1: 100 pinned open-test messages, three rounds, ten warmups.
  Full SDK **p50 13.06 / p95 17.12 / p99 18.33 ms** on this M4 Max/MPS desktop.
  V1 p95 was 13.72 ms. The <=20 ms p95 gate passes, not an every-message latency bound.
  Maximum observed latency was 20.53 ms; 2/300 observations exceeded 20 ms.
- Peak evaluator process RSS 1,252,589,568 bytes; this is not total device/GPU memory.
  No physical-mobile, cold-start, external-IPC or battery claim.
- System load was high and variable. Read-only uptime snapshots immediately before/after
  evaluation reported load averages 175.01/394.37/369.02 and 36.47/156.08/262.98.
  Other processes were not changed. These observations are context, not proof of a causal
  explanation for the latency difference.

### All declared populations

Rates use each population's own reference-label denominators; no pooled score is used.
A dash means no reference examples for that binary metric. Macro F1 always uses the fixed
SAFE/UNCERTAIN/SCAM label set: a perfect SAFE-only slice therefore scores 0.3333, and a
perfect two-label slice at most 0.6667. Do not compare those ceilings as model regressions.

| Population | Rows | SCAM recall | SAFE FPR | Three-class macro F1 |
| --- | ---: | ---: | ---: | ---: |
| dev | 2,634 | 98.83% | 1.74% | 0.7107 |
| test | 2,374 | 99.83% | 1.32% | 0.7919 |
| forum_validation | 1,125 | 98.10% | 8.00% | 0.7370 |
| ood_financial | 431 | 93.08% | 52.05% | 0.4615 |
| ood_wspr | 488 | 99.51% | — | 0.6228 |
| ood_azsc | 4,327 | 97.55% | 24.27% | 0.2499 |
| call_state_validation | 2,048 | 100.00% | 0.00% | 1.0000 |
| call_window_validation | 447 | — | 0.00% | 0.3333 |
| ftc_pattern_validation | 84 | 100.00% | 0.00% | 1.0000 |
| multidogo_call_validation | 896 | — | 1.79% | 0.3296 |
| multidogo_state_validation | 592 | 100.00% | 0.00% | 1.0000 |
| phone_scam_validation | 344 | 98.28% | 8.82% | 0.6317 |
| scam_dialogue_validation | 294 | 99.29% | 18.30% | 0.5954 |
| taskmaster_validation | 450 | — | 0.00% | 0.3333 |
| apptek_call_selection | 348 | — | 0.29% | 0.3324 |
| ppone_validation | 33 | 91.67% | — | 0.2996 |
| banking77_validation | 770 | — | 9.87% | 0.3064 |
| vystadial_safe_diagnostic | 1,024 | — | 2.73% | 0.3135 |
| international_robocalls_diagnostic | 47 | 94.74% | — | 0.2412 |
| adversarial | 320 | 96.88% | 15.63% | 0.5789 |
| youtube_scam_validation | 70 | 100.00% | — | 0.3333 |
| ood_chichewa | 677 | 70.48% | 20.17% | 0.4664 |

Per-source/category/language details and row-level Wilson intervals are in the full open
report. Those intervals do not correct for related templates, dialogue families, repeated
research selection or base-model pretraining contamination. For example, the MultiDoGO
point estimate passes 2%, but that does not establish a population FPR upper bound below 2%.

### Frozen larger-model comparison

The completed Qwen3.8-27B Q4_K_M zero-shot, reasoning-off teacher ledger was reused without
another teacher run. Identity/text/label/source hashes match all 300 preselected rows.

| Metric on the identical open 300 | Ettin v2 | Frozen Qwen teacher |
| --- | ---: | ---: |
| SCAM detections | 76/76 | 65/76 |
| SAFE false SCAM alerts | 2/210 | 0/210 |
| Correct UNCERTAIN | 8/14 | 3/14 |
| Three-class macro F1 | 0.8352 | 0.6515 |

Qwen's 11 missed SCAM references were deferred as UNCERTAIN, not marked SAFE. The paired
binary correctness comparison is 11 candidate-only versus 2 teacher-only correct
(McNemar two-sided p=0.02246). The paired bootstrap recall difference is +14.47 percentage
points, 95% interval +6.94 to +22.83; FPR is +0.95 points, interval 0 to +2.43.
These row-level, open-sample diagnostics are not SOTA proof and cannot override rejection.

### Reproducibility and evidence

The recipe, implementation and CPU replay receipts were committed and pushed before fitting
at `a3637e77499a4f2f09d74238adb01069cdeaa53a`.
Configuration SHA-256:
`5c6a241a6994d0a64ea7d92c8052b9b4d1a76e917b553683ba1207babd7986b5`.
Final model weights:
`d03120a4982d1191eb6257c081aedf13e18301f5c291ebd93f7ab2e0a76419e9`.
Calibration:
`d31108b08ad213ebb0c0e3db1794393a119f34dda8a8aa3af6146d5fab2eb08d`.

Temperature = 2.3687777066724722; SCAM threshold = 0.18446353822946548.
Its minimum observed development score margin is 0.0005237981677055359.
The original dev-only objective and all quality gates were preserved.
The independent ledger audit found exact saved-model serving parity on all 2,634 development
rows: zero probability differences and zero verdict/argmax/SCAM-mask differences. It separately
recomputed all 18 gates, all 22 population counts and the 300 latency observations. This is
an automated reproducibility check, not independent human label review.

- [Training receipt](runs/sg-ettin150m-recovery-v2.train.json) and
  [development ledger](runs/sg-ettin150m-recovery-v2.train.dev.predictions.jsonl).
- [Full open report and latency trace](runs/sg-ettin150m-recovery-v2.open.json) and
  [text-free SDK ledger](runs/sg-ettin150m-recovery-v2.open.predictions.jsonl).
- [Cached-teacher paired comparison](runs/sg-ettin150m-recovery-v2-vs-qwen38-27b-open300-20261010.paired.json).
- [Independent calculation/integrity receipt](runs/sg-ettin150m-recovery-v2.integrity.json).
- [V2 audit-packet reuse binding](../docs/ETTIN_V2_AUDIT_REUSE.md).
- Regression suite: **672 tests passed**; new calibration/replay modules have 60 dedicated
  tests including an actual CPU Trainer data-loader order/count check without model forward.
  Historical v1 source code and evidence are unchanged.

### What is still needed to publish

This bounded recovery is complete, but a passing release is not. Do not automatically
continue training. The remaining quality issue is legitimate phone/service requests being
misclassified as SCAM; any new data-label work or experiment must be separately authorized
and versioned, with fresh validation beyond these repeatedly opened sets.

The owner can proceed in parallel with the existing human review: appoint an independent
reviewer/coordinator, arrange competent human language help, send the existing private ZIP
and [forwardable note](../docs/INDEPENDENT_REVIEW_HANDOFF.md), and return the completed
`scamguard_blind_audit.csv`, truthful `reviewer-attestation.txt`, and
`reviewer-assignments.tsv`. All 635 decisions are still blank. Estimate 8–14 base person-hours
plus language coordination/adjudication; do not substitute AI labels.

After a genuinely passing candidate exists, release also needs appropriate frozen final
validation, provenance/privacy checks (including unresolved phone-generator version terms
and existing source-rights review), an Ettin-specific package/card and export parity if its
format changes. HF authentication was checked on 2026-10-11: **not logged in**.
Use local `hf auth login` when publication is ready; never send a token in chat.
No upload occurred, and no claim of release readiness, mobile validation or SOTA is made.

## Question and fixed intervention

Can one low-learning-rate continuation improve identity-scam development recall and
legitimate-dialogue false alarms without losing the previously passing gates?

- Parent: the exact completed `sg-ettin150m-bounded-v1` checkpoint; fresh optimizer.
- Data: unchanged schema-25 train (24,436) and development (2,634) rows, hash-pinned.
- Present every training row once; present 1,206 identity-SCAM and 1,400 phone/MultiDoGO-SAFE
  training rows two additional times. Shuffle this finite list once with seed 20261011.
- Exactly 29,648 presentations / 1,853 batches of 16; learning rate 5e-6; original-row
  square-root class-weighted three-class cross entropy; 8% warmup; no auxiliary losses.
- Preserve the 256-token evidence/recent preprocessing policy, labels, sources, families,
  tokenizer and three-way head. No new texts, weak short-view labels or evaluation rows.
- Fit cap: 2,700 seconds. Outer process cap: 3,100 seconds including save/calibration.
  Incomplete fit, numerical failure, invalid protocol or failed gates ends this experiment.
- Fixed terminal checkpoint only. No intermediate validation, best-checkpoint selection,
  additional seeds, post-benchmark threshold changes or automatic retraining.

The diagnosis used previously open results: 33 of v1's 35 development misses belonged to
one held-out identity-impersonation family; legitimate service requests dominated dialogue
false alarms. Error inputs were not truncated. These observations inform the recipe, but
their texts/families are not added to training. MultiDoGO short agent views are omitted
because inherited service-domain SAFE labels do not establish standalone safety.

Development calibration now uses batch-one, unpadded, stripped inputs and MPS float32
one-dimensional Torch softmax, matching serving arithmetic. Temperature is fitted on dev
only. Threshold selection still maximizes recall under SAFE FPR <=2%, then precision and
highest observed threshold. An interior midpoint is allowed only if every dev alert decision
is unchanged and its margin is at least one float32 ULP; this is not a drift guarantee.
Actual reopened-checkpoint SDK predictions, not training metrics, determine gate outcomes.

## Evidence and scope

- [Frozen configuration](../configs/ettin150m-recovery-v2.json).
- [CPU metadata/replay preflight](runs/sg-ettin150m-recovery-v2.train.preflight.json).
- [Exact replay hash manifest](runs/sg-ettin150m-recovery-v2.train.replay.json).
- All 21 existing open evaluation slices and all 18 promotion gates remain unchanged.
- Train versus dev and every open slice: zero exact ID, stripped-text SHA-256 or family-ID
  overlap; all frozen file hashes and row counts verified before fit. This is not semantic
  deduplication, proof of no base-model pretraining contamination, or independent holdout status.
- The 300-row Qwen3.8-27B comparison reuses its completed, hash-bound teacher predictions;
  no new teacher inference. It remains an open diagnostic, not SOTA evidence.
- No sealed test is opened. No model release or physical-mobile claim is authorized by
  passing this screening experiment alone.

SCAM recall = detected SCAM / reference SCAM. SAFE FPR = SAFE references predicted SCAM /
reference SAFE; UNCERTAIN predictions are not false-positive SCAM alerts. Report uncertainty
and three-class macro F1 separately, without treating abstention as successful detection.
Latency is warm batch-one full `Scanner.scan` including preprocessing, tokenization,
calibration, extraction and synchronization; it excludes loading and external IPC.

## Independent review and release

The [existing review handoff](../docs/INDEPENDENT_REVIEW_HANDOFF.md) still applies to the
unchanged train/dev corpus. Its 635 decisions remain blank. Resampling existing rows does
not complete that review or establish label correctness. Keep the private ZIP unchanged;
do not expose canonical labels or model predictions to reviewers.

An HF release still requires passing quality/final-validation gates, a truthful completed
human review and rights/privacy checks, an encoder-specific release package and card,
and local HF authentication. The historical Qwen/GGUF release validator is not an Ettin
release certificate. A failed recovery will be documented, not published as a passing model.
