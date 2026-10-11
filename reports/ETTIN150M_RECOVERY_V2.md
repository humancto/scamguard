# Ettin-150M: one bounded recovery experiment

Protocol frozen on 2026-10-11 before training. Results are pending in this revision.
The user explicitly authorized one recovery experiment after v1's terminal rejection.
This does not reopen any previous run or authorize a sweep.

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
