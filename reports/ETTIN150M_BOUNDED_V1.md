# Ettin-150M: one bounded ScamGuard experiment

## Status

The single frozen fit is running. No result, promotion, release, or SOTA claim is established
yet. This report will record the terminal outcome, including failures. No sealed evaluation is
authorized by this pilot.

## What was fixed before fitting

Commit `8ad5cd9161965227bc18e5809f1d3703d1a905f3` freezes the
[experiment configuration](../configs/ettin150m-bounded-v1.json), trainer, open evaluator,
300-row comparison manifest, and independent-review handoff. It was created before the fit and
pushed while the fit was running.

- Fresh [Ettin encoder 150M](https://huggingface.co/jhu-clsp/ettin-encoder-150m), MIT licensed,
  pinned revision `45d08642849e5c5701b162671ac811b7654bfd9f`; 149,607,171 parameters including
  the new three-way classification head. This is not a continuation of a rejected ScamGuard
  adapter. Upstream general-language benchmark results are not ScamGuard results.
- Exactly 24,436 schema-25 training rows and 2,634 development rows. Exact-text, ID and family
  overlap checks found no train/dev or train/open-test overlap. The open test is a repeatedly
  used regression set, not an untouched final test.
- Two epochs, batch 16, learning rate 2e-5, seed 20261010, 256-token input, three-way weighted
  cross entropy, 3,056 effective optimizer steps, 60-minute training cap. One terminal
  checkpoint; no sweep, restart, best-checkpoint selection or continued training.
- Temperature and the SCAM threshold fit on development data only. No threshold refit after
  open evaluation. The SAFE decision and evidence-compaction rule are fixed in the config.
- All 21 declared open slices, plus development, through the actual batch-one `Scanner.scan`.
  Measure 300 synchronized warm full-SDK calls on the Mac; no simultaneous teacher workload.
  Open evaluation has a 30-minute cap.
- A separate 300-row, preselected, label-stratified open comparison with cached Qwen3.8-27B
  Q4_K_M, official embedded chat template, reasoning off, temperature zero, constrained JSON
  verdict output, and a 30-minute cap including server startup. This is not a sealed SOTA test.

Core promotion requires at least 97% scam recall and at most 2% SAFE false-positive rate on
development, open regression and applicable required dialogue slices; eligible core categories
also need 97% recall. Full-scanner p95 must be at most 20 ms. Macro F1 of 0.94 remains the PRD's
reported stretch target, not a retrospectively added mandatory gate. All other slice results
must be reported even on failure. A promotion would authorize only a separately frozen final
validation/export experiment, not publication.

## Results

Pending the fixed terminal checkpoint and benchmarks.

## Independent review and publication

The [review handoff](../docs/INDEPENDENT_REVIEW_HANDOFF.md) identifies the exact 635-message
blind ZIP, reviewer qualifications, return files, and binding checks. All real review decisions
remain blank. It audits sampled training/development supervision and privacy, not held-out
benchmark labels, semantic explanation quality, runtime performance or model superiority.

No HF upload is attempted by this pilot. Publication still needs an accepted candidate,
independent review and adjudication, final same-set evidence, artifact/runtime verification,
source-license and attribution checks, a limitations/model card, and a candidate-specific release
manifest. The historical Qwen-only release validator must not be represented as validating this
encoder. A desktop-only claim is distinct from an unmeasured physical-mobile claim.

## Reproducibility notes

The first synthetic preflight stopped before inference because the installed HF Hub cache lookup
requested unneeded repository metadata. Restricting the lookup to the five pinned model/tokenizer
files fixed that implementation issue. Both preflight receipts are preserved; the second passed.
Neither preflight fit any model. The 611-test repository suite passed after the freeze commit.

Before any teacher inference, a read-only check against the pinned server parser found that its
API requires the schema inside `response_format.json_schema.schema`, whereas its README showed
a sibling `schema`. The harness now transports the identical frozen schema in the correct API
envelope. A dedicated wire-format test passes; the semantic protocol and 300 selected IDs are
unchanged. This compatibility repair is recorded in the teacher receipt, not a model retry.

Raw training data, the private answer key and the blind-review ZIP are not published to GitHub.
Text-free predictions and reports retain dataset, checkpoint, configuration and code identities.
