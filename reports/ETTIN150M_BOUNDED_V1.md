# Ettin-150M: one bounded ScamGuard experiment

## Status

**Rejected: 14/18 promotion checks pass.** The single frozen fit and all 21 open benchmark slices
are complete. Development recall and false-positive rates on three required dialogue slices
fail. Warm desktop full-scanner latency passes at **13.72 ms p95**. There will be no continuation
or threshold refit. The 300-row larger-model comparison is also complete: Ettin outperforms the
fixed Qwen-27B baseline on that open sample, but this does not repair the broader quality failures.
No export, sealed evaluation, release or SOTA claim is authorized by this pilot. All compute jobs
started for this experiment have finished; the local teacher server is stopped.

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

The fit completed in 1,256.58 seconds; setup, saving and development calibration brought total
trainer wall time to 1,291.27 seconds (21.52 minutes). Its fixed terminal checkpoint is
`artifacts/checkpoints/sg-ettin150m-bounded-v1`, weights SHA-256
`d7ee45bb5d2cc12e0b0f26171cca45234cb9a0a7c517237971e31c2766f06803`.

Development calibration chose temperature 2.55844629 and SCAM threshold 0.120099813 without
opening test results. It detected 480/514 scams (93.39%, below the 97% gate) and falsely alerted
on 38/2,008 SAFE messages (1.89%, within the 2% limit). Three-class macro F1 was 0.6929, below
the 0.94 stretch target. Training completion is not quality success. This rejects the specific
frozen recipe, not every possible use of the Ettin architecture.

The actual batch-one SDK development pass detects **479/514 scams (93.19%)**, with the same
38/2,008 SAFE false positives and macro F1 0.6925. One SCAM row crosses the exact calibrated
threshold by one float32 ULP: NumPy's calibration softmax yields 0.12009981274604797, while
PyTorch softmax yields 0.12009980529546738. A CPU-only recomputation on the same saved logits
reproduces the difference; text preparation, label mapping and threshold loading match. Across
development rows the largest probability difference is 1.02e-6 and only that verdict changes.
**SDK results control the product-facing report.** No numerical tolerance or threshold adjustment
was introduced after observing this result.

### All declared open results

The actual SDK evaluated **19,823 unique `(split,id)` records**: development plus 17,189 examples
across all 21 declared open slices. Evaluation and latency measurement took 219.05 seconds.
An independent read-only recomputation verified all data/ledger hashes, row counts, confusion
matrices, metrics and promotion gates. The source's financial-label noise caveat remains relevant;
it does not justify silently changing labels or omitting that diagnostic.

| Split | N | SAFE / UNC / SCAM | Scam recall % | SAFE FPR % | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Development | 2,634 | 2008 / 112 / 514 | 93.191 | 1.892 | 0.6925 |
| Open regression | 2,374 | 1746 / 41 / 587 | 99.830 | 0.344 | 0.8141 |
| Forum | 1,125 | 25 / 100 / 1000 | 99.200 | 12.000 | 0.7860 |
| Financial OOD | 431 | 171 / 0 / 260 | 97.308 | 61.404 | 0.4396 |
| WSPR OOD | 488 | 0 / 79 / 409 | 99.511 | — | 0.6138 |
| Azerbaijani OOD | 4,327 | 2963 / 1160 / 204 | 97.549 | 23.422 | 0.1724 |
| Controlled call states | 2,048 | 1024 / 512 / 512 | 100.000 | 0.000 | 1.0000 |
| Call windows | 447 | 447 / 0 / 0 | — | 0.224 | 0.3330 |
| FTC patterns | 84 | 42 / 21 / 21 | 100.000 | 0.000 | 1.0000 |
| MultiDoGO complete calls | 896 | 896 / 0 / 0 | — | 3.571 | 0.3267 |
| MultiDoGO controlled states | 592 | 296 / 148 / 148 | 100.000 | 0.338 | 0.9983 |
| Phone-scam conversations | 344 | 170 / 0 / 174 | 99.425 | 13.529 | 0.6199 |
| Scam dialogue | 294 | 153 / 0 / 141 | 100.000 | 39.216 | 0.5252 |
| Taskmaster | 450 | 450 / 0 / 0 | — | 0.222 | 0.3330 |
| AppTek | 348 | 348 / 0 / 0 | — | 4.023 | 0.3260 |
| PPoNE | 33 | 0 / 21 / 12 | 91.667 | — | 0.2092 |
| BANKING77 | 770 | 770 / 0 / 0 | — | 13.247 | 0.3041 |
| Vystadial | 1,024 | 1024 / 0 / 0 | — | 9.277 | 0.2851 |
| International robocalls | 47 | 0 / 28 / 19 | 94.737 | — | 0.2612 |
| Adversarial | 320 | 160 / 0 / 160 | 100.000 | 21.875 | 0.5750 |
| YouTube scam calls | 70 | 0 / 0 / 70 | 100.000 | — | 0.3333 |
| Chichewa OOD | 677 | 362 / 0 / 315 | 69.524 | 21.271 | 0.4184 |

Macro F1 always uses three fixed labels; absent classes contribute zero. Thus perfect single-class
performance scores 0.3333, and a binary-only slice has a maximum of 0.6667. Compare like-for-like
populations, not these heterogeneous macro scores against one another. A dash means no applicable
reference examples, not a measured zero error rate. Confidence intervals and source/category/
language breakdowns are in the [full report](runs/sg-ettin150m-bounded-v1.open.json).

The four failed mandatory checks are development recall (479/514), scam-dialogue SAFE false
positives (60/153), MultiDoGO complete-call false positives (32/896), and phone false positives
(23/170). All six supported core regression categories passed: credential theft 144/144,
delivery/toll 72/72, financial 72/72, identity/impersonation 144/144, opportunity 72/72 and
relationship 72/72. UNKNOWN has 10/11 and is below the predeclared 20-example category minimum.

Uncertainty remains weak outside controlled templates. Open regression correctly defers 22/41
UNCERTAIN references and alerts on 17/41. Its 609 alerts comprise 586 SCAM, six SAFE and 17
UNCERTAIN references. Conditional binary precision is 98.99%; the known-SCAM fraction among all
alerts is 96.22%, a different quantity. Development correctly defers only 30/112 UNCERTAIN rows.
PPoNE alerts on 18/21 uncertain references; international robocalls on 24/28. No SAFE examples
exist in those two slices, so their conditional binary precision does not establish alert precision.

### Measured desktop runtime

On Apple M4 Max, 128 GiB RAM, macOS 26.3 (25D5101c), Python 3.11/arm64 and PyTorch MPS:

- Warm batch-one full SDK, 100 frozen examples over three rounds: median **9.254 ms**,
  p95 **13.719 ms**, p99 **15.678 ms**, maximum **31.487 ms**.
- Includes preprocessing, tokenization, model, calibration, evidence/action extraction and
  synchronization; excludes model loading and external IPC. This is not an all-input 20 ms bound.
- Saved FP32 checkpoint: **602,034,667 bytes (574.145 MiB)** across six files. No quantized export.
- Peak process RSS: **1,260,011,520 bytes (1.173 GiB)**. This is a process metric, not an
  independent measurement of total Metal/unified-memory use.
- Zero invalid evidence-span offsets across all 19,823 scans. Exact offsets do not establish
  semantic explanation correctness. No physical-mobile measurement was performed.

### Fixed larger-model comparison

All **300/300** preselected examples completed in **550.18 seconds**, including server startup,
identity checks and three format-only smoke requests, within the 1,800-second cap. The sample
contains 210 SAFE, 14 UNCERTAIN and 76 SCAM references. It was frozen before model predictions;
the comparison joins exact IDs, labels, source hashes and raw-text hashes. There was no retry,
resampling, prompt search or use of sealed data.

| Metric on the identical 300 rows | Ettin-150M | Qwen3.8-27B Q4_K_M |
|---|---:|---:|
| SCAM alerts on reference SCAM | 76/76 (100%) | 65/76 (85.53%) |
| SCAM alerts on reference SAFE | 0/210 (0%) | 0/210 (0%) |
| Correct UNCERTAIN verdicts | 7/14 (50%) | 3/14 (21.43%) |
| Three-class macro F1 | 0.8367 | 0.6515 |
| Three-class accuracy | 96.33% | 86.67% |

Qwen deferred its 11 missed SCAM references as UNCERTAIN; it did not mark them SAFE. Ettin
alerted on six UNCERTAIN references and Qwen on seven. Both have perfect conditional binary
precision in this sample, but known-SCAM fractions among all alerts are 92.68% and 90.28%.

The paired scam-recall difference is +14.47 percentage points, with a 95% paired-bootstrap
interval of +6.94 to +22.83 points (5,000 iterations). Binary-correctness discordance is 11:0;
exact two-sided McNemar p = 0.0009765625. Zero observed SAFE false positives is not proof of zero
population risk: each model's Wilson 95% upper FPR bound is 1.80% on these 210 SAFE examples.

**Scope:** this is an open, stratified diagnostic with a dev-calibrated task specialist versus a
zero-shot, quantized, reasoning-off larger model using its official embedded chat template. The
sample overrepresents UNCERTAIN relative to the full open test. No general SOTA, untouched-test,
English-population or real-phone superiority is established. Neither an optimized larger-model
baseline nor a same-sealed-set release comparison was run. Do not use the positive paired result
to override the frozen rejection decision.

See the [paired statistics](runs/sg-ettin150m-vs-qwen38-27b-open300-20261010.paired.json),
[teacher receipt](runs/qwen38-27b-open300-20261010.teacher.json) and
[teacher preflight](runs/qwen38-27b-open300-20261010.teacher.preflight.json).
After the complete receipt was saved, SIGINT shutdown exited nonzero during Metal cleanup;
the process exited and a subsequent listener check confirmed port 8097 closed. No benchmark
requests were interrupted and no replacement run was performed.

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

The exact training metadata contains no non-commercial or share-alike source: 6,979 CC-BY-4.0
rows, 1,913 WSPR MIT rows, 1,190 phone-synthetic publisher-MIT rows, 1,656 CDLA-Permissive-1.0
MultiDoGO/derived rows, 12,263 original Apache-2.0 synthetic rows and 435 publisher-CC0
YouTube-derived rows. These declarations do not independently clear privacy, copied content or
downstream weight publication. Preserve all applicable notices and attribution. In particular,
the phone dataset README identifies only a "Llama 8B" generator, not its exact version/terms;
this is an unresolved release-provenance question, not evidence that publication is prohibited.
See [source research](ONLINE_SOURCE_RESEARCH.md) and [data governance](../docs/DATA_GOVERNANCE.md).
Restricted international-robocall, Vystadial and Chichewa material remains evaluation-only;
raw Reddit posts are not in this fit. No raw evaluation content is included in published ledgers.

### Exactly what the owner needs to supply

1. Appoint an independent primary reviewer/coordinator, with qualified human language help, and
   send the single ZIP and forwardable note from the handoff. All 635 decisions remain blank.
   Budget 8–14 base person-hours, plus language coordination and adjudication. Metadata marks
   27 rows as non-English across 19 languages; 208 rows have unspecified language metadata.
2. Return the unchanged completed `scamguard_blind_audit.csv`, truthful
   `reviewer-attestation.txt`, and `reviewer-assignments.tsv`. The handoff specifies fields,
   validation and import commands. Honest disagreements require adjudication, not a forced match.
3. Only when a release-eligible package exists, authenticate locally using
   `.venv/bin/hf auth login` and confirm the intended HF account/repository and visibility.
   A live CLI check found no logged-in account. Never put a token in chat.

No additional user input was needed to finish this bounded experiment. **Human review alone will
not make this rejected model meet the requested quality.** Final artifact validation, the
encoder-specific release contract and publication packaging are engineering work, not paperwork
the reviewer can certify. No automatic next-model search or further fitting is running.

## Reproducibility notes

The first synthetic preflight stopped before inference because the installed HF Hub cache lookup
requested unneeded repository metadata. Restricting the lookup to the five pinned model/tokenizer
files fixed that implementation issue. Both preflight receipts are preserved; the second passed.
Neither preflight fit any model. The 611-test repository suite passed after the freeze commit;
after the teacher API-envelope test was added, the full suite passed **612 tests in 39.47 seconds**.

Before any teacher inference, a read-only check against the pinned server parser found that its
API requires the schema inside `response_format.json_schema.schema`, whereas its README showed
a sibling `schema`. The harness now transports the identical frozen schema in the correct API
envelope. Commit `51dab9e` contains that pre-inference repair and its dedicated passing wire-format
test; the semantic protocol and 300 selected IDs are unchanged. This compatibility repair is
recorded in the teacher receipt, not a model retry.

Raw training data, the private answer key and the blind-review ZIP are not published to GitHub.
Text-free predictions and reports retain dataset, checkpoint, configuration and code identities.

Primary receipts:

- [Training](runs/sg-ettin150m-bounded-v1.train.json) and
  [development ledger](runs/sg-ettin150m-bounded-v1.train.dev.predictions.jsonl).
- [All open metrics and raw latency trace](runs/sg-ettin150m-bounded-v1.open.json) and
  [text-free SDK ledger](runs/sg-ettin150m-bounded-v1.open.predictions.jsonl).
- [Frozen teacher manifest](runs/qwen38-27b-open300-20261010.manifest.json),
  [text-free teacher ledger](runs/qwen38-27b-open300-20261010.teacher.predictions.jsonl) and paired
  report linked above.
