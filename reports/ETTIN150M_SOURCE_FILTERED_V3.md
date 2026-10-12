# Ettin-150M v3: final bounded attempt rejected

Completed 2026-10-12 UTC (2026-10-11 local). **12/18 gates pass; six fail.**
The one authorized fresh-base fit and all 21 declared open benchmarks are complete.
This candidate is **not production-ready and will not be uploaded to Hugging Face**.
The experiment is closed: no continuation, second seed, threshold refit or automatic next model.

The chosen goal was one defensible desktop-first release-candidate decision.
The experiment delivered a reproducible negative result; the broader production/HF goal is unmet.
Speed is adequate. Safety generalization is not.

## Decisive results

| Metric | v2 (historical) | v3 | Frozen requirement |
| --- | ---: | ---: | --- |
| Development SCAM recall | 508/514 (98.83%) | **441/514 (85.80%)** | >=97% |
| Development SAFE false alarms | 35/2,008 (1.74%) | 23/2,008 (1.15%) | <=2% |
| Open regression SCAM recall | 586/587 (99.83%) | 586/587 (99.83%) | >=97% |
| Open regression SAFE false alarms | 23/1,746 (1.32%) | **52/1,746 (2.98%)** | <=2% |
| MultiDoGO complete-call SAFE false alarms | 16/896 (1.79%) | **20/896 (2.23%)** | <=2% |
| Phone SCAM recall | 171/174 (98.28%) | **47/174 (27.01%)** | >=97% |
| Phone SAFE false alarms | 15/170 (8.82%) | **29/170 (17.06%)** | <=2% |
| BothBosu SCAM recall | 140/141 (99.29%) | **8/141 (5.67%)** | >=97% |
| BothBosu SAFE false alarms | 28/153 (18.30%) | 0/153 (0%) | <=2% |
| Full desktop SDK warm p95 | 17.12 ms | 11.02 ms | <=20 ms |
| Open regression three-class macro-F1 | 0.7919 | 0.7245 | 0.94 stretch |
| Promotion gates | 16/18 | **12/18** | All pass |

All six eligible regression scam categories still have 100% recall, and invalid evidence
spans remain zero. These successes do not override the six failures. Zero BothBosu false
alarms is not a safe improvement: 126 of its 141 reference scams are called SAFE, seven are
deferred, and only eight are detected. Phone has 116 scams called SAFE and eleven deferred.

## What was actually run

The [prospective contract](../docs/ETTIN_V3_FINAL_EXPERIMENT.md) and
[configuration](../configs/ettin150m-source-filtered-v3.json) were committed and pushed at
`ccecf2a` before the fit. The final local freeze additionally binds exact source/config/data
hashes and successful metadata preflight. No model forward occurred in metadata preflight.

- Original Ettin-150M base revision `45d08642849e5c5701b162671ac811b7654bfd9f`;
  149,607,171 parameters, fresh classification head and optimizer. No v1/v2 ancestral weights.
- 22,811 training examples from 12 previously admitted source IDs. Removed 1,190 phone
  synthetic and 435 YouTube rows; no attributable derivatives detected. Every retained row
  stayed byte-identical. No new examples, BANKING77 fit rows or AI-reviewed labels were added.
- Training labels: 13,082 SAFE / 7,038 SCAM / 2,691 UNCERTAIN. Development remains exactly
  the original 2,634 rows: 2,008 SAFE / 514 SCAM / 112 UNCERTAIN.
- Two epochs, batch 16, seed 20261012, learning rate 0.00002, weighted three-class CE,
  action-preserving v3 preprocessing, right truncation at 256 tokens.
- Exactly 2,852 steps. Fit runtime 1,176.35 seconds; train/calibration process 1,220.36 seconds.
  The 3,600-second fit envelope and 4,500-second total process cap were respected.
- Development-only batch-one FP32 calibration: temperature 2.9458531462;
  SCAM threshold 0.2022846490; SAFE maximum risk 0.2. No post-open refit.
- All 21 open populations plus development: 19,823 ledger predictions.
  Evaluation process 227.43 seconds, below the 1,800-second cap.
- Both supervised processes completed successfully. Exclusive attempt receipts prevent retries.

[Corpus receipt](data/ettin-v3-corpus.json) records hashes, source counts, privacy checks
and remaining qualifications. Zero ID, normalized-text or family-ID overlap was found between
training and each of the 22 development/open populations. Automated privacy patterns found no
flags. Neither check establishes semantic/pretraining independence or human privacy clearance.

## Development failure diagnosis

All **72/72 development identity-impersonation scams are predicted SAFE**. They account for
72 of the 73 misses; the other is one delivery/toll scam deferred as UNCERTAIN.
V2 detected 68 of those 72 identity cases. V3 assigns SAFE argmax to all 72, with SAFE
probability 0.88268–0.97121 and SCAM probability 0.00668–0.02859. This is not a small
threshold-rounding issue.

The exact v2/v3 join confirms all 2,634 IDs, reference labels and text hashes are unchanged.
V3 newly loses 68 identity detections, recovers one prior UNKNOWN-category miss, and retains
five previous misses. True UNCERTAIN outcomes remain 31/112 UNCERTAIN and 81/112 SCAM.
SAFE deferrals increase from 84 to 165. Development macro-F1 falls from 0.7107 to 0.6556.

All 514 scam rows have unspecified language metadata in this ledger; do not invent a
language-specific explanation. The observed concentration is in source
`scamguard_synthetic_v5`, category `IDENTITY_IMPERSONATION`, not a demonstrated language effect.

This is one bundled intervention: source subtraction, preprocessing, fresh initialization,
seed and the absence of v2's recovery replay differ. It does **not** identify which change
caused the regressions or prove that tiny models cannot work. It establishes that this
specific frozen candidate fails. The AI 96-case diagnostic did not replace ground truth.

## Complete open-population results

SCAM recall is reference scams detected / reference scams. SAFE FPR is reference SAFE rows
predicted SCAM / reference SAFE rows; it excludes UNCERTAIN references. A dash means that
reference class is absent, not a perfect result.

| Population | Rows | SCAM n | SCAM recall | SAFE n | SAFE FPR | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 2634 | 514 | 85.80% | 2008 | 1.15% | 0.6556 |
| test | 2374 | 587 | 99.83% | 1746 | 2.98% | 0.7245 |
| forum_validation | 1125 | 1000 | 97.80% | 25 | 16.00% | 0.7736 |
| ood_financial | 431 | 260 | 88.08% | 171 | 55.56% | 0.3798 |
| ood_wspr | 488 | 409 | 99.51% | 0 | — | 0.6107 |
| ood_azsc | 4327 | 204 | 92.16% | 2963 | 4.66% | 0.2889 |
| call_state_validation | 2048 | 512 | 100.00% | 1024 | 0.00% | 1.0000 |
| call_window_validation | 447 | 0 | — | 447 | 0.00% | 0.3333 |
| ftc_pattern_validation | 84 | 21 | 100.00% | 42 | 0.00% | 1.0000 |
| multidogo_call_validation | 896 | 0 | — | 896 | 2.23% | 0.3240 |
| multidogo_state_validation | 592 | 148 | 100.00% | 296 | 0.00% | 1.0000 |
| phone_scam_validation | 344 | 174 | 27.01% | 170 | 17.06% | 0.3315 |
| scam_dialogue_validation | 294 | 141 | 5.67% | 153 | 0.00% | 0.2719 |
| taskmaster_validation | 450 | 0 | — | 450 | 0.00% | 0.3333 |
| apptek_call_selection | 348 | 0 | — | 348 | 0.00% | 0.3333 |
| ppone_validation | 33 | 12 | 100.00% | 0 | — | 0.1905 |
| banking77_validation | 770 | 0 | — | 770 | 6.10% | 0.3117 |
| vystadial_safe_diagnostic | 1024 | 0 | — | 1024 | 1.76% | 0.2975 |
| international_robocalls_diagnostic | 47 | 19 | 94.74% | 0 | — | 0.1846 |
| adversarial | 320 | 160 | 96.25% | 160 | 3.13% | 0.6171 |
| youtube_scam_validation | 70 | 70 | 11.43% | 0 | — | 0.0684 |
| ood_chichewa | 677 | 315 | 57.14% | 362 | 6.35% | 0.4508 |

Macro-F1 always uses all three labels, including absent classes. A perfectly classified
SAFE-only population therefore has 0.3333, not 1.0, under this frozen implementation.
The perfect controlled action-state results are narrow synthetic successes, not evidence
of real-call generalization. Financial OOD SAFE FPR is 55.56%; Chichewa recall is 57.14%;
adversarial recall is 96.25%. They remain visible even though not all are promotion gates.

The [public aggregate receipt](data/ettin-v3-result.json) includes all population counts,
confusion matrices, uncertainty behavior and row-level Wilson intervals. Those intervals
do not account for shared template families, prior-open selection or reference-label error.
The original local evaluation receipt also retains source/category/language breakdowns.
No raw texts or per-example IDs are copied into the public summary.

## Runtime and integrity

- Full warm `Scanner.scan`, batch one on Apple Silicon MPS: p50 **9.36 ms**,
  p95 **11.02 ms**, p99 **11.52 ms**, maximum **16.09 ms**.
- Unchanged seeded 100-row regression sample, three rounds, ten warmups:
  300 timed scans and zero above 20 ms. Includes preprocessing, tokenization, model,
  calibration, evidence/action extraction and synchronization; excludes model load and IPC.
- Five runtime files total **602,030,752 bytes** (602.03 decimal MB), still FP32.
  Complete checkpoint directory including training arguments is 602,036,017 bytes.
  This is not a quantized or physical-mobile package.
- Process peak RSS: 1,248,493,568 bytes. This is not an audited total Metal-memory peak.
- Exact saved-checkpoint SDK parity: **2,634/2,634** development IDs, probability triples
  and calibrated verdicts. All source/config/checkpoint/corpus/ledger bindings verified.
- Independent read-only arithmetic review recomputed every aggregate and subgroup
  confusion/count/recall/FPR/macro-F1 from the text-free ledger, all 18 gates, timing
  percentiles/sample membership and supervisor bindings. This is an AI engineering audit,
  not the required independent human label/privacy review.

864 code tests and full Ruff checks passed before fit. Test success is separate from the
six failed model gates. Frozen code was not modified after fitting began.

## Cached larger-model comparison: descriptive only

Reused the previously frozen 300-row open sample and cached Qwen3.8-27B Q4 predictions;
no new teacher inference. V3 macro-F1 is 0.8077 versus 0.6515 for that fixed zero-shot
baseline; recall is 76/76 versus 65/76, but SAFE false alarms are 7/210 versus 0/210.
This selected short-message diagnostic neither repairs the dialogue failures nor establishes
SOTA. It is not an independent final test, population-wide result or release authorization.

## Evidence and close-out

Local immutable evidence:

- [Training](runs/sg-ettin150m-source-filtered-v3.train.json)
- [All open evaluation and timing trace](runs/sg-ettin150m-source-filtered-v3.open.json)
- [Exact integrity/parity receipt](runs/sg-ettin150m-source-filtered-v3.integrity.json)
- [Cached paired comparison](runs/sg-ettin150m-source-filtered-v3-vs-qwen38-27b-open300-20261010.paired.json)
- [Public text-free aggregate](data/ettin-v3-result.json)

`reports/runs/` and model checkpoints are local ignored artifacts; their links do not
imply public distribution. The committed public aggregate preserves the decision, metrics
and exact bindings without redistributing raw source texts or checkpoint weights.

| Artifact | SHA-256 |
| --- | --- |
| Config | `af97b5e2de2f9450bf88d3deacfa7559c7854b8a15cdc1be605021884f80c086` |
| Prospective freeze | `9ca8b127d493f606d4a4ef8374a0ac70238375b3a2a9c131e1ad1ea32e49b879` |
| Model | `853f56c940a0e06c66f90096230952db15da6d8893c0228168097754cd966871` |
| Calibration | `307eda637a3e22d7ac7d39445e3e2af233784cb9dfe95c0de1159ef5d656302e` |
| Train report | `1d3e9c6a97ccf4f9d2966af8f7277829f0f88d5ebf1548e2beca6c2d78ccb555` |
| Open report | `c4d850f0c18d2ae3795ad87e9c68519a8a91f2142363b2dcd391258aaa69a0b4` |
| Open ledger | `d101380f0b8c84fac10e24f429272c0666740f158eba2650f39f881dd7a5c7e8` |
| Integrity receipt | `c878b71711a93f10d0edb4a151fe0e2c7db1bf827ab80f6f4585d2d1b056d0d1` |

Historical v1/v2 reports, checkpoints, original benchmark labels, 635-row human packet and
96-row dialogue packet remain unchanged. No sealed tests, mobile export, HF upload, new
publisher contact or paid cloud computation occurred.

**Decision:** retain the code, data provenance and negative findings; close the current
model-training attempt. Do not spend on final human certification or packaging this rejected
candidate. Restarting would require a separate, explicitly authorized research question and
materially better independently reviewed supervision—not another automatic tuning loop.
The independent human/source-rights/final-validation requirements remain unmet, but even
completing that paperwork would not repair the measured quality failures.
