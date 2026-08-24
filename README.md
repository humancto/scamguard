# ScamGuard

ScamGuard is a local-first experiment in small, safety-calibrated models for detecting scams,
phishing, impersonation, credential theft, and manipulative payment requests. The product contract
is deliberately stricter than a spam filter: it returns `SAFE`, `UNCERTAIN`, or `SCAM`, quotes the
textual evidence it actually found, and recommends a reversible next action.

The experiment is quality-first. Model size is minimized only after candidates clear the same
contamination-controlled benchmark.

## Current model ladder

| Track | Role | Approximate mobile package | Decision |
|---|---|---:|---|
| TF-IDF logistic regression | cheap reproducible floor | 2.7 MB measured | 1.16 ms p95; 58.1% test recall, gate failed |
| ModernBERT-base, 149M (schema v6) | historical calibrated classifier | 574 MiB FP32 artifact | 92.67% test recall / 5.10% FPR; 19.06 ms MPS-forward p95 |
| ModernBERT-base, 149M (schema v9) | fast-router candidate | 574 MiB FP32 artifact | 99.83% regression recall / 6.30% FPR; 16.67 ms MPS-forward p95; sole-detector gates failed |
| ModernBERT-large, 395M (schema v9) | historical capacity control | 1.48 GiB FP32 artifact | 99.83% regression recall / 0.23% FPR, but 0/72 held identity family; 26.78 ms p95; rejected |
| ModernBERT-base, 149M (schema v10) | rejected dialogue ablation | 574 MiB FP32 artifact | Taskmaster FPR 0%, but BothBosu scam recall 13.48% and regression FPR 11.23%; rejected |
| ModernBERT-base, 149M (schema v11) | rejected fast-router ablation | 574 MiB FP32 artifact | 97.86% dev recall / 1.99% FPR and 15.55 ms end-to-end p95, but 11.57% regression FPR; rejected |
| ModernBERT-base, 149M (schema v12) | rejected counterfactual ablation | 574 MiB FP32 artifact | 99.49% regression recall / 4.18% FPR and 14.27 ms end-to-end p95, but 0/72 held identity-family recall; rejected |
| ModernBERT-base, 149M (schema v13 dose-16) | current fast-path research candidate | 602.5 MB complete FP32 Core ML pack | exact open-set verdict parity and 5.65 ms Core ML p95 on Mac; policy clears open short-message binary gates, but dialogue and macro F1 still fail |
| ModernBERT-base, 149M (schema v14 real dialogue) | rejected positive-only call ablation | 602.0 MB training artifact | new real-call recall 34.29%→100%, but regression SAFE FPR 8.48% and dialogue SAFE FPR 73.20%; rejected before export |
| ModernBERT-base, 149M (schema v15 legitimate openings) | rejected matched-negative call ablation | 602.0 MB training artifact | AppTek FPR 15.52%, regression FPR 18.84%, BothBosu 69.50% recall / 35.29% FPR; rejected despite 12.12 ms p95 |
| ModernBERT-base, 149M (schema v17 pair retention) | rejected paired-action ablation | 602.0 MB training artifact | BothBosu recall reached 98.58%, but SAFE FPR was 23.53% and held-pair recall was 70.83%; rejected |
| ModernBERT-base, 149M (schema v18 action retention) | rejected stronger paired-action ablation | 602.0 MB training artifact | held pairs reached 100% recall / 0% FPR, but regression FPR was 2.23% and BothBosu was 73.76% recall / 15.69% FPR |
| ModernBERT-base, 149M (schema v19 long windows) | rejected length-matched call ablation | 602.0 MB training artifact | long pairs and Taskmaster passed, but regression FPR was 3.15% and BothBosu was 92.91% recall / 42.48% FPR; 21.51 ms PyTorch p95 |
| ModernBERT-base, 149M (schema v20 action states) | rejected multi-task teacher | 602.0 MB training artifact | dev/regression and controlled states passed, but BothBosu was 93.62% recall / 42.48% FPR; rejected |
| ModernBERT-base, 149M (schema v21 human calls) | rejected full-dose call teacher | 602.0 MB training artifact | regression FPR 4.64%, Harper SAFE FPR 4.24%, BothBosu 90.07% recall / 39.22% FPR; rejected |
| ModernBERT-base, 149M (schema v22 service evidence) | rejected conservative teacher | 602.1 MB training artifact | 20/29 gates passed; regression FPR 0.63% and BothBosu FPR 1.96%, but BothBosu recall collapsed to 41.13%; rejected |
| ModernBERT-base, 149M (schema v23 evidence compaction) | rejected bounded-evidence teacher | 602.1 MB training artifact | 18/36 gates passed; 16.97 ms PyTorch/MPS p95, but MultiDoGO SAFE FPR 23.10% and BothBosu 77.30% recall / 13.73% FPR |
| Qwen3.5-0.8B, BF16 base / Q4 control | untouched runtime and capacity control | 537 MiB verified Q4_0 plus 5.27 MB portable arm64 runner | product-shaped BF16 base is 30.32% recall / 2.52% FPR / 0.4295 macro F1; hash-verified public SDK path is 39.60 ms p95 with no Transformers runtime |
| Qwen3.5-0.8B, schema-v24 AI-internal LoRA | trained specialist; rejected before quantization | 41.3 MB adapter plus BF16 base | primary regression is 99.66% recall / 0.115% FPR, but macro F1 is 0.7407, MultiDoGO complete-call FPR is 5.69%, and BothBosu is 65.96% recall / 3.27% FPR; 29/39 gates pass |
| ModernBERT schema v23 + Qwen 0.8B base | rejected routed control | 4.68% test escalation; 1.13 GB process peak RSS | exact product-shape parity; fast-path p95 10.71 ms and routed p95 17.26 ms, but p99 190.79 ms, escalated p95 216.66 ms, and macro F1 0.7730; rejected |
| Qwen3.5-2B, BF16 LoRA reference | high-recall teacher/explainer | 83 MiB adapter plus 4.19 GiB base | 100% test recall / 4.52% FPR; 354.8/579.9 ms median/p95; gates failed |
| Qwen3.5-4B, BF16 LoRA | rejected sole detector; possible teacher | 9.21 GB measured | historical core is strong, but selection dialogue is 92.91% recall / 24.84% FPR and Taskmaster FPR is 4.44% |
| Qwen3.5-4B, Q4/Q5/Q6 GGUF | deployable escalation candidates | 2.71/3.07/3.46 GB measured | Q4 and Q5 rejected after regression loss; Q6 is frozen on dev only |
| Qwen3.5-9B, 4-bit | desktop teacher/upper bound | roughly 6 GB; measure after export | distillation source if needed |

The final release can be a hybrid: the encoder supplies calibrated risk and the Qwen specialist is
invoked only for uncertain cases. A smaller artifact wins only if it meets the exact same gates.
The 4B candidate is mandatory when 2B fails a gate; the 9B model is a desktop teacher, not the
default product payload.

The 0.8B train/merge/quantize path and fail-closed Hugging Face publication contract are in
[docs/HUGGING_FACE_RELEASE.md](docs/HUGGING_FACE_RELEASE.md). Training uses the official
Transformers safetensors checkpoint; GGUF is produced only after the selected adapter is merged.
The untouched 0.8B control, including category failures, publisher-held behavior, latency, and
artifact hashes, is in
[reports/QWEN08_BASE_SCHEMA24_BASELINE.md](reports/QWEN08_BASE_SCHEMA24_BASELINE.md).
The verified upstream Q4_0 artifact, Metal/CPU sweep, exact three-verdict scorer, memory results,
and routed deployment decision are in
[reports/QWEN08_Q4_RUNTIME_FLOOR.md](reports/QWEN08_Q4_RUNTIME_FLOOR.md).
The strict text-free ledger join, development-only routing freeze, and rejected base-specialist
control are in [reports/ROUTED_BASE_DIAGNOSTIC.md](reports/ROUTED_BASE_DIAGNOSTIC.md).
The three-pass persistent MPS trace, tail latency, scoring-batch mismatch, and exact parity failure
are in [reports/ROUTED_BASE_RUNTIME.md](reports/ROUTED_BASE_RUNTIME.md).
The trained 0.8B result, corrected branch-token scorer, complete gate rejection, and split-safe
call-robustness continuation are in
[reports/QWEN08_AI_INTERNAL_BRANCH_EXPERIMENT.md](reports/QWEN08_AI_INTERNAL_BRANCH_EXPERIMENT.md).

The complete 2B evaluation, confidence intervals, OOD failures, paired DeBERTa comparison, latency
scope, and artifact hashes are in [reports/QWEN2B_REFERENCE.md](reports/QWEN2B_REFERENCE.md).
The historical schema-v6 4B adapter's new, selection-only multi-turn diagnostics are in
[reports/QWEN4B_DIALOGUE_DIAGNOSTIC.md](reports/QWEN4B_DIALOGUE_DIAGNOSTIC.md).
The historical 149M schema-v9 evaluation and held-out-family failure analysis are in
[reports/ENCODER_SCHEMA9.md](reports/ENCODER_SCHEMA9.md).
The completed schema-v11 result, dialogue-policy ablations, and false-positive family audit are in
[reports/ENCODER_SCHEMA11.md](reports/ENCODER_SCHEMA11.md).
The rejected schema-v12 dataset decision and immutable identities are in
[reports/DATASET_SCHEMA12.md](reports/DATASET_SCHEMA12.md). The isolated schema-v13 dose result and
post-hoc policy audit are in [reports/ENCODER_SCHEMA13.md](reports/ENCODER_SCHEMA13.md); schema v11
and the failed schema-v10 experiment remain documented in
[reports/DATASET_SCHEMA11.md](reports/DATASET_SCHEMA11.md) and
[reports/DATASET_SCHEMA10.md](reports/DATASET_SCHEMA10.md).
The CC0 real-call source admission and rejected schema-v14 checkpoint are documented in
[reports/DATASET_SCHEMA14_REAL_DIALOGUE.md](reports/DATASET_SCHEMA14_REAL_DIALOGUE.md).
The text-only legitimate-call benchmark, schema-v13/v14 false-positive localization, and sealed
AppTek partition are documented in
[reports/APPTEK_CALL_BENCHMARK.md](reports/APPTEK_CALL_BENCHMARK.md).
The 256-row matched-opening experiment and its rejection are documented in
[reports/DATASET_SCHEMA15_LEGITIMATE_OPENINGS.md](reports/DATASET_SCHEMA15_LEGITIMATE_OPENINGS.md).
The full FP32/INT8 ONNX fidelity, latency, memory, and rejection record is in
[reports/ONNX_SCHEMA13.md](reports/ONNX_SCHEMA13.md).
The native FP32/FP16 Core ML conversion, full parity, latency, and rejection record is in
[reports/COREML_SCHEMA13.md](reports/COREML_SCHEMA13.md).
The successive pair, long-window, and action-state decisions are in
[reports/ENCODER_SCHEMA17_PAIR_RETENTION.md](reports/ENCODER_SCHEMA17_PAIR_RETENTION.md),
[reports/ENCODER_SCHEMA18_ACTION_RETENTION.md](reports/ENCODER_SCHEMA18_ACTION_RETENTION.md),
[reports/ENCODER_SCHEMA19_WINDOWMIX.md](reports/ENCODER_SCHEMA19_WINDOWMIX.md), and
[reports/DATASET_SCHEMA20_ACTION_STATES.md](reports/DATASET_SCHEMA20_ACTION_STATES.md). The rejected
human-call ablation, bounded schema-v22 dataset, and rejected schema-v22 result are in
[reports/ENCODER_SCHEMA21_HUMAN_CALLS.md](reports/ENCODER_SCHEMA21_HUMAN_CALLS.md) and
[reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md](reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md), with
the measured result in
[reports/ENCODER_SCHEMA22_SERVICE_EVIDENCE.md](reports/ENCODER_SCHEMA22_SERVICE_EVIDENCE.md).
The frozen schema-v23 evidence-compaction dataset, FTC pattern boundary, licensed-real accounting,
and preflight identities are in
[reports/DATASET_SCHEMA23_EVIDENCE_COMPACTION.md](reports/DATASET_SCHEMA23_EVIDENCE_COMPACTION.md);
the rejected 18/36-gate result, controlled compactor diagnosis, and next data-semantics experiment
are in
[reports/ENCODER_SCHEMA23_EVIDENCE_COMPACTION.md](reports/ENCODER_SCHEMA23_EVIDENCE_COMPACTION.md).
The audited publisher-annotation curriculum, schema-24 row accounting, held-slice baseline, and
Qwen SFT preflight are in
[reports/SCHEMA24_ANNOTATION_BASELINE.md](reports/SCHEMA24_ANNOTATION_BASELINE.md).

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra train --extra dev
make data
make baseline
uv run scamguard scan --model artifacts/sg-linear-v0.3.joblib \
  "Urgent: send the verification code to stop this wire transfer"
```

After building the schema-v13 ONNX pack, run the actual neural candidate locally:

```bash
uv run --extra onnx --extra neural scamguard scan \
  --model artifacts/onnx/schema13-dynamic-pack/scamguard-modernbert-seqdynamic-fp32.onnx \
  "Urgent: send the verification code to stop this wire transfer"
```

Run the localhost-only demo:

```bash
uv run scamguard demo --model artifacts/sg-linear-v0.3.joblib
```

The SDK is a single call:

```python
from scamguard import scan

result = scan("Paste a suspicious message here", model_path="artifacts/sg-linear-v0.3.joblib")
print(result.to_dict())
```

The GGUF path uses the same API. A runtime pack contains the quantized model, statically linked
native runner, calibration, and frozen prompt framing; every component is hash-checked before the
model is loaded. Keep a `Scanner` open to reuse the model and prefix cache across messages:

```bash
make qwen-08b-base-runtime-pack
make qwen-08b-base-runtime-pack-benchmark
```

```python
from scamguard import Scanner

with Scanner(model_path="artifacts/runtime-packs/qwen35-08b-upstream-q4-control") as guard:
    result = guard.scan("Paste a suspicious message here")
    print(result.to_dict())
```

That packaged artifact is the untouched upstream runtime control, not a trained ScamGuard release.
It is deliberately marked `publication_authorized: false`; the final trained Q4/Q5 pack must repeat
all quality, parity, desktop, and physical-mobile gates.

The built-in heuristic exists only for SDK smoke tests. It is visibly identified as
`heuristic-unbenchmarked-v0` and must not be presented as a trained model.

Exact training, native arm64 llama.cpp, merge, Q4_K_M export, and post-quantization evaluation
commands are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The measured ONNX/Core ML/Android export gates are in
[docs/DEPLOYMENT_PLAN.md](docs/DEPLOYMENT_PLAN.md).
The physical iOS/Android raw-trace receipt and verifier are in
[docs/MOBILE_BENCHMARK_PROTOCOL.md](docs/MOBILE_BENCHMARK_PROTOCOL.md).
The shared in-process C ABI, iOS XCFramework build, Swift wrapper, Android JNI build, and Kotlin
wrapper are in [docs/MOBILE_RUNTIME_INTEGRATION.md](docs/MOBILE_RUNTIME_INTEGRATION.md).

## ScamBench gates

Every candidate is evaluated at one threshold fitted on development data, then frozen:

- scam recall at least 97%;
- clean-message false-positive rate no more than 2%;
- macro F1 above 0.94 as a stretch target;
- calibration reported as Brier score and 15-bin ECE;
- exact and normalized-template families isolated between train/dev/test;
- a separately sourced financial-scam set used only as an out-of-domain holdout;
- a 488-row WSPR holdout with no SAFE rows (409 SCAM and 79 UNCERTAIN);
- 2,300 disjoint forum OOD rows, including 100 SAFE controls, plus 2,079
  realistic-placeholder derivatives;
- a 677-row CC-BY-4.0 Chichewa external diagnostic, privacy-normalized and collapsed to one row per
  family without counting augmented rows as independently collected real data;
- a 1,343-row Apache-2.0 multi-turn telephone-scam diagnostic, reduced from 1,600 upstream
  synthetic dialogues and split by family into 294 selection and 1,049 prediction-sealed rows;
- a CC0 real-scam-call-derived corpus with 161 early training windows, 70 source-family-held open
  validation windows, and 80 prediction-sealed OOD windows; the first positive-only training dose
  is rejected for false-positive regression;
- a CC-BY-SA-4.0 AppTek legitimate-call roleplay benchmark with 348 selection windows and 1,396
  prediction-sealed windows, split by shared-speaker/call components and never used for fitting;
- 600 CC-BY-4.0 human-authored Taskmaster transactional dialogues used as weak SAFE training
  examples, plus a disjoint 450-dialogue selection slice;
- a sealed 1,820-row newly sourced Portuguese/Mozambican mobile-money test with 526 SCAM and
  1,294 SAFE family representatives;
- desktop single-message p95 verdict latency at most 20 ms, plus separately measured mobile latency;
- median/p95 latency, routed-model escalation rate, peak memory, and artifact size published.

Scores from a different dataset are context, not a valid head-to-head comparison. See
[docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md) and the
[dataset-size decision](docs/DATASET_SIZE_DECISION.md).

## Data policy

The build uses hash-pinned datasets, provenance-tagged synthetic variants, hard negatives, exact
deduplication, and near-template split isolation. ScamBench schema v12 validates 28,125 unique
processed examples with no family leakage. Its 14,446-row training split retains the unchanged
schema-v9 dev and regression sets, uses 600 human-authored Taskmaster roleplay dialogues, and admits
1,495 of 1,536 generated paired synthetic conversations. The processed core and named diagnostics
now contain 23,798 rows: 13,127 naturally occurring licensed-source rows, 600 human-authored
crowdsourced roleplays reported separately, and 10,071 controlled synthetic rows.
Training includes 1,000 evidence-grounded public-forum scam reports, 100 ambiguous forum messages,
148 real forum SAFE controls, 1,913 WSPR campaign representatives, licensed SMS data, and the three
synthetic curricula. Synthetic v5 contributes 5,040 original English scenario variants plus 3,024 benign
lookalikes across seven languages so multilingual input does not become a scam shortcut. Its 24 new
paired families cover official-advisory gaps such as protect-your-money transfers, task scams,
government-benefit identity theft, family-bail secrecy, and reshipping jobs. A development-only
0/1k/3k/all-5,672 learning curve selected the 1k quality-first cap without opening test or OOD outcomes.
Synthetic dialogue v2 adds 12 balanced scam/legitimate scenarios after the first dialogue
correction exposed a source-format shortcut. A versioned speaker-neutral transform removes corpus
role-name cues and retains complete recent turns. The external BothBosu OOD partition and the
1,820-row MOZ holdout were initially prediction-sealed. BothBosu was later opened for model-error
diagnosis and is now reported only as a prior-open regression diagnostic; the MOZ holdout remains
sealed.
Synthetic counterfactual v1 adds 512 balanced train-only messages in eight paired families after
schema v11's frozen regression ledger exposed false alarms on known-contact transfers, ordinary
family updates, in-platform marketplace activity, and official-app reviews. The scam counterparts
change the trust boundary or requested action; none copies a regression or external-diagnostic row.
This is an ablation, not an accepted dataset improvement: the schema-v12 run reduced regression FPR
from 11.57% to 4.18% but generalized the repair into an unsafe identity-scam veto and missed all 72
rows in the unchanged `identity_case_callback` development family. See
[`reports/ENCODER_SCHEMA12.md`](reports/ENCODER_SCHEMA12.md) for the frozen failure analysis.
Schema v13 reduces the correction to 128 rows. Its model-only result is still rejected, while a
separately reported deterministic trusted-channel policy clears the open short-message binary gates
without changing dialogue results. See [`reports/ENCODER_SCHEMA13.md`](reports/ENCODER_SCHEMA13.md).
Schema v14 then adds 161 early windows from 145 CC0 scam-call source families. It completely repairs
the new source-family-held positive recall diagnostic but creates a broad call/prose shortcut,
doubles regression false positives, and is rejected. The 80-window OOD partition remains sealed.
The AppTek open legitimate-call slice localizes that shortcut to early service-call openings:
schema v13 falsely flags 31/348 SAFE windows and schema v14 flags 77/348, with zero false positives
from either model on all 174 recent windows. Schema v15 is therefore a 256-row matched-negative
ablation spanning 16 service scenarios and four opening structures. It copies no AppTek text,
contains no explicit anti-scam safety cues, and leaves AppTek's 1,396-window OOD partition sealed.
That experiment is also rejected: it lowers AppTek FPR relative to schema v14 but remains worse
than schema v13, while unchanged-regression FPR rises to 18.84%. No export or sealed evaluation is
performed for schema v15.
Schema v16 changes the objective without adding rows: it initializes from schema v13, preserves
schema-v13 logits on all 14,062 inherited examples, and square-root balances the fixed schema-v15
sources. It improves development and unchanged-regression behavior, but is still rejected: AppTek
SAFE FPR rises to 30.17%, BothBosu SAFE FPR rises to 69.93%, and desktop CPU end-to-end p95 is
32.31 ms. The result proves that retention and source balancing do not remove the call-opening
shortcut. No export or sealed evaluation is performed; the next increment must use
structure-matched minimal contrasts. See
[`reports/ENCODER_SCHEMA16_RETENTION.md`](reports/ENCODER_SCHEMA16_RETENTION.md).
Schema v17 is the frozen next-data increment, not a model claim. It starts again from schema v14 and
adds 576 balanced structure-matched minimal-contrast rows: each pair shares four call turns and
changes only the final legitimate/risky action. Another 192 rows are held out by four complete
service scenarios. The resulting 14,799-row training set validates at 28,670 unique processed rows;
all new dialogues fit within 114 ModernBERT tokens. See
[`reports/DATASET_SCHEMA17_CALL_MINIMAL_PAIRS.md`](reports/DATASET_SCHEMA17_CALL_MINIMAL_PAIRS.md).
Schemas v18 and v19 prove that larger and length-matched binary action pairs can reach perfect
held-pair ordering without producing a transferable absolute call boundary. Schema v20 therefore
changes the target instead of adding undifferentiated volume. Its 21,234-row training mix includes
6,144 four-state long contrasts whose byte-identical histories end in routine SAFE, independently
verified SAFE, unresolved UNCERTAIN, or caller-controlled SCAM actions. Seven dense action/context
targets support a multi-task teacher; 2,048 rows from four complete service domains remain held out.
The full frozen composition, hashes, source rights, and latency rationale are in
[`reports/DATASET_SCHEMA20_ACTION_STATES.md`](reports/DATASET_SCHEMA20_ACTION_STATES.md).
Schema v21 then added a full-weight HarperValleyBank roleplay dose. It passed controlled states but
failed unchanged-regression, original-call, and BothBosu gates, so no export or sealed evaluation
was run. Schema v22 returns to schema v20 and adds 1,790 licensed MultiDoGO human-authored service
views plus 1,184 human-grounded state variants. Action states train on airline, fast food, finance,
and media while insurance and software stay validation-only. The 24,208-row teacher recipe and all
source, overlap, token-window, and gate identities are frozen in
[`reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md`](reports/DATASET_SCHEMA22_SERVICE_EVIDENCE.md).
Schema v23 returns to schema v20, adds only 215 licensed MultiDoGO action turns and 1,216 controlled
state rows, and applies a participant-balanced evidence-plus-recent input contract before the
256-token window. The compactor helps in controlled post-hoc diagnostics, but the trained candidate
passes only 18/36 frozen gates: MultiDoGO original-call SAFE FPR is 23.10% and prior-open BothBosu
is 77.30% recall at 13.73% FPR. No external selection, export, or sealed evaluation was run. Its
frozen dataset and measured rejection are documented in
[`reports/DATASET_SCHEMA23_EVIDENCE_COMPACTION.md`](reports/DATASET_SCHEMA23_EVIDENCE_COMPACTION.md)
and
[`reports/ENCODER_SCHEMA23_EVIDENCE_COMPACTION.md`](reports/ENCODER_SCHEMA23_EVIDENCE_COMPACTION.md).
The schema-v24 path begins with `make multidogo-annotation-curriculum`. All 36 pinned publisher
intent/slot files are now materialized and covered by a text-free integrity audit. Their IDs do not
join to the separately released unannotated collection, so the builder selects privacy-normalized
turn-level customer rows directly, preserves the publisher's train/dev/test boundary, and stratifies difficult
legitimate-service examples without misrepresenting intent labels as independently reviewed scam
labels. `make schema24-annotated-hard-negatives` then removes parent/held collisions and whole
families with exact or radius-six near overlap before admitting publisher-train rows. Schema 24
also replaces contextual access codes, account fragments, postal codes, and credential-like values
with typed placeholders before repeating contamination control; the validator rejects any residual
value or missing row-level normalization record. The separate `schema24-audit` workbook is bound to
that exact manifest and a frozen blind-review rubric. `make schema24-audit-bundle` produces a
deterministic four-file ZIP for an independent reviewer; the ZIP contains only opaque row IDs, message
text in a deterministic shuffled order, blank decision fields, the frozen rubric, and a
dependency-free localhost app. Project
labels, source labels, source names, categories, splits, and model outputs are absent from the
artifact. `make schema24-audit-handoff-preflight` black-box checks the production ZIP with isolated
Python, an ephemeral loopback server, the public API schema, and a disposable save/resume cycle;
it never writes a decision back to the source ZIP. After the reviewer returns the completed blind
CSV, `make schema24-audit-import` verifies
the ZIP, protocol, immutable messages, IDs, canonical audit manifest, and dataset manifest before
joining decisions to the sealed answer key. `schema24-audit-check` then rejects every incomplete
decision, label disagreement, sensitive-data finding, rubric change, workbook change, or dataset
change by reconstructing the blind import in a temporary directory and byte-checking the reviewed
workbook without overwriting its provenance report. Its text-free report
adds Wilson-bound agreement, Cohen's kappa, confusion counts, and source/label diagnostics. The
schema-24 dataset validates, but the release training freeze still fails closed because the
independent human workbook is 0/635 complete.
Generic spam and evidence-free wrong-number openers are `UNCERTAIN`; defensive scam education and
standalone authentication-code notifications are `SAFE` unless the text itself adds a risky
external action. Source-reported positives without strong message-local fraud evidence are
`UNCERTAIN`, not SCAM. A separate answer-key-free AI-internal review is now 635/635 complete. It
found 83.46% agreement (95% Wilson lower bound 80.38%, Cohen's kappa 0.655), 105 label
disagreements, and 22 sensitive rows. `make schema24-ai-internal-audit` reproduces the text-free
comparison report while hard-coding `release_gate_passed=false` and
`publication_authorized=false`. `make schema24-ai-internal-overlay` preserves the canonical corpus,
quarantines those 22 rows, applies 98 non-sensitive corrections in a separate exploratory overlay,
and rebuilds evidence-grounded Qwen SFT data. `make qwen-08b-ai-internal-freeze` binds that overlay,
the 640-token audit, and the measured 4x4 batch geometry into a non-release experiment config.
This allows useful model work to continue without misrepresenting the assistant as an independent
human reviewer. The original 635-row workbook still needs independent human decisions, and the
repository and canonical audit workbook must not be given to that reviewer because they contain the
answer key.

The previous schema-v6, schema-v9, and rejected schema-v10/v11 model reports remain historical regression
evidence; they are not relabeled as schema-v12 results. Schema v8 adds a separate,
prediction-sealed 1,820-row MOZ-Smishing holdout after privacy
normalization, label-conflict quarantine, one-per-family collapse, and overlap removal against every
previously processed benchmark. It is local-evaluation-only and excluded from training/public row
redistribution because the publisher's model-oriented OpenRAIL tag lacks a dataset-specific license
file. See [the online source research](reports/ONLINE_SOURCE_RESEARCH.md) for measured admissions
and rejections. Its 2026-08-22 contingency refresh separately records five new Hub/paper candidates
with zero admissions, preserving schema v24 while the human audit is active; dataset size is never
inflated with unlicensed GitHub collections or duplicate repackagings.

Schema v6 replaces real-source email addresses, long phone-like values, and long account-like digit
sequences with typed placeholders before IDs, family clustering, splitting, or fitting. The validator
fails closed if any such value survives in a real-source parent row.

Reddit user content is excluded from training and redistribution because Reddit's current Data
API Terms prohibit ML/AI training on user content without rightsholder permission. Public scam
reports may inform taxonomy research but cannot become rows in this repository under those terms.
See [data/README.md](data/README.md) for sources and [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md)
for the acceptance policy.

## Repository map

```text
src/scamguard/       local SDK, output schema, signals, CLI, demo
scripts/             hash-pinned fetch, generation, build, validation
training/            model-specific training and evaluation
configs/             immutable experiment configurations and data hashes
tests/               contract and metric tests
data/                source documentation; downloaded/generated rows are ignored
artifacts/           local model outputs; large exports are ignored
reports/             benchmark protocol and local run records
```

## Safety boundary

ScamGuard is decision support, not a guarantee that a message is safe or fraudulent. `UNCERTAIN`
is a first-class result. Evidence spans are extracted from the input; the SDK does not generate a
hidden chain of thought. Do not automatically delete messages, contact senders, make payments, or
submit reports without the user's review.

## License

Source code is Apache-2.0. Dataset rows retain their source licenses and attribution requirements;
model artifacts require a separate release card listing every included source.
