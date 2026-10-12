# Final bounded Ettin experiment

Authorized by the owner: “Decide the goal and pursue it.” Frozen before any v3 fit.
The goal is a defensible desktop-first release candidate, with one final attempt and a
clear decision—not an indefinite model search. Open screening is not final release evidence.

## Hypothesis and recipe

Start from the original pinned `jhu-clsp/ettin-encoder-150m` revision, never ScamGuard v1/v2.
Remove the unresolved phone-generator and YouTube training sources and attributable derivatives.
Use `speaker-neutral-action-preserving-v3` to preserve concrete requested actions, safety
boundaries and conversation context. This preprocessing already passed synthetic regression
tests; it has not established model quality.

The source-filtered corpus has 22,811 training rows from 12 source IDs:
13,082 SAFE, 7,038 SCAM and 2,691 UNCERTAIN. All retained row bytes are unchanged.
The development set remains byte-identical: 2,634 rows (2,008 SAFE, 514 SCAM, 112 UNCERTAIN).
The builder removed 1,190 phone-synthetic and 435 YouTube rows; no extra BANKING77,
AI-adjudicated or benchmark examples enter training. Automated privacy-pattern checks
found zero flags; this is not independent privacy or rights clearance.

- One seed: 20261012. Two epochs, batch 16, exactly 2,852 steps at learning rate 0.00002.
- Square-root class-weighted three-class cross-entropy; fresh optimizer and classification head.
- Right truncation, 256 tokens, v3 dialogue policy. No extra objective or selective replay.
- One terminal checkpoint. No best-checkpoint selection, second seed, sweep, resume or repair fit.
- Fit envelope: at most 3,600 seconds including load/tokenization in the runner.
  A process supervisor also caps launch-to-observed-terminal-step at 3,600 seconds.
- Checkpoint save/calibration: at most 900 seconds after terminal-step observation;
  total training process at most 4,500 seconds. Open evaluation process at most 1,800 seconds.
  Timeout termination allows a five-second graceful exit before forced process-group kill.
- Offline cached files only; local single-process MPS. No cloud expense or new model download.

## Frozen evidence and unchanged decision rule

[Configuration](../configs/ettin150m-source-filtered-v3.json) binds all data/model files.
[Corpus receipt](../reports/data/ettin-v3-corpus.json) binds parent sources, byte-preserving
filter, privacy checks, labels, counts and remaining release-review qualifications.
The metadata preflight verifies cached base bytes without a model forward pass. The final
freeze also binds training/SDK/evaluator/verifier/supervisor code and checks zero train overlap
with development and all 21 open populations by ID, normalized text and family ID.
This does not establish absence of semantic or base-pretraining contamination.

Calibration uses only development, batch one, unpadded input, FP32 device-local Torch softmax,
and the serving-identical stable threshold procedure. SAFE risk remains 0.2. There is no
threshold refit after opening predictions. The saved checkpoint is evaluated through the real
SDK; every development ID, probability triple and verdict must exactly match calibration.

Keep all original 18 gates: development/regression SCAM recall >=97%, SAFE FPR <=2%,
eligible regression-category recall >=97%, required dialogue recall >=97% and SAFE FPR <=2%,
full warm desktop `Scanner.scan` p95 <=20 ms, and zero invalid evidence spans.
Latency uses the unchanged 100 IDs, three rounds and ten warmups; it excludes load/IPC.
The 0.94 three-class macro-F1 target stays an explicitly reported stretch. All 21 open
populations, confusion matrices, uncertainty behavior, source and language diagnostics must
be reported even if quality fails. Phone/BothBosu labels and denominators do not change.
The 96-case AI review is neither human adjudication nor replacement truth.

Any failed mandatory gate, incomplete required evaluation, timeout, numerical failure or
invalid protocol rejects this attempt. Do not automatically train again. Passing these gates
would qualify only for separately frozen final validation, not establish SOTA or authorize HF.

## Commands and one-attempt receipts

Run from the repository root using its existing virtual environment:

```bash
rtk proxy .venv/bin/python -m training.run_bounded_encoder --config configs/ettin150m-source-filtered-v3.json --metadata-preflight
rtk proxy .venv/bin/python scripts/run_ettin_v3_once.py freeze --config configs/ettin150m-source-filtered-v3.json --freeze reports/runs/sg-ettin150m-source-filtered-v3.freeze.json
rtk proxy .venv/bin/python scripts/run_ettin_v3_once.py train --config configs/ettin150m-source-filtered-v3.json --freeze reports/runs/sg-ettin150m-source-filtered-v3.freeze.json
rtk proxy .venv/bin/python scripts/run_ettin_v3_once.py evaluate --config configs/ettin150m-source-filtered-v3.json --freeze reports/runs/sg-ettin150m-source-filtered-v3.freeze.json
rtk proxy .venv/bin/python scripts/verify_ettin_experiment.py --config configs/ettin150m-source-filtered-v3.json --output reports/runs/sg-ettin150m-source-filtered-v3.integrity.json
```

These are ordered phases, not commands to rerun blindly. Existing attempts and outputs are
never overwritten. Evaluation requires the successful, matching training supervisor receipt,
not just a checkpoint or a child process's completion claim.

## Publication remains a separate decision

Retained sources are previously admitted research inputs, not newly certified rights-cleared
data. Independent human label/privacy review, source-specific downstream-use decisions,
untouched final validation, package/runtime parity and normal HF authentication remain release
requirements. The old 635-row packet does not automatically certify the new corpus.
No sealed inputs, model upload, automatic account creation, SOTA claim or physical-mobile claim
are authorized by this local experiment. If screening fails, publish the research findings in
the repository and close this attempt; do not label the checkpoint production-ready.
