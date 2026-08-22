# Encoder schema 18: evidence-action retention result

## Decision

**Rejected. Do not run external selection benchmarks, export this checkpoint, or evaluate
sealed OOD splits.**

Schema 18 tested whether a much larger action-counterfactual curriculum, stronger retention,
repeated complete pairs, and latest-context truncation could teach the schema-13 ModernBERT
checkpoint the difference between an ordinary service call and the same call ending in a
harmful external action. It learned the new paired task perfectly, but it did not retain a
usable absolute boundary on established real-dialogue and regression gates.

The experiment was frozen and pushed before training in commits `fa330ee` and `728d935`. Its
configuration SHA-256 was
`87f60eaafaa03bdad32c08ca72e9e380d2a45f19df52c9180e4e9bde7f373e5d`.

The first execution stopped at step 1,097 because the loss rejected two complete repeated
copies of a family when they landed in one batch. It produced no accepted result. Commit
`27abba8`, pushed before the successful rerun, made the loss average repeated SAFE and SCAM
margins while continuing to reject odd or label-imbalanced families. Lint and all 130 tests
passed before the clean restart, which began again from schema 13 rather than partial weights.

## Frozen recipe

- Initialization: `sg-modernbert-schema13-dose16`
  (`31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99`)
- Unique training rows: 17,295
- Teacher anchors: 14,062 text-free schema-13 logits
- New unanchored rows: 161 licensed real scam-call rows plus 3,072 original synthetic pair rows
- Synthetic source: 2,048 balanced pair families / 4,096 rows across 16 scenarios, four
  dialogue structures, four context frames, and eight harmful-action mechanisms
- Pair split: 1,536 train families and 512 validation families from four entirely held-out
  scenarios
- Effective epoch: all legacy rows once and every train pair twice, or 20,367 sampled rows
- Training: one epoch, batch 16, learning rate `5e-6`, retention weight 4, pair weight 2,
  pair margin 3, no source replacement sampling
- Input contract: speaker-neutral text, latest 256 tokens (`truncation_side=left`)
- Threshold: fitted only on the unchanged development SAFE/SCAM rows at the 2% FPR cap

Pair examples were 87–116 tokens (median 102, p95 110), so none were truncated. Preflight
verified input hashes, exact shared-context hashes, one SAFE plus one SCAM row per family,
held-out scenario separation, teacher coverage, and sealed-data flags.

## Result

| Gate | Required | Schema 13 baseline | Schema 18 | Pass |
|---|---:|---:|---:|:---:|
| Development scam recall | >=97% | 95.91% | 99.42% | Yes |
| Development FPR | <=2% | 1.94% | 1.94% | Yes |
| Regression scam recall | >=97% | 99.32% | 99.49% | Yes |
| Regression FPR | <=2% | 4.18% | 2.23% | **No** |
| BothBosu latest-window recall | >=97% | not frozen for this policy | 73.76% | **No** |
| BothBosu latest-window FPR | <=2% | not frozen for this policy | 15.69% | **No** |
| Taskmaster FPR | <=2% | not frozen for this policy | 0.22% | Yes |
| Paired validation recall | >=97% | not measured on schema 18 pairs | 100.00% | Yes |
| Paired validation FPR | <=2% | not measured on schema 18 pairs | 0.00% | Yes |
| Paired validation ordering | 100% | not measured on schema 18 pairs | 100.00% | Yes |
| PyTorch desktop end-to-end p95 | diagnostic | 55.65 ms | 33.11 ms | Not a release result |

The unseen 512-family paired holdout had a mean SCAM-probability gap of 0.9253, p05 of
0.8763, and minimum of 0.7825. This is a decisive improvement on the targeted action
distinction, not a marginal threshold effect.

The regression miss is small in absolute terms but still a failure under the frozen contract.
The BothBosu miss is decisive: 37 of 141 scams were missed and 24 of 153 safe calls were
flagged. The model's argmax confusion was 47 SAFE-to-SCAM errors and 20 SCAM-to-SAFE errors.

The run took 1,067.70 seconds on Apple MPS. The PyTorch artifact is 602,034,366 bytes. Its
weight SHA-256 is
`3d129e9bd913849d65e76cc69a3b835cf4aa3a85ca98e351b39dd31c12039c97`; calibration
SHA-256 is `29e06286b9502d97fb9ff8129168e96ecc6aa9fb1ebf16550ea6e2a0d3ad4853`;
and run-report SHA-256 is
`e0bc1a2dcebfc7a102759f316f93aa8aca313d05be4910e00ccbcf9e018d19a9`.

No AppTek, YouTube, Core ML, ONNX, or sealed-OOD result was run for this rejected checkpoint.

## What the experiment established

The paired curriculum is now sufficiently expressive and sufficiently optimized. Unlike
schema 17, schema 18 achieved a perfect relative and absolute boundary on entirely held-out
paired scenarios. More copies of the same short-pair distribution are therefore not the next
move.

The remaining failure is distributional retention on long, real dialogues. BothBosu has a
median length of 319 tokens and 73.13% of its examples exceed the 256-token limit, while the
new action pairs have a median length of 102 and never exceed the limit. Latest-context
truncation fixes the historical error of discarding recent turns, but it cannot make a model
trained mostly on short constructed calls robust to real 256-token dialogue windows.

The 3,072 train-pair rows were also visited twice, giving 6,144 effective synthetic action rows
against only 161 new real call rows. Teacher retention protected the legacy text boundary but
did not supply representative long-call windows. The result therefore learned the intended
action feature while moving the real-dialogue calibration too far.

## Next experiment requirements

Do not continue from schema-18 weights. Initialize again from schema 13 and keep the successful
schema-18 pair generator as one component of a broader window curriculum:

1. Add licensed, conversation-ID-separated normal-call transcripts as 64/128/256-token recent
   windows, with a source-held-out validation partition.
2. Convert licensed real scam calls into the same recent-window representation and retain
   source-level separation; never split windows from one conversation across train and eval.
3. Add long synthetic counterfactual pairs only by padding shared benign context around the
   same final safe/scam action, so context length cannot become a label shortcut.
4. Reduce pair exposure from 2x to 1x or use an explicit real-window/pair mixture, while keeping
   the pair loss and testing that the perfect paired boundary survives.
5. Fit the development threshold under a stricter predeclared cap (for example 1%) to leave
   transfer margin, but do not treat thresholding as a remedy for the BothBosu failure.
6. Add a chunked streaming aggregation contract (recent 128-token windows with overlap) and
   gate both per-window latency and full-call alert behavior before distillation.
7. Keep the same fail-closed order: internal gates, then licensed external selection, then
   export/runtime, and only then a sealed evaluation.
