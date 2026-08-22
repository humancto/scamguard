# Encoder schema 23: evidence-compaction result

## Decision

**Rejected. Do not run external selection, distill, quantize, export, open sealed OOD, or promote
this checkpoint.**

Schema 23 tested a participant-balanced evidence-plus-recent input contract with a deliberately
small dose of licensed MultiDoGO service language and FTC-pattern-grounded action contrasts. The
candidate preserved core short-message quality and met the unoptimized desktop latency diagnostic,
but passed only **18 of 36** frozen internal quality gates. It falsely alerted on 207 of 896 held
MultiDoGO service calls, missed 32 of 141 scams in the prior-open BothBosu diagnostic, and did not
learn jointly reliable action states. The release ladder stopped before external selection.

The complete dataset, recipe, preflight, and gates were frozen and pushed before training in
commits `ff41ef2` and `7c452b4`. The experiment configuration SHA-256 was
`754f60cd90085082a7ca8df7606fecc9951c2bf1f061ffcdb7bdbfabe9dbd535`; the processed manifest
SHA-256 was `c1a4e4d6e8c52a3d499044857ba785459a4f6c9de5ebf1e7f2a8999466224d1a`.

## Frozen recipe

- Initialization: `sg-modernbert-schema20-actionheads-ret4-aw05-vw025-left`
  (`586edb11d1deb511565108f5630fb9581ddd1f25f1f18d759b9a877b809c46ee`)
- Unique fitting rows: 22,665
- Licensed non-synthetic rows: 9,542, comprising 8,134 naturally occurring or real-call-derived
  rows, 1,193 Taskmaster roleplays, and 215 MultiDoGO human agent turns
- Controlled synthetic rows: 13,123, including 860 MultiDoGO-grounded and 356
  FTC-pattern-grounded action states added over the schema-20 parent
- Action-supervised fitting rows: 7,575
- Calibration: 400 rows from 80 fitting- and validation-disjoint MultiDoGO families; auxiliary
  thresholds only
- Held additions: 592 four-state MultiDoGO rows from 148 insurance/software families, 896
  original SAFE MultiDoGO views from 448 families, and 84 FTC-pattern rows from 21 families
- Training: one epoch, 1,417 optimizer steps, batch 16, learning rate `2e-6`, verdict-retention
  weight 4, action loss weight 0.5, and action-row verdict weight 0.25
- Input contract: `speaker-neutral-evidence-recent-v2`, right truncation at 256 tokens
- Product threshold: fitted only on unchanged development SAFE/SCAM rows at the 2% FPR cap

The processed training JSONL SHA-256 was
`b31ae7519b1fffc985a9825301ccacfd6d2603279096b6f102355798e105833b`.

## Frozen gate result

| Gate | Required | Schema 23 | Pass |
|---|---:|---:|:---:|
| Development scam recall | >=97% | 99.61% (512/514) | Yes |
| Development SAFE FPR | <=2% | 1.44% (29/2,008) | Yes |
| Unchanged regression scam recall | >=97% | 100.00% (587/587) | Yes |
| Unchanged regression SAFE FPR | <=2% | 1.03% (18/1,746) | Yes |
| Original state harmful recall | >=97% | 99.61% (510/512) | Yes |
| Original state routine SAFE FPR | <=2% | 1.56% (8/512) | Yes |
| Original state verified SAFE FPR | <=2% | 5.08% (26/512) | **No** |
| Original state unresolved scam rate | <=10% | 24.80% (127/512) | **No** |
| Original state ordered contrasts | >=95% | 86.91% (445/512) | **No** |
| Original state action macro AUC | >=97% | 99.19% | Yes |
| Original state calibrated action exact match | >=90% | 65.28% (1,337/2,048) | **No** |
| FTC holdout harmful recall | >=97% | 100.00% (21/21) | Yes |
| FTC holdout routine SAFE FPR | <=2% | 0.00% (0/21) | Yes |
| FTC holdout verified SAFE FPR | <=2% | 0.00% (0/21) | Yes |
| FTC holdout unresolved scam rate | <=10% | 19.05% (4/21) | **No** |
| FTC holdout ordered contrasts | >=95% | 100.00% (21/21) | Yes |
| FTC holdout action macro AUC | >=97% | 95.91% | **No** |
| FTC holdout calibrated action exact match | >=90% | 48.81% (41/84) | **No** |
| Held MultiDoGO state harmful recall | >=97% | 100.00% (148/148) | Yes |
| Held MultiDoGO state routine SAFE FPR | <=2% | 27.03% (40/148) | **No** |
| Held MultiDoGO state verified SAFE FPR | <=2% | 0.00% (0/148) | Yes |
| Held MultiDoGO state unresolved scam rate | <=10% | 0.68% (1/148) | Yes |
| Held MultiDoGO state ordered contrasts | >=95% | 98.65% (146/148) | Yes |
| Held MultiDoGO state action macro AUC | >=97% | 97.93% | Yes |
| Held MultiDoGO state calibrated action exact match | >=90% | 63.85% (378/592) | **No** |
| MultiDoGO original-call SAFE FPR | <=2% | 23.10% (207/896) | **No** |
| MultiDoGO airline SAFE FPR | <=3% | 12.00% (18/150) | **No** |
| MultiDoGO fast-food SAFE FPR | <=3% | 29.33% (44/150) | **No** |
| MultiDoGO finance SAFE FPR | <=3% | 23.33% (35/150) | **No** |
| MultiDoGO insurance SAFE FPR | <=3% | 33.33% (50/150) | **No** |
| MultiDoGO media SAFE FPR | <=3% | 20.00% (30/150) | **No** |
| MultiDoGO software SAFE FPR | <=3% | 20.55% (30/146) | **No** |
| Long-call SAFE FPR | <=2% | 0.45% (2/447) | Yes |
| Taskmaster SAFE FPR | <=2% | 0.00% (0/450) | Yes |
| Prior-open BothBosu recall | >=97% | 77.30% (109/141) | **No** |
| Prior-open BothBosu SAFE FPR | <=2% | 13.73% (21/153) | **No** |

The candidate therefore passed 18/36 gates. The independent gate report records
`external_selection_authorized=false`, `sealed_evaluation_authorized=false`, and
`distillation_or_export_authorized=false`.

## What the experiment established

Evidence compaction helps. On the same 896 held SAFE MultiDoGO calls, the unchanged schema-20
teacher under its original latest-token contract produced 50.22% FPR. Applying only the new
compactor to that unchanged checkpoint lowered FPR to 38.06%. The schema-23 weights under the old
contract reached 33.82%, and the combined schema-23 path reached 23.10%. These are post-hoc failure
diagnostics, not selection results, but they rule out compaction as the primary cause of the
remaining service-call error.

The bounded fitting dose also moved the sensitivity-specificity frontier in the expected direction:

| Candidate | MultiDoGO SAFE FPR | Prior-open BothBosu recall | Prior-open BothBosu FPR |
|---|---:|---:|---:|
| Schema 20 sensitive teacher | 50.22% post-hoc | 93.62% | 42.48% |
| Schema 22 full service dose | 2.90% | 41.13% | 1.96% |
| Schema 23 bounded dose + compaction | 23.10% | 77.30% | 13.73% |

Schema 23 is not a solution between those endpoints. It is evidence that undifferentiated SAFE
exposure rotates one shared verdict boundary rather than learning the caller-control distinction.

Auxiliary-threshold fitting is not the blocker. On the permitted 400-row calibration split, the
frozen per-head thresholds achieved 78.25% joint exact match. A post-hoc coordinate search directly
maximizing exact match could not improve beyond 78.25%. Sensitive-action AUC was 87.72%, requested
disclosure/transfer AUC 94.20%, and irreversible-action AUC 95.94% on that split. The representation
and weak labels need repair before another thresholding strategy can satisfy the 90% gate.

## Runtime and identities

Training completed in 570.1 seconds on Apple-silicon MPS. The unoptimized FP32 PyTorch artifact is
602,057,462 bytes. Batch-one model-forward latency over 250 samples was 9.55 ms median and 16.03 ms
p95; tokenizer, transfer, model, and probability transform together were 10.28 ms median and
16.97 ms p95. This is a desktop diagnostic only. Quality failed, so no Core ML/ONNX/quantized pack
was built and no physical-device latency claim is possible.

- Model weights SHA-256:
  `20f9287fb5c0fff238d0a64710d6bb1557a94ce75afc9b9d7f02ca2b29febc57`
- Calibration SHA-256:
  `a383f4ac1a609ee7b93fb52ee2bc6b2d883cf4380d005fd19daeeb048f2c3a14`
- Full run report SHA-256:
  `15bd7fd7ad4686edf9bcfc747d8e43565a13025820be2e13fbbfa15ec8e99fef`
- Independent gate report SHA-256:
  `a3639326f3aa23a8edaa63a2a384e6979237b083f19b2d7d65badb1c4e23991a`

No AppTek/YouTube external selection, distillation, Core ML, ONNX, quantization, or sealed-OOD
evaluation was run for this rejected checkpoint.

## Next bounded experiment

Schema 24 should retain the v2 compactor but change supervision and sampling rather than model size:

1. Replace single-turn lexical weak labels with participant-aware labels derived from dialogue
   intent/slot metadata and complete compacted context. Distinguish a service agent requesting an
   account identifier for an in-channel task from a caller directing disclosure, remote access, or
   transfer to a caller-controlled destination.
2. Construct complete-family minimal contrasts around the three weak heads: sensitive action,
   requested disclosure/transfer, and irreversible action. Each family must include legitimate
   in-channel service, official self-navigation, unresolved request, and harmful caller-controlled
   action with the same topic and participants.
3. Sample complete contrast families, not independent rows. Pair each licensed legitimate action
   anchor with a real-scam-call-derived or officially grounded harmful counterpart in the same
   optimizer neighborhood, while keeping verdict retention on the unchanged parent rows.
4. Use the family-disjoint calibration split to validate label consistency before training. Require
   at least 90% leave-family-out exact match from a simple lexical/metadata audit model; if the
   labels themselves cannot clear that check, do not spend another neural training run.
5. Keep the fitting corpus in the tens of thousands. Dataset volume does not set inference latency;
   retain one 256-token encoder pass and postpone any student, Core ML, or quantized work until all
   quality gates pass.
6. Continue seeking admissible positive human scam dialogue. Do not bypass TeleAntiFraud's gated
   access, use Sting9's non-commercial corpus, or scrape Reddit user content for training.

This is a data-semantics experiment, not a request for a larger Qwen payload. The current 149M
encoder is already fast enough on desktop to justify solving quality before compression.
