# Encoder schema 22: service-evidence result

## Decision

**Rejected. Do not run external selection, distill, export, open sealed OOD, or promote this
checkpoint.**

Schema 22 returned to the schema-20 parent and added a bounded dose of CDLA-Permissive-1.0
MultiDoGO human-authored service roleplay plus four-state action evidence grounded in held-out
insurance and software conversations. It fixed schema 21's unchanged-regression false-positive
regression and nearly eliminated BothBosu false alarms. It did not solve the joint safety problem:
BothBosu recall collapsed to 41.13%, held-domain harmful-state recall was 93.92%, and 6.08% of
held routine states crossed the scam threshold. The checkpoint passed 20 of 29 frozen gates and
is rejected.

The complete dataset, configuration, preflight, and gates were frozen and pushed before training
in commit `41b45d3`. The configuration SHA-256 was
`36aadf8df9878f2b841e2342d30cf81705f4bf68340dde0fa4c3fd693203dcbd`; the processed manifest
SHA-256 was `814414895d7cb808a5a28fa31675c23e068d3e7f0bf642cf41daef805088d2ec`.

## Frozen recipe

- Initialization: `sg-modernbert-schema13-dose16`
  (`31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99`)
- Unique training rows: 24,208
- Licensed non-synthetic rows: 11,117, comprising 8,134 naturally occurring or
  real-scam-call-derived rows, 1,193 Taskmaster roleplays, and 1,790 MultiDoGO roleplays
- Controlled synthetic rows: 13,091, including 1,184 MultiDoGO-grounded state rows
- MultiDoGO validation: 896 original SAFE views from 448 unseen families and 592 four-state rows
  from 148 insurance/software families excluded from state training
- Auxiliary targets: sensitive-action language, requested disclosure/transfer, caller-controlled
  target, official self-navigation, independent verification, pressure/secrecy, and irreversible
  action
- Training: one epoch, 1,513 optimizer steps, batch 16, learning rate `5e-6`, retention weight
  4, action loss weight 0.5, generated action-row verdict weight 0.25, and original MultiDoGO
  verdict weight 0.5
- Input contract: speaker-neutral text and the latest 256 tokens (`truncation_side=left`)
- Threshold: fitted only on the unchanged development SAFE/SCAM rows at the 2% FPR cap

The processed training JSONL SHA-256 was
`1c076d0f6d98d39178fdc503345d4a0c85dc5ed4a8d410789cb84f86d2afbfcc`.
MultiDoGO was pinned at Git revision `baa30639c4b271f394b81443c842193407cdf26d` under
CDLA-Permissive-1.0; no audio was downloaded.

## Result

| Gate | Required | Schema 22 | Pass |
|---|---:|---:|:---:|
| Development scam recall | >=97% | 99.81% (513/514) | Yes |
| Development SAFE FPR | <=2% | 1.84% (37/2,008) | Yes |
| Unchanged regression scam recall | >=97% | 97.96% (575/587) | Yes |
| Unchanged regression SAFE FPR | <=2% | 0.63% (11/1,746) | Yes |
| Original held-state harmful recall | >=97% | 100.00% (512/512) | Yes |
| Original held-state routine SAFE FPR | <=2% | 0.00% (0/512) | Yes |
| Original held-state verified SAFE FPR | <=2% | 0.00% (0/512) | Yes |
| Original held-state unresolved scam rate | <=10% | 0.00% (0/512) | Yes |
| Original held-state contrast ordering | >=95% | 85.35% (437/512) | **No** |
| Original held-state action macro AUC | >=97% | 100.00% | Yes |
| Original held-state action exact match | >=90% | 100.00% | Yes |
| Held-domain MultiDoGO harmful recall | >=97% | 93.92% (139/148) | **No** |
| Held-domain MultiDoGO routine SAFE FPR | <=2% | 6.08% (9/148) | **No** |
| Held-domain MultiDoGO verified SAFE FPR | <=2% | 0.00% (0/148) | Yes |
| Held-domain MultiDoGO unresolved scam rate | <=10% | 0.00% (0/148) | Yes |
| Held-domain MultiDoGO contrast ordering | >=95% | 100.00% (148/148) | Yes |
| Held-domain MultiDoGO action macro AUC | >=97% | 98.09% | Yes |
| Held-domain MultiDoGO action exact match | >=90% | 75.84% (449/592) | **No** |
| MultiDoGO original-call SAFE FPR | <=2% | 2.90% (26/896) | **No** |
| MultiDoGO airline SAFE FPR | <=3% | 0.00% (0/150) | Yes |
| MultiDoGO fast-food SAFE FPR | <=3% | 6.00% (9/150) | **No** |
| MultiDoGO finance SAFE FPR | <=3% | 2.67% (4/150) | Yes |
| MultiDoGO insurance SAFE FPR | <=3% | 3.33% (5/150) | **No** |
| MultiDoGO media SAFE FPR | <=3% | 3.33% (5/150) | **No** |
| MultiDoGO software SAFE FPR | <=3% | 2.05% (3/146) | Yes |
| Long-call SAFE FPR | <=2% | 0.00% (0/447) | Yes |
| Taskmaster SAFE FPR | <=2% | 0.00% (0/450) | Yes |
| Prior-open BothBosu recall | >=97% | 41.13% (58/141) | **No** |
| Prior-open BothBosu SAFE FPR | <=2% | 1.96% (3/153) | Yes |

The calibrated threshold was 0.218690. BothBosu precision was 95.08%, but 83/141 scams were
missed. Recall was weak across every reported category: credential theft 34.29%, financial
35.14%, identity impersonation 48.57%, and opportunity 47.06%. Schema 22 therefore learned a
much more conservative boundary, not a generally superior dialogue detector.

On the held MultiDoGO states, all nine harmful misses were software-domain credential-theft
families: insurance harmful recall was 75/75, while software was 64/73. The nine routine false
alerts split into six insurance and three software families. The harmful-versus-safe probability
gap remained positive for every family, but its p05 narrowed to 0.1855.

The held action heads ranked examples reasonably well (98.09% macro AUC) but did not produce
reliable joint decisions at the frozen 0.5 threshold (75.84% exact match). Caller-controlled-target
recall was only 56.08%; sensitive-action AUC was 93.17%, and requested-disclosure AUC was 96.64%.
This is partly a calibration problem, but the two weaker AUCs show that per-head threshold tuning
alone cannot establish the required action representation.

The original state set still made perfect thresholded verdicts, but 75 families failed the stricter
ordering because the unresolved score did not exceed the verified-safe score. This is a genuine
fine-grained ranking failure, not a reporting defect: harmful-minus-safe gaps were positive in all
512 families, while the ordering gate additionally requires
`harmful > unresolved > verified_safe`.

## Runtime and identities

The trainer reported about 1,340 seconds on Apple-silicon CPU. The unoptimized FP32 PyTorch
artifact is 602,056,856 bytes. Batch-one end-to-end latency over 250 samples was 19.28 ms median
and 32.31 ms p95. This is diagnostic only: quality failed, so no Core ML or quantized runtime was
built, and no physical-phone claim is possible.

- Model weights SHA-256:
  `3e3e2fb3d7fa84a61eb6e59307bee7e513682bffa7b247082b33c86e234d4c40`
- Calibration SHA-256:
  `9df0363dc786acd533fcabd82c1f3895a1a7ccef395755cdd511fd2f99ff1b65`
- Full run report SHA-256:
  `ba9cdc88cbc36f11eebf0f43fbc3c5d5f02677e7a5a086e568c0982e1e948148`
- Independent gate report SHA-256:
  `0d5fa3ced5a7a7cd6d72419d8aeb3b8417fbc2d3c2aede5f418b9ea8cb38d67c`

No AppTek or YouTube external selection, distillation, Core ML, ONNX, or sealed-OOD evaluation was
run for this rejected checkpoint.

## Next experiment

Schema 23 should change evidence handling, not merely enlarge the SAFE corpus:

1. Freeze a single-pass evidence-compaction policy that retains action-bearing turns plus recent
   conversational context inside 256 tokens. Use source-derived annotations and rules, never
   BothBosu text, to define the selector.
2. Add span/turn-level supervision for the decisive requested action, destination control, payment
   rail, credential, remote-access, pressure, and verified-channel evidence. Pool the strongest
   harmful evidence so a later refusal or polite close cannot erase an earlier request.
3. Counterbalance each action with legitimate agent-assisted and customer-self-navigated examples
   across domains. Do not add another broad undifferentiated SAFE dose.
4. Calibrate auxiliary heads on a dedicated family-disjoint calibration split, then freeze their
   thresholds before validation. Keep the verdict threshold independently fitted on development.
5. Treat BothBosu only as prior-open regression evidence. Continue searching for admissible,
   licensed positive human scam dialogue; do not train on or tune to benchmark text.
6. Preserve the current tens-of-thousands scale. Distill and quantize only after every quality gate
   passes; then require Core ML p95 <=20 ms and a physical-device measurement.

This targets the observed frontier directly: schema 20 was sensitive but noisy, while schema 22 is
specific but dangerously insensitive. A successful successor must dominate both at the same frozen
operating point rather than choosing one side of that tradeoff.
