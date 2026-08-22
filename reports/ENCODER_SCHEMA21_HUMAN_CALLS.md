# Encoder schema 21: human-call action-state result

## Decision

**Rejected. Do not export, distill, open sealed OOD, or promote this checkpoint.**

Schema 21 added CC-BY-4.0 HarperValleyBank human-spoken roleplay plus call-grounded action-state
variants to schema 20. The teacher passed development, original state, Taskmaster, and long-call
gates, and its unoptimized PyTorch CPU median fell below 20 ms. It nevertheless failed the
unchanged regression false-positive gate, the original held-out Harper call gate, both frozen
BothBosu gates, and two held-out Harper action-transfer gates. The experiment therefore rejects
the hypothesis that adding one legitimate human banking-call distribution is sufficient.

The complete dataset and recipe were frozen and pushed before training in commits `9c3b893` and
`0a48e13`. The configuration SHA-256 was
`88f2e6cf8b84caa8732dc52ff9d5c9d2a32f0cea7446b4ccc5a1411ea0dadb1`; the processed
manifest SHA-256 was
`1172b5045a943f00f923319807a95ea1c762eafcee439e9efe28c8c8cc000edf`.

## Frozen recipe

- Initialization: `sg-modernbert-schema13-dose16`
  (`31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99`)
- Unique training rows: 26,579
- Licensed non-synthetic rows: 10,396, comprising 8,134 naturally occurring or
  real-call-derived rows, 1,193 Taskmaster human-authored roleplays, and 1,069 Harper human-spoken
  roleplays
- Controlled synthetic rows: 16,183, including 4,276 Harper-grounded final-turn variants
- Harper task-disjoint validation: 377 original SAFE calls and 1,508 four-state rows; branch-hours
  and card-replacement tasks never appeared in training
- Auxiliary targets: sensitive-action language, requested disclosure/transfer, caller-controlled
  target, official self-navigation, independent verification, pressure/secrecy, and irreversible
  action
- Training: one epoch, batch 16, learning rate `5e-6`, retention weight 4, action loss weight 0.5,
  generated action-row verdict weight 0.25, and original Harper verdict weight 1.0
- Input contract: speaker-neutral text, latest 256 tokens (`truncation_side=left`)
- Product alert score: calibrated probability from the preserved first three verdict logits;
  auxiliary heads remained diagnostic signals only
- Threshold: fitted only on unchanged development SAFE/SCAM rows at the 2% FPR cap

The processed training JSONL SHA-256 was
`00f1dcb2fd20b2e6d3390979e46ad76f69422e2056deafd8b75244cf06581e03`.
The Harper source was pinned at Git revision
`0bd721e877c4a85d8c13ff837e68661ea6200a98` under CC-BY-4.0; no audio was downloaded.

## Result

| Gate | Required | Schema 21 | Pass |
|---|---:|---:|:---:|
| Development scam recall | >=97% | 99.61% (512/514) | Yes |
| Development FPR | <=2% | 1.94% (39/2,008) | Yes |
| Unchanged regression scam recall | >=97% | 100.00% (587/587) | Yes |
| Unchanged regression FPR | <=2% | 4.64% (81/1,746) | **No** |
| Original held-state harmful recall | >=97% | 100.00% (512/512) | Yes |
| Original held-state routine SAFE FPR | <=2% | 0.00% (0/512) | Yes |
| Original held-state verified SAFE FPR | <=2% | 0.00% (0/512) | Yes |
| Original held-state unresolved SCAM rate | <=10% | 0.00% (0/512) | Yes |
| Original held-state contrast ordering | >=95% | 100.00% (512/512) | Yes |
| Original held-state action macro AUC | >=97% | 100.00% | Yes |
| Original held-state action exact match | >=90% | 100.00% | Yes |
| Original held-out Harper-call FPR | <=2% | 4.24% (16/377) | **No** |
| Harper held-state harmful recall | >=97% | 100.00% (377/377) | Yes |
| Harper held-state routine SAFE FPR | <=2% | 2.92% (11/377) | **No** |
| Harper held-state verified SAFE FPR | <=2% | 0.00% (0/377) | Yes |
| Harper held-state unresolved SCAM rate | <=10% | 1.59% (6/377) | Yes |
| Harper held-state contrast ordering | >=95% | 98.67% (372/377) | Yes |
| Harper held-state action macro AUC | >=97% | 93.44% | **No** |
| Harper held-state action exact match | >=90% | 84.68% | **No** |
| BothBosu latest-window recall | >=97% | 90.07% (127/141) | **No** |
| BothBosu latest-window FPR | <=2% | 39.22% (60/153) | **No** |
| Taskmaster SAFE FPR | <=2% | 0.22% (1/450) | Yes |
| Long Taskmaster SAFE FPR | <=2% | 0.67% (3/447) | Yes |
| PyTorch desktop end-to-end median | diagnostic | 18.94 ms | Not a release result |
| PyTorch desktop end-to-end p95 | diagnostic | 32.11 ms | Not a release result |

The original generated state set remained perfect. On held-out human-call-derived states, the
mean harmful-versus-routine gap was 0.8832, its p05 was 0.7594, and 372/377 families had the
correct risk ordering. However, the action heads transferred unevenly: sensitive-action AUC was
0.6214 and requested-disclosure AUC was 0.9198. The model can learn the generated action schema,
but it has not learned a sufficiently general action representation from these weak labels.

## Failure analysis

BothBosu false alarms remain concentrated in ordinary commercial calls: 25/42 telemarketing,
23/42 insurance, 11/49 delivery, and 1/20 wrong-number SAFE calls. The 14 missed scams comprise
five support, five reward, two refund, and two Social Security impersonation dialogues.

All 14 misses share a temporal failure: the decisive credential, remote-access, identity, or
payment request appears earlier, while the retained final window ends with the receiver delaying,
refusing, checking independently, or asking the caller to hold. A safe final exchange must not
erase earlier harmful evidence. Conversely, false positives contain sensitive commercial words
without strong caller control. Mean caller-control probability was only 0.060 for false positives
and 0.029 for false negatives, versus 0.185 for true positives; the auxiliary head is directionally
useful but not strong enough to drive a reliable decision.

Harper also reveals a dose problem. Adding every original train call at full verdict weight lowered
the frozen threshold from schema 20's 0.324 to 0.189 and increased unchanged-regression FPR from
1.78% to 4.64%, while still producing 16 false alerts on the entirely held-out Harper tasks. A
larger, single-domain SAFE dose shifted calibration without teaching the cross-domain boundary.

Training took 1,641.19 seconds on CPU. The PyTorch artifact is 602,056,851 bytes. Its weight
SHA-256 is
`4c21c6f779c5736548119ecc6ae73cbe5e36554da59963970f2e197f1e38f8b1`; calibration
SHA-256 is `644e8a8aab237134d87fe09d6bcb0420e865b5cf2caa07cd73235912b32a274d`; run-report
SHA-256 is `49003283a66fd4ad36af66aa8f969e9af080024ff124ffb34decec5047448dbb`.
The error-ledger SHA-256 is
`f4d99ef631bdb268e846de7a7532706094c0d029c7d134874b7a79ecd32beab3`.

No AppTek, YouTube external selection, Core ML, ONNX, or sealed-OOD result was run for this
rejected checkpoint. The PyTorch latency is diagnostic and does not establish physical-mobile
latency.

## Next experiment

Schema 22 should be an evidence-aggregation experiment with a smaller, broader data increment:

1. Add licensed human-authored transactional dialogue from task-disjoint Schema-Guided Dialogue
   domains that cover delivery, insurance-like service, payment, support, and ordinary commerce.
2. Preserve complete dialogues, then create overlapping 256-token windows with dialogue-level
   labels and explicit evidence-window labels. Aggregate the maximum calibrated harmful-evidence
   score across windows so a later refusal cannot erase an earlier scam action.
3. Counterbalance each sensitive-action concept with legitimate caller-controlled and
   user-self-navigated examples across multiple domains. Do not repeat the full Harper dose.
4. Keep BothBosu validation diagnostic-only. Use its ledger to define failure families, not to copy
   benchmark text into training.
5. Freeze dataset hashes, aggregation policy, thresholds, and gates before training. Reject before
   distillation or sealed evaluation if any earlier gate fails.

This keeps the experiment in the tens of thousands of rows. Dataset size still does not determine
latency; the final latency target will be measured only after a teacher passes and is distilled and
quantized into a substantially smaller deployment model.
