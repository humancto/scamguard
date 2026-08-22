# Encoder schema 20: action-state multi-task result

## Decision

**Rejected. Do not export, distill, open sealed OOD, or promote this checkpoint.**

Schema 20 tested whether seven explicit action/context targets could separate harmful requests
from routine, independently verified, and unresolved call states while retaining the schema-13
verdict boundary. It learned the controlled task perfectly and passed the unchanged development,
regression, Taskmaster, and long-SAFE gates. It nevertheless failed both predeclared BothBosu
dialogue gates. Synthetic state separation is therefore not sufficient evidence of real-dialogue
transfer.

The complete dataset and recipe were frozen and pushed before training in commits `4b9d65d` and
`0a48e13`. The configuration SHA-256 was
`c2ca3df6335bf81962aed54281aab904c3074d468cb7cca676f40a0c2d8d9886`; the processed
manifest SHA-256 was
`1bba8a07fc0efc236d34d6edacc9bdb2a338765ab88b6633246a176c515f8e4e`.

## Frozen recipe

- Initialization: `sg-modernbert-schema13-dose16`
  (`31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99`)
- Unique training rows: 21,234
- Licensed-source or original non-synthetic rows: 9,327, of which 8,134 are naturally occurring
  or real-call-derived and 1,193 are human-authored Taskmaster roleplay
- Controlled synthetic rows: 11,907
- New state supervision: 6,144 train rows in 1,536 four-state contrast families and 2,048
  validation rows in 512 families from four entirely held-out scenarios
- State labels: `routine_safe`, `verified_safe`, `unresolved`, and `harmful_scam`
- Auxiliary targets: sensitive-action language, requested disclosure/transfer, caller-controlled
  target, official self-navigation, independent verification, pressure/secrecy, and irreversible
  action
- Training: one epoch, batch 16, learning rate `5e-6`, retention weight 4, action loss weight 0.5,
  and action-row verdict weight 0.25
- Input contract: speaker-neutral text, latest 256 tokens (`truncation_side=left`)
- Product alert score: calibrated probability from the preserved first three verdict logits;
  auxiliary heads were training and diagnostic signals only
- Threshold: fitted only on unchanged development SAFE/SCAM rows at the 2% FPR cap

The processed training JSONL SHA-256 was
`6fbecab488b743c7d8ffca4326f9a8a9e70497df3ce9147b08616627b4b75dcb`; held-out
state validation SHA-256 was
`c6d9aa782739990f7dcd0fd5956d90274bfffcf89e868da394d0a458087918fe`.

## Result

| Gate | Required | Schema 20 | Pass |
|---|---:|---:|:---:|
| Development scam recall | >=97% | 99.61% (512/514) | Yes |
| Development FPR | <=2% | 1.64% (33/2,008) | Yes |
| Regression scam recall | >=97% | 99.83% (586/587) | Yes |
| Regression FPR | <=2% | 1.78% (31/1,746) | Yes |
| Held-state harmful recall | >=97% | 100.00% (512/512) | Yes |
| Held-state routine SAFE FPR | <=2% | 0.00% (0/512) | Yes |
| Held-state verified SAFE FPR | <=2% | 0.00% (0/512) | Yes |
| Held-state unresolved SCAM rate | <=10% | 0.00% (0/512) | Yes |
| Held-state contrast ordering | >=95% | 100.00% (512/512) | Yes |
| Held-state action macro F1 | >=97% | 100.00% | Yes |
| Held-state action exact match | >=90% | 100.00% | Yes |
| BothBosu latest-window recall | >=97% | 93.62% (132/141) | **No** |
| BothBosu latest-window FPR | <=2% | 42.48% (65/153) | **No** |
| Taskmaster SAFE FPR | <=2% | 0.00% (0/450) | Yes |
| Long Taskmaster SAFE FPR | <=2% | 0.22% (1/447) | Yes |
| PyTorch desktop end-to-end median | diagnostic | 20.09 ms | Not a release result |
| PyTorch desktop end-to-end p95 | diagnostic | 32.43 ms | Not a release result |

The held-out state set had a mean harmful-versus-routine gap of 0.8881, a p05 gap of 0.8175,
and a minimum gap of 0.7434. All seven auxiliary targets had 100% ROC AUC and F1 on these
scenario-held-out generated families. This proves that the frozen representation can express the
desired distinction, but the perfect result must be interpreted as controlled-generator evidence,
not as a real-world accuracy claim.

BothBosu provides the decisive counterevidence. False alarms concentrate in ordinary commercial
calls: 24/42 telemarketing, 23/42 insurance, 16/49 delivery, and 2/20 wrong-number SAFE calls.
The nine missed scams comprise four support, three refund, and two reward dialogues. Many false
alarms contain a legitimate purchase, transfer, license, or contact-verification discussion; many
misses end with the receiver delaying or refusing even though a harmful request appeared earlier.
The model still entangles sensitive commercial content with malicious caller control and remains
too dependent on the most recent turn.

The auxiliary heads did not solve that transfer gap. On BothBosu, mean caller-control probability
was 0.405 for true positives but only 0.201 for false positives; requested-disclosure probability
was 0.205 versus 0.145. The direction is useful, but the separation is far below the controlled
holdout. Real or human-authored dialogue needs dense action supervision, and the deployed decision
needs evidence aggregation across windows rather than a single verdict logit.

Training took 1,158.78 seconds on CPU. The PyTorch artifact is 602,056,838 bytes. Its weight
SHA-256 is
`586edb11d1deb511565108f5630fb9581ddd1f25f1f18d759b9a877b809c46ee`; calibration
SHA-256 is `a17d97ec7e7796c948f363c3dc9b58d2fc41cf556974dec295e1ae5d994d12b5`; run-report
SHA-256 is `695d00f6f902301884243f13b7c3c1f4f24cb7832d703eac76e8703346ed99de`.

No AppTek, YouTube external selection, Core ML, ONNX, or sealed-OOD result was run for this
rejected checkpoint. The full teacher's latency is diagnostic only: it narrowly misses 20 ms at
the median and is not a mobile artifact.

## Next data increment

The next increment should add human-authored and human-spoken legitimate call distributions that
contain genuine sensitive actions, not just more generic SAFE dialogue:

1. Admit the CC-BY-4.0 HarperValleyBank transcript corpus as human-human roleplay: 1,446 simulated
   banking calls, 25,381 human-corrected utterances, and tasks covering transfers, bill payment,
   password reset, balance checks, card replacement, checks, appointments, and branch hours.
2. Add dense action labels to the original SAFE calls so sensitive-action and irreversible-action
   heads learn that those concepts can coexist with a legitimate verdict.
3. Create conversation-linked harmful variants by minimally changing only the decisive agent turn;
   keep original and transformed calls in the same family and cap transformed exposure.
4. Add a bounded CC-BY-SA-4.0 Schema-Guided Dialogue slice for human-authored banks, payment,
   delivery, insurance-like service, and transactional domains. Preserve service/dialogue groups
   and count these separately from naturally occurring communication.
5. Train temporal evidence aggregation across overlapping windows. A later refusal must not erase
   an earlier credential, remote-access, or irreversible-transfer request.
6. Keep TeleAntiFraud-28k as a candidate rather than an admitted source until its gated archive is
   accessible and its real-derived versus generated rows can be distinguished from provenance.

This keeps the training set in the tens of thousands. Dataset scale does not control inference
latency; encoder depth, sequence length, runtime, and quantization do. The quality target should be
met before distilling to a 10M-30M parameter student and measuring the final mobile path.
