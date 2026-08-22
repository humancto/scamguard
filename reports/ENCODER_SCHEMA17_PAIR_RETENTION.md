# Encoder schema 17: paired-call retention result

## Decision

**Rejected. Do not export and do not evaluate sealed OOD splits.**

Schema 17 tested whether a small, structure-matched minimal-pair curriculum plus a paired
ranking loss could teach the schema-13 ModernBERT checkpoint to distinguish ordinary call
language from a harmful final action. It improved held-out paired-call scam recall, but it did
not learn a usable absolute boundary and it regressed established safety gates.

The experiment was predeclared and pushed before training in commit `54be7ca`. The frozen
configuration SHA-256 was
`4a16bd5c33d8a7142739e027af96e68e4d8c677cc879e75aed153d88faf41e3b`.

## Frozen recipe

- Initialization: `sg-modernbert-schema13-dose16`
  (`31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99`)
- Training rows: 14,799, each visited once
- Teacher anchors: 14,062 text-free schema-13 logits
- New unanchored rows: 161 licensed real scam-call rows and 576 original paired synthetic rows
- Minimal pairs: 288 train families; 96 validation families from four entirely held-out scenarios
- Training: one epoch, batch 16, learning rate `5e-6`, retention weight 2, pair weight 0.5,
  pair margin 2, no source replacement sampling
- Pair sampler: both members of every pair were kept in the same even-sized batch
- Threshold: fitted only on the unchanged development SAFE/SCAM rows at the 2% FPR cap

Preflight verified all input hashes, pair composition and shared-context hashes, teacher ID
coverage, held-out scenario separation, and sealed-OOD flags. The complete run took 779.73
seconds on Apple MPS and passed the pair-batch runtime guards.

## Result

| Gate | Required | Schema 13 baseline | Schema 17 | Pass |
|---|---:|---:|---:|:---:|
| Development scam recall | >=97% | 95.91% | 95.14% | No |
| Development FPR | <=2% | 1.94% | 1.99% | Yes |
| Regression scam recall | >=97% | 99.32% | 98.98% | Yes |
| Regression FPR | <=2% | 4.18% | 3.32% | No |
| BothBosu selection recall | >=97% | 51.06% | 98.58% | Yes |
| BothBosu selection FPR | <=2% | 18.30% | 55.56% | No |
| Taskmaster selection FPR | <=2% | 0.22% | 0.00% | Yes |
| Paired validation recall | >=97% | 61.46% | 70.83% | No |
| Paired validation FPR | <=2% | 12.50% | 21.88% | No |
| Paired validation ordering | 100% | 83.33% | 83.33% | No |
| PyTorch desktop end-to-end p95 | <=20 ms | 55.65 ms | 32.09 ms | No |

The schema-17 model is 602,034,360 bytes. Its weight SHA-256 is
`8907a9970dc8846b1804d366a288efabdf5c985a8654048858024a3e9b687aaa`; the calibration
SHA-256 is `ab558ef082e6052657df9841fee1389105563d76f3031524ecc8e4027fdb50c8`; and the full
run-report SHA-256 is `704fe270cfe62bbd738db07bb2511381c2ff4db2b0efa88449df70e3d308917c`.

No AppTek or YouTube external selection score was run for this rejected checkpoint. No Core ML
or ONNX export was attempted.

## What the paired objective learned

The failure is not evenly distributed. On all 384 generated pair families, five of six action
mechanisms reached 100% relative ordering. `secrecy_isolation` reached 0%, making the aggregate
ordering exactly 5/6, or 83.33%.

| Mechanism | Scam recall | Safe FPR | Pair ordering |
|---|---:|---:|---:|
| Credential/code request | 68.75% | 7.81% | 100% |
| Cryptocurrency fee | 100% | 17.19% | 100% |
| Login link | 98.44% | 32.81% | 100% |
| Protection transfer | 98.44% | 26.56% | 100% |
| Remote access | 85.94% | 21.88% | 100% |
| Secrecy/isolation | 0% | 15.63% | 0% |

The fit remained weak even on the 288 seen pair families: 76.74% recall, 19.79% FPR, and
83.33% ordering. The 96 held-out families scored 70.83% recall, 21.88% FPR, and 83.33%
ordering. This is underfitting of the paired curriculum, not merely scenario generalization.

The stronger BothBosu recall came from shifting the whole call distribution toward SCAM rather
than learning the harmful-action boundary. That is why its FPR rose to 55.56%.

## Additional input-window finding

The frozen BothBosu score used the historical right-truncation policy. Its dialogues have a
median length of 319 tokens, and 73.13% exceed the 256-token limit, so that policy discards the
newest turns. A post-rejection, open-selection diagnostic with left truncation reduced FPR to
23.53% but also reduced recall to 70.21%; it still fails decisively. This exploratory result must
not replace the frozen metric. A future streaming contract should score recent sliding windows
explicitly rather than treat either one-sided truncation as the product behavior.

## Next experiment requirements

Do not continue from the rejected schema-17 weights. A next candidate must initialize from
schema 13 and change the curriculum, not just the threshold:

1. Add substantially more diverse paired families that keep suspicious conversational framing
   constant while varying only the external action.
2. Expand secrecy/isolation language and benign privacy language as explicit counterfactuals.
3. Increase paired-objective exposure enough to fit seen pairs while strengthening retention on
   the original boundary.
4. Predeclare a recent-window/sliding-window inference policy and evaluate it consistently.
5. Keep the same sealed-data rule: reject before export or sealed OOD evaluation if any open gate
   fails.
