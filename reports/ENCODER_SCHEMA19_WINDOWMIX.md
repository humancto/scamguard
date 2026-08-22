# Encoder schema 19: licensed call-window result

## Decision

**Rejected. Do not run additional external selection benchmarks, export this checkpoint, or
evaluate sealed OOD splits.**

Schema 19 tested the frozen hypothesis that schema 18 failed mainly because its constructed
action pairs were short and overrepresented. It added licensed long SAFE and SCAM-derived call
windows, changed every synthetic pair into a length-matched recent-call window, reduced pair
exposure from two passes to one, and retained the latest 256 tokens. The result preserved the
targeted action distinction and passed both normal-call window gates, but it did not transfer a
usable absolute boundary to BothBosu or the unchanged regression suite.

The complete dataset and recipe were frozen and pushed before training in commit `03bc1e1`.
The configuration SHA-256 was
`2d880a113b656ab9af3b8ab88bc96bca1695e46554bf415d3b5249a7c776ad4a`; the processed
manifest SHA-256 was
`a87c1cfdf6e13c7e070499c9fbe8f2fe8d29f3ee0ccf4535c1ffbbd987b3147a`.

## Frozen recipe

- Initialization: `sg-modernbert-schema13-dose16`
  (`31b4d17cf752e6d79319506c62f0cc8e406ddb70d317fae0421672f7854f2d99`)
- Unique training rows: 18,162
- Teacher anchors: 14,062 text-free schema-13 logits
- New licensed and original rows: 1,193 Taskmaster SAFE windows, 435 YouTube
  scam-call-derived windows, and 3,072 original long counterfactual pair rows
- Taskmaster families: 600 conversation-disjoint train conversations plus 447 held-out long
  SAFE-call conversations
- YouTube families: 145 connected train families; publisher validation and OOD families were
  not expanded
- Synthetic pairs: 1,536 train families plus 512 scenario-held-out validation families; every
  raw pair row exceeded 256 tokens and therefore exercised latest-context truncation
- Training: one epoch, batch 16, learning rate `5e-6`, retention weight 4, pair weight 1,
  pair margin 3, one pair exposure, no source replacement sampling
- Input contract: speaker-neutral text, latest 256 tokens (`truncation_side=left`)
- Threshold: fitted only on the unchanged development SAFE/SCAM rows at the 2% FPR cap

The processed training JSONL SHA-256 was
`e120edbd0ceec7684b0b67bd3034b3bcc4796d9d53a99b928323e4340601a383`; paired validation
SHA-256 was `b3423195f1c1adb71b54a48b49f3b0fb54ed8115b33ea650120365cb18a8b3e4`; and long
SAFE-call validation SHA-256 was
`c96a642d8d3a98f510eb14929032260055f6c71dfda7153f4e5699900743171c`.

## Result

| Gate | Required | Schema 19 | Pass |
|---|---:|---:|:---:|
| Development scam recall | >=97% | 99.42% | Yes |
| Development FPR | <=2% | 1.29% | Yes |
| Regression scam recall | >=97% | 99.66% | Yes |
| Regression FPR | <=2% | 3.15% | **No** |
| BothBosu latest-window recall | >=97% | 92.91% | **No** |
| BothBosu latest-window FPR | <=2% | 42.48% | **No** |
| Taskmaster short-window FPR | <=2% | 0.22% | Yes |
| Taskmaster long-call FPR | <=2% | 0.22% | Yes |
| Long paired validation recall | >=97% | 100.00% | Yes |
| Long paired validation FPR | <=2% | 0.59% | Yes |
| Long paired validation ordering | 100% | 100.00% | Yes |
| PyTorch desktop end-to-end p95 | diagnostic | 21.51 ms | Not a release result |

The 512-family long paired holdout had a mean SCAM-probability gap of 0.7672, p05 of 0.5646,
and minimum of 0.3678. The long normal-call validation produced one false positive in 447
conversation-disjoint Taskmaster calls. These results confirm that the latest-window length
contract and the explicit action counterfactuals are learnable.

The absolute transfer boundary nevertheless failed decisively. On BothBosu, 65 of 153 SAFE
calls were flagged and 10 of 141 scams were missed. SAFE false-positive rates were broad rather
than isolated to a length bucket: delivery 44.90%, insurance 45.24%, telemarketing 50.00%, and
wrong-number calls 15.00%. SCAM recall was 97.30% for refund, 94.12% for reward, 97.14% for SSN,
and only 82.86% for support calls.

The calibrated threshold fell to 0.4131. BothBosu SAFE false positives contained more genuine
risk-like language than its true negatives: credential requests appeared in 10/65 false
positives versus 2/88 true negatives, and contact-diversion language appeared in 21/65 versus
15/88. This is not merely a category or length shortcut. The binary label asks one score to
represent both the presence of an action and whether the surrounding context makes that action
coercive, irreversible, or independently verifiable.

Training took 791.79 seconds on Apple MPS. The PyTorch artifact is 602,034,361 bytes. Its weight
SHA-256 is
`9580999e8544b6f2fb56e9c6244db656c7a4adae5fb600e2dcdc7c340193168b`; calibration
SHA-256 is `81f045493cda37fe49b6e9fd66e25f84cc9234e6d03f0f5ad0222422d5982a1f`;
and run-report SHA-256 is
`6439b2d4bae3289b876e74ec05b3c69fe3beaada81e61c512100be8c18cdf393`.

No AppTek, YouTube external selection, Core ML, ONNX, or sealed-OOD result was run for this
rejected checkpoint. PyTorch latency is diagnostic only and does not establish mobile latency.

## What the experiment established

The schema-18 diagnosis was incomplete. Length-matched call windows removed the obvious length
shortcut, preserved perfect pair ordering, and made held-out ordinary service calls extremely
safe, yet real-dialogue transfer became worse. Adding more generic normal calls or more copies
of the same binary counterfactual task is therefore not a justified next step.

The data now reveal two different prediction problems:

1. Detect evidence that a caller proposes a high-risk action such as disclosing a credential,
   installing remote access, transferring funds, or moving to an untrusted channel.
2. Decide whether that action is unsafe in context, including independent verification,
   caller control, urgency, secrecy, payment irreversibility, and official-channel recovery.

Schema 19 teaches the first distinction well but forces the second into the same scalar label.
BothBosu contains legitimate calls with credential and contact language, exposing that
identifiability failure. Its failure despite passing Taskmaster also shows that
human-authored roleplay from one service-call distribution is not a substitute for diverse,
naturally styled legitimate-call negatives.

## Next experiment requirements

Do not continue from schema-19 weights. The next frozen experiment should initialize from
schema 13 and change the target representation before increasing dataset volume:

1. Add explicit multi-task labels for proposed action, coercion/urgency, channel trust,
   irreversible value transfer, credential disclosure, and independent-verification language.
2. Generate original hard-negative counterfactuals in delivery, insurance, support, and
   telemarketing settings where the same credential or contact terms occur but the user is
   directed to independently open an official app, call a verified number, or disclose nothing.
   Share all surrounding turns between SAFE and SCAM versions so surface terms cannot solve the
   task.
3. Train separate evidence and context heads. The product alert should require a high-risk
   proposed action plus unsafe context, while allowing independently verified recovery paths to
   suppress a premature alert.
4. Evaluate streaming aggregation across overlapping recent windows. Conversation-level alert
   logic must be predeclared and cannot be tuned on BothBosu.
5. Preserve all schema-19 gates, including the long SAFE-call and long paired holdouts, while
   keeping AppTek, YouTube external selection, and sealed OOD closed until the internal suite
   passes.
6. Optimize or distill only after the larger teacher passes quality. The current 149M PyTorch
   checkpoint is already close to the desktop 20 ms target, but its 602 MB size and failed
   quality gates make it neither a mobile candidate nor a release result.

The right dataset is therefore not a much larger unlabeled scrape. It is a source-diverse,
conversation-disjoint corpus with dense action-state supervision and hard legitimate
look-alikes. Row count remains secondary to independent source families, label precision, and
unopened evaluation partitions.
