# Qwen3.5-0.8B Stage 9 and Stage 10 results

## Decision

**Reject Stage 9 and Stage 10 before the full regression. Keep Stage 7 as the current reference.
Do not merge, quantize, package, inspect the sealed PPoNE or publisher-phone tests, or publish either
candidate.**

Stage 9 validated the data-quality hypothesis but did not satisfy the frozen open-promotion
contract. Its development quality and phone scam recall improved, but it lost one additional PPoNE
SCAM and added one alarm on the predeclared 152-row ordinary publisher-SAFE phone stratum. Six of
eight gates passed; both failures are binding.

## Stage 9 receipt

- Base: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Initialization: Stage 7 adapter SHA-256
  `14f1d2bf121e76fa158cea722416994ec9cbfbc545956d24b9693b7808441357`.
- Stage 9 adapter SHA-256:
  `0e853e5b2408805a7eb5743fe3f02230bb600767a7dc9850e6e5ddcfee07baf5`.
- Optimization: one epoch, `2e-6`, seed `20260911`, MPS, 1,239 train rows, 884 families;
  10,822,656 trainable LoRA parameters out of 863,808,576 total.
- Runtime: 966.9 seconds including the trainer development pass; final train loss `0.03894` and
  trainer eval loss `0.03341`.
- Development report SHA-256:
  `d6e5a8e698dee4c20ce0832076d54635b65ab6911998fdea04e28ac1470b3bf3`.
- Open selection report SHA-256:
  `1b0d4fd57ffdf555b30b2064cb696072e8d01dead3714b22d20e499539e7e73c`.
- Promotion-gate report SHA-256:
  `5d902bc148ee40fa0226bd70560684772c479155882b1d06058629b0fde59717`.

| Metric | Stage 7 | Stage 9 | Frozen requirement | Result |
|---|---:|---:|---:|---|
| Dev scam recall | 97.08% | 97.08% | at least 97% | pass |
| Dev SAFE FPR | 0.25% | 0.20% | at most 2% | pass |
| Dev macro F1 | 0.7856 | 0.8301 | no regression | pass |
| Phone scam recall | 78.16% (136/174) | 79.89% (139/174) | no regression | pass |
| Phone ordinary-SAFE FPR | 17.11% (26/152) | 17.76% (27/152) | no regression | **fail** |
| PPoNE scam recall | 91.67% (11/12) | 83.33% (10/12) | no regression | **fail** |
| PPoNE macro F1 | 0.2398 | 0.2984 | at least +0.03 | pass |
| PPoNE UNCERTAIN recall | 4.76% (1/21) | 14.29% (3/21) | at least 3/21 | pass |

Publisher-phone FPR on all 170 SAFE-labeled rows is retained only as a diagnostic because the
tracked source audit found 18 high-risk caller-action conflicts. Stage 9 alarms on 2/18 of those
rows versus Stage 7's 1/18; no publisher label is rewritten.

## Text-free error localization

The exact Stage 7-to-Stage 9 joins emit no message text. The PPoNE audit finds four changed verdicts:
two improved UNCERTAIN decisions, one retained error, and one SCAM regression. The phone audit finds
five changed binary decisions: three improved SCAM calls and two publisher-SAFE regressions. One
SAFE regression is already in the high-risk conflict stratum; the binding ordinary-SAFE regression
uses independent official navigation and asks for no secret.

- PPoNE transition report SHA-256:
  `8c222384962668712ee72b6146084c01cc2c1bfe12e8ad85870fe9a81d7df2a0`.
- Phone transition report SHA-256:
  `a04b26bd5054d69e2c71d762730f4f736f6d7a7113be6f320b701c9e00fa032b`.

The binding PPoNE regression is an identity-investigation robocall with coercive consequences and
an in-call digit action. The evidence contrast—not the learning rate—is the next intervention.

## Stage 10 design receipt

Stage 10 restarts from Stage 7 rather than continuing from rejected Stage 9. It holds the one-epoch
`2e-6` recipe constant and changes only the curriculum. Its new Apache-2.0 source contains 96
original messages in 32 matched families: 32 coercive SCAM calls, 32 unresolved IVRs, and 32 SAFE
independent-navigation calls. Stage 9's prior-open errors informed only the abstract boundary; no
evaluation message is copied, paraphrased, transformed, or used for fitting.

The complete Stage 10 curriculum has 1,404 rows across 704 families: 535 SAFE, 500 SCAM, and 369
UNCERTAIN. It includes all 178 admitted PPoNE training calls, 57 quality-filtered long phone SCAM
calls, 378 matched government replay rows, the complete 356-row action-state source, and
family-diverse Stage 7 retention anchors. The builder quarantines all 78 phone-training audit flags
without relabeling. The dev split remains byte-identical to Stage 7, all six overlap references are
clean, and all 4,038 train-plus-dev sequences fit the 640-token contract (maximum 636).

Frozen identities:

- new contrast rows:
  `9703aa3a83d5de71c637c8981494e8a76e362d6bf766c488214e488b475c3797`;
- contrast manifest:
  `ce984b65357d4b47fb79f8c0f062ea4d6488e242f8351583dced2d6da286b1f2`;
- curriculum manifest:
  `cf8875e285d6c5c525bcd24d276da43b3b7649d56c3470b070b352d24e15df92`;
- token audit:
  `7ae777dc422e4178205d2ae04958d343a36f070a33b0bd6c461be6136ad055c7`;
- experiment config:
  `5f14910bc43b3166bf75722cd4796b543aef09e424f3bb4719c0883b4a759ad0`.

Stage 10 uses the unchanged Stage 9 promotion logic. It must pass every development, PPoNE, and
quality-stratified phone gate before a full regression.

## Stage 10 result

Stage 10 completed the frozen one-epoch MPS run, but failed one of eight open promotion gates. The
matched-evidence curriculum improved development macro F1, phone recall, ordinary-phone SAFE FPR,
PPoNE macro F1, and PPoNE UNCERTAIN recall relative to Stage 7. It nevertheless retained Stage 9's
one-SCAM PPoNE regression: 10/12 detected versus Stage 7's 11/12. That non-regression gate is
binding, so the full regression was not run.

- Stage 10 adapter SHA-256:
  `3eb2974d7db70d2f859a10d1876a9f25e290bbe487abae27f837dfc949b59cf0`.
- Adapter configuration SHA-256:
  `cefdc82c9bde541da713975d1d22607f82792e0b1615126577d316e26cf8b7bb`.
- Training receipt SHA-256:
  `5f28c320c04d0c435a9a447442c0ca81156c43beae205bbb1748242f5c1fd4cc`.
- Runtime: 1,411 seconds including the trainer development pass; final train loss `0.03531` and
  trainer eval loss `0.03374`.
- Development report SHA-256:
  `73cf728ea5d60728bd9f6cf78964f1873d2681d4ca22653290dab7bef17ee1e7`.
- Open selection report SHA-256:
  `da23314cf38e638d86a46560b853178c67619aa955aa1242c8e385e073c8b747`.
- Promotion-gate report SHA-256:
  `9a9d783e4ee8330a66b533337410751f46e67bdc47d33e30a4c8306e6ba09c01`.

| Metric | Stage 7 | Stage 10 | Frozen requirement | Result |
|---|---:|---:|---:|---|
| Dev scam recall | 97.08% | 97.08% | at least 97% | pass |
| Dev SAFE FPR | 0.25% | 0.15% | at most 2% | pass |
| Dev macro F1 | 0.7856 | 0.8479 | no regression | pass |
| Phone scam recall | 78.16% (136/174) | 78.74% (137/174) | no regression | pass |
| Phone ordinary-SAFE FPR | 17.11% (26/152) | 16.45% (25/152) | no regression | pass |
| PPoNE scam recall | 91.67% (11/12) | 83.33% (10/12) | no regression | **fail** |
| PPoNE macro F1 | 0.2398 | 0.2984 | at least +0.03 | pass |
| PPoNE UNCERTAIN recall | 4.76% (1/21) | 14.29% (3/21) | at least 3/21 | pass |

The publisher-phone diagnostic is unchanged at 27/170 SAFE alarms. Stage 10 shifts one error from
the ordinary stratum into the 18-row high-risk caller-action stratum: 25/152 ordinary alarms and
2/18 high-risk alarms, versus Stage 7's 26/152 and 1/18. No publisher label is rewritten.

The text-free transition audit records the same aggregate topology as Stage 9: four changed PPoNE
verdicts (two improved, one retained error, one SCAM regression) and five changed phone decisions
(three improved SCAM calls, two SAFE regressions). This is evidence that adding more matched
government-call wording did not move the remaining PPoNE decision boundary; another lexical replay
iteration would not be justified by the result.

- PPoNE transition report SHA-256:
  `f02df68066ece23b025d157a9a4059734487e17ededf0acf78b20aed18e2da69`.
- Phone transition report SHA-256:
  `b66fad266e98ba6727976deedc66a7e8ac8e3fc5332093734f6fa6501a3c139a`.

PPoNE test, publisher-phone test, primary sealed evaluation, quantization, physical-device
benchmarking, independent human review, and Hugging Face publication remain downstream and
unauthorized.
