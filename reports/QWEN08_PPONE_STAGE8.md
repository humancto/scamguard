# Qwen3.5-0.8B Stage 8 PPoNE-abstention screen

## Decision

**Reject all three Stage 8 continuations before the full regression. Keep Stage 7 as the current
phone-oriented reference. Do not merge, quantize, package, open either sealed source test, or
publish any Stage 8 checkpoint.**

Stage 8 answers a narrow question: can a small, rights-clear curriculum teach the Stage 7 adapter
to abstain on ambiguous real robocalls without losing scam recall or phone behavior? The one-epoch
`2e-6` run proves that the 0.8B model can recover some of the missing abstentions, but the frozen
promotion contract exposes the tradeoff: it loses one of twelve open PPoNE scams and adds a phone
false alarm. The lower learning rates preserve or improve development quality but do not jointly
retain phone and PPoNE behavior. None is eligible for a sealed or release evaluation.

## Real-data boundary

The new source is the WSPR/NCSU PPoNE robocall dataset at revision
`5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272`. Its pinned publisher README places the data records
in the public domain. The 1,432 records are real-world automated or semi-automated calls published
through FTC Project Point of No Entry enforcement materials, but that provenance does not prove
every individual call is fraud. ScamGuard therefore maps only strong-evidence examples to SCAM and
maps the remainder to UNCERTAIN; the source supplies no SAFE truth.

The source admission pipeline keeps 255 representatives after removing 676 exact duplicates,
three near-overlaps with existing references, and 342 same-label near-template repetitions. It
quarantines 102 rows in mixed-label near-template clusters. Family-salted partitioning produces 179
train, 33 open validation, and 43 sealed-test rows. Only the train partition enters Stage 8. The
43-row PPoNE test and the unrelated 172-row publisher phone test remain unopened.

Reddit is deliberately absent. Its current official Data API Terms do not permit ML training on
Reddit Services or Data without the relevant permission. Public visibility is not treated as a
training license. The detailed source decision and provenance links are in
`reports/ONLINE_SOURCE_RESEARCH.md`.

## Curriculum and frozen protocol

- Base: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Initialization: rejected Stage 7 adapter, weights SHA-256
  `14f1d2bf121e76fa158cea722416994ec9cbfbc545956d24b9693b7808441357`.
- Curriculum: 890 one-row families: 311 SAFE, 353 SCAM, and 226 UNCERTAIN.
- Composition: 178 admitted PPoNE train rows, 88 long-refund phone contrasts, 88 weak-category
  anchors, and retention anchors sampled across the prior rights-clear sources.
- Held-data protection: zero exact or near-template overlap with the development, primary test,
  BothBosu, or MultiDoGO validation references; the parent development split is byte-identical.
- Token audit: 3,524 train plus development examples, maximum 639 tokens, and zero examples above
  the 640-token contract. Curriculum manifest SHA-256 is
  `39e1058f1e902e4cd2a27078bd4f920742b5cb84afa2c80e910966d146dac02f`; token-audit SHA-256 is
  `5343ddf663641c973401ec8119c33d0e0f5e731a71da172920ec1b0357c207c5`.
- Optimization control: LoRA rank and targets inherited from Stage 7, batch 4, gradient
  accumulation 4, Apple MPS required, fixed seeds, and no threshold selection on PPoNE or phone
  labels.
- Evaluation control: thresholds are fitted once on the unchanged development split. The open
  PPoNE and phone validation sets are reporting-only promotion screens. The full regression cannot
  run unless the frozen promotion checker passes.

## Results

The Stage 7 columns are the predeclared non-regression reference. `U correct` is the number of the
21 open PPoNE UNCERTAIN rows assigned UNCERTAIN. PPoNE has only 12 SCAM and 21 UNCERTAIN examples,
so every row is reported as a count and the result is not presented as a population estimate.

| Candidate | Epochs / LR | Dev recall | Dev SAFE FPR | Dev macro F1 | PPoNE recall | PPoNE U correct | PPoNE macro F1 | Phone recall | Phone SAFE FPR | Phone macro F1 | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 7 reference | - | 97.08% | 0.25% | 0.7856 | 91.67% (11/12) | 1/21 | 0.2398 | 78.16% (136/174) | 15.88% (27/170) | 0.5414 | reference |
| Stage 8 | 3 / `5e-7` | 97.08% | 0.25% | 0.8096 | 91.67% (11/12) | 1/21 | 0.2460 | 79.31% (138/174) | 16.47% (28/170) | 0.5434 | no |
| Stage 8b | 1 / `2e-6` | 97.08% | 0.20% | 0.8223 | 83.33% (10/12) | 4/21 | 0.3217 | 79.89% (139/174) | 16.47% (28/170) | 0.5454 | no |
| Stage 8c | 1 / `1e-6` | 97.08% | 0.20% | 0.8256 | 83.33% (10/12) | 2/21 | 0.2663 | 75.86% (132/174) | 14.71% (25/170) | 0.5382 | no |

Stage 8 fails phone-FPR non-regression, the required PPoNE macro-F1 gain, and minimum PPoNE
abstention recovery. Stage 8b passes the PPoNE macro and abstention gates but fails phone-FPR and
PPoNE-recall non-regression. Stage 8c produces the best development macro-F1 and phone FPR of the
three, but fails phone recall/macro-F1 and all three PPoNE improvement gates. The three-way result
shows that learning rate is not the missing control; the target examples and/or objective need to
separate uncertain evidence from scam evidence more explicitly.

## Training receipts

| Candidate | Seed | Config SHA-256 | Adapter SHA-256 | Active training runtime | Final train loss |
|---|---:|---|---|---:|---:|
| Stage 8 | 20260908 | `5d15e2823f249dab9be67c5f5159421b0990fec931b1b5401257f78c18dc7442` | `ffb47e6f23171af58b2d8f0c8a7ec1811d1a6069b0d3183548fb77cd05f991bd` | 24,965.3 s wall-clock, inflated by machine sleep | 0.05560 |
| Stage 8b | 20260909 | `48fa1f0d2429f45c4202770fb97e8aa3d50583478765ebc27134a0a4d442088f` | `da8670a6fca4313a8baf5e5b0c2700b04ce619f7d3374b0583b17922730b1ded` | 643.5 s | 0.05615 |
| Stage 8c | 20260910 | `583f6b9b73cee62bcb937ce138dd49fad4d942334cc4f6fa5a70f9d1237a15af` | `533839921203894154c9ce5d8ac13d1686866b36f4cdc1984ba7ff507fb4754c` | 837.7 s | 0.05797 |

The Stage 8 wall-clock receipt includes a long host-sleep interval and must not be interpreted as
active accelerator time. Stage 8b and Stage 8c are uninterrupted local MPS runs. All adapters are
development artifacts and remain ignored from Git.

Promotion-report SHA-256 identities are:

- Stage 8: `8a965d272dfc12e820bfa5dff2601ecf0629b007748967b627de0ab7d97c7900`;
- Stage 8b: `4223b9fa5950f8e89b396344e645384a42981e6f6992fe32a8ec383227fb811c`;
- Stage 8c: `b013e6970f8b8b16bf6b1b199b15eea2b63dbcf5edbeb50ec269b5363d9fe06f`.

## Next experiment boundary

Do not run another learning-rate interpolation or increase the PPoNE dose unchanged. First audit
the exact Stage 7-to-Stage 8b transitions for the one lost PPoNE scam, the three recovered
abstentions, and the added phone SAFE alarm. Build paired, family-separated examples that hold the
surface form constant while changing only the evidentiary fact: verified scam action, ambiguous
automated solicitation, or legitimate service action. Preserve Stage 7 scam and phone-safe anchors
at higher sampling weight, then require the same development and open-promotion gates.

Only a promoted single adapter may enter the full regression. The PPoNE test, publisher phone test,
primary sealed benchmark, quantization, physical-device latency, independent human review, and
Hugging Face publication remain downstream gates.
