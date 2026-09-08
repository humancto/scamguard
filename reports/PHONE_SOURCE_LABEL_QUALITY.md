# Publisher phone-source label-quality audit

## Decision

The pinned `shakeleoatmeal/phone-scam-detection-synthetic` corpus remains useful as synthetic
language coverage, but its publisher labels are not authoritative product-safety truth. Before the
next training run, quarantine flagged training rows instead of teaching the model that a risky
caller action is SAFE. Keep the publisher validation split as a prior-open, source-specific
diagnostic and report both the original metric and a predeclared action-risk stratification.

This is deterministic triage, not independent human relabeling. It does not authorize silently
changing labels, opening the 172-row test, or promoting a model. A blinded human audit remains
required before the source supports a public quality claim.

## Scope and controls

The audit reads the immutable, privacy-normalized 1,190-row training and 344-row open validation
derivatives. It refuses any row marked `test`, emits no message text, binds every input and
prediction ledger by SHA-256, and never changes a label. It parses caller and receiver turns so a
receiver asking about their own account is not mistaken for a caller request.

High-risk SAFE conflicts are deliberately narrow. A row is flagged when a publisher-SAFE caller
requests a returned code, payment, or remote access, or when the caller combines a sensitive-value
request with a caller-supplied link or portal. Separate flags identify generation instructions,
speaker-parse failures, and publisher-SCAM rows with neither a supported caller action nor an
existing deterministic scam signal.

## Findings

| Split | SAFE | SCAM | SAFE high-risk conflicts | SCAM weak-evidence flags | Generation artifacts | Parse failures |
|---|---:|---:|---:|---:|---:|---:|
| Train | 588 | 602 | 52 | 23 | 2 | 1 |
| Open validation | 170 | 174 | 18 | 4 | 2 | 0 |

The SAFE conflict rate is 8.84% in training and 10.59% in open validation. These are review or
quarantine candidates, not automatically proven scams. The source also exposes a known shortcut:
its `variation_style` metadata perfectly predicts the publisher label. ScamGuard never supplies
that metadata to a model.

## Stage 8b disagreement localization

The frozen Stage 7 and Stage 8b ledgers were joined to all 344 open validation IDs. Stage 8b added
one publisher-SAFE threshold alarm relative to Stage 7. That sole additional alarm belongs to the
18-row high-risk-action stratum: the caller supplies a link/portal and asks for a code to be
returned. On the other 152 publisher-SAFE rows, both adapters produce exactly 26 alarms.

| Model | All publisher-SAFE alarms | High-risk stratum (18) | Other SAFE stratum (152) |
|---|---:|---:|---:|
| Stage 7 | 27/170 | 1/18 | 26/152 |
| Stage 8b | 28/170 | 2/18 | 26/152 |

Therefore the Stage 8b phone-FPR failure is real disagreement with the frozen publisher label, but
it is not evidence that the model became less safe on ordinary legitimate calls. The Stage 8 gate
is not rewritten after seeing this result; Stage 8b remains rejected because it also loses one of
twelve open PPoNE scams. The finding changes the next predeclared benchmark design, not the past
decision.

## Transition evidence

The text-free Stage 7-to-Stage 8b comparison found five changed PPoNE decisions: three corrected
UNCERTAIN verdicts, one still-wrong UNCERTAIN transition, and one SCAM regression. The lost SCAM is
a government/identity investigation call with coercive consequences and an IVR action. This is the
positive retention boundary Stage 9 must protect. On phone validation, three long, very-subtle
support/refund SCAM calls become detected and the single disputed SAFE call becomes an alarm.

Immutable evidence:

- label-quality audit SHA-256:
  `b9216bb3da50f723f3ed0e017bbad33eb959645db16075252a880b91fb73e0fd`;
- Stage 7 phone prediction ledger SHA-256:
  `4b6f1b79a63ad4a6e625b8e157d072c0687a1b4c0c8afda881eac19809595821`;
- Stage 8b selection ledger SHA-256:
  `4e6c705d7df235ca6b33f6b861dc23376c890ed51917ea2d151a8833787156e1`;
- PPoNE transition audit SHA-256:
  `b0e17536bb238444ceda2b5887069429a9bae324aff441588139b533c561a102`;
- phone transition audit SHA-256:
  `bbb3421129d80e2700e9fbab7c6f6ea80bd3588ff1c30c2f659e99a895a492bb`.

The ignored transition artifacts can be reproduced with
`make qwen-08b-ppone-stage8-transition-audit`. The tracked, text-free label audit can be reproduced
with `make phone-scam-label-audit`.

## Stage 9 boundary

Stage 9 must be frozen before training with these controls:

1. exclude, rather than silently relabel, flagged phone-training rows;
2. retain explicit government/identity coercion SCAM anchors;
3. use paired evidence states that distinguish a generic promotional IVR, a coercive credential
   demand, and legitimate self-navigation through an official channel;
4. preserve the unchanged dev recall/FPR contract and Stage 7 PPoNE SCAM recall;
5. predeclare phone SCAM recall plus FPR on the 152-row unflagged SAFE stratum, while retaining the
   original publisher metric as a diagnostic; and
6. keep PPoNE and phone tests sealed until all open promotion gates pass.
