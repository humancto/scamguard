# Schema v14 real-dialogue experiment

Decision date: 2026-08-21. **Reject the schema-v14 checkpoint. Retain the source audit and the
family-isolated data partition as research assets.** A small, genuinely different real-data dose
fixed the measured scam-call recall gap, but it also taught an unsafe call/prose shortcut and more
than doubled false positives on the unchanged regression set.

## Source and admission

The source is Kaggle dataset `rivalcults/youtube-scam-phone-call-transcripts`, version 2. The
publisher declares CC0 and describes 243 manually corrected partial transcripts sourced from
YouTube scam calls, mostly scammer/scambaiter interactions plus some autodialer messages. This is
real scam-call-derived language, not 243 ordinary victim calls and not independently reviewed
ground truth.

- Archive: 149,701 bytes; SHA-256
  `3f67497736e9421c2f6e59efc46c129006419d40fc752cbb981042940384cedd`.
- Source: 243 rows and 222 unique upstream source URLs.
- Privacy scan before normalization: zero email-like rows, two phone-like rows, and zero long-digit
  rows. ScamGuard applies its own masking before IDs, clustering, or materialization.
- Windowing: at most one early and one recent whitespace-complete 425-character window per source
  record.
- Admission: 449 candidate windows; one exact duplicate removed; zero exact or near overlaps with
  the existing processed corpus; 448 windows across 220 connected source/template families.
- Partition: deterministic source-and-near-template-family split, yielding 298 train windows from
  145 families, 70 open validation windows from 35 families, and 80 prediction-sealed OOD windows
  from 40 families.
- Schema v14 fits only the 161 early windows from the 145 training families. The recent windows do
  not double-weight the same calls. Development, regression, and all prior diagnostics remain
  byte-identical to schema v13.

The text-free source audit is
[`source-audits/youtube-scam-calls.json`](source-audits/youtube-scam-calls.json). The external
validation artifact SHA-256 is
`108daf448b01e64f1f6f58228b295119b536c44cede1e6b512abc1eaf9262b1e`; the sealed OOD identity is
`69b0d22ee2a9d7b0d44df80fec30e1032bd16d24e5f078f105354c3eb6890cf8`. The sealed rows were not
scored.

## Frozen experiment

Schema v14 contains 14,223 training examples: 8,005 SAFE, 855 UNCERTAIN, and 5,363 SCAM. Its
processed corpus validates at 27,902 unique examples with no family leakage.

| Artifact | SHA-256 |
|---|---|
| Schema-v14 manifest | `83ee137212eb99bac46f81a8ce265e7bad003c58cef35f5f951a9024a6ecc09b` |
| Schema-v14 train | `93fa3d3ddea2c51b8093bc8fa486d26e345c08011da5de27e6d6dc1b42e6da97` |
| Source audit | `a394be257adc881bf9d0de24e10df2e4180dc207fd3792eb1b71869556b4be22` |
| Schema-v13 source-validation report | `d9c21b1b0083b2ff853805cbb601dbdecac8dc78a25e88f6c51db7c973411282` |
| Schema-v14 model report | `09699851c9bb802c37e852ffd14b3c0e4eead8cd90674858349fd74ebbb06e86` |
| Schema-v14 source-validation report | `8b43d87f07d14f64c1db5522ecb9bc5701c1708fa0defd707e31dd9a3313d168` |

The model is the pinned 149M-parameter ModernBERT-base encoder, trained for the predeclared three
epochs with the existing speaker-neutral policy. Its threshold is fit only on development SAFE
and SCAM rows. The YouTube validation slice is used for candidate diagnosis, never fitting or
threshold calibration.

## Result

| Open slice | Schema v13 | Schema v14 | Gate |
|---|---:|---:|---|
| Development recall | 95.91% | 99.61% | at least 97% |
| Development SAFE FPR | 1.94% | 1.74% | at most 2% |
| Development three-way macro F1 | 0.7303 | 0.7627 | above 0.94 stretch |
| Unchanged regression recall | 99.32% (583/587) | 99.83% (586/587) | at least 97% |
| Unchanged regression SAFE FPR | 4.18% (73/1,746) | **8.48% (148/1,746)** | at most 2% |
| Unchanged regression macro F1 | 0.8817 | **0.8576** | above 0.94 stretch |
| BothBosu dialogue recall | 51.06% | 100% | diagnostic |
| BothBosu dialogue SAFE FPR | 18.30% | **73.20%** | diagnostic |
| Taskmaster SAFE FPR | 0% (0/450) | 0% (0/450) | diagnostic |
| New real-call validation recall | 34.29% (24/70) | **100% (70/70)** | positive-only diagnostic |

The real-call increase is large and source-family-held, but it is not a product win. The frozen
threshold moved from 0.27175 to 0.14992, synthetic-v5 SAFE false positives rose from 73 to 145, and
the balanced telephone-dialogue selection slice shows the intended call-language correction is not
class-specific. Positive-only source validation cannot measure precision or false-positive rate.

The checkpoint still meets the desktop latency ceiling: 10.97 ms median and 15.80 ms p95 for
batch-one tokenizer-through-probability inference on Apple MPS, with a 602,034,250-byte training
artifact. This is a desktop measurement, not physical-mobile evidence. The checkpoint is rejected
on quality before Core ML export; speed cannot rescue it.

## Next data decision

Do not add more positive-only call text or increase Taskmaster blindly. The earlier 1,800-row
Taskmaster dose already overfit dialogue provenance. The next controlled dataset must pair scammer
language with natural, scam-adjacent legitimate service-call language—refunds, support, delivery,
insurance, banking, and account verification—while preserving source families and an untouched
English SAFE-call denominator.

Two new sources are promising but are not yet admitted:

- AppTek Call-Center Dialogues is a fresh CC-BY-SA-4.0 collection of 873 English, multi-accent,
  spontaneous role-played service calls. Its publisher explicitly designates it for evaluation and
  analysis rather than training, so it should become a family-split SAFE-call benchmark, not a
  fitting source.
- ES-Port contains anonymized transcripts of real Spanish telecom technical-support calls under
  CC-BY-SA-3.0 Spain. The 3,681,616-byte publisher archive fetched for this audit had SHA-256
  `0017a2d6bbbf57d2971872c7a8eb7c1bf266c76dfb8ba2736c17086a948f1f2c`; it needs a full privacy,
  weak-label, language, and share-alike review before any admission.

Authorized TeleAntiFraud access remains the highest-value balanced call-level candidate. The
existing manual 240-row audit also remains incomplete. No “beats SOTA,” release, or mobile-runtime
claim is justified yet.
