# AppTek legitimate-call benchmark

Freeze date: 2026-08-21.

## Decision

The [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues)
text metadata is admitted as an **evaluation-only SAFE-call diagnostic**. It is not training data,
not real customer data, and not independently reviewed row-level ground truth. The publisher
describes 873 newly collected spontaneous English service-call roleplays across 14 accents and 16
domains, licenses the release CC-BY-SA-4.0, and explicitly places training out of scope.

ScamGuard downloads no audio. The fetcher pins publisher revision
`95a8c157e4fd6df2f3c77593160c83db79b75dc7` and verifies all 14 diarization metadata files by byte
count and SHA-256 before use. The pinned text metadata totals 17,438,486 bytes.

## Construction

The builder validates the exact call and segment schemas, requires 873 unique call names, checks
segment timing and the `agent`/`customer` role vocabulary, and hashes source call and speaker IDs.
It emits an early complete-turn window and a recent complete-turn window, each capped at 425
characters. Role prefixes survive truncation. Privacy-like email, phone, and long-digit patterns
are normalized before a row can be emitted.

The 1,746 candidate windows contained no exact duplicates and no near overlap with the existing
ScamGuard development corpora. Two same-source near-template rows were collapsed, leaving 1,744
representatives. Shared-call and shared-speaker edges form 77 connected components. A fixed salt
selects the component subset closest to 20% while keeping complete components together:

| Partition | Components | Calls | Windows | Policy |
|---|---:|---:|---:|---|
| open selection | 18 | 174 | 348 | candidate selection only |
| sealed OOD | 59 | 699 | 1,396 | no predictions until candidate freeze |

The split contains no shared hashed speaker identity. Its tracked manifest contains hashes,
counts, domains, and accents but no raw dialogue text or speaker hashes. The selection artifact is
SHA-256 `e0e7ad4de8d378061159df18a8fa6c39fedd69d0cdd0dffe3f72579158290b62`; the unopened OOD
artifact is `1e6d1176936324f073ca7dd5746bcce5b7849d5be5c4973782479e595f3c3ade`.

## Frozen-threshold baseline result

Both models were scored with their existing temperature and SCAM threshold. AppTek supplied no
fitting row and did not refit calibration or thresholds.

| Candidate | SAFE false positives | SAFE FPR (95% Wilson CI) | Early window | Recent window | Decision |
|---|---:|---:|---:|---:|---|
| schema 13 dose-16 | 31 / 348 | 8.91% (6.35–12.37%) | 31 / 174 | 0 / 174 | reject |
| schema 14 natural dialogue | 77 / 348 | 22.13% (18.08–26.78%) | 77 / 174 | 0 / 174 | reject |
| schema 15 legitimate openings | 54 / 348 | 15.52% (12.09–19.70%) | 53 / 174 | 1 / 174 | reject |
| schema 16 continual retention | 105 / 348 | 30.17% (25.59–35.19%) | 105 / 174 | 0 / 174 | reject |

Schema 14's positive-only real scam-call increment made legitimate openings substantially worse.
All 108 false alarms across the two candidates occur in early windows, while both produce zero
false alarms on all 174 recent windows. This localizes a model shortcut: introductory service-call
language is being treated as scam evidence. It does not show that accents cause errors; accent
slices are small and confounded with speaker, domain, and call script.

The largest schema-13 domain rates were finance 2/6, banking 6/24, and delivery service 6/32. For
schema 14 they were finance 3/6, insurance 11/22, and delivery service 14/32. These are selection
diagnostics with small denominators, not population estimates.

## Corrective experiment

Schema 15 is a predeclared 256-row training correction: 16 original service scenarios × four call
opening structures × four variants. It uses deterministic synthetic SAFE dialogue only. It copies
zero AppTek rows and deliberately omits artificial safety phrases such as “never ask,” “official
app,” and “gift card.” The AppTek aggregate and metadata slices informed the gap definition; this
means the open slice is now candidate-selection evidence, never an independent release test.

The schema-15 run must retain the unchanged development/regression recall and FPR gates, improve
AppTek selection FPR, retain the positive YouTube-call recall, and avoid regression on BothBosu and
Taskmaster. The 1,396-window AppTek OOD partition stays sealed until the model, threshold, routing,
and export choices are frozen.

The completed run fails. AppTek FPR improves from schema 14's 22.13% to 15.52%, but remains worse
than schema 13's 8.91%. Unchanged-regression FPR rises to 18.84%, and BothBosu reaches only 69.50%
recall / 35.29% SAFE FPR. YouTube-call recall remains 100% and Taskmaster remains 0% FPR, but those
narrow successes cannot compensate for the safety regressions. Schema 15 and its increment are
rejected; see [`DATASET_SCHEMA15_LEGITIMATE_OPENINGS.md`](DATASET_SCHEMA15_LEGITIMATE_OPENINGS.md).

Schema 16 changes the learning formulation without adding rows: it initializes from schema 13,
retains schema-13 logits on all inherited rows, and applies square-root source balancing to the
fixed schema-15 corpus. It improves unchanged-regression FPR to 3.15% and reaches 98.57% YouTube
call recall, but worsens AppTek to 30.17% FPR. All 105 false alarms occur in early openings. This
rejects source balancing plus generic retention as a sufficient correction and motivates
structure-matched scam/legitimate minimal contrasts. See
[`ENCODER_SCHEMA16_RETENTION.md`](ENCODER_SCHEMA16_RETENTION.md).

## Reproduction

```bash
make apptek-callcenter
make apptek-eval-schema13
make apptek-eval-schema14
make schema15-legitimate-openings
make encoder-schema15-legitimate-openings
make apptek-eval-schema15
make encoder-schema16-retention
make apptek-eval-schema16
make youtube-eval-schema16
```

Generated/raw/model artifacts are ignored by Git. The repository publishes acquisition,
verification, construction, and evaluation code plus text-free audit reports.
