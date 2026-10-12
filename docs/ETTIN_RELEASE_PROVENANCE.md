# Ettin release provenance: blocked, not cleared

Checked 2026-10-11. The [machine-readable inventory](../configs/ettin150m-release-provenance.json)
records evidence for the current schema-25 Ettin lineage; it does **not** authorize publication.
The inventory is `blocked`, every source remains `review_required` for final release review,
and no human approval is invented. This is a factual provenance/compliance aid, not legal advice
or a guarantee of rights. A qualified reviewer must assess unresolved rights for the intended use.

## What is established, and what is not

| Item | Verified evidence | Remaining limitation |
| --- | --- | --- |
| Ettin base | Official HF model card at `45d08642849e5c5701b162671ac811b7654bfd9f` declares MIT. The linked official JHU repository supplies an MIT license with the actual JHU copyright notice. | No standalone LICENSE is present in that HF revision. The base declaration does not clear fine-tuning data or third-party privacy rights. |
| Phone synthetic source | Pinned/current HF revision is `27a1f6d0cf21dda995130d221383900b060dbcc9`; README SHA-256 is `38171953e18bba0a66f5aa7f3bf11039ebbb4e033afbde95d38b46dd2a04d7ac`. All six README revisions and the sole public discussion were checked. | The card says only “Llama 8B”; exact model/version, generation date/host terms and downstream-use obligations are unresolved. MIT front matter is not a substitute for that evidence. |
| YouTube transcripts | Live official Kaggle metadata still identifies version 2 and publisher-declared CC0. The 149,701-byte archive matches `3f67497736e9421c2f6e59efc46c129006419d40fc752cbb981042940384cedd`. | No underlying creator/participant permission receipt was located. The publisher's PII-removal claim is not independent clearance; the pinned source audit found two phone-like rows before masking. |
| Corpus/ancestry | Inventory matches 14 sources, 24,436 train rows, 2,634 dev rows and 29,648 v2 presentations. Exact corpus, parent/current training reports and source evidence are hash-bound. | Integrity and license metadata are not proof of all rights or label quality. Current generator code is evidence, not a reconstructed historical execution receipt. |
| Human review | Existing ZIP is unchanged; all 635 decisions are blank and all three return files are absent. | No completed independent label/privacy review. The sample is not a copyright clearance audit or model-accuracy test. |
| HF authentication | Normal `.venv/bin/hf auth whoami` returned `Error: Not logged in`. | Signing in later is an operational step, not publication approval. No credential was read, copied or requested. |

Primary sources: [pinned Ettin card](https://huggingface.co/jhu-clsp/ettin-encoder-150m/blob/45d08642849e5c5701b162671ac811b7654bfd9f/README.md),
[official JHU license](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/6cbf954a2ecaa7c753be6468a1bccdef401d5f93/LICENSE),
[pinned phone card](https://huggingface.co/datasets/shakeleoatmeal/phone-scam-detection-synthetic/blob/27a1f6d0cf21dda995130d221383900b060dbcc9/README.md),
[Kaggle publisher page](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts),
and [official Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/rivalcults/youtube-scam-phone-call-transcripts).
For the last item, the official API supplied readable metadata where the rendered page did not.

CC0 applies to the affirmer's rights and disclaims responsibility for clearing others' rights;
it does not establish that an uploader can waive every creator's or participant's rights.
See [CC0 section 4](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en#s4).
The phone-generator uncertainty is material: model families can have different output-use
conditions. We do not infer a version or legal outcome from the generic name.

## Base notice and data notices

[LICENSES/Ettin-MIT.txt](../LICENSES/Ettin-MIT.txt) preserves the official license wording and
`Copyright (c) 2025 Johns Hopkins University Center for Language and Speech Processing`.
The source is pinned to JHU repository commit `6cbf954a2ecaa7c753be6468a1bccdef401d5f93`.

- Upstream license SHA-256: `7e096e455b9dac457008fa960d25e64066416f417a71c7ad61fbdd65158ba9b9`.
- Local license SHA-256: `b866194b77a237d30106629df409eeafea233c019373e0b902332b4c3d517cd5`.
- Only terminal whitespace differs: local bytes equal `upstream.rstrip() + '\n'`.
  No legal wording or copyright notice was changed.
- Pinned HF model-card SHA-256: `95b8dbb4c4ba78b2855bdfc6ebefe8eac4c39d63c20f92e2aced3f47cd7e1ff6`.

The [data notice](ETTIN_DATA_NOTICE.md) identifies all training sources and their declared terms.
ScamGuard's Apache-2.0 code license must not be substituted for the base's MIT notice or external
data terms. A future model license and attribution package need artifact-specific review;
dataset licensing obligations must not be asserted to apply identically to raw data and weights.

## Ancestry cannot be erased by changing a replay

Both v1 and v2 were trained with the phone and YouTube sources. V2 begins from the v1 weights.
Removing these sources only from a later continuation does not undo their ancestral training
influence or resolve their rights. Keep both sources in the ancestry inventory unless a separate,
appropriately frozen fresh-base experiment excludes them. That alternative needs authorization,
new evidence and reassessed review coverage; this document does not authorize it.

The unchanged 635-row packet remains the one review for the current corpus; see
[audit reuse](ETTIN_V2_AUDIT_REUSE.md). New or altered data is not silently covered by that packet.

## Proposed fail-closed machine contract

The JSON is an evidence inventory, not a completed release validator. A consumer must:

1. Verify schema, data/base/checkpoint/parent/report/evidence hashes; match all source IDs,
   unique row counts, selected-ID digests and ancestral source coverage. Reject missing/stale bindings.
2. Keep a local staged package visibly blocked while any source is `review_required`, restricted,
   unknown, or lacks a genuine human review decision and its scope/date/evidence receipt.
   A license tag, free-form assertion, or a manually flipped boolean is not clearance.
3. Require the exact three human returns, packet-bound import verification, and manual checks of
   actual reviewer independence and per-row language/assignment coverage. Do not fabricate reviewers.
4. Verify packaged base/data notices and an explicit artifact allowlist. Exclude raw training,
   blind-audit and sealed text, credentials, unlisted source URLs, and training/optimizer pickles.
5. Evaluate quality, final-validation, runtime parity, performance and package safety separately.
   Passing provenance cannot override a failed model gate. Validation must never authenticate,
   contact a publisher or upload as a side effect.

## Minimal owner decisions and actions

1. Appoint an independent review coordinator with qualified human language help. Send the existing
   ZIP and forwardable note in [the handoff](INDEPENDENT_REVIEW_HANDOFF.md), not this owner-side
   document or canonical answers. Collect `scamguard_blind_audit.csv`, `reviewer-attestation.txt`
   and `reviewer-assignments.tsv`. Source metadata identifies 400 English, 27 non-English and
   208 unspecified rows; the unspecified rows are not assumed English. Estimate 8–14 base
   person-hours plus language assistance/adjudication. Do not guess unreadable messages UNCERTAIN.
2. Provide authoritative phone-generator details and YouTube underlying-rights evidence for
   qualified review, or choose a separately authorized clean-data fresh-base route. No publisher
   was contacted and no permission is presumed. Human label review cannot resolve these rights
   questions by itself.
3. Only when the model and release evidence genuinely qualify, select the HF account/organization,
   repository and visibility, and authenticate locally using the normal HF flow. Never place a
   token in the CSV, JSON, notice or chat. Authentication alone must not trigger publication.

The current quality and release state remains in the
[v2 experiment report](../reports/ETTIN150M_RECOVERY_V2.md). These new notices do not make v2
production-ready, SOTA, physically mobile-verified, or authorized for HF upload.
