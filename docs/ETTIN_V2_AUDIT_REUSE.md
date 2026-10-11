# Ettin v2: reuse the existing independent-review packet

Owner-side binding note, verified 2026-10-11. **One 635-row human review is still required;
no second 635-row review is needed solely because v2 repeats existing training rows.**
This is reuse of a corpus-label review, not evidence that either model passed evaluation.
All 635 review decisions remain blank; no independent human review is complete.

## Why the same packet applies

`sg-ettin150m-recovery-v2` uses the same immutable schema-25 train/dev files as the existing
`sg-ettin150m-bounded-v1` review packet. Its frozen recipe changes presentation weights,
not source text, labels, or the set of training examples. It adds no rows, short views,
translations, or evaluation examples to training; development rows are not replayed into training.

The recipe presents each of the 24,436 original training rows once, then adds two presentations
for each of 1,206 identity-impersonation SCAM rows and each of 1,400 SAFE rows from the human
MultiDoGO service-dialogue and synthetic phone-scam sources. That is 29,648 presentations,
shuffled with the frozen seed. These are recipe counts, not a claim that fitting completed.

The 635-row sample is an enriched, stratified audit of underlying train/dev records. It is not
a new sample weighted by v2's replay frequency, and cannot estimate presentation-weighted
error rates, population prevalence, or model accuracy. New or changed text, labels, sources,
or translations would require explicit versioned provenance and reassessed review coverage;
they are not silently covered by this reuse note.

## Frozen bindings

Paths are relative to the repository root. SHA-256 values were computed from the current files,
and train/dev hashes were checked against both the v2 configuration and the original audit manifest.

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| V2 configuration | `configs/ettin150m-recovery-v2.json` | `5c6a241a6994d0a64ea7d92c8052b9b4d1a76e917b553683ba1207babd7986b5` |
| Training corpus, 24,436 rows | `data/experiments/schema25-full-call-curriculum/processed/train.jsonl` | `d4e3e100e97a937a789aa984d4ce75b545592886099a0027dbd758bbf0b2e614` |
| Development corpus, 2,634 rows | `data/experiments/schema25-full-call-curriculum/processed/dev.jsonl` | `04201ee86325306cf1751dbf848ddca360386de707069c0c4bba0c9c28a21c07` |
| Corpus manifest | `data/experiments/schema25-full-call-curriculum/processed/manifest.json` | `6c1096f1017edb10b690fde8968d9de892d2bff8ae49b9183c5da15e0380ad57` |
| Blank canonical audit CSV | `data/audit/ettin150m-v1/label-audit.csv` | `51c8b448826a6bada655763cf746e9537936e9ae53961ada20bd4c81f5930d93` |
| Canonical audit manifest | `data/audit/ettin150m-v1/label-audit.manifest.json` | `db6aad6806107e306ce5709be59c32fa6380644bbba98cfd18e395e28b98aca7` |
| Existing reviewer ZIP, 113,760 bytes | `data/audit/ettin150m-v1/scamguard-ettin150m-v1-blind-audit.zip` | `4f11ee5e63536785ee6a6f56c1bc7345e3de7d26315380853b11ce2334356ea5` |

The manifest's selected-ID digest is
`cdf04399b6040525cd16f7d39edf9a7854b58e4268165973ac7ef80ecb4dbb72`;
its selected-input digest is
`173ba2ce2cd0ed110e2536356f5781f1f0fb85cc3d390eb9fc6e2da1499e9fc6`.
The ZIP's embedded canonical-manifest digest matches the manifest above.
The original manifest intentionally retains `sg-ettin150m-bounded-v1`; neither the packet
nor its manifest is renamed, regenerated, or rewritten to imply a new review.

## The owner's remaining review action

Use the [existing handoff and forwardable note](INDEPENDENT_REVIEW_HANDOFF.md#owner-action).
Send only that existing ZIP and the forwardable note, not this owner-side document, the repository,
or canonical audit files. Arrange an independent primary reviewer/coordinator plus qualified
human language assistance as described in the
[language-staffing instructions](INDEPENDENT_REVIEW_HANDOFF.md#language-staffing--owner-only).
Unreadable messages must not receive guessed labels or automatic AI label decisions.

Collect these three human-produced returns under `data/audit/ettin150m-v1/returned/`:

- `scamguard_blind_audit.csv`: all 635 message decisions, retaining the packet's IDs and text.
- `reviewer-attestation.txt`: truthful identities, roles, independence, languages, and dates.
- `reviewer-assignments.tsv`: per-row reviewer/language/helper mapping, separate from the CSV.

None of those return files existed at verification time. Follow the exact
[return checks and version-bound import/verification commands](INDEPENDENT_REVIEW_HANDOFF.md#return-and-import--owner-only).
Preserve disagreements and sensitivity findings; never change answers to force the gate to pass.

## Verification boundary

CPU-only checks recomputed all file hashes and row counts, validated canonical audit binding
with `validate_audit_binding`, and verified the ZIP with `load_and_verify_bundle`. The audit has
635 rows with blank `auditor_label`, `label_correct`, `contains_sensitive_data`, and `notes` fields.
These checks establish packet integrity and corpus reuse, not reviewer independence, label quality,
or a passed model/release gate. No model was loaded and no inference was run for this note.

Research fitting can proceed while the review is pending. This note does not authorize an HF
upload, a SOTA claim, or deployment: human review and the candidate's separate quality,
runtime-parity, performance, provenance, and release checks remain necessary.
