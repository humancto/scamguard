# Ettin HF readiness: hardened staging, not a production release

2026-10-11. **The model still fails two quality gates. No HF upload, new fit,
new model inference, calibration change, or sealed evaluation was performed in this pass.**
The latest request authorized repair and release preparation; the old experiment's
rejection decision and measured results have not been rewritten.

## Completed and verified

- Added opt-in `speaker-neutral-action-preserving-v3` preprocessing. It reserves
  evidence for an access request or a concrete requested secret/account number/payment,
  ahead of topic-rich questions or later identity reassurance. It retains negation,
  speaker identity and available safety context; it never assigns a verdict.
- Reproduced two observed losses in already-open dialogue diagnostics. V2 dropped
  both requested actions; V3 retains both after the actual 256-token tokenizer limit.
  The served inputs are 181 and 170 tokens. This is **retention evidence, not
  evidence that either model prediction is now correct**.
- Compared frozen V2 and current V2 on all 44,259 pinned train/dev/open rows:
  **44,259 identical outputs**. V3 changes 2,051 inputs, remains opt-in, and needs
  its own calibration, quality and latency evaluation before adoption. No existing
  checkpoint manifest selects V3. See the [text-only regression receipt](../reports/data/ettin-action-preserving-v3-regression.json).
- Prepared a 96-row blind adjudication packet covering all 43 reference-SAFE alerts,
  four reference-SCAM messages assigned SAFE, and 49 concordant controls across the
  two failing dialogue sources. Labels, predictions and selection reasons are hidden
  from reviewers. See the [text-free preparation receipt](../reports/data/ettin-dialogue-v3-adjudication.json).
- Added encoder-specific local staging, strict file/hash/source bindings, tensor-container
  checks, required notices, and a visibly blocked model card. The five runtime files
  remain byte-identical to the V2 checkpoint; training pickles, corpus text, prediction
  ledgers and credentials are excluded. The tool cannot authenticate or upload.
- Checked the complete 14-source training ancestry and preserved the actual Ettin MIT
  notice separately from the SDK's Apache-2.0 license. [Provenance findings](ETTIN_RELEASE_PROVENANCE.md)
  are an evidence inventory, **not rights clearance**.

Verification: **745 tests passed**, targeted Ruff checks passed, and the full pinned
text-transform regression completed. Tests do not substitute for model benchmarks.

## Why another fit did not run

The remaining phone/BothBosu SAFE FPR results are still **15/170 (8.82%)** and
**28/153 (18.30%)**, against the unchanged 2% limits. Some reference-SAFE messages
request a returned verification code, SSN or caller-supplied verification path.
Ordinary sales offers also conflict with the product rubric's UNCERTAIN class.
These are candidates for adjudication, not automatic relabeling; genuine model
false alarms also exist. Training to suppress all these warnings before review could
make the detector less safe. No error has been removed from the historical denominator.

Two source-provenance issues also remain: the phone generator is described only as
“Llama 8B,” and the YouTube transcript publisher's CC0 declaration does not establish
underlying creator/participant rights or independent privacy clearance. Both sources
already influenced V1 and V2. Excluding them from another continuation does not erase
that ancestry. If authoritative evidence cannot resolve them, the recommended route
is a **fresh-base, clean-data experiment**, not another V2 continuation.

## Exactly what the human handoff needs

Use one independent coordinator; do not send this owner-side document, canonical
answers, selection manifests or predictions to the reviewers.

1. **Existing training/privacy audit — 635 rows, still 0 completed.** Send
   `data/audit/ettin150m-v1/scamguard-ettin150m-v1-blind-audit.zip` with the forwardable
   note in [the original handoff](INDEPENDENT_REVIEW_HANDOFF.md). Collect its completed
   `scamguard_blind_audit.csv`, `reviewer-attestation.txt` and `reviewer-assignments.tsv`.
   Language help is required; unspecified language is not assumed English. This is
   the original audit, not a second 635-row request.
2. **New benchmark-label adjudication — 96 rows, still 0 completed.** Send only
   `data/audit/ettin-dialogue-v3/dialogue-adjudication-blind.zip` and
   `data/audit/ettin-dialogue-v3/reviewer-requirements.json`. Return the completed
   CSV plus the actual reviewer's identity, qualifications and independence attestation.
   Review every row under the frozen rubric. Disagreements are findings, not failures
   to be erased. Do **not** apply the old zero-disagreement release importer to this
   packet: its role is adjudicating potentially incorrect benchmark labels. A later
   adjudicated dataset must preserve the original references, reasons and version history.
3. **Rights decision.** Have a qualified reviewer assess the inventory and authoritative
   phone-generator/YouTube permission evidence, or select the fresh-base exclusion route.
   Label review and model performance cannot provide rights clearance.

The 96 rows are deliberately error-enriched and are not an unbiased accuracy estimate
or an independent final test. Human review is not something the training agent can
attest on its own behalf. The existing audit's estimate remains 8–14 base person-hours
plus language coordination; the new adjudication work is additional, not included in it.

## Local staging and verification

The latest local artifact is `dist/sg-ettin150m-recovery-v2-blocked-r2/`.
It is a **blocked research staging directory**, not a public HF repository or a mobile
export. Its [receipt](../reports/data/ettin-release-staging.json) records hashes and blockers.

```bash
.venv/bin/python scripts/stage_ettin_release.py verify \
  --package dist/sg-ettin150m-recovery-v2-blocked-r2
```

This confirms package integrity only. Adding `--require-release` intentionally fails.
The package binds the original SDK revision `bd96b81e05a15b69f64bf036ba922f9ce155013a`;
the new preprocessing source is reported as drift, despite the V2 output-compatibility
check. Optional model smoke refuses a mismatched SDK. No new package inference was
run in this pass. Do not bypass this check or treat raw Transformers argmax as the
calibrated ScamGuard verdict.

To reproduce staging, use `configs/ettin150m-recovery-v2-staging.json` with the `stage`
subcommand and a **new** output directory; existing evidence cannot be overwritten.
Do not upload either local staging attempt or the private review packets.

## Exit path

After human findings and the ancestry decision are available: version the adjudicated
labels without changing historical metrics; freeze one bounded experiment on approved
data and the chosen input policy; calibrate on development only; evaluate all declared
open slices without retries or hidden omissions. A passing candidate then needs a
candidate-specific, frozen unopened final-validation contract and complete serving
evaluation. Open selection data cannot be renamed an independent final test.

The current staging tool is intentionally not that final-release verifier. HF packaging
alone does not solve quality, independent evaluation or rights. Desktop-first release is
consistent with the user's desktop-and/or-mobile scope; physical mobile evidence is
needed only for actual mobile claims, not as an invented requirement for desktop HF use.

Once the evidence qualifies, choose the HF namespace/repository/visibility and sign in
locally with the normal HF flow. Authentication currently reports **Not logged in**.
Never paste tokens into a review packet or chat. Publication must use the reviewed
artifact and model card, with no SOTA or mobile-performance claim beyond the evidence.
