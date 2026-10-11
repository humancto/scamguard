# Independent human review: Ettin150M bounded v1

Prepared and verified on 2026-10-10 for `sg-ettin150m-bounded-v1`. **Send this one new
635-message packet, not the historical schema-24 packet.** It is bound to the experiment's
schema-25 train/dev corpus. All 635 decisions are blank; no independent human review is complete.

## Owner action

Choose one independent human reviewer, send only the ZIP below and the forwardable note, and
agree a return date. Do not send the repository, this document's owner-only sections, model
outputs, source metadata, or canonical label workbook.

- [Blind-review ZIP](../data/audit/ettin150m-v1/scamguard-ettin150m-v1-blind-audit.zip), 113,760 bytes.
- Absolute path: `/Users/archithrapaka/Documents/Codex/2026-08-20/work/outputs/scamguard/data/audit/ettin150m-v1/scamguard-ettin150m-v1-blind-audit.zip`.
- SHA-256: `4f11ee5e63536785ee6a6f56c1bc7345e3de7d26315380853b11ce2334356ea5`.
- Four files: `README.txt`, `review.py`, `scamguard_blind_audit.csv`,
  `scamguard_blind_audit.manifest.json`. No answer key is included.

The reviewer must not have authored/corrected these project labels or seen their answer key or
model predictions. Choose someone fluent in the displayed language and comfortable assessing
ordinary financial/service messages and scam evidence. They must make their own decisions from
the frozen rubric. AI-generated decisions do not count as independent human review.

Plan **8–14 person-hours**, commonly over two days, plus any later adjudication. This is an
estimate, not a deadline guarantee: the sample contains about 58,000 words (median 65 per row;
51 rows exceed 250 words). Reading at 200 words/minute alone takes 4.8 hours; decisions, difficult
dialogues, setup and breaks add time. Refine the estimate after the first 25 rows. The existing
importer accepts one complete review; use another independent person for disputed cases as needed.

## Forward this note with the ZIP

> Please independently review all 635 messages in the attached ScamGuard blind-review ZIP for
> experiment `sg-ettin150m-bounded-v1`. You need Python 3.11 or newer and a modern browser.
> No Python packages or internet access are required by the review application.
>
> Extract the ZIP into a separate folder. Open a terminal there and run
> `python3 review.py --open`. The local page displays the frozen review rubric and saves progress
> to `scamguard_blind_audit.csv`. Stop with Ctrl-C and run the same command later to resume.
> If port 8765 is occupied, use `python3 review.py --port 0 --open`.
>
> For every message, choose SAFE, UNCERTAIN or SCAM from the displayed text and rubric. Check
> whether real-looking sensitive information is present. Use notes for ambiguity, language
> difficulties or the reason for a difficult decision. Generic spam is not automatically a scam;
> insufficient fraud evidence belongs in UNCERTAIN. Do not consult project labels, source
> metadata or model predictions, and do not use an AI assistant to supply review decisions.
>
> Preserve IDs, text, columns, row count and the manifest. Only `auditor_label`,
> `contains_sensitive_data` and `notes` are review fields; the UI preserves their format.
>
> Once finished, stop the server and run `python3 review.py --check`. Check all four values:
> `valid: true`, `rows: 635`, `complete_rows: 635`, `remaining_rows: 0`. A valid file can still
> be incomplete.
>
> Return the completed `scamguard_blind_audit.csv`. In a separate plain-text note, provide:
> reviewer name or agreed identifier; review dates; languages assessed; any assistance used;
> and a truthful confirmation that you made these decisions personally, did not create/correct
> the original project labels and did not consult the answer key or model predictions.
> Mention any exception to that confirmation. Do not add this declaration as CSV columns.
>
> Please report your actual judgments, including disagreements and sensitive-data findings.
> Agreement with an unseen project answer is not your objective.

## Return and import — owner only

The CSV retains exactly five columns, in order:

```text
id,text,auditor_label,contains_sensitive_data,notes
```

Each row requires `SAFE`, `UNCERTAIN` or `SCAM`, and a sensitivity decision of `yes` or `no`.
Notes are optional, maximum 2,000 characters. Preserve the separate identity/independence note
unchanged as `data/audit/ettin150m-v1/returned/reviewer-attestation.txt`; do not fill it for the
reviewer. The importer does not identify people or verify independence, so successful import
alone is insufficient to assert an independent human review.

Save the original returned CSV unchanged at
`data/audit/ettin150m-v1/returned/scamguard_blind_audit.csv`. From the repository root, run these
commands with explicit new bindings. **Do not use the schema-24 Make targets.**

```bash
rtk .venv/bin/python scripts/import_blind_audit.py \
  --returned-audit data/audit/ettin150m-v1/returned/scamguard_blind_audit.csv \
  --bundle data/audit/ettin150m-v1/scamguard-ettin150m-v1-blind-audit.zip \
  --canonical-audit data/audit/ettin150m-v1/label-audit.csv \
  --canonical-manifest data/audit/ettin150m-v1/label-audit.manifest.json \
  --output data/audit/ettin150m-v1/label-audit.reviewed.csv \
  --report reports/data/ettin150m-v1-review-completion.json

rtk .venv/bin/python scripts/verify_imported_blind_audit.py \
  --returned-audit data/audit/ettin150m-v1/returned/scamguard_blind_audit.csv \
  --bundle data/audit/ettin150m-v1/scamguard-ettin150m-v1-blind-audit.zip \
  --canonical-audit data/audit/ettin150m-v1/label-audit.csv \
  --canonical-manifest data/audit/ettin150m-v1/label-audit.manifest.json \
  --reviewed-audit data/audit/ettin150m-v1/label-audit.reviewed.csv \
  --report reports/data/ettin150m-v1-review-completion.json
```

The importer checks the bundle, rubric, dataset binding, immutable IDs/text, complete decisions
and one-to-one join before computing agreement. The verifier reconstructs that import in a
temporary directory and compares outputs. Never import an AI/internal review as a human return.

The current gate requires all 635 rows complete, zero disagreements, zero sensitive-data findings
and no binding errors. An honest disagreement is useful. The importer preserves it and exits
unsuccessfully **after it may have written the reviewed CSV/report**. Do not overwrite those
outputs, coerce the reviewer's decision, or change the frozen answer key in place. Preserve the
return, independently adjudicate disputed items, and version any justified correction/quarantine.
Additional review is needed only if adjudication or a changed dataset justifies it, not simply
because this candidate replaced the old one. There is no automatic adjudication workflow.

## Frozen scope and exclusions — owner only

The frame is the 24,436 training and 2,634 development rows in
`data/experiments/schema25-full-call-curriculum/processed`. No evaluation/held-out label files or
model predictions were read by this preparation. The sample has 558 train and 77 dev rows,
212 SAFE, 212 SCAM and 211 UNCERTAIN, covering all 14 sources and 58 source/domain/label/split
strata. Seed: `sg-ettin150m-bounded-v1-review-v1`.

Selection ranks rows by SHA-256 of `seed:id`, takes six per stratum (or all when fewer exist),
anchors at least 24 phone examples per available label, then fills each label target round-robin
over its sorted strata. The new-source coverage is:

| Source | Rows | Coverage |
| --- | ---: | --- |
| Synthetic phone-scam conversations | 55 | 25 SAFE, 30 SCAM |
| Human MultiDoGO service conversations | 42 | Six domains, SAFE |
| MultiDoGO-grounded evidence states | 141 | Four domains; 28 SAFE, 49 SCAM, 64 UNCERTAIN |

This is an intentionally enriched supervision-quality sample, **not** a prevalence-weighted
accuracy estimate or proof that every corpus row is correct/private. It does not audit held-out
benchmarks, other evaluation datasets, extraction/explanation quality, category/severity, runtime
speed, SOTA superiority or device support. Those require separate evidence. Keep dev rows out of
model-fitting data; do not recycle evaluation labels as training feedback.

The sample is bound to the experiment ID and exact dataset bytes, not to a particular checkpoint
or a successful result. Training may proceed as unaudited research while review is pending.
Neither preparing the ZIP nor training a checkpoint satisfies the human-review release gate.
Existing release configurations that reference schema-24 evidence must be explicitly updated for
this candidate before publication; this handoff does not silently change those configurations.

| Artifact | SHA-256 |
| --- | --- |
| Schema-25 manifest | `6c1096f1017edb10b690fde8968d9de892d2bff8ae49b9183c5da15e0380ad57` |
| Train file | `d4e3e100e97a937a789aa984d4ce75b545592886099a0027dbd758bbf0b2e614` |
| Dev file | `04201ee86325306cf1751dbf848ddca360386de707069c0c4bba0c9c28a21c07` |
| Canonical review manifest | `db6aad6806107e306ce5709be59c32fa6380644bbba98cfd18e395e28b98aca7` |
| Selected IDs | `cdf04399b6040525cd16f7d39edf9a7854b58e4268165973ac7ef80ecb4dbb72` |

## Verification and reproducibility

- [`prepare_ettin_review.py`](../scripts/prepare_ettin_review.py) pins all three dataset hashes,
  reads only train/dev, refuses existing destinations and reuses the existing binding/bundle
  helpers. The [preparation report](../reports/data/ettin150m-v1-review-preparation.json) records
  exact source/stratum counts and hashes. No existing audit artifacts were replaced.
- The [new handoff preflight](../reports/data/ettin150m-v1-review-preflight.json) passed isolated
  Python validation and local save/resume checks: 635 blank rows, no answer key, ZIP unchanged.
  Its test decision existed only in a disposable temporary copy, not in the real review.
- Focused Ruff passed. `rtk .venv/bin/pytest -q -o addopts='' tests/test_prepare_ettin_review.py
  tests/test_blind_audit_bundle.py tests/test_audit_completion.py`: **20 passed**. Tests cover
  deterministic selection, coverage, rejected changed/invalid frames, no answer-key leakage,
  and round-trip import/reverification with synthetic fixtures and schema-25 bindings.
- Original schema-24 ZIP, canonical CSV and manifest hashes are unchanged. They are historical
  artifacts, not the review requested from the user now.

The creation command was `rtk .venv/bin/python scripts/prepare_ettin_review.py`; rerunning it
against the same destinations deliberately fails. Do not regenerate or alter a packet already
sent to a reviewer. No reviewer was contacted and no real decisions or attestation were invented.
