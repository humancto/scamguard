# International Robocalls external diagnostic

## Frozen decision

The International Robocalls Dataset is useful new real-call evidence, but it is not training data
for ScamGuard. The pinned Zenodo release is CC-BY-NC-4.0, so every row is excluded from fitting,
threshold selection, commercial artifacts, and public row redistribution. The text-only release is
retained locally as a newly sourced, reporting-only diagnostic.

- DOI: <https://doi.org/10.5281/zenodo.21066049>
- Zenodo record: `21066049`, metadata revision 4, modified
  `2026-07-01T07:00:08.494820+00:00`
- Archive SHA-256:
  `2a5b09a9ea260001f67465bed329dbf51f9e1d82fbe3d11aa1656105ef3a7995`
- Publisher MD5: `318c485aa38e1760fe7ac3338f20526c`
- License: CC-BY-NC-4.0; commercial use requires a separate license.

The artifact contains real inbound honeypot calls. The publisher says every released recording was
verified by at least one human as a robocall. That does not make every row adjudicated scam truth:
campaign categories were produced by the publisher's analysis pipeline and are treated here as
silver labels.

## Predeclared construction

1. Download only `text.zip`; do not acquire the 155 MB audio or 154 MB call-detail archive.
2. Require the exact archive hash, exact member list, exact CSV schema, and observed row counts.
3. Replace URLs, email addresses, phone-like values, and long digit sequences before IDs,
   clustering, or output.
4. Map the ten publisher campaign names containing `Scam` to `SCAM`. Map financial spam,
   insurance spam, political, telemarketing, and unclassified campaigns to `UNCERTAIN`.
5. Supply zero `SAFE` rows. Human robocall verification is not legitimate-call ground truth.
6. Remove overlap against the current canonical corpus, schema-v25 call curriculum, YouTube-call,
   PPoNE, and BothBosu diagnostics; then retain one representative per near-template family.
7. Evaluate the frozen Stage 7 adapter once with its existing development calibration. Do not tune
   a threshold, model, prompt, label map, or later training curriculum against these predictions.

The Zenodo description reports 724 US and 115 international transcripts. Direct inspection of the
hash-pinned archive finds 725 and 116 data records. The builder fails closed on those observed
counts and records the discrepancy instead of silently conforming the files to the prose.

## Metrics frozen before predictions

This diagnostic can report publisher-scam recall, `UNCERTAIN` recall, macro F1, and the fraction of
all verified robocalls that avoid a `SAFE` verdict. It cannot estimate ordinary-call false-positive
rate, scam prevalence, or mobile latency. It is not a promotion gate and cannot authorize
quantization or Hugging Face publication.

## Result

The immutable build is complete. The 841 source rows collapse to 47 independent family
representatives: 19 silver `SCAM` and 28 `UNCERTAIN`. It removes 529 exact duplicates and 201
same-label near-template repetitions, quarantines 64 rows from five mixed-label near-template
groups, and finds zero near overlaps across 79,021 prior reference rows.

The publisher's claim that phone numbers were replaced is not borne out by the pinned text ZIP.
Before normalization, 601 rows contain phone-like values, 40 contain URL-like values, and 25
contain long digit sequences. No such value survives the built diagnostic.

- Diagnostic SHA-256:
  `4604d1566dc09a616bf57318359ac2946036606d5ed03acb874b83ff9e5967af`
- Quarantine SHA-256:
  `a6d52f4297f5d69c493161dd87f72d5fc276d7b3d0e0449f9d85ac982aa832dd`
- Machine-readable source audit:
  `reports/source-audits/international-robocalls.json`

The one frozen Stage 7 evaluation remains pending. No model prediction has been inspected while
freezing these construction rules and metrics.
