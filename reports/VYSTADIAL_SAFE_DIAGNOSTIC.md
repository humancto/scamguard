# Vystadial real-call SAFE diagnostic

## Frozen decision

Vystadial adds a useful real-human telephone negative control, but it does not enter ScamGuard
training. The source is narrow: crowdworkers called a benign restaurant-finding dialogue system,
and the released English corpus contains only the callers' short utterances rather than complete
two-sided calls. The result therefore measures false alarms on ordinary service requests; it does
not establish safety on banking, account-recovery, delivery, health-care, or other action-bearing
calls.

- Primary release: <https://www.openslr.org/6/>
- Primary paper: *Free English and Czech Telephone Speech Corpus Shared Under the CC-BY-SA 3.0
  License*, LREC 2014
- Processed text/audio derivative: <https://huggingface.co/datasets/BrunoHays/vystadial-telephony-en>
- Pinned derivative revision: `b7454512a3547c829be201d9afceb64af8c28e61`
- License: CC-BY-SA-3.0; commercial use is permitted with attribution and share-alike.

Because share-alike obligations for trained weights deserve a separate release decision, this
experiment uses the corpus only as a local diagnostic. It does not redistribute audio or rows and
does not use any row for fitting, calibration, threshold selection, or promotion tuning.

## Source audit and construction

The two pinned parquet shards contain 48,254 rows and 42.60 hours from 30 speakers. Their hashes
are:

- `test-00000-of-00002.parquet`:
  `399e6e1f13846b1bbad1ddb60df552c443137db56b86f7153d1a0b6e76b5844c`
- `test-00001-of-00002.parquet`:
  `3c7108edc251d5f7d95c2bbab53c1716b7d9c2ebe6d6670142602cec2b24fb92`

The builder reads transcript and provenance columns only; embedded audio is never decoded. It
fails closed unless every row retains the expected English, 16 kHz, mono, one-segment metadata,
the declared license, and an unchanged upstream orthographic transcript. URLs, email addresses,
phone-like values, and long digit sequences are normalized before IDs or clustering.

Of 48,254 source rows, 1,504 are too short for admission. The remaining 46,750 rows collapse by
35,599 exact duplicates. Four more rows overlap an existing ScamGuard reference family. SimHash
clustering turns the remaining 11,147 rows into 7,718 near-template families; the largest family
contains 1,387 variants. A deterministic round-robin cap retains 1,024 families across all 30
speakers. No mixed-label conflict is possible because the collection context supplies SAFE truth
only.

- Diagnostic SHA-256:
  `0a2c9a8e1e100fddd338d22acf2dc4ee85401ac40710199196e7171d1b463042`
- Machine-readable source audit: `reports/source-audits/vystadial-safe.json`
- Raw and derived rows: local and Git-ignored

## Frozen Stage 7 result

The Stage 7 adapter was evaluated once using its existing development temperature and thresholds.
The evaluator ran on Apple MPS, loaded the identity-checked development score cache, and computed
all 1,024 new scores from scratch. This was a selection-only diagnostic, not a release gate. The
construction code existed before scoring, but was not committed as an immutable prospective
declaration, so the result must not be presented as a sealed benchmark.

| Metric | Frozen result |
|---|---:|
| Thresholded SAFE false positives | **0/1,024 (0.000%)** |
| SAFE FPR 95% Wilson interval | 0.000–0.374% |
| Calibrated verdicts | 1,024 SAFE, 0 UNCERTAIN, 0 SCAM |
| Raw argmax verdicts | 1,023 SAFE, 1 UNCERTAIN, 0 SCAM |
| Binary Brier score | 0.0000105 |

The one non-SAFE raw argmax had four words and was recovered to SAFE by the already-frozen
three-verdict policy; no threshold was changed. The score contains no evidence of a broad false-
positive problem on simple benign telephone requests.

- Evaluation report SHA-256:
  `067203758a8165f5be8393c023135702533515275b890c0813fae908a268769c`
- Text-free prediction ledger SHA-256:
  `dddf39d13cd0c58596672b40835e575054e12400f75005311f590dc0a9529b3f`
- Adapter SHA-256:
  `14f1d2bf121e76fa158cea722416994ec9cbfbc545956d24b9693b7808441357`
- Frozen calibration SHA-256:
  `a2309cc5479a85efa51ab74219411593a1e8b372888b4b8f97aaa1bf741b078b`

The BF16-plus-LoRA reference run measured 79.45 ms median and 92.77 ms p95 over the standard
50-example development latency loop, with a sampled MPS driver allocation peak of 2.50 GB. Those
numbers are diagnostic runtime evidence, not the mobile GGUF target.

## What this changes

Adding these rows to training would supply little boundary information: Stage 7 already classifies
every retained family correctly. It would also risk over-weighting a narrow restaurant-request
domain and would bring share-alike release obligations into the training artifact. Vystadial stays
frozen and evaluation-only.

The next real-data increment must instead contain independently licensed, legitimate but
action-bearing calls: identity checks, billing disputes, appointment scheduling, delivery changes,
account recovery, or financial-service conversations. That source must be split by participant or
conversation before prediction and must not be selected using the already-open phone, PPoNE,
international-robocall, or Vystadial outcomes.

Reproduce the local build and frozen diagnostic with:

```bash
make vystadial-safe-calls
make qwen-08b-stage7-vystadial-safe
```

The MPS evaluation command must run outside a restricted sandbox; `--require-mps` fails closed
rather than silently substituting CPU.
