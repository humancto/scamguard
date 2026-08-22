# Dataset schema 20: action-state supervision

## Decision

Schema 20 is accepted as the next frozen dataset experiment. It changes the supervision target
after schema 19 proved that more long binary pairs could learn action ranking without learning a
reliable real-dialogue boundary. It does not open any prediction-sealed benchmark and it does not
authorize a release claim.

The processed manifest SHA-256 is
`1bba8a07fc0efc236d34d6edacc9bdb2a338765ab88b6633246a176c515f8e4e`. The training
artifact SHA-256 is
`6fbecab488b743c7d8ffca4326f9a8a9e70497df3ce9147b08616627b4b75dcb`.

## Right-sized training mix

The training artifact contains 21,234 rows:

| Evidence class | Rows | What the count means |
|---|---:|---|
| Licensed naturally occurring or real-call-derived sources | 8,134 | UCI and Mendeley SMS, WSPR, the IMC forum research release, and CC0 scam-call-derived transcripts |
| Human-authored Taskmaster roleplay | 1,193 | 600 conversation families represented by short and long recent windows; reported separately from real calls |
| Controlled original synthetic data | 11,907 | Existing audited message/dialogue curricula plus the new action-state contrasts |

The model sees no lookup table at inference time. Training-row count therefore does not determine
latency; architecture, token length, quantization, runtime, and hardware do. The schema keeps the
256-token mobile-window contract. Every one of the 6,144 new training contrasts is 326–359 tokens
before truncation, so every row exercises latest-256-token truncation. The held-out state rows are
326–361 tokens. This is enough data for the causal supervision experiment without pretending that
duplicated templates improve deployment speed or generalization.

## New four-state unit

The generator creates 2,048 four-row semantic families across 16 legitimate service domains, four
dialogue structures, four context frames, and eight risk mechanisms. Twelve complete domains form
1,536 training families / 6,144 rows. Four complete domains—financial planning, health scheduling,
parcel service, and technology service—form 512 validation families / 2,048 rows.

Every family has byte-identical prior turns and changes only the final proposed action:

| State | Product label | Purpose |
|---|---|---|
| `routine_safe` | SAFE | Ordinary service action with no sensitive-action language |
| `verified_safe` | SAFE | The same risk vocabulary, but no disclosure or caller-controlled target; the user independently navigates to a verified official channel |
| `unresolved` | UNCERTAIN | Sensitive action or channel is mentioned, but authority and destination are not established |
| `harmful_scam` | SCAM | The caller requests a sensitive action and controls the recipient, destination, device access, or isolation |

Each row carries seven dense boolean targets in addition to the public three-way label:

- sensitive-action language
- requested disclosure or transfer
- caller-controlled target
- official self-navigation
- independent verification
- pressure or secrecy
- irreversible action

The new train slice is exactly balanced by state at 1,536 rows each. Its auxiliary positives are
4,608 sensitive-language, 1,152 disclosure-or-transfer, 1,536 caller-controlled-target, 1,536
official-self-navigation, 1,536 independent-verification, 192 pressure-or-secrecy, and 768
irreversible-action rows. These are supervision labels, not claims that the surface term itself
makes a communication fraudulent.

## Real and online source boundary

- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
  licensed Mendeley SMS releases, the [WSPR/NCSU phishing release](https://github.com/wspr-ncsu/sms-phishing),
  and the IMC 2025 public-forum research artifact remain the primary licensed message/report
  sources already admitted through source-wide privacy normalization and family deduplication.
- [Taskmaster-1](https://github.com/google-research-datasets/Taskmaster/tree/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019)
  supplies CC-BY-4.0 human-authored legitimate transactional dialogue. Its 600 train conversations
  and 447 long-window validation conversations are disjoint by conversation ID.
- [YouTube Scam Phone Call Transcripts](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts)
  supplies the pinned publisher-declared CC0 scam-call-derived positive source. Schema 20 uses 435
  windows from only the 145 connected source families already assigned to training; its validation
  and OOD families are unchanged.
- Direct Reddit scraping remains excluded. Reddit's current
  [Data API Terms](https://redditinc.com/policies/data-api-terms) require separate rights for model
  training. Public reports guide taxonomy, but their text is not copied. The admitted IMC research
  artifact is the rights- and privacy-reviewed forum-derived path.
- BothBosu and AppTek contribute zero training rows. BothBosu remains an open selection slice plus
  a prediction-sealed OOD slice; AppTek remains evaluation-only under its publisher's stated scope.
  GrandgemMa remains excluded because its published composite contains BothBosu rows.

## Leakage, privacy, and benchmark controls

The general validator passed 37,408 unique open examples with no family leakage. The 2,048-row
action-state validation SHA-256 is
`c6d9aa782739990f7dcd0fd5956d90274bfffcf89e868da394d0a458087918fe`.
The unchanged 447-row long SAFE-call validation SHA-256 is
`c96a64bb3c6e5e88419997031b09c72188a41f66be170172726d0afbc1bc317c`.

The 1,820-row primary MOZ holdout was manifest- and count-checked only; no prediction was run.
AppTek's prediction-sealed partition, BothBosu OOD, and YouTube OOD were not opened. Synthetic rows
record their generator, advisory reference, family, state, shared-context hash, and an explicit
`external_benchmark_text_copied: false` assertion. Real-source rows retain the existing email,
URL, phone, and account-like privacy normalization.

## Experiment contract

Schema 20 must be trained with separate evidence and context supervision; feeding these rows into
the old single scalar pair loss would not test the hypothesis. The proposed alert boundary is:

`high-risk requested action AND unsafe context`, with independent official verification allowed to
reduce a premature alert. The exact combination, loss weights, initialization, and thresholds must
be frozen before fitting. Model selection remains development-only, followed by regression,
state-heldout, long SAFE-call, Taskmaster, and BothBosu open gates. Only a complete quality pass can
advance to AppTek/YouTube external selection, export, and physical-device latency.

Under 20 ms remains a deployment gate, not a data-size gate. The 149M teacher is used to establish
the boundary; after it passes, its dense labels and logits can supervise a smaller student. A tiny
student that misses the 97% recall or 2% FPR requirements is not a win, regardless of latency.

The first multi-task teacher recipe was subsequently frozen before fitting at configuration
SHA-256 `c2ca3df6335bf81962aed54281aab904c3074d468cb7cca676f40a0c2d8d9886`. It copies
schema 13's three verdict rows exactly, appends seven deterministic auxiliary logits, weights the
auxiliary loss at 0.5, and limits each dense action-state row to 0.25 weight on the main verdict
loss. The primary alert remains the calibrated three-way verdict for this controlled experiment;
auxiliary logits are diagnostic until a separate combination rule passes independent calibration.
