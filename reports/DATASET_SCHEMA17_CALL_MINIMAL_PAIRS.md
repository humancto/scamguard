# Schema v17 structure-matched call minimal pairs

Freeze date: 2026-08-21. Status: **dataset admitted for the next open experiment; no model result
yet**.

## Why this dataset exists

Schema v16 proved that regression-logit retention and source balancing can preserve broad text
behavior while still learning the wrong call boundary. It reached 98.57% on the open YouTube
scam-call selection but falsely flagged 105/348 AppTek legitimate-call windows and 107/153
legitimate BothBosu dialogues. Every AppTek false alarm occurred in an early call opening.

The next increment therefore changes the supervision geometry rather than adding another pile of
independent examples. Each SAFE/SCAM pair holds the service domain, opening structure, speaking
style, length, neutral vocabulary, and first four turns constant. Only the final agent action
changes.

## Construction

The deterministic Apache-2.0 generator emits 384 pair families, or 768 rows:

- 16 service scenarios;
- four structures: inbound, requested callback, transfer, and requested outbound update;
- six risk mechanisms: verification-code request, remote access, protection-account wire,
  cryptocurrency fee, secrecy/isolation, and login link;
- exactly one SAFE and one SCAM row per scenario × structure × mechanism family.

SCAM actions are original copy grounded in the FTC scam-mechanics taxonomy and must expose at least
one extractive ScamGuard signal. SAFE actions contain no explicit anti-scam phrases or risk tokens;
they simply complete the ordinary service request. The generator rejects duplicate IDs or text,
unbalanced families, differing pair-context hashes, SCAM rows without evidence, and safety cues in
SAFE rows. It copies no AppTek, YouTube, BothBosu, Reddit, or regression text.

The pinned ModernBERT tokenizer measures 85–114 tokens per generated dialogue (median 99, p95 109),
with 0/768 exceeding the 256-token mobile window.

## Size and family split

Schema v17 deliberately starts from schema v14, not schema v15. This retains the 161 CC0 real
scam-call-derived early windows but discards the rejected independent synthetic SAFE increment.

Four complete service scenarios are held out before fitting: financial planning, health
scheduling, parcel service, and technology service. Their 96 pair families (192 balanced rows)
form `call_pair_validation.jsonl`. The remaining 288 pair families add 576 balanced training rows.

| Partition | Pair families | SAFE | SCAM | Rows |
|---|---:|---:|---:|---:|
| training increment | 288 | 288 | 288 | 576 |
| family-held paired validation | 96 | 96 | 96 | 192 |
| generated total | 384 | 384 | 384 | 768 |

The processed training set has 14,799 rows. All processed artifacts contain 28,670 unique rows.
The schema-v14 development, unchanged regression, and inherited diagnostics remain byte-identical.
Validation reports no duplicate IDs, exact text overlap, or family leakage. The 1,820-row primary
holdout was checked only for its existing manifest/count contract and remains prediction-sealed.

This is the minimum complete mechanism matrix: there is one pair per scenario, structure, and risk
mechanism, with no surface-form replicas. More variants are not authorized until this dose shows
family-held and external legitimate-call benefit.

## Evaluation and use contract

The 192 paired-validation rows are not fitting or threshold rows. They test whether the model
learns the changed action across unseen service scenarios. They cannot establish real-world
performance because they are synthetic minimal contrasts.

The next candidate must also clear the unchanged development/regression, AppTek open selection,
YouTube open selection, BothBosu selection, and Taskmaster selection gates. AppTek remains
evaluation-only. Its 1,396-window OOD split, YouTube's 80-window OOD split, BothBosu OOD, and the
MOZ holdout remain sealed. Failure on any open gate blocks export and mobile claims.

Training should initialize from schema 13 and retain its logits on the 14,062 inherited rows. The
576 new paired rows and 161 real scam-call rows have no teacher anchor. A pair-aware objective must
reward the SCAM member's scam margin over its matched SAFE member; ordinary source weighting alone
was rejected by schema 16.

## Online-source decision

The live GitHub/Hugging Face refresh did not justify adding bulk rows. ThaiScamCall is CC-BY-4.0
but AI-generated TTS audio, not real call text, and its current viewer exposes only 100 rows while
the card claims 21,287 clips. `adamtc/scam_dialogues` is an Apache-2.0 Vietnamese synthetic corpus
without sufficient row-generation provenance for fitting. FraudLens-RU is CC-BY-4.0 public
anti-fraud channel/article material, not victim-message or call transcripts. Korean voice-phishing
repositories still do not establish usable rights for the underlying transcripts. These sources
remain audit or taxonomy candidates, not training mass.

Direct Reddit posts remain excluded. Reddit's Data API Terms state that user content cannot be used
to train an ML/AI model without express permission from the applicable rightsholders. ScamGuard's
forum-derived fitting rows continue to come only from the CC-BY-4.0 IMC 2025 research release.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| generated 768-row source | `375402c70f6ff5dce26fcaf6d8862d61ccd82de106106f0cbbe43066941f44b9` |
| generator manifest | `d4f17384ffba8b35c048ea122dfe1eecde0b4e140e3096ac7b4b6aa00bafd840` |
| schema-v17 manifest | `b5184b9c08de3582e72935d10ea3dac429b5efdd04d13de7044a7c172a6e5f37` |
| schema-v17 train | `256c0f653799b454c01eaaf7f6b749d8c0b5508e89cea4f92f8a41034d836c14` |
| paired validation | `9208cd3df9568b7ccbc97266bd2c726f7d57c7fbd4cc01ba3d92aed7fa885205` |
| schema-v14 parent manifest | `83ee137212eb99bac46f81a8ce265e7bad003c58cef35f5f951a9024a6ecc09b` |
| schema-v14 parent train | `93fa3d3ddea2c51b8093bc8fa486d26e345c08011da5de27e6d6dc1b42e6da97` |

Reproduce and validate with `make schema17-call-minimal-pairs`.
