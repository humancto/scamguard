# Ettin: fresh-context AI second opinion

2026-10-11 local / 2026-10-12 UTC. **AI_INTERNAL_SECOND_OPINION — not independent human adjudication, new ground truth, rights clearance or release approval.**

## Outcome

Keep V2 rejected. The review supports investigating label/rubric conflicts, but **does not support treating the remaining failures as mostly bad labels**: 29 of the 43 disputed SAFE-reference alerts were independently judged SAFE by this AI reviewer, including 25 of 28 BothBosu alerts. There are also plausible missed scams hidden among supposedly concordant SAFE controls. Do not suppress every disputed warning or automatically relabel this packet.

## Method and limits

A fresh-context agent read the blind ZIP's complete instructions and frozen rubric, then individually assessed all 96 messages in six batches of 16. Every row has a manually authored AI label, reason, confidence, ambiguity and possible-sensitive-data flag. Scripts performed integrity checks and arithmetic only. Decisions and exact UTF-8 text hashes were frozen at **2026-10-12 03:17:27 UTC**, and their hashes reported to the coordinating agent **before** opening owner labels, model verdicts or project reports. No memory was consulted during review.

Phase 2 verified all 96 unique joins through the existing hash-bound private selection: exact message hashes, canonical input digest, metadata, ID coverage and manifest bindings. The comparison uses the recorded V2 selection verdicts, not new inference. Original prediction-ledger/model bytes were not independently revalidated by this reviewer. The coordinating agent separately checked the blind-phase coverage and hashes.

The sample intentionally contains all 43 SAFE-reference alerts, four SCAM-reference SAFE predictions and 49 selected concordant controls. It is **not a random population sample, independent final test or accuracy estimate**. AI confidence is subjective, not calibrated; 23 decisions have medium confidence. No training, downloads, inference, sealed-test access, dataset edits, human-return edits, external upload or publication occurred. The data-validation skill guided exact joins, denominator checks and selection-bias disclosure.

## Frozen comparison

AI decisions: **60 SAFE, 26 UNCERTAIN, 10 SCAM**. Source references: 88 SAFE / 8 SCAM. V2 verdicts: 49 SAFE / 47 SCAM.

| Selected group | Rows | AI SAFE | AI UNCERTAIN | AI SCAM |
| --- | ---: | ---: | ---: | ---: |
| BothBosu: reference SAFE, V2 SCAM | 28 | 25 | 3 | 0 |
| Phone: reference SAFE, V2 SCAM | 15 | 4 | 10 | 1 |
| BothBosu: reference SCAM, V2 SAFE | 1 | 0 | 0 | 1 |
| Phone: reference SCAM, V2 SAFE | 3 | 0 | 2 | 1 |
| BothBosu: concordant SAFE controls | 30 | 29 | 1 | 0 |
| Phone: concordant SAFE controls | 15 | 2 | 10 | 3 |
| Concordant SCAM controls, both sources | 4 | 0 | 0 | 4 |
| Total | 96 | 60 | 26 | 10 |

AI/reference agreement is **66/96 (68.75%)**, disagreement **30/96 (31.25%)**. AI/V2 agreement is **36/96 (37.50%)**, disagreement **60/96 (62.50%)**. Reference/V2 agreement is 49/96 by the packet's selection design. These are descriptive agreement fractions, **not model accuracy or corrected FPR**. Source: hash-verified private canonical/selection files; aggregate calculations are in the local comparison receipt.

## Prioritized findings

1. **High — substantial model false-alarm evidence survives adjudication.** Of 43 reference-SAFE alerts, 29 support the false-alarm interpretation, 13 are suspicious/ambiguous and one supports SCAM. Ordinary insurance, software and delivery conversations dominate the AI-judged SAFE errors. The frozen rubric does not automatically make all unsolicited advertising SCAM, nor does it unambiguously make every ordinary sales exchange UNCERTAIN. Human reviewers should resolve the ordinary-advertising/generic-spam boundary explicitly. Do not use a favorable reinterpretation to erase historical failures. V2's recorded 15/170 and 28/153 SAFE FPR failures remain unchanged. [Recorded results](../ETTIN150M_RECOVERY_V2.md), [readiness analysis](../../docs/ETTIN_HF_READINESS.md).

2. **High — error-only review misses potentially unsafe controls.** Two of four reference-SCAM/V2-SAFE cases support genuine missed-scam interpretations; two remain UNCERTAIN, not confirmed SAFE. Three reference-SAFE/V2-SAFE controls were AI-judged SCAM: explicit password solicitation, authentication-code solicitation, and a branded lookalike support destination. The last is a medium-confidence boundary case, not verified domain ownership. These are candidates for qualified review, not automatic new labels. Across the packet, 30 source-label disagreements comprise 26 UNCERTAIN decisions and four SAFE-to-SCAM proposals. Zero disagreement is the wrong completion criterion. [Adjudication receipt](../data/ettin-dialogue-v3-adjudication.json).

3. **High — fresh-base exclusion is an ancestry strategy, not a clearance certificate.** Prefer preparing a fresh-base experiment over another V2 continuation if the recorded phone-generator and YouTube-source questions remain unresolved. Excluding those sources only from future replay cannot remove V1/V2 ancestry. However, the inventory marks **all 14 training sources review_required**: dropping two does not establish the remaining twelve are approved. Resolve training-use scope, retained derivatives, provenance and applicable privacy review before calling a corpus clean. This review assessed the local evidence inventory, not external rights evidence or legal permissions. [Provenance](../../docs/ETTIN_RELEASE_PROVENANCE.md), [inventory](../../configs/ettin150m-release-provenance.json).

4. **Medium — rebind audit work to the actual next corpus and serving policy.** The existing 635-row packet remains valid for its unchanged corpus; this AI review completes none of it. A fresh subset needs a new explicit corpus binding and reviewed coverage plan, not an automatic carry-over of the old certificate. Preserve the old packet; exact unchanged rows may eventually reuse genuine human decisions only with matching text, labels, rubric and documented assignments. There are no completed human decisions to reuse today. Decide the new corpus before commissioning review of rows that may be excluded. V3 preprocessing changes served inputs and requires its own frozen calibration, quality and latency evidence; text-retention tests do not prove improved predictions. [Audit reuse scope](../../docs/ETTIN_V2_AUDIT_REUSE.md), [V3 limits](../../docs/ETTIN_HF_READINESS.md).

Thirteen rows received conservative possible-sensitive-data flags, including real-looking names, a name/date combination and non-obvious case/order identifiers. These are privacy-review candidates, not confirmed real-person exposures. No raw message, row identifier, secret or row mapping is included in this report; all row-level decisions and comparisons remain in the private local review directory.

## One finite next step

**Prepare one owner-approved fresh-base experiment contract; do not fit yet.** Within that single preparation pass: select the permitted source subset (excluding unresolved sources and attributable derivatives), bind the exact corpus and base, settle rubric boundary examples with a qualified independent reviewer, and define candidate-specific audit coverage. Do not place this opened 96-row diagnostic packet into training or relabel it as a final test.

The contract must name one input policy, one fixed fit/compute cap and terminal checkpoint, development-only calibration, all declared per-slice recall/FPR and UNCERTAIN diagnostics, desktop serving-parity/latency/size checks, and an unopened candidate-specific final-validation set. Freeze these before fitting. If source scope, rubric or review coverage cannot be resolved, end preparation with a blocked record. If a later separately authorized single fit fails any frozen requirement, reject it without retries or benchmark-driven threshold changes. A passing open screen still requires genuine human/provenance review and the predeclared final evaluation before publication.

Desktop-first is a sensible explicit scope: the recorded desktop warm-latency result supports further desktop work, not release readiness or mobile performance. Require cold-start/operational measurements for the actual desktop package; physical-mobile evidence is only needed if making mobile claims. Neither HF login nor packaging can cure quality or provenance failures.

## Exact evidence bindings

SHA-256 values; full local input-file inventory and all grouped counts are in `data/audit/ettin-dialogue-v3/ai-second-opinion-20261011/comparison-receipt.json`.

| Artifact | SHA-256 |
| --- | --- |
| Original blind ZIP | `3fd7a7be3557ab9a7fb57c089a09f989873d8f47e3388f2d686c43f520958d5e` |
| Frozen rubric, canonical JSON | `d9dcc931447ce5229ca5e07398b20944759dfe0f0224369c14c2729be10cbb59` |
| Blind input digest | `85ca1afb7e42e6eec743bea3679e05052699679403a3c248ec15331df5e1da54` |
| Frozen AI decisions | `882e2088135dbdbd0c0604e9bdc662d28dbe244f0c1c3d9585c480acbdd28280` |
| Exact text-hash bindings | `d2fc7350a1c77ace80ed496b97bc85bd7f295a4771b9293eed89a8b21e418634` |
| Phase-1 freeze receipt | `ad18a83c4fcb9a76c10d8d0065c0ee88d5c6c70cc7bab1ae22510e87b128117f` |
| Aggregate comparison receipt | `b97b316554d9a9b033c796b511b1b647130afeb651c7f01d7e7d01f501b6b757` |
| Private row comparison | `d1533f825b09600781019effb04dc4df49ea963a7bcecd778d88f04436fa6202` |
| Owner canonical CSV | `8a296312f5cf04067fdab677e8fa8e3246a1a47a6556363f4e805b7d4d79fde1` |
| Owner selection mapping | `ef0b289d719680997207e6a2d5cfc3bd537186623cabe1544d4613448451f795` |
| Owner canonical input digest | `e8586eec3a6b172e42ebad66e2216b7a5737ee26ab11c63b702a94bdbac0732b` |
| V2 result document | `c916e6f1e7e760a10d86eb0a3b1051b0a0d16df22c93f3c106b84511bd1abbe0` |
| Provenance inventory | `791fb85a06a8594983104219c51ffb531989e038fb02f849451d10cbc56368c0` |

The freeze is procedural and hash-bound, not an operating-system immutability claim. Original labels, model verdicts, failed gates and human-review requirements are unchanged.
