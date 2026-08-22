# Schema v18 call evidence-action pairs

Freeze date: 2026-08-21. Status: **dataset built and validated; model experiment not yet run**.

## Why schema 18 exists

Schema 17 established that matched contexts are necessary but its minimum 288-family training
matrix was too small. Even on seen pair families, the candidate reached only 76.74% scam recall,
19.79% SAFE false-positive rate, and 83.33% pair ordering. Five mechanisms reached perfect
relative ordering while every `secrecy_isolation` pair was reversed. The model moved the entire
call distribution toward SCAM instead of learning that a harmful external action—not an unusual
opening or a skeptical customer—is the decision boundary.

Schema 18 therefore expands variation along the dimensions schema 17 proved were underfit. It is
not an unfiltered synthetic-data increase.

## Construction

The deterministic Apache-2.0 generator emits 2,048 pair families, or 4,096 rows:

- 16 service scenarios;
- four call structures: inbound, requested callback, transfer, and requested outbound update;
- four shared framing conditions: neutral, unexpected call, customer skepticism, and privacy
  concern;
- eight harmful action mechanisms: credential code, remote access, protection transfer,
  cryptocurrency fee, secrecy/isolation, login link, gift card, and advance fee;
- four independently written action phrasings per mechanism, rotated across context frames;
- exactly one SAFE and one SCAM continuation for each shared four-turn context.

The framing condition is identical within each pair. This deliberately places phrases such as
"unexpected call," "suspicious message," and privacy concern in both labels so the model cannot
use customer skepticism as a shortcut. Only the final agent action changes. SAFE endings mix
ordinary service completion, independent verification, explanation without forced action, and a
normal follow-up. SCAM endings request a concrete harmful action and must trigger at least one
auditable ScamGuard evidence signal.

No AppTek, BothBosu, YouTube, Reddit, regression, or other benchmark text is copied. The generator
rejects duplicate IDs or normalized text, incomplete/unbalanced pairs, changed shared contexts,
unsafe cues in SAFE endings, and SCAM endings without extractive evidence.

The pinned ModernBERT tokenizer measures 87–116 tokens per row (median 102, p95 110); 0/4,096
exceed the 256-token inference window.

## Size and split

Schema 18 starts from schema 14. It preserves the 161 CC0 real scam-call early windows and excludes
both rejected schema-15 independent SAFE rows and rejected schema-17 pair rows.

Financial planning, health scheduling, parcel service, and technology service are complete
scenario holdouts. Their 512 pair families form a 1,024-row balanced validation split. The other
12 scenarios contribute 1,536 pair families, or 3,072 balanced training rows.

| Partition | Pair families | SAFE | SCAM | Rows |
|---|---:|---:|---:|---:|
| training increment | 1,536 | 1,536 | 1,536 | 3,072 |
| scenario-held validation | 512 | 512 | 512 | 1,024 |
| generated total | 2,048 | 2,048 | 2,048 | 4,096 |

The processed training set contains 17,295 rows. Across its training, development, inherited
open diagnostics, unchanged regression split, and paired validation, schema 18 contains 31,998
rows with no ID, exact-text, or family leakage detected by the repository validator. Inherited
artifacts remain byte-identical to schema 14.

## Training and evaluation contract

The candidate initializes from schema 13, not a rejected call checkpoint. Its 14,062 inherited
rows retain text-free teacher logits. The 161 real scam-call rows and 3,072 pair rows are the only
3,233 unanchored training examples.

The frozen recipe visits inherited rows once and paired rows twice, for 20,367 sample exposures
and 1,273 optimizer steps at batch 16. It uses retention weight 4, pair loss weight 2, pair margin
3, and a latest-256-token input policy. The increased pair exposure is predeclared because schema
17 underfit its seen families; it is not chosen after observing schema-18 validation.

The paired validation is not used for fitting, calibration, or threshold selection. The threshold
still comes only from unchanged development SAFE/SCAM rows at the 2% FPR cap. All quality gates
must pass before Core ML export. The optimized Core ML path—not training-time PyTorch—is the
product's <=20 ms desktop gate, followed by a required physical-mobile measurement.

AppTek, YouTube, and BothBosu validation remain open selection diagnostics only. Their OOD splits,
the MOZ holdout, and the primary sealed holdout remain unopened.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| generated 4,096-row source | `414a6c461c0ca794fbda6e51ca8e927023266ce6082337c678bd4b600ebc7dbf` |
| generator manifest | `0c01817dce9ec6952ac1d777df5a47fe3ed7383cc6a6ae9e47ffdbe3292e3c5c` |
| schema-v18 manifest | `2df6ee1774f06beb2aa82e2d5df67289003710c296e2b4a5eb9c99036b83d605` |
| schema-v18 train | `e69a3421fe6f736662a9aca796eae80da3a61a585c29a3b0c1d2713dd4a1cb04` |
| paired validation | `bfb28504072ac9f785dbda2f5bbab9a8837f6f6f5310827516d639a7b50ae8dd` |
| schema-v14 parent manifest | `83ee137212eb99bac46f81a8ce265e7bad003c58cef35f5f951a9024a6ecc09b` |
| schema-v14 parent train | `93fa3d3ddea2c51b8093bc8fa486d26e345c08011da5de27e6d6dc1b42e6da97` |

Reproduce and validate with `make schema18-call-evidence-pairs`. Run the frozen preflight with
`make encoder-schema18-preflight`.
