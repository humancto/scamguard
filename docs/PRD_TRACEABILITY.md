# ScamGuard PRD traceability

This ledger maps the supplied TinySpecialists / ScamGuard PRD v0.1 to current, inspectable
evidence. Its purpose is to prevent a research checkpoint, simulator smoke, or different-dataset
score from being presented as the finished product. The supplied PRD file is SHA-256
`621c85a90c273fd41d84d176864c857159c349abb11c522247151129024ff25e`.

Status meanings:

- **Proven**: current repository or measured evidence establishes the requirement.
- **Partial**: a reproducible implementation exists, but release-level evidence is incomplete.
- **Pending**: the required result or independent receipt does not exist yet.

| PRD requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Open-source ScamGuard repository | **Proven** | Apache-2.0 [`LICENSE`](../LICENSE), reproducible [`Makefile`](../Makefile), and public `humancto/scamguard` origin | Keep release artifacts and source revisions hash-bound. |
| Frozen `SAFE` / `UNCERTAIN` / `SCAM` product contract | **Proven** | [`src/scamguard/schema.py`](../src/scamguard/schema.py), [`src/scamguard/taxonomy.py`](../src/scamguard/taxonomy.py), and contract tests | None for the schema; model quality remains separate. |
| One-call Python SDK, CLI, and local web demo | **Proven** | [`src/scamguard/scanner.py`](../src/scamguard/scanner.py), [`src/scamguard/cli.py`](../src/scamguard/cli.py), [`src/scamguard/demo.py`](../src/scamguard/demo.py), and scanner tests | The final accepted runtime pack must replace research controls in release examples. |
| Explain why with grounded evidence and a safe action | **Partial** | Extractive spans and deterministic actions in [`src/scamguard/signals.py`](../src/scamguard/signals.py); schema/scanner tests require exact source offsets and SAFE invariants; the text-free frozen-ledger audit and predeclared coverage gates are implemented in [`scripts/evaluate_product_contract.py`](../scripts/evaluate_product_contract.py) and [`scripts/check_product_contract_gates.py`](../scripts/check_product_contract_gates.py) | Pass the quantized core-test and dialogue gates, then independently review semantic category/evidence/action correctness; do not describe deterministic signals as generated reasoning. |
| Appropriately licensed real data plus controlled synthetic data | **Partial: source declarations pinned; release rights unresolved** | [`ETTIN_RELEASE_PROVENANCE.md`](ETTIN_RELEASE_PROVENANCE.md), the hash-bound 14-source inventory and [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md) | Phone generator/version/terms and underlying YouTube rights/privacy require review or a fresh-base exclusion route. Independent label/privacy review is also pending. Raw Reddit posts remain excluded; source license tags alone are not blanket clearance. |
| Hard negatives, adversarial positives, multilingual variation, deduplication, and family isolation | **Proven for the benchmark/data factory** | [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md), [`docs/DATASET_SIZE_DECISION.md`](DATASET_SIZE_DECISION.md), validators, overlap audits, and frozen source-family splits | Final model must pass the untouched gates; data engineering alone does not prove model quality. |
| Human-audited representative slice | **Pending** | Current schema-25 635-row packet and exact return requirements in [`INDEPENDENT_REVIEW_HANDOFF.md`](INDEPENDENT_REVIEW_HANDOFF.md); unchanged-corpus v2 bindings in [`ETTIN_V2_AUDIT_REUSE.md`](ETTIN_V2_AUDIT_REUSE.md) | All 635 decisions remain blank. Independent humans must complete this one blind label/privacy audit, including qualified language assistance. Replay-weight changes do not require a duplicate packet or make it a presentation-weighted audit; AI-internal review cannot authorize release. |
| Same benchmark for rules, classical, open neural, and larger-model baselines | **Proven for recorded research baselines** | Model ladder and linked run reports in [`README.md`](../README.md); paired-comparison tooling in [`benchmarks/compare_paired.py`](../benchmarks/compare_paired.py) | Rerun any claimed external winner on the same final sealed benchmark before a SOTA claim. |
| Sub-1B serious specialist | **Partial** | Ettin-150M v2 completed its separately authorized one-pass recovery and 21 open slices; 16/18 promotion checks pass, but phone/BothBosu SAFE FPR still fails. See [`ETTIN150M_RECOVERY_V2.md`](../reports/ETTIN150M_RECOVERY_V2.md). The rejected fresh v1 fit remains in [`ETTIN150M_BOUNDED_V1.md`](../reports/ETTIN150M_BOUNDED_V1.md); historical Qwen results remain in [`QWEN08_PHONE_STAGE7.md`](../reports/QWEN08_PHONE_STAGE7.md) and [`BANKING77_SOURCE_ADMISSION.md`](../reports/BANKING77_SOURCE_ADMISSION.md). | No tested candidate satisfies the complete quality requirements. The Qwen line and completed Ettin recovery do not authorize further fitting. |
| Scam recall at least 97%, SAFE FPR at most 2%, core-category recall at least 97%, macro F1 above 0.94 stretch | **Pending for a final candidate** | Fail-closed definitions in [`scripts/check_qwen08_full_gates.py`](../scripts/check_qwen08_full_gates.py) and [`scripts/check_primary_v8_gates.py`](../scripts/check_primary_v8_gates.py) | Obtain passing frozen regression and untouched primary receipts from the same final artifact. |
| Publish calibration, adversarial/OOD behavior, false-positive analysis, latency, RAM, and artifact size | **Partial** | Existing run reports and required-report contract in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | Repeat and publish the complete evidence bundle for the accepted quantized model. |
| Quantized local artifact | **Partial** | Hash-verified upstream Q4 runtime control and native scorer; final merge/export chain in [`Makefile`](../Makefile) | Merge and Q4 export are forbidden until the trained BF16 challenger passes every gate; then quantized parity and quality must pass again. |
| Normal-laptop local inference | **Partial** | Ettin v2's actual SDK completed 19,823 scans on the Apple Silicon/MPS desktop and measured 17.12 ms warm p95, with the raw 300-call trace in [`sg-ettin150m-recovery-v2.open.json`](../reports/runs/sg-ettin150m-recovery-v2.open.json); historical v1 and GGUF controls remain separately documented. | This passes the bounded experiment's 20 ms desktop limit, not a mobile claim or a general laptop guarantee. The encoder is rejected on quality; a final accepted artifact must retain its own runtime evidence. Q4 is not a universal encoder requirement. |
| Credible mobile path | **Partial** | Shared C ABI, Swift wrapper, Android JNI/Kotlin wrapper, package builders, and correctness smokes in [`docs/MOBILE_RUNTIME_INTEGRATION.md`](MOBILE_RUNTIME_INTEGRATION.md) | Run the final artifact on physical iOS and Android devices using [`docs/MOBILE_BENCHMARK_PROTOCOL.md`](MOBILE_BENCHMARK_PROTOCOL.md). Simulator and desktop evidence cannot substitute. |
| Small model competitive with or better than much larger models | **Pending** | Comparison discipline and same-row paired test contract in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | A final sealed, same-benchmark comparison has not yet established “beats SOTA.” Do not make that claim from different datasets or the open regression split. |
| Reproducible training and evaluation | **Proven for research workflow** | Hash-pinned configs, data manifests, [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md), tests, and fail-closed Make targets | Final release must bind the accepted adapter, merged model, GGUF, calibration, ledgers, runtime packages, and source revision. |
| Hugging Face publication | **Pending** | New encoder-only blocked staging in [`stage_ettin_release.py`](../scripts/stage_ettin_release.py); exact remaining steps in [`ETTIN_HF_READINESS.md`](ETTIN_HF_READINESS.md). Historical Qwen-specific release verifier remains unchanged. | Ettin still needs passing quality, genuine human review, rights clearance and its own frozen final-validation contract/results. Runtime/hash/model-card evidence must bind the accepted artifact. Physical-mobile proof is required for mobile claims, not desktop-only HF publication. |

## Completion rule

The subsequent [release-preparation pass](ETTIN_HF_READINESS.md) adds opt-in action-preserving
preprocessing, a 96-row blind benchmark-label adjudication packet, a source-rights inventory and
a blocked local encoder package. All 745 tests pass; original V2 inputs match on 44,259 rows.
No new model prediction, training, calibrated threshold or sealed evaluation was produced.
Neither the retention fix nor the package changes the failed quality results below.

The 2026-10-11 [Ettin-150M v2 recovery](../reports/ETTIN150M_RECOVERY_V2.md) completed its one
authorized 1,853-step replay pass, fixed terminal checkpoint, and all 21 declared open slices.
It measured 17.12 ms warm full-scanner desktop p95 and passed 16/18 quality/runtime promotion
checks. Development SCAM recall improved from 93.19% to 98.83%; MultiDoGO SAFE FPR improved
from 3.57% to 1.79%. Both now pass their frozen limits. Phone-source SAFE FPR remains 8.82%
(15/170) and BothBosu dialogue SAFE FPR remains 18.30% (28/153), both above 2%; v2 is rejected.
Three-class macro F1 is 0.7919 on open regression and 0.7107 on development, below the 0.94
stretch target. No further fit, export, sealed evaluation or HF upload is authorized by this result.

The [fresh v1 experiment](../reports/ETTIN150M_BOUNDED_V1.md) remains a separate, rejected
historical result: 14/18 gates and 13.72 ms desktop p95. The preselected 300-row open comparison
now records v2 macro F1 of 0.835 versus 0.652 for the fixed Qwen3.8-27B Q4 baseline; v2 detects
76/76 versus 65/76 scams, but raises 2/210 versus 0/210 SAFE false alerts. This is not a sealed
comparison, SOTA proof, or release authorization. No model is release-ready.

The existing independent-review packet is bound to the unchanged schema-25 train/dev corpus,
not the historical schema-24 packet. All 635 decisions are still blank. The
[v2 reuse note](ETTIN_V2_AUDIT_REUSE.md) preserves the original packet and bindings: the owner
needs one completed review, not a second 635-row review for repeated training presentations.
Completing it would address sampled supervision/privacy review, not repair model quality,
estimate replay-weighted error, or prove held-out labels and semantic explanations correct.
Any future accepted encoder requires its own final-validation/release contract; the
Qwen-specific verifier cannot validate it unchanged.

Native protocol 5 corrects NFC normalization as well as chat-token parsing. The expanded real-model
probe matches the published tokenizer on 81/81 token sequences, but historical training inputs
remain different. Existing native measurements do not establish the corrected runtime's quality
or latency. See
[`reports/GGUF_PROTOCOL5_TOKENIZATION.md`](../reports/GGUF_PROTOCOL5_TOKENIZATION.md).

The original goal is still incomplete. Stage 7's regression recall is 96.934%, SAFE FPR is 2.005%,
credential-theft recall is 90.278%, and opportunity-scam recall is 94.444%. These fail the original
core requirements even before the additional Qwen-lineage gates are considered. Strong development
scores, completed training, or runnable packages do not establish success on a final held benchmark.

The current Qwen publication validator remains unchanged. It requires that lineage's frozen BF16,
quantization, primary-test, desktop/mobile, audit, and release receipts. None of its rejected
artifacts is newly authorized by this document. The source-of-requirements distinction below must
be used when freezing a future model's contract so that historical implementation choices are not
mistaken for user requirements.

## Requirement provenance and experiment boundary

The supplied PRD sections 7, 8, and 12 require representative human review, core-category scam recall
above 97%, false-positive rate below 2%, same-held-set comparison against larger models, local
laptop inference, a credible mobile path, and reproducible artifacts. The user subsequently asked
for under 20 ms and accepted desktop and/or mobile execution. Both remain goals; no claim that the
current Qwen artifact achieves either is supported.

Several later conditions are project-added experiment decisions:

- The PRD calls macro F1 above 0.94 a **stretch** target. The Qwen gate checker makes it mandatory.
- The PRD asks for a credible mobile path; the Qwen release checker requires physical evidence on
  **both** iOS and Android and both platform packages.
- BANKING77's at-most-three-errors and 0.5% FPR limits are continuation-promotion criteria, stricter
  than the product's general 2% FPR target. Stage 14 meets 2% on that particular slice but fails
  the predeclared continuation and regresses other evidence.
- Exact GGUF verdict parity and the particular Qwen scorer are lineage-specific implementation
  requirements, not an instruction to deploy every future encoder through GGUF.

These distinctions preserve the historical rejection decisions. They prevent adding requirements
after results are observed and prevent using relaxed retrospective criteria to declare a winner.
For the next model, freeze the dataset hashes, inference rule, target hardware, numerical limits,
required versus diagnostic slices, and finite run budget **before** observing its results. Keep
every phone, uncertainty, multilingual, and adversarial weakness in the published report even when
it is diagnostic. Do not aggregate them into an easy headline score.

Ettin-150M's fresh v1 fit and the subsequently authorized single-pass v2 recovery both failed
their prospectively frozen promotion checks. An open baseline can precede completed human
review; a release claim cannot. The unchanged schema-25 blind packet provides the outstanding
supervision/privacy review process, and an AI review cannot certify its own labels. Neither
experiment authorizes automatic next-model search or additional fitting. The Qwen line remains
closed, this Ettin recovery is terminal, and sealed tests remain unopened.
