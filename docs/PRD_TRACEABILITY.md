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
| Appropriately licensed real data plus controlled synthetic data | **Proven for the research corpus** | [`reports/ONLINE_SOURCE_RESEARCH.md`](../reports/ONLINE_SOURCE_RESEARCH.md), [`docs/DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md), and the dataset manifests/builders | Independent label/privacy review is still required for release. Raw Reddit posts remain excluded because public visibility is not a training or redistribution grant; the admitted forum-derived research artifact has explicit release terms. |
| Hard negatives, adversarial positives, multilingual variation, deduplication, and family isolation | **Proven for the benchmark/data factory** | [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md), [`docs/DATASET_SIZE_DECISION.md`](DATASET_SIZE_DECISION.md), validators, overlap audits, and frozen source-family splits | Final model must pass the untouched gates; data engineering alone does not prove model quality. |
| Human-audited representative slice | **Pending** | Frozen 635-row blind handoff and verifier documented in [`docs/HUGGING_FACE_RELEASE.md`](HUGGING_FACE_RELEASE.md) | An independent human must complete the blind label/privacy audit. AI-internal review cannot authorize release. |
| Same benchmark for rules, classical, open neural, and larger-model baselines | **Proven for recorded research baselines** | Model ladder and linked run reports in [`README.md`](../README.md); paired-comparison tooling in [`benchmarks/compare_paired.py`](../benchmarks/compare_paired.py) | Rerun any claimed external winner on the same final sealed benchmark before a SOTA claim. |
| Sub-1B serious specialist | **Partial** | Frozen Qwen3.5-0.8B base revision; rejected 35/39 stage-2 and 36/39 stage-3 results; stage-4 data/config targets in the [`Makefile`](../Makefile) | Stage 3 cut complete-call SAFE FPR to 1.12% and long-call FPR to 0.22%, but held-test FPR, macro F1, and BothBosu recall still fail. Stage 4 must first clear BF16 gates; only then may quantized and sealed-primary evaluation begin. |
| Scam recall at least 97%, SAFE FPR at most 2%, core-category recall at least 97%, macro F1 above 0.94 stretch | **Pending for a final candidate** | Fail-closed definitions in [`scripts/check_qwen08_full_gates.py`](../scripts/check_qwen08_full_gates.py) and [`scripts/check_primary_v8_gates.py`](../scripts/check_primary_v8_gates.py) | Obtain passing frozen regression and untouched primary receipts from the same final artifact. |
| Publish calibration, adversarial/OOD behavior, false-positive analysis, latency, RAM, and artifact size | **Partial** | Existing run reports and required-report contract in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | Repeat and publish the complete evidence bundle for the accepted quantized model. |
| Quantized local artifact | **Partial** | Hash-verified upstream Q4 runtime control and native scorer; final merge/export chain in [`Makefile`](../Makefile) | Merge and Q4 export are forbidden until the trained BF16 challenger passes every gate; then quantized parity and quality must pass again. |
| Normal-laptop local inference | **Partial** | CLI/SDK/runtime pack and measured upstream control in [`reports/QWEN08_Q4_RUNTIME_FLOOR.md`](../reports/QWEN08_Q4_RUNTIME_FLOOR.md) | Benchmark the final trained Q4 artifact without concurrent training and retain the complete raw trace. |
| Credible mobile path | **Partial** | Shared C ABI, Swift wrapper, Android JNI/Kotlin wrapper, package builders, and correctness smokes in [`docs/MOBILE_RUNTIME_INTEGRATION.md`](MOBILE_RUNTIME_INTEGRATION.md) | Run the final artifact on physical iOS and Android devices using [`docs/MOBILE_BENCHMARK_PROTOCOL.md`](MOBILE_BENCHMARK_PROTOCOL.md). Simulator and desktop evidence cannot substitute. |
| Small model competitive with or better than much larger models | **Pending** | Comparison discipline and same-row paired test contract in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | A final sealed, same-benchmark comparison has not yet established “beats SOTA.” Do not make that claim from different datasets or the open regression split. |
| Reproducible training and evaluation | **Proven for research workflow** | Hash-pinned configs, data manifests, [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md), tests, and fail-closed Make targets | Final release must bind the accepted adapter, merged model, GGUF, calibration, ledgers, runtime packages, and source revision. |
| Hugging Face publication | **Pending** | Release manifest template and verifier in [`configs/huggingface-release-qwen35-08b.template.json`](../configs/huggingface-release-qwen35-08b.template.json) and [`scripts/verify_huggingface_release.py`](../scripts/verify_huggingface_release.py) | Quality, human audit, provenance, quantized parity, physical-mobile, limitations, hashes, and model-card gates must all pass first. |

## Completion rule

ScamGuard is not complete merely because a LoRA adapter trains, a GGUF loads, or an open regression
score looks strong. Completion requires the same frozen sub-1B artifact to pass the BF16 regression,
native quantized parity and quality, the prediction-sealed primary test, laptop and physical-mobile
measurement, independent human audit, and the release verifier. Only then may the project claim a
validated release or publish the model to Hugging Face.
