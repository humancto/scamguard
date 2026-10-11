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
| Human-audited representative slice | **Pending** | Current schema-25 635-row packet and exact return requirements in [`INDEPENDENT_REVIEW_HANDOFF.md`](INDEPENDENT_REVIEW_HANDOFF.md) | Independent humans must complete the blind label/privacy audit, including qualified language assistance. AI-internal review cannot authorize release. |
| Same benchmark for rules, classical, open neural, and larger-model baselines | **Proven for recorded research baselines** | Model ladder and linked run reports in [`README.md`](../README.md); paired-comparison tooling in [`benchmarks/compare_paired.py`](../benchmarks/compare_paired.py) | Rerun any claimed external winner on the same final sealed benchmark before a SOTA claim. |
| Sub-1B serious specialist | **Partial** | Fresh Ettin-150M completed its bounded fit and 21 open slices, but passed only 14/18 promotion checks; see [`ETTIN150M_BOUNDED_V1.md`](../reports/ETTIN150M_BOUNDED_V1.md). Historical Qwen results remain in [`QWEN08_PHONE_STAGE7.md`](../reports/QWEN08_PHONE_STAGE7.md) and [`BANKING77_SOURCE_ADMISSION.md`](../reports/BANKING77_SOURCE_ADMISSION.md). | No tested candidate satisfies the complete quality requirements. The Qwen continuation line and this single-fit Ettin pilot are closed to further fitting. |
| Scam recall at least 97%, SAFE FPR at most 2%, core-category recall at least 97%, macro F1 above 0.94 stretch | **Pending for a final candidate** | Fail-closed definitions in [`scripts/check_qwen08_full_gates.py`](../scripts/check_qwen08_full_gates.py) and [`scripts/check_primary_v8_gates.py`](../scripts/check_primary_v8_gates.py) | Obtain passing frozen regression and untouched primary receipts from the same final artifact. |
| Publish calibration, adversarial/OOD behavior, false-positive analysis, latency, RAM, and artifact size | **Partial** | Existing run reports and required-report contract in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | Repeat and publish the complete evidence bundle for the accepted quantized model. |
| Quantized local artifact | **Partial** | Hash-verified upstream Q4 runtime control and native scorer; final merge/export chain in [`Makefile`](../Makefile) | Merge and Q4 export are forbidden until the trained BF16 challenger passes every gate; then quantized parity and quality must pass again. |
| Normal-laptop local inference | **Partial** | Ettin's actual SDK completed 19,823 scans on M4 Max and measured 13.72 ms warm p95, with the raw 300-call trace in [`sg-ettin150m-bounded-v1.open.json`](../reports/runs/sg-ettin150m-bounded-v1.open.json); historical GGUF controls remain separately documented. | The measured encoder is rejected on quality. A final accepted artifact must retain its own runtime evidence; Q4 is not a universal encoder requirement. |
| Credible mobile path | **Partial** | Shared C ABI, Swift wrapper, Android JNI/Kotlin wrapper, package builders, and correctness smokes in [`docs/MOBILE_RUNTIME_INTEGRATION.md`](MOBILE_RUNTIME_INTEGRATION.md) | Run the final artifact on physical iOS and Android devices using [`docs/MOBILE_BENCHMARK_PROTOCOL.md`](MOBILE_BENCHMARK_PROTOCOL.md). Simulator and desktop evidence cannot substitute. |
| Small model competitive with or better than much larger models | **Pending** | Comparison discipline and same-row paired test contract in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | A final sealed, same-benchmark comparison has not yet established “beats SOTA.” Do not make that claim from different datasets or the open regression split. |
| Reproducible training and evaluation | **Proven for research workflow** | Hash-pinned configs, data manifests, [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md), tests, and fail-closed Make targets | Final release must bind the accepted adapter, merged model, GGUF, calibration, ledgers, runtime packages, and source revision. |
| Hugging Face publication | **Pending** | Release manifest template and verifier in [`configs/huggingface-release-qwen35-08b.template.json`](../configs/huggingface-release-qwen35-08b.template.json) and [`scripts/verify_huggingface_release.py`](../scripts/verify_huggingface_release.py) | Quality, human audit, provenance, quantized parity, physical-mobile, limitations, hashes, and model-card gates must all pass first. |

## Completion rule

The 2026-10-10 [fresh Ettin-150M bounded experiment](../reports/ETTIN150M_BOUNDED_V1.md) completed
all 3,056 fit steps and all 21 declared open slices. It measured 13.72 ms warm full-scanner p95 on
M4 Max, satisfying the experiment's desktop latency limit, but passed only 14/18 quality/runtime
promotion checks. Development recall and three dialogue false-positive rates fail. It is rejected;
no further training, export or sealed evaluation is authorized for that candidate. Its final
comparison status and full diagnostic results are in the linked report. No model is release-ready.

The new independent-review packet is bound to schema-25 train/dev, not the historical schema-24
packet. Completing it would address sampled supervision/privacy review, not repair model quality
or prove held-out labels and semantic explanations correct. Any future accepted encoder requires
its own final-validation/release contract; the Qwen-specific verifier cannot validate it unchanged.

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

The next architecture was named and tested once: Ettin-150M. It failed its prospectively frozen
promotion checks. An open baseline can precede completed human review; a release claim cannot.
The current schema-25 blind packet supplies the requested supervision/privacy review, and an AI
review cannot certify its own labels. No automatic next-model search is authorized by the bounded
experiment. Both the Qwen continuation line and this Ettin pilot are closed to further fitting;
sealed tests remain unopened.
