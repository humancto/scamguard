# Qwen uncertainty and alert-composition diagnosis

The closed Qwen line has a measurable decision-policy limitation in addition to its model-quality
failures. This diagnosis uses only existing, hash-bound development predictions. It changes no
labels, prompts, probabilities, thresholds, weights, selection results, or publication decisions.

Stage 7 and Stage 14 make the same number of SCAM decisions on development: 581 alerts, containing
499 reference SCAM examples, 77 reference UNCERTAIN examples, and 5 reference SAFE examples.
Their binary precision is 499/504 (99.01%) because that metric excludes UNCERTAIN references.
Reference SCAM examples constitute 499/581 (85.89%) of all alerts. The remaining ambiguous examples
must not be silently counted as false positives or confirmed scams; their labels do not establish
either conclusion.

There are 112 reference UNCERTAIN examples in this development set. The frozen recall-first policy
flags 77 of them, or 68.75%. This explains why the binary safety numbers can look strong while
three-way macro F1 remains 0.7856. The evaluator now reports the reference-label composition of all
alerts beside its calibrated confusion matrix, including in GGUF evaluations using the shared
slice evaluator. Old frozen reports remain unchanged.

## Fixed-SCAM diagnostic ceiling

An oracle can assign every remaining true SAFE or UNCERTAIN example correctly, then choose the
best SAFE/UNCERTAIN allocation for the missed SCAM examples. With the SCAM prediction set fixed,
this bounds every possible SAFE-threshold adjustment from above. For both development ledgers the
bound is **0.794213 macro F1**, below 0.94. It is a label-aware mathematical bound, not a deployable
decision policy or achieved model score. It says nothing about the best attainable result after
changing model representations or the SCAM decision set.

This rules out further SAFE-threshold tuning as a solution for these frozen scores. The source
breakdown also matters: of Stage 14's 72 synthetic reference-UNCERTAIN examples, raw argmax already
calls 50 SCAM; the calibrated policy raises that to 67. Weak class ranking and the recall-first
decision rule both contribute. A better SAFE cutoff alone cannot repair either mechanism.

## Label validity remains unresolved

The existing 635-row AI review agreed with only 37/96 reference UNCERTAIN labels and 90/132 SCAM
labels. These are disagreement diagnostics from an internal AI review, not a measured human error
rate and not proof that the model or reviewer is correct. The development counts above use the
original reference labels without modifying them. The result motivates adjudicating the label
boundary before using these scores to justify another student-training campaign.

The independent blind-review ZIP was reverified on 2026-10-04: all 635 decisions are still blank,
its isolated reviewer loads, and disposable save/resume checks pass without modifying the source
ZIP. It remains `dist/scamguard-schema24-blind-audit.zip`, SHA-256
`dd0c99d97894ace0e3868f947b08b5eab6ccfb6d03b7930e726c91ec1bbf6691`.
The current handoff-preflight report is
`reports/runs/schema24-blind-handoff-2026-10-04.json`, SHA-256
`cd4db519e14f018e0538c3bc35f291d6d7e46567b05689d542dc59db36cb42d5`.

The original product requirements and added Qwen experiment conditions are distinguished in
[`docs/PRD_TRACEABILITY.md`](../docs/PRD_TRACEABILITY.md#requirement-provenance-and-experiment-boundary).
Neither this diagnosis nor that clarification promotes a rejected adapter. The core-category
quality requirement, final held-set comparison, human review, and target-runtime evidence remain
incomplete.

## Reproduce the diagnostic

The command verifies the supplied report/ledger binding and reconstructs development decisions
before computing the ceiling. It rejects malformed or inconsistent probabilities, duplicate IDs,
confusion/count drift, and input overwrites. Existing diagnostic outputs require explicit
`--replace`. It performs no model inference and ignores non-development records.

```bash
.venv/bin/python scripts/audit_verdict_tradeoff.py \
  --report reports/runs/qwen35-08b-phone-generalization-stage7-dev.json \
  --predictions reports/runs/qwen35-08b-phone-generalization-stage7-dev.predictions.jsonl \
  --output reports/runs/qwen35-08b-phone-generalization-stage7-dev-verdict-tradeoff.json

.venv/bin/python scripts/audit_verdict_tradeoff.py \
  --report reports/runs/qwen35-08b-banking-boundary-stage14-low-lr-dev.json \
  --predictions reports/runs/qwen35-08b-banking-boundary-stage14-low-lr-dev.predictions.jsonl \
  --output reports/runs/qwen35-08b-banking-boundary-stage14-low-lr-dev-verdict-tradeoff.json
```

The Stage 7 diagnostic SHA-256 is
`ac17822aa6a722eab2a9728594e2979a926c2891f559952646ea94e57316f15f`.
The Stage 14 diagnostic SHA-256 is
`69adea41c0a25832cb3f3616a5844a8cd5c1fb15ee91258890918cb604af567b`.
Each receipt records the exact source-report and prediction-ledger hashes. A synthetic exhaustive
test compares the oracle formula with every possible SAFE/UNCERTAIN assignment on a small fixture;
it does not inspect an additional benchmark split.
