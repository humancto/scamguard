# Linear baseline — fast, rejected on quality

Post-freeze ScamBench v0.3 run: `sg-linear-v0.3`, native arm64 Python, seed `20260820`. Threshold
`0.7230530` was selected once on development SAFE/SCAM rows under the 2% FPR ceiling. Test and OOD
outcomes were opened only after the 1,000-row forum cap was frozen.

| Slice | Scam recall | FPR | Macro F1 | Interpretation |
|---|---:|---:|---:|---|
| Development | 48.05% | 1.99% | 0.687 | release gate failed |
| Untouched test | 58.09% | 2/1,746 (0.115%) | 0.815 | release gate failed |
| Financial OOD | 59.39% | 27.91% | 0.385 | noisy-domain failure |
| WSPR OOD (no SAFE rows) | 96.09% | n/a | 0.627 | campaign recall diagnostic |
| Forum validation | 91.90% | 1/25 (4.00%) | 0.704 | selection-only, SAFE denominator is too small |
| Forum OOD | 91.80% | 6/100 (6.00%) | 0.691 | unseen forum families plus SAFE controls |
| Materialized forum OOD | 79.65% | 5/78 (6.41%) | 0.521 | placeholder shortcut exposed |
| Adversarial derivatives | 52.50% | 0% | 0.582 | robustness gate failed |

The baseline meets the desktop speed goal decisively: 0.67 ms median and 1.16 ms p95 for batch-one
prediction across 250 messages. Its compressed artifact is 2,682,613 bytes. Those properties make
it a useful fast-path floor, but not the product detector: it misses 246 of 587 untouched test scams
at the safety-constrained threshold.

The text-free prediction ledger contains 11,082 scored binary examples for paired comparisons and
error auditing. Hardware-specific reports remain under ignored `reports/runs/`.
