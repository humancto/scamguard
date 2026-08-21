# Encoder comparison — schema v6

Both encoders were rerun on the privacy-normalized schema-v6 rows. The schema-v5 ModernBERT run
was interrupted and rejected before completion. DeBERTa's CC-BY-NC license makes it a research
reference rather than a redistributable product dependency.

| Candidate and threshold | Dev recall / FPR | Test recall / FPR | Financial OOD recall / FPR | Forum OOD recall / FPR | Adversarial recall / FPR |
|---|---:|---:|---:|---:|---:|
| ScamGuard ModernBERT, dev-fitted `0.294633` | 85.02% / 1.74% | 92.67% / 5.10% | 66.28% / 32.56% | 98.70% / 14.00% | 79.38% / 2.50% |
| Public DeBERTa, ScamBench dev-fitted `0.9187091` | 11.48% / 1.89% | 8.18% / 1.32% | 8.43% / 0.00% | 12.20% / 6.00% | 7.50% / 0.00% |
| Public DeBERTa, its published `0.7229` | 98.64% / 49.95% | 68.82% / 60.42% | 65.52% / 30.23% | 81.20% / 72.00% | 53.12% / 17.50% |

ModernBERT reaches 92.67% test recall and an 0.854 three-class test macro-F1, but misses the frozen
97% recall and 2% FPR safety gates. Its Apple-MPS batch-one forward pass measured 12.42 ms median
and 19.06 ms p95 after tokenization, excluding tokenization itself. The isolated DeBERTa CPU run
measured 10.39 ms median and 36.94 ms p95 per message at batch 32; that is not batch-one product
latency. The comparison demonstrates why the threshold and latency scope must travel with every
performance claim.

The ModernBERT report initially joined length-grouped logits to original-order rows and was
rejected. The corrected evaluator forces sequential prediction and asserts the returned labels
match source-row order before producing any source or category slice.

Machine-readable local reports are generated in `reports/runs/` and ignored because they include
hardware-specific runtime measurements.
