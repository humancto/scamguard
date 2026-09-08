# Qwen3.5-0.8B Stage 7 phone-generalization result

## Decision

**Rejected. Do not merge, quantize, package, open the sealed phone test, or publish this
checkpoint.**

Stage 7 is a useful negative result. Replaying the complete Stage 3 curriculum once with 1,052
rights-clear phone-training examples improved development precision and the previously opened
publisher phone diagnostic. It did not meet the frozen general-regression, category, dialogue, or
phone gates. The fail-closed checker passed 33 of 41 gates and correctly denied quantization and
Hugging Face publication.

The publisher phone validation split remains a confounded, previously opened development
diagnostic. Its upstream `variation_style` metadata predicts the label perfectly; that metadata is
not provided to the model. The untouched 172-row publisher phone test remains prediction-sealed.

## Frozen run

- Base: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Initialization: Stage 3 adapter weights SHA-256
  `403c2e176997311396780a79b890654e8165c2a996e77d2fbcf0b7c41ac6ce9b`.
- Curriculum: all 25,098 Stage 3 fitting rows plus 1,052 admitted phone rows; 26,150 total.
  There are zero held-reference overlaps and zero examples above the 640-token contract.
- Optimization: one epoch, batch 4, gradient accumulation 4, learning rate `1e-6`, seed
  `20260906`, 10,822,656 trainable parameters.
- Observed training: 1,635 optimizer steps, 14,721.5 seconds on Apple MPS, final reported training
  loss `0.01011764`.
- Output: 60 MiB adapter directory; `adapter_model.safetensors` is 41.3 MB.

## Fair Stage 3 comparison

Both adapters were scored with the same schema-25 data, branch-token scorer, development-fitted
thresholds, and unchanged phone validation rows. No phone-validation label selected the threshold.

| Split and metric | Stage 3 | Stage 7 | Frozen Stage 7 gate |
|---|---:|---:|---:|
| Development scam recall | 97.08% (499/514) | 97.08% (499/514) | >=97% |
| Development SAFE FPR | 0.60% (12/2,008) | 0.25% (5/2,008) | <=2% |
| Development macro F1 | 0.7116 | 0.7856 | diagnostic selection metric |
| Regression scam recall | 99.83% (586/587) | 96.93% (569/587) | >=97% |
| Regression SAFE FPR | 4.07% (71/1,746) | 2.00% (35/1,746) | <=2% |
| Regression macro F1 | 0.7370 | 0.7778 | >=0.94 |
| Phone scam recall | 77.01% (134/174) | 78.16% (136/174) | >=90% |
| Phone SAFE FPR | 20.59% (35/170) | 15.88% (27/170) | <=10% |
| Phone macro F1 | 0.5221 | 0.5414 | diagnostic only |

Stage 7 improved phone recall by only 1.15 percentage points and reduced phone SAFE FPR by 4.71
points. The residual phone errors are still material: 38 missed scams and 27 false alarms in 344
examples. On the unchanged regression it also narrowly missed the aggregate recall/FPR limits,
missed `CREDENTIAL_THEFT` recall at 90.28%, missed `OPPORTUNITY` recall at 94.44%, and reached only
62.41% scam recall on the prior-open BothBosu dialogue slice.

## Score-interpolation diagnostic

A strict ID-and-metadata join compared 202 arithmetic and log-linear Stage 3/Stage 7 score blends.
Only development labels selected the blend and thresholds; test and phone labels were reporting
only. The contract-first selector chose the Stage 7 endpoint (`right_weight = 1.0`). The
macro-F1-ranked alternative used 95% Stage 7 but merely tied the development result and produced no
better held phone result. Two-adapter inference therefore adds no demonstrated value and is not a
mobile candidate.

This is post-hoc diagnostic evidence, not fresh confirmation. It should inform a new single-adapter
curriculum or optimization experiment; it cannot authorize weight merging or publication.

## Failed gates

The eight failed gates were:

1. regression scam recall: 96.93%, required >=97%;
2. regression SAFE FPR: 2.0046%, required <=2%;
3. regression calibrated macro F1: 0.7778, required >=0.94;
4. regression `CREDENTIAL_THEFT` recall: 90.28%, required >=97%;
5. regression `OPPORTUNITY` recall: 94.44%, required >=97%;
6. prior-open BothBosu scam recall: 62.41%, required >=97%;
7. phone validation scam recall: 78.16%, required >=90%; and
8. phone validation SAFE FPR: 15.88%, required <=10%.

## Evidence identities

- Adapter weights SHA-256:
  `14f1d2bf121e76fa158cea722416994ec9cbfbc545956d24b9693b7808441357`.
- Adapter config SHA-256:
  `27e4683d15e5335ffa0cb6e11640f0812a29b324a4940adcc3634ec86e2c019b`.
- Training receipt SHA-256:
  `7dfec17cd6322cae2b94350b312dd8b6efb55f05dc41a1da170e766d922178fd`.
- Full regression report SHA-256:
  `5d9b1b30e22c83e07d7a19a1b71ebbb3c6c07462a1c0d5a1ffb8ba1485ab4cbb`.
- Stage 7 prediction ledger SHA-256:
  `4b6f1b79a63ad4a6e625b8e157d072c0687a1b4c0c8afda881eac19809595821`.
- Gate report SHA-256:
  `48a2e7db6f0bd1abfa4f75d7a4cbdfeddb1ddda66a826e7b60bd4d5bc2583c22`.
- Fair Stage 3 phone-baseline report SHA-256:
  `0ee191d627c528dee9127cfb69b29128d8851b4968d30962dcedc8d73b70436e`.
- Stage 3 prediction ledger SHA-256:
  `2589ba395908f36e987f844f429b56d89ff2bcbbbb9fb9ef62d40852bc229c36`.
- Score-blend report SHA-256:
  `08cf56379cc49a710e2f39cd220881e17276f3b0b71c79bb07a86de914bf2e27`.

## Next experiment boundary

Do not repeat another full-replay epoch unchanged. First localize Stage 3-to-Stage 7 phone error
transitions by dialogue style, length, source batch, and evidence-bearing request type. The next
single-adapter run should use those results to construct family-separated, evidence-matched
positive/negative pairs while preserving explicit Stage 3 retention slices. It must earn a
development-only promotion before reopening the full regression. Quantization, physical-device
latency, the sealed primary benchmark, independent human review, and Hugging Face publication all
remain downstream gates.
