# Schema-v13 ONNX deployment experiment

Run date: 2026-08-21. FP32 export decision: **accepted for desktop fidelity testing**. Dynamic
INT8 decision: **rejected as the release artifact**. The underlying schema-v13 model and policy
remain research candidates because dialogue and macro-F1 gates still fail.

## Export contract

The exporter freezes batch one, permits a variable sequence dimension up to the training-time
256-token cap, forces eager attention for traceability, validates the graph with ONNX's checker,
loads both outputs in ONNX Runtime, and binds every runtime file to SHA-256 in a self-contained
manifest. The SDK's ONNX backend rechecks the selected graph hash before loading it and never falls
back to a network model.

| Artifact | Model bytes | Complete pack bytes | SHA-256 |
|---|---:|---:|---|
| dynamic FP32 | 599,075,489 | 602,662,339 | `77efc6b5703bae49578d453d8eb5649fab185c7de4974be6547a04d11011e3df` |
| dynamic INT8 | 267,432,599 | 271,019,449 | `0308a8b04bf104baa191bb283f7df7d112944b66a871ce93d42aff091d3ee0c9` |

The complete pack count includes the selected graph, tokenizer, calibration, and common manifest.
The INT8 graph is 55.4% smaller than FP32. It uses per-channel QInt8 dynamic quantization for
`MatMul` and `Gemm`; it is not a blanket claim that every operator or activation is eight-bit.

## Full open-set parity

Both graphs were scored batch-one through ONNX Runtime 1.29.0 on all 2,634 development and 2,374
unchanged-regression rows using the original schema-v13 temperature and threshold. No sealed source
was read.

| Candidate | Dev verdict parity | Regression verdict parity | Dev recall / FPR | Regression recall / FPR | Decision |
|---|---:|---:|---:|---:|---|
| PyTorch/MPS reference | reference | reference | 95.91% / 1.94% | 99.32% / 4.18% | model-only rejected |
| ONNX FP32 dynamic | **100%** | **100%** | 95.91% / 1.94% | 99.32% / 4.18% | exact deployment-fidelity pass |
| ONNX INT8 dynamic | 98.97% | 99.28% | 98.83% / 2.34% | 98.98% / 4.24% | reject: quality shifts |

FP32 maximum absolute calibrated SCAM-probability error is 0.0000114 on development and 0.0000109
on regression, with 100% argmax agreement. INT8 changes 27 development and 17 regression frozen
binary verdicts, misses two additional regression scams, crosses the raw-model 2% development FPR
ceiling, reduces regression macro F1 from 0.8817 to 0.8686, and reduces the nine-row development
UNKNOWN-family recall from 88.89% to 77.78%. Its policy-adjusted open binary metrics remain above
the minimum gates, but that post-hoc policy cannot excuse a measurable quantization regression.

## Latency and memory

Measurements use batch one, four ONNX CPU intra-op threads, one inter-op thread, 250 unchanged
regression messages, graph optimizations enabled, and natural variable sequence lengths. They
include speaker-neutral preprocessing, tokenization, ONNX inference, and the probability transform.

| Candidate | Median | p95 | Whole-process peak RSS | Result |
|---|---:|---:|---:|---|
| ONNX FP32 dynamic | 7.07 ms | **13.92 ms** | 1,512,833,024 bytes | desktop under-20-ms pass |
| ONNX INT8 dynamic | 6.44 ms | **13.87 ms** | 1,342,734,336 bytes | latency pass, quality reject |
| ONNX FP32 static 128 | 24.76 ms | 34.73 ms | 1,580,990,464 bytes | reject padding cost |
| ONNX INT8 static 128 | 25.79 ms | 26.48 ms | 1,369,112,576 bytes | reject padding and quality |

Peak RSS is the full Python evaluator process, including tokenizer and runtime initialization; it is
not the model's isolated resident set and not a phone measurement. The dynamic result proves that
the under-20-ms desktop path is portable to ONNX CPU on the reference Mac. It does not prove mobile
latency, Core ML/NNAPI partitioning, or acceptable memory on a physical device.

## Runnable SDK proof

After export, the normal CLI accepts the hash-bound `.onnx` file directly:

```bash
uv run --extra onnx --extra neural scamguard scan \
  --model artifacts/onnx/schema13-dynamic-pack/scamguard-modernbert-seqdynamic-fp32.onnx \
  "Urgent: share the verification code now so we can stop the wire transfer."
```

The exercised result was `SCAM`, category `CREDENTIAL_MFA`, with three exact evidence spans and
`DO_NOT_SHARE_CODE`. This is a functional local inference proof, not an independent quality score.

## Immutable evidence

| Evidence | SHA-256 |
|---|---|
| export manifest | `19fa8fe091e556fd6ed463344668ece7b3b66b127980eb0b8b1dab3c2223247e` |
| FP32 full evaluation | `667033676158b8f3c5a82b7105165af2686fa598c94227e0722750253a1d0c3c` |
| INT8 full evaluation | `60a645560b826d083a4bc1daac4a19df0b0d5fd1f30876a35b28841207b70e9d` |

The next footprint experiment should use quantization-aware training or a smaller distilled
student. Repeating naive post-training dynamic INT8 with a newly tuned threshold would hide rather
than remove the observed per-example probability shifts.
