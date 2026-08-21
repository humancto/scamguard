# Schema-v13 Core ML deployment experiment

Run date: 2026-08-21. FP32 decision: **accepted as the Apple desktop fidelity artifact**. FP16
decision: **rejected as a release artifact**. The schema-v13 classifier remains a research candidate
because its original dialogue, raw regression-FPR, and macro-F1 gates still fail.

## Conversion contract

The exporter uses the stable TorchScript-trace path into an iOS 17 ML Program with batch one and a
fixed 128-token input. It replaces two trace-hostile implementation details with verified-equivalent
fixed-shape expressions: attention masks use a representable `-10000` softmax bias, and the
64-dimensional rotary head uses a fixed half-slice. Four varied PyTorch probes show zero logit error
for the mask wrapper; the rotary rewrite and TorchScript trace also show zero error.

The runtime pack contains the Core ML package, tokenizer, calibration, and manifest. Every source
checkpoint file and runtime asset is SHA-256 bound. One development example has 139 tokens and is
truncated at 128; no unchanged-regression example is truncated.

| Artifact | Model package bytes | Complete pack bytes | Export probe max logit error | Decision |
|---|---:|---:|---:|---|
| Core ML FP32 | 598,917,520 | 602,503,913 | 0.00000572 | fidelity pass |
| Core ML FP16 | 299,634,858 | 303,221,250 | 0.06456 | reject pending better compression |

An earlier FP16 attempt used the float32 minimum as the attention-mask sentinel. It overflowed while
lowering to FP16 and produced 7.06 maximum logit error. That artifact was rejected; `-10000` removed
the overflow from the attention mask and reduced the probe error to 0.06456, but full-set drift still
fails the quality-first rule.

## Full open-set fidelity

Both packages were run through Core ML with `ComputeUnit.ALL` on all 2,634 development and 2,374
unchanged-regression rows at the original temperature and threshold. No sealed data was opened.

| Candidate | Dev verdict parity | Regression verdict parity | Dev recall / FPR | Regression recall / FPR |
|---|---:|---:|---:|---:|
| PyTorch reference | reference | reference | 95.91% / 1.94% | 99.32% / 4.18% |
| Core ML FP32 | **100%** | **100%** | 95.91% / 1.94% | 99.32% / 4.18% |
| Core ML FP16 | 99.35% (17 changes) | 99.92% (2 changes) | 93.19% / 1.99% | 99.32% / 4.07% |

FP32 also preserves 100% three-class argmax agreement. Its maximum calibrated SCAM-probability
error is 0.003685 on development, caused by the single truncated 139-token example, and 0.00000721
on unchanged regression. FP16 changes 19 frozen decisions and loses 14 additional development scam
detections. It is rejected despite being smaller and faster.

## Latency and memory

The reference machine reports model identifier `Mac16,5`, 16 CPU cores, 128 GB memory, and macOS
26.3. Measurements use Core ML 9.0, `ComputeUnit.ALL`, batch one, fixed 128-token inputs, eight warmup
calls, and 250 unchanged-regression messages. End-to-end timing includes speaker-neutral
preprocessing, tokenization, Core ML prediction, and the probability transform.

| Candidate | Median | p95 | Whole-process peak RSS | Result |
|---|---:|---:|---:|---|
| Core ML FP32 | 5.48 ms | **5.65 ms** | 1,817,657,344 bytes | under-20-ms fidelity pass |
| Core ML FP16 | 3.60 ms | **3.83 ms** | 1,275,772,928 bytes | latency pass, quality reject |

Peak RSS is the entire Python evaluator, including Core ML tooling and tokenizer; it is not isolated
model memory. The timing proves a native Core ML path on the reference Mac, not phone latency.
Physical iPhone/iPad testing must record device, thermal state, compiled-model size, cold start,
sustained latency, and peak memory before any mobile-ready claim.

## ONNX mobile-routing diagnostic

ONNX Runtime's mobile-usability checker does not recommend NNAPI or either Core ML execution
provider for the exported dynamic graphs. The FP32 graph exposes only 26.0% of nodes to NNAPI,
20.2% to the Core ML NeuralNetwork provider, and 16.0% to Core ML ML Program; hypothetical fixed
shapes raise supported-node counts but leave 90-137 partitions. The INT8 graph is similarly
fragmented and adds dynamic-quantization operators. CPU EP is the checker's recommendation.

This is why the accepted Apple experiment uses direct PyTorch-to-Core ML conversion rather than an
ONNX/Core ML provider route. The checker is a compatibility heuristic, not a device benchmark.

## Immutable evidence

| Evidence | SHA-256 |
|---|---|
| FP32 export manifest | `6280941eec96417c51e26d6f45383fd575894d117217df1e6ca2e5f2e06e9b78` |
| FP32 full evaluation | `2985e9649805574cc242967fa6f438d441b04c740e2766c55ac05f0b665ea45c` |
| FP16 export manifest | `44d723c94abdad6cd60f7d18d64f3d570e880c074e00feef9cb1a619018410ea` |
| FP16 full evaluation | `ef93eb92635ec0beff3e0d51deae97e95877fe0a94759a7d953f91d51b655cfd` |

The next size experiment should use accuracy-aware palettization, quantization-aware training, or a
distilled smaller encoder and must rerun the same frozen per-example parity and quality checks.
