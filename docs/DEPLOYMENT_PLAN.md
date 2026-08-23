# Desktop and mobile deployment plan

ScamGuard does not call a model “mobile-ready” from parameter count or desktop timing alone. The
schema-v13 149M encoder plus separately versioned trust-channel policy is the current fast-path
research candidate. Tokenizer-to-probability latency measured 14.01 ms p95 on desktop MPS. The
variable-sequence FP32 ONNX export reproduces every frozen binary verdict across 5,008 open rows and
measures 13.92 ms p95 on four CPU threads; signal extraction plus policy measured 0.11 ms p95 in a
separate run. Direct Core ML FP32 also preserves every frozen binary verdict and measures 5.65 ms
p95 end-to-end on the reference Mac; its complete pack is 602.5 MB. Core ML FP16 measures 3.83 ms
p95 but changes 19 decisions, so it is rejected. None is a phone result, and the candidate still
fails the open dialogue and macro-F1 gates. Its self-contained FP32 ONNX pack is 602.7 MB. The
upstream ModernBERT-base repository
publishes an approximately 151 MB INT8 ONNX
variant, which is useful sizing evidence—not proof that the ScamGuard checkpoint will have the same
size, accuracy, operator coverage, or latency.

## Artifact ladder

1. Export the selected encoder with fixed 128- and 256-token shapes plus a dynamic-shape control.
2. Verify FP32 ONNX logits and verdicts against the frozen PyTorch checkpoint on development and
   every already-open diagnostic.
3. Produce INT8 dynamic and static-calibrated candidates. ONNX Runtime recommends dynamic
   quantization for transformers, but the winning method is selected by measured quality and
   hardware latency, not the generic recommendation.
4. Run ONNX Runtime's mobile-usability checker and record Core ML/NNAPI partition coverage. A model
   that repeatedly crosses between accelerator and CPU partitions is not accepted on file size
   alone.
5. Convert the best Apple candidate to a Core ML `mlprogram` with enumerated/static shapes and test
   FP16, weight-only INT8, and supported weight-plus-activation INT8 paths.
6. Package tokenizer assets, calibration, labels, and model hash as one versioned model pack.

For the Qwen escalation path, `make qwen-08b-base-runtime-pack` exercises that final packaging
shape today. The pack embeds the exact Qwen3.5-0.8B prompt prefix/suffix instead of importing
Transformers at runtime, statically links llama.cpp/ggml into one arm64 executable, and binds the
GGUF, executable, calibration, platform, machine, prompt, and runtime settings by SHA-256. The
public `Scanner` API keeps that process and its 141-token prefix cache alive across calls. The
upstream control pack measures 39.60 ms p95 across 150 Scanner requests on the reference Mac. This
is desktop deployment evidence only; its manifest is non-publishable and carries no trained-model
quality claim.

Primary references:

- [ModernBERT-base ONNX artifacts](https://huggingface.co/answerdotai/ModernBERT-base/tree/main/onnx)
- [ONNX Runtime mobile](https://onnxruntime.ai/docs/get-started/with-mobile.html)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime mobile usability checker](https://onnxruntime.ai/docs/tutorials/mobile/helpers/model-usability-checker.html)
- [Core ML conversion formats](https://apple.github.io/coremltools/docs-guides/source/target-conversion-formats.html)
- [Core ML input and output types](https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html)
- [Core ML optimization and quantization API](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html)

## Quantization acceptance

Quantization creates a new candidate. It must use only development data for any calibration or
threshold selection and then pass the same frozen quality gates as the source checkpoint. Reports
must include:

- exact artifact and tokenizer hashes;
- artifact bytes and peak resident memory;
- maximum/mean logit error and verdict agreement versus FP32;
- recall, false-positive rate, macro-F1, Brier score, ECE, and category slices;
- cold-start, warm p50/p95, and sustained p95 for tokenization, inference, post-processing, and the
  full verdict path separately;
- thermal state, OS/runtime version, device model, thread count, sequence length, and sample count.

A quantized artifact is rejected if it crosses the 2% false-positive ceiling, loses a required core
category, or silently changes the abstention policy. A smaller file is not a quality trade.

The first schema-v13 dynamic INT8 attempt is rejected under this rule. It reduces the complete pack
to 271.0 MB and measures 13.87 ms p95, but changes 27 development and 17 regression frozen binary
verdicts, crosses the raw-model development FPR ceiling at 2.34%, and degrades category/macro-F1
metrics. The FP32 dynamic graph is the current fidelity reference. See
[`reports/ONNX_SCHEMA13.md`](../reports/ONNX_SCHEMA13.md).

The direct Core ML conversion reaches the opposite size/quality boundary. FP32 is accepted as an
Apple desktop fidelity artifact: 100% frozen binary-verdict parity and 5.65 ms p95 across the full
open evaluation. FP16 cuts the package roughly in half and reaches 3.83 ms p95, but changes 17
development and two regression decisions and reduces development scam recall from 95.91% to
93.19%. It is rejected. See [`reports/COREML_SCHEMA13.md`](../reports/COREML_SCHEMA13.md).

## Hardware gate

The release matrix needs at least one recent iPhone and one representative Android device. Apple
testing compares Core ML ML Program and ONNX Runtime's CPU/CoreML execution providers; Android
testing compares CPU/XNNPACK and NNAPI where supported. The ONNX Runtime checker is only a routing
hint; its own documentation requires performance testing.

The product target remains p95 at or below 20 ms for the complete local fast verdict on the declared
device class. The original PRD's sub-50-ms laptop requirement is a separate, less strict gate. If a
quality-preserving Qwen fallback is used, publish its escalation rate and full routed latency; do
not relabel the encoder's 20-ms number as end-to-end latency.

## Release boundary

No mobile claim is allowed until a physical-device run records the model pack, binary/runtime,
hardware, benchmark corpus hashes, and raw timing samples. Simulator, desktop MPS, and upstream
artifact sizes establish feasibility only.
