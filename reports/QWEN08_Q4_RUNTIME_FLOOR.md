# Qwen3.5-0.8B upstream Q4 runtime floor

## Decision

Qwen3.5-0.8B is small enough to ship as an optional local specialist, but it is not the under-20-ms
fast path. The verified upstream Q4_0 control occupies 563,036,064 bytes. On a 40-GPU-core M4 Max,
192-token prompt processing alone measures 33.26 ms p95. The exact three-candidate ScamGuard
verdict scorer measures a 50.24 ms p95 of ten run means at a short-text 256-token context and
53.98 ms at the quality-preserving 640-token context. It therefore fails the strict fast-path
target and narrowly misses the PRD's formal `<50 ms` laptop target in the measured scorer. A newer
persistent native runner now supplies the stronger measurement the aggregate scorer could not:
across 50 individually timed requests it measures 59.65 ms p95 with four threads, so the upstream
base definitively fails the escalated-path latency gate as well.

This freezes the deployment role, not the final model: a fast encoder/rule router owns the common
path, and a trained 0.8B Qwen specialist may handle only the uncertainty band if the complete routed
system passes quality, escalation-rate, memory, and latency gates. Direct Qwen remains a desktop
option, not the default mobile path. No quality claim transfers from this upstream base to the
future fine-tuned and quantized artifact.

## Pinned artifact and runtime

- Upstream repository: `ggml-org/Qwen3.5-0.8B-GGUF`
- Repository revision: `8fea620810c4afa23dd6443f999a48574c1611a3`
- File: `Qwen3.5-0.8B-Q4_0.gguf`
- License: Apache-2.0
- File bytes: `563036064` (537.0 MiB)
- File SHA-256: `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`
- llama.cpp revision: `521a64cd01979bb5b1a466152c576a9d809b068d`
- Native runtime: arm64, Metal + Accelerate, Release build, HTTPS disabled
- Host: Apple M4 Max, 40 GPU cores, 128 GiB unified memory, macOS 26.3

The source checkout has exactly the tracked ScamGuard multiple-choice patch applied and therefore
appears dirty. `git apply --reverse --check` passes against
`scripts/llama_cpp_multiple_choice_scores.patch`; its SHA-256 is
`00c85d37cd84efae576efaf32bc2a4c4a3e20927aeebf1876cd4ff950c6efdfc`.

## Generic llama-bench floor

Each row contains 20 warmed repetitions. These are model-kernel floors, not complete
tokenizer-to-verdict latency.

| Backend | Workload | Mean | p95 | Throughput |
|---|---:|---:|---:|---:|
| Metal | 32-token prompt | 11.30 ms | 12.06 ms | 2,834 tok/s |
| Metal | 128-token prompt | 20.88 ms | 21.39 ms | 6,132 tok/s |
| Metal | 192-token prompt | 30.02 ms | 33.26 ms | 6,411 tok/s |
| Metal | 256-token prompt | 35.08 ms | 36.86 ms | 7,303 tok/s |
| Metal | 1 generated token | 4.09 ms | 4.47 ms | 245 tok/s |
| Metal | 4 generated tokens | 15.99 ms | 16.95 ms | 250 tok/s |
| CPU, 8 threads | 192-token prompt | 149.50 ms | 151.33 ms | 1,285 tok/s |
| CPU, 8 threads | 4 generated tokens | 19.27 ms | 22.83 ms | 208 tok/s |

The 192-token Metal prompt floor passes 50 ms but fails 20 ms before candidate scoring. The CPU
thread sweep covered 4, 8, and 12 threads; eight is the stable short-prompt winner. The whole
llama-bench process peaks at 728,973,312 bytes RSS on Metal and 1,302,986,752 bytes on CPU.

## Exact ScamGuard verdict scorer

The exact scorer uses the frozen prompt and length-normalized teacher-forced likelihoods for
`SAFE`, `UNCERTAIN`, and `SCAM`. It runs product batch one (`parallel=1`) over the same first 50
test messages used by the BF16 latency protocol. Candidate-inclusive input length is 153 minimum,
167 p50, 194.55 p95, and 196 maximum tokens.

| Context / threads | Repetitions | Mean ms/message | Median | p95 of run means | Peak process RSS |
|---|---:|---:|---:|---:|---:|
| 224 / 12 | 10 | 53.43 | 51.53 | 59.25 | 1.81 GB |
| 256 / 4 | 5 | 51.99 | 52.26 | 52.71 | 1.81 GB |
| 256 / 8 | 5 | 49.79 | 49.44 | 50.61 | 1.82 GB |
| 256 / 12 | 10 | **49.74** | **49.84** | **50.24** | **1.81 GB** |
| 640 / 12 | 10 | 53.27 | 53.18 | 53.98 | 3.38 GB |

Context 256 / 12 threads is the measured short-text configuration. Context 224 is rejected because
it is slower and less stable despite being large enough for this sample. Four and eight threads do
not improve the boundary. Context 640 remains the quality path for longer dialogue; reducing the
training ceiling would truncate supervision and is not authorized as a latency shortcut.

The exact-scorer p95 is a percentile across ten per-run mean task times, not a per-message p95:
upstream llama.cpp exposes score-phase start/end timestamps but not per-task timings. Full
structured category/evidence/action generation is also outside this measurement. Both limitations
make the routing decision conservative rather than a claim of final product latency.

## Persistent native per-request control

`native/gguf_verdict_runner.cpp` removes the aggregate-timing ambiguity. It loads the GGUF once,
accepts one locally hex-framed prompt at a time, tokenizes inside the timed native request, scores
the exact three continuations, and returns raw likelihoods plus elapsed microseconds. The Python
round-trip measurement includes pipe I/O. The first request is an unmeasured warmup for lazy Metal
kernel compilation. The runner uses the complete 640-token admission ceiling plus one 64-token
suffix-headroom bucket; no input is truncated.

The same 50 frozen test messages were run once per configuration after warmup:

| Threads | Mean round trip | p50 | Per-request p95 | p99 | Maximum |
|---:|---:|---:|---:|---:|---:|
| 4 | 56.66 ms | 55.60 ms | **59.65 ms** | 59.96 ms | 60.15 ms |
| 8 | 56.61 ms | 55.58 ms | 59.83 ms | 59.90 ms | **59.91 ms** |
| 12 | 56.74 ms | 55.63 ms | 60.04 ms | 60.22 ms | 60.35 ms |

Four threads is frozen as the default because it has the best p95. On an independent smoke prompt,
all three raw candidate scores match the patched reference evaluator within `2.46e-5` maximum
absolute error; the runtime release gate still requires exact calibrated-verdict parity on every
frozen example. These numbers apply to the untouched upstream Q4_0 control, not to the future
trained Q4_K_M or Q5_K_M candidates.

## Deployment contract

1. Keep the 640-token training ceiling and reject truncation. At runtime, allocate the smallest
   context that contains the complete tokenized request; short messages may use 256, while longer
   dialogues retain the slower 640-token path.
2. The fast router must independently clear its frozen safety gates and the complete fast path must
   stay below 20 ms p95. Qwen is invoked only for a frozen uncertainty region; its escalation rate
   and routed end-to-end p50/p95 must be published.
3. The upstream Q4_0 file is only a runtime control. After LoRA quality passes, merge the selected
   adapter, produce Q4_K_M and Q5_K_M, and choose the smallest quantization with decision and gate
   parity. Do not infer trained-model quality from this base.
4. Repeat artifact size, peak RAM, cold load, warm verdict, and energy measurements on a physical
   phone. This M4 Max result proves local desktop execution and a plausible payload size; it does
   not prove mobile performance.

## Evidence identities

- Stable upstream receipt SHA-256:
  `6adb95d6adbf711c9171928d6be0e6c71b317dce96eeb0b3c1165b85676ea21e`
- Generic runtime report SHA-256:
  `6cc032e4cc3571c179652774a90c36c697c852cfb3574e78d41dd8e0a8ae0ce2`
- Exact ctx-256 report SHA-256:
  `0b8349490a2a7eeb171fa4448da80fc0af462dffdc0ff6efd96771da463245f4`
- Exact ctx-640 report SHA-256:
  `bc6940df5c40c6ccb8df18f1505aec16048a54423294bdc533bafcdef5c310e9`
- llama-bench binary SHA-256:
  `9e88febf53bd7c64aad81af1ccd2c73726bf04d5f32d99aa543b32b48d904534`
- Patched llama-perplexity binary SHA-256:
  `de64bbd9844dafde58929f4b214ccb6004889ff5ef473a661b17c8e0344f9def`
- Persistent runner source SHA-256:
  `3930297c87ac0098597183b8f8875d68db5599ac1e631cd47811ce90af4d7487`
- Persistent runner build script SHA-256:
  `489a58ee8189b98b8f3c13c708311e05ef77e75a85cc9540b15af051956b3a05`
- Local persistent runner binary SHA-256:
  `4bc0784a89bf81d36b2044be734f488d653a699a011a2c9477392138883ea4ca`

The raw benchmark reports are ignored local run artifacts. They contain all timing samples and no
message text. The stable upstream receipt is tracked. Hugging Face publication remains
unauthorized until the audited trained model passes every downstream release gate.
