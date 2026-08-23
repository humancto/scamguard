# Qwen3.5-0.8B upstream Q4 runtime floor

## Decision

Qwen3.5-0.8B is small enough to ship as an optional local specialist, but it is not the under-20-ms
fast path. The verified upstream Q4_0 control occupies 563,036,064 bytes. On a 40-GPU-core M4 Max,
192-token prompt processing alone measures 33.26 ms p95. The exact three-candidate ScamGuard
verdict scorer measures a 50.24 ms p95 of ten run means at a short-text 256-token context and
53.98 ms at the quality-preserving 640-token context. It therefore fails the strict fast-path
target and narrowly misses the PRD's formal `<50 ms` laptop target in the measured scorer. A newer
persistent native runner now supplies the stronger measurement the aggregate scorer could not.
Uncached, 50 individually timed requests measure 59.65 ms p95 with four threads and fail the
escalated-path latency gate. Reusing the exact 141-token fixed prompt state reduces the same
requests to 42.50 ms p95 with zero calibrated-verdict mismatches, establishing a viable desktop
specialist path that still requires trained-artifact confirmation.

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

The same 50 frozen test messages were run after warmup. The fixed-prefix configuration uses three
repetitions (150 requests) to make its release-relevant percentile less sensitive to a single run:

| Threads | Requests | Mean round trip | p50 | Per-request p95 | p99 | Maximum |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 56.66 ms | 55.60 ms | **59.65 ms** | 59.96 ms | 60.15 ms |
| 8 | 50 | 56.61 ms | 55.58 ms | 59.83 ms | 59.90 ms | **59.91 ms** |
| 12 | 50 | 56.74 ms | 55.63 ms | 60.04 ms | 60.22 ms | 60.35 ms |
| 4, fixed-prefix cache | 150 | **38.64 ms** | **38.20 ms** | **42.50 ms** | **43.17 ms** | **43.36 ms** |

Four threads is frozen as the default because it has the best p95. On an independent smoke prompt,
all three raw candidate scores match the patched reference evaluator within `2.46e-5` maximum
absolute error; the runtime release gate still requires exact calibrated-verdict parity on every
frozen example. Across the 50-message prefix-cache comparison, maximum raw-score drift versus the
uncached native runner is `0.003125`, maximum calibrated-probability drift is `0.00004351`, and
calibrated-verdict mismatches are zero. Prefix reuse is verified on every escalated trace row and
cannot silently fall back during release measurement. These numbers apply to the untouched upstream
Q4_0 control, not to the future trained Q4_K_M or Q5_K_M candidates.

The same control is now assembled as a self-contained desktop runtime pack: the 563,036,064-byte
GGUF, a 5,267,976-byte statically linked arm64 Metal/Accelerate runner, normalized calibration, and
the frozen prompt fragments. The public `Scanner` path ran 150 requests at 35.86 ms mean, 35.38 ms
p50, **39.60 ms p95**, 39.93 ms p99, and 40.14 ms maximum. Every request reused the 141-token
prefix, and the runtime imported no Transformers package. The pack manifest remains explicitly
non-publishable because this is the untouched base-model control.

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
- Persistent fixed-prefix control report SHA-256:
  `26beda09e2d66372d667e078c1e5bcb8fe002698c1684b1ffadb8968eae2c144`
- Portable runtime-pack manifest SHA-256:
  `d9a8dfa44bcf5268bf2ce8ad16bd33d406df3cfd892963e40697b332aa159d3f`
- Portable public-SDK benchmark report SHA-256:
  `040add6f0b4f3a88b315fb8a283491af9b6a89881f07a3a8e2797fead27680ef`
- Portable static runner binary SHA-256:
  `a715ccdebe50041d13265532230a154d9a7dec4729113ceca8f09fda188799fd`
- llama-bench binary SHA-256:
  `9e88febf53bd7c64aad81af1ccd2c73726bf04d5f32d99aa543b32b48d904534`
- Patched llama-perplexity binary SHA-256:
  `de64bbd9844dafde58929f4b214ccb6004889ff5ef473a661b17c8e0344f9def`
- Persistent runner source SHA-256:
  `36ce1cf425a97b763d23308525a9bb156d4f34541f228e88c50bdcbb47a31884`
- Persistent runner build script SHA-256:
  `b9dbf5f95968d075ec99e7ebfac8e7c849439ac9c656dadcc214fac4407b960a`
- Local persistent runner binary SHA-256:
  `9c3b2525b72a0f56684a9c6ba63b14366c59d0a5a16b09f02f898afba267f8c3`

The raw benchmark reports are ignored local run artifacts. They contain all timing samples and no
message text. The stable upstream receipt is tracked. Hugging Face publication remains
unauthorized until the audited trained model passes every downstream release gate.
