# Model selection record

## Decision

Quality is the first constraint; phone/desktop viability is the second. The controlled sweep is:

1. ModernBERT-base (149M) as the fast calibrated classifier/router;
2. ModernBERT-large (395M) as the sub-1B capacity and distillation-teacher candidate;
3. Qwen3.5-0.8B as the smallest multimodal/generative challenger;
4. Qwen3.5-2B as the quality-first Qwen candidate, trained as a 16.8M-parameter LoRA adapter;
5. Qwen3.5-4B as the required escalation if the 2B candidate misses any safety gate;
6. Qwen3.5-9B as a desktop teacher/upper bound if the 4B model still leaves a material quality gap;
7. a hybrid where ModernBERT handles clear cases and Qwen handles the uncertainty band.

Qwen3-0.6B remains a fallback stability/size control. Qwen3.6 is not a small-model family: the
current official releases begin at 27B dense and 35B-A3B MoE, so it is outside the product budget.
The 4B experiment pins official revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and preserves
the 2B run's effective batch size of 16 through batch 8 with two-step accumulation.

## Why not choose by parameter count

The output requires calibrated probabilities and an abstention band, not merely valid JSON.
Encoder classifiers are usually faster and easier to calibrate; the Qwen candidates can add
cross-channel context, categories, and constrained explanations. The winner is the smallest
deployment that passes recall/FPR/OOD gates, whether that is one model or a routed pair.

The schema-v9 149M run passes the desktop-forward latency target at 16.67 ms p95 but fails the
sole-detector false-positive gate and misses one complete held-out development family. The 395M
control also misses all 72 examples in that family, fails dialogue and multilingual diagnostics,
and measures 26.78 ms p95 before tokenization at 1.59 GB unquantized. Capacity did not repair the
data shortcut; ModernBERT-large is rejected as the product model and retained only as a historical
capacity control.

Schema v10 reran the 149M candidate after adding 1,800 human-authored legitimate transactional
dialogues and 768 paired multi-turn synthetic counterfactuals. It repaired the held-out development
identity family and reduced Taskmaster selection FPR from 95.78% to 0%, but learned the wrong broad
rule: BothBosu dialogue scam recall collapsed from 99.29% to 13.48%, regression FPR rose to 11.23%,
and macro F1 remained below target. The schema-v10 model is rejected.

Schema v11 is the controlled correction. It uses 600 Taskmaster fitting families, 1,536 balanced
paired synthetic dialogues across 12 scenarios, a complete-recent-turn cap verified below 256
tokens on all eligible Taskmaster conversations, and a versioned speaker-neutral transform that
removes `USER`/`ASSISTANT` versus `caller`/`receiver` corpus cues before tokenization. Development
and regression remain byte-identical to schema v9. Candidate selection must improve both the
450-row Taskmaster SAFE FPR and the 294-row BothBosu balance without degrading short-message recall.
The 1,049-row BothBosu OOD and 1,820-row MOZ primary test remain sealed.

The three-epoch schema-v11 result clears the development boundary at 97.86% recall / 1.99% FPR and
measures 15.55 ms p95 from tokenizer entry through probability output. It is still rejected:
regression FPR is 11.57%, financial OOD FPR is 67.25%, and the BothBosu selection result is 77.30%
recall / 45.75% FPR. An incoming-turn maximum raises dialogue recall to 100% only by flagging 93.46%
of SAFE dialogue and is also rejected.

Schema v12 tested the narrow correction by adding 512 balanced, train-only counterfactual rows in
eight paired families. It cut unchanged-regression FPR from 11.57% to 4.18% and retained 99.49%
recall at 14.27 ms end-to-end p95, but overcorrected: development recall fell to 85.60%, including
0/72 recall on the untouched `identity_case_callback` family. On the open selection slices it
reached only 61.70% recall / 34.64% FPR on BothBosu and 0% FPR on Taskmaster. Chichewa recall fell to
58.73% at 11.88% FPR. Text-free ledgers localize the failure: all 72 catastrophic development misses
come from that one identity family, while 59 of 73 remaining regression false positives come from
`family_transfer_verified`. Schema v12 is rejected as a data-mixture ablation.

The next encoder experiment must vary counterfactual dose and pairing coverage rather than stack
another broad corpus increment. The development, regression, Taskmaster, and BothBosu selection
rows stay fixed; the 1,049-row BothBosu OOD and MOZ primary test remain unopened. A schema-v13
candidate may advance only if it preserves every development scam family while improving the same
frozen regression and open-dialogue gates.

Schema v13 reduces the increment to 16 rows per family (128 total). The selected checkpoint recovers
development to 95.91% recall / 1.94% FPR and 53/72 identity-family detections, but unchanged
regression remains at 99.32% recall / 4.18% FPR. BothBosu dialogue selection deteriorates to 51.06%
recall / 18.30% FPR. The neural candidate alone is rejected.

A separately frozen `trusted-channel-v1` policy uses two extractive trust-boundary rules and adds
0.11 ms p95 after signal extraction. On the open short-message slices it changes schema v13 to
99.61% recall / 0.10% FPR on development and 99.32% / 0.11% on regression; an 18,729-row open
coverage audit finds no wrong-direction override. This is a promising fast-path candidate, not an
independent release score: the policy was designed from already-open errors, macro F1 remains below
stretch, and dialogue selection still fails. The sealed dialogue and MOZ sources remain unopened.

A deployment-only export then demonstrated exact verdict portability: a self-contained dynamic
FP32 ONNX pack matches all 5,008 open development/regression frozen decisions and measures 13.92 ms
p95 on four CPU threads. The 55.4%-smaller dynamic INT8 graph measures 13.87 ms but changes 44
frozen decisions across those splits and degrades category/macro metrics, so it is rejected rather
than accepted on size alone. Neither result changes the model-selection decision or substitutes for
physical-mobile measurement.

Direct Core ML conversion strengthens deployment feasibility without changing that rejection. The
FP32 128-token ML Program preserves all 5,008 open frozen binary verdicts and measures 5.65 ms p95
end-to-end on the reference Mac. Its complete pack is 602.5 MB. FP16 reaches 3.83 ms p95 and cuts
the package roughly in half, but changes 19 frozen decisions and reduces development recall to
93.19%; it is rejected. Neither result repairs the open dialogue, raw regression-FPR, or macro-F1
gates, and neither is a physical-mobile result.

The verified upstream Qwen3.5-0.8B Q4_0 control is 563,036,064 bytes. On the 40-GPU-core M4 Max,
192-token prompt processing measures 33.26 ms p95, already above the strict 20 ms fast-path budget.
The historical protocol-v2 three-candidate verdict scorer reaches a 50.24 ms p95 across ten run
means at context 256 and 53.98 ms at the quality-preserving context 640; process RSS peaks at 1.81
GB and 3.38 GB, respectively. Protocol v3's one-pass branch-token scorer requires a new benchmark;
the older figures are not reused. This freezes 0.8B Qwen as
a routed specialist rather than the default fast path. It does not pre-judge trained quality or
replace required physical-phone measurements.

The paired routing evaluator confirms the architecture contract without promoting the base model.
On identical schema-v23/24 test IDs, mandatory encoder-uncertain escalation to untouched Qwen base
preserves 100% binary recall and improves calibrated three-way macro F1 from 0.7519 to 0.7730, but
raises SAFE FPR from 1.03% to 1.37% and still fails the 0.94 macro-F1 gate. Development selection
therefore chooses no additional confidence-margin traffic. A trained 0.8B specialist must beat the
encoder-only baseline on the same rows and clear every gate; model size alone earns no routing role.

The first schema-24 Qwen3.5-0.8B adapter passed 29/39 protocol-v3 branch-token gates and was
rejected, chiefly because complete MultiDoGO calls produced a 5.69% SAFE false-positive rate. A
family-disjoint call-robustness continuation reduced that result to 1.56% and passed 35/39 gates.
It remains rejected: unchanged-test SAFE FPR is 3.72%, calibrated macro F1 is 0.7502, the software
call subgroup is 3.42% FPR, and the prior-open BothBosu scam recall is 56.74%. Quantization is not
authorized for either adapter.

The frozen stage-3 correction is deliberately narrower than another broad-data increment. It
replays all 23,435 parent SFT rows once, presents 895 publisher-training complete calls once, and
adds 768 balanced advisory-grounded dialogues where label-matched neutral closing turns follow the
decisive earlier behavior. The supplement has zero SimHash-radius-6 overlap with the open
development/regression rows, BothBosu validation and sealed OOD rows, held MultiDoGO calls, or the
sealed primary test. The 25,098-row curriculum has no sequence above 598 tokens at the frozen
640-token limit. Stage 3 continues from the rejected stage-2 adapter at `1e-5`; it earns no release
or quantization status until it clears the same 39 gates without moving any threshold or slice.

The completed stage-3 result passes 36/39 gates and remains rejected. It preserves 99.83% held-test
scam recall, reduces complete MultiDoGO call SAFE FPR to 1.12% with every domain at or below 2.05%,
and cuts long-call SAFE FPR from stage 2's 22.37% to 0.22%. Those are real corrections, but held-test
SAFE FPR regresses from 3.72% to 4.07%, calibrated macro F1 is 0.7370, and prior-open BothBosu scam
recall rises only from 56.74% to 58.16%. Its BF16 reference path measures 93.54 ms median and 99.11
ms p95 at 163--190 input tokens. Quantization remains unauthorized.

The frozen stage-4 design addresses those three failures without fitting any evaluation row. It
continues from the exact stage-3 adapter, fully replays the parent corpus and publisher-training
complete calls, retains the stage-3 persistence rows, adds advisory-grounded SAFE/UNCERTAIN/SCAM
dialogue triads for the four weak BothBosu categories, repeats training-only UNCERTAIN rows twice,
and repeats training-only `scamguard_synthetic_v5` SAFE rows once. The lower `5e-6` learning rate is
intended to preserve stage-3 call behavior while separating the underlearned three-way boundary.
The same frozen 39 gates remain authoritative; overlap, token-length, and config hashes must be
recorded before training.

Stage 4 completed and is rejected at 33/39 gates. It improved the prior-open BothBosu scam recall
from 58.16% to 82.98%, but raised BothBosu SAFE FPR from 0% to 9.15%, held-test SAFE FPR from 4.07%
to 5.50%, and complete MultiDoGO-call SAFE FPR from 1.12% to 1.90%; calibrated held-test macro F1
fell to 0.7146. Its 82.83 ms median / 85.59 ms p95 BF16 reference latency is encouraging but does
not compensate for failed quality gates. Adapter SHA-256 is
`b61000295dda5afc10dce44d9d15c3c94d461947a922cad78e0f4d394a24b556`.

A post-hoc diagnostic then tested 202 arithmetic and log-linear interpolations of the text-free
stage-3 and stage-4 ledgers. Method, weight, and thresholds were selected on development only. The
leakage-safe policy selected arithmetic weight `0.0` for stage 4: exactly stage 3, at 97.08% dev
recall and 0.99% dev SAFE FPR. Although 64 interpolations met the dev contract, no blend improved
dev recall and every nonzero stage-4 weight tied while spending safety margin or degraded it.
Simple output-score interpolation is therefore ruled out as the next mobile candidate. This is a
post-hoc direction-finding result, not fresh held-out confirmation; previously inspected splits
remain diagnostic, and the sealed primary test remains unopened. Stage 5 must recover dialogue
sensitivity in one adapter without the broad SAFE-boundary shift before quantization can begin.

Linear LoRA weight interpolation was then tested as a cheaper single-adapter alternative. The
hash-bound 6.25% stage-4 candidate still meets the binary dev contract, but only ties stage 3 at
97.08% recall while worsening SAFE FPR from 0.99% to 1.79% and calibrated macro F1 from 0.7165 to
0.6831. At 12.5% stage-4 weight, recall falls to 96.11% and the joint contract fails. Both were
scored with the explicit development-only evaluator mode, which requires `--splits dev` exactly,
does not read regression data, emits no release gates, and refuses to persist release calibration.
The coarse higher-weight sweep was stopped after the trajectory was dominated; no interpolated
adapter was evaluated on regression splits. Weight interpolation is therefore also rejected.

The 4B and 9B checkpoints are escalation tools, not automatic winners. Qwen3.5-4B remains practical
as a roughly 3 GB-class Q4 desktop/high-memory-mobile artifact; Qwen3.5-9B is a desktop teacher whose
errors and soft labels can improve a smaller student. If the 2B model already passes every frozen
gate, larger models must demonstrate a statistically meaningful gain before they justify their
latency and footprint.

The historical schema-v6 Qwen3.5-4B LoRA was also scored on the two open multi-turn selection
slices using its original frozen calibration. On 294 BothBosu conversations it reached 92.91% scam
recall but 24.84% SAFE FPR. On 450 Taskmaster SAFE conversations it raised 20 scam alarms, a 4.44%
FPR. The reference runtime sampled roughly 9.2 GB allocated on MPS. These are architecture
diagnostics, not schema-v12 results, but they reject the hypothesis that a larger generative model
automatically solves the dialogue boundary. The 4B model remains useful as a teacher or selective
explainer; it is not the sole detector.

## Training route

Use Transformers/PEFT for Qwen3.5 adapters and record the exact model revision. Do not make MLX the
only training path yet: open MLX-VLM and MLX-LM issues report corrupted Qwen3.5 LoRA generation and
a Metal descriptor leak, including on an M4 Max with 128 GB. MLX/GGUF/Core ML remain export/runtime
targets after adapter correctness is verified against the reference checkpoint.

The reproducible environment pins the exact upstream Transformers revision needed for the
`qwen3_5` architecture. Qwen3.5-2B text-only LoRA was verified on native Apple MPS at batch 16. A
pinned tokenizer audit of 12,745 schema-v6 chat examples found eight sequences above 384 tokens, so the
quality-first run uses a 512-token ceiling; the observed maximum is 508 and no supervision is
truncated. Its explicit module allowlist covers only the language tower; a runtime assertion
rejects any trainable visual-tower tensor. The 0.8B model remains the post-quality compression
target rather than the default merely because its download is smaller.

On the 128 GB M4 Max, the longest 508-token grouped batch measured 36 seconds with gradient
checkpointing and 62 seconds without it. The no-checkpoint variant fit in memory but lost on unified
memory bandwidth, so the frozen 2B run keeps checkpointing enabled.

## Release rule

Do not publish a “SOTA” label from an internal split. First rerun allowed baselines on ScamBench,
freeze the evaluator, verify OOD and adversarial slices, measure quantized deployment, and publish
all failures as well as aggregate scores.
