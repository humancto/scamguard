# ScamGuard Qwen3.5-2B schema-v6 model card

## Status

Historical schema-v6 research candidate, not the current schema-v13 fast-path experiment and not a
production safety guarantee. No SOTA claim is authorized until an
artifact passes the frozen ScamBench protocol and eligible external baselines are rerun on the
same examples, labels, and threshold policy.

## Model

- Base: `Qwen/Qwen3.5-2B`
- Base revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Adaptation: rank-16 LoRA on an explicit language-tower module allowlist
- Trainable parameters: 16,819,200 (0.7542% of the loaded 2.23B-parameter multimodal base)
- Training runtime: native arm64 PyTorch MPS with Transformers/PEFT
- Planned deployment: text-only GGUF Q4_K_M; the visual tower is neither trained nor exported

The smaller Qwen3.5-0.8B candidate is a post-quality compression challenger. It does not become
the release model merely because it is smaller.

## Intended use

Local classification of user-supplied SMS, email, chat, marketplace, job, delivery, payment, and
account messages into `SAFE`, `UNCERTAIN`, or `SCAM`. The SDK adds a calibrated risk score,
fine-grained category, deterministic verbatim evidence, and a reversible recommended action.
Benchmark and SDK likelihood scoring share the same full-continuation tokenizer helper so BPE
merges at the JSON/verdict boundary cannot make product predictions differ from evaluation. The
SDK serializes evidence as PRD-compatible `evidence_spans` objects with verified text offsets.

Do not use the model to automatically delete messages, contact senders, transfer money, submit
reports, or make legal determinations. `UNCERTAIN` requires user review or verification through a
known official channel.

## Training data

ScamBench v0.3 schema v6 contains 10,111 train, 2,634 dev, and 2,374 untouched test rows, plus 433
financial OOD, 488 WSPR OOD with no SAFE rows, 1,125 forum validation, and 2,300 forum OOD rows. The
forum validation slice selects training volume only; it is not used for fitting or threshold
selection. The 2,079-row realistic-placeholder derivative and all OOD outcomes remain untouched
until the cap is frozen. Sources are hash-pinned CC-BY-4.0 and MIT datasets plus 6,336 Apache-2.0 synthetic v4
examples across 130 isolated base/multilingual template families.

The Qwen train/development projection contains 12,745 strict JSON examples. All 4,025 SCAM
targets have non-empty verbatim evidence. Under the pinned Qwen tokenizer, full prompt-plus-target
length is 239 tokens at p95, 274 at p99, and 508 maximum; zero examples exceed the 512-token limit.
Only eight exceed 384 tokens, which is why the frozen run uses 512 rather than silently truncating
rare long messages.

The multilingual synthetic hard negatives are curated but have not yet completed native-speaker
review. That review is a production-release gate, not a completed quality claim.

Generic advertising/spam is `UNCERTAIN`, not automatically `SCAM`. Defensive guidance and
standalone authentication-code notifications are `SAFE` unless the message itself directs the
reader to an external action or contact; ambiguous defensive messages with such routing are
`UNCERTAIN`. A source-reported positive is SCAM only when the message itself contains a strong
fraud cue; otherwise it is UNCERTAIN. Bare-domain evidence uses a curated web-TLD allowlist so
measurement abbreviations such as `400sq.mtr` cannot become fake URLs. Exact text, masked-template,
and 64-bit character SimHash controls prevent family leakage. Every top-level scam category has an
independent family in train, dev, and test. Every train/dev/test SCAM row has deterministic
verbatim evidence. Evidence coverage is 77.4% on noisy financial OOD scams and 100% on the admitted
forum validation/OOD SCAM rows; source-reported low-cue rows remain UNCERTAIN rather than becoming
generative SCAM supervision.

Schema v6 privacy-normalizes every real source before IDs, deduplication, family clustering, and
splitting. Email addresses, long phone-like values, and long account-like digit sequences become
typed placeholders; the independent validator rejects a surviving pattern.

Direct Reddit user-content scraping is excluded because the current Reddit Data API Terms do not
grant model training rights without express rightsholder permission. The admitted forum source is
a separately published CC-BY-4.0 research artifact derived from five public reporting forums.
Raw and processed datasets are not committed to this source repository.

## Evaluation

The scam threshold and temperature are fitted on dev SAFE/SCAM rows only. Untouched test gates are:

- overall scam recall at least 97%;
- false-positive rate at most 2%;
- recall at least 97% for every scam category with 20 or more test examples;
- macro F1 at least 0.94 as a stretch target;
- separately published financial, WSPR, forum, realistic-placeholder, and adversarial results;
- desktop batch-one verdict latency at or below 20 ms for the fast path, with Qwen and routed
  end-to-end latency reported separately;
- strict generated JSON, enum, semantic consistency, and verbatim-evidence auditing;
- post-Q4 rerun with the same calibration and threshold.

The post-freeze 2.7 MB lexical floor measures 1.16 ms p95 on the development machine but fails the
quality gate (58.1% test recall at 0.115% FPR). The complete measured 2B result is recorded in
`reports/QWEN2B_REFERENCE.md`; this card remains schema-v6 historical evidence and must not be
presented as a schema-v12 result.

## Known limitations

- The current 240-row stratified audit sample has not received independent human labels.
- Multilingual synthetic hard negatives have not completed native-speaker review.
- Public datasets contain label noise, encoding artifacts, and regional/language imbalance.
- Cross-split deduplication controls contamination inside ScamBench, but cannot prove that a public
  UCI, Mendeley, WSPR, or forum message was absent from the base model's pretraining corpus. OOD,
  unseen-family, and synthetic tests reduce reliance on memorization but do not eliminate this risk.
- The deterministic evidence taxonomy is strongest in English; financial OOD evidence coverage is
  explicitly lower in Bengali and other multilingual cases.
- Messages can be novel, compromised, or deliberately evasive. A SAFE result is not proof of
  legitimacy; users should independently verify consequential requests.
- Mobile feasibility requires the measured Q4 artifact size, memory, latency, and quantized quality
  results; a successful desktop run alone is insufficient evidence of phone performance.
