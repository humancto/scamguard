.PHONY: install test lint fetch data youtube-scam-calls apptek-callcenter harper-valley multidogo multidogo-annotations multidogo-annotation-curriculum schema13-dose16 schema14-natural-dialogue schema15-legitimate-openings schema17-call-minimal-pairs schema18-call-evidence-pairs schema19-call-windows schema20-action-states schema21-human-calls schema22-service-evidence schema23-evidence-compaction schema24-annotated-hard-negatives schema24-audit schema24-audit-review schema24-audit-bundle schema24-audit-import schema24-audit-check schema24-ai-internal-audit encoder-schema13-dose16 encoder-schema14-natural-dialogue encoder-schema15-legitimate-openings encoder-schema16-cache encoder-schema16-preflight encoder-schema16-retention encoder-schema17-preflight encoder-schema17-pair-retention encoder-schema18-preflight encoder-schema18-action encoder-schema19-preflight encoder-schema19-windowmix encoder-schema20-preflight encoder-schema20-actionheads encoder-schema21-preflight encoder-schema21-human-calls encoder-schema22-preflight encoder-schema22-service-evidence encoder-schema22-gates encoder-schema23-cache encoder-schema23-preflight encoder-schema23-evidence-compaction encoder-schema23-gates apptek-eval-schema13 apptek-eval-schema14 apptek-eval-schema15 apptek-eval-schema16 apptek-eval-schema17 apptek-eval-schema18 youtube-eval-schema16 youtube-eval-schema17 youtube-eval-schema18 bothbosu-eval-schema18 encoder-onnx-export encoder-onnx-eval-fp32 encoder-onnx-eval-int8 encoder-coreml-export encoder-coreml-eval teleantifraud-fetch teleantifraud-audit fresh-holdout chichewa-holdout scam-dialogue-holdout taskmaster-dialogues audit audit-check baseline encoder encoder-large encoder-schema12 encoder-dialogue-base encoder-dialogue-large encoder-taskmaster-base encoder-taskmaster-large reference forum-learning-data forum-learning-curve qwen-data qwen-token-audit qwen-08b qwen-08b-full-data qwen-08b-full-token-audit qwen-08b-full-freeze qwen-08b-full qwen-08b-full-eval qwen-08b-full-gates qwen-08b-ai-internal-errors qwen-2b qwen-4b qwen-4b-schema9 qwen-batch-benchmark qwen-4b-batch-benchmark qwen-eval qwen-4b-core-eval qwen-4b-eval qwen-generation qwen-error-audit paired qwen-merge qwen-gguf qwen-gguf-eval huggingface-release-check demo
.PHONY: schema24-audit-handoff-preflight schema24-ai-internal-overlay qwen-08b-ai-internal-token-audit qwen-08b-ai-internal-freeze qwen-08b-ai-internal qwen-08b-ai-internal-eval qwen-08b-ai-internal-gates qwen-08b-ai-internal-branch-eval qwen-08b-ai-internal-branch-gates qwen-08b-call-robustness-data qwen-08b-call-robustness-token-audit qwen-08b-call-robustness-freeze qwen-08b-call-robustness qwen-08b-call-robustness-eval qwen-08b-call-robustness-gates qwen-08b-call-robustness-merge qwen-08b-call-robustness-gguf qwen-08b-call-robustness-native-q4-eval qwen-08b-call-robustness-native-q4-gates qwen-08b-call-robustness-primary-v8-freeze qwen-08b-call-robustness-primary-v8-eval qwen-08b-call-robustness-primary-v8-gates qwen-08b-training-preflight qwen-08b-batch-preflight qwen-08b-base-product-eval qwen-08b-base-gguf qwen-08b-base-gguf-benchmark qwen-08b-base-gguf-verdict-benchmark qwen-08b-base-native-gguf-prefix-benchmark encoder-schema23-ledger qwen-08b-base-routed-diagnostic qwen-08b-base-routed-runtime qwen-08b-full-routed-diagnostic qwen-08b-full-routed-runtime qwen-08b-full-merge qwen-08b-full-gguf qwen-08b-full-gguf-eval-q4 qwen-08b-full-gguf-eval-q5 routed-eval
.PHONY: gguf-verdict-runner portable-gguf-verdict-runner gguf-runtime-pack qwen-08b-base-runtime-pack qwen-08b-base-runtime-pack-benchmark qwen-08b-full-gguf-routed-diagnostic-q4 qwen-08b-full-gguf-routed-diagnostic-q5 qwen-08b-full-gguf-routed-runtime-q4 qwen-08b-full-gguf-routed-runtime-q5
.PHONY: mobile-benchmark-check mobile-ios-xcframework mobile-ios-simulator-smoke-build mobile-ios-simulator-smoke-run mobile-ios-simulator-smoke-verify mobile-android-jni mobile-android-smoke-apk mobile-android-physical-smoke-run mobile-android-physical-smoke-verify mobile-ios-package mobile-android-package

PYTHON_BIN ?= .venv/bin/python
QWEN08_FULL_DATA ?= data/experiments/schema24-annotated-hard-negatives/processed
QWEN08_FULL_CONFIG ?= configs/qwen35-08b-schema24-lora.json
QWEN08_FULL_OUTPUT ?= artifacts/checkpoints/qwen35-08b-schema24-lora
QWEN08_FULL_TOKEN_AUDIT ?= reports/runs/qwen35-08b-schema24-token-audit.json
QWEN08_FULL_LABEL_AUDIT ?= reports/data/schema24-label-audit-reviewed-completion.json
SCHEMA24_AUDIT_BUNDLE ?= dist/scamguard-schema24-blind-audit.zip
SCHEMA24_AUDIT_HANDOFF_REPORT ?= reports/SCHEMA24_AUDIT_HANDOFF_PREFLIGHT.json
SCHEMA24_RETURNED_AUDIT ?= data/audit/returned/scamguard_blind_audit.csv
SCHEMA24_REVIEWED_AUDIT ?= data/audit/schema24-label-audit.reviewed.csv
SCHEMA24_AI_AUDIT ?= data/audit/internal-ai/scamguard_blind_audit.csv
SCHEMA24_AI_AUDIT_REPORT ?= reports/data/schema24-ai-internal-audit.json
SCHEMA24_AI_OVERLAY ?= data/experiments/schema24-ai-internal-overlay/processed
QWEN08_AI_TOKEN_AUDIT ?= reports/runs/qwen35-08b-schema24-ai-internal-token-audit.json
QWEN08_AI_CONFIG ?= configs/qwen35-08b-schema24-ai-internal-exploratory.json
QWEN08_AI_OUTPUT ?= artifacts/checkpoints/qwen35-08b-schema24-ai-internal-lora
QWEN08_AI_REPORT ?= reports/runs/qwen35-08b-schema24-ai-internal.json
QWEN08_AI_GATE_REPORT ?= reports/runs/qwen35-08b-schema24-ai-internal-gates.json
QWEN08_AI_BRANCH_REPORT ?= reports/runs/qwen35-08b-schema24-ai-internal-branch.json
QWEN08_AI_BRANCH_GATE_REPORT ?= reports/runs/qwen35-08b-schema24-ai-internal-branch-gates.json
QWEN08_ROBUST_DATA ?= data/experiments/qwen35-08b-call-robustness-stage2
QWEN08_ROBUST_TOKEN_AUDIT ?= reports/runs/qwen35-08b-call-robustness-stage2-token-audit.json
QWEN08_ROBUST_CONFIG ?= configs/qwen35-08b-call-robustness-stage2.json
QWEN08_ROBUST_OUTPUT ?= artifacts/checkpoints/qwen35-08b-call-robustness-stage2-lora
QWEN08_ROBUST_REPORT ?= reports/runs/qwen35-08b-call-robustness-stage2-regression.json
QWEN08_ROBUST_GATE_REPORT ?= reports/runs/qwen35-08b-call-robustness-stage2-regression-gates.json
QWEN08_ROBUST_MERGED ?= artifacts/merged/qwen35-08b-call-robustness-stage2
QWEN08_ROBUST_GGUF_PREFIX ?= artifacts/gguf/scamguard-qwen35-08b-call-robustness-stage2
QWEN08_ROBUST_Q4_GGUF ?= $(QWEN08_ROBUST_GGUF_PREFIX)-q4_k_m.gguf
QWEN08_ROBUST_NATIVE_Q4_REPORT ?= reports/runs/qwen35-08b-call-robustness-stage2-q4-k-m-native.json
QWEN08_ROBUST_NATIVE_Q4_GATE_REPORT ?= reports/runs/qwen35-08b-call-robustness-stage2-q4-k-m-native-gates.json
QWEN08_ROBUST_PRIMARY_DECLARATION ?= configs/qwen35-08b-call-robustness-stage2-q4-primary-v8.json
QWEN08_ROBUST_PRIMARY_REPORT ?= reports/runs/qwen35-08b-call-robustness-stage2-q4-primary-v8.json
QWEN08_ROBUST_PRIMARY_GATE_REPORT ?= reports/runs/qwen35-08b-call-robustness-stage2-q4-primary-v8-gates.json
QWEN08_FULL_REPORT ?= reports/runs/qwen35-08b-schema24-full.json
QWEN08_FULL_GATE_REPORT ?= reports/runs/qwen35-08b-schema24-full-gates.json
QWEN08_FULL_EVAL_SPLITS ?= dev test ood_financial forum_validation ood_wspr ood_forum ood_azsc call_state_validation call_window_validation multidogo_call_validation multidogo_state_validation ftc_pattern_validation multidogo_annotation_dev multidogo_annotation_test ood_chichewa scam_dialogue_validation taskmaster_validation
QWEN08_FULL_MERGED ?= artifacts/merged/qwen35-08b-schema24-scamguard
QWEN08_FULL_GGUF_PREFIX ?= artifacts/gguf/scamguard-qwen35-08b-schema24
QWEN08_FULL_Q4_GGUF ?= $(QWEN08_FULL_GGUF_PREFIX)-q4_k_m.gguf
QWEN08_FULL_Q5_GGUF ?= $(QWEN08_FULL_GGUF_PREFIX)-q5_k_m.gguf
QWEN08_FULL_Q4_REPORT ?= reports/runs/qwen35-08b-schema24-q4-k-m.json
QWEN08_FULL_Q5_REPORT ?= reports/runs/qwen35-08b-schema24-q5-k-m.json
QWEN08_FULL_Q4_ROUTED_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-schema24-q4-k-m-routed.json
QWEN08_FULL_Q5_ROUTED_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-schema24-q5-k-m-routed.json
QWEN08_FULL_Q4_ROUTED_RUNTIME_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-schema24-q4-k-m-routed-runtime.json
QWEN08_FULL_Q5_ROUTED_RUNTIME_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-schema24-q5-k-m-routed-runtime.json
QWEN08_BASE_GGUF ?= artifacts/gguf/Qwen3.5-0.8B-Q4_0.gguf
LLAMA_CPP_DIR ?= ../llama.cpp
LLAMA_CPP_REVISION ?= 521a64cd01979bb5b1a466152c576a9d809b068d
LLAMA_BENCH ?= $(LLAMA_CPP_DIR)/build-scamguard-arm64/bin/llama-bench
LLAMA_PERPLEXITY ?= $(LLAMA_CPP_DIR)/build-scamguard-arm64/bin/llama-perplexity
GGUF_VERDICT_RUNNER_BUILD ?= build/native-gguf-verdict
GGUF_VERDICT_RUNNER ?= $(GGUF_VERDICT_RUNNER_BUILD)/scamguard-gguf-verdict
PORTABLE_GGUF_RUNNER_BUILD ?= build/native-gguf-verdict-portable
PORTABLE_GGUF_RUNNER ?= $(PORTABLE_GGUF_RUNNER_BUILD)/scamguard-gguf-verdict
GGUF_RUNTIME_PACK_MODEL ?=
GGUF_RUNTIME_PACK_CALIBRATION_SOURCE ?=
GGUF_RUNTIME_PACK_PURPOSE ?= release_candidate
GGUF_RUNTIME_PACK_OUTPUT ?= artifacts/runtime-packs/scamguard-qwen35-08b
QWEN08_BASE_RUNTIME_PACK ?= artifacts/runtime-packs/qwen35-08b-upstream-q4-control
QWEN08_BASE_RUNTIME_PACK_REPORT ?= reports/runs/qwen35-08b-upstream-q4-runtime-pack.json
ENCODER23_OUTPUT ?= artifacts/checkpoints/sg-modernbert-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right
ENCODER23_ROUTER_REPORT ?= reports/runs/sg-modernbert-schema23-router-source.json
ENCODER23_ROUTER_PREDICTIONS ?= reports/runs/sg-modernbert-schema23-router-source.predictions.jsonl
ROUTER_PREDICTIONS ?= $(ENCODER23_ROUTER_PREDICTIONS)
SPECIALIST_PREDICTIONS ?= $(QWEN08_FULL_REPORT:.json=.predictions.jsonl)
ROUTED_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-schema24-routed.json
QWEN08_FULL_ROUTED_RUNTIME_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-schema24-routed-runtime.json
QWEN08_BASE_REPORT ?= reports/runs/qwen35-08b-base-schema24-product-batch1.json
QWEN08_BASE_ROUTED_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-base-product-batch1-routed.json
QWEN08_BASE_ROUTED_RUNTIME_REPORT ?= reports/runs/sg-modernbert-schema23-qwen08-base-product-batch1-runtime.json
QWEN08_BASE_NATIVE_PREFIX_REPORT ?= reports/runs/qwen35-08b-upstream-q4-native-prefix.json
QWEN08_TRAINING_PREFLIGHT_REPORT ?= reports/runs/qwen35-08b-training-preflight.json
QWEN08_BATCH_PREFLIGHT_REPORT ?= reports/runs/qwen35-08b-microbatch4-accum4-preflight.json
ROUTED_RUNTIME_REPETITIONS ?= 3
MOBILE_BENCHMARK_REPORT ?=
MOBILE_GGUF_MODEL ?=
MOBILE_RUNTIME_CALIBRATION ?=
MOBILE_QUANTIZED_QUALITY ?=
MOBILE_PREDICTION_LEDGER ?=
IOS_RUNTIME_PACKAGE ?=
ANDROID_RUNTIME_PACKAGE ?=
IOS_XCFRAMEWORK ?= build/ScamGuardGGUF.xcframework
IOS_SIMULATOR_SMOKE_APP ?= build/ScamGuardSmoke.app
IOS_SIMULATOR_UDID ?=
IOS_SIMULATOR_SMOKE_RESULT ?= reports/runs/qwen35-08b-upstream-q4-ios-simulator-smoke.raw.json
IOS_SIMULATOR_SMOKE_EVIDENCE ?= reports/runs/qwen35-08b-upstream-q4-ios-simulator-smoke.json
IOS_SIMULATOR_SMOKE_REQUEST ?= mobile/ios/smoke/control-request.json
IOS_SIMULATOR_RUNTIME_PACK ?= $(QWEN08_BASE_RUNTIME_PACK)
ANDROID_NDK_DIR ?=
ANDROID_RUNTIME_BUILD ?= build/android-arm64
ANDROID_SDK_DIR ?=
ANDROID_SMOKE_KEYSTORE ?= $(HOME)/.android/debug.keystore
ANDROID_SMOKE_JNI ?= build/android-arm64-r27d/libscamguard-jni.so
ANDROID_SMOKE_APK ?= build/ScamGuardSmokeStable.apk
ANDROID_PHYSICAL_SERIAL ?=
ANDROID_PHYSICAL_SMOKE_RESULT ?= reports/runs/qwen35-08b-upstream-q4-android-physical-smoke.raw.json
ANDROID_PHYSICAL_SMOKE_EVIDENCE ?= reports/runs/qwen35-08b-upstream-q4-android-physical-smoke.json
IOS_RUNTIME_PACKAGE_BUILD ?= dist/scamguard-ios-runtime.zip
ANDROID_RUNTIME_PACKAGE_BUILD ?= dist/scamguard-android-runtime.zip

install:
	uv sync --extra train --extra dev

test:
	uv run pytest

mobile-benchmark-check:
	test -n "$(MOBILE_BENCHMARK_REPORT)"
	test -n "$(MOBILE_GGUF_MODEL)"
	test -n "$(MOBILE_RUNTIME_CALIBRATION)"
	test -n "$(MOBILE_QUANTIZED_QUALITY)"
	test -n "$(MOBILE_PREDICTION_LEDGER)"
	test -n "$(IOS_RUNTIME_PACKAGE)"
	test -n "$(ANDROID_RUNTIME_PACKAGE)"
	$(PYTHON_BIN) scripts/verify_mobile_benchmark.py \
		--report "$(MOBILE_BENCHMARK_REPORT)" \
		--gguf "$(MOBILE_GGUF_MODEL)" \
		--calibration "$(MOBILE_RUNTIME_CALIBRATION)" \
		--quantized-quality "$(MOBILE_QUANTIZED_QUALITY)" \
		--prediction-ledger "$(MOBILE_PREDICTION_LEDGER)" \
		--ios-runtime-package "$(IOS_RUNTIME_PACKAGE)" \
		--android-runtime-package "$(ANDROID_RUNTIME_PACKAGE)"

mobile-ios-xcframework:
	test -n "$(LLAMA_CPP_DIR)"
	scripts/build_ios_xcframework.sh "$(LLAMA_CPP_DIR)" "$(IOS_XCFRAMEWORK)"

mobile-ios-simulator-smoke-build:
	test -d "$(IOS_XCFRAMEWORK)"
	scripts/build_ios_simulator_smoke.sh "$(IOS_XCFRAMEWORK)" \
		"$(IOS_SIMULATOR_SMOKE_APP)"

mobile-ios-simulator-smoke-run:
	test -n "$(IOS_SIMULATOR_UDID)"
	scripts/run_ios_simulator_smoke.sh "$(IOS_SIMULATOR_UDID)" \
		"$(IOS_SIMULATOR_SMOKE_APP)" "$(IOS_SIMULATOR_RUNTIME_PACK)" \
		"$(IOS_SIMULATOR_SMOKE_RESULT)"

mobile-ios-simulator-smoke-verify:
	test -n "$(IOS_SIMULATOR_SMOKE_RESULT)"
	$(PYTHON_BIN) scripts/verify_ios_simulator_smoke.py \
		--result "$(IOS_SIMULATOR_SMOKE_RESULT)" \
		--request "$(IOS_SIMULATOR_SMOKE_REQUEST)" \
		--runtime-pack "$(IOS_SIMULATOR_RUNTIME_PACK)" \
		--evidence-output "$(IOS_SIMULATOR_SMOKE_EVIDENCE)"

mobile-android-jni:
	test -n "$(LLAMA_CPP_DIR)"
	test -n "$(ANDROID_NDK_DIR)"
	scripts/build_android_runtime.sh "$(LLAMA_CPP_DIR)" "$(ANDROID_NDK_DIR)" \
		"$(ANDROID_RUNTIME_BUILD)"

mobile-android-smoke-apk:
	test -n "$(ANDROID_SDK_DIR)"
	test -f "$(ANDROID_SMOKE_KEYSTORE)"
	test -f "$(ANDROID_SMOKE_JNI)"
	scripts/build_android_smoke_apk.sh "$(ANDROID_SMOKE_JNI)" \
		"$(ANDROID_SDK_DIR)" "$(ANDROID_SMOKE_KEYSTORE)" "$(ANDROID_SMOKE_APK)"

mobile-android-physical-smoke-run:
	test -n "$(ANDROID_PHYSICAL_SERIAL)"
	scripts/run_android_physical_smoke.sh "$(ANDROID_PHYSICAL_SERIAL)" \
		"$(ANDROID_SMOKE_APK)" "$(QWEN08_BASE_RUNTIME_PACK)" \
		"$(ANDROID_PHYSICAL_SMOKE_RESULT)"

mobile-android-physical-smoke-verify:
	$(PYTHON_BIN) scripts/verify_android_physical_smoke.py \
		--result "$(ANDROID_PHYSICAL_SMOKE_RESULT)" \
		--request mobile/android/smoke/control-request.json \
		--runtime-pack "$(QWEN08_BASE_RUNTIME_PACK)" \
		--apk "$(ANDROID_SMOKE_APK)" \
		--jni-library "$(ANDROID_SMOKE_JNI)" \
		--evidence-output "$(ANDROID_PHYSICAL_SMOKE_EVIDENCE)"

mobile-ios-package:
	test -d "$(IOS_XCFRAMEWORK)"
	uv run python scripts/build_mobile_runtime_package.py --platform ios \
		--runtime "$(IOS_XCFRAMEWORK)" --wrapper mobile/ios/ScamGuardRuntime.swift \
		--llama-source "$(LLAMA_CPP_DIR)" --toolchain-name Xcode \
		--toolchain-version "$$(xcodebuild -version | head -1 | awk '{print $$2}')" \
		--minimum-os-version 16.4 --output "$(IOS_RUNTIME_PACKAGE_BUILD)"

mobile-android-package:
	test -f "$(ANDROID_RUNTIME_BUILD)/libscamguard-jni.so"
	uv run python scripts/build_mobile_runtime_package.py --platform android \
		--runtime "$(ANDROID_RUNTIME_BUILD)/libscamguard-jni.so" \
		--wrapper mobile/android/src/main/kotlin/com/scamguard/runtime/ScamGuardNative.kt \
		--llama-source "$(LLAMA_CPP_DIR)" --toolchain-name "Android NDK" \
		--toolchain-version 27.3.13750724 --minimum-os-version 28 \
		--output "$(ANDROID_RUNTIME_PACKAGE_BUILD)"

lint:
	uv run ruff check .

fetch:
	uv run --extra train python scripts/fetch_datasets.py

data: fetch
	uv run --extra train python scripts/generate_synthetic.py
	uv run --extra train python scripts/generate_dialogue_curriculum.py
	uv run --extra train python scripts/build_taskmaster_hard_negatives.py
	uv run --extra train python scripts/build_dataset.py
	uv run --extra train python scripts/generate_adversarial.py
	uv run --extra train python scripts/materialize_forum_placeholders.py
	uv run --extra train python scripts/build_fresh_holdout.py
	uv run --extra train python scripts/build_chichewa_holdout.py
	uv run --extra train python scripts/build_scam_dialogue_holdout.py
	uv run --extra train python scripts/validate_dataset.py

fresh-holdout:
	uv run --extra train python scripts/build_fresh_holdout.py

chichewa-holdout: fetch
	uv run --extra train python scripts/build_chichewa_holdout.py

scam-dialogue-holdout: fetch
	uv run --extra train python scripts/build_scam_dialogue_holdout.py

taskmaster-dialogues: fetch
	uv run --extra train python scripts/build_taskmaster_hard_negatives.py

teleantifraud-fetch:
	uv run --extra train --extra neural python scripts/fetch_teleantifraud.py

teleantifraud-audit: teleantifraud-fetch
	uv run --extra train python scripts/audit_teleantifraud.py

audit: data
	uv run --extra train python scripts/create_audit_sample.py

audit-check:
	uv run python scripts/check_audit_completion.py

schema24-ai-internal-audit:
	$(PYTHON_BIN) scripts/analyze_internal_ai_audit.py \
		--decisions "$(SCHEMA24_AI_AUDIT)" \
		--bundle "$(SCHEMA24_AUDIT_BUNDLE)" \
		--canonical-audit data/audit/schema24-label-audit.csv \
		--canonical-manifest data/audit/schema24-label-audit.manifest.json \
		--output "$(SCHEMA24_AI_AUDIT_REPORT)"

schema24-ai-internal-overlay: schema24-ai-internal-audit
	@if [ -f "$(SCHEMA24_AI_OVERLAY)/manifest.json" ]; then \
		echo "Reusing AI-internal overlay: $(SCHEMA24_AI_OVERLAY)"; \
	else \
		$(PYTHON_BIN) scripts/build_ai_internal_overlay.py \
			--source "$(QWEN08_FULL_DATA)" --output "$(SCHEMA24_AI_OVERLAY)" \
			--decisions "$(SCHEMA24_AI_AUDIT)" \
			--canonical-audit data/audit/schema24-label-audit.csv \
			--internal-report "$(SCHEMA24_AI_AUDIT_REPORT)"; \
	fi
	$(PYTHON_BIN) training/build_qwen_sft.py \
		--data "$(SCHEMA24_AI_OVERLAY)" --output "$(SCHEMA24_AI_OVERLAY)/qwen_sft"

qwen-08b-ai-internal-token-audit: schema24-ai-internal-overlay
	$(PYTHON_BIN) scripts/audit_qwen_tokens.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--data "$(SCHEMA24_AI_OVERLAY)/qwen_sft" --max-length 640 \
		--local-files-only \
		--output "$(QWEN08_AI_TOKEN_AUDIT)"

qwen-08b-ai-internal-freeze: qwen-08b-ai-internal-token-audit
	@if [ -f "$(QWEN08_AI_CONFIG)" ]; then \
		echo "Reusing immutable exploratory config: $(QWEN08_AI_CONFIG)"; \
	else \
		$(PYTHON_BIN) scripts/freeze_qwen08_ai_exploratory.py \
			--processed "$(SCHEMA24_AI_OVERLAY)" \
			--token-audit "$(QWEN08_AI_TOKEN_AUDIT)" \
			--internal-audit "$(SCHEMA24_AI_AUDIT_REPORT)" \
			--batch-selection reports/QWEN08_BATCH_GEOMETRY_SELECTION.json \
			--output "$(QWEN08_AI_CONFIG)" \
			--checkpoint-output "$(QWEN08_AI_OUTPUT)" \
			--experiment-id sg-qwen35-08b-schema24-ai-internal-v1; \
	fi

qwen-08b-ai-internal: qwen-08b-ai-internal-freeze
	$(PYTHON_BIN) scripts/verify_experiment_config.py \
		--config "$(QWEN08_AI_CONFIG)" --data "$(SCHEMA24_AI_OVERLAY)"
	@if [ -f "$(QWEN08_AI_OUTPUT)/adapter_model.safetensors" ]; then \
		echo "Reusing exploratory adapter: $(QWEN08_AI_OUTPUT)"; \
	else \
		$(PYTHON_BIN) training/train_qwen_lora.py \
			--experiment-config "$(QWEN08_AI_CONFIG)" \
			--model Qwen/Qwen3.5-0.8B \
			--revision 2fc06364715b967f1860aea9cf38778875588b17 \
			--local-files-only \
			--data "$(SCHEMA24_AI_OVERLAY)/qwen_sft" \
			--batch-size 4 --eval-batch-size 4 \
			--gradient-accumulation 4 --gradient-checkpointing \
			--max-length 640 --sampling-strategy group_by_length --require-mps \
			--output "$(QWEN08_AI_OUTPUT)"; \
	fi

qwen-08b-ai-internal-eval: qwen-08b-ai-internal
	$(PYTHON_BIN) training/eval_qwen.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--local-files-only --adapter "$(QWEN08_AI_OUTPUT)" \
		--data "$(SCHEMA24_AI_OVERLAY)" --external-data data/external \
		--splits $(QWEN08_FULL_EVAL_SPLITS) \
		--batch-size 1 --sequence-bucket-size 64 --require-mps \
		--report "$(QWEN08_AI_REPORT)"

qwen-08b-ai-internal-gates: qwen-08b-ai-internal-eval
	$(PYTHON_BIN) scripts/check_qwen08_full_gates.py \
		--report "$(QWEN08_AI_REPORT)" --output "$(QWEN08_AI_GATE_REPORT)"

qwen-08b-ai-internal-branch-eval: qwen-08b-ai-internal
	$(PYTHON_BIN) training/eval_qwen.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--local-files-only --adapter "$(QWEN08_AI_OUTPUT)" \
		--data "$(SCHEMA24_AI_OVERLAY)" --external-data data/external \
		--splits $(QWEN08_FULL_EVAL_SPLITS) \
		--batch-size 1 --sequence-bucket-size 64 --scoring-mode branch_token \
		--min-recall-for-threshold 0.97 --require-mps \
		--report "$(QWEN08_AI_BRANCH_REPORT)"

qwen-08b-ai-internal-branch-gates: qwen-08b-ai-internal-branch-eval
	$(PYTHON_BIN) scripts/check_qwen08_full_gates.py \
		--report "$(QWEN08_AI_BRANCH_REPORT)" \
		--output "$(QWEN08_AI_BRANCH_GATE_REPORT)"

qwen-08b-call-robustness-data:
	@if [ ! -f "$(QWEN08_ROBUST_DATA)/manifest.json" ]; then \
		$(PYTHON_BIN) scripts/build_qwen_call_robustness_curriculum.py \
			--parent "$(SCHEMA24_AI_OVERLAY)" --multidogo data/external/multidogo \
			--output "$(QWEN08_ROBUST_DATA)" --multidogo-repetitions 3 \
			--core-per-label 1000; \
	fi

qwen-08b-call-robustness-token-audit: qwen-08b-call-robustness-data
	$(PYTHON_BIN) scripts/audit_qwen_tokens.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--local-files-only --data "$(QWEN08_ROBUST_DATA)/qwen_sft" \
		--max-length 640 --output "$(QWEN08_ROBUST_TOKEN_AUDIT)"

qwen-08b-call-robustness-freeze: qwen-08b-call-robustness-token-audit
	@if [ ! -f "$(QWEN08_ROBUST_CONFIG)" ]; then \
		$(PYTHON_BIN) scripts/freeze_qwen08_call_robustness.py \
			--curriculum "$(QWEN08_ROBUST_DATA)" \
			--token-audit "$(QWEN08_ROBUST_TOKEN_AUDIT)" \
			--initial-adapter "$(QWEN08_AI_OUTPUT)" \
			--source-report "$(QWEN08_AI_BRANCH_REPORT)" \
			--output "$(QWEN08_ROBUST_CONFIG)" \
			--checkpoint-output "$(QWEN08_ROBUST_OUTPUT)"; \
	fi

qwen-08b-call-robustness: qwen-08b-call-robustness-freeze
	@if [ ! -f "$(QWEN08_ROBUST_OUTPUT)/adapter_model.safetensors" ]; then \
		$(PYTHON_BIN) training/train_qwen_lora.py \
			--model Qwen/Qwen3.5-0.8B \
			--revision 2fc06364715b967f1860aea9cf38778875588b17 \
			--local-files-only --experiment-config "$(QWEN08_ROBUST_CONFIG)" \
			--data "$(QWEN08_ROBUST_DATA)/qwen_sft" \
			--initial-adapter "$(QWEN08_AI_OUTPUT)" \
			--epochs 1 --batch-size 4 --eval-batch-size 4 \
			--gradient-accumulation 4 --learning-rate 0.00002 --max-length 640 \
			--sampling-strategy group_by_length --seed 20260824 --require-mps \
			--output "$(QWEN08_ROBUST_OUTPUT)"; \
	fi

qwen-08b-call-robustness-eval: qwen-08b-call-robustness
	$(PYTHON_BIN) training/eval_qwen.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--local-files-only --adapter "$(QWEN08_ROBUST_OUTPUT)" \
		--data "$(SCHEMA24_AI_OVERLAY)" --external-data data/external \
		--splits $(QWEN08_FULL_EVAL_SPLITS) \
		--batch-size 1 --sequence-bucket-size 64 --scoring-mode branch_token \
		--min-recall-for-threshold 0.97 --require-mps \
		--report "$(QWEN08_ROBUST_REPORT)"

qwen-08b-call-robustness-gates: qwen-08b-call-robustness-eval
	$(PYTHON_BIN) scripts/check_qwen08_full_gates.py \
		--report "$(QWEN08_ROBUST_REPORT)" \
		--output "$(QWEN08_ROBUST_GATE_REPORT)"

qwen-08b-call-robustness-merge: qwen-08b-call-robustness-gates
	$(PYTHON_BIN) training/merge_qwen_adapter.py \
		--base Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--adapter "$(QWEN08_ROBUST_OUTPUT)" --device cpu --dtype float32 \
		--output "$(QWEN08_ROBUST_MERGED)"

qwen-08b-call-robustness-gguf: qwen-08b-call-robustness-merge
	test -n "$(LLAMA_CPP_DIR)"
	scripts/export_gguf.sh "$(LLAMA_CPP_DIR)" "$(QWEN08_ROBUST_MERGED)" \
		"$(QWEN08_ROBUST_GGUF_PREFIX)" "$(PYTHON_BIN)"

qwen-08b-call-robustness-native-q4-eval: qwen-08b-call-robustness-gguf gguf-verdict-runner
	$(PYTHON_BIN) training/eval_gguf_native.py \
		--model "$(QWEN08_ROBUST_Q4_GGUF)" --runner "$(GGUF_VERDICT_RUNNER)" \
		--data "$(SCHEMA24_AI_OVERLAY)" --external-data data/external \
		--splits $(QWEN08_FULL_EVAL_SPLITS) --fit-calibration-on-dev \
		--reference-predictions "$(QWEN08_ROBUST_REPORT:.json=.predictions.jsonl)" \
		--ctx-size 640 --batch-size 640 --ubatch-size 128 --threads 4 \
		--n-gpu-layers 99 --max-fpr 0.02 --min-recall 0.97 \
		--report "$(QWEN08_ROBUST_NATIVE_Q4_REPORT)"

qwen-08b-call-robustness-native-q4-gates: qwen-08b-call-robustness-native-q4-eval
	$(PYTHON_BIN) scripts/check_qwen08_full_gates.py \
		--report "$(QWEN08_ROBUST_NATIVE_Q4_REPORT)" \
		--output "$(QWEN08_ROBUST_NATIVE_Q4_GATE_REPORT)"

qwen-08b-call-robustness-primary-v8-freeze: qwen-08b-call-robustness-native-q4-gates
	$(PYTHON_BIN) scripts/freeze_gguf_primary_v8.py \
		--model "$(QWEN08_ROBUST_Q4_GGUF)" --runner "$(GGUF_VERDICT_RUNNER)" \
		--regression-report "$(QWEN08_ROBUST_NATIVE_Q4_REPORT)" \
		--gate-report "$(QWEN08_ROBUST_NATIVE_Q4_GATE_REPORT)" \
		--primary-test-v8 data/processed/primary_test_v8.jsonl \
		--quantization Q4_K_M --output "$(QWEN08_ROBUST_PRIMARY_DECLARATION)"

qwen-08b-call-robustness-primary-v8-eval: qwen-08b-call-robustness-primary-v8-freeze
	$(PYTHON_BIN) training/eval_gguf_native.py \
		--model "$(QWEN08_ROBUST_Q4_GGUF)" --runner "$(GGUF_VERDICT_RUNNER)" \
		--data "$(SCHEMA24_AI_OVERLAY)" --external-data data/external \
		--splits primary_test_v8 \
		--calibration-report "$(QWEN08_ROBUST_NATIVE_Q4_REPORT)" \
		--primary-test-v8 data/processed/primary_test_v8.jsonl \
		--final-artifact-declaration "$(QWEN08_ROBUST_PRIMARY_DECLARATION)" \
		--ctx-size 640 --batch-size 640 --ubatch-size 128 --threads 4 \
		--n-gpu-layers 99 --max-fpr 0.02 --min-recall 0.97 \
		--report "$(QWEN08_ROBUST_PRIMARY_REPORT)"

qwen-08b-call-robustness-primary-v8-gates: qwen-08b-call-robustness-primary-v8-eval
	$(PYTHON_BIN) scripts/check_primary_v8_gates.py \
		--report "$(QWEN08_ROBUST_PRIMARY_REPORT)" \
		--output "$(QWEN08_ROBUST_PRIMARY_GATE_REPORT)"

baseline: data
	uv run --extra train python training/train_linear.py

encoder: data
	uv run --extra train --extra neural python training/train_encoder.py

encoder-large: data
	uv run --extra train --extra neural python training/train_encoder.py \
		--model answerdotai/ModernBERT-large \
		--revision 45bb4654a4d5aaff24dd11d4781fa46d39bf8c13 \
		--batch-size 8 --gradient-accumulation 2 --no-gradient-checkpointing \
		--output artifacts/checkpoints/sg-modernbert-large-schema9-safety \
		--report reports/runs/sg-modernbert-large-schema9-safety.json

encoder-schema12: data
	uv run --extra train --extra neural python training/train_encoder.py \
		--epochs 3 --dialogue-policy speaker-neutral-v1 \
		--output artifacts/checkpoints/sg-modernbert-schema12-counterfactual \
		--report reports/runs/sg-modernbert-schema12-counterfactual.json

schema13-dose16:
	$(PYTHON_BIN) scripts/generate_synthetic.py \
		--targeted-per-family 16 \
		--output data/experiments/schema13-dose16/generated/synthetic.jsonl
	$(PYTHON_BIN) scripts/build_dataset.py \
		--schema-version 13 \
		--synthetic data/experiments/schema13-dose16/generated/synthetic.jsonl \
		--output data/experiments/schema13-dose16/processed
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 13 \
		--sealed-data data/processed \
		--data data/experiments/schema13-dose16/processed

encoder-schema13-dose16: schema13-dose16
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema13-dose16/processed \
		--epochs 3 --dialogue-policy speaker-neutral-v1 \
		--output artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--report reports/runs/sg-modernbert-schema13-dose16.json

youtube-scam-calls:
	$(PYTHON_BIN) scripts/fetch_youtube_scam_calls.py
	$(PYTHON_BIN) scripts/build_youtube_scam_calls.py

apptek-callcenter:
	$(PYTHON_BIN) scripts/fetch_apptek_callcenter.py
	$(PYTHON_BIN) scripts/build_apptek_callcenter.py

schema14-natural-dialogue: youtube-scam-calls
	@if [ -f data/experiments/schema14-natural-dialogue/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v14 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema14_natural_dialogue.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 14 \
		--sealed-data data/processed \
		--data data/experiments/schema14-natural-dialogue/processed

encoder-schema14-natural-dialogue: schema14-natural-dialogue
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema14-natural-dialogue/processed \
		--epochs 3 --dialogue-policy speaker-neutral-v1 \
		--output artifacts/checkpoints/sg-modernbert-schema14-natural-dialogue \
		--report reports/runs/sg-modernbert-schema14-natural-dialogue.json

schema15-legitimate-openings:
	$(PYTHON_BIN) scripts/generate_legitimate_call_openings.py
	@if [ -f data/experiments/schema15-legitimate-openings-dose16/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v15 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema15_legitimate_openings.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 15 --sealed-data data/processed \
		--data data/experiments/schema15-legitimate-openings-dose16/processed

encoder-schema15-legitimate-openings: schema15-legitimate-openings
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema15-legitimate-openings-dose16/processed \
		--epochs 3 --dialogue-policy speaker-neutral-v1 \
		--output artifacts/checkpoints/sg-modernbert-schema15-legitimate-openings-dose16 \
		--report reports/runs/sg-modernbert-schema15-legitimate-openings-dose16.json

schema17-call-minimal-pairs:
	$(PYTHON_BIN) scripts/generate_call_minimal_pairs.py
	@if [ -f data/experiments/schema17-call-minimal-pairs/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v17 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema17_call_minimal_pairs.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 17 --sealed-data data/processed \
		--data data/experiments/schema17-call-minimal-pairs/processed

schema18-call-evidence-pairs:
	$(PYTHON_BIN) scripts/generate_call_evidence_pairs.py
	@if [ -f data/experiments/schema18-call-evidence-pairs/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v18 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema18_call_evidence_pairs.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 18 --sealed-data data/processed \
		--data data/experiments/schema18-call-evidence-pairs/processed

schema19-call-windows: taskmaster-dialogues youtube-scam-calls
	$(PYTHON_BIN) scripts/generate_call_evidence_pairs.py
	@if [ -f data/experiments/schema19-call-windows/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v19 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema19_call_windows.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 19 --sealed-data data/processed \
		--data data/experiments/schema19-call-windows/processed

schema20-action-states: taskmaster-dialogues youtube-scam-calls
	$(PYTHON_BIN) scripts/generate_call_action_states.py
	@if [ -f data/experiments/schema20-action-states/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v20 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema20_action_states.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 20 --sealed-data data/processed \
		--data data/experiments/schema20-action-states/processed

harper-valley:
	$(PYTHON_BIN) scripts/fetch_harper_valley.py
	@if [ -f data/external/harper_valley/manifest.json ]; then \
		echo "Reusing immutable HarperValleyBank derivative; manifest will be rechecked"; \
	else \
		$(PYTHON_BIN) scripts/build_harper_valley_calls.py; \
	fi

schema21-human-calls: schema20-action-states harper-valley
	@if [ -f data/experiments/schema21-human-calls/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v21 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema21_human_calls.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 21 --sealed-data data/processed \
		--data data/experiments/schema21-human-calls/processed

multidogo: schema20-action-states
	$(PYTHON_BIN) scripts/fetch_multidogo.py
	@if [ -f data/external/multidogo/manifest.json ]; then \
		echo "Reusing immutable MultiDoGO derivative; manifest will be rechecked"; \
	else \
		$(PYTHON_BIN) scripts/build_multidogo_dialogues.py; \
	fi

multidogo-annotations:
	$(PYTHON_BIN) scripts/fetch_multidogo.py --annotations
	$(PYTHON_BIN) scripts/audit_multidogo_annotations.py

multidogo-annotation-curriculum: multidogo multidogo-annotations
	$(PYTHON_BIN) scripts/build_multidogo_annotation_curriculum.py

schema22-service-evidence: schema20-action-states multidogo
	@if [ -f data/experiments/schema22-service-evidence/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v22 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema22_service_evidence.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 22 --sealed-data data/processed \
		--data data/experiments/schema22-service-evidence/processed

schema23-evidence-compaction: schema20-action-states multidogo
	$(PYTHON_BIN) scripts/generate_ftc_pattern_action_states.py
	@if [ -f data/experiments/schema23-evidence-compaction/processed/manifest.json ]; then \
		echo "Reusing immutable schema-v23 experiment; preflight will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema23_evidence_compaction.py; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 23 --sealed-data data/processed \
		--data data/experiments/schema23-evidence-compaction/processed

schema24-annotated-hard-negatives: schema23-evidence-compaction multidogo-annotation-curriculum
	@if [ -f $(QWEN08_FULL_DATA)/manifest.json ]; then \
		echo "Reusing immutable schema-v24 experiment; validation will recheck it"; \
	else \
		$(PYTHON_BIN) scripts/build_schema24_annotated_hard_negatives.py \
			--output "$(QWEN08_FULL_DATA)"; \
	fi
	$(PYTHON_BIN) scripts/validate_dataset.py \
		--expected-schema-version 24 --sealed-data data/processed \
		--data "$(QWEN08_FULL_DATA)"

schema24-audit: schema24-annotated-hard-negatives
	$(PYTHON_BIN) scripts/create_audit_sample.py \
		--data "$(QWEN08_FULL_DATA)" \
		--output data/audit/schema24-label-audit.csv \
		--manifest-output data/audit/schema24-label-audit.manifest.json \
		--seed scamguard-schema24-audit-v1 \
		--extra-split multidogo_annotation_dev \
		--extra-split multidogo_annotation_test

schema24-audit-review: schema24-audit-handoff-preflight
	@echo "Send $(SCHEMA24_AUDIT_BUNDLE) to an independent reviewer; do not send the repository."
	@echo "Reviewer: extract the ZIP, then run python3 review.py --open"

schema24-audit-bundle:
	$(PYTHON_BIN) scripts/build_blind_audit_bundle.py \
		--audit data/audit/schema24-label-audit.csv \
		--audit-manifest data/audit/schema24-label-audit.manifest.json \
		--output "$(SCHEMA24_AUDIT_BUNDLE)" \
		--replace

schema24-audit-handoff-preflight: schema24-audit-bundle
	$(PYTHON_BIN) scripts/verify_blind_audit_handoff.py \
		--bundle "$(SCHEMA24_AUDIT_BUNDLE)" \
		--output "$(SCHEMA24_AUDIT_HANDOFF_REPORT)" \
		--replace

schema24-audit-import:
	$(PYTHON_BIN) scripts/import_blind_audit.py \
		--returned-audit "$(SCHEMA24_RETURNED_AUDIT)" \
		--bundle "$(SCHEMA24_AUDIT_BUNDLE)" \
		--canonical-audit data/audit/schema24-label-audit.csv \
		--canonical-manifest data/audit/schema24-label-audit.manifest.json \
		--output "$(SCHEMA24_REVIEWED_AUDIT)" \
		--report "$(QWEN08_FULL_LABEL_AUDIT)"

schema24-audit-check:
	$(PYTHON_BIN) scripts/verify_imported_blind_audit.py \
		--returned-audit "$(SCHEMA24_RETURNED_AUDIT)" \
		--bundle "$(SCHEMA24_AUDIT_BUNDLE)" \
		--canonical-audit data/audit/schema24-label-audit.csv \
		--canonical-manifest data/audit/schema24-label-audit.manifest.json \
		--reviewed-audit "$(SCHEMA24_REVIEWED_AUDIT)" \
		--report "$(QWEN08_FULL_LABEL_AUDIT)"

encoder-schema16-cache:
	$(PYTHON_BIN) training/cache_encoder_teacher_logits.py --require-mps

encoder-schema16-preflight: encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_continual_config.py

encoder-schema16-retention: encoder-schema16-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema15-legitimate-openings-dose16/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--dialogue-policy speaker-neutral-v1 \
		--retention-weight 2 --retention-temperature 2 --source-balance-alpha 0.5 \
		--output artifacts/checkpoints/sg-modernbert-schema16-retention-alpha05-w2 \
		--report reports/runs/sg-modernbert-schema16-retention-alpha05-w2.json

encoder-schema17-preflight: schema17-call-minimal-pairs encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_pair_config.py

encoder-schema17-pair-retention: encoder-schema17-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema17-call-minimal-pairs/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--dialogue-policy speaker-neutral-v1 \
		--retention-weight 2 --retention-temperature 2 \
		--pair-loss-weight 0.5 --pair-margin 2 \
		--output artifacts/checkpoints/sg-modernbert-schema17-pair-retention-w05-m2 \
		--report reports/runs/sg-modernbert-schema17-pair-retention-w05-m2.json

encoder-schema18-preflight: schema18-call-evidence-pairs encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_schema18_config.py

encoder-schema18-action: encoder-schema18-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema18-call-evidence-pairs/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--truncation-side left --dialogue-policy speaker-neutral-v1 \
		--retention-weight 4 --retention-temperature 2 \
		--pair-loss-weight 2 --pair-margin 3 --pair-repeats 2 \
		--output artifacts/checkpoints/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left \
		--report reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.json

encoder-schema19-preflight: schema19-call-windows encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_schema19_config.py

encoder-schema19-windowmix: encoder-schema19-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema19-call-windows/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--truncation-side left --dialogue-policy speaker-neutral-v1 \
		--retention-weight 4 --retention-temperature 2 \
		--pair-loss-weight 1 --pair-margin 3 \
		--output artifacts/checkpoints/sg-modernbert-schema19-windowmix-ret4-w1-m3-left \
		--report reports/runs/sg-modernbert-schema19-windowmix-ret4-w1-m3-left.json

encoder-schema20-preflight: schema20-action-states encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_schema20_config.py

encoder-schema20-actionheads: encoder-schema20-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema20-action-states/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--truncation-side left --dialogue-policy speaker-neutral-v1 \
		--retention-weight 4 --retention-temperature 2 \
		--action-targets sensitive_action_language,requested_disclosure_or_transfer,caller_controls_target,official_self_navigation,independent_verification,pressure_or_secrecy,irreversible_action \
		--action-loss-weight 0.5 --action-verdict-weight 0.25 \
		--output artifacts/checkpoints/sg-modernbert-schema20-actionheads-ret4-aw05-vw025-left \
		--report reports/runs/sg-modernbert-schema20-actionheads-ret4-aw05-vw025-left.json

encoder-schema21-preflight: schema21-human-calls encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_schema21_config.py

encoder-schema21-human-calls: encoder-schema21-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema21-human-calls/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--truncation-side left --dialogue-policy speaker-neutral-v1 \
		--retention-weight 4 --retention-temperature 2 \
		--action-targets sensitive_action_language,requested_disclosure_or_transfer,caller_controls_target,official_self_navigation,independent_verification,pressure_or_secrecy,irreversible_action \
		--action-loss-weight 0.5 --action-verdict-weight 0.25 \
		--output artifacts/checkpoints/sg-modernbert-schema21-human-calls-actionheads-ret4-aw05-vw025-left \
		--report reports/runs/sg-modernbert-schema21-human-calls-actionheads-ret4-aw05-vw025-left.json

encoder-schema22-preflight: schema22-service-evidence encoder-schema16-cache
	$(PYTHON_BIN) scripts/verify_encoder_schema22_config.py

encoder-schema22-service-evidence: encoder-schema22-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema22-service-evidence/processed \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--teacher-logits data/experiments/schema16-retention-alpha05-w2/teacher/schema13-train-logits.jsonl \
		--teacher-manifest data/experiments/schema16-retention-alpha05-w2/teacher/manifest.json \
		--epochs 1 --learning-rate 5e-6 --batch-size 16 \
		--truncation-side left --dialogue-policy speaker-neutral-v1 \
		--retention-weight 4 --retention-temperature 2 \
		--action-targets sensitive_action_language,requested_disclosure_or_transfer,caller_controls_target,official_self_navigation,independent_verification,pressure_or_secrecy,irreversible_action \
		--action-loss-weight 0.5 --action-verdict-weight 0.25 \
		--output artifacts/checkpoints/sg-modernbert-schema22-service-evidence-actionheads-ret4-aw05-vw025-left \
		--report reports/runs/sg-modernbert-schema22-service-evidence-actionheads-ret4-aw05-vw025-left.json
	$(MAKE) encoder-schema22-gates

encoder-schema22-gates:
	$(PYTHON_BIN) scripts/check_encoder_schema22_gates.py \
		--output reports/runs/sg-modernbert-schema22-service-evidence-actionheads-ret4-aw05-vw025-left.gates.json

encoder-schema23-cache: schema23-evidence-compaction
	$(PYTHON_BIN) training/cache_encoder_teacher_logits.py --require-mps \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema20-actionheads-ret4-aw05-vw025-left \
		--data data/experiments/schema20-action-states/processed/train.jsonl \
		--output data/experiments/schema23-evidence-compaction/teacher/schema20-evidence-recent-v2-verdict-logits.jsonl \
		--manifest data/experiments/schema23-evidence-compaction/teacher/manifest-v2.json \
		--dialogue-policy speaker-neutral-evidence-recent-v2 --max-length 256 --batch-size 32

encoder-schema23-preflight: encoder-schema23-cache
	$(PYTHON_BIN) scripts/verify_encoder_schema23_config.py

encoder-schema23-evidence-compaction: encoder-schema23-preflight
	$(PYTHON_BIN) training/train_encoder.py \
		--data data/experiments/schema23-evidence-compaction/processed \
		--external-data data/external \
		--init-checkpoint artifacts/checkpoints/sg-modernbert-schema20-actionheads-ret4-aw05-vw025-left \
		--teacher-logits data/experiments/schema23-evidence-compaction/teacher/schema20-evidence-recent-v2-verdict-logits.jsonl \
		--teacher-manifest data/experiments/schema23-evidence-compaction/teacher/manifest-v2.json \
		--epochs 1 --batch-size 16 --gradient-accumulation 1 --learning-rate 2e-6 \
		--max-length 256 --truncation-side right --dialogue-policy speaker-neutral-evidence-recent-v2 \
		--binary-loss-weight 1 --retention-weight 4 --retention-temperature 2 \
		--action-targets sensitive_action_language,requested_disclosure_or_transfer,caller_controls_target,official_self_navigation,independent_verification,pressure_or_secrecy,irreversible_action \
		--action-loss-weight 0.5 --action-verdict-weight 0.25 --seed 20260821 \
		--output artifacts/checkpoints/sg-modernbert-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right \
		--report reports/runs/sg-modernbert-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right.json
	$(MAKE) encoder-schema23-gates

encoder-schema23-gates:
	$(PYTHON_BIN) scripts/check_encoder_schema23_gates.py \
		--output reports/runs/sg-modernbert-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right.gates.json

encoder-schema23-ledger:
	test -f "$(ENCODER23_OUTPUT)/model.safetensors"
	$(PYTHON_BIN) training/train_encoder.py \
		--evaluate-only --data data/experiments/schema23-evidence-compaction/processed \
		--external-data data/external --output "$(ENCODER23_OUTPUT)" \
		--max-length 256 --truncation-side right \
		--dialogue-policy speaker-neutral-evidence-recent-v2 --batch-size 32 \
		--report "$(ENCODER23_ROUTER_REPORT)" \
		--predictions "$(ENCODER23_ROUTER_PREDICTIONS)"

routed-eval:
	test -f "$(ROUTER_PREDICTIONS)"
	test -f "$(SPECIALIST_PREDICTIONS)"
	$(PYTHON_BIN) training/eval_routed.py \
		--router-predictions "$(ROUTER_PREDICTIONS)" \
		--specialist-predictions "$(SPECIALIST_PREDICTIONS)" \
		--report "$(ROUTED_REPORT)"

qwen-08b-base-routed-diagnostic: qwen-08b-base-product-eval
	$(MAKE) routed-eval \
		SPECIALIST_PREDICTIONS="$(QWEN08_BASE_REPORT:.json=.predictions.jsonl)" \
		ROUTED_REPORT="$(QWEN08_BASE_ROUTED_REPORT)"

qwen-08b-base-routed-runtime: qwen-08b-base-routed-diagnostic
	test -f "$(ENCODER23_ROUTER_PREDICTIONS)"
	test -f "$(QWEN08_BASE_REPORT)"
	test -f "$(QWEN08_BASE_REPORT:.json=.predictions.jsonl)"
	test -f "$(QWEN08_BASE_ROUTED_REPORT)"
	$(PYTHON_BIN) benchmarks/benchmark_routed_transformers_runtime.py \
		--router-checkpoint "$(ENCODER23_OUTPUT)" \
		--router-predictions "$(ENCODER23_ROUTER_PREDICTIONS)" \
		--specialist-revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--specialist-report "$(QWEN08_BASE_REPORT)" \
		--specialist-score-cache-metadata "$(QWEN08_BASE_REPORT:.json=.scores/test.json)" \
		--specialist-predictions "$(QWEN08_BASE_REPORT:.json=.predictions.jsonl)" \
		--routed-report "$(QWEN08_BASE_ROUTED_REPORT)" \
		--data "$(QWEN08_FULL_DATA)" --repetitions "$(ROUTED_RUNTIME_REPETITIONS)" --require-mps \
		--report "$(QWEN08_BASE_ROUTED_RUNTIME_REPORT)"

qwen-08b-full-routed-diagnostic: qwen-08b-full-eval
	$(MAKE) routed-eval \
		SPECIALIST_PREDICTIONS="$(QWEN08_FULL_REPORT:.json=.predictions.jsonl)" \
		ROUTED_REPORT="$(ROUTED_REPORT)"

qwen-08b-full-routed-runtime: qwen-08b-full-routed-diagnostic
	test -f "$(QWEN08_FULL_OUTPUT)/adapter_model.safetensors"
	test -f "$(QWEN08_FULL_REPORT:.json=.scores/test.json)"
	$(PYTHON_BIN) benchmarks/benchmark_routed_transformers_runtime.py \
		--router-checkpoint "$(ENCODER23_OUTPUT)" \
		--router-predictions "$(ENCODER23_ROUTER_PREDICTIONS)" \
		--specialist-revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--specialist-adapter "$(QWEN08_FULL_OUTPUT)" \
		--specialist-report "$(QWEN08_FULL_REPORT)" \
		--specialist-score-cache-metadata "$(QWEN08_FULL_REPORT:.json=.scores/test.json)" \
		--specialist-predictions "$(QWEN08_FULL_REPORT:.json=.predictions.jsonl)" \
		--routed-report "$(ROUTED_REPORT)" \
		--data "$(QWEN08_FULL_DATA)" --repetitions "$(ROUTED_RUNTIME_REPETITIONS)" \
		--require-mps --report "$(QWEN08_FULL_ROUTED_RUNTIME_REPORT)"

apptek-eval-schema13: apptek-callcenter
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--data data/external/apptek_callcenter/apptek_call_selection.jsonl \
		--manifest data/external/apptek_callcenter/manifest.json \
		--split apptek_call_selection --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema13-dose16.apptek-call-selection.json \
		--predictions reports/runs/sg-modernbert-schema13-dose16.apptek-call-selection.predictions.jsonl

apptek-eval-schema14: apptek-callcenter
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema14-natural-dialogue \
		--data data/external/apptek_callcenter/apptek_call_selection.jsonl \
		--manifest data/external/apptek_callcenter/manifest.json \
		--split apptek_call_selection --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema14-natural-dialogue.apptek-call-selection.json \
		--predictions reports/runs/sg-modernbert-schema14-natural-dialogue.apptek-call-selection.predictions.jsonl

apptek-eval-schema15: apptek-callcenter
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema15-legitimate-openings-dose16 \
		--data data/external/apptek_callcenter/apptek_call_selection.jsonl \
		--manifest data/external/apptek_callcenter/manifest.json \
		--split apptek_call_selection --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema15-legitimate-openings-dose16.apptek-call-selection.json \
		--predictions reports/runs/sg-modernbert-schema15-legitimate-openings-dose16.apptek-call-selection.predictions.jsonl

apptek-eval-schema16: apptek-callcenter
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema16-retention-alpha05-w2 \
		--data data/external/apptek_callcenter/apptek_call_selection.jsonl \
		--manifest data/external/apptek_callcenter/manifest.json \
		--split apptek_call_selection --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema16-retention-alpha05-w2.apptek-call-selection.json \
		--predictions reports/runs/sg-modernbert-schema16-retention-alpha05-w2.apptek-call-selection.predictions.jsonl

youtube-eval-schema16: youtube-scam-calls
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema16-retention-alpha05-w2 \
		--data data/external/youtube_scam_calls/youtube_scam_validation.jsonl \
		--manifest data/external/youtube_scam_calls/manifest.json \
		--split youtube_scam_validation --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema16-retention-alpha05-w2.youtube-scam-validation.json \
		--predictions reports/runs/sg-modernbert-schema16-retention-alpha05-w2.youtube-scam-validation.predictions.jsonl

apptek-eval-schema17: apptek-callcenter
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema17-pair-retention-w05-m2 \
		--data data/external/apptek_callcenter/apptek_call_selection.jsonl \
		--manifest data/external/apptek_callcenter/manifest.json \
		--split apptek_call_selection --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema17-pair-retention-w05-m2.apptek-call-selection.json \
		--predictions reports/runs/sg-modernbert-schema17-pair-retention-w05-m2.apptek-call-selection.predictions.jsonl

youtube-eval-schema17: youtube-scam-calls
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema17-pair-retention-w05-m2 \
		--data data/external/youtube_scam_calls/youtube_scam_validation.jsonl \
		--manifest data/external/youtube_scam_calls/manifest.json \
		--split youtube_scam_validation --dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema17-pair-retention-w05-m2.youtube-scam-validation.json \
		--predictions reports/runs/sg-modernbert-schema17-pair-retention-w05-m2.youtube-scam-validation.predictions.jsonl

apptek-eval-schema18: apptek-callcenter
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left \
		--data data/external/apptek_callcenter/apptek_call_selection.jsonl \
		--manifest data/external/apptek_callcenter/manifest.json \
		--split apptek_call_selection --truncation-side left \
		--dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.apptek-call-selection.json \
		--predictions reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.apptek-call-selection.predictions.jsonl

youtube-eval-schema18: youtube-scam-calls
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left \
		--data data/external/youtube_scam_calls/youtube_scam_validation.jsonl \
		--manifest data/external/youtube_scam_calls/manifest.json \
		--split youtube_scam_validation --truncation-side left \
		--dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.youtube-scam-validation.json \
		--predictions reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.youtube-scam-validation.predictions.jsonl

bothbosu-eval-schema18: scam-dialogue-holdout
	$(PYTHON_BIN) training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left \
		--data data/external/scam_dialogue/scam_dialogue_validation.jsonl \
		--manifest data/external/scam_dialogue/manifest.json \
		--split scam_dialogue_validation_latest_window --truncation-side left \
		--dialogue-policy speaker-neutral-v1 \
		--report reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.bothbosu-validation-left.json \
		--predictions reports/runs/sg-modernbert-schema18-action-pairx2-ret4-w2-m3-left.bothbosu-validation-left.predictions.jsonl

encoder-onnx-export:
	$(PYTHON_BIN) scripts/export_encoder_onnx.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--output-dir artifacts/onnx/schema13-dynamic-pack \
		--sequence-length 256 --dynamic-sequence --dialogue-policy speaker-neutral-v1

encoder-onnx-eval-fp32:
	$(PYTHON_BIN) training/eval_encoder_onnx.py \
		--onnx artifacts/onnx/schema13-dynamic-pack/scamguard-modernbert-seqdynamic-fp32.onnx \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--model-report reports/runs/sg-modernbert-schema13-dose16.json \
		--data-dir data/experiments/schema13-dose16/processed \
		--dev-predictions reports/runs/sg-modernbert-schema13-dose16.dev.predictions.jsonl \
		--test-predictions reports/runs/sg-modernbert-schema13-dose16.regression.predictions.jsonl \
		--sequence-length 256 --dynamic-sequence --threads 4 \
		--output reports/runs/sg-modernbert-schema13-seqdynamic-fp32-onnx.json

encoder-onnx-eval-int8:
	$(PYTHON_BIN) training/eval_encoder_onnx.py \
		--onnx artifacts/onnx/schema13-dynamic-pack/scamguard-modernbert-seqdynamic-int8.onnx \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--model-report reports/runs/sg-modernbert-schema13-dose16.json \
		--data-dir data/experiments/schema13-dose16/processed \
		--dev-predictions reports/runs/sg-modernbert-schema13-dose16.dev.predictions.jsonl \
		--test-predictions reports/runs/sg-modernbert-schema13-dose16.regression.predictions.jsonl \
		--sequence-length 256 --dynamic-sequence --threads 4 \
		--output reports/runs/sg-modernbert-schema13-seqdynamic-int8-onnx.json

encoder-coreml-export:
	$(PYTHON_BIN) scripts/export_encoder_coreml.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--output-dir artifacts/coreml/schema13-seq128-fp32 \
		--sequence-length 128 --precision fp32 --dialogue-policy speaker-neutral-v1

encoder-coreml-eval:
	$(PYTHON_BIN) training/eval_encoder_coreml.py \
		--package artifacts/coreml/schema13-seq128-fp32/scamguard-modernbert-seq128-fp32.mlpackage \
		--manifest artifacts/coreml/schema13-seq128-fp32/manifest.json \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema13-dose16 \
		--model-report reports/runs/sg-modernbert-schema13-dose16.json \
		--data-dir data/experiments/schema13-dose16/processed \
		--dev-predictions reports/runs/sg-modernbert-schema13-dose16.dev.predictions.jsonl \
		--test-predictions reports/runs/sg-modernbert-schema13-dose16.regression.predictions.jsonl \
		--output reports/runs/sg-modernbert-schema13-seq128-fp32-coreml.json

encoder-dialogue-base: scam-dialogue-holdout
	uv run --extra neural python training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema9-safety \
		--data data/external/scam_dialogue/scam_dialogue_validation.jsonl \
		--manifest data/external/scam_dialogue/manifest.json \
		--split scam_dialogue_validation --require-mps \
		--report reports/runs/sg-modernbert-schema9-safety.dialogue-validation.json

encoder-dialogue-large: scam-dialogue-holdout
	uv run --extra neural python training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-large-schema9-safety \
		--data data/external/scam_dialogue/scam_dialogue_validation.jsonl \
		--manifest data/external/scam_dialogue/manifest.json \
		--split scam_dialogue_validation --batch-size 16 --require-mps \
		--report reports/runs/sg-modernbert-large-schema9-safety.dialogue-validation.json

encoder-taskmaster-base: taskmaster-dialogues
	uv run --extra neural python training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-schema9-safety \
		--data data/external/taskmaster/taskmaster_validation.jsonl \
		--manifest data/external/taskmaster/manifest.json \
		--split taskmaster_validation --require-mps \
		--report reports/runs/sg-modernbert-schema9-safety.taskmaster-validation.json

encoder-taskmaster-large: taskmaster-dialogues
	uv run --extra neural python training/eval_encoder_external.py \
		--checkpoint artifacts/checkpoints/sg-modernbert-large-schema9-safety \
		--data data/external/taskmaster/taskmaster_validation.jsonl \
		--manifest data/external/taskmaster/manifest.json \
		--split taskmaster_validation --batch-size 16 --require-mps \
		--report reports/runs/sg-modernbert-large-schema9-safety.taskmaster-validation.json

reference: data
	uv run --extra train --extra neural python benchmarks/eval_deberta_reference.py --device cpu

forum-learning-data: data
	uv run --extra train python scripts/build_dataset.py --output data/learning_curve/forum-0 \
		--forum-train-scam-limit 0 --forum-train-uncertain-limit 0
	uv run --extra train python scripts/build_dataset.py --output data/learning_curve/forum-1000 \
		--forum-train-scam-limit 1000 --forum-train-uncertain-limit 100
	uv run --extra train python scripts/build_dataset.py --output data/learning_curve/forum-3000 \
		--forum-train-scam-limit 3000 --forum-train-uncertain-limit 300
	uv run --extra train python scripts/build_dataset.py --output data/learning_curve/forum-5672 \
		--forum-train-scam-limit 5672 --forum-train-uncertain-limit 600

forum-learning-curve: forum-learning-data
	uv run --extra train python benchmarks/forum_learning_curve.py

qwen-08b-base-gguf:
	uv run --extra train python scripts/fetch_qwen08_base_gguf.py

qwen-08b-base-gguf-benchmark: qwen-08b-base-gguf
	$(PYTHON_BIN) benchmarks/benchmark_gguf_runtime.py \
		--llama-bench "$(LLAMA_BENCH)" --llama-source "$(LLAMA_CPP_DIR)" \
		--model "$(QWEN08_BASE_GGUF)" \
		--expected-model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
		--report reports/runs/qwen35-08b-upstream-q4-runtime.json

qwen-08b-base-gguf-verdict-benchmark: qwen-08b-base-gguf
	uv run --extra train --extra qwen python benchmarks/benchmark_gguf_verdict_latency.py \
		--llama-perplexity "$(LLAMA_PERPLEXITY)" --model "$(QWEN08_BASE_GGUF)" \
		--expected-model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
		--data "$(QWEN08_FULL_DATA)" \
		--processor-revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--ctx-size 256 \
		--report reports/runs/qwen35-08b-upstream-q4-verdict-latency-ctx256.json
	uv run --extra train --extra qwen python benchmarks/benchmark_gguf_verdict_latency.py \
		--llama-perplexity "$(LLAMA_PERPLEXITY)" --model "$(QWEN08_BASE_GGUF)" \
		--expected-model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
		--data "$(QWEN08_FULL_DATA)" \
		--processor-revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--ctx-size 640 \
		--report reports/runs/qwen35-08b-upstream-q4-verdict-latency-ctx640.json

qwen-08b-base-native-gguf-prefix-benchmark: qwen-08b-base-gguf gguf-verdict-runner
	uv run --extra train --extra qwen python benchmarks/benchmark_native_gguf_prefix.py \
		--runner "$(GGUF_VERDICT_RUNNER)" \
		--runner-sha256 "$$(shasum -a 256 "$(GGUF_VERDICT_RUNNER)" | awk '{print $$1}')" \
		--model "$(QWEN08_BASE_GGUF)" \
		--model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
		--processor Qwen/Qwen3.5-0.8B \
		--processor-revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--calibration-report "$(QWEN08_BASE_REPORT)" --data "$(QWEN08_FULL_DATA)" \
		--rows 50 --repetitions 3 --ctx-size 640 --batch-size 640 \
		--ubatch-size 128 --threads 4 --n-gpu-layers 99 \
		--report "$(QWEN08_BASE_NATIVE_PREFIX_REPORT)"

qwen-data: data
	uv run --extra train python training/build_qwen_sft.py

qwen-08b-training-preflight:
	uv run --extra train --extra qwen python scripts/preflight_qwen08_training.py \
		--local-files-only --output "$(QWEN08_TRAINING_PREFLIGHT_REPORT)"

qwen-08b-batch-preflight:
	uv run --extra train --extra qwen python scripts/preflight_qwen08_batch.py \
		--batch-size 4 --sequence-length 640 --gradient-accumulation 4 \
		--local-files-only \
		--output "$(QWEN08_BATCH_PREFLIGHT_REPORT)"

qwen-token-audit: qwen-data
	uv run --extra train --extra qwen python scripts/audit_qwen_tokens.py \
		--revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
		--output reports/runs/qwen-token-audit-schema9.json

qwen-08b: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-08b-lora.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--experiment-config configs/qwen35-08b-lora.json \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--batch-size 16 --eval-batch-size 4 \
		--gradient-accumulation 1 --gradient-checkpointing \
		--sampling-strategy group_by_length --require-mps \
		--output artifacts/checkpoints/qwen35-08b-schema6-lora

qwen-08b-full-data:
	test -f "$(QWEN08_FULL_DATA)/manifest.json"
	uv run --extra train python training/build_qwen_sft.py \
		--data "$(QWEN08_FULL_DATA)" --output "$(QWEN08_FULL_DATA)/qwen_sft"

qwen-08b-full-token-audit: qwen-08b-full-data
	uv run --extra train --extra qwen python scripts/audit_qwen_tokens.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--data "$(QWEN08_FULL_DATA)/qwen_sft" --max-length 640 \
		--output "$(QWEN08_FULL_TOKEN_AUDIT)"

qwen-08b-full-freeze: qwen-08b-full-token-audit schema24-audit-check
	@if [ -f "$(QWEN08_FULL_CONFIG)" ]; then \
		echo "Reusing immutable full-run config: $(QWEN08_FULL_CONFIG)"; \
	else \
		uv run --extra train --extra qwen python scripts/freeze_qwen08_full_experiment.py \
			--processed "$(QWEN08_FULL_DATA)" --token-audit "$(QWEN08_FULL_TOKEN_AUDIT)" \
			--label-audit "$(QWEN08_FULL_LABEL_AUDIT)" \
			--batch-selection reports/QWEN08_BATCH_GEOMETRY_SELECTION.json \
			--output "$(QWEN08_FULL_CONFIG)" --checkpoint-output "$(QWEN08_FULL_OUTPUT)" \
			--experiment-id sg-qwen35-08b-schema24-full-v1; \
	fi

qwen-08b-full:
	test -f "$(QWEN08_FULL_CONFIG)"
	uv run python scripts/verify_experiment_config.py \
		--config "$(QWEN08_FULL_CONFIG)" --data "$(QWEN08_FULL_DATA)"
	@if [ -f "$(QWEN08_FULL_OUTPUT)/adapter_model.safetensors" ]; then \
		echo "Reusing immutable full-run adapter: $(QWEN08_FULL_OUTPUT)"; \
	else \
		uv run --extra train --extra qwen python training/train_qwen_lora.py \
			--experiment-config "$(QWEN08_FULL_CONFIG)" \
			--model Qwen/Qwen3.5-0.8B \
			--revision 2fc06364715b967f1860aea9cf38778875588b17 \
			--data "$(QWEN08_FULL_DATA)/qwen_sft" \
			--batch-size 4 --eval-batch-size 4 \
			--gradient-accumulation 4 --gradient-checkpointing \
			--max-length 640 \
			--sampling-strategy group_by_length --require-mps \
			--output "$(QWEN08_FULL_OUTPUT)"; \
	fi

qwen-08b-full-eval: qwen-08b-full
	uv run --extra train --extra qwen python training/eval_qwen.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--adapter "$(QWEN08_FULL_OUTPUT)" --data "$(QWEN08_FULL_DATA)" \
		--external-data data/external --splits $(QWEN08_FULL_EVAL_SPLITS) \
		--batch-size 1 --sequence-bucket-size 64 --require-mps \
		--report "$(QWEN08_FULL_REPORT)"

qwen-08b-base-product-eval:
	uv run --extra train --extra qwen python training/eval_qwen.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--data "$(QWEN08_FULL_DATA)" --external-data data/external \
		--splits dev test multidogo_annotation_dev multidogo_annotation_test \
		--batch-size 1 --sequence-bucket-size 64 --require-mps \
		--report "$(QWEN08_BASE_REPORT)"

qwen-08b-full-gates: qwen-08b-full-eval
	$(PYTHON_BIN) scripts/check_qwen08_full_gates.py \
		--report "$(QWEN08_FULL_REPORT)" --output "$(QWEN08_FULL_GATE_REPORT)"

qwen-08b-full-merge: qwen-08b-full-gates
	uv run --extra train --extra qwen python training/merge_qwen_adapter.py \
		--base Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--adapter "$(QWEN08_FULL_OUTPUT)" --device cpu --dtype float32 \
		--output "$(QWEN08_FULL_MERGED)"

qwen-08b-full-gguf:
	test -f "$(QWEN08_FULL_MERGED)/scamguard_merge.json"
	test -n "$(LLAMA_CPP_DIR)"
	scripts/export_gguf.sh "$(LLAMA_CPP_DIR)" "$(QWEN08_FULL_MERGED)" \
		"$(QWEN08_FULL_GGUF_PREFIX)" "$(PYTHON_BIN)"

qwen-08b-full-gguf-eval-q4:
	test -f "$(QWEN08_FULL_Q4_GGUF)"
	uv run --extra train --extra qwen python training/eval_gguf.py \
		--model "$(QWEN08_FULL_Q4_GGUF)" --processor "$(QWEN08_FULL_OUTPUT)" \
		--calibration "$(QWEN08_FULL_OUTPUT)/scamguard_calibration.json" \
		--llama-perplexity "$(LLAMA_PERPLEXITY)" --data "$(QWEN08_FULL_DATA)" \
		--external-data data/external --splits $(QWEN08_FULL_EVAL_SPLITS) \
		--ctx-size 640 --batch-size 640 --ubatch-size 128 --parallel 1 \
		--reference-predictions "$(QWEN08_FULL_REPORT:.json=.predictions.jsonl)" \
		--report "$(QWEN08_FULL_Q4_REPORT)"

qwen-08b-full-gguf-eval-q5:
	test -f "$(QWEN08_FULL_Q5_GGUF)"
	uv run --extra train --extra qwen python training/eval_gguf.py \
		--model "$(QWEN08_FULL_Q5_GGUF)" --processor "$(QWEN08_FULL_OUTPUT)" \
		--calibration "$(QWEN08_FULL_OUTPUT)/scamguard_calibration.json" \
		--llama-perplexity "$(LLAMA_PERPLEXITY)" --data "$(QWEN08_FULL_DATA)" \
		--external-data data/external --splits $(QWEN08_FULL_EVAL_SPLITS) \
		--ctx-size 640 --batch-size 640 --ubatch-size 128 --parallel 1 \
		--reference-predictions "$(QWEN08_FULL_REPORT:.json=.predictions.jsonl)" \
		--report "$(QWEN08_FULL_Q5_REPORT)"

gguf-verdict-runner:
	test -n "$(LLAMA_CPP_DIR)"
	scripts/build_gguf_verdict_runner.sh "$(LLAMA_CPP_DIR)" "$(GGUF_VERDICT_RUNNER_BUILD)"
	test -x "$(GGUF_VERDICT_RUNNER)"

portable-gguf-verdict-runner:
	test -n "$(LLAMA_CPP_DIR)"
	scripts/build_gguf_verdict_runner.sh "$(LLAMA_CPP_DIR)" "$(PORTABLE_GGUF_RUNNER_BUILD)"
	test -x "$(PORTABLE_GGUF_RUNNER)"

gguf-runtime-pack: portable-gguf-verdict-runner
	test -n "$(GGUF_RUNTIME_PACK_MODEL)"
	test -n "$(GGUF_RUNTIME_PACK_CALIBRATION_SOURCE)"
	uv run --extra train --extra qwen python scripts/build_gguf_runtime_pack.py \
		--model "$(GGUF_RUNTIME_PACK_MODEL)" --runner "$(PORTABLE_GGUF_RUNNER)" \
		--calibration-source "$(GGUF_RUNTIME_PACK_CALIBRATION_SOURCE)" \
		--purpose "$(GGUF_RUNTIME_PACK_PURPOSE)" --output "$(GGUF_RUNTIME_PACK_OUTPUT)"

qwen-08b-base-runtime-pack:
	@if [ -f "$(QWEN08_BASE_RUNTIME_PACK)/scamguard_gguf_pack.json" ]; then \
		echo "Reusing immutable upstream runtime control pack"; \
	else \
		$(MAKE) gguf-runtime-pack \
			GGUF_RUNTIME_PACK_MODEL="$(QWEN08_BASE_GGUF)" \
			GGUF_RUNTIME_PACK_CALIBRATION_SOURCE="$(QWEN08_BASE_REPORT)" \
			GGUF_RUNTIME_PACK_PURPOSE=upstream_base_control \
			GGUF_RUNTIME_PACK_OUTPUT="$(QWEN08_BASE_RUNTIME_PACK)"; \
	fi

qwen-08b-base-runtime-pack-benchmark: qwen-08b-base-runtime-pack
	uv run python benchmarks/benchmark_gguf_runtime_pack.py \
		--pack "$(QWEN08_BASE_RUNTIME_PACK)" --data "$(QWEN08_FULL_DATA)" \
		--split test --rows 50 --repetitions 3 \
		--report "$(QWEN08_BASE_RUNTIME_PACK_REPORT)"

qwen-08b-full-gguf-routed-diagnostic-q4: qwen-08b-full-gguf-eval-q4
	$(MAKE) routed-eval \
		SPECIALIST_PREDICTIONS="$(QWEN08_FULL_Q4_REPORT:.json=.predictions.jsonl)" \
		ROUTED_REPORT="$(QWEN08_FULL_Q4_ROUTED_REPORT)"

qwen-08b-full-gguf-routed-diagnostic-q5: qwen-08b-full-gguf-eval-q5
	$(MAKE) routed-eval \
		SPECIALIST_PREDICTIONS="$(QWEN08_FULL_Q5_REPORT:.json=.predictions.jsonl)" \
		ROUTED_REPORT="$(QWEN08_FULL_Q5_ROUTED_REPORT)"

qwen-08b-full-gguf-routed-runtime-q4: qwen-08b-full-gguf-routed-diagnostic-q4 gguf-verdict-runner
	uv run --extra train --extra qwen python benchmarks/benchmark_routed_gguf_runtime.py \
		--router-checkpoint "$(ENCODER23_OUTPUT)" \
		--router-predictions "$(ENCODER23_ROUTER_PREDICTIONS)" \
		--runner "$(GGUF_VERDICT_RUNNER)" \
		--runner-sha256 "$$(shasum -a 256 "$(GGUF_VERDICT_RUNNER)" | awk '{print $$1}')" \
		--llama-source "$(LLAMA_CPP_DIR)" --llama-revision "$(LLAMA_CPP_REVISION)" \
		--model "$(QWEN08_FULL_Q4_GGUF)" \
		--model-sha256 "$$(shasum -a 256 "$(QWEN08_FULL_Q4_GGUF)" | awk '{print $$1}')" \
		--quantization Q4_K_M --processor "$(QWEN08_FULL_OUTPUT)" \
		--calibration "$(QWEN08_FULL_OUTPUT)/scamguard_calibration.json" \
		--bf16-report "$(QWEN08_FULL_REPORT)" --specialist-report "$(QWEN08_FULL_Q4_REPORT)" \
		--specialist-predictions "$(QWEN08_FULL_Q4_REPORT:.json=.predictions.jsonl)" \
		--routed-report "$(QWEN08_FULL_Q4_ROUTED_REPORT)" --data "$(QWEN08_FULL_DATA)" \
		--repetitions "$(ROUTED_RUNTIME_REPETITIONS)" --ctx-size 640 --batch-size 640 \
		--ubatch-size 128 --threads 4 --n-gpu-layers 99 --require-mps \
		--report "$(QWEN08_FULL_Q4_ROUTED_RUNTIME_REPORT)"

qwen-08b-full-gguf-routed-runtime-q5: qwen-08b-full-gguf-routed-diagnostic-q5 gguf-verdict-runner
	uv run --extra train --extra qwen python benchmarks/benchmark_routed_gguf_runtime.py \
		--router-checkpoint "$(ENCODER23_OUTPUT)" \
		--router-predictions "$(ENCODER23_ROUTER_PREDICTIONS)" \
		--runner "$(GGUF_VERDICT_RUNNER)" \
		--runner-sha256 "$$(shasum -a 256 "$(GGUF_VERDICT_RUNNER)" | awk '{print $$1}')" \
		--llama-source "$(LLAMA_CPP_DIR)" --llama-revision "$(LLAMA_CPP_REVISION)" \
		--model "$(QWEN08_FULL_Q5_GGUF)" \
		--model-sha256 "$$(shasum -a 256 "$(QWEN08_FULL_Q5_GGUF)" | awk '{print $$1}')" \
		--quantization Q5_K_M --processor "$(QWEN08_FULL_OUTPUT)" \
		--calibration "$(QWEN08_FULL_OUTPUT)/scamguard_calibration.json" \
		--bf16-report "$(QWEN08_FULL_REPORT)" --specialist-report "$(QWEN08_FULL_Q5_REPORT)" \
		--specialist-predictions "$(QWEN08_FULL_Q5_REPORT:.json=.predictions.jsonl)" \
		--routed-report "$(QWEN08_FULL_Q5_ROUTED_REPORT)" --data "$(QWEN08_FULL_DATA)" \
		--repetitions "$(ROUTED_RUNTIME_REPETITIONS)" --ctx-size 640 --batch-size 640 \
		--ubatch-size 128 --threads 4 --n-gpu-layers 99 --require-mps \
		--report "$(QWEN08_FULL_Q5_ROUTED_RUNTIME_REPORT)"

qwen-2b: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-2b-lora.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--experiment-config configs/qwen35-2b-lora.json \
		--model Qwen/Qwen3.5-2B \
		--revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
		--batch-size 16 --eval-batch-size 4 \
		--gradient-accumulation 1 --gradient-checkpointing \
		--sampling-strategy group_by_length --skip-eval --require-mps \
		--output artifacts/checkpoints/qwen35-2b-schema6-lora

qwen-4b: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-4b-lora.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--experiment-config configs/qwen35-4b-lora.json \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--batch-size 8 --eval-batch-size 2 \
		--gradient-accumulation 2 --gradient-checkpointing \
		--sampling-strategy group_by_length --skip-eval --require-mps \
		--output artifacts/checkpoints/qwen35-4b-schema6-lora

qwen-4b-schema9: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-4b-lora-schema9.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--experiment-config configs/qwen35-4b-lora-schema9.json \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--batch-size 8 --eval-batch-size 2 \
		--gradient-accumulation 2 --gradient-checkpointing \
		--sampling-strategy group_by_length --skip-eval --require-mps \
		--output artifacts/checkpoints/qwen35-4b-schema9-lora

qwen-batch-benchmark:
	uv run --extra train --extra qwen python benchmarks/qwen_batch_benchmark.py \
		--model Qwen/Qwen3.5-2B \
		--revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
		--adapter artifacts/checkpoints/qwen35-2b-schema6-lora \
		--batch-sizes 1 2 4 8 --require-mps \
		--report reports/runs/qwen35-2b-batch-benchmark.json

qwen-4b-batch-benchmark:
	uv run --extra train --extra qwen python benchmarks/qwen_batch_benchmark.py \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--adapter artifacts/checkpoints/qwen35-4b-schema6-lora \
		--batch-sizes 1 2 4 8 --require-mps \
		--report reports/runs/qwen35-4b-batch-benchmark.json

qwen-eval:
	uv run --extra train --extra qwen python training/eval_qwen.py \
		--model Qwen/Qwen3.5-2B \
		--revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
		--adapter artifacts/checkpoints/qwen35-2b-schema6-lora \
		--batch-size 1 --require-mps \
		--report reports/runs/qwen35-2b-schema6-lora.json

qwen-4b-core-eval:
	uv run --extra train --extra qwen python training/eval_qwen.py \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--adapter artifacts/checkpoints/qwen35-4b-schema6-lora \
		--batch-size 1 --splits dev test --require-mps \
		--cache-dir reports/runs/qwen35-4b-schema6-lora.scores \
		--report reports/runs/qwen35-4b-schema6-lora.core.json

qwen-4b-eval:
	uv run --extra train --extra qwen python training/eval_qwen.py \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--adapter artifacts/checkpoints/qwen35-4b-schema6-lora \
		--batch-size 1 --require-mps \
		--cache-dir reports/runs/qwen35-4b-schema6-lora.scores \
		--report reports/runs/qwen35-4b-schema6-lora.json

qwen-generation:
	uv run --extra train --extra qwen python training/eval_qwen_generation.py \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--adapter artifacts/checkpoints/qwen35-4b-schema6-lora \
		--require-mps \
		--report reports/runs/qwen35-4b-generation.json

qwen-error-audit:
	uv run python scripts/create_error_audit.py \
		--predictions reports/runs/qwen35-4b-schema6-lora.predictions.jsonl \
		--output reports/audits/qwen35-4b-schema6-lora-errors.csv

qwen-08b-ai-internal-errors:
	.venv/bin/python scripts/summarize_qwen_errors.py \
		--predictions reports/runs/qwen35-08b-schema24-ai-internal-branch.predictions.jsonl \
		--output reports/runs/qwen35-08b-schema24-ai-internal-branch-errors.json

paired:
	uv run python benchmarks/compare_paired.py \
		--candidate reports/runs/qwen35-4b-schema6-lora.predictions.jsonl \
		--reference reports/runs/deberta-v022.predictions.jsonl \
		--output reports/runs/qwen35-4b-vs-deberta-v022-paired.json

qwen-merge:
	uv run --extra train --extra qwen python training/merge_qwen_adapter.py \
		--base Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--adapter artifacts/checkpoints/qwen35-4b-schema6-lora \
		--device cpu --dtype float32 \
		--output artifacts/merged/qwen35-4b-scamguard

qwen-gguf: qwen-merge
	test -n "$(LLAMA_CPP_DIR)"
	mkdir -p artifacts/gguf
	scripts/export_gguf.sh "$(LLAMA_CPP_DIR)" artifacts/merged/qwen35-4b-scamguard \
		artifacts/gguf/scamguard-qwen35-4b "$(PYTHON_BIN)"

qwen-gguf-eval:
	test -n "$(LLAMA_CPP_DIR)"
	uv run --extra train --extra qwen python training/eval_gguf.py \
		--model artifacts/gguf/scamguard-qwen35-4b-q4_k_m.gguf \
		--processor artifacts/checkpoints/qwen35-4b-schema6-lora \
		--calibration artifacts/checkpoints/qwen35-4b-schema6-lora/scamguard_calibration.json \
		--llama-perplexity "$(LLAMA_CPP_DIR)/build-arm64/bin/llama-perplexity" \
		--report reports/runs/qwen35-4b-q4-k-m.json

huggingface-release-check:
	test -n "$(HF_RELEASE_MANIFEST)"
	$(PYTHON_BIN) scripts/verify_huggingface_release.py \
		--manifest "$(HF_RELEASE_MANIFEST)" --repo-root "$(CURDIR)"

demo:
	uv run scamguard demo --model artifacts/sg-linear-v0.3.joblib
