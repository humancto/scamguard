.PHONY: install test lint fetch data youtube-scam-calls apptek-callcenter harper-valley multidogo multidogo-annotations schema13-dose16 schema14-natural-dialogue schema15-legitimate-openings schema17-call-minimal-pairs schema18-call-evidence-pairs schema19-call-windows schema20-action-states schema21-human-calls schema22-service-evidence schema23-evidence-compaction encoder-schema13-dose16 encoder-schema14-natural-dialogue encoder-schema15-legitimate-openings encoder-schema16-cache encoder-schema16-preflight encoder-schema16-retention encoder-schema17-preflight encoder-schema17-pair-retention encoder-schema18-preflight encoder-schema18-action encoder-schema19-preflight encoder-schema19-windowmix encoder-schema20-preflight encoder-schema20-actionheads encoder-schema21-preflight encoder-schema21-human-calls encoder-schema22-preflight encoder-schema22-service-evidence encoder-schema22-gates encoder-schema23-cache encoder-schema23-preflight encoder-schema23-evidence-compaction encoder-schema23-gates apptek-eval-schema13 apptek-eval-schema14 apptek-eval-schema15 apptek-eval-schema16 apptek-eval-schema17 apptek-eval-schema18 youtube-eval-schema16 youtube-eval-schema17 youtube-eval-schema18 bothbosu-eval-schema18 encoder-onnx-export encoder-onnx-eval-fp32 encoder-onnx-eval-int8 encoder-coreml-export encoder-coreml-eval teleantifraud-fetch teleantifraud-audit fresh-holdout chichewa-holdout scam-dialogue-holdout taskmaster-dialogues audit audit-check baseline encoder encoder-large encoder-schema12 encoder-dialogue-base encoder-dialogue-large encoder-taskmaster-base encoder-taskmaster-large reference forum-learning-data forum-learning-curve qwen-data qwen-token-audit qwen-08b qwen-2b qwen-4b qwen-4b-schema9 qwen-batch-benchmark qwen-4b-batch-benchmark qwen-eval qwen-4b-core-eval qwen-4b-eval qwen-generation qwen-error-audit paired qwen-merge qwen-gguf qwen-gguf-eval huggingface-release-check demo

PYTHON_BIN ?= .venv/bin/python

install:
	uv sync --extra train --extra dev

test:
	uv run pytest

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

qwen-data: data
	uv run --extra train python training/build_qwen_sft.py

qwen-token-audit: qwen-data
	uv run --extra train --extra qwen python scripts/audit_qwen_tokens.py \
		--revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
		--output reports/runs/qwen-token-audit-schema9.json

qwen-08b: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-08b-lora.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--model Qwen/Qwen3.5-0.8B \
		--revision 2fc06364715b967f1860aea9cf38778875588b17 \
		--batch-size 16 --eval-batch-size 4 \
		--gradient-accumulation 1 --gradient-checkpointing \
		--sampling-strategy group_by_length --require-mps \
		--output artifacts/checkpoints/qwen35-08b-schema6-lora

qwen-2b: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-2b-lora.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--model Qwen/Qwen3.5-2B \
		--revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
		--batch-size 16 --eval-batch-size 4 \
		--gradient-accumulation 1 --gradient-checkpointing \
		--sampling-strategy group_by_length --skip-eval --require-mps \
		--output artifacts/checkpoints/qwen35-2b-schema6-lora

qwen-4b: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-4b-lora.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
		--model Qwen/Qwen3.5-4B \
		--revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
		--batch-size 8 --eval-batch-size 2 \
		--gradient-accumulation 2 --gradient-checkpointing \
		--sampling-strategy group_by_length --skip-eval --require-mps \
		--output artifacts/checkpoints/qwen35-4b-schema6-lora

qwen-4b-schema9: qwen-data
	uv run python scripts/verify_experiment_config.py --config configs/qwen35-4b-lora-schema9.json
	uv run --extra train --extra qwen python training/train_qwen_lora.py \
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
