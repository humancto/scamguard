from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(
    "configs/encoder-schema23-evidencecompact-ret4-aw05-vw025-lr2e6-right.json"
)


def test_schema23_config_freezes_evidence_and_data_boundaries() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["predeclared_before_training"] is True
    assert config["training"]["dialogue_policy"] == (
        "speaker-neutral-evidence-recent-v2"
    )
    assert config["training"]["truncation_side"] == "right"
    assert config["data"]["direct_reddit_rows_scraped"] == 0
    assert config["data"]["external_transcript_text_copied"] is False
    assert config["data"]["bothbosu_rows_used_for_fitting_or_threshold"] == 0
    assert config["teacher"]["logit_scope"] == "first three verdict logits only"


def test_schema23_distillation_is_quality_gated() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["runtime_acceptance_after_quality_pass"][
        "distillation_and_quantization_require_all_quality_gates"
    ] is True
    assert "reject before external selection" in config["failure_policy"]
