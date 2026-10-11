import copy
import json

import pytest

from scamguard.published_qwen_tokenizer import assert_serialized_backend, load_serialized_tokenizer


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "added_tokens",
    ],
)
def test_serialized_contract_rejects_any_backend_component_drift(field: str) -> None:
    original = dict.fromkeys(
        [
            "model",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "added_tokens",
        ]
    )
    actual = copy.deepcopy(original)
    assert_serialized_backend(original, actual)
    actual[field] = {"changed": True}
    with pytest.raises(ValueError, match=field):
        assert_serialized_backend(original, actual)


def test_generic_loader_preserves_normalization_and_ids(tmp_path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"café": 0, "[UNK]": 1}, "[UNK]"))
    backend.normalizer = tokenizers.normalizers.NFC()
    backend.post_processor = tokenizers.processors.TemplateProcessing(single="$A", pair="$A $B:1")
    tokenizer_file = tmp_path / "tokenizer.json"
    backend.save(str(tokenizer_file))
    config_file = tmp_path / "tokenizer_config.json"
    config_file.write_text(json.dumps({"model_max_length": 128}), encoding="utf-8")
    template_file = tmp_path / "chat_template.jinja"
    template_file.write_text("{{ messages[0]['content'] }}", encoding="utf-8")
    tokenizer = load_serialized_tokenizer(tokenizer_file, config_file, template_file)
    assert tokenizer.encode("cafe\u0301", add_special_tokens=False) == [0]
    assert tokenizer.encode("café", add_special_tokens=False) == [0]
    assert (
        tokenizer.apply_chat_template([{"role": "user", "content": "cafe\u0301"}], tokenize=False)
        == "cafe\u0301"
    )
