from __future__ import annotations

from scripts.build_phone_scam_synthetic import one_per_near_family, source_row


def record(dialogue: str, label: int = 1) -> dict[str, object]:
    return {
        "dialogue": dialogue,
        "type": "support",
        "label": label,
        "length_category": "short",
        "variation_style": "direct" if label else "efficient_professional",
        "target_exchanges": "2-4",
        "batch_num": 1,
    }


def test_source_row_quarantines_generation_prompt_and_masks_access_code() -> None:
    assert source_row(record("Here is a legitimate phone call dialogue."), "train", 0) is None
    row = source_row(
        record("caller: Read me verification code 928104. receiver: I will not."),
        "train",
        1,
    )

    assert row is not None
    assert "928104" not in str(row["text"])
    assert row["metadata_used_as_model_input"] is False
    assert row["is_synthetic"] is True


def test_source_row_masks_publisher_generated_email_and_account_values() -> None:
    row = source_row(
        record(
            "caller: Confirm user@gmail.com and account 1234567890. receiver: No.",
            1,
        ),
        "train",
        2,
    )

    assert row is not None
    assert "user@gmail.com" not in str(row["text"])
    assert "1234567890" not in str(row["text"])
    assert row["broad_privacy_values_replaced"] is True


def test_near_family_control_reserves_test_and_quarantines_label_conflict() -> None:
    test_row = source_row(record("caller: Pay by gift card now.", 0), "test", 0)
    train_duplicate = source_row(record("caller: Pay by gift card now.", 1), "train", 1)
    safe = source_row(record("caller: Your appointment is tomorrow.", 0), "train", 2)
    assert test_row is not None and train_duplicate is not None and safe is not None

    retained, report = one_per_near_family(
        {"train": [train_duplicate, safe], "validation": [], "test": [test_row]}
    )

    assert report["mixed_label_families_quarantined"] == 1
    assert retained["test"] == []
    assert [row["id"] for row in retained["train"]] == [safe["id"]]


def test_near_family_control_keeps_highest_priority_split_for_same_label() -> None:
    held = source_row(record("caller: Open the official app.", 0), "test", 0)
    fit = source_row(record("caller: Open the official app.", 0), "train", 1)
    assert held is not None and fit is not None

    retained, report = one_per_near_family({"train": [fit], "validation": [], "test": [held]})

    assert report["cross_split_near_families"] == 1
    assert len(retained["test"]) == 1
    assert retained["train"] == []
