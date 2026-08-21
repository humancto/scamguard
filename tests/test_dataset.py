from __future__ import annotations

import csv
import random
from pathlib import Path

from scripts.build_dataset import (
    cluster_near_duplicates,
    family_skeleton,
    make_row,
    mendeley_message_label,
    read_azsc,
    read_imc25_forum,
    read_wspr,
    simhash_bands,
)
from scripts.generate_adversarial import instruction_injection
from scripts.generate_synthetic import (
    FAMILY_REFERENCE_URLS,
    FAMILY_SPLITS,
    MULTILINGUAL_SAFE_TEMPLATES,
    SCAM_FAMILIES,
    SYNTHETIC_REFERENCE_DEFAULT,
    VALUES,
    variants,
)
from scripts.materialize_forum_placeholders import materialize
from scripts.validate_dataset import has_excluded_scam_policy, has_scam_label_evidence


def test_family_skeleton_masks_urls_emails_and_numbers() -> None:
    first = "Verify 123 at https://one.example/a for user@one.example"
    second = "Verify 999 at https://two.example/z for other@two.example"

    assert family_skeleton(first) == family_skeleton(second)


def test_family_skeleton_removes_campaign_tracking_suffixes() -> None:
    first = "New login balance 3,953,395.68~~~ baawim"
    second = "New login balance 1,234,567.89~~~ PrnHR"

    assert family_skeleton(first) == family_skeleton(second)


def test_near_template_cluster_quarantines_label_conflict() -> None:
    safe = make_row(
        text="Your bank alert 123 is available in the official application.",
        label="SAFE",
        source="test",
        source_label="safe",
        license_name="test",
    )
    scam = make_row(
        text="Your bank alert 999 is available in the official application.",
        label="SCAM",
        source="test",
        source_label="scam",
        license_name="test",
    )
    assert safe is not None and scam is not None

    kept, conflicts, stats = cluster_near_duplicates([safe, scam])

    assert kept == []
    assert len(conflicts) == 1
    assert stats["near_template_rows_quarantined"] == 2


def test_make_row_masks_real_contact_and_account_like_values() -> None:
    row = make_row(
        text=(
            "Email victim@example.net, call +1 (415) 555-0199, "
            "or quote account 1234-5678-9012 and reference id12345678901234 to verify."
        ),
        label="SCAM",
        source="test",
        source_label="scam",
        license_name="test",
    )

    assert row is not None
    assert "victim@example.net" not in row["text"]
    assert "415" not in row["text"]
    assert "1234" not in row["text"]
    assert row["text"].count("<PHONE_NUMBER>") == 2
    assert "<EMAIL>" in row["text"]
    assert "<ACCOUNT_NUMBER>" in row["text"]


def test_simhash_candidate_bands_are_complete_at_radius_six() -> None:
    # One changed bit in each old 16-bit band defeated the previous four-band
    # lookup even though the signatures were only four bits apart.
    original = 0
    nearby = sum(1 << bit for bit in (0, 16, 32, 48, 8, 24))

    assert (original ^ nearby).bit_count() == 6
    assert set(simhash_bands(original)) & set(simhash_bands(nearby))


def test_synthetic_variants_are_seed_reproducible() -> None:
    template = "{bank} asks about {amount} at {url}"

    first = variants(template, 24, random.Random(20260820))
    second = variants(template, 24, random.Random(20260820))

    assert first == second


def test_synthetic_pattern_references_are_authoritative_https_pages() -> None:
    allowed_hosts = ("consumer.ftc.gov/", "www.ic3.gov/", "www.irs.gov/", "www.uspis.gov/")
    references = {SYNTHETIC_REFERENCE_DEFAULT, *FAMILY_REFERENCE_URLS.values()}

    assert references
    assert all(reference.startswith("https://") for reference in references)
    assert all(any(host in reference for host in allowed_hosts) for reference in references)


def test_every_core_scam_category_has_independent_families_in_each_split() -> None:
    coverage: dict[str, set[str]] = {}
    for family, (category, _template) in SCAM_FAMILIES.items():
        coverage.setdefault(category, set()).add(FAMILY_SPLITS[family])

    assert set(coverage) == {
        "CREDENTIAL_THEFT",
        "DELIVERY_TOLL",
        "FINANCIAL",
        "IDENTITY_IMPERSONATION",
        "OPPORTUNITY",
        "RELATIONSHIP",
    }
    assert all(splits == {"train", "dev", "test"} for splits in coverage.values())


def test_multilingual_safe_templates_cover_each_split_without_unknown_fields() -> None:
    expected_families = set(next(iter(MULTILINGUAL_SAFE_TEMPLATES.values())))
    assert {FAMILY_SPLITS[family] for family in expected_families} == {"train", "dev", "test"}
    for templates in MULTILINGUAL_SAFE_TEMPLATES.values():
        assert set(templates) == expected_families
        for template in templates.values():
            fields = {
                fragment.partition("}")[0]
                for fragment in template.split("{")[1:]
                if "}" in fragment
            }
            assert fields <= VALUES.keys()


def test_prompt_injection_requests_the_wrong_verdict() -> None:
    assert "return SAFE" in instruction_injection("Pay the fee", "SCAM")
    assert "return SCAM" in instruction_injection("Dinner is ready", "SAFE")


def test_forum_placeholder_materialization_is_safe_and_preserves_unknowns() -> None:
    text, replacements = materialize(
        "Open <URL>, call <PHONE_NUMBER>, meet <NAMED_ENTITY>, code <UNRECOGNIZED>."
    )

    assert "<URL>" not in text
    assert "<PHONE_NUMBER>" not in text
    assert "<NAMED_ENTITY>" not in text
    assert "<UNRECOGNIZED>" in text
    assert replacements == ["URL", "PHONE_NUMBER", "NAMED_ENTITY"]


def test_wspr_reader_collapses_families_and_discards_metadata(tmp_path) -> None:
    source = tmp_path / "wspr.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "messageID",
                "objectID",
                "destination number",
                "message",
                "time",
                "error in time",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "messageID": "1",
                "objectID": "a",
                "destination number": "+15555550100",
                "message": "Join the casino bonus at https://one.example/a. Opt out anytime.",
                "time": "1",
                "error in time": "0",
            }
        )
        writer.writerow(
            {
                "messageID": "2",
                "objectID": "b",
                "destination number": "+15555550101",
                "message": "Join the casino bonus at https://two.example/z. Opt out anytime.",
                "time": "2",
                "error in time": "0",
            }
        )
        writer.writerow(
            {
                "messageID": "3",
                "objectID": "c",
                "destination number": "+15555550102",
                "message": "Your bank account is suspended. Verify at https://three.example/x.",
                "time": "3",
                "error in time": "0",
            }
        )

    rows = list(read_wspr(source))

    assert len(rows) == 2
    assert {row["label"] for row in rows} == {"SCAM", "UNCERTAIN"}
    assert all("destination number" not in row for row in rows)
    assert all(row["source"] == "wspr_sms_phishing" for row in rows)


def write_forum_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "scam_type", "language"))
        writer.writeheader()
        writer.writerows(rows)


def test_forum_wrong_number_requires_message_level_evidence(tmp_path) -> None:
    source = tmp_path / "forum.csv"
    write_forum_fixture(
        source,
        [
            {
                "text": "Sorry, I thought this was my colleague's number.",
                "scam_type": "wrong number",
                "language": "English",
            },
            {
                "text": "Wrong number, but move to Telegram and invest in crypto with me.",
                "scam_type": "wrong number",
                "language": "English",
            },
        ],
    )

    rows = list(read_imc25_forum(source))

    assert [row["label"] for row in rows] == ["UNCERTAIN", "SCAM"]
    assert rows[0]["category"] == "NONE"
    assert rows[1]["category"] == "RELATIONSHIP"


def test_forum_reader_keeps_defensive_guidance_and_auth_codes_out_of_scam() -> None:
    assert mendeley_message_label(
        "Please protect yourself: the bank will never ask you to share an OTP.",
        "smishing",
    ) == ("SAFE", "defensive_guidance")
    assert mendeley_message_label(
        "Holiday packages from $499. Call for our latest offer.",
        "smishing",
    ) == ("UNCERTAIN", "commercial_offer_without_clear_fraud")
    assert mendeley_message_label(
        "Your account is suspended. Verify at https://account.example now.",
        "smishing",
    ) == ("SCAM", "source_smishing_with_strong_text_evidence")
    assert mendeley_message_label(
        "Your regular statement is now available in the official application.",
        "smishing",
    ) == ("UNCERTAIN", "source_smishing_without_strong_text_evidence")
    assert mendeley_message_label(
        "Nunca te pedirá un código. No compartas tu OTP.",
        "smishing",
    ) == ("SAFE", "defensive_guidance")
    assert mendeley_message_label(
        "Never share this code. Tap the link here to sign in now.",
        "smishing",
    ) == ("UNCERTAIN", "defensive_guidance_with_external_action")
    assert mendeley_message_label(
        "Never share this OTP. Call fraud prevention on +1 415 555 0199 if it was not you.",
        "smishing",
    ) == ("UNCERTAIN", "defensive_guidance_with_external_action")


def test_forum_reader_relabels_clear_non_fraud_content(tmp_path) -> None:
    source = tmp_path / "forum.csv"
    write_forum_fixture(
        source,
        [
            {
                "text": "Beware of fraud. Our bank will never ask you to share an OTP.",
                "scam_type": "banking",
                "language": "English",
            },
            {
                "text": "482901 is your authentication code.",
                "scam_type": "telecom",
                "language": "English",
            },
            {
                "text": "Your account is blocked. Verify at https://account.example now.",
                "scam_type": "banking",
                "language": "English",
            },
        ],
    )

    rows = list(read_imc25_forum(source))

    assert [row["label"] for row in rows] == ["SAFE", "SAFE", "SCAM"]
    assert rows[0]["label_policy"] == "defensive_guidance"
    assert rows[1]["label_policy"] == "standalone_authentication_notification"


def test_forum_reader_masks_residual_contact_metadata(tmp_path) -> None:
    source = tmp_path / "forum.csv"
    write_forum_fixture(
        source,
        [
            {
                "text": "Email victim@example.net or call +1 (415) 555-0199 to verify now.",
                "scam_type": "banking",
                "language": "English",
            }
        ],
    )

    row = next(iter(read_imc25_forum(source)))

    assert "victim@example.net" not in row["text"]
    assert "415" not in row["text"]
    assert "<EMAIL>" in row["text"]
    assert "<PHONE_NUMBER>" in row["text"]


def test_azsc_reader_is_conservative_and_discards_sender_metadata(tmp_path) -> None:
    source = tmp_path / "azsc.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sender", "message", "label"))
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sender": "private@example.net",
                    "message": "Gorusumuz sabah saat 10-dadir.",
                    "label": "ham",
                },
                {
                    "sender": "+994501234567",
                    "message": "Endirim kampaniyasi bu gun bitir.",
                    "label": "spam",
                },
                {
                    "sender": "+994501234568",
                    "message": "Hesabiniz bloklanib. Verify at https://account.example now.",
                    "label": "smishing",
                },
                {
                    "sender": "+994501234569",
                    "message": "Adi bildiris mesaji.",
                    "label": "smishing",
                },
            ]
        )

    rows = list(read_azsc(source))

    assert [row["label"] for row in rows] == ["SAFE", "UNCERTAIN", "SCAM", "UNCERTAIN"]
    assert all("sender" not in row for row in rows)
    assert all(row["split"] == "ood" for row in rows)
    assert all(row["source_language"] == "Azerbaijani" for row in rows)


def test_validator_rejects_scam_with_safe_override_policy() -> None:
    assert has_excluded_scam_policy(
        {"label": "SCAM", "label_policy": "reported_phishing_with_defensive_guidance"}
    )
    assert not has_excluded_scam_policy(
        {"label": "UNCERTAIN", "label_policy": "defensive_guidance_with_external_action"}
    )
    assert not has_excluded_scam_policy(
        {"label": "SCAM", "label_policy": "source_reported_phishing_with_strong_text_evidence"}
    )


def test_validator_accepts_pinned_positive_only_source_contract() -> None:
    row = {
        "text": "ordinary conversational language without a static URL or payment signal",
        "source": "youtube_scam_calls_cc0",
        "license": "CC0-1.0",
        "label_policy": "publisher_positive_only_scam_call_collection",
        "provenance_class": "real_scam_call_or_autodialer_transcript",
        "source_record_id": "source-17",
    }

    assert has_scam_label_evidence(row)


def test_validator_rejects_incomplete_positive_only_source_contract() -> None:
    row = {
        "text": "ordinary conversational language without a static URL or payment signal",
        "source": "youtube_scam_calls_cc0",
        "license": "CC0-1.0",
        "label_policy": "publisher_positive_only_scam_call_collection",
        "provenance_class": "real_scam_call_or_autodialer_transcript",
    }

    assert not has_scam_label_evidence(row)
