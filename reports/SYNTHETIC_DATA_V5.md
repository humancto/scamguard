# Synthetic data v5 methodology

Freeze date: 2026-08-21. ScamGuard synthetic v5 contains 8,064 Apache-2.0 rows: 5,040
English base-family variants and 3,024 multilingual benign lookalikes. The v5 increment adds 1,728
training-only rows in 12 scam families and 12 paired legitimate families. Development, regression,
OOD, and the sealed schema-v8 source receive no new synthetic message families.

## What changed and why

The first 149M encoder missed an independent identity-impersonation development family and showed
that the original scenario set did not span enough scam mechanics. The v5 increment targets the
gap with original messages grounded in these authoritative descriptions:

| Mechanics represented | Primary reference |
|---|---|
| government benefit, Medicare, jury-duty, and immigration impersonation | [FTC government impersonation guidance](https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam) |
| layered tech/bank/government “safe account” transfers | [FBI IC3 Phantom Hacker PSA](https://www.ic3.gov/PSA/2023/PSA230929) |
| “move money to protect it” and verification-code requests | [FTC safe-account warning](https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam) |
| tax refund and identity lures | [IRS tax scam guidance](https://www.irs.gov/help/tax-scams) |
| gamified task deposits | [FTC task-scam guidance](https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams) |
| family emergency and bail secrecy | [FTC family-emergency guidance](https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money) |
| package-identity smishing | [USPIS package-tracking smishing guidance](https://www.uspis.gov/news/scam-article/smishing-package-tracking-text-scams) |
| reshipping jobs | [USPIS reshipping scam guidance](https://www.uspis.gov/wp-content/uploads/2021/08/uspis-be-smart-reshipping-scams-handout_508.pdf) |

The source pages define threat mechanics and legitimate-channel safeguards. They are not scraped
as examples, and their prose is not copied. `scripts/generate_synthetic.py` creates original copy
from deterministic scenario grammars and records both
`synthetic_method=deterministic_slot_filling_original_copy` and the relevant `pattern_reference`
on each row.

## Counterfactual design

Each new risk area has a legitimate lookalike that shares topical vocabulary while reversing the
harmful action. Examples include a bank warning never to move money, an opted-in tax appointment
that asks for no information, a court reminder that directs the recipient to the official clerk,
and a legitimate job interview that requires no deposit. These pairs are designed to prevent
shortcuts such as “bank,” “tax,” “package,” or “verification code” automatically implying SCAM.

Families, not individual slot-filled messages, are the split unit. Every new family is assigned to
training only; no exact family can cross into development or evaluation. Existing development and
regression families remain independent, and the newly sourced 1,820-row MOZ-Smishing test remains
prediction-sealed and excluded from training.

## Fail-closed checks

The independent validator requires every synthetic row to have an approved Apache-2.0 license,
generation-method tag, authoritative HTTPS pattern reference, non-empty provenance, isolated family,
and no PII-like value. Every core SCAM row must also yield a verbatim deterministic evidence span.
The first v5 draft failed this last condition for 86 messages; the underlying scam requests were
made explicit and the build was rerun. The validator was not relaxed.

The accepted snapshot has 25,518 unique processed examples across core and diagnostics, no family
leakage, and zero unmasked PII-like values under the project rules. The synthetic artifact SHA-256
is `e9fbc2cd32cfc1f0115bca82231f3d19a9dab280a381c91566581d15253e00e6`;
its manifest SHA-256 is
`478de3cc87ea0a62441d1a8eb5db3a6e9d04771dce000869fa79840f1b9dd39a`.

## Limitations

Deterministic variants are controlled experiments, not a substitute for diverse real messages.
They can overrepresent polished wording and repeated structures. Model results must therefore be
reported separately on real-source, OOD, adversarial, multilingual, and synthetic slices. The
180-row audit workbook still needs independent human labels, and the multilingual benign families
still need native-speaker review. Until those gates close, v5 is an auditable research dataset—not
a human-certified or SOTA corpus claim.
