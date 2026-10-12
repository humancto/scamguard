# Ettin training-data notice — provisional, release blocked

Checked 2026-10-11. This notice discloses the actual schema-25 training sources for Ettin v1 and
v2. It is not a new license for third-party content, an independent rights clearance, a completed
human review, or authorization to redistribute raw messages. Every source's final review is pending.
Exact revisions, artifact hashes, selected-row digests and status are in the
[provenance inventory](../configs/ettin150m-release-provenance.json).

## License separation

- Base: [JHU Ettin-150M](https://huggingface.co/jhu-clsp/ettin-encoder-150m) declares MIT at
  revision `45d08642849e5c5701b162671ac811b7654bfd9f`. Preserve its actual JHU copyright and
  permission notice in [Ettin-MIT.txt](../LICENSES/Ettin-MIT.txt).
- ScamGuard code and original template-copy contributions declare Apache-2.0; see
  [the repository license](../LICENSE). This does not relicense external data or erase base notices.
- A final trained-weight license has not been cleared by this notice. Data, modified data,
  computational results and code can have different obligations. Do not equate a permissive
  metadata tag with verification of upstream ownership, generator terms or privacy rights.

## All training sources

Counts are unique training rows, not v2 replay frequency. Total: **24,436 rows from 14 sources**.
All are `review_required`; some have specific unresolved rights chains in addition to incomplete
human label/privacy review.

| Source ID | Rows | Publisher / provenance | Declared terms and primary reference |
| --- | ---: | --- | --- |
| `imc25_public_forum_smishing` | 1,248 | reportsmishing IMC25 release; forum-derived, not a direct Reddit scrape | [CC-BY-4.0; pinned release](https://github.com/reportsmishing/Smishing-Dataset-IMC25/tree/a6175560b57387199871e51fbef6bc523d2516b4); [paper](https://doi.org/10.1145/3730567.3764431) |
| `mendeley_sms_phishing` | 4,391 | Sandhya Mishra and Devpriya Soni; collected message corpus | [CC-BY-4.0; version 1](https://doi.org/10.17632/f45bkkt8pr.1) |
| `multidogo_human_service_dialogues` | 812 | AWS Labs MultiDoGO; human customer/agent roleplay, not ordinary victim calls | [CDLA-Permissive-1.0; pinned release](https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/tree/baa30639c4b271f394b81443c842193407cdf26d); [paper](https://aclanthology.org/D19-1460/) |
| `scamguard_ftc_pattern_action_states_v1` | 356 | ScamGuard original template copy grounded in FTC-authored pattern descriptions | Apache-2.0 project contributions; [FTC reference](https://consumer.ftc.gov/features/robocall-scam-examples) |
| `scamguard_multidogo_grounded_evidence_states_v1` | 844 | ScamGuard synthetic transformations retaining MultiDoGO human context | [CDLA-Permissive-1.0 source](https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/tree/baa30639c4b271f394b81443c842193407cdf26d); not relabeled wholly Apache-2.0 |
| `scamguard_synthetic_counterfactual_v1` | 128 | ScamGuard deterministic counterfactual slot filling, version 1 | Apache-2.0 project contributions; [advisory grounding](https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money) |
| `scamguard_synthetic_dialogue_v2` | 1,495 | ScamGuard paired scenario grammars, version 2 | Apache-2.0 project contributions; [advisory grounding](https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam) |
| `scamguard_synthetic_long_call_action_states_v1` | 6,144 | ScamGuard original long-context action-state templates, version 1 | Apache-2.0 project contributions; [advisory grounding](https://consumer.ftc.gov/articles/how-avoid-scam) |
| `scamguard_synthetic_v5` | 4,140 | ScamGuard original deterministic message templates, version 5 | Apache-2.0 project contributions; [advisory grounding](https://consumer.ftc.gov/consumer-alerts/2025/03/what-are-signs-scam) |
| `shakeleoatmeal_phone_scam_synthetic` | 1,190 | shakeleoatmeal; LLM-synthetic, generic “Llama 8B” disclosure only | [MIT front matter; pinned card](https://huggingface.co/datasets/shakeleoatmeal/phone-scam-detection-synthetic/blob/27a1f6d0cf21dda995130d221383900b060dbcc9/README.md); generator terms unresolved |
| `taskmaster1_woz_dialogues` | 1,193 | Google Research Taskmaster-1 human Wizard-of-Oz roleplay | [CC-BY-4.0; pinned source](https://github.com/google-research-datasets/Taskmaster/tree/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019) |
| `uci_sms_spam` | 147 | Tiago Almeida and Jos Hidalgo; SMS Spam Collection | [CC-BY-4.0; dataset DOI](https://doi.org/10.24432/C5CC84) |
| `wspr_sms_phishing` | 1,913 | WSPR / North Carolina State University project release | [MIT; pinned release](https://github.com/wspr-ncsu/sms-phishing/tree/ef7da01dfc145ce283a2af380e5dd8b817191ee7) |
| `youtube_scam_calls_cc0` | 435 | Rival Cults / rivalcults; manually corrected YouTube scammer/scambaiter/autodialer transcripts | [Publisher-declared CC0, version 2](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts); underlying rights/privacy unresolved |

The 2,634 development rows comprise Mendeley (531), UCI (15) and synthetic v5 (2,088).
Development is calibration/evaluation data, not v2 replay training. Data was normalized,
privacy-masked, deduplicated/family-partitioned, relabeled under the ScamGuard taxonomy and, for
some sources, windowed or transformed into contrast states. Exact policies and counts live in
the hash-pinned manifests; do not imply original publishers supplied ScamGuard's final labels.

Source: United States Federal Trade Commission, [ftc.gov](https://www.ftc.gov), for identified
fraud-pattern descriptions. No FTC or dataset-provider endorsement is claimed. The
[FTC website policy](https://www.ftc.gov/policy-notices/website-policy) distinguishes government
material from third-party material; it does not clear everything linked from an FTC page.

## Conditional redistribution obligations and open reviews

- [MIT](https://opensource.org/license/mit): preserve applicable copyright and permission notices
  in covered copies/substantial portions. The base notice is included; a release assembler still
  must preserve relevant upstream data notices where applicable, not invent an author's copyright.
- [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en): sharing covered material
  requires retention of supplied attribution/notices and a license link, plus indication of changes.
  Model-card source disclosure is our policy; this notice does not decide whether every weight
  artifact constitutes adapted material under every source's terms.
- [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/): sections 3.1–3.2 address Data/Enhanced
  Data notices and credits; section 3.4 imposes no obligations on Results as defined in section 1.10.
  Do not apply either category blindly to a release that may contain source text.
- [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) does not prove that a
  transcript uploader owns or cleared all underlying rights. The YouTube chain remains unresolved.
- The phone card's MIT declaration does not identify the exact Llama version or applicable
  generation/host terms. Obtain evidence and qualified review before distribution eligibility.

All external-message privacy and sampled-label reviews remain incomplete. No raw source,
training, blind-review or sealed-evaluation text should be bundled in a model-only staging pack.
Evaluation-only corpora are not included in these training counts and must not be silently
redistributed; their separate diagnostic restrictions remain in force.

See [release provenance and owner actions](ETTIN_RELEASE_PROVENANCE.md) and the
[one existing human-review handoff](INDEPENDENT_REVIEW_HANDOFF.md). No phone/YouTube rights
uncertainty is cured merely by omitting those sources from a continuation of already exposed weights.
