# Online real-data source research

Research freeze: 2026-08-21. This review treats a large download count as irrelevant unless the
source has usable rights, row-level provenance, privacy controls, and meaningful novelty after
family-level overlap removal. Raw and processed corpora are ignored by Git; the public repository
ships hash-pinned acquisition and audit code, not sensitive message dumps.

## Admitted sources

Schema v12 retains 13,127 naturally occurring licensed-source examples, adds 600 separately
reported human-authored crowdsourced roleplay dialogues, and uses 10,071 controlled synthetic
examples. Its primary naturally occurring sources are UCI SMS Spam (CC-BY-4.0), the Mendeley SMS Phishing and
financial datasets (CC-BY-4.0), WSPR/NCSU gateway-confirmed phishing (MIT), and the IMC 2025
public-forum artifact (CC-BY-4.0). The forum artifact is the rights-safe way ScamGuard learns from
Reddit-like public reports: it is a research release with privacy normalization and source-wide
deduplication, not a fresh scrape of individual posts.

After the source review, synthetic v5 adds 24 paired, original scenario families grounded in FTC,
FBI/IC3, IRS, and USPIS advisories. This targeted increment covers layered safe-account transfer,
government-benefit and tax identity theft, jury-duty threats, task deposits, family-bail secrecy,
reshipping jobs, and their legitimate lookalikes. It does not copy source prose. The generator
records the relevant official URL on every row and remains Apache-2.0.

Synthetic dialogue v2 generates 1,536 training-only rows in 12 balanced scam/legitimate five-turn
scenario pairs; 1,495 pass the standard admission gates. It was created after held dialogue
selection slices showed both excessive false positives and a later corpus-format shortcut. The
generator retains advisory URLs and never copies source prose or any external diagnostic row.

Synthetic counterfactual v1 adds 512 balanced train-only messages after schema-v11's frozen error
ledger isolated false alarms around known contacts, verified transfers, in-platform marketplace
actions, and official-app review instructions. Its eight paired families alter the trust boundary
or requested action without copying any benchmark text. The schema-v12 result rejected this full
dose after it caused a catastrophic held identity-family regression. Schema v13 keeps an isolated
128-row dose-16 ablation; its neural result also fails the open model gates. No additional scraped
or weakly licensed bulk corpus was admitted.

### MOZ-Smishing: sealed schema-v8 holdout

- Primary artifact: <https://huggingface.co/datasets/MOZNLP/MOZ-Smishing>
- Paper: <https://doi.org/10.18653/v1/2025.africanlp-1.23>
- Pinned revision: `1092f9d9a545b29ae6be030ee9713b615fc2d987`
- Raw SHA-256: `814a11d9b05741c4b47eb0d0784b1fd12a2a076f83a714a9908bdda594986ab8`
- Provenance: crowd-sourced messages from Mozambican mobile-money users; 552 reported smishing
  messages and 2,009 legitimate messages in Portuguese.
- Privacy finding: the publisher says legitimate identifiers were anonymized, but ScamGuard's
  source-wide scan still found 1,103 rows with phone-like strings, 94 with long digit sequences,
  and two with email addresses. ScamGuard normalizes all labels before IDs, clustering, or output.
- Quality result: five exact mixed-label groups and two near-template mixed-label groups were
  quarantined. One representative per remaining campaign family was retained. The final sealed
  set has 1,820 rows: 526 SCAM and 1,294 SAFE.
- Independence: zero exact overlaps remained against the ten pre-existing processed benchmarks;
  four representative families with near overlap were removed. The artifact hash is
  `07edf56aea1704d86dbf2b71512fa59049b9d0cbc44d92eda942b67ecfc6b092`.
- License boundary: Hugging Face declares `creativeml-openrail-m`, but provides no
  dataset-specific license file. ScamGuard therefore permits local evaluation only, excludes it
  from training, and does not redistribute its rows pending clarification. This limitation is
  machine-recorded in `primary_test_v8.manifest.json`.

### Chichewa SMS: licensed external diagnostic

- Primary artifact: <https://doi.org/10.5281/zenodo.14607454>
- Paper: <https://arxiv.org/abs/2502.16947>
- Zenodo metadata revision: 12, modified 2026-02-05
- License: CC-BY-4.0 in the publisher's Zenodo API metadata
- Workbook SHA-256: `4f83cfaab196f8fab3bdbf9c89e15313ddaa889da066335fcc2f35cc6b3f487a`
- Audit finding: the Chichewa sheet has 676 balanced rows and 675 exact-unique texts; 197 rows
  contain phone-like values and 13 contain live-looking URLs. The paper describes an initial 101
  crowd-sourced fraud messages plus 25 from a telecom company, followed by augmentation, but the
  workbook does not mark row-level provenance. The separate 148-row legitimate telecom sheet has
  only 74 exact-unique texts and one template appears 28 times.
- Decision: admit only as a privacy-normalized, one-per-family multilingual diagnostic. The built
  artifact has 677 rows (315 SCAM, 362 SAFE), zero exact/near overlap with existing processed
  benchmarks, and SHA-256
  `b429621d3508b9cb3f1a7c7f079603f10966cdcec81b339d5d176aa8c77b0b38`. Do not count all source
  rows as real, do not use it for fitting, and do not mix its human or machine translations with
  the original language. Native-language label review is required before a public benchmark claim.

### BothBosu scam dialogue: external synthetic diagnostic

- Primary artifact: <https://huggingface.co/datasets/BothBosu/scam-dialogue>
- Pinned revision: `321b961b5ae353e19ed479b960658dcd223d5e06`
- License: Apache-2.0 in the dataset card
- Raw SHA-256: `fe8a8fa0aa2b8afb0b0a672fb7f9739b323cb6dd12064f786a68c2a1f49a4e0b`
- Audit finding: all 1,600 rows are synthetic Llama-3-70B telephone dialogues, evenly divided
  between four scam and four legitimate-call types. Two exact duplicates were removed; one
  mixed-label near-template cluster was quarantined. The largest remaining cluster has 88 rows.
- Decision: retain one representative per near-template family as a 1,343-row external stress test
  (684 SCAM, 659 SAFE). It has zero exact/near overlap with existing processed benchmarks. A salted
  family hash creates a 294-row candidate-selection slice (SHA-256
  `c473c94a6d3cc7b6c114c5e6b29f86a31e454310558f5282d9c1133bb51741a0`) and a 1,049-row
  prediction-sealed OOD slice (SHA-256
  `33d480aa505f16014e18a7193f379b618e7a9feeb90262c93e77433c022c1193`). It is never used for
  fitting or threshold selection and never counted as real-world data.

### Taskmaster-1: admitted human-authored hard negatives

- Primary artifact: <https://github.com/google-research-datasets/Taskmaster/tree/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019>
- Pinned revision: `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`
- License: CC-BY-4.0 in the publisher README
- Raw two-person-dialogue SHA-256:
  `cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f`
- Provenance: 5,507 two-person Wizard-of-Oz transactional dialogues written by human participants
  across six task domains. They are realistic roleplay, not naturally occurring calls or messages.
- Privacy: ScamGuard retains only recent complete turns under 1,100 characters and replaces email,
  URL, and phone/account-like strings before materialization.
- Partition: a salted conversation-family hash runs before sampling. Training receives 100 rows per
  domain (600 total); a disjoint selection slice receives 75 per domain (450 total).
- Label boundary: Taskmaster has no scam label. The admitted rows receive a weak SAFE label from the
  legitimate task domain and are reported as human-authored roleplay, not real-world negative truth.
- Baseline result: before fitting on Taskmaster, the schema-v9 149M and 395M encoders falsely flagged
  95.78% and 20.22% of the 450 SAFE selection rows at their frozen thresholds. This is selection
  evidence motivating schema v10, not a release score.

### Dialogue-source refresh after schema v13

The schema-v13 149M encoder plus `trusted-channel-v1` policy clears the open short-message binary
gate but still reaches only 51.06% recall / 18.30% SAFE FPR on the 294-row BothBosu dialogue
selection slice. A new source search therefore focused on licensed call-level dialogue rather than
adding more short-message rows.

| Candidate | Evidence | Decision |
|---|---|---|
| [YouTube Scam Phone Call Transcripts](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts) | Version 2 is publisher-declared CC0: 243 manually corrected partial transcripts sourced from scam-call videos, mostly scammer/scambaiter calls plus some autodialer messages. The 149,701-byte archive is pinned at SHA-256 `3f67497736e9421c2f6e59efc46c129006419d40fc752cbb981042940384cedd`. ScamGuard found 222 source URLs, removed one exact duplicate, found no existing-corpus exact/near overlap, and retained 448 windows in 220 connected source/template families. | Admit as real scam-call-derived, positive-only research data—not ordinary victim calls or independently reviewed truth. Schema v14 fits only 161 early windows from 145 train families, keeps 70 windows/35 families open for selection, and leaves 80 windows/40 families prediction-sealed. The first model is rejected despite 100% selection recall because regression and dialogue FPR become unsafe. |
| [TeleAntiFraud-28k](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud) | Apache-2.0 dataset card; pinned public revision `0872e54b584b28d34e0911dffcf696f0b2e5e49a`; 4,000/400 call-level binary train/test rows plus 27,146/6,807 SFT rows. The paper reports a mixture of privacy-preserved real-call ASR, LLM semantic augmentation, and multi-agent synthesis. | Highest-priority dialogue candidate, but the repository is gated and this machine is not authenticated. Do not bypass the gate through a derivative mirror. After access is approved, inspect whether construction provenance is available per row, privacy-normalize text, and keep the publisher test split external. |
| [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | Pinned revision `95a8c157e4fd6df2f3c77593160c83db79b75dc7`; CC-BY-SA-4.0; 873 newly collected English, multi-accent, spontaneous role-played service calls across 16 domains. The card says the corpus is intended exclusively for evaluation and analysis and explicitly puts training out of scope. | Admit text metadata as evaluation-only SAFE-call roleplay. After exact/near-template control, 348 windows from 174 calls form the open selection slice and 1,396 windows from 699 calls remain prediction-sealed. Shared-speaker/call components cannot cross the split. Zero AppTek rows enter fitting. |
| [ES-Port](https://github.com/Vicomtech/esport-corpus) | Anonymized transcripts of spontaneous real Spanish telecom technical-support calls, released under CC-BY-SA-3.0 Spain. The 3,681,616-byte publisher ZIP is SHA-256 `0017a2d6bbbf57d2971872c7a8eb7c1bf266c76dfb8ba2736c17086a948f1f2c`; the documentation repository is pinned at `b575ec6d7925ea4f83ef00a302682a6f58ad788d`. | Promising authentic SAFE-call source, but not yet admitted. First audit residual identifiers, verify the weak legitimate-domain label, isolate whole-call families, evaluate Spanish/source shortcuts, and resolve model-artifact share-alike obligations. |
| [KorCCVi v2](https://github.com/selfcontrol7/Korean_Voice_Phishing_Detection/blob/main/DATA.md) | 1,417 Korean call transcripts, including 706 real FSS vishing calls. The publisher explicitly says transcript derivatives retain third-party terms, must not be redistributed, and require source permission. Scam label and source are also perfectly correlated. | Reject for current training and benchmarking. The code's MIT license does not license the transcripts, and the source/label confound invalidates a headline detector score. |
| [Arabic Scam and Legitimate Call Conversation Dataset](https://data.mendeley.com/datasets/p384bgyzz3/2) | CC-BY-4.0, 448 balanced five-turn Arabic conversations across nine dialects and 23 categories. The calls and audio are simulated rather than naturally occurring. | Candidate multilingual synthetic OOD diagnostic after native-language review; it cannot satisfy the real-dialogue gap or increase the licensed-real count. |
| [Scam Conversation Corpus](https://zenodo.org/records/15212527) | English multi-platform conversations with real scammers and an automated GPT-4o victim, but files are restricted, message bodies remain unaltered, and use is stated as research-only. | Reject for the distributable product corpus absent access, explicit commercial-training permission, and a privacy review. |

This refresh confirms that rights-clean authentic scam dialogue is the scarce resource. TeleAntiFraud
is worth requesting because it combines real-call-derived language with licensed release terms, but
its mixed construction must be reported honestly; it is not 28,511 independently collected calls.

### HarperValleyBank: admitted human-spoken call structure

- Primary repository: <https://github.com/cricketclub/gridspace-stanford-harper-valley>
- Paper: <https://arxiv.org/abs/2010.13929>
- License: CC-BY-4.0
- Pinned revision: `0bd721e877c4a85d8c13ff837e68661ea6200a98`
- Provenance: 1,446 human-human simulated telephone-banking calls across 59 speakers and eight
  transactional tasks. These are real human utterances in roleplay, not naturally occurring calls.
- Acquisition: transcript and metadata trees only; no audio. Both trees and the publisher license
  are independently SHA-256 pinned in addition to the Git revision.
- Partition: six entire tasks and 1,069 call families enter training; branch-hours and card-replace
  tasks form a 377-family validation split. No call or task crosses the split.
- Decision: admit the original SAFE calls with weak action labels and full verdict weight. Add four
  same-context final-turn states per call, but count all 5,784 transformed rows as controlled
  synthetic derivatives. See `DATASET_SCHEMA21_HUMAN_CALLS.md` for exact counts and hashes.

This is a bounded response to schema 20's external error ledger, not an invitation to ingest every
available service dialogue. It adds authentic conversational rhythm and sensitive legitimate
actions while preserving interpretable task-disjoint measurement.

### MultiDoGO: admitted human-authored service evidence

- Primary repository:
  <https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset>
- Paper: <https://aclanthology.org/D19-1460/>
- License: CDLA-Permissive-1.0
- Pinned revision: `baa30639c4b271f394b81443c842193407cdf26d`
- Provenance: human customers roleplayed six service domains with trained human agents. These are
  human-authored conversations, not naturally occurring customer-support logs.
- Raw scope: 86,719 conversations; 84,129 pass frozen structural quality filters. The fetcher
  acquires only six dialogue files, README, notice, and license; no audio.
- Template control: a deterministic 18,000-conversation audit pool collapses to 3,485 exact/near-
  template representatives before sampling. One source conversation may enter only one split.
- Admission: 1,790 weak-SAFE training views from 895 families and 896 validation views from 448
  disjoint families. Another 1,184 train and 592 validation rows are controlled four-state
  derivatives and remain synthetic in all accounting.
- Domain holdout: action states train on airline, fast food, finance, and media; insurance and
  software are validation-only. Five train and two validation families were removed because a view
  was near a schema-v20 artifact.
- Decision: admit as schema v22's bounded service-dialogue increment. Original rows use half verdict
  weight and weak auxiliary action labels; the experiment starts from the safer schema-v20 parent.
  See `DATASET_SCHEMA22_SERVICE_EVIDENCE.md` for exact hashes and gates.

MultiDoGO was selected over broad Schema-Guided Dialogue ingestion because its trained-agent roleplay
more directly matches ScamGuard's unresolved gap: sensitive but legitimate service actions. The
bounded family sample makes failure attribution possible and avoids turning repeated service scripts
into an inflated data-volume claim.

The repository now includes a pinned CC0 fetcher/builder for the YouTube-call source, plus the
pinned gated TeleAntiFraud fetcher and text-free admission auditor. The YouTube builder rejects
unsafe ZIP members, validates the exact publisher schema and character lengths, masks privacy-like
values, removes existing-corpus overlaps, and partitions connected source/template families before
materialization. Its first 161-row schema-v14 dose increased the new 70-row selection recall from
34.29% to 100%, but also raised unchanged-regression FPR from 4.18% to 8.48% and the balanced
telephone-dialogue SAFE FPR from 18.30% to 73.20%; the checkpoint is rejected.

The TeleAntiFraud fetcher
requests only `binary_classification.zip` (60.3 kB on the publisher's file listing),
`dataset_manifest.json`, and `README.md`; it deliberately excludes the 12.7 GB audio archive and
the SFT archive. The auditor accepts exactly 4,000 train and 400 publisher-test records, rejects
unsafe ZIP paths or unknown labels, and emits only schema/count/duplication/privacy/provenance-field
statistics. Its admission count is hard-coded to zero pending inspection. The live fetch currently
fails with the publisher's expected 401 gated-repository response because this machine has no
authorized Hugging Face session; no TeleAntiFraud row has been downloaded or viewed.

The AppTek text-only builder is now complete. It pins and verifies 14 metadata files totaling
17,438,486 bytes, downloads no audio, validates 873 unique calls, collapses two same-source near
templates, and partitions 1,744 retained early/recent windows through 77 shared-speaker/call
components. Schema v13 falsely flags 31/348 open SAFE windows (8.91%, 95% CI 6.35–12.37%); schema
v14 falsely flags 77/348 (22.13%, 18.08–26.78%). Every false alarm occurs in an early-call window;
both models score all 174 recent windows SAFE. This selects a 256-row original synthetic
legitimate-opening correction for schema v15 without copying or fitting AppTek text. The open
slice is consequently candidate-selection evidence, and the 1,396-window OOD slice remains sealed.
The completed correction is rejected: AppTek FPR remains 15.52%, unchanged-regression FPR rises to
18.84%, and BothBosu records 69.50% recall / 35.29% FPR. This is evidence against simply increasing
the same synthetic SAFE dose; it does not justify opening the sealed AppTek partition.

### 2026-08-21 GitHub and Hub refresh

| Candidate | Current evidence | Decision |
|---|---|---|
| [ThaiScamCall](https://huggingface.co/datasets/Paam1/ThaiScamCall) | CC-BY-4.0 and Thai, but the card says the calls are AI-generated scripted TTS. The current Hub viewer exposes 100 rows while the card claims 21,287 audio clips. | Do not count as real calls or text-training data. Revisit only as an audio-robustness candidate after the release shape, scripts, and transcript availability are reconciled with native-language review. |
| [Vietnamese scam dialogues](https://huggingface.co/datasets/adamtc/scam_dialogues) | Apache-2.0, 3,840 rows, explicitly synthetic, with long generated dialogue plus generated explanations. The card does not identify a generator, prompt, revision, or independent label audit. | Do not fit. It may become a multilingual synthetic diagnostic after native review, generator provenance, family clustering, and overlap checks against known synthetic dialogue releases. |
| [FraudLens-RU v1](https://huggingface.co/datasets/Abdurohman/fraudlens-ru-v1) | CC-BY-4.0, 6,330 Russian rows from nine public anti-fraud channels, with article/post text, summaries, and analyst-style fraud taxonomy. The examples are educational reports, not message-local victim communications or call transcripts. | Taxonomy research only. Do not let educational phrases become an easy SCAM shortcut or count the rows as real scam conversations. Source-level rights and duplication still require audit. |
| [Korean voice-phishing GitHub fork](https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts) | Claims 2,927 call transcripts and sub-10-ms inference, but the repository exposes no dataset license and says required source text files are not provided there. The reported benign and scam classes also come from different source domains. | Reject for training and headline comparison. Repository-code licensing cannot grant rights to missing underlying transcripts, and the source/label confound makes the reported latency/accuracy non-comparable. |

The refresh does not change the rights-clean source decision. It supports a small, controlled
structure-matched increment over bulk ingestion. Schema v17 therefore adds 576 balanced original
minimal-pair training rows and holds 192 paired rows out by complete service scenario; see
[`DATASET_SCHEMA17_CALL_MINIMAL_PAIRS.md`](DATASET_SCHEMA17_CALL_MINIMAL_PAIRS.md).

### 2026-08-22 pattern-rights refresh

| Candidate | Current evidence | Decision |
|---|---|---|
| [FTC Robocall Scam Examples](https://consumer.ftc.gov/features/robocall-scam-examples) | FTC-authored pages describe current scam mechanisms and link to individual call examples. The [FTC Website Policy](https://www.ftc.gov/policy-notices/website-policy) says most FTC-authored material is public domain but warns that some site material may be third-party. | Use scenario descriptions as grounding only. Copy zero audio or transcript wording; generate original Apache-2.0 four-state contrasts and retain the FTC URL on each row. |
| [TeleAntiFraud-28k](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud) | The Apache-2.0 repository remains gated and the pinned fetch returns 401 without an authorized account. | Keep the fail-closed fetch/audit path; zero downloaded, viewed, or fitted rows and no derivative-mirror bypass. |
| [Sting9](https://sting9.org/license) | The governing license is ODC-BY-NC and excludes startup/product development and commercial model training despite a conflicting CC0 marketing statement. | Reject for ScamGuard's commercial-capable corpus. |
| Raw Reddit and forum posts | Platform terms, privacy, deletion, provenance, and contamination cannot be resolved by treating public visibility as a training grant. | Copy zero raw posts. Continue using only the licensed, privacy-normalized IMC 2025 research artifact already admitted in the parent corpus. |

The resulting schema-v23 increment is 1,431 rows rather than a bulk scrape: 215 licensed human
MultiDoGO turns, 860 human-grounded MultiDoGO states, and 356 admitted original FTC-pattern states.
It is frozen in `DATASET_SCHEMA23_EVIDENCE_COMPACTION.md`. The trained candidate is rejected after
18/36 gates: held MultiDoGO SAFE FPR is 23.10%, while prior-open BothBosu is 77.30% recall at
13.73% FPR. Controlled scoring shows that evidence compaction helps, but neither the schema-v20,
v22, nor v23 training mixture learns the joint sensitivity-specificity boundary.

The same pinned MultiDoGO revision contains 36 CDLA-Permissive turn- and sentence-level customer
intent/slot files that were omitted from the original sparse checkout. Schema-v24 preparation adds
a pinned-tree fetch contract and text-free source audit for those files. Materialization is pending
because the current Codex network approval is usage-limited; no mirror, revision change, or download
bypass was attempted, and zero annotation-derived rows are admitted. Once available, intent/slot
metadata may ground participant-aware matched contrast families; it will not be counted as new
human dialogue beyond the underlying conversations.

## Audited but rejected or quarantined

| Candidate | Finding | Decision |
| --- | --- | --- |
| [SmishX](https://github.com/yizhu-joy/SmishX) | Strong SOUPS 2025 relabeling, MIT repository, but 807/1,200 rows exactly match and 1,023/1,200 are near matches to already processed data; 98 internal duplicate rows. The paper says it samples older public datasets plus 22 personal messages without per-row source IDs. | Keep the audit as external evidence; do not call it a fresh test or add it to training. |
| [SMISH_DT](https://github.com/MarazMia/SMISH_DT) | Unlicensed aggregate of at least 15 datasets, including noncommercial and unlicensed components. | Reject the aggregate; inspect only original publishers. |
| [SecureBharat 50k](https://github.com/AdityaKhaire45/sms-scam-dataset) | No license or collection methodology; repeated parameterized templates and `.example` URLs show that it is generated, not 50,000 real reports. | Reject as real data and as uncontrolled synthetic data. |
| [SpamHunter dataset](https://github.com/opmusic/SpamHunter_dataset) | Useful Twitter-screenshot research artifact, but the dataset repository has no license and includes sender/URL files. | Reject pending explicit rights and privacy clarification. |
| [Bengali SMS Smishing](https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset) | MIT tag and 7,005 rows, but no collection/annotation provenance in the card; inspection shows repeated templates, live-looking domains, and phone numbers. | Do not count as real. May inform taxonomy only after sanitization. |
| [Sting9](https://sting9.org/license) | The marketing page says CC0, while the governing legal page says ODC-BY-NC and expressly disallows startup/product development and commercial model training. | Reject for this product unless commercial permission is obtained. |
| [DIFrauD](https://huggingface.co/datasets/difraud/difraud) | MIT and manually cleaned, but its SMS slice combines UCI and Mendeley sources already used by ScamGuard. | No novelty after source accounting; exclude from size claims. |
| [ScamGuardBench](https://huggingface.co/datasets/flowxai/scamguardbench) | Apache-2.0 external benchmark with useful hard-legitimate cases, but a mixed/generated evaluation artifact rather than a new real-message training source. Its model-card scores are self-reported. | Reserve for named external comparison; never merge into training. |
| [Malicious/Benign SMS/MMS v3](https://huggingface.co/datasets/notd5a/malicious-benign-sms-mms-dataset) | CC-BY-NC-4.0 derivative with 442,282 rows, including 59,035 declared AI-generated rows and upstream UCI/Mendeley content already represented here. | Reject for the commercial-capable product corpus; no novelty-adjusted size claim. Its hard-negative categories may guide original counterfactual design. |
| [Africa Smishing](https://huggingface.co/datasets/electricsheepafrica/africa-smishing-sms-phishing) | MIT-tagged but all 10,000 rows are synthetic; the card gives regional categories but no row-level generator, prompts, model revision, or independent quality audit. | Taxonomy research only; do not replace the controlled generator with an unaudited synthetic bulk set. |
| [Scam Classification Multiclass](https://huggingface.co/datasets/Shade63/scam-classification-multiclass) | 14,000 Indian-context rows generated by an ML agent; the card says only “same as original dataset” and does not identify a usable source license or collection method. | Reject pending source and rights evidence. |
| [FraudSMSWalker](https://arxiv.org/abs/2606.16659) | Strong new 699-chain bilingual SMS-to-webpage benchmark with 367 hard benign cases, but it evaluates sanitized webpage evidence as well as SMS and currently points to an anonymous artifact without a verified release license. | Track as a cross-channel external benchmark; do not merge it into message-only training or current ScamBench scores. |
| [COVA-X / ScamLingua](https://scamlingua.org/) | Approximately 11,000 fully synthetic multi-turn conversations generated with Qwen 2.5 14B. Access is by request, noncommercial-research-only, and redistribution is prohibited. | Valuable future research comparison, but reject for the distributable product corpus. The admitted Apache-2.0 dialogue diagnostic supplies a reproducible first stress test. |

Direct Reddit collection remains off-limits under Reddit's Data API Terms, last revised July 20,
2026. The current
[Data API Terms](https://redditinc.com/policies/data-api-terms). Public discussions may guide the
threat taxonomy, but individual content is not copied into the corpus. The IMC 2025 release is the
approved forum-derived source because its researchers handled collection, labeling, release rights,
and privacy as a dataset artifact.

## Dataset-size decision

More rows do not make inference slower; model architecture, sequence length, runtime, and hardware
determine the under-20-ms target. More low-quality or duplicated rows do make experiments worse by
inflating metrics and teaching shortcuts. Current status and next actions are:

1. Keep schema v23 rejected; do not export, externally select, or tune it on BothBosu.
2. Materialize and audit the pinned MultiDoGO intent/slot tree before defining schema-v24 rows.
3. Require source-aligned, participant-aware action labels and complete matched contrast families;
   do not spend another neural run if a family-disjoint label audit cannot reach 90% exact match.
4. Keep AppTek evaluation-only and its 1,396-window OOD partition sealed.
5. Obtain authorized access to TeleAntiFraud-28k, then audit per-row construction provenance,
   privacy, family duplication, source/label shortcuts, and license files before admitting a row.
6. Keep the publisher's TeleAntiFraud test split external and create a family-disjoint open
   selection slice before any new dialogue training run.
7. Audit ES-Port as a possible authentic Spanish SAFE-call source, but do not admit it until privacy,
   weak-label, language-shortcut, and share-alike questions are resolved.
8. Measure the fast encoder and any Qwen fallback separately; do not represent 4B generation
   latency as a sub-20-ms mobile detector.

This gives ScamGuard enough training mass for a serious experiment while preserving an honest,
new-source denominator for the next model decision.
