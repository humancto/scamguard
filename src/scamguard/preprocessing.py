"""Versioned, deployment-safe model input transforms."""

from __future__ import annotations

import re

DIALOGUE_POLICY_NONE = "none"
DIALOGUE_POLICY_SPEAKER_NEUTRAL_V1 = "speaker-neutral-v1"
DIALOGUE_POLICY_EVIDENCE_RECENT_V2 = "speaker-neutral-evidence-recent-v2"
DIALOGUE_POLICIES = (
    DIALOGUE_POLICY_NONE,
    DIALOGUE_POLICY_SPEAKER_NEUTRAL_V1,
    DIALOGUE_POLICY_EVIDENCE_RECENT_V2,
)

# The compactor is deliberately character-bounded instead of tokenizer-specific. That keeps the
# transform identical in Python, ONNX, and eventual native runtimes. The model still enforces its
# independently frozen token limit after this transform.
EVIDENCE_RECENT_MAX_CHARS = 1_400
EVIDENCE_RECENT_MAX_EVIDENCE_TURNS = 3
EVIDENCE_RECENT_MAX_RECENT_TURNS = 3
EVIDENCE_RECENT_MAX_TURN_CHARS = 520

_HEADER_RE = re.compile(
    r"^\s*(?:transcript|chat\s+log|recent\s+conversation|dialogue)\s*:\s*",
    re.IGNORECASE,
)
_SPEAKER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<label>caller|receiver|recipient|agent|customer|bank|courier|recruiter|"
    r"applicant|sender|clerk|resident|participant|user|assistant)\s*:",
    re.IGNORECASE,
)

# Source-grounded action vocabulary, frozen before schema-23 fitting. These expressions select
# complete turns; they never assign a verdict. Both harmful requests and explicit safe-navigation
# guidance are retained so a sensitive topic alone cannot determine the result.
_HIGH_RISK_ACTION_RE = re.compile(
    r"\b(?:one[- ]time (?:code|passcode|password)|verification code|security code|"
    r"password|passcode|\bpin\b|cvv|routing number|bank account|card number|"
    r"gift card|prepaid card|bitcoin|crypto(?:currency)?|wire (?:money|funds)|"
    r"transfer (?:money|funds|the balance)|remote (?:access|desktop|support)|"
    r"screen shar(?:e|ing)|install (?:the |this |a )?(?:app|application|software|tool)|"
    r"download (?:the |this |a )?(?:app|application|software|tool)|"
    r"click (?:the |this |a )?(?:link|button)|log ?in (?:through|using|at)|"
    r"keep (?:this|it|the matter) (?:secret|confidential)|do not tell|don't tell|"
    r"warrant|arrest|lawsuit|suspend(?:ed|ing)? (?:your )?(?:account|number|service)|"
    r"protection account|safe account|release fee|processing fee|clearance fee)\b",
    re.IGNORECASE,
)
_REQUEST_OR_PRESSURE_RE = re.compile(
    r"\b(?:can you|could you|do you have|may i (?:have|know)|need you to|please|"
    r"what is|what's|would you|send|share|read|tell|give|provide|confirm|enter|buy|"
    r"purchase|pay|deposit|"
    r"move|transfer|wire|open|visit|press|act|respond|stay on the line|do it now|"
    r"immediately|urgent|final (?:attempt|notice|chance|courtesy call)|today only)\b",
    re.IGNORECASE,
)
_SERVICE_ACTION_RE = re.compile(
    r"\b(?:account|address|bank|balance|billing|card|claim|code|credit|deductible|"
    r"deposit|download|email|identity|install|insurance|license|loan|order|password|"
    r"payment|phone|policy|purchase|refund|routing|security|social security|"
    r"subscription|transfer|verify)\b",
    re.IGNORECASE,
)
_IRREVERSIBLE_ACTION_RE = re.compile(
    r"\b(?:book|buy|cancel|charge|confirm the order|deposit|pay|place the order|"
    r"purchase|submit|transfer)\b",
    re.IGNORECASE,
)
_SAFE_BOUNDARY_RE = re.compile(
    r"\b(?:do not|don't|never|decline|avoid|hang up|end (?:this|the) call|"
    r"open the official (?:app|site|website)|official app (?:yourself|instead)|"
    r"type the (?:known|official) (?:address|website)|number on (?:your|the) card|"
    r"published (?:service |support |billing )?number|verified (?:service |support )?number|"
    r"independently (?:call|contact|check|confirm|verify|navigate|review)|"
    r"verify (?:it|this|the request|the charge|the account) independently|"
    r"keep (?:the |your |every )?(?:code|password|pin|card|account) (?:private|secret)|"
    r"no (?:payment|transfer|code|remote access|gift card|crypto(?:currency)?) (?:is |was )?"
    r"(?:needed|required))\b",
    re.IGNORECASE,
)
_SCAM_FRAME_RE = re.compile(
    r"\b(?:refund|rebate|reward|prize|lottery|suspicious (?:charge|activity)|"
    r"unauthorized (?:charge|purchase|transaction)|customer support|tech support|"
    r"social security|tax debt|student loan|utility|customs|border protection|"
    r"package (?:was |is )?(?:seized|intercepted)|account (?:was |is )?compromised)\b",
    re.IGNORECASE,
)


def parse_dialogue_turns(text: str) -> list[tuple[str, str]]:
    """Return recognized ``(speaker, content)`` turns, including one-sided transcripts."""

    body = _HEADER_RE.sub("", text, count=1)
    matches = list(_SPEAKER_RE.finditer(body))
    if not matches:
        return []
    turns: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = " ".join(body[match.end() : end].split())
        if content:
            turns.append((match.group("label").casefold(), content))
    return turns


def canonicalize_dialogue_speakers(text: str) -> str:
    """Remove corpus-specific dialogue labels while retaining turn order and content.

    The transform intentionally activates only when at least four turns and two distinct known
    speaker labels are present. That avoids rewriting ordinary short messages which happen to use
    a word such as ``caller:`` once.
    """

    body = _HEADER_RE.sub("", text, count=1)
    matches = list(_SPEAKER_RE.finditer(body))
    labels = {match.group("label").casefold() for match in matches}
    if len(matches) < 4 or len(labels) < 2:
        return text

    mapping: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        label = match.group("label").casefold()
        if label not in mapping:
            ordinal = len(mapping)
            suffix = chr(ord("A") + ordinal) if ordinal < 26 else str(ordinal + 1)
            mapping[label] = suffix
        return " " + mapping[label] + ":"

    canonical = _SPEAKER_RE.sub(replace, body)
    return " ".join(canonical.split())


def _turn_score(content: str) -> tuple[int, int, int, re.Match[str] | None]:
    high_risk = list(_HIGH_RISK_ACTION_RE.finditer(content))
    safe_boundary = list(_SAFE_BOUNDARY_RE.finditer(content))
    requests = list(_REQUEST_OR_PRESSURE_RE.finditer(content))
    service_actions = list(_SERVICE_ACTION_RE.finditer(content))
    irreversible = list(_IRREVERSIBLE_ACTION_RE.finditer(content))
    frames = list(_SCAM_FRAME_RE.finditer(content))
    # Repetition is weak evidence. Capping match counts prevents long benign turns which repeat a
    # topic word (for example, "account") from crowding out a short explicit transfer request.
    def count(matches: list[re.Match[str]]) -> int:
        return min(len(matches), 3)
    service_score = 5 * count(service_actions) + 3 * count(requests) + 4 * count(irreversible)
    safe_score = (
        20 * count(safe_boundary) + 2 * count(high_risk) + count(service_actions)
        if safe_boundary
        else 0
    )
    score = (
        12 * count(high_risk)
        + 10 * count(safe_boundary)
        + 3 * count(requests)
        + 2 * count(service_actions)
        + 4 * count(irreversible)
        + count(frames)
    )
    matches = high_risk + safe_boundary + requests + service_actions + irreversible + frames
    anchor = min(matches, key=lambda match: match.start()) if matches else None
    return score, service_score, safe_score, anchor


def _evidence_turns(
    turns: list[tuple[str, str]],
) -> list[tuple[int, int, re.Match[str] | None]]:
    """Select participant-balanced action evidence plus the strongest safety boundary.

    A global top-k can spend every evidence slot on one participant and erase the other
    participant's requested or completed action. Version 2 reserves a representative for each
    speaker, prioritizing service actions and requests while demoting generic safety closures,
    then adds the strongest explicit safety boundary and fills any remaining capacity globally.
    """

    metadata: list[tuple[int, str, int, int, int, re.Match[str] | None]] = []
    for index, (speaker, content) in enumerate(turns):
        score, service_score, safe_score, anchor = _turn_score(content)
        metadata.append((index, speaker, score, service_score, safe_score, anchor))

    chosen: dict[int, tuple[int, int, re.Match[str] | None]] = {}
    speakers = list(dict.fromkeys(speaker for speaker, _ in turns))
    for speaker in speakers:
        candidates = [item for item in metadata if item[1] == speaker and item[2] > 0]
        if not candidates:
            continue
        # Safety-only closings are useful, but should not displace the participant's actual
        # service action. They receive their own dedicated selector below.
        representative = max(
            candidates,
            key=lambda item: (
                int(item[4] == 0),
                item[2],
                item[3],
                len(turns[item[0]][1]),
                item[0],
            ),
        )
        index, _, score, _, _, anchor = representative
        chosen[index] = (score, index, anchor)
        if len(chosen) == EVIDENCE_RECENT_MAX_EVIDENCE_TURNS:
            break

    safe_candidates = sorted(
        (item for item in metadata if item[4] > 0),
        key=lambda item: (item[4], item[3], len(turns[item[0]][1]), item[0]),
        reverse=True,
    )
    for boundary in safe_candidates:
        if len(chosen) == EVIDENCE_RECENT_MAX_EVIDENCE_TURNS:
            break
        index, _, score, _, _, anchor = boundary
        chosen.setdefault(index, (score, index, anchor))

    for item in sorted(
        metadata,
        key=lambda item: (item[2], item[3], len(turns[item[0]][1]), item[0]),
        reverse=True,
    ):
        if len(chosen) == EVIDENCE_RECENT_MAX_EVIDENCE_TURNS:
            break
        index, _, score, _, _, anchor = item
        chosen.setdefault(index, (score, index, anchor))

    return sorted(chosen.values(), key=lambda item: (item[0], item[1]), reverse=True)


def _bounded_turn(content: str, anchor: re.Match[str] | None, *, recent: bool) -> str:
    content = " ".join(content.split())
    if len(content) <= EVIDENCE_RECENT_MAX_TURN_CHARS:
        return content
    if anchor is None:
        return "…" + content[-(EVIDENCE_RECENT_MAX_TURN_CHARS - 1) :] if recent else (
            content[: EVIDENCE_RECENT_MAX_TURN_CHARS - 1] + "…"
        )
    before = 250
    start = max(0, anchor.start() - before)
    end = min(len(content), start + EVIDENCE_RECENT_MAX_TURN_CHARS)
    start = max(0, end - EVIDENCE_RECENT_MAX_TURN_CHARS)
    fragment = content[start:end]
    if start:
        fragment = "…" + fragment[1:]
    if end < len(content):
        fragment = fragment[:-1] + "…"
    return fragment


def compact_dialogue_evidence_recent(text: str) -> str:
    """Retain strongest action evidence and recent context in a bounded single-pass input.

    The transform activates only for dialogues with at least four recognized turns and two
    speakers. Evidence turns are ordered by frozen selector strength, followed by non-duplicate
    recent turns in chronological order. It does not infer whether an action is legitimate.
    """

    turns = parse_dialogue_turns(text)
    if len(turns) < 4 or len({speaker for speaker, _ in turns}) < 2:
        return text

    speaker_map: dict[str, str] = {}
    for speaker, _ in turns:
        if speaker not in speaker_map:
            ordinal = len(speaker_map)
            speaker_map[speaker] = (
                chr(ord("A") + ordinal) if ordinal < 26 else str(ordinal + 1)
            )

    def neutral_speaker(speaker: str) -> str:
        return speaker_map[speaker]

    evidence = _evidence_turns(turns)
    evidence_indices = {index for _, index, _ in evidence}
    recent_indices = [
        index
        for index in range(max(0, len(turns) - EVIDENCE_RECENT_MAX_RECENT_TURNS), len(turns))
        if index not in evidence_indices
    ]

    sections: list[str] = []
    if evidence:
        evidence_lines = []
        for _, index, anchor in evidence:
            speaker, content = turns[index]
            evidence_lines.append(
                f"{neutral_speaker(speaker)}: {_bounded_turn(content, anchor, recent=False)}"
            )
        sections.append("EVIDENCE: " + " ".join(evidence_lines))
    if recent_indices:
        recent_lines = []
        for index in recent_indices:
            speaker, content = turns[index]
            recent_lines.append(
                f"{neutral_speaker(speaker)}: {_bounded_turn(content, None, recent=True)}"
            )
        sections.append("RECENT: " + " ".join(recent_lines))

    compacted = " ".join(sections)
    if not compacted:
        return canonicalize_dialogue_speakers(text)
    if len(compacted) > EVIDENCE_RECENT_MAX_CHARS:
        compacted = compacted[: EVIDENCE_RECENT_MAX_CHARS - 1].rstrip() + "…"
    return compacted


def prepare_model_text(text: str, dialogue_policy: str = DIALOGUE_POLICY_NONE) -> str:
    if dialogue_policy == DIALOGUE_POLICY_NONE:
        return text
    if dialogue_policy == DIALOGUE_POLICY_SPEAKER_NEUTRAL_V1:
        return canonicalize_dialogue_speakers(text)
    if dialogue_policy == DIALOGUE_POLICY_EVIDENCE_RECENT_V2:
        return compact_dialogue_evidence_recent(text)
    raise ValueError(f"unsupported dialogue policy: {dialogue_policy}")
