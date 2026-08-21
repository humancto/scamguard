"""Versioned, deployment-safe model input transforms."""

from __future__ import annotations

import re

DIALOGUE_POLICY_NONE = "none"
DIALOGUE_POLICY_SPEAKER_NEUTRAL_V1 = "speaker-neutral-v1"
DIALOGUE_POLICIES = (DIALOGUE_POLICY_NONE, DIALOGUE_POLICY_SPEAKER_NEUTRAL_V1)

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


def prepare_model_text(text: str, dialogue_policy: str = DIALOGUE_POLICY_NONE) -> str:
    if dialogue_policy == DIALOGUE_POLICY_NONE:
        return text
    if dialogue_policy == DIALOGUE_POLICY_SPEAKER_NEUTRAL_V1:
        return canonicalize_dialogue_speakers(text)
    raise ValueError(f"unsupported dialogue policy: {dialogue_policy}")
