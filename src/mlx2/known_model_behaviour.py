"""Declared, artifact-scoped known model behaviour for qualification checks.

A qualification check may fail because of what a specific model artifact
chooses to say, not because the runtime is wrong.  An entry here records
that decision for exactly one check on exactly the artifact fingerprints the
receipts record (``/v1/status`` ``artifact``), with a reason and the evidence
it rests on.  It never turns the check into a pass: the producer records it
as ``known_model_behaviour`` and keeps running the remaining checks, and the
route loader accepts the receipt only when the recorded result still matches
this declaration and its failure shape.  Any other failure of that check, or
the same failure on any other artifact, fails as before.

The lookup is by (check, artifact fingerprint); generic qualification code
never branches on model names.  Add an entry only on an explicit decision,
with the evidence path, and re-pin ``APPROVED_QUALIFICATION_HARNESS`` when
the producer's handling of a shape changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

KNOWN_MODEL_BEHAVIOUR_STATUS = "known_model_behaviour"

# The ``tools`` check's request: one user turn, one weather(city) tool,
# tool_choice auto.  The shape below re-asks the same request with
# tool_choice "required" so a declared text reply is only accepted while the
# same artifact still calls the tool correctly when the call is enforced.
TOOLS_AUTO_TEXT_REPLY = "tools_auto_text_reply"
TOOLS_FORCED_CHOICE = "required"


@dataclass(frozen=True)
class KnownModelBehaviour:
    id: str
    check: str
    artifacts: tuple[tuple[str, str], ...]  # (fingerprint, what it identifies)
    shape: str
    reason: str
    evidence: str
    declared: str

    def artifact_label(self, artifact):
        return dict(self.artifacts).get(artifact)

    def record(self, artifact):
        return {
            "id": self.id,
            "check": self.check,
            "artifact": artifact,
            "artifact_label": self.artifact_label(artifact),
            "shape": self.shape,
            "reason": self.reason,
            "evidence": self.evidence,
            "declared": self.declared,
        }


KNOWN_MODEL_BEHAVIOURS: tuple[KnownModelBehaviour, ...] = (
    KnownModelBehaviour(
        id="muse-glimmer-30b-8bit-tools-auto-asks-for-city",
        check="tools",
        artifacts=(
            (
                "e4f23559c63fcf614116136c1205fd9bf1795e289001c2e2441f783f356c4793",
                "Muse-Glimmer-30B-mlx-8bit (ordinary and prompt-lookup routes)",
            ),
            (
                # Composite identity of the same target with the
                # Muse-Glimmer-30B-DFlash2 drafter and the adapter-default
                # proposal composition (muse_glimmer.py, external-dflash2-v1).
                # Any change to the drafter or its policy changes this value
                # and the exception stops applying (fails closed).
                "116136e41ac5017b9823bcbbd5a8806deaced3bed049694efa3d3ea05a53f065",
                "Muse-Glimmer-30B-mlx-8bit + Muse-Glimmer-30B-DFlash2 (dflash2 route)",
            ),
        ),
        shape=TOOLS_AUTO_TEXT_REPLY,
        reason=(
            "With tool_choice auto and greedy decoding the 8-bit artifact answers "
            "\"Use the weather tool to get the weather in Toronto.\" in text, asking "
            "which city, on every route (same text on ordinary, prompt-lookup and "
            "DFlash2, so it is the target model's choice). The rendered prompt is "
            "intact (402 tokens, weather tool listed, Toronto in the user turn), and "
            "the same model calls weather(city=Toronto) when the call is required "
            "or named. Declared model behaviour by Pierre, 2026-10-08."
        ),
        evidence="qualification/runs/qualify-1007-extra/QUALIFIED.md",
        declared="2026-10-08",
    ),
)


def known_model_behaviour_for(check, artifact):
    """The declared exception for ``check`` on ``artifact``, or ``None``."""
    for entry in KNOWN_MODEL_BEHAVIOURS:
        if entry.check == check and entry.artifact_label(artifact) is not None:
            return entry
    return None


def _single_weather_toronto_call(response):
    try:
        choice = response["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        return (
            choice["finish_reason"] == "tool_calls"
            and len(calls) == 1
            and calls[0]["function"]["name"] == "weather"
            and json.loads(calls[0]["function"]["arguments"]).get("city") == "Toronto"
        )
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def tools_check_passes(response):
    """The ``tools`` check's own pass condition."""
    return _single_weather_toronto_call(response)


def match_shape(shape, evidence):
    """Whether ``evidence`` is exactly the declared failure ``shape``.

    Returns ``(matched, reasons)``; ``reasons`` lists every mismatch.
    """
    if shape != TOOLS_AUTO_TEXT_REPLY:
        return False, [f"unknown shape {shape!r}"]
    reasons = []
    auto = (evidence or {}).get("auto")
    forced = (evidence or {}).get("forced")
    try:
        choice = auto["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if choice["finish_reason"] != "stop":
            reasons.append(f"auto finish_reason {choice['finish_reason']!r}, not 'stop'")
        if message.get("tool_calls"):
            reasons.append("auto response carries tool calls")
        if not (isinstance(content, str) and content.strip()):
            reasons.append("auto response has no text")
    except (KeyError, IndexError, TypeError, AttributeError):
        reasons.append("auto response is malformed")
    if not _single_weather_toronto_call(forced):
        reasons.append(
            "forced (tool_choice required) response is not one weather(city=Toronto) call"
        )
    return not reasons, reasons


def validate_known_check(name, recorded, artifact):
    """Re-verify a receipt's ``known_model_behaviour`` check at load time.

    Returns the declaration's record when the receipt's entry names the
    declared exception for this check and artifact and its evidence still
    matches the declared shape; otherwise ``None``.
    """
    if not isinstance(recorded, dict):
        return None
    if (recorded.get("passed") is not False
            or recorded.get("status") != KNOWN_MODEL_BEHAVIOUR_STATUS):
        return None
    entry = known_model_behaviour_for(name, artifact)
    if entry is None or recorded.get("known_model_behaviour") != entry.record(artifact):
        return None
    matched, _ = match_shape(entry.shape, recorded.get("evidence"))
    return entry.record(artifact) if matched else None
