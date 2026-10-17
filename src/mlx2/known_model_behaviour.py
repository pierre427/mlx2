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
import re
from dataclasses import dataclass

KNOWN_MODEL_BEHAVIOUR_STATUS = "known_model_behaviour"

# The ``tools`` check's request: one user turn, one weather(city) tool,
# tool_choice auto.  The shape below re-asks the same request with
# tool_choice "required" so a declared text reply is only accepted while the
# same artifact still calls the tool correctly when the call is enforced.
TOOLS_AUTO_TEXT_REPLY = "tools_auto_text_reply"
TOOLS_FORCED_CHOICE = "required"

# Plain-text checks that ask for one exact reply, and that reply.  The
# declared shape is the exact reply written with Markdown backslash escapes
# (``MLX2\\_READY``): the model obeyed and formatted it for Markdown, which
# renders identically.  Only the plain-text form of a check can be excused;
# a source-bound token comparison is a parity check and never is.
EXACT_REPLY_TEXT = {"hermes_client": "HERMES_READY", "cold_text": "MLX2_READY"}
MARKDOWN_ESCAPED_EXACT_REPLY = "markdown_escaped_exact_reply"
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|~<>])")

# A context-ladder lane whose cold and warm requests both completed with the
# same tokens and neither states the needle: the artifact's own retrieval
# miss, reproduced exactly through APCv2, not a serving fault.  Judged by the
# ladder verdict (qualify-1010-correctness/qualification_verdict.py), never
# by the serving qualifier; ``limit`` bounds the missed share of lane-runs.
LADDER_RETRIEVAL_CHECK = "ladder_retrieval"
LADDER_RETRIEVAL_MODEL_MISS = "ladder_retrieval_model_miss"
# Misses under a smaller forced answer are budget exhaustion, not this shape.
LADDER_RETRIEVAL_MIN_MAX_TOKENS = 512


@dataclass(frozen=True)
class KnownModelBehaviour:
    id: str
    check: str
    artifacts: tuple[tuple[str, str], ...]  # (fingerprint, what it identifies)
    shape: str
    reason: str
    evidence: str
    declared: str
    limit: float | None = None
    # Where the behaviour was observed: only these (context, width) cells
    # and at most ``max_count`` instances are excused (None: unrestricted).
    cells: tuple[tuple[int, int], ...] | None = None
    max_count: int | None = None

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
        } | ({"limit": self.limit} if self.limit is not None else {}) | (
            {"cells": [list(cell) for cell in self.cells]} if self.cells is not None else {}
        ) | ({"max_count": self.max_count} if self.max_count is not None else {})


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
    *(
        KnownModelBehaviour(
            id=f"minicpm-o-2_6-{check.replace('_', '-')}-markdown-escaped",
            check=check,
            artifacts=(
                (
                    "312ea377015805b4d83bd10d37105fea90c278ed19b3bd8f46a75ced42145b8f",
                    "openbmb/MiniCPM-o-2_6 (native VLM route)",
                ),
            ),
            shape=MARKDOWN_ESCAPED_EXACT_REPLY,
            reason=(
                f"Asked to reply with exactly {text} (greedy), the model answers "
                f"{text.replace('_', chr(92) + '_')}: the exact text with every "
                "underscore Markdown-escaped, then stops (HERMES and MLX2 probes, "
                "2026-10-10 and 2026-10-14). Declared model behaviour by Pierre, "
                "2026-10-14."
            ),
            evidence="qualification/runs/vlm-companions-20261014/README.md",
            declared="2026-10-14",
        )
        for check, text in EXACT_REPLY_TEXT.items()
    ),
    KnownModelBehaviour(
        id="lfm25-24b-a2b-8bit-ladder-retrieval-miss",
        check=LADDER_RETRIEVAL_CHECK,
        artifacts=(
            (
                "ce07be388dd4e5a6e5f4c48f015c0d1234e23ec73f1637b80bd3e26a36c8bc52",
                "LFM2.5-24B-A2B-MLX-8bit (ordinary route)",
            ),
        ),
        shape=LADDER_RETRIEVAL_MODEL_MISS,
        reason=(
            "On the realistic-artifact ladder with the 512-token reasoning budget "
            "the always-reasoning model misses 3 of 60 lane-runs, each identically "
            "cold and warm: it copies NEEDLE-7D13C8-3 as NELL-7D13C8-3 (1K w4), "
            "writes 'the launch code is NEVER ...' and never recovers the code "
            "(1K w4), or searches 16K of material for 512 tokens without finding "
            "the note (16K w4). Every stated code is otherwise exact, and an earlier "
            "replay reproduced the NEEDLE miscopy at width 1. Declared model "
            "behaviour by Pierre, 2026-10-14."
        ),
        evidence="qualification/runs/qualify-ce017ce9-lfm25-q36int8/README.md",
        declared="2026-10-14",
        limit=0.10,
        # Narrowed 2026-10-16 (sweep ADAPTERS-4, Pierre): only the observed
        # cells and count, so a width-4 serving fault elsewhere still fails.
        cells=((1024, 4), (16384, 4)),
        max_count=3,
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


def _match_markdown_escaped(check, response):
    expected = EXACT_REPLY_TEXT.get(check)
    if expected is None:
        return False, [f"check {check!r} asks for no exact reply"]
    reasons = []
    try:
        choice = response["choices"][0]
        content = choice["message"].get("content")
        if choice["finish_reason"] != "stop":
            reasons.append(f"finish_reason {choice['finish_reason']!r}, not 'stop'")
        if choice["message"].get("tool_calls"):
            reasons.append("response carries tool calls")
        if response["mlx2"]["request_controls"]["thinking"] is not False:
            reasons.append("thinking was not off")
        if not isinstance(content, str):
            reasons.append("response has no text")
        elif content == expected or "\\" not in content:
            reasons.append("reply carries no Markdown escape")
        elif _MARKDOWN_ESCAPE.sub(r"\1", content.strip()) != expected:
            reasons.append(f"unescaped reply {content!r} is not {expected}")
    except (KeyError, IndexError, TypeError, AttributeError):
        reasons.append("response is malformed")
    return not reasons, reasons


def match_shape(shape, evidence, check=None):
    """Whether ``evidence`` is exactly the declared failure ``shape``.

    Returns ``(matched, reasons)``; ``reasons`` lists every mismatch.
    """
    if shape == MARKDOWN_ESCAPED_EXACT_REPLY:
        return _match_markdown_escaped(check, evidence)
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
    matched, _ = match_shape(entry.shape, recorded.get("evidence"), name)
    return entry.record(artifact) if matched else None
