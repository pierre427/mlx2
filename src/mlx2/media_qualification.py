"""Pure, fail-closed evaluation of source-bound media qualification traces.

The live producer gathers observations. Both it and the route loader use this
module so a hand-edited ``passed`` field cannot replace missing trace data.
"""

from __future__ import annotations

import math
import hashlib
import json


SMOL_MEDIA_CHECKS = frozenset({
    "multimodal_image", "multimodal_video", "image_parity", "video_parity",
    "media_token_alignment", "multimodal_state_replay", "multimodal_apcv2_reuse",
    "text_source_parity",
})
LOGIT_MAX_ABS = 1e-4
TEXT_TOKENS = 16
NEAR_CONTEXT_TOKENS = 64
TEXT_PROMPTS = {
    "hermes_client": "Reply with exactly HERMES_READY",
    "cold_text": "Reply with exactly MLX2_READY",
}
TEXT_SAMPLING = {"temperature": 0, "repetition_penalty": 1.0,
                 "presence_penalty": 0.0, "frequency_penalty": 0.0}
NEAR_CONTEXT_PROMPT_SHA256 = {
    "near_context": "71615d9b4f60fe2b46ac9b38703ea2125c4ea36863b9596760204d03ed43bd3d",
    "near_context_control": "69ff49655e1104cf95f7d479686ef5488591aeda8697b13c1f7bfe15d9549854",
}


# The identity ``adapters/vlm_runtime.bind_backend`` publishes as
# ``adapter.mlx_vlm_runtime`` and the engine records as ``settings.mlx_vlm``:
# the digest of the executed mlx-vlm source closure plus the reviewed
# reference revision.  The pre-contract ``{version, source, editable,
# revision}`` shape named an install, not the executed bytes, and is refused.
VLM_DEPENDENCY_SCHEMA = "mlx2.vlm-dependencies.v1"


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        digit in "0123456789abcdef" for digit in value
    )


def source_contract_identity(runtime, revision, family=None):
    """True when ``runtime`` is a dependency-content identity bound to
    ``revision`` (and, when given, to the contract ``family``).  Both are the
    reviewed values the caller is written against, never read from the
    report itself."""
    return (isinstance(runtime, dict)
            and runtime.get("schema") == VLM_DEPENDENCY_SCHEMA
            and isinstance(runtime.get("family"), str) and bool(runtime["family"])
            and (family is None or runtime["family"] == family)
            and _sha256(runtime.get("source_sha256"))
            and type(runtime.get("dependency_files")) is int
            and runtime["dependency_files"] > 0
            and isinstance(revision, str) and len(revision) == 40
            and runtime.get("reference_revision") == revision)


def parity_source_bound(parity, runtime, revision):
    """A parity trace is evidence only against the source bytes that produced
    it: its recorded revision and digest must equal the arm's bound identity."""
    return (source_contract_identity(runtime, revision)
            and isinstance(parity, dict)
            and parity.get("source_revision") == revision
            and parity.get("source_sha256") == runtime["source_sha256"])


def arm_source_bound(arm, *, settings, family, revision):
    """An arm is evidence only for the identity the route served under: its
    ``mlx_vlm_runtime`` must equal the report's served ``settings.mlx_vlm``,
    name the family the report qualifies, and bind its parity trace.  A
    self-asserted identity with a matching fabricated digest is not evidence."""
    if not isinstance(arm, dict) or not isinstance(settings, dict):
        return False
    runtime = arm.get("mlx_vlm_runtime")
    return (source_contract_identity(runtime, revision)
            and runtime["family"] == family
            and settings.get("mlx_vlm") == runtime
            and parity_source_bound(arm.get("parity"), runtime, revision))


def _finished(row):
    return (isinstance(row, dict)
            and row.get("route") == "ordinary"
            and row.get("qualification") == "candidate"
            and row.get("finish_reason") in {"length", "stop"}
            and type(row.get("prompt_tokens")) is int
            and row["prompt_tokens"] > 1
            and type(row.get("cached_tokens")) is int
            and row["cached_tokens"] >= 0
            and isinstance(row.get("output"), str))


def _aligned(fixture, kind):
    if not isinstance(fixture, dict):
        return False
    prompt, end, count = (fixture.get(name) for name in (
        "prompt_tokens", "media_token_end", "media_token_positions_count"
    ))
    positions = fixture.get("media_token_positions")
    frames = fixture.get("ordered_frame_sha256")
    changed_frames = fixture.get("changed_ordered_frame_sha256")
    return (fixture.get("trusted") is True
            and type(prompt) is int and type(end) is int and type(count) is int
            and 0 < count <= end < prompt
            and isinstance(positions, list) and len(positions) == count
            and all(type(position) is int and 0 <= position < end for position in positions)
            and positions == sorted(set(positions)) and positions[-1] + 1 == end
            and hashlib.sha256(json.dumps(positions, separators=(",", ":")).encode()).hexdigest()
                == fixture.get("media_token_positions_sha256")
            and isinstance(frames, list) and len(frames) >= (2 if kind == "video" else 1)
            and isinstance(changed_frames, list) and len(changed_frames) == len(frames)
            and all(_sha256(value) for value in frames + changed_frames)
            and _sha256(fixture.get("changed_media_sha256"))
            and fixture["changed_media_sha256"] != fixture.get("media_sha256")
            and all(_sha256(fixture.get(name)) for name in (
                "media_sha256", "prompt_tokens_sha256",
                "media_token_positions_sha256",
            )))


def _parity(parity):
    if not isinstance(parity, dict):
        return False
    rows = [{"max_abs": parity.get("prefill_max_abs"),
             "argmax_match": parity.get("prefill_argmax_match")},
            *(parity.get("decode") or [])]
    return (len(rows) == 5 and all(
        isinstance(row, dict)
        and row.get("argmax_match") is True
        and type(row.get("max_abs")) in (int, float)
        and math.isfinite(row["max_abs"])
        and 0 <= row["max_abs"] <= LOGIT_MAX_ABS
        for row in rows
    ))


def _arm(report, kind):
    arms = report.get("arms")
    arm = arms.get(kind) if isinstance(arms, dict) else None
    return arm if isinstance(arm, dict) else {}


def _text_case(case, prompt, *, max_tokens=TEXT_TOKENS, min_tokens=0):
    if not isinstance(case, dict) or case.get("prompt") != prompt:
        return False
    count = case.get("prompt_tokens")
    prompt_ids = case.get("prompt_token_ids")
    generated = case.get("generated_token_ids")
    stop_ids = case.get("stop_token_ids")
    prefill = case.get("prefill")
    decode = case.get("decode")
    if not (type(count) is int and count > 1
            and isinstance(prompt_ids, list) and len(prompt_ids) == count
            and all(type(value) is int and value >= 0 for value in prompt_ids)
            and hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest()
                == case.get("prompt_tokens_sha256")
            and case.get("max_tokens") == max_tokens
            and case.get("min_tokens") == min_tokens
            and case.get("sampling") == TEXT_SAMPLING
            and isinstance(stop_ids, list)
            and all(type(value) is int and value >= 0 for value in stop_ids)
            and stop_ids == sorted(set(stop_ids))
            and isinstance(generated, list) and len(generated) == max_tokens
            and all(type(value) is int and value >= 0 and value not in stop_ids
                    for value in generated)
            and isinstance(prefill, dict)
            and isinstance(decode, list) and len(decode) == max_tokens - 1):
        return False
    rows = [prefill, *decode]
    if not all(
        isinstance(row, dict)
        and row.get("argmax_match") is True
        and type(row.get("source_argmax")) is int
        and row.get("source_argmax") == row.get("adapter_argmax")
        and type(row.get("max_abs")) in (int, float)
        and math.isfinite(row["max_abs"])
        and 0 <= row["max_abs"] <= LOGIT_MAX_ABS
        and isinstance(row.get("shape"), list)
        and len(row["shape"]) == 3
        and row["shape"][0] == 1
        and row["shape"][1] == (count if index == 0 else 1)
        and type(row["shape"][2]) is int and row["shape"][2] > 1
        and row["source_argmax"] < row["shape"][2]
        and type(row.get("sampled_argmax")) is int
        and 0 <= row["sampled_argmax"] < row["shape"][2]
        and (row["sampled_argmax"] == row["source_argmax"]
             if row["source_argmax"] not in stop_ids or not min_tokens
             else row["sampled_argmax"] != row["source_argmax"])
        and generated[index] == row["sampled_argmax"]
        for index, row in enumerate(rows)
    ):
        return False
    return all(row.get("step") == step
               and row.get("input_token") == generated[step]
               for step, row in enumerate(decode))


def _text_source(report, digest):
    trace = report.get("text_source")
    if not isinstance(trace, dict) or trace.get("artifact") != report.get("artifact"):
        return False
    if not _sha256(digest):
        return False
    cases = trace.get("cases")
    if not isinstance(cases, dict) or set(cases) != (
        set(TEXT_PROMPTS) | set(NEAR_CONTEXT_PROMPT_SHA256)
    ):
        return False
    if not all(_text_case(cases[name], prompt) and
               cases[name].get("source_revision") == report.get("source_revision")
               and cases[name].get("source_sha256") == digest
               for name, prompt in TEXT_PROMPTS.items()):
        return False
    if cases["hermes_client"]["prompt_tokens_sha256"] == cases["cold_text"]["prompt_tokens_sha256"]:
        return False
    for name, prompt_sha in NEAR_CONTEXT_PROMPT_SHA256.items():
        case = cases[name]
        prompt = case.get("prompt")
        if not (isinstance(prompt, str)
                and hashlib.sha256(prompt.encode()).hexdigest() == prompt_sha
                and _text_case(case, prompt, max_tokens=NEAR_CONTEXT_TOKENS,
                               min_tokens=NEAR_CONTEXT_TOKENS)
                and case.get("source_revision") == report.get("source_revision")
                and case.get("source_sha256") == digest
                and 3840 < case["prompt_tokens"] <= 4096 - NEAR_CONTEXT_TOKENS):
            return False
    return (cases["near_context"]["generated_token_ids"]
            != cases["near_context_control"]["generated_token_ids"])


def evaluate_smol_media_report(report):
    """Recompute eight required checks from media and source-text traces."""
    if not isinstance(report, dict):
        return {name: False for name in SMOL_MEDIA_CHECKS}
    arms = {kind: _arm(report, kind) for kind in ("image", "video")}
    aligned = {kind: _aligned(arm.get("fixture"), kind) for kind, arm in arms.items()}
    # ``source_revision`` is pinned by the route validator; here every arm and
    # text trace must be bound to the one served identity for that revision.
    bound = {kind: arm_source_bound(
        arm, settings=report.get("settings"), family="smolvlm",
        revision=report.get("source_revision"),
    ) for kind, arm in arms.items()}
    if not all(bound.values()):
        return {name: False for name in SMOL_MEDIA_CHECKS}
    digest = arms["image"]["mlx_vlm_runtime"]["source_sha256"]
    parity = {kind: _parity(arm.get("parity")) for kind, arm in arms.items()}
    serving = {kind: arm.get("serving") if isinstance(arm.get("serving"), dict)
               else {} for kind, arm in arms.items()}
    completed = {kind: all(_finished(rows.get(name)) for name in (
        "cold", "warm1", "warm2", "changed_tail", "changed_lead",
        "changed_pixels", "return_original",
    )) for kind, rows in serving.items()}

    def replay(kind):
        rows = serving[kind]
        if not completed[kind]:
            return False
        cold = rows["cold"]
        return (all(rows[name]["output"] == cold["output"]
                    for name in ("warm1", "warm2", "return_original"))
                and all(rows[name]["prompt_tokens"] == cold["prompt_tokens"]
                        for name in ("warm1", "warm2", "return_original")))

    def apcv2(kind):
        rows = serving[kind]
        if not completed[kind] or not aligned[kind]:
            return False
        prompt = rows["cold"]["prompt_tokens"]
        media_end = arms[kind]["fixture"]["media_token_end"]
        return (rows["cold"]["cached_tokens"] == 0
                and all(rows[name]["cached_tokens"] == prompt - 1
                        for name in ("warm1", "warm2", "return_original"))
                and media_end <= rows["changed_tail"]["cached_tokens"]
                < rows["changed_tail"]["prompt_tokens"]
                and rows["changed_lead"]["cached_tokens"] == 0
                and rows["changed_pixels"]["cached_tokens"] == 0
                and type(rows.get("apcv2_hits_delta")) is int
                and rows["apcv2_hits_delta"] >= 3)

    return {
        "multimodal_image": aligned["image"] and completed["image"],
        "multimodal_video": aligned["video"] and completed["video"],
        "image_parity": aligned["image"] and parity["image"],
        "video_parity": aligned["video"] and parity["video"],
        "media_token_alignment": all(aligned.values()),
        "multimodal_state_replay": all(replay(kind) for kind in arms),
        "multimodal_apcv2_reuse": all(apcv2(kind) for kind in arms),
        "text_source_parity": _text_source(report, digest),
    }
