"""Pure recomputation of Qwen2.5-VL media and request-owned M-RoPE evidence.

The live producer records observations; its ``checks`` and ``passed`` fields
carry no authority. A route loader may use this evaluator after separately
binding the producer SHA, runtime, artifact, settings, and source revision.
"""

from __future__ import annotations

import hashlib
import json
import math


CHECKS = frozenset({
    "multimodal_image", "multimodal_video", "image_parity", "video_parity",
    "media_token_alignment", "multimodal_state_replay", "multimodal_apcv2_reuse",
})
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
DECODE_STEPS = 4
MAX_ABS = 1e-4
ROW_NAMES = ("cold", "warm1", "warm2", "changed_tail", "changed_lead",
             "changed_pixels", "return_original")


def _hex(value):
    return type(value) is str and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _fixture(value, kind):
    if not isinstance(value, dict):
        return False
    positions = value.get("media_token_positions")
    prompt = value.get("prompt_tokens")
    end = value.get("media_token_end")
    frames = value.get("ordered_frame_sha256")
    changed = value.get("changed_ordered_frame_sha256")
    if (value.get("trusted_media_preparation") is not True
            or type(prompt) is not int or type(end) is not int
            or not 0 < end < prompt or value.get("mrope_media_end") != end
            or not isinstance(positions, list) or not positions
            or any(type(i) is not int for i in positions)
            or positions != sorted(set(positions)) or positions[-1] + 1 != end
            or value.get("media_token_count") != len(positions)
            or value.get("media_token_positions_count") != len(positions)
            or value.get("media_token_positions_sha256") != _digest(positions)
            or type(value.get("image_token_count")) is not int
            or type(value.get("video_token_count")) is not int
            or value["image_token_count"] + value["video_token_count"] != len(positions)
            or (kind == "image" and value["image_token_count"] < 1)
            or (kind == "video" and value["video_token_count"] < 1)
            or not isinstance(frames, list) or len(frames) < (2 if kind == "video" else 1)
            or not isinstance(changed, list) or len(changed) != len(frames)
            or not all(_hex(digest) for digest in frames + changed)
            or frames == changed
            or not all(_hex(value.get(key)) for key in (
                "media_sha256", "changed_media_sha256", "prompt_tokens_sha256",
            ))
            or value["media_sha256"] == value["changed_media_sha256"]
            or not isinstance(value.get("media_fingerprint"), str)
            or not value["media_fingerprint"]):
        return False
    return True


def _logit(row, *, prompt=None):
    return (isinstance(row, dict)
            and row.get("argmax_match") is True
            and row.get("source_argmax") == row.get("adapter_argmax")
            and type(row.get("max_abs")) in (float, int)
            and math.isfinite(row["max_abs"])
            and 0 <= row["max_abs"] <= MAX_ABS
            and isinstance(row.get("shape"), list)
            and len(row["shape"]) == 3 and row["shape"][0] == 1
            and row["shape"][1] == (prompt if prompt is not None else 1)
            and type(row["shape"][2]) is int and row["shape"][2] > 1)


def _parity(value, fixture):
    if not isinstance(value, dict) or not isinstance(fixture, dict):
        return False
    prompt = fixture.get("prompt_tokens")
    decode = value.get("decode")
    rope = value.get("mrope")
    if (value.get("source_revision") != SOURCE_REVISION
            or not _logit(value.get("prefill"), prompt=prompt)
            or value.get("prefill_max_abs") != value["prefill"]["max_abs"]
            or value.get("prefill_argmax_match") is not True
            or not isinstance(decode, list) or len(decode) != DECODE_STEPS
            or any(not _logit(row) or row.get("step") != step
                   or type(row.get("input_token")) is not int
                   for step, row in enumerate(decode))
            or not isinstance(rope, dict)
            or type(rope.get("source_delta")) is not int
            or rope.get("request_owned_prefill") != {
                "length": prompt, "delta": rope["source_delta"],
                "media": True, "media_end": fixture.get("media_token_end"),
            }
            or rope.get("request_owned_decode_length") != prompt + DECODE_STEPS
            or rope.get("model_global_state_cleared") is not True):
        return False
    return (decode[0]["input_token"] == value["prefill"]["source_argmax"]
            and all(decode[step]["input_token"] == decode[step - 1]["source_argmax"]
                    for step in range(1, DECODE_STEPS)))


def _row(value, media_end):
    if not isinstance(value, dict):
        return False
    receipt = value.get("receipt")
    return (isinstance(receipt, dict)
            and receipt.get("cache") == "apcv2"
            and receipt.get("route") == value.get("route") == "ordinary"
            and receipt.get("qualification") == value.get("qualification") == "candidate"
            and receipt.get("prompt_tokens") == value.get("prompt_tokens")
            and receipt.get("cached_tokens") == value.get("cached_tokens")
            and receipt.get("cache_checkpoint_role") == value.get("cache_checkpoint_role")
            and value.get("finish_reason") in ("length", "stop")
            and type(value.get("prompt_tokens")) is int
            and value["prompt_tokens"] > media_end
            and type(value.get("cached_tokens")) is int
            and 0 <= value["cached_tokens"] < value["prompt_tokens"]
            and isinstance(value.get("output"), str)
            and isinstance(value.get("reasoning"), str))


def _serving(value, fixture):
    if not isinstance(value, dict) or not isinstance(fixture, dict):
        return False, False
    end = fixture.get("media_token_end")
    positions = fixture.get("media_token_positions")
    if type(end) is not int or not all(_row(value.get(name), end) for name in ROW_NAMES):
        return False, False
    if not isinstance(positions, list) or not positions or type(positions[0]) is not int:
        return False, False
    # Only the prefix before the media may be reused by a request whose media
    # or leading text changed: a restore landing inside [start, end) reuses
    # KV computed from the original pixels.
    start = positions[0]
    cold = value["cold"]
    prompt = cold["prompt_tokens"]
    same = (cold["output"], cold["reasoning"], cold["finish_reason"])
    replay = all(
        value[name]["prompt_tokens"] == prompt
        and (value[name]["output"], value[name]["reasoning"],
             value[name]["finish_reason"]) == same
        and value[name]["cache_checkpoint_role"] == "committed_prompt_boundary"
        for name in ("warm1", "warm2", "return_original")
    )
    apcv2 = (replay and cold["cached_tokens"] == 0
             and prompt == fixture["prompt_tokens"]
             and all(value[name]["cached_tokens"] == prompt - 1
                     for name in ("warm1", "warm2", "return_original"))
             and end <= value["changed_tail"]["cached_tokens"]
             and value["changed_tail"]["cached_tokens"] < value["changed_tail"]["prompt_tokens"]
             and value["changed_lead"]["cached_tokens"] <= start
             and value["changed_pixels"]["cached_tokens"] <= start
             and type(value.get("apcv2_hits_delta")) is int
             and value["apcv2_hits_delta"] >= 3
             and type(value.get("apcv2_hits_before")) is int
             and type(value.get("apcv2_hits_after")) is int
             and value.get("apcv2_hits_after") - value["apcv2_hits_before"]
                == value["apcv2_hits_delta"])
    return replay, apcv2


def evaluate_qwen25_media_report(report):
    """Recompute the seven media checks without importing MLX or trusting labels."""
    failed = {name: False for name in CHECKS}
    if (not isinstance(report, dict)
            or report.get("schema") != "mlx2.media-serving-qualification.v1"
            or report.get("model_type") != "qwen2_5_vl"
            or report.get("source_revision") != SOURCE_REVISION
            or not isinstance(report.get("arms"), dict)
            or set(report["arms"]) != {"image", "video"}):
        return failed
    evidence = {}
    for kind in ("image", "video"):
        arm = report["arms"][kind]
        if not isinstance(arm, dict):
            return failed
        fixture = arm.get("fixture")
        aligned = _fixture(fixture, kind)
        parity = aligned and _parity(arm.get("parity"), fixture)
        replay, apcv2 = _serving(arm.get("serving"), fixture) if aligned else (False, False)
        runtime = arm.get("mlx_vlm_runtime")
        source = isinstance(runtime, dict) and runtime.get("revision") == SOURCE_REVISION
        binding = arm.get("binding") == {key: report.get(key) for key in (
            "runtime", "artifact", "settings",
        )}
        evidence[kind] = (aligned and source and binding, parity and source,
                          replay and source, apcv2 and source)
    return {
        "multimodal_image": evidence["image"][0] and evidence["image"][2],
        "multimodal_video": evidence["video"][0] and evidence["video"][2],
        "image_parity": evidence["image"][1],
        "video_parity": evidence["video"][1],
        "media_token_alignment": all(evidence[kind][0] for kind in evidence),
        "multimodal_state_replay": all(evidence[kind][2] for kind in evidence),
        "multimodal_apcv2_reuse": all(evidence[kind][3] for kind in evidence),
    }
