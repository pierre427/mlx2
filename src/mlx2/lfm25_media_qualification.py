"""Pure recomputation of LFM2.5-VL hybrid image/video qualification traces."""

from __future__ import annotations

import hashlib
import json
import math

DECODE_STEPS = 4
MAX_ABS_TOLERANCE = 1e-4
EXPECTED_CONV_LAYERS = 22
EXPECTED_ATTENTION_LAYERS = 8
ATTENTION_LAYERS = frozenset({2, 5, 9, 13, 17, 21, 24, 27})
CONV_LAYERS = frozenset(range(30)) - ATTENTION_LAYERS
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_serving_rows(rows: dict, media_end: int, media_start: int) -> dict:
    """Return derived predicates; the independent loader recomputes these."""
    required = ("cold", "warm1", "warm2", "changed_tail", "changed_lead",
                "changed_pixels", "return_original")
    if set(rows) != set(required):
        raise AssertionError("serving trace has missing or extra arms")
    def receipt(name):
        return rows[name]["receipt"]
    valid = all(
        rows[name]["finish_reason"] in ("length", "stop")
        and receipt(name).get("cache") == "apcv2"
        and receipt(name).get("route") == "ordinary"
        and receipt(name).get("qualification") == "candidate"
        and receipt(name).get("prompt_tokens", 0) > media_end
        for name in required
    )
    cold = receipt("cold").get("cached_tokens") == 0
    cold_count = receipt("cold").get("prompt_tokens")
    warm = all(
        receipt(name).get("cached_tokens") == cold_count - 1
        and receipt(name).get("prompt_tokens") == cold_count
        and receipt(name).get("cache_checkpoint_role") == "committed_prompt_boundary"
        and (rows[name]["output"], rows[name]["reasoning"], rows[name]["finish_reason"])
            == (rows["cold"]["output"], rows["cold"]["reasoning"], rows["cold"]["finish_reason"])
        for name in ("warm1", "warm2", "return_original")
    )
    tail = (type(receipt("changed_tail").get("cached_tokens")) is int
            and media_end <= receipt("changed_tail")["cached_tokens"]
            < receipt("changed_tail")["prompt_tokens"])
    # Only the prefix before the media may be reused once the media or the
    # leading text changed: a restore inside [media_start, media_end) reuses
    # KV computed from the original pixels.
    lead = (type(receipt("changed_lead").get("cached_tokens")) is int
            and receipt("changed_lead")["cached_tokens"]
            <= min(media_start, receipt("changed_lead")["prompt_tokens"] - 1))
    pixels = (type(receipt("changed_pixels").get("cached_tokens")) is int
              and receipt("changed_pixels")["cached_tokens"] <= media_start)
    return {"route_receipts": valid, "cold_apcv2_miss": cold,
            "warm_apcv2_restore": warm, "post_media_branch": tail,
            "changed_leading_text_refuses_media_reuse": lead,
            "changed_pixels_refuses_media_reuse": pixels}



LFM_MEDIA_CHECKS = frozenset({
    "image_placeholder_alignment", "image_apcv2_reuse",
    "video_frame_order_alignment", "video_apcv2_reuse",
    "hybrid_cache_replay", "image_parity", "video_parity",
    "multimodal_image", "multimodal_video", "text_ordinary",
    "batching_if_requested",
})


def _text_ordinary(rows):
    if not isinstance(rows, dict) or set(rows) != {"cold", "warm"}:
        return False
    cold, warm = rows["cold"], rows["warm"]
    if not all(isinstance(row, dict) and isinstance(row.get("receipt"), dict)
               for row in (cold, warm)):
        return False
    return (all(row.get("route") == row["receipt"].get("route") == "ordinary"
                and row.get("qualification") == row["receipt"].get("qualification") == "candidate"
                and row.get("finish_reason") in {"length", "stop"}
                and isinstance(row.get("output"), str) and bool(row["output"])
                for row in (cold, warm))
            and cold["output"] == warm["output"]
            and cold.get("cached_tokens") == 0
            and type(warm.get("prompt_tokens")) is int
            and warm.get("cached_tokens") == warm["prompt_tokens"] - 1)


def _parity_passes(parity: dict, prompt: int) -> bool:
    if not isinstance(parity, dict) or not isinstance(parity.get("decode"), list):
        return False
    if (type(prompt) is not int or parity.get("source_revision") != SOURCE_REVISION
            or len(parity["decode"]) != DECODE_STEPS):
        return False
    rows = [parity.get("prefill"), *parity["decode"]]
    return (all(isinstance(row, dict) and row.get("argmax_match") is True
               and row.get("source_argmax") == row.get("adapter_argmax")
               and isinstance(row.get("shape"), list)
               and len(row["shape"]) == 3 and row["shape"][0] == 1
               and row["shape"][1] == (prompt if index == 0 else 1)
               and type(row["shape"][2]) is int and row["shape"][2] > 1
               and type(row.get("max_abs")) in (int, float)
               and math.isfinite(row["max_abs"])
               and 0 <= row["max_abs"] <= MAX_ABS_TOLERANCE
               for index, row in enumerate(rows))
            and all(row.get("step") == index
                    and row.get("input_token") == rows[index].get("source_argmax")
                    for index, row in enumerate(parity["decode"])))


def _hybrid_passes(parity: dict) -> bool:
    rows = [parity.get("prefill_hybrid_state"), *[
        row.get("hybrid_state") for row in parity.get("decode", [])
    ]]
    return len(rows) == DECODE_STEPS + 1 and all(
        isinstance(row, dict)
        and row.get("shortconv_all_match") is True
        and row.get("attention_offsets_match") is True
        and len(row.get("shortconv", [])) == EXPECTED_CONV_LAYERS
        and len(row.get("attention_kv", [])) == EXPECTED_ATTENTION_LAYERS
        and all(type(state.get("layer")) is int for state in row["shortconv"])
        and all(type(kv.get("layer")) is int for kv in row["attention_kv"])
        and {state.get("layer") for state in row["shortconv"]} == CONV_LAYERS
        and {kv.get("layer") for kv in row["attention_kv"]} == ATTENTION_LAYERS
        and all(type(state.get("layer")) is int
                and state.get("source_size") == state.get("adapter_size")
                and type(state.get("max_abs")) in (int, float)
                and math.isfinite(state["max_abs"])
                and 0 <= state["max_abs"] <= MAX_ABS_TOLERANCE
                for state in row["shortconv"])
        and all(type(kv.get("layer")) is int
                and type(kv.get("source_offset")) is int
                and kv.get("source_offset") == kv.get("adapter_offset")
                for kv in row["attention_kv"])
        for row in rows
    )


def evaluate_lfm_media_report(report: dict) -> dict[str, bool]:
    """Recompute candidate checks from the two arms; never trust flags alone."""
    result = {name: False for name in LFM_MEDIA_CHECKS}
    if not isinstance(report, dict):
        return result
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"image", "video"}:
        return result
    for kind in ("image", "video"):
        arm = arms[kind]
        if not isinstance(arm, dict):
            return result
        fixture, parity, serving = (
            arm.get("fixture"), arm.get("parity"), arm.get("serving")
        )
        if not all(isinstance(value, dict) for value in (fixture, parity, serving)):
            return result
        positions = fixture.get("media_token_positions")
        frames = fixture.get("ordered_frame_sha256")
        changed_frames = fixture.get("changed_ordered_frame_sha256")
        times = fixture.get("ordered_frame_timestamps_seconds")
        changed_times = fixture.get("changed_ordered_frame_timestamps_seconds")
        end, count, prompt = (fixture.get(key) for key in (
            "media_token_end", "media_token_positions_count", "prompt_tokens"
        ))
        aligned = (
            fixture.get("post_media_checkpoint_proof_accepted") is True
            and type(end) is int and type(count) is int and type(prompt) is int
            and 0 < count <= end < prompt
            and isinstance(positions, list) and len(positions) == count
            and all(type(position) is int and 0 <= position < end for position in positions)
            and positions == sorted(set(positions)) and positions[-1] + 1 == end
            and sha256(json.dumps(positions, separators=(",", ":")).encode())
                == fixture.get("media_token_positions_sha256")
            and isinstance(frames, list) and len(frames) >= (2 if kind == "video" else 1)
            and isinstance(changed_frames, list) and len(changed_frames) == len(frames)
            and isinstance(times, list) and isinstance(changed_times, list)
            and (len(times) == len(changed_times) == len(frames) if kind == "video"
                 else times == changed_times == [])
            and (all(type(t) in (int, float) and math.isfinite(t) for t in times + changed_times)
                 and times == sorted(set(times)) and changed_times == sorted(set(changed_times)))
            and all(isinstance(value, str) and len(value) == 64
                    for value in frames + changed_frames)
            and len(set(frames)) == len(frames)
            and frames != changed_frames
            and fixture.get("media_sha256") != fixture.get("changed_media_sha256")
        )
        row_checks = serving.get("derived_checks")
        if not isinstance(row_checks, dict):
            return result
        try:
            recomputed = check_serving_rows(
                {key: serving[key] for key in (
                    "cold", "warm1", "warm2", "changed_tail", "changed_lead",
                    "changed_pixels", "return_original"
                )}, end if type(end) is int else -1,
                positions[0] if isinstance(positions, list) and positions
                and type(positions[0]) is int else -1,
            )
        except (KeyError, TypeError, ValueError):
            return result
        hits = serving.get("apcv2_hits_delta")
        reuse = (all(recomputed.values())
                 and all(row_checks.get(key) is value for key, value in recomputed.items())
                 and type(hits) is int and hits >= 3)
        result[f"{kind}_parity"] = _parity_passes(parity, prompt)
        result["image_placeholder_alignment" if kind == "image" else
               "video_frame_order_alignment"] = aligned
        result[f"{kind}_apcv2_reuse"] = aligned and reuse
        result["hybrid_cache_replay"] = (
            result["hybrid_cache_replay"] or kind == "image"
        ) and _hybrid_passes(parity) and reuse
    result["multimodal_image"] = (result["image_placeholder_alignment"]
                                   and result["image_parity"] and result["image_apcv2_reuse"])
    result["multimodal_video"] = (result["video_frame_order_alignment"]
                                   and result["video_parity"] and result["video_apcv2_reuse"])
    result["text_ordinary"] = _text_ordinary(report.get("text_ordinary"))
    settings = report.get("settings")
    result["batching_if_requested"] = (
        isinstance(settings, dict) and settings.get("max_lanes") == 1
        and settings.get("max_inflight") == 1
        and report.get("batching_requested") is False
    )
    return result
