"""Host-only checks for the LFM image/video hybrid evidence producer."""

import importlib.util
from pathlib import Path

import pytest
from mlx2 import lfm25_media_qualification as media_evaluator


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualify_lfm25_media_serving.py"
SPEC = importlib.util.spec_from_file_location("qualify_lfm25_media_serving", SCRIPT)
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


def _state():
    return {
        "shortconv": [
            {"layer": index, "shape": [1, 2, 2048], "max_abs": 0.0,
             "source_size": 0, "adapter_size": 0}
            for index in sorted(media_evaluator.CONV_LAYERS)
        ],
        "attention_kv": [
            {"layer": index, "source_offset": 12, "adapter_offset": 12}
            for index in sorted(media_evaluator.ATTENTION_LAYERS)
        ],
        "shortconv_all_match": True, "attention_offsets_match": True,
    }


def _parity():
    row = {"shape": [1, 12, 128000], "max_abs": 0.0,
           "argmax_match": True, "source_argmax": 101,
           "adapter_argmax": 101}
    return {"prefill": dict(row), "prefill_hybrid_state": _state(),
            "decode": [{**row, "shape": [1, 1, 128000],
                        "hybrid_state": _state(), "step": step,
                        "input_token": 101}
                       for step in range(producer.DECODE_STEPS)],
            "source_revision": media_evaluator.SOURCE_REVISION}


def _serving(end=7):
    def row(cached, *, output="blue"):
        receipt = {"cache": "apcv2", "route": "ordinary",
                   "qualification": "candidate", "prompt_tokens": 12,
                   "cached_tokens": cached,
                   "cache_checkpoint_role": "committed_prompt_boundary"}
        return {"output": output, "reasoning": "", "finish_reason": "length",
                "receipt": receipt, **{name: receipt[name] for name in (
                    "route", "qualification", "prompt_tokens", "cached_tokens",
                    "cache_checkpoint_role")}}
    rows = {"cold": row(0), "warm1": row(11), "warm2": row(11),
            "changed_tail": row(end, output="other"),
            "changed_lead": row(0, output="other"),
            "changed_pixels": row(0, output="other"),
            "return_original": row(11)}
    checks = producer.check_serving_rows(rows, end, 4)
    return {**rows, "derived_checks": checks, "apcv2_hits_delta": 4}


def _arm(kind):
    positions = [4, 5, 6]
    return {
        "fixture": {
            "media_token_end": 7, "media_token_positions": positions,
            "media_token_positions_count": 3, "prompt_tokens": 12,
            "media_token_positions_sha256": producer.sha256(
                b"[4,5,6]"),
            "media_sha256": "a" * 64, "changed_media_sha256": "b" * 64,
            "ordered_frame_sha256": (["c" * 64, "e" * 64] if kind == "video"
                                     else ["c" * 64]),
            "changed_ordered_frame_sha256": (["d" * 64, "f" * 64] if kind == "video"
                                             else ["d" * 64]),
            "ordered_frame_timestamps_seconds": ([0.0, 1.0] if kind == "video" else []),
            "changed_ordered_frame_timestamps_seconds": ([0.0, 1.0] if kind == "video" else []),
            "post_media_checkpoint_proof_accepted": True,
        },
        "parity": _parity(), "serving": _serving(),
    }


def _report():
    text = _serving()
    return {"arms": {kind: _arm(kind) for kind in ("image", "video")},
            "text_ordinary": {"cold": text["cold"], "warm": text["warm1"]},
            "settings": {"max_lanes": 1, "max_inflight": 1},
            "batching_requested": False}


def test_recomputed_lfm_media_checks_cover_both_arms():
    report = _report()
    assert producer.evaluate_lfm_media_report(report) == {
        name: True for name in media_evaluator.LFM_MEDIA_CHECKS
    }

    report["arms"]["video"]["parity"]["decode"][2]["hybrid_state"][
        "shortconv"][0]["max_abs"] = 0.2
    checks = producer.evaluate_lfm_media_report(report)
    assert checks["video_parity"]
    assert not checks["hybrid_cache_replay"]


def test_changed_media_cannot_reuse_prefix_and_missing_frames_fail():
    report = _report()
    report["arms"]["image"]["serving"]["changed_pixels"]["receipt"][
        "cached_tokens"] = 7
    checks = producer.evaluate_lfm_media_report(report)
    assert not checks["image_apcv2_reuse"]

    report = _report()
    report["arms"]["video"]["fixture"]["ordered_frame_sha256"] = []
    checks = producer.evaluate_lfm_media_report(report)
    assert not checks["video_frame_order_alignment"]
    assert not checks["video_apcv2_reuse"]


def test_processor_boundary_must_leave_decode_anchor():
    prepared = {"_mlx2_prompt_tokens": [1, 42, 42, 9],
                "_mlx2_media_token_end": 3,
                "_mlx2_media_fingerprint": "media-id"}
    assert producer.prompt_alignment(prepared, 42)["media_token_end"] == 3
    prepared["_mlx2_prompt_tokens"].pop()
    with pytest.raises(AssertionError, match="boundary"):
        producer.prompt_alignment(prepared, 42)


def test_changed_pixels_reuse_inside_the_media_span_fails():
    """media occupies [4, 7); a restore at 5 or 6 reuses the original pixels'
    KV, which the evaluator accepted as long as it stayed below media_end."""
    for cached in (5, 6):
        report = _report()
        report["arms"]["image"]["serving"]["changed_pixels"]["receipt"]["cached_tokens"] = cached
        assert not producer.evaluate_lfm_media_report(report)["image_apcv2_reuse"]
