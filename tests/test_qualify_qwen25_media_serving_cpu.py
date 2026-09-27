"""CPU-only Qwen media producer and receipt recomputation contract."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from mlx2 import qwen25_media_qualification as evaluator


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


producer = load("qualify_qwen25_media_serving")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def row(cached, *, prompt=10, output="blue"):
    receipt = {
        "cache": "apcv2", "route": "ordinary", "qualification": "candidate",
        "prompt_tokens": prompt, "cached_tokens": cached,
        "cache_checkpoint_role": "committed_prompt_boundary",
    }
    return {"receipt": receipt, "finish_reason": "length", "output": output,
            "reasoning": "", **{key: receipt[key] for key in (
                "route", "qualification", "prompt_tokens", "cached_tokens",
                "cache_checkpoint_role",
            )}}


def serving():
    return {
        "cold": row(0), "warm1": row(9), "warm2": row(9),
        "changed_tail": row(7, output="other"),
        "changed_lead": row(3, output="other"),
        "changed_pixels": row(3, output="other"),
        "return_original": row(9),
        "apcv2_hits_before": 0, "apcv2_hits_after": 4, "apcv2_hits_delta": 4,
    }


def logit(width):
    return {"shape": [1, width, 151936], "max_abs": 0.0,
            "argmax_match": True, "source_argmax": 101,
            "adapter_argmax": 101}


def arm(kind, binding):
    positions = [4, 5, 6]
    fixture = {
        "trusted_media_preparation": True, "prompt_tokens": 10,
        "prompt_tokens_sha256": "a" * 64,
        "media_token_end": 7, "mrope_media_end": 7,
        "media_token_positions": positions,
        "media_token_positions_sha256": digest(positions),
        "media_token_count": 3, "media_token_positions_count": 3,
        "image_token_count": 3 if kind == "image" else 0,
        "video_token_count": 3 if kind == "video" else 0,
        "media_fingerprint": "opaque-proof",
        "media_sha256": "b" * 64, "changed_media_sha256": "c" * 64,
        "ordered_frame_sha256": ["d" * 64] * (2 if kind == "video" else 1),
        "changed_ordered_frame_sha256": ["e" * 64] * (2 if kind == "video" else 1),
    }
    decode = [{**logit(1), "step": i, "input_token": 101} for i in range(4)]
    parity = {
        "prefill": logit(10), "prefill_max_abs": 0.0,
        "prefill_argmax_match": True, "decode": decode,
        "source_revision": evaluator.SOURCE_REVISION,
        "mrope": {
            "source_delta": -2,
            "request_owned_prefill": {
                "length": 10, "delta": -2, "media": True, "media_end": 7,
            },
            "request_owned_decode_length": 14,
            "model_global_state_cleared": True,
        },
    }
    return {"fixture": fixture, "parity": parity, "serving": serving(),
            "binding": binding,
            "mlx_vlm_runtime": {"revision": evaluator.SOURCE_REVISION}}


def report():
    binding = {"runtime": {"source_sha256": "f" * 64},
               "artifact": "artifact-1", "settings": {"max_lanes": 1}}
    return {"schema": producer.SCHEMA, "model_type": "qwen2_5_vl",
            "source_revision": evaluator.SOURCE_REVISION, **binding,
            "arms": {kind: arm(kind, binding) for kind in ("image", "video")}}


def test_prompt_alignment_checks_both_media_ids_and_mrope_boundary():
    prepared = {"_mlx2_prompt_tokens": [1, 42, 42, 9],
                "_mlx2_media_token_end": 3,
                "_mlx2_prefill_inputs": {"_mlx2_rope_media_end": 3},
                "_mlx2_media_fingerprint": "media-id"}
    record = producer.prompt_alignment(prepared, 42, 43, "image")
    assert record["media_token_positions"] == [1, 2]
    assert (record["image_token_count"], record["video_token_count"]) == (2, 0)
    prepared["_mlx2_prefill_inputs"]["_mlx2_rope_media_end"] = 2
    with pytest.raises(AssertionError, match="boundary"):
        producer.prompt_alignment(prepared, 42, 43, "image")
    prepared["_mlx2_prefill_inputs"]["_mlx2_rope_media_end"] = 3
    with pytest.raises(AssertionError, match="omitted"):
        producer.prompt_alignment(prepared, 42, 43, "video")


def test_serving_predicates_allow_leading_prefix_only_before_media():
    rows = {
        key: value for key, value in serving().items() if key in (
            "cold", "warm1", "warm2", "changed_tail", "changed_lead",
            "changed_pixels", "return_original",
        )
    }
    assert all(producer.check_serving_rows(rows, 7).values())
    rows["changed_lead"]["receipt"]["cached_tokens"] = 7
    assert not producer.check_serving_rows(rows, 7)[
        "changed_leading_text_refuses_media_reuse"
    ]


def test_evaluator_recomputes_mrope_parity_and_apcv2_evidence():
    valid = report()
    assert set(evaluator.evaluate_qwen25_media_report(valid)) == evaluator.CHECKS
    assert all(evaluator.evaluate_qwen25_media_report(valid).values())
    for path, replacement in (
        (("arms", "video", "parity", "mrope", "source_delta"), 0),
        (("arms", "image", "parity", "decode", 2, "max_abs"), 0.1),
        (("arms", "video", "fixture", "mrope_media_end"), 6),
        (("arms", "video", "fixture", "ordered_frame_sha256"), []),
        (("arms", "image", "serving", "warm2", "output"), "red"),
        (("arms", "video", "serving", "changed_pixels", "cached_tokens"), 7),
        (("arms", "video", "mlx_vlm_runtime", "revision"), "untrusted"),
        (("arms", "image", "binding", "artifact"), "other"),
    ):
        tampered = copy.deepcopy(valid)
        item = tampered
        for key in path[:-1]:
            item = item[key]
        item[path[-1]] = replacement
        assert not all(evaluator.evaluate_qwen25_media_report(tampered).values()), path


def test_evaluator_fails_closed_on_incomplete_and_type_damaged_receipts():
    for damaged in (None, {}, {"schema": producer.SCHEMA},
                    {**report(), "arms": {"image": report()["arms"]["image"]}}):
        assert not any(evaluator.evaluate_qwen25_media_report(damaged).values())
    damaged = report()
    damaged["arms"]["image"]["serving"]["apcv2_hits_after"] = None
    assert evaluator.evaluate_qwen25_media_report(damaged)["multimodal_apcv2_reuse"] is False
