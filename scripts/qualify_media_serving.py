#!/usr/bin/env python3
"""Source-bound SmolVLM2 image/video serving qualification producer.

Run on the M3 under an exclusive CPG GPU lease. The script acquires the host
GPU lock, uses one model at a time, and emits a JSON candidate evidence record.
It does not select a production route; a separate validator must bind this
record to the exact runtime, artifact, settings, and producer SHA.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile


SCHEMA = "mlx2.media-serving-qualification.v1"
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
SETUP = {
    "qualification_mode": True,
    "mtp": False,
    "max_lanes": 1,
    "max_inflight": 1,
    "max_context": 4096,
    "prefill_step": 256,
    "cache_bytes": 1 << 30,
    "execution_policy": None,
    "route_selection_source": "explicit_flag",
}
MAX_TOKENS = 8
DECODE_STEPS = 4
TEXT_TOKENS = 16
NEAR_CONTEXT_TOKENS = 64
TEXT_PROMPTS = {
    "hermes_client": "Reply with exactly HERMES_READY",
    "cold_text": "Reply with exactly MLX2_READY",
}
MAX_ABS_TOLERANCE = 1e-4


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(kind: str, *, changed_pixels: bool = False) -> dict:
    """Create a small, bounded fixture; return exact encoded-media identity."""
    if kind == "image":
        from PIL import Image

        stream = io.BytesIO()
        color = (41, 90, 205) if changed_pixels else (41, 90, 160)
        Image.new("RGB", (32, 32), color).save(stream, format="PNG")
        data = stream.getvalue()
        media_type = "image/png"
    elif kind == "video":
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="mlx2-smol-qualification-") as tmp:
            path = Path(tmp) / "frames.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 32)
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV video writer failed")
            try:
                levels = (35, 80, 210, 180 if changed_pixels else 240)
                for level in levels:
                    writer.write(np.full((32, 32, 3), level, dtype=np.uint8))
            finally:
                writer.release()
            data = path.read_bytes()
        media_type = "video/mp4"
    else:
        raise ValueError(f"unsupported media kind: {kind}")
    return {
        "uri": f"data:{media_type};base64,{base64.b64encode(data).decode()}",
        "media_sha256": sha256(data),
        "media_bytes": len(data),
    }


def media_frame_hashes(kind: str, uri: str) -> list[str]:
    """Hash decoded frames in order, as the adapter's resolver sees them."""
    import numpy as np
    from mlx2.multimodal import resolve_media

    value = (resolve_media(uri, kind=kind, fps=1, max_frames=16)
             if kind == "video" else resolve_media(uri, kind=kind))
    frames = list(value.value) if kind == "video" else [value.value]
    if not frames:
        raise AssertionError("media resolver produced no frames")
    result = []
    for frame in frames:
        pixels = np.asarray(frame)
        result.append(sha256(
            json.dumps([list(pixels.shape), str(pixels.dtype)], separators=(",", ":")).encode()
            + pixels.tobytes()
        ))
    return result


def request(kind: str, uri: str, tail: str, *, lead: str = "Look at this media.") -> dict:
    media = ({"type": "input_image", "image_url": uri} if kind == "image"
             else {"type": "input_video", "video_url": uri})
    return {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": lead}, media,
            {"type": "text", "text": tail},
        ]}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }


def prompt_alignment(prepared: dict, image_token_id: int) -> dict:
    """Extract an auditable processor token/media boundary receipt."""
    tokens = prepared["_mlx2_prompt_tokens"]
    positions = [i for i, token in enumerate(tokens) if token == image_token_id]
    end = prepared["_mlx2_media_token_end"]
    if (not positions or end != positions[-1] + 1 or end >= len(tokens)):
        raise AssertionError("processor media token boundary is invalid")
    if not prepared.get("_mlx2_media_fingerprint"):
        raise AssertionError("missing media fingerprint")
    return {
        "prompt_tokens": len(tokens),
        "prompt_tokens_sha256": sha256(json.dumps(tokens, separators=(",", ":")).encode()),
        "media_token_positions": positions,
        "media_token_positions_sha256": sha256(
            json.dumps(positions, separators=(",", ":")).encode()
        ),
        "media_token_count": len(positions),
        "media_token_positions_count": len(positions),
        "media_token_end": end,
        "media_fingerprint": prepared["_mlx2_media_fingerprint"],
    }


def compare_logits(mx, source, candidate) -> dict:
    if tuple(source.shape) != tuple(candidate.shape):
        raise AssertionError(f"source/adapter logits shape mismatch: {source.shape}, {candidate.shape}")
    mx.eval(source, candidate)
    difference = float(mx.max(mx.abs(source.astype(mx.float32) - candidate.astype(mx.float32))).item())
    if not math.isfinite(difference):
        raise AssertionError("source/adapter logits contain non-finite differences")
    source_argmax = int(mx.argmax(source[0, -1]).item())
    candidate_argmax = int(mx.argmax(candidate[0, -1]).item())
    return {
        "shape": list(source.shape),
        "max_abs": difference,
        "source_argmax": source_argmax,
        "adapter_argmax": candidate_argmax,
        "argmax_match": source_argmax == candidate_argmax,
    }


def parity_arm(adapter, prepared: dict) -> dict:
    """Compare every prefill logit and four teacher-forced decode vectors."""
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache as SourceKVCache

    if adapter.mlx_vlm_runtime.get("revision") != SOURCE_REVISION:
        raise AssertionError("source mlx-vlm revision drift")
    source_model = adapter.model._model
    source_cache = [SourceKVCache() for _ in source_model.language_model.layers]
    adapter_cache = adapter.model.make_cache()
    tokens = list(prepared["_mlx2_prompt_tokens"])
    inputs = mx.array([tokens], dtype=mx.int32)
    kwargs = dict(prepared["_mlx2_prefill_inputs"])
    source_logits = source_model(inputs, cache=source_cache, **kwargs).logits
    adapter_kwargs = adapter.validate_prefill_inputs(prepared, tokens, kwargs)
    adapter_logits = adapter.model(inputs, cache=adapter_cache, **adapter_kwargs)
    prefill = compare_logits(mx, source_logits, adapter_logits)
    if prefill["shape"][:2] != [1, len(tokens)]:
        raise AssertionError("prefill did not cover the complete prompt")
    decode = []
    next_token = prefill["source_argmax"]
    for step in range(DECODE_STEPS):
        token = mx.array([[next_token]], dtype=mx.int32)
        source_logits = source_model(token, pixel_values=None, cache=source_cache).logits
        adapter_logits = adapter.model(token, pixel_values=None, cache=adapter_cache)
        row = compare_logits(mx, source_logits, adapter_logits)
        row.update({"step": step, "input_token": next_token})
        decode.append(row)
        next_token = row["source_argmax"]
    return {"prefill": prefill, "prefill_max_abs": prefill["max_abs"],
            "prefill_argmax_match": prefill["argmax_match"], "decode": decode,
            "source_revision": adapter.mlx_vlm_runtime["revision"]}


def text_parity_arm(adapter, prompt: str, *, max_tokens: int = TEXT_TOKENS,
                    min_tokens: int = 0) -> dict:
    """Record the source's greedy text tokens and every adapter logit comparison.

    Generic HTTP qualification can compare returned token IDs to this trace
    without assuming a small vision model obeys an exact-word instruction.
    A full bounded token trace prevents early source stop from being mistaken
    for a complete protocol or near-context test.
    """
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache as SourceKVCache

    if adapter.mlx_vlm_runtime.get("revision") != SOURCE_REVISION:
        raise AssertionError("source mlx-vlm revision drift")
    request_body = {"messages": [{"role": "user", "content": prompt}]}
    tokens = list(adapter.prompt_tokens(request_body))
    if len(tokens) <= 1:
        raise AssertionError("text prompt was not rendered")
    source_model = adapter.model._model
    source_cache = [SourceKVCache() for _ in source_model.language_model.layers]
    adapter_cache = adapter.model.make_cache()
    stop_ids = sorted(set(adapter._eos_ids()))

    def selected_id(logits, row):
        if min_tokens == 0 or row["source_argmax"] not in stop_ids:
            return row["source_argmax"]
        scores = logits[0, -1].astype(mx.float32)
        indices = mx.arange(scores.shape[0])
        for stop_id in stop_ids:
            scores = mx.where(indices == stop_id, float("-inf"), scores)
        return int(mx.argmax(scores).item())

    inputs = mx.array([tokens], dtype=mx.int32)
    source_logits = source_model(inputs, pixel_values=None, cache=source_cache).logits
    adapter_logits = adapter.model(inputs, pixel_values=None, cache=adapter_cache)
    prefill = compare_logits(mx, source_logits, adapter_logits)
    if prefill["shape"][:2] != [1, len(tokens)]:
        raise AssertionError("text prefill did not cover the complete prompt")
    prefill["sampled_argmax"] = selected_id(source_logits, prefill)
    generated = [prefill["sampled_argmax"]]
    decode = []
    for step in range(max_tokens - 1):
        input_token = generated[-1]
        token = mx.array([[input_token]], dtype=mx.int32)
        source_logits = source_model(token, pixel_values=None, cache=source_cache).logits
        adapter_logits = adapter.model(token, pixel_values=None, cache=adapter_cache)
        row = compare_logits(mx, source_logits, adapter_logits)
        row["sampled_argmax"] = selected_id(source_logits, row)
        row.update({"step": step, "input_token": input_token})
        decode.append(row)
        generated.append(row["sampled_argmax"])
    return {
        "prompt": prompt,
        "prompt_tokens": len(tokens),
        "prompt_token_ids": tokens,
        "prompt_tokens_sha256": sha256(json.dumps(tokens, separators=(",", ":")).encode()),
        "generated_token_ids": generated,
        "stop_token_ids": stop_ids,
        "prefill": prefill,
        "decode": decode,
        "source_revision": adapter.mlx_vlm_runtime["revision"],
        "sampling": {"temperature": 0, "repetition_penalty": 1.0,
                     "presence_penalty": 0.0, "frequency_penalty": 0.0},
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
    }


def text_source_trace(model_path: str) -> dict:
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter
    import importlib.util

    harness_path = Path(__file__).with_name("qualify_serving.py")
    spec = importlib.util.spec_from_file_location("qualify_serving_prompts", harness_path)
    if spec is None or spec.loader is None:
        raise AssertionError("generic qualification prompt source is unavailable")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    filler = harness.long_context_filler(harness.near_limit_prompt_floor(4096))
    near_prompts = {
        "near_context": (filler + "\nQuestion: What topic should the answer discuss? "
                         "Answer: compiler optimization. Start your answer with compiler."),
        "near_context_control": (filler + "\nQuestion: What is the final secret word? "
                                 "The final secret word is SAPPHIRE. Answer: SAPPHIRE."),
    }

    adapter = SmolVLM2CandidateAdapter(model_path)
    try:
        return {
            "artifact": adapter.identity["fingerprint"],
            "cases": {**{name: text_parity_arm(adapter, prompt)
                         for name, prompt in TEXT_PROMPTS.items()},
                      **{name: text_parity_arm(
                          adapter, prompt, max_tokens=NEAR_CONTEXT_TOKENS,
                          min_tokens=NEAR_CONTEXT_TOKENS,
                      ) for name, prompt in near_prompts.items()}},
        }
    finally:
        adapter.close()


def collect(job) -> dict:
    output, reasoning = [], []
    while True:
        event = job.events.get(timeout=180)
        if "error" in event:
            raise RuntimeError(f"serving error: {event}")
        delta = event.get("delta")
        if isinstance(delta, dict):
            output.append(str(delta.get("content", "")))
            reasoning.append(str(delta.get("reasoning_content", "")))
        elif delta is not None:
            output.append(str(delta))
        if "finish_reason" in event:
            receipt = event.get("receipt") or {}
            return {"output": "".join(output), "reasoning": "".join(reasoning),
                    "finish_reason": event["finish_reason"], "receipt": receipt,
                    **{key: receipt.get(key) for key in (
                        "prompt_tokens", "cached_tokens", "route", "qualification",
                        "cache_checkpoint_role",
                    )}}


def check_serving_rows(rows: dict, media_end: int) -> dict:
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
    lead = (type(receipt("changed_lead").get("cached_tokens")) is int
            and receipt("changed_lead")["cached_tokens"]
            < min(media_end, receipt("changed_lead")["prompt_tokens"]))
    pixels = (type(receipt("changed_pixels").get("cached_tokens")) is int
              and receipt("changed_pixels")["cached_tokens"] < media_end)
    return {"route_receipts": valid, "cold_apcv2_miss": cold,
            "warm_apcv2_restore": warm, "post_media_branch": tail,
            "changed_leading_text_refuses_media_reuse": lead,
            "changed_pixels_refuses_media_reuse": pixels}


def run_arm(model_path: str, kind: str) -> dict:
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter
    from mlx2.serving import ServingEngine

    original = fixture(kind)
    changed = fixture(kind, changed_pixels=True)
    base = request(kind, original["uri"], "Describe it in one sentence.")
    inputs = {
        "cold": base,
        "warm1": base,
        "warm2": base,
        "changed_tail": request(kind, original["uri"], "Name its colors only."),
        "changed_lead": request(kind, original["uri"], "Describe it in one sentence.",
                                lead="Please inspect the attached media carefully."),
        "changed_pixels": request(kind, changed["uri"], "Describe it in one sentence."),
        "return_original": base,
    }
    adapter = SmolVLM2CandidateAdapter(model_path)
    try:
        prepared = adapter.prepare_multimodal_request(base)
        alignment = prompt_alignment(prepared, int(adapter.identity["config"]["image_token_id"]))
        if not adapter.has_trusted_media_preparation(prepared):
            raise AssertionError("adapter refused its own media preparation proof")
        parity = parity_arm(adapter, prepared)
        artifact = adapter.identity["fingerprint"]
        model_identity = {"files": adapter.identity["files"],
                          "tensor_count": adapter.identity["tensor_count"],
                          "vision_present": adapter.identity["vision_present"]}
        runtime = adapter.mlx_vlm_runtime
    finally:
        adapter.close()
    original_frames = media_frame_hashes(kind, original["uri"])
    changed_frames = media_frame_hashes(kind, changed["uri"])
    engine = ServingEngine(model_path, **SETUP)
    try:
        if not engine.ready.wait(150) or engine.error:
            raise RuntimeError(f"engine load failed: {engine.error}")
        if engine.snapshot["artifact"] != artifact:
            raise AssertionError("direct/serving artifact mismatch")
        before = dict(engine.apc.apc_stats)
        rows = {name: collect(engine.submit(body)) for name, body in inputs.items()}
        after = dict(engine.apc.apc_stats)
        checks = check_serving_rows(rows, alignment["media_token_end"])
        checks["apcv2_hit_counter"] = after.get("hits", 0) - before.get("hits", 0) >= 3
        if rows["cold"]["receipt"].get("prompt_tokens") != alignment["prompt_tokens"]:
            raise AssertionError("direct/serving prompt token count mismatch")
        return {
            "fixture": {
                "media_sha256": original["media_sha256"],
                "media_bytes": original["media_bytes"],
                "ordered_frame_sha256": original_frames,
                "changed_media_sha256": changed["media_sha256"],
                "changed_ordered_frame_sha256": changed_frames,
                **alignment,
                "trusted": True,
                "trusted_media_preparation": True,
            },
            "parity": parity,
            "serving": {**rows, "apcv2_hits_before": before.get("hits", 0),
                        "apcv2_hits_after": after.get("hits", 0),
                        "apcv2_hits_delta": after.get("hits", 0) - before.get("hits", 0),
                        "derived_checks": checks},
            "binding": {"runtime": engine.snapshot["runtime"],
                        "artifact": engine.snapshot["artifact"],
                        "settings": engine.snapshot["settings"]},
            "model_identity": model_identity,
            "mlx_vlm_runtime": runtime,
        }
    finally:
        engine.close()


def run(model_path: str) -> dict:
    import mlx.core as mx
    from mlx2.media_qualification import evaluate_smol_media_report

    mx.set_default_device(mx.gpu)
    result = {
        "schema": SCHEMA,
        "family": "smolvlm2",
        "model_type": "smolvlm",
        "qualification_harness": {
            "name": "scripts/qualify_media_serving.py",
            "sha256": sha256(Path(__file__).read_bytes()),
        },
        "source_revision": SOURCE_REVISION,
        "settings_request": dict(SETUP),
        "model_path": str(Path(model_path).resolve()),
        "arms": {}, "checks": {}, "passed": False,
    }
    for kind in ("image", "video"):
        result["arms"][kind] = run_arm(model_path, kind)
    bindings = [result["arms"][kind]["binding"] for kind in ("image", "video")]
    if bindings[0] != bindings[1]:
        raise AssertionError("image/video runtime, artifact, or settings mismatch")
    result.update(bindings[0])
    result["text_source"] = text_source_trace(model_path)
    if result["text_source"]["artifact"] != result["artifact"]:
        raise AssertionError("text/media source artifact mismatch")
    evaluated = evaluate_smol_media_report(result)
    result["checks"] = {
        name: {"passed": passed, "evidence": "recomputed_from_image_video_arms"}
        for name, passed in sorted(evaluated.items())
    }
    result["passed"] = (all(evaluated.values()) and all(
        all(arm["serving"]["derived_checks"].values())
        for arm in result["arms"].values()
    ))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args(argv)
    if not os.environ.get("MLX2_CPG_GENERATION"):
        parser.error("an exclusive CPG gpu:m3 generation is required")
    if not args.model_path.is_dir():
        parser.error(f"model artifact missing: {args.model_path}")
    fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, (
            f"mlx2 SmolVLM2 qualification pid={os.getpid()} "
            f"CPG generation={os.environ['MLX2_CPG_GENERATION']}\n"
        ).encode())
        os.close(fd)
        fd = -1
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = {"schema": SCHEMA, "family": "smolvlm2",
                  "model_type": "smolvlm", "passed": False}
        try:
            result = run(str(args.model_path))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, default=str), flush=True)
        return 0 if result["passed"] else 1
    finally:
        if fd >= 0:
            os.close(fd)
        LOCK.unlink()


if __name__ == "__main__":
    sys.exit(main())
