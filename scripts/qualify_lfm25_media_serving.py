#!/usr/bin/env python3
"""Source-bound LFM2.5-VL hybrid image/video serving qualification producer.

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

from mlx2.lfm25_media_qualification import check_serving_rows, evaluate_lfm_media_report


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
    "execution_policy": {"lfm_media_checkpoint": "candidate_v1"},
}
MAX_TOKENS = 8
DECODE_STEPS = 4
MAX_ABS_TOLERANCE = 1e-4
EXPECTED_CONV_LAYERS = 22
EXPECTED_ATTENTION_LAYERS = 8


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

        with tempfile.TemporaryDirectory(prefix="mlx2-lfm-qualification-") as tmp:
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


def media_frame_hashes(kind: str, uri: str) -> dict:
    """Hash decoded frames and their ordered timestamps as the resolver sees them."""
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
    times = list(value.metadata.get("timestamps_seconds", ())) if kind == "video" else []
    if kind == "video" and (len(times) != len(result) or
                            any(not math.isfinite(float(t)) for t in times) or
                            times != sorted(set(times))):
        raise AssertionError("video frame timestamps missing or unordered")
    return {"hashes": result, "timestamps_seconds": [float(t) for t in times]}


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


def hybrid_cache_snapshot(mx, source_cache, adapter_cache, layer_types) -> dict:
    """Compare every ShortConv state plane and every attention KV position."""
    if len(layer_types) != len(source_cache) or len(layer_types) != len(adapter_cache):
        raise AssertionError("LFM cache layer count drift")
    conv, attention = [], []
    for index, (kind, source, candidate) in enumerate(zip(
        layer_types, source_cache, adapter_cache, strict=True
    )):
        if kind == "conv":
            source_state, candidate_state = source[0], candidate[0]
            if source_state is None or candidate_state is None:
                raise AssertionError(f"missing ShortConv state at layer {index}")
            if tuple(source_state.shape) != tuple(candidate_state.shape):
                raise AssertionError(f"ShortConv state shape differs at layer {index}")
            mx.eval(source_state, candidate_state)
            difference = float(mx.max(mx.abs(
                source_state.astype(mx.float32) - candidate_state.astype(mx.float32)
            )).item())
            if not math.isfinite(difference):
                raise AssertionError(f"non-finite ShortConv state at layer {index}")
            conv.append({"layer": index, "shape": list(source_state.shape),
                         "max_abs": difference,
                         "source_size": int(source.size()),
                         "adapter_size": int(candidate.size())})
        elif kind == "full_attention":
            attention.append({"layer": index, "source_offset": int(source.offset),
                              "adapter_offset": int(candidate.offset)})
        else:
            raise AssertionError(f"unknown LFM layer type {kind!r}")
    if len(conv) != EXPECTED_CONV_LAYERS or len(attention) != EXPECTED_ATTENTION_LAYERS:
        raise AssertionError("LFM 22 ShortConv / 8 attention topology drift")
    return {"shortconv": conv, "attention_kv": attention,
            "shortconv_max_abs": max(row["max_abs"] for row in conv),
            "shortconv_all_match": all(row["max_abs"] <= MAX_ABS_TOLERANCE
                                       and row["source_size"] == row["adapter_size"]
                                       for row in conv),
            "attention_offsets_match": all(row["source_offset"] == row["adapter_offset"]
                                            for row in attention)}


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
    """Compare full logits and all hybrid state planes through decode."""
    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache as SourceArraysCache
    from mlx_vlm.models.cache import KVCache as SourceKVCache

    if adapter.mlx_vlm_runtime.get("revision") != SOURCE_REVISION:
        raise AssertionError("source mlx-vlm revision drift")
    source_model = adapter.model._model
    layer_types = adapter.identity["config"]["text_config"]["layer_types"]
    source_cache = [
        SourceKVCache() if kind == "full_attention" else SourceArraysCache(size=1)
        for kind in layer_types
    ]
    adapter_cache = adapter.model.make_cache()
    tokens = list(prepared["_mlx2_prompt_tokens"])
    inputs = mx.array([tokens], dtype=mx.int32)
    kwargs = dict(prepared["_mlx2_prefill_inputs"])
    source_logits = source_model(
        inputs, kwargs.pop("pixel_values", None), None, cache=source_cache, **kwargs
    ).logits
    adapter_logits = adapter.model(
        inputs, cache=adapter_cache,
        pixel_values=prepared["_mlx2_prefill_inputs"]["pixel_values"], **kwargs
    )
    prefill = compare_logits(mx, source_logits, adapter_logits)
    prefill_state = hybrid_cache_snapshot(mx, source_cache, adapter_cache, layer_types)
    if prefill["shape"][:2] != [1, len(tokens)]:
        raise AssertionError("prefill did not cover the complete prompt")
    decode = []
    next_token = prefill["source_argmax"]
    for step in range(DECODE_STEPS):
        token = mx.array([[next_token]], dtype=mx.int32)
        source_logits = source_model(token, None, None, cache=source_cache).logits
        adapter_logits = adapter.model(token, pixel_values=None, cache=adapter_cache)
        row = compare_logits(mx, source_logits, adapter_logits)
        row["hybrid_state"] = hybrid_cache_snapshot(
            mx, source_cache, adapter_cache, layer_types
        )
        row.update({"step": step, "input_token": next_token})
        decode.append(row)
        next_token = row["source_argmax"]
    return {"prefill": prefill, "prefill_hybrid_state": prefill_state,
            "prefill_max_abs": prefill["max_abs"],
            "prefill_argmax_match": prefill["argmax_match"], "decode": decode,
            "source_revision": adapter.mlx_vlm_runtime["revision"]}


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


def run_arm(model_path: str, kind: str) -> dict:
    from mlx2.adapters.lfm25_vl import LFM25VLAdapter
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
    adapter = LFM25VLAdapter(
        model_path, execution_policy={"lfm_media_checkpoint": "candidate_v1"}
    )
    try:
        prepared = adapter.prepare_multimodal_request(base)
        alignment = prompt_alignment(prepared, int(adapter.identity["image_token_id"]))
        if adapter.apc_media_checkpoint_position(
            prepared, prepared["_mlx2_prompt_tokens"], cached_tokens=0
        ) != alignment["media_token_end"]:
            raise AssertionError("adapter refused its bound post-media checkpoint")
        if adapter.fused_shortconv_opt_in_requested or adapter.fused_shortconv_counters.installed:
            raise AssertionError("source ShortConv qualification requires fused opt-in off")
        parity = parity_arm(adapter, prepared)
        artifact = adapter.identity["fingerprint"]
        model_identity = {"files": adapter.identity["files"],
                          "tensor_count": adapter.identity["tensor_count"],
                          "header_sha256": adapter.identity["header_sha256"]}
        runtime = adapter.mlx_vlm_runtime
        shortconv = adapter.diagnostics()["shortconv_fused_candidate"]
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
        text_rows = None
        if kind == "image":
            text_body = {"messages": [{"role": "user", "content":
                          "Explain a compiler in one short sentence."}],
                         "temperature": 0, "max_tokens": MAX_TOKENS}
            text_rows = {"cold": collect(engine.submit(text_body)),
                         "warm": collect(engine.submit(text_body))}
        checks = check_serving_rows(rows, alignment["media_token_end"])
        checks["apcv2_hit_counter"] = after.get("hits", 0) - before.get("hits", 0) >= 3
        if rows["cold"]["receipt"].get("prompt_tokens") != alignment["prompt_tokens"]:
            raise AssertionError("direct/serving prompt token count mismatch")
        return {
            "fixture": {
                "media_sha256": original["media_sha256"],
                "media_bytes": original["media_bytes"],
                "ordered_frame_sha256": original_frames["hashes"],
                "ordered_frame_timestamps_seconds": original_frames["timestamps_seconds"],
                "changed_media_sha256": changed["media_sha256"],
                "changed_ordered_frame_sha256": changed_frames["hashes"],
                "changed_ordered_frame_timestamps_seconds": changed_frames["timestamps_seconds"],
                **alignment,
                "trusted": True,
                "trusted_media_preparation": True,
                "post_media_checkpoint_proof_accepted": True,
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
            "shortconv_reference": shortconv,
            "text_ordinary": text_rows,
        }
    finally:
        engine.close()


def run(model_path: str) -> dict:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    if os.environ.get("MLX2_LFM25_FUSED_SHORTCONV", "0").strip().lower() in {
        "1", "true", "on", "yes"
    }:
        raise RuntimeError("LFM source baseline requires fused ShortConv disabled")
    result = {
        "schema": SCHEMA,
        "family": "lfm2.5-vl",
        "model_type": "lfm2_vl",
        "qualification_harness": {
            "name": "scripts/qualify_lfm25_media_serving.py",
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
    result["text_ordinary"] = result["arms"]["image"]["text_ordinary"]
    result["batching_requested"] = False
    evaluated = evaluate_lfm_media_report(result)
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
            f"mlx2 LFM2.5-VL qualification pid={os.getpid()} "
            f"CPG generation={os.environ['MLX2_CPG_GENERATION']}\n"
        ).encode())
        os.close(fd)
        fd = -1
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = {"schema": SCHEMA, "family": "lfm2.5-vl",
                  "model_type": "lfm2_vl", "passed": False}
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
