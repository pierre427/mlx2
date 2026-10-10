#!/usr/bin/env python3
"""Source-bound Qwen2.5-VL image/video serving qualification producer.

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
FAMILY = "qwen2_5_vl"  # mlx-vlm contract family == report model_type
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
}
MAX_TOKENS = 8
DECODE_STEPS = 4
MAX_ABS_TOLERANCE = 1e-4


def source_contract(adapter) -> dict:
    """The adapter's bound mlx-vlm dependency-content identity, or fail closed.

    ``bind_backend`` verified the executed source closure against the reviewed
    contract for ``SOURCE_REVISION`` and published that identity as
    ``adapter.mlx_vlm_runtime``; the installed-build provenance
    (``adapter.mlx_vlm_build``) is diagnostic and is not a binding.
    """
    from mlx2.media_qualification import source_contract_identity

    runtime = getattr(adapter, "mlx_vlm_runtime", None)
    if not source_contract_identity(runtime, SOURCE_REVISION, family=FAMILY):
        raise AssertionError("source mlx-vlm dependency contract drift")
    return runtime


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(kind: str, *, changed_pixels: bool = False) -> dict:
    """Create a small, bounded fixture; return exact encoded-media identity."""
    if kind == "image":
        from PIL import Image

        stream = io.BytesIO()
        color = (41, 90, 205) if changed_pixels else (41, 90, 160)
        Image.new("RGB", (64, 64), color).save(stream, format="PNG")
        data = stream.getvalue()
        media_type = "image/png"
    elif kind == "video":
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="mlx2-qwen25-qualification-") as tmp:
            path = Path(tmp) / "frames.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 64)
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV video writer failed")
            try:
                levels = (35, 80, 210, 180 if changed_pixels else 240)
                for level in levels:
                    writer.write(np.full((64, 64, 3), level, dtype=np.uint8))
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


def prompt_alignment(prepared: dict, image_token_id: int,
                     video_token_id: int, kind: str) -> dict:
    """Bind Qwen processor placeholders and the request-owned M-RoPE boundary."""
    tokens = prepared["_mlx2_prompt_tokens"]
    if kind not in ("image", "video") or image_token_id == video_token_id:
        raise AssertionError("invalid Qwen media token IDs")
    media_ids = {image_token_id, video_token_id}
    positions = [i for i, token in enumerate(tokens) if token in media_ids]
    selected = image_token_id if kind == "image" else video_token_id
    if not any(tokens[i] == selected for i in positions):
        raise AssertionError("Qwen processor omitted requested media placeholder")
    end = prepared["_mlx2_media_token_end"]
    prefill_end = prepared["_mlx2_prefill_inputs"].get("_mlx2_rope_media_end")
    if (not positions or end != positions[-1] + 1 or end >= len(tokens)
            or prefill_end != end):
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
        "mrope_media_end": prefill_end,
        "image_token_count": sum(tokens[i] == image_token_id for i in positions),
        "video_token_count": sum(tokens[i] == video_token_id for i in positions),
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
    """Compare full logits while separately preserving source-global M-RoPE."""
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache as SourceKVCache

    contract = source_contract(adapter)
    source_model = adapter.model._model
    source_cache = [SourceKVCache() for _ in source_model.language_model.layers]
    adapter_cache = adapter.model.make_cache()
    tokens = list(prepared["_mlx2_prompt_tokens"])
    inputs = mx.array([tokens], dtype=mx.int32)
    kwargs = dict(prepared["_mlx2_prefill_inputs"])
    source_logits = source_model(inputs, cache=source_cache, **kwargs).logits
    language = source_model.language_model
    source_delta = language._rope_deltas
    source_positions = language._position_ids
    if source_delta is None:
        raise AssertionError("pinned Qwen source omitted media RoPE delta")
    source_delta_value = int(source_delta.reshape(-1)[0].item())
    adapter_kwargs = adapter.validate_prefill_inputs(prepared, tokens, kwargs)
    adapter_logits = adapter.model(inputs, cache=adapter_cache, **adapter_kwargs)
    rope = adapter_cache[-1]
    media_end = prepared["_mlx2_media_token_end"]
    expected_row = {"length": len(tokens), "delta": source_delta_value,
                    "media": True, "media_end": media_end}
    if rope.rows != [expected_row]:
        raise AssertionError(f"request-owned media M-RoPE differs from pinned source: {rope.rows}")
    if language._rope_deltas is not None or language._position_ids is not None:
        raise AssertionError("adapter left model-global M-RoPE state behind")
    prefill = compare_logits(mx, source_logits, adapter_logits)
    if prefill["shape"][:2] != [1, len(tokens)]:
        raise AssertionError("prefill did not cover the complete prompt")
    decode = []
    next_token = prefill["source_argmax"]
    for step in range(DECODE_STEPS):
        token = mx.array([[next_token]], dtype=mx.int32)
        # The same source object backs the adapter. Restore the pinned path's
        # independent state for this arm, then ensure the adapter clears it.
        language._rope_deltas = source_delta
        language._position_ids = source_positions
        source_logits = source_model(token, pixel_values=None, cache=source_cache).logits
        adapter_logits = adapter.model(token, pixel_values=None, cache=adapter_cache)
        if rope.rows != [{**expected_row, "length": len(tokens) + step + 1}]:
            raise AssertionError("request-owned M-RoPE drifted during decode")
        if language._rope_deltas is not None or language._position_ids is not None:
            raise AssertionError("adapter decode leaked model-global M-RoPE state")
        row = compare_logits(mx, source_logits, adapter_logits)
        row.update({"step": step, "input_token": next_token})
        decode.append(row)
        next_token = row["source_argmax"]
    return {"prefill": prefill, "prefill_max_abs": prefill["max_abs"],
            "prefill_argmax_match": prefill["argmax_match"], "decode": decode,
            "mrope": {"source_delta": source_delta_value,
                      "request_owned_prefill": expected_row,
                      "request_owned_decode_length": rope.rows[0]["length"],
                      "model_global_state_cleared": True},
            "source_revision": contract["reference_revision"],
            "source_sha256": contract["source_sha256"]}


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


def run_arm(model_path: str, kind: str) -> dict:
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter
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
    adapter = Qwen25VLCandidateAdapter(model_path)
    try:
        prepared = adapter.prepare_multimodal_request(base)
        alignment = prompt_alignment(
            prepared, int(adapter.identity["config"]["image_token_id"]),
            int(adapter.identity["config"]["video_token_id"]), kind,
        )
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
        checks = check_serving_rows(
            rows, alignment["media_token_end"], alignment["media_token_positions"][0]
        )
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
    from mlx2.qwen25_media_qualification import evaluate_qwen25_media_report

    for name in (
        "MLX2_QWEN25_GROUPED_VISION_CANDIDATE",
        "MLX2_QWEN25_VISION_FEATURE_REUSE_CANDIDATE",
        "MLX2_QWEN25_VISION_TOWER_REUSE_CANDIDATE",
    ):
        if os.environ.get(name, "0") != "0":
            raise ValueError(f"source-path qualification requires {name}=0")
    mx.set_default_device(mx.gpu)
    result = {
        "schema": SCHEMA,
        "family": "qwen2.5-vl-3b",
        "model_type": FAMILY,
        "qualification_harness": {
            "name": "scripts/qualify_qwen25_media_serving.py",
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
    evaluated = evaluate_qwen25_media_report(result)
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
            f"mlx2 Qwen2.5-VL qualification pid={os.getpid()} "
            f"CPG generation={os.environ['MLX2_CPG_GENERATION']}\n"
        ).encode())
        os.close(fd)
        fd = -1
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        result = {"schema": SCHEMA, "family": "qwen2.5-vl-3b",
                  "model_type": FAMILY, "passed": False}
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
