"""Bounded M3 tower-only vision reuse and APCv2 coexistence gate.

Usage: PYTHONPATH=/tmp/mlx2-candidate-m3.oOgHrM/src \
    /tmp/mlx2-candidate-m3.oOgHrM/.venv/bin/python \
    scripts/qualify_media_tower_apcv2.py {smol,qwen} {image,video} {source,candidate}

The caller must hold the CPG M3 GPU lease. This script acquires the host lock.
It never changes any launchd service, including StoryForge.
"""

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path("/tmp/mlx2-candidate-m3.oOgHrM")
MODELS = {name: ROOT / "models" / name for name in ("smol", "qwen")}
SIZES = {"smol": 32, "qwen": 64}
LOCK = "/Users/Shared/mlxuag/gpu.lock"
MAX_TOKENS = 8


def image_uri(size, color=(41, 90, 160)):
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (size, size), color).save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def video_uri(size, *, changed=False):
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="mlx2-media-gate-") as directory:
        path = Path(directory) / "two-frames.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (size, size)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV video writer failed")
        try:
            for level in ((35, 80, 210, 240) if not changed else (35, 80, 210, 180)):
                writer.write(np.full((size, size, 3), level, dtype=np.uint8))
        finally:
            writer.release()
        return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


def request_for(kind, media_kind, uri, tail, *, lead="Look at this media."):
    parts = [{"type": "text", "text": lead}]
    if media_kind == "image":
        parts.append({"type": "input_image", "image_url": uri})
    else:
        parts.append({"type": "input_video", "video_url": uri})
    parts.append({"type": "text", "text": tail})
    return {
        "messages": [{"role": "user", "content": parts}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }


def collect(job):
    visible = []
    reasoning = []
    while True:
        event = job.events.get(timeout=180)
        if "error" in event:
            raise RuntimeError(f"serving error: {event}")
        delta = event.get("delta")
        if isinstance(delta, dict):
            visible.append(str(delta.get("content", "")))
            reasoning.append(str(delta.get("reasoning_content", "")))
        elif delta is not None:
            visible.append(str(delta))
        if "finish_reason" in event:
            receipt = event.get("receipt") or {}
            return {
                "output": "".join(visible),
                "reasoning": "".join(reasoning),
                "finish_reason": event["finish_reason"],
                "prompt_tokens": receipt.get("prompt_tokens"),
                "completion_tokens": receipt.get("completion_tokens"),
                "cached_tokens": receipt.get("cached_tokens"),
                "route": receipt.get("route"),
                "qualification": receipt.get("qualification"),
                "cache_checkpoint_role": receipt.get("cache_checkpoint_role"),
                "ordinary_compute_width": receipt.get("ordinary_compute_width"),
            }


def check_row(row, *, cold=False):
    if row["finish_reason"] not in ("length", "stop"):
        raise AssertionError(f"request did not finish: {row['finish_reason']}")
    if row["route"] != "ordinary" or row["qualification"] != "candidate":
        raise AssertionError(f"wrong route or qualification: {row}")
    if not cold and row["cache_checkpoint_role"] != "committed_prompt_boundary":
        raise AssertionError(f"wrong checkpoint role: {row}")
    if type(row["prompt_tokens"]) is not int or row["prompt_tokens"] <= 0:
        raise AssertionError(f"missing prompt token count: {row}")
    if cold and row["cached_tokens"] != 0:
        raise AssertionError(f"cold request unexpectedly reused APCv2: {row}")


def executed_source_root(root=ROOT):
    """The ``mlx2`` package that will run, refusing one outside ``root/src``.

    The report hashes files under ``root/src``; importing ``mlx2`` from
    anywhere else would record hashes of source that never ran.
    """
    import mlx2

    package = Path(mlx2.__file__).resolve().parent
    expected = (Path(root) / "src" / "mlx2").resolve()
    if package != expected:
        raise AssertionError(f"executing mlx2 from {package}, not {expected}")
    return package


def _feature_misses(engine, kind):
    diagnostics = engine.adapter.diagnostics() if hasattr(engine.adapter, "diagnostics") else {}
    counters = (diagnostics.get("qwen25_vision_feature_reuse") if kind == "qwen"
                else diagnostics.get("vision_feature_reuse"))
    return None if not isinstance(counters, dict) else counters.get("misses")


def media_span(adapter, request):
    """(first media position, media end, prompt tokens) of ``request``'s prompt."""
    prepared = adapter.prepare_multimodal_request(request)
    ids = [int(token) for token in prepared["_mlx2_prompt_tokens"]]
    config = adapter.identity["config"]
    media_ids = {int(config[key]) for key in ("image_token_id", "video_token_id")
                 if config.get(key) is not None}
    positions = [index for index, token in enumerate(ids) if token in media_ids]
    if not positions:
        raise AssertionError("prepared prompt has no media placeholders")
    return positions[0], int(prepared["_mlx2_media_token_end"]), len(ids)


def check_media_reuse(changed_tail, changed_lead, changed_pixels, spans):
    """Changed media or leading text may reuse only the prefix before the media.

    ``spans`` maps each arm to ``media_span``.  The previous gate only checked
    that a request reporting zero reuse was cold, so a cache keyed on tokens
    alone (reusing 99 of 100 tokens across different pixels) passed.
    """
    start, _end, _prompt = spans["changed_pixels"]
    if changed_pixels["cached_tokens"] > start:
        raise AssertionError(f"changed pixels reused media KV: {changed_pixels}")
    start, _end, _prompt = spans["changed_lead"]
    if changed_lead["cached_tokens"] > start:
        raise AssertionError(f"changed leading text reused media KV: {changed_lead}")
    _start, end, _prompt = spans["changed_tail"]
    if not end <= changed_tail["cached_tokens"] < changed_tail["prompt_tokens"]:
        raise AssertionError(f"changed tail did not branch after the media: {changed_tail}")


def main(kind, media_kind, mode, report):
    if (kind not in MODELS or media_kind not in ("image", "video")
            or mode not in ("source", "candidate")):
        raise ValueError("usage: {smol,qwen} {image,video} {source,candidate}")
    report.update({
        "family": kind, "media": media_kind, "mode": mode,
        "model": str(MODELS[kind]), "max_tokens": MAX_TOKENS,
        "source_sha256": {
            name: hashlib.sha256((ROOT / "src" / "mlx2" / name).read_bytes()).hexdigest()
            for name in (
                "serving.py", "adapters/pinned_vlm_candidate.py",
                "adapters/vision_feature_reuse.py",
                f"adapters/{'smolvlm2' if kind == 'smol' else 'qwen25_vl'}.py",
            )
        },
    })
    if not MODELS[kind].is_dir():
        raise FileNotFoundError(MODELS[kind])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    policy = None
    if mode == "candidate":
        if kind == "smol":
            policy = {"vision_feature_reuse": "tower_only_candidate_v1"}
        else:
            os.environ["MLX2_QWEN25_VISION_TOWER_REUSE_CANDIDATE"] = "1"

    import mlx.core as mx
    executed_source_root()
    from mlx2.serving import ServingEngine

    mx.set_default_device(mx.gpu)
    uri = image_uri(SIZES[kind]) if media_kind == "image" else video_uri(SIZES[kind])
    changed_uri = (image_uri(SIZES[kind], color=(41, 90, 205))
                   if media_kind == "image" else video_uri(SIZES[kind], changed=True))
    base = request_for(kind, media_kind, uri, "Describe it in one sentence.")
    changed = request_for(kind, media_kind, uri, "Name its colors only.")
    changed_lead = request_for(
        kind, media_kind, uri, "Describe it in one sentence.",
        lead="Please inspect the attached media carefully.",
    )
    changed_pixels = request_for(kind, media_kind, changed_uri, "Describe it in one sentence.")
    engine = ServingEngine(
        str(MODELS[kind]), qualification_mode=True, mtp=False,
        max_lanes=1, max_inflight=1, max_context=4096,
        prefill_step=256, cache_bytes=1 << 30, execution_policy=policy,
    )
    try:
        if not engine.ready.wait(150) or engine.error:
            raise RuntimeError(f"engine load failed: {engine.error}")
        before = dict(engine.apc.apc_stats)
        report["stage"] = "original_cold_and_warm"
        rows = [collect(engine.submit(base)) for _ in range(3)]
        report["runs"] = rows
        report["stage"] = "changed_tail"
        branch = collect(engine.submit(changed))
        report["changed_tail_branch"] = branch
        report["stage"] = "changed_lead"
        prefix_branch = collect(engine.submit(changed_lead))
        report["changed_lead_branch"] = prefix_branch
        report["stage"] = "changed_pixels"
        misses_before = _feature_misses(engine, kind)
        changed_media = collect(engine.submit(changed_pixels))
        misses_after = _feature_misses(engine, kind)
        report["changed_pixels"] = changed_media
        report["stage"] = "return_original"
        returned = collect(engine.submit(base))
        after = dict(engine.apc.apc_stats)
        report.update({
            "runs": rows, "changed_tail_branch": branch,
            "changed_lead_branch": prefix_branch,
            "changed_pixels": changed_media,
            "return_to_original": returned,
            "apcv2_hits_before": before.get("hits", 0),
            "apcv2_hits_after": after.get("hits", 0),
            "adapter_diagnostics": (
                engine.adapter.diagnostics()
                if hasattr(engine.adapter, "diagnostics") else None
            ),
            "feature_cache": engine.adapter.media_feature_cache.snapshot(),
            "execution_config": engine.adapter.execution_config(
                max_lanes=1, prefill_step=256),
        })
        check_row(rows[0], cold=True)
        for row in [*rows[1:], returned]:
            check_row(row)
            if row["prompt_tokens"] != rows[0]["prompt_tokens"]:
                raise AssertionError("identical media prompt changed token count")
            if row["cached_tokens"] != row["prompt_tokens"] - 1:
                raise AssertionError(f"warm APCv2 boundary mismatch: {row}")
            for key in ("output", "reasoning", "finish_reason"):
                if row[key] != rows[0][key]:
                    raise AssertionError(f"warm {key} differs from cold request")
        check_row(branch, cold=branch["cached_tokens"] == 0)
        check_row(prefix_branch, cold=prefix_branch["cached_tokens"] == 0)
        check_row(changed_media, cold=changed_media["cached_tokens"] == 0)
        spans = {name: media_span(engine.adapter, request) for name, request in (
            ("changed_tail", changed), ("changed_lead", changed_lead),
            ("changed_pixels", changed_pixels))}
        report["media_spans"] = spans
        check_media_reuse(branch, prefix_branch, changed_media, spans)
        if after.get("hits", 0) - before.get("hits", 0) < 3:
            raise AssertionError("three repeated/returned warm APCv2 hits missing")
        if mode == "candidate":
            assert report["execution_config"]["vision_feature_reuse"] == "tower_only_candidate_v1"
            counters = (report["adapter_diagnostics"].get("qwen25_vision_feature_reuse")
                        if kind == "qwen" else
                        report["adapter_diagnostics"].get("vision_feature_reuse"))
            report["feature_counters"] = counters
            if (counters is None or counters["misses"] < 1
                    or counters["stores"] < 1 or counters["hits"] < 1):
                raise AssertionError(f"tower feature candidate did not engage: {counters}")
            if misses_after is None or misses_before is None or misses_after <= misses_before:
                raise AssertionError("changed pixels did not miss the tower feature cache")
        report["ok"] = True
    finally:
        engine.close()


if __name__ == "__main__":
    result = {"schema": "mlx2.m3-tower-apcv2.v1", "ok": False}
    lock_fd = None
    lock_owned = False
    try:
        if len(sys.argv) != 4:
            raise ValueError("usage: script {smol,qwen} {image,video} {source,candidate}")
        lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        lock_owned = True
        os.write(lock_fd, (
            f"mlx2 M3 media replay gate pid={os.getpid()} "
            f"CPG generation={os.environ.get('MLX2_CPG_GENERATION', 'unbound')}\n"
        ).encode())
        os.close(lock_fd)
        lock_fd = None
        main(*sys.argv[1:], result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        print(json.dumps(result, indent=2, default=str), flush=True)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_owned:
            try:
                os.unlink(LOCK)
            except FileNotFoundError:
                pass
    sys.exit(0 if result["ok"] else 1)
