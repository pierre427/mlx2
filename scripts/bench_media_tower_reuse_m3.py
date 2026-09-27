"""Bounded direct-model tower reuse timing screen on the pinned M3 artifacts.

Run one family/mode per process under the exclusive CPG gpu:m3 lease. This
measures request preparation plus an evaluated cold prefill, not HTTP serving.
Every timed prompt changes leading text while retaining identical media.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import statistics
import time
from pathlib import Path


ROOT = Path("/tmp/mlx2-candidate-m3.oOgHrM")
LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
SOURCE_REV = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"


def image_uri(size):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (size, size), (31, 92, 174)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def request(uri, index):
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": f"Inspect this image for pass {index}."},
        {"type": "input_image", "image_url": uri},
        {"type": "text", "text": "Name its main color."},
    ]}], "skip_writing_prefix_cache": True, "max_tokens": 1}


def run(family, mode, repeats):
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    if family == "qwen":
        os.environ["MLX2_QWEN25_VISION_TOWER_REUSE_CANDIDATE"] = (
            "1" if mode == "candidate" else "0"
        )
        from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter

        adapter = Qwen25VLCandidateAdapter(str(ROOT / "models/qwen"))
        uri = image_uri(448)
    else:
        from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

        policy = ({"vision_feature_reuse": "tower_only_candidate_v1"}
                  if mode == "candidate" else None)
        adapter = SmolVLM2CandidateAdapter(
            str(ROOT / "models/smol"), execution_policy=policy
        )
        uri = image_uri(32)
    try:
        source = adapter.mlx_vlm_runtime.get("revision")
        if source != SOURCE_REV:
            raise RuntimeError(f"unexpected pinned source revision: {source}")
        rows = []
        for index in range(-2, repeats):
            mx.synchronize()
            started = time.perf_counter_ns()
            prepared = adapter.prepare_multimodal_request(request(uri, index))
            ids = prepared["_mlx2_prompt_tokens"]
            kwargs = dict(prepared["_mlx2_prefill_inputs"])
            if mode == "candidate":
                kwargs = adapter.validate_prefill_inputs(prepared, ids, kwargs)
                if not kwargs.get("_mlx2_vision_feature_verified"):
                    raise RuntimeError("tower-only certificate refused")
            prep_ms = (time.perf_counter_ns() - started) / 1e6
            cache = adapter.model.make_cache()
            prefill_started = time.perf_counter_ns()
            logits = adapter.model(mx.array([ids], dtype=mx.int32), cache=cache, **kwargs)
            mx.eval(logits)
            prefill_ms = (time.perf_counter_ns() - prefill_started) / 1e6
            if index >= 0:
                rows.append({"index": index, "prompt_tokens": len(ids),
                             "prepare_ms": prep_ms, "prefill_ms": prefill_ms,
                             "total_ms": prep_ms + prefill_ms})
        counters = getattr(adapter, "_vision_feature_reuse_counters", None)
        counter_values = vars(counters).copy() if counters is not None else None
        if mode == "candidate" and (
            counter_values["hits"] < repeats
            or counter_values["misses"] != 1
            or counter_values["stores"] != 1
        ):
            raise RuntimeError(f"tower candidate did not reuse features: {counter_values}")
        return {
            "schema": "mlx2.m3-tower-direct-timing.v1",
            "family": family, "mode": mode,
            "source_revision": source, "repeats": repeats,
            "execution_config": adapter.execution_config(max_lanes=1, prefill_step=128),
            "source_sha256": {
                name: hashlib.sha256((ROOT / "src/mlx2" / name).read_bytes()).hexdigest()
                for name in ("adapters/vision_feature_reuse.py",
                             "adapters/pinned_vlm_candidate.py",
                             f"adapters/{'qwen25_vl' if family == 'qwen' else 'smolvlm2'}.py")
            },
            "rows": rows,
            "median_prepare_ms": statistics.median(x["prepare_ms"] for x in rows),
            "median_prefill_ms": statistics.median(x["prefill_ms"] for x in rows),
            "median_total_ms": statistics.median(x["total_ms"] for x in rows),
            "feature_counters": counter_values,
            "feature_cache": adapter.media_feature_cache.snapshot(),
            "limitation": "Direct model cold prefill screen; no APCv2, HTTP, concurrent lanes, or serving throughput.",
        }
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("smol", "qwen"))
    parser.add_argument("mode", choices=("source", "candidate"))
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 20:
        parser.error("repeats must be in [3, 20]")
    if not os.environ.get("MLX2_CPG_GENERATION"):
        parser.error("an exclusive CPG gpu:m3 generation is required")
    fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, f"mlx2 tower timing pid={os.getpid()} generation={os.environ['MLX2_CPG_GENERATION']}\n".encode())
        os.close(fd)
        fd = -1
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(json.dumps(run(args.family, args.mode, args.repeats), indent=2), flush=True)
    finally:
        if fd >= 0:
            os.close(fd)
        LOCK.unlink()


if __name__ == "__main__":
    main()
