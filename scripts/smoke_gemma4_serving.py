#!/usr/bin/env python3
"""Candidate mlx2 serving smoke for one local Gemma 4 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")


def _drain(job):
    events = []
    while True:
        event = job.events.get(timeout=120)
        events.append(event)
        if "error" in event:
            raise RuntimeError(str(event))
        if "finish_reason" in event:
            return {
                "finish_reason": event["finish_reason"],
                "receipt": event.get("receipt"),
                "delta_count": sum("delta" in item for item in events),
                "events": len(events),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    owner = json.loads(LOCK.read_text())
    if owner.get("pid") != os.getppid() or not owner.get("cpg_generation"):
        raise RuntimeError("Serving smoke requires cpg_job.py GPU lease and lock")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MLX_ENABLE_TF32"] = "0"
    from smoke_gemma4_adapters import _image_url, _video_url

    from mlx2.serving import ServingEngine

    engine = ServingEngine(
        str(args.model_path), qualification_mode=True, mtp=False,
        max_lanes=2, max_inflight=4, max_context=1024,
        prefill_step=256, cache_bytes=2 << 30,
    )
    try:
        if not engine.ready.wait(90):
            raise TimeoutError(f"Gemma 4 candidate engine did not become ready: {engine.error}")
        if engine.error:
            raise RuntimeError(str(engine.error))
        before = dict(engine.apc.apc_stats)
        prompt = {"prompt": "The capital of France is", "temperature": 0,
                  "max_tokens": 4}
        cold = _drain(engine.submit(prompt))
        warm = _drain(engine.submit(prompt))
        after = dict(engine.apc.apc_stats)
        parallel = [engine.submit({"prompt": value, "temperature": 0, "max_tokens": 4})
                    for value in ("Two plus two is", "Three plus three is")]
        batched = [_drain(job) for job in parallel]
        image_request = {"messages": [{"role": "user", "content": [
            {"type": "input_image", "image_url": _image_url()},
            {"type": "text", "text": "Describe the color."},
        ]}], "temperature": 0, "max_tokens": 2}
        image = _drain(engine.submit(image_request))
        image_repeat = _drain(engine.submit(image_request))
        video = _drain(engine.submit({"messages": [{"role": "user", "content": [
            {"type": "input_video", "video_url": _video_url()},
            {"type": "text", "text": "Describe the clip."},
        ]}], "temperature": 0, "max_tokens": 2}))
        status = engine.status()
        if any(row["finish_reason"] not in {"length", "stop"} for row in (
                cold, warm, *batched, image, image_repeat, video)):
            raise RuntimeError("Candidate serving request had an unexpected finish reason")
        if any(row["receipt"].get("route") != "ordinary" for row in (
                cold, warm, *batched, image, image_repeat, video)):
            raise RuntimeError("Candidate serving request did not use the ordinary route")
        if warm["receipt"].get("cached_tokens", 0) <= 0:
            raise RuntimeError("Repeated text request did not reuse an APCv2 prefix")
        if image_repeat["receipt"].get("cached_tokens", 0) < image["receipt"]["prompt_tokens"] - 1:
            raise RuntimeError("Repeated image request did not reuse the full APCv2 prefix")
        if status["counts"].get("peak_observed_width", 0) < 2:
            raise RuntimeError("Concurrent requests did not run at two-lane compute width")
        receipt = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_path": str(args.model_path.resolve()),
            "adapter_source_sha256": hashlib.sha256(
                (ROOT / "src/mlx2/adapters/gemma4.py").read_bytes()
            ).hexdigest(),
            "gpu_owner": owner,
            "candidate": True,
            "cold": cold,
            "warm": warm,
            "apcv2_before": before,
            "apcv2_after": after,
            "parallel": batched,
            "image": image,
            "image_repeat": image_repeat,
            "video": video,
            "route": cold["receipt"]["route"],
            "qualification": status.get("qualification"),
            "batch_metrics": status.get("batch_runtime"),
            "counts": {k: v for k, v in status["counts"].items() if v},
            "scope": "Candidate ordinary route, APCv2 repeat and two concurrent text requests",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print(json.dumps(receipt, indent=2, default=str), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
