#!/usr/bin/env python3
"""Bounded Mellum 2.1 ordinary-route load, decode and APCv2 smoke."""

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
LOCKS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def _gpu_owner() -> dict:
    owners = [json.loads(path.read_text()) for path in LOCKS]
    if owners[0] != owners[1] or owners[0].get("pid") != os.getppid():
        raise RuntimeError("Mellum smoke requires a matching parent-owned two-lock lease")
    return owners[0]


def _drain(job):
    events = []
    while True:
        event = job.events.get(timeout=300)
        events.append(event)
        if "error" in event:
            raise RuntimeError(str(event))
        if "finish_reason" in event:
            deltas = [item.get("delta", {}) for item in events]
            return {
                "finish_reason": event["finish_reason"],
                "receipt": event.get("receipt"),
                "content": "".join(
                    delta.get("content", "")
                    for delta in deltas
                    if isinstance(delta, dict)
                ),
                "reasoning": "".join(
                    delta.get("reasoning_content", "")
                    for delta in deltas
                    if isinstance(delta, dict)
                ),
                "events": len(events),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    owner = _gpu_owner()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MLX_ENABLE_TF32="0",
    )

    from mlx2.adapters.mellum21 import inspect_artifact
    from mlx2.serving import ServingEngine

    artifact = inspect_artifact(args.model_path)
    engine = ServingEngine(
        str(args.model_path),
        mtp=False,
        max_lanes=2,
        max_inflight=4,
        max_context=2048,
        prefill_step=256,
        cache_bytes=1 << 30,
    )
    try:
        if not engine.ready.wait(300):
            raise TimeoutError(f"Mellum engine did not become ready: {engine.error}")
        if engine.error:
            raise RuntimeError(str(engine.error))
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: MLX2_MELLUM_OK",
                }
            ],
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": 32,
        }
        before = dict(engine.apc.apc_stats)
        cold = _drain(engine.submit(request))
        warm = _drain(engine.submit(request))
        after = dict(engine.apc.apc_stats)
        if cold["finish_reason"] not in {"length", "stop"} or warm[
            "finish_reason"
        ] not in {"length", "stop"}:
            raise RuntimeError("Mellum request ended unexpectedly")
        for row in (cold, warm):
            receipt = row.get("receipt") or {}
            if receipt.get("route") != "ordinary":
                raise RuntimeError("Mellum smoke did not use ordinary decode")
            if receipt.get("qualification") != "unqualified":
                raise RuntimeError("Mellum route was not truthfully labelled unqualified")
        if (warm.get("receipt") or {}).get("cached_tokens", 0) <= 0:
            raise RuntimeError("Repeated Mellum request did not reuse an APCv2 prefix")
        status = engine.status()
        receipt = {
            "schema": "mlx2.mellum21-smoke.v1",
            "timestamp": datetime.now(UTC).isoformat(),
            "model_path": str(args.model_path.resolve()),
            "artifact_identity": artifact["identity"],
            "gpu_owner": owner,
            "adapter_source_sha256": hashlib.sha256(
                (ROOT / "src/mlx2/adapters/mellum21.py").read_bytes()
            ).hexdigest(),
            "model_source_sha256": hashlib.sha256(
                (ROOT / "src/mlx2/runtime/models/mellum.py").read_bytes()
            ).hexdigest(),
            "cold": cold,
            "warm": warm,
            "apcv2_before": before,
            "apcv2_after": after,
            "qualification": status.get("qualification"),
            "execution": status.get("execution"),
            "counts": {key: value for key, value in status["counts"].items() if value},
            "scope": "bounded load, ordinary decode and repeated-prefix APCv2 smoke",
            "performance_claim": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print(json.dumps(receipt, indent=2, default=str), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
