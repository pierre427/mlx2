"""Check a real ordinary route and APCv2 with explicit GPU ownership."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
parser = argparse.ArgumentParser(
    description="Real-artifact route identity and APCv2 smoke; requires owned GPU locks."
)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--i-own-the-gpu", action="store_true")
args = parser.parse_args()
if not args.i_own_the_gpu:
    parser.error("explicit GPU ownership required")
import mlx.core as mx

mx.set_default_device(mx.gpu)
from mlx2.route_identity import recompute_runtime_identity
from mlx2.serving import ServingEngine, runtime_identity

path = args.model
engine = ServingEngine(
    str(path),
    max_lanes=2,
    max_inflight=4,
    max_context=4096,
    default_max_tokens=32,
    cache_bytes=128 << 20,
    mtp=False,
    execution_policy={"decode_first": {"shared_prefill_budget": False}},
    prefill_step=128,
    qualification_mode=True,
)
try:
    deadline = time.monotonic() + 60
    while not engine.ready.wait(0.1):
        if engine.error or time.monotonic() > deadline:
            raise RuntimeError(engine.error or "startup timeout")
    status = engine.status()
    assert status["runtime_source_scope"]["mode"] == "route_source", status[
        "runtime_source_scope"
    ]
    assert (
        recompute_runtime_identity(status["runtime"], build=runtime_identity())
        == status["runtime"]
    )
    results = []
    request = {
        "messages": [{"role": "user", "content": "Reply with the word READY."}],
        "max_tokens": 16,
        "temperature": 0,
    }
    for repeat in range(2):
        job = engine.submit(dict(request))
        events = []
        for _ in range(100):
            event = job.events.get(timeout=30)
            events.append(event)
            if "error" in event:
                raise RuntimeError(event)
            if "finish_reason" in event:
                break
        else:
            raise RuntimeError("no terminal response")
        results.append({"events": events, "cached_tokens": job.cached_tokens})
    # Run once cold and once through APCv2 on the same scoped namespace.
    assert results[1]["cached_tokens"] > 0, results
    assert [x.get("delta") for x in results[0]["events"] if "delta" in x] == [
        x.get("delta") for x in results[1]["events"] if "delta" in x
    ]
    proof = {
        "scope": "real artifact internal serving event queue and APCv2 smoke, not HTTP qualification",
        "device": str(mx.default_device()),
        "runtime": status["runtime"],
        "build_runtime": status["build_runtime"],
        "runtime_source_scope": status["runtime_source_scope"],
        "settings": status["settings"],
        "results": results,
    }
    proof["git_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    proof["git_dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "scope": status["runtime_source_scope"],
                "cached_tokens": [x["cached_tokens"] for x in results],
            }
        )
    )
finally:
    engine.close()
