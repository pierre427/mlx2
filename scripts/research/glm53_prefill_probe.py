"""One GLM-5.3 chunked prefill + short greedy decode: TTFT, peak memory, logits digest.

For process-level A/B of MLX environment knobs (glm53_env_ab.py --child
glm53_prefill_probe.py): prefill ``--tokens`` in ``--chunk`` pieces as the
replay recorder does, then ``--decode`` serial greedy steps. Records prefill
seconds, the MLX peak memory over the prefill, the last-row logits sha256 and
the greedy tokens (exactness across arms). Research only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=32)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    import os

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text())
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model = adapter.model
    ids = prompt_ids(adapter, args.tokens)
    cache = model.make_cache()
    mx.eval([c.state for c in cache if hasattr(c, "state")] or [mx.zeros(1)])
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for begin in range(0, len(ids), args.chunk):
        piece = mx.array([ids[begin:begin + args.chunk]])
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    prefill_s = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    row = logits[0, -1].astype(mx.float32)
    sha = hashlib.sha256(np.array(row).tobytes()).hexdigest()
    tokens = [int(mx.argmax(row).item())]
    steps = []
    for _ in range(args.decode):
        t1 = time.perf_counter()
        logits = model(mx.array([[tokens[-1]]]), cache, logits_to_keep=1)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        steps.append(time.perf_counter() - t1)
    result = {"schema": "mlx2.glm53-prefill-probe.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "policy": policy, "prompt_tokens": len(ids),
              "chunk": args.chunk, "prefill_seconds": prefill_s, "prefill_tok_s": len(ids) / prefill_s,
              "peak_gb": peak / 1e9, "peak_above_start_gb": (peak - base) / 1e9,
              "prefill_logits_sha256": sha, "tokens": tokens,
              "decode_serial_p50_ms": sorted(steps)[len(steps) // 2] * 1e3 if steps else None,
              "mlx_env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")}}
    (args.out / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("prefill_tok_s", "peak_gb", "peak_above_start_gb",
                                             "prefill_logits_sha256", "decode_serial_p50_ms")}), flush=True)


if __name__ == "__main__":
    main()
