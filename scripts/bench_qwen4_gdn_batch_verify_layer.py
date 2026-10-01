"""Layer-level timing of the batched fused GDN verify vs the stock chain.

One real Flash-Next GatedDeltaNet layer from the artifact safetensors, a
speculating merged ``ArraysCache`` with per-lane histories, a ``(B, S)``
verify block (full or ragged spans) followed by a per-lane partial-accept
``trim_ragged``, evaluated per iteration.  Arms (stock chain = batch verify
off; batched fused = row_exact) alternate per rep; median ms per verify+trim
and per verify alone.  Explains the full-model A/B; it is not a quoted number.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python \\
      scripts/bench_qwen4_gdn_batch_verify_layer.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP --out t.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 9])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        ap.error("set MLX_ENABLE_TF32=0")

    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models import qwen4_fused_gdn_verify as V
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.models.qwen4_exp import GatedDeltaNet, TextModelArgs

    mx.set_cache_limit(4 << 30)
    V.set_verify_max_steps(17)
    path = Path(a.model).expanduser()
    config = json.loads((path / "config.json").read_text())
    text = TextModelArgs.from_dict(config["text_config"])
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{a.layer}.linear_attn."
    weights = {}
    for shard in sorted({v for k, v in weight_map.items() if k.startswith(prefix)}):
        loaded = mx.load(str(path / shard))
        weights.update({k[len(prefix):]: v for k, v in loaded.items() if k.startswith(prefix)})
    layer = GatedDeltaNet(text)
    quant = config["quantization"]

    def predicate(p, module):
        if not hasattr(module, "to_quantized") or f"{p}.scales" not in weights:
            return False
        per = quant.get(prefix + p)
        return {"group_size": per["group_size"], "bits": per["bits"]} if isinstance(per, dict) else True

    nn.quantize(layer, group_size=quant["group_size"], bits=quant["bits"], class_predicate=predicate)
    layer.load_weights(list(weights.items()), strict=True)
    layer.eval()
    layer.set_fused_gdn_verify_mode("fused")
    layer.set_fused_gdn_decode_mode("fused")
    layer.set_fused_gdn_replay_rollback_mode("compact")
    mx.eval(layer.parameters())
    hidden = int(config["text_config"]["hidden_size"])
    key = mx.random.key(3)
    report = {"layer": a.layer, "mlx": mx.__version__, "cells": {}}
    for rows in a.rows:
        for steps in a.steps:
            for pattern in ("full", "ragged"):
                spans = [steps] * rows if pattern == "full" else [
                    steps if r == 0 else 1 + (5 * r + 3 * steps) % steps for r in range(rows)]
                accepts = [1 + (3 * r) % s for r, s in enumerate(spans)]
                conv = (mx.random.normal((rows, 3, layer.conv_dim), key=key) * 0.5).astype(mx.bfloat16)
                state = mx.random.normal((rows, 48, 128, 128), key=key) * 0.05
                x = (mx.random.normal((rows, steps, hidden), key=key) * 0.8).astype(mx.bfloat16)
                mx.eval(conv, state, x)

                def once(timed_trim=True):
                    cache = ArraysCache(2)
                    cache[0], cache[1] = conv, state
                    cache.start_speculation()
                    cache.prepare(lengths=spans)
                    t0 = time.perf_counter()
                    out = layer(x, mask=cache.make_mask(steps), cache=cache)
                    mx.eval(out, cache[0], cache[1])
                    t1 = time.perf_counter()
                    cache.finalize()
                    cache.trim_ragged([s - m for s, m in zip(spans, accepts)])
                    mx.eval(cache[0], cache[1])
                    t2 = time.perf_counter()
                    return (t1 - t0) * 1e3, (t2 - t0) * 1e3

                per = {"off": {"verify": [], "verify_trim": []},
                       "row_exact": {"verify": [], "verify_trim": []}}
                for arm in per:  # warm-up
                    layer.set_fused_gdn_batch_verify_mode(arm)
                    for _ in range(3):
                        once()
                for rep in range(a.reps):
                    for arm in (("off", "row_exact") if rep % 2 == 0 else ("row_exact", "off")):
                        layer.set_fused_gdn_batch_verify_mode(arm)
                        calls0 = layer.fused_gdn_batch_verify_calls
                        samples = [once() for _ in range(a.iters)]
                        if arm == "row_exact":
                            assert layer.fused_gdn_batch_verify_calls - calls0 == a.iters
                        per[arm]["verify"].append(statistics.median(s[0] for s in samples))
                        per[arm]["verify_trim"].append(statistics.median(s[1] for s in samples))
                cell = {k: {m: statistics.median(v) for m, v in d.items()} for k, d in per.items()}
                cell["spans"] = spans
                cell["ratio_verify_trim"] = cell["row_exact"]["verify_trim"] / cell["off"]["verify_trim"]
                cell["ratio_verify"] = cell["row_exact"]["verify"] / cell["off"]["verify"]
                name = f"B={rows} S={steps} {pattern}"
                report["cells"][name] = cell
                print(name, json.dumps(cell), flush=True)
                mx.clear_cache()
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
