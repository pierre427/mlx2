#!/usr/bin/env python3
"""Kimi MLA prompt-chunk cost versus context depth, one real-weight layer.

Arms: explicit visibility array vs causal offset mode (fused SDPA), the
per-chunk kv_b re-expansion alone, and expanded attention over precomputed
K/V (the lower bound of an expanded cache). B1, no thermal hold.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--contexts", default="8192,32768,131072")
    parser.add_argument("--prefill", default="512,2048")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty benchmark evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"],
        generation=owner["cpg_generation"], cpg_owner_lock=args.cpg_lock)

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.kimi_linear import REPO, REVISION
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.kimi_linear import ModelArgs, MultiLatentAttention

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(8 << 30)
    config = json.loads((args.model/"config.json").read_text())
    index = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    a = ModelArgs.from_dict(config)
    mx.random.seed(7)
    receipt = {"checkpoint": REPO+"@"+REVISION,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
        "ownership": ownership, "device": mx.device_info(), "mlx": mx.__version__,
        "kind": "real-weight single-layer timing and numerics; no thermal hold",
        "qualified": False, "timings": [], "numerics": []}

    def load_layer(module, layer):
        prefix = f"model.layers.{layer}.self_attn."
        weights = {}
        for name in {index[k] for k in index if k.startswith(prefix)}:
            for key, value in mx.load(str(args.model/name)).items():
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix):]
                if "conv1d.weight" in key:
                    key, value = key.replace(".weight", ".conv.weight"), value.moveaxis(2, 1)
                if key == "A_log":
                    value = value.reshape(-1)
                weights[key] = value
        module.load_weights(list(weights.items()), strict=True)
        module.eval()
        mx.eval(module.parameters())

    def timed(name, fn, repeats=5, **extra):
        mx.eval(fn())
        mx.synchronize()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            samples.append(time.perf_counter()-start)
        row = {"name": name, "median_ms": statistics.median(samples)*1e3,
               "min_ms": min(samples)*1e3, "repeats": repeats, **extra}
        receipt["timings"].append(row)
        print(json.dumps(row), flush=True)

    def compare(name, ref, got, **extra):
        mx.eval(ref, got)
        r, g = np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32))
        row = {"name": name, "relative_l2": float(np.linalg.norm(g-r)/max(np.linalg.norm(r), 1e-20)),
               "max_abs": float(np.max(np.abs(g-r))), "finite": bool(np.isfinite(g).all()), **extra}
        receipt["numerics"].append(row)
        print(json.dumps(row), flush=True)

    before = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True).strip()
    receipt["thermal_before"] = subprocess.check_output(["/usr/bin/pmset", "-g", "therm"], text=True).strip()
    try:
        from mlx2.runtime.models.base import scaled_dot_product_attention
        mla = MultiLatentAttention(a)
        load_layer(mla, 3)
        mla.policy = "auto"
        c, h = a.kv_lora_rank, a.num_attention_heads
        n, r, v = a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim
        for context in [int(x) for x in args.contexts.split(",")]:
            hidden = mx.random.normal((1, 4096, a.hidden_size)).astype(mx.bfloat16)
            latent, shared = mx.split(mla.kv_a_proj_with_mqa(hidden), [c], -1)
            latent = mla.kv_a_layernorm(latent)
            reps = (context+4095)//4096
            latent = mx.tile(latent, (1, reps, 1))[:, :context][:, None]
            shared = mx.tile(shared, (1, reps, 1))[:, :context][:, None]
            mx.eval(latent, shared)

            def cache_at():
                cache = KVCache()
                cache.update_and_fetch(latent, shared)
                return cache
            for rows in [int(x) for x in args.prefill.split(",")]:
                x = mx.random.normal((1, rows, a.hidden_size)).astype(mx.bfloat16)
                explicit = mx.arange(context, context+rows)[:, None] >= mx.arange(context+rows)[None, :]
                info = {"context": context, "rows": rows}
                timed("mla_auto_array_mask", lambda: mla(x, mask=explicit, cache=cache_at()), repeats=3, **info)
                timed("mla_auto_causal_mode", lambda: mla(x, mask="causal", cache=cache_at()), repeats=3, **info)
                compare("causal_mode_vs_array", mla(x, mask=explicit, cache=cache_at()),
                        mla(x, mask="causal", cache=cache_at()), **info)
                full = mx.concatenate([latent[:, 0], latent[:, 0, :rows]], axis=1)
                timed("kv_b_reexpansion_only", lambda: mla.kv_b_proj(full), repeats=3, **info)
                kv = mla.kv_b_proj(full).reshape(1, -1, h, n+v).transpose(0, 2, 1, 3)
                keys, values = mx.split(kv, [n], -1)
                sh = mx.concatenate([shared[:, :, :], shared[:, :, :rows]], axis=2)
                keys = mx.concatenate([keys, mx.broadcast_to(sh, (1, h, sh.shape[2], r))], -1)
                mx.eval(keys, values)
                q = mx.random.normal((1, h, rows, n+r)).astype(mx.bfloat16)
                timed("sdpa_precomputed_kv_causal", lambda: scaled_dot_product_attention(
                    q, keys, values, cache=None, scale=mla.scale, mask="causal"), repeats=3, **info)
                kv = keys = values = full = None  # release before the next shape
                mx.clear_cache()
            latent = shared = None
            mx.clear_cache()
        receipt["mla_counters"] = dict(mla.calls)
        receipt["passed"] = all(row["finite"] and row["relative_l2"] < 1e-2 for row in receipt["numerics"])
    except BaseException as error:
        receipt["passed"] = False
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["swap_before"] = before
        receipt["swap_after"] = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True).strip()
        receipt["thermal_after"] = subprocess.check_output(["/usr/bin/pmset", "-g", "therm"], text=True).strip()
        path = args.output/"receipt.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, default=str)+"\n")
        os.replace(tmp, path)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
