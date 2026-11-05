#!/usr/bin/env python3
"""Real-weight KDA/MLA layer timings and numerics for Kimi Linear policies.

One KDA layer and one MLA layer, B1, no thermal hold: these are bounded
sanity timings for choosing what to A/B end to end, not publishable numbers.
The caller owns the CPG lease and paired GPU locks.
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
    parser.add_argument("--contexts", default="4096,32768,131072")
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
    from mlx2.runtime.models.cache import ArraysCache, KVCache
    from mlx2.runtime.models.kimi_linear import KimiDeltaAttention, ModelArgs, MultiLatentAttention

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
        kda = KimiDeltaAttention(a)
        load_layer(kda, 0)
        kda.prepare_projections()
        mx.eval(kda._joined, kda._second, kda._conv)
        arms = {"kernel": {}, "chunk32": {"chunk32": True}, "chunk_wy": {"chunk_wy": True},
                "kernel+fused_proj+norm": {"fused_projections": True, "fused_norm": True},
                "chunk32+fused_proj+norm": {"chunk32": True, "fused_projections": True, "fused_norm": True}}

        def kda_run(x, flags, state=None):
            for key in ("chunk32", "chunk_wy", "fused_projections", "fused_norm", "reference"):
                setattr(kda, key, bool(flags.get(key, False)))
            cache = ArraysCache(size=4)
            if state is not None:
                for i, value in enumerate(state):
                    cache[i] = value
            return kda(x, cache=cache), cache
        warm_prefix = mx.random.normal((1, 256, a.hidden_size)).astype(mx.bfloat16)
        _, warm = kda_run(warm_prefix, {})
        warm_state = [warm[i] for i in range(4)]
        mx.eval(warm_state)
        for rows in [int(n) for n in args.prefill.split(",")]:
            x = mx.random.normal((1, rows, a.hidden_size)).astype(mx.bfloat16)
            ref, ref_cache = kda_run(x, {"reference": True}, warm_state)
            mx.eval(ref, ref_cache[3])
            for name, flags in arms.items():
                timed(f"kda_prefill_{name}", lambda: kda_run(x, flags, warm_state)[0], rows=rows)
                out, cache = kda_run(x, flags, warm_state)
                compare(f"kda_prefill_{name}_out", ref, out, rows=rows)
                compare(f"kda_prefill_{name}_state", ref_cache[3], cache[3], rows=rows)
        x = mx.random.normal((1, 1, a.hidden_size)).astype(mx.bfloat16)
        for name in ("kernel", "kernel+fused_proj+norm"):
            timed(f"kda_decode_{name}", lambda: kda_run(x, arms[name], warm_state)[0], repeats=20, rows=1)
        del kda

        mla = MultiLatentAttention(a)
        load_layer(mla, 3)
        c, r = a.kv_lora_rank, a.qk_rope_head_dim
        for context in [int(n) for n in args.contexts.split(",")]:
            # Realistic scale: latents pass the checkpoint's own kv_a path.
            hidden = mx.random.normal((1, min(context, 4096), a.hidden_size)).astype(mx.bfloat16)
            latent, shared = mx.split(mla.kv_a_proj_with_mqa(hidden), [c], -1)
            latent = mla.kv_a_layernorm(latent)
            reps = (context+latent.shape[1]-1)//latent.shape[1]
            latent = mx.tile(latent, (1, reps, 1))[:, :context][:, None]
            shared = mx.tile(shared, (1, reps, 1))[:, :context][:, None]
            mx.eval(latent, shared)

            def cache_at():
                cache = KVCache()
                cache.update_and_fetch(latent, shared)
                return cache
            x = mx.random.normal((1, 1, a.hidden_size)).astype(mx.bfloat16)
            outputs = {}
            for policy in ("expanded", "latent_tiled", "latent_fused", "latent_single", "auto"):
                mla.policy = policy
                cache = cache_at()
                mx.eval(cache.state)
                timed(f"mla_decode_{policy}", lambda: mla(x, cache=cache_at()), repeats=10, context=context)
                outputs[policy] = mla(x, cache=cache_at())
            for policy in ("latent_tiled", "latent_fused", "latent_single"):
                compare(f"mla_decode_{policy}_vs_expanded", outputs["expanded"], outputs[policy], context=context)
            for rows in [int(n) for n in args.prefill.split(",")]:
                if context*rows > 32768*2048:
                    continue
                x = mx.random.normal((1, rows, a.hidden_size)).astype(mx.bfloat16)
                mask = mx.arange(context, context+rows)[:, None] >= mx.arange(context+rows)[None, :]
                outputs = {}
                for policy in ("expanded", "latent_tiled", "latent_single", "auto"):
                    mla.policy = policy
                    timed(f"mla_prefill_{policy}", lambda: mla(x, mask=mask, cache=cache_at()),
                          repeats=3, context=context, rows=rows)
                    outputs[policy] = mla(x, mask=mask, cache=cache_at())
                for policy in ("latent_tiled", "latent_single"):
                    compare(f"mla_prefill_{policy}_vs_expanded", outputs["expanded"], outputs[policy],
                            context=context, rows=rows)
            del latent, shared
            mx.clear_cache()
        receipt["mla_counters"] = dict(mla.calls)
        receipt["passed"] = all(row["finite"] for row in receipt["numerics"])
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
