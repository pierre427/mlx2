#!/usr/bin/env python3
"""Does MLX fuse SDPA for Kimi's expanded 192/128 head dims? Synthetic tensors.

Times and peak memory for qk192/v128 (as served), zero-padded qk256/v256 and
qk128/v128 at Kimi's 32 heads. Caller owns the CPG lease and paired locks.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                cpg_owner_lock=args.cpg_lock)
    import mlx.core as mx
    rows_out = []
    for n, rows in ((8192, 512), (32768, 512), (32768, 2048), (131072, 512), (131072, 2048)):
        q = mx.random.normal((1, 32, rows, 192)).astype(mx.bfloat16)*0.1
        k = mx.random.normal((1, 32, n+rows, 192)).astype(mx.bfloat16)
        v = mx.random.normal((1, 32, n+rows, 128)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        pad = lambda x, d: mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, d-x.shape[-1])])
        arms = {"qk192_v128": (q, k, v, False), "qk192_v128_forced": (q, k, v, True),
                "qk256_v256_padded": (pad(q, 256), pad(k, 256), pad(v, 256), False),
                "qk256_v256_padded_forced": (pad(q, 256), pad(k, 256), pad(v, 256), True),
                "qk192_v192_padded": (q, k, pad(v, 192), False),
                "qk192_v192_padded_forced": (q, k, pad(v, 192), True),
                "qk128_v128": (q[..., :128], k[..., :128], v, False)}
        ref = None
        for name, (a, b, c, forced) in arms.items():
            mx.eval(a, b, c)
            f = lambda: mx.fast.scaled_dot_product_attention(a, b, c, scale=192**-0.5, mask="causal",
                                                             force_fused=forced)
            try:
                mx.eval(f())
            except (ValueError, RuntimeError) as error:
                rows_out.append({"arm": name, "context": n, "rows": rows, "error": str(error)[:200]})
                print(json.dumps(rows_out[-1]), flush=True)
                continue
            mx.eval(f()); mx.synchronize(); mx.reset_peak_memory()
            base = mx.get_active_memory()
            samples = []
            for _ in range(3):
                t = time.perf_counter(); out = f(); mx.eval(out); mx.synchronize(); samples.append(time.perf_counter()-t)
            row = {"arm": name, "context": n, "rows": rows, "median_ms": sorted(samples)[1]*1e3,
                   "peak_extra_gib": (mx.get_peak_memory()-base)/2**30}
            if name != "qk128_v128":
                o = out[..., :128].astype(mx.float32)
                if ref is None:
                    ref = o
                else:
                    row["max_abs_vs_qk192"] = float(mx.abs(o-ref).max())
            rows_out.append(row); print(json.dumps(row), flush=True)
        del q, k, v, arms, ref
        mx.clear_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"receipt.json").write_text(json.dumps({"mlx": mx.__version__, "rows": rows_out}, indent=2))


if __name__ == "__main__":
    main()
