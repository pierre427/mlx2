#!/usr/bin/env python3
"""Kimi MLA NAX flash kernels: numerics and timings. Unqualified research.

Phases (one per GPU unit):
  smoke  small-shape numerics of both kernels against their FP32 ops laws,
         hidden-row zeros, split invariance; no timing.
  bench  synthetic BF16 B1/32-head kernel timings against the padded-256 and
         D128 fused SDPA baselines and the latent_single decode ops law.
  layer  one real-weight MLA layer (index 3): module policies' numerics and
         per-layer timings, ``auto`` with and without the flash sub-routes.
Caller owns the CPG lease and paired GPU locks; no thermal hold.
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
    parser.add_argument("--phase", choices=("smoke", "bench", "layer", "review", "crossover"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/Models/Huihui-Kimi-Linear-REAP-35B-A3B-Instruct-abliterated"))
    parser.add_argument("--shapes", default="8192x512,32768x512,32768x2048,131072x512,131072x2048")
    parser.add_argument("--decode", default="4096,32768,131072")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--parts", default="expanded,absorbed,decode")
    parser.add_argument("--review-cells", default=(  # batch x context x rows
        "1x4096x1,1x32768x1,1x131072x1,1x32768x16,1x131072x16,1x131072x128,4x32768x1,4x131072x1"))
    parser.add_argument("--crossover-cells", default=(  # B<batch> x context x rows/rows/...
        "B1x8192x1/16/64/128/256/512,B1x32768x1/16/64/128/256/512,B1x131072x1/16/64/128/256/512,"
        "B4x8192x1/16/64/128/256,B4x32768x1/16/64/128/256,B4x131072x1/16/64/128,"
        "B1x524288x128/256/512"))
    parser.add_argument("--layer-cells", default=(
        "8192x1,8192x16,8192x64,8192x512,32768x1,32768x4,32768x16,32768x32,32768x64,32768x128,"
        "32768x512,32768x2048,131072x1,131072x4,131072x16,131072x32,131072x64,131072x128,"
        "131072x512,131072x2048"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"scripts"))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"],
        generation=owner["cpg_generation"], cpg_owner_lock=args.cpg_lock)

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import kimi_linear_mla_flash as flash

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(8 << 30)
    snapshot = root/".snapshot-rev"
    if snapshot.exists():
        # `git archive` export of a commit, so edits elsewhere cannot leak in.
        revision, dirty = snapshot.read_text().strip(), False
    else:
        git = lambda *a: subprocess.check_output(["git", "-C", str(root), *a], text=True).strip()
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    receipt = {"phase": args.phase, "source_revision": revision, "source_dirty": dirty,
               "source_root": str(root), "ownership": ownership, "device": mx.device_info(), "mlx": mx.__version__,
               "kind": "unqualified kernel research; no thermal hold", "qualified": False,
               "numerics": [], "timings": []}
    swap = lambda: subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True).strip()
    receipt["swap_before"] = swap()

    def compare(name, ref, got, **extra):
        mx.eval(ref, got)
        r, g = np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32))
        row = {"name": name, "max_abs": float(np.max(np.abs(g-r))) if r.size else 0.0,
               "relative_l2": float(np.linalg.norm(g-r)/max(np.linalg.norm(r), 1e-20)),
               "ref_max_abs": float(np.max(np.abs(r))) if r.size else 0.0,
               "finite": bool(np.isfinite(g).all()), **extra}
        receipt["numerics"].append(row)
        print(json.dumps(row), flush=True)
        return row

    def timed(name, fn, repeats=args.repeats, **extra):
        mx.eval(fn())
        mx.synchronize()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            samples.append(time.perf_counter()-start)
        row = {"name": name, "median_ms": statistics.median(samples)*1e3, "min_ms": min(samples)*1e3,
               "repeats": repeats, **extra}
        receipt["timings"].append(row)
        print(json.dumps(row), flush=True)
        return row

    H, N, R, V, C = flash.HEADS, flash.NOPE, flash.SHARED, flash.VDIM, flash.LATENT
    scale = (N+R)**-0.5
    rand = lambda shape, s=1.0: (mx.random.normal(shape)*s).astype(mx.bfloat16)
    mx.random.seed(11)
    try:
        if args.phase == "smoke":
            receipt["available"] = {k: flash.available(k) for k in ("expanded", "absorbed")}
            print(json.dumps(receipt["available"]), flush=True)
            for b, t, s, mask, off in ((1, 1, 1, None, None), (1, 5, 37, "causal", None),
                                       (1, 64, 64, "causal", None), (1, 77, 300, "causal", None),
                                       (2, 40, 100, "causal", None), (1, 130, 1000, "causal", None),
                                       (1, 33, 90, None, None), (1, 24, 60, "causal", -8)):
                q, kv, sh = rand((b, t, H*(N+R)), 0.5), rand((b, s, H*(N+V))), rand((b, 1, s, R))
                ref = flash.expanded_reference(q, kv, sh, mask, scale=scale, offset=off)
                kvh = mx.contiguous(kv.reshape(b, s, H, N+V).transpose(0, 2, 1, 3))
                for layout, operand in (("tokens", kv), ("heads", kvh)):
                    info = {"batch": b, "rows": t, "width": s, "mask": mask, "offset_override": off,
                            "layout": layout}
                    got = flash.expanded(q, operand, sh, mask, scale=scale, offset=off)
                    compare("expanded_vs_fp32", ref, got, **info)
                    if off is not None:
                        hidden = got.reshape(b, t, H, V)[:, :-off]
                        compare("expanded_hidden_rows_zero", mx.zeros_like(hidden), hidden, **info)
                    if layout == "heads":
                        _, lse = flash.expanded(q, operand, sh, mask, scale=scale, offset=off, return_lse=True)
                        _, ref_lse = flash.expanded_reference(q, operand, sh, mask, scale=scale, offset=off,
                                                              return_lse=True)
                        finite = mx.isfinite(ref_lse)
                        compare("expanded_lse_vs_fp32", mx.where(finite, ref_lse, 0), mx.where(finite, lse, 0),
                                **info)
                        if off is not None:
                            receipt["numerics"][-1]["hidden_lse_neg_inf"] = bool(
                                mx.all(mx.isneginf(lse[:, :-off])))
            for b, t, s, mask, window in ((1, 40, 300, "causal", 128), (2, 70, 1000, "causal", 256),
                                          (1, 16, 500, None, 128)):
                q, lat, sh = rand((b, t, H*(N+R)), 0.5), rand((b, 1, s, C)), rand((b, 1, s, R))
                wkv = rand((H*(N+V), C), 0.04)
                proj = lambda z: z @ wkv.T
                ref = flash.expanded_reference(q, proj(lat[:, 0]), sh, mask, scale=scale)
                got = flash.expanded_windows(q, lat, sh, proj, mask, scale=scale, window=window)
                compare("expanded_windows_vs_fp32", ref, got, batch=b, rows=t, width=s, mask=mask,
                        window=window)
            for b, t, s, mask, split, off in ((1, 1, 1, None, None, None), (1, 1, 300, None, None, None),
                                              (1, 1, 300, None, 128, None), (1, 4, 1000, "causal", None, None),
                                              (2, 3, 700, "causal", 256, None), (1, 40, 100, "causal", None, None),
                                              (1, 17, 5000, "causal", None, None), (1, 17, 5000, "causal", 128, None),
                                              (1, 12, 40, "causal", None, -5)):
                q, qs = rand((b, H, t, C), 0.05), rand((b, H, t, R), 0.5)
                lat, sh = rand((b, 1, s, C)), rand((b, 1, s, R))
                ref = flash.absorbed_reference(q, qs, lat, sh, mask, scale=scale, offset=off)
                for variant in flash.ABSORBED_VARIANTS:
                    info = {"batch": b, "rows": t, "width": s, "mask": mask, "split": split,
                            "offset_override": off, "variant": variant}
                    got = flash.absorbed(q, qs, lat, sh, mask, scale=scale, split=split, offset=off,
                                         variant=variant)
                    compare("absorbed_vs_fp32", ref, got, **info)
                    if off is not None:
                        hidden = got[:, :, :-off]
                        compare("absorbed_hidden_rows_zero", mx.zeros_like(hidden), hidden, **info)
            poison(flash, mx, rand, compare, receipt, scale, H, N, R, V, C)
            receipt["passed"] = all(r["finite"] and r["relative_l2"] < 1e-2 for r in receipt["numerics"]
                                    if r["ref_max_abs"] > 0) and all(
                r["max_abs"] == 0 for r in receipt["numerics"] if "hidden" in r["name"]
                or "poison" in r["name"]) and all(
                r.get("hidden_lse_neg_inf", True) for r in receipt["numerics"])
        elif args.phase == "bench":
            parts = set(args.parts.split(","))
            for shape in args.shapes.split(","):
                ctx, rows = (int(x) for x in shape.split("x"))
                s = ctx+rows
                info = {"context": ctx, "rows": rows}
                if "expanded" in parts:
                    q, kv, sh = rand((1, rows, H*(N+R)), 0.5), rand((1, s, H*(N+V))), rand((1, 1, s, R))
                    qh = q.reshape(1, rows, H, N+R).transpose(0, 2, 1, 3)
                    kvh = mx.contiguous(kv.reshape(1, s, H, N+V).transpose(0, 2, 1, 3))
                    keys = mx.concatenate([kvh[..., :N], mx.broadcast_to(sh, (1, H, s, R))], -1)
                    pad = lambda z: mx.pad(z, [(0, 0), (0, 0), (0, 0), (0, 256-z.shape[-1])])
                    qp, kp, vp = pad(qh), pad(keys), pad(kvh[..., N:])
                    k128, v128 = mx.contiguous(keys[..., :128]), mx.contiguous(kvh[..., N:])
                    q128 = mx.contiguous(qh[..., :128])
                    mx.eval(q, kv, sh, kvh, qp, kp, vp, k128, v128, q128)
                    del keys
                    timed("d128_sdpa_bound", lambda: mx.fast.scaled_dot_product_attention(
                        q128, k128, v128, scale=scale, mask="causal"), **info)
                    timed("padded256_sdpa_only", lambda: mx.fast.scaled_dot_product_attention(
                        qp, kp, vp, scale=scale, mask="causal", force_fused=True), **info)
                    ref = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=scale, mask="causal")[..., :V]
                    ref = ref.transpose(0, 2, 1, 3).reshape(1, rows, H*V)
                    for layout, operand in (("tokens", kv), ("heads", kvh)):
                        run = lambda: flash.expanded(q, operand, sh, "causal", scale=scale)
                        timed(f"flash192_{layout}", run, **info)
                        compare(f"flash192_{layout}_vs_padded256", ref, run(), **info)
                    del q, kv, sh, qh, kvh, qp, kp, vp, k128, v128, q128, ref
                    mx.clear_cache()
                if "reexpand" in parts or "absorbed" in parts:
                    lat = rand((1, 1, s, C))
                    wkv = rand((H*(N+V), C), 0.04)
                    wh = mx.contiguous(wkv.reshape(H, N+V, C).swapaxes(-1, -2))
                    mx.eval(lat, wkv, wh)
                    timed("kv_b_reexpansion_tokens", lambda: lat[:, 0] @ wkv.T, **info)
                    timed("kv_b_reexpansion_heads", lambda: lat @ wh, **info)
                    del wkv, wh
                    if "absorbed" in parts:
                        qa, qs, sh = rand((1, H, rows, C), 0.05), rand((1, H, rows, R), 0.5), rand((1, 1, s, R))
                        mx.eval(qa, qs, sh)
                        timed("absorbed_flash", lambda: flash.absorbed(qa, qs, lat, sh, "causal", scale=scale),
                              **info)
                        del qa, qs, sh
                    del lat
                    mx.clear_cache()
            for ctx in ((int(x) for x in args.decode.split(",")) if "decode" in parts else ()):
                lat, sh = rand((1, 1, ctx, C)), rand((1, 1, ctx, R))
                mx.eval(lat, sh)
                for rows in (1, 4, 16):
                    info = {"context": ctx, "rows": rows}
                    qa, qs = rand((1, H, rows, C), 0.05), rand((1, H, rows, R), 0.5)
                    mask = "causal" if rows > 1 else None
                    explicit = mx.arange(ctx-rows, ctx)[:, None] >= mx.arange(ctx)[None, :]
                    mx.eval(qa, qs, explicit)

                    def single():
                        score = ((qa @ lat.swapaxes(-1, -2)).astype(mx.float32)
                                 + (qs @ sh.swapaxes(-1, -2)).astype(mx.float32))*scale
                        if rows > 1:
                            score = mx.where(explicit, score, -float("inf"))
                        top = score.max(-1, keepdims=True)
                        prob = mx.exp(score-top)
                        return ((prob @ lat.astype(mx.float32))/prob.sum(-1, keepdims=True)).astype(mx.bfloat16)
                    timed("latent_single_ops", single, **info)
                    timed("absorbed_flash", lambda: flash.absorbed(qa, qs, lat, sh, mask, scale=scale), **info)
                    compare("absorbed_vs_latent_single", single(),
                            flash.absorbed(qa, qs, lat, sh, mask, scale=scale), **info)
                del lat, sh
                mx.clear_cache()
            receipt["passed"] = all(r["finite"] and r["relative_l2"] < 1e-2 for r in receipt["numerics"])
        elif args.phase == "review":
            receipt.update(review(args, mx, flash, rand, compare, receipt, scale, H, N, R, V, C))
        elif args.phase == "crossover":
            receipt.update(crossover(args, mx, np, compare, receipt))
        else:
            receipt.update(layer(args, mx, np, compare, timed, receipt))
    except BaseException as error:
        receipt["passed"] = False
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["swap_after"] = swap()
        path = args.output/"receipt.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, default=str)+"\n")
        os.replace(tmp, path)
    return 0 if receipt["passed"] else 1


def paired(mx, arms, *, rounds=4, launches=1, **info):
    """A/B/B/A... timing: arm order reverses every round. With launches > 1
    each sample evaluates that many independent launches of the arm in one
    graph and reports wall/launches (device time with launch overhead
    amortized); launches == 1 is plain wall time of one evaluated call."""
    names = list(arms)
    for fn in arms.values():  # warm-up / compile, outside the clock
        mx.eval(fn())
    mx.synchronize()
    samples = {name: [] for name in names}
    order = []
    for r in range(rounds):
        for name in (names if r % 2 == 0 else names[::-1]):
            fn = arms[name]
            start = time.perf_counter()
            mx.eval([fn() for _ in range(launches)])
            mx.synchronize()
            samples[name].append((time.perf_counter()-start)*1e3/launches)
            order.append(chr(65+names.index(name)))
    return [{"name": name, "median_ms": statistics.median(v), "min_ms": min(v), "samples_ms": v,
             "launches_per_sample": launches, "order": "".join(order), **info}
            for name, v in samples.items()]


def _emit(receipt, rows):
    for row in rows:
        receipt["timings"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)


def review(args, mx, flash, rand, compare, receipt, scale, H, N, R, V, C):
    """Kernel-only device time (8 launches per sample, A/B/B/A order).

    Inside every timed window: only the named kernel(s). Absorbed inputs are
    already absorbed (qn @ W_k done), the latent cache is materialized; the V
    un-absorption, projections and cache update are outside."""
    out = {"timed_window": "kernel only: absorbed partial (+ combine where named) on precomputed "
           "absorbed q, shared q, latent and shared caches; absorption, unabsorption, projections, "
           "cache update outside; 8 launches per sample, wall/8; A/B/B/A order"}
    cells = [tuple(int(v) for v in c.split("x")) for c in args.review_cells.split(",")]
    for b, ctx, t in cells:
        lat, sh = rand((b, 1, ctx, C)), rand((b, 1, ctx, R))
        qa, qs = rand((b, H, t, C), 0.05), rand((b, H, t, R), 0.5)
        mask = "causal" if t > 1 else None
        mx.eval(lat, sh, qa, qs)
        info = {"batch": b, "context": ctx, "rows": t}
        # Q1: latent fragments loaded once for K and V (reuse) vs tiled.
        arms = {v: (lambda v=v: flash.absorbed(qa, qs, lat, sh, mask, scale=scale, variant=v))
                for v in flash.ABSORBED_VARIANTS}
        ref = arms["tiled"]()
        for v in ("reuse", "reuse_q"):
            compare(f"absorbed_{v}_vs_tiled", ref, arms[v](), variant=v, **info)
        _emit(receipt, paired(mx, arms, launches=8, study="reuse", **info))
        # Q2: split sweep; partial alone, combine alone, total.
        auto_split, _ = flash.split_plan(b, t, ctx)
        for split in sorted({128, 256, 512, 1024, 2048, 4096, 8192, 32768, auto_split}):
            if split > ctx:
                continue
            params, count = flash.absorbed_params(qa, lat, mask, split=split)
            parts = flash.absorbed(qa, qs, lat, sh, mask, scale=scale, split=split, stage="partial")
            mx.eval(parts)
            arms = {"partial": lambda split=split: flash.absorbed(qa, qs, lat, sh, mask, scale=scale,
                                                                  split=split, stage="partial"),
                    "combine": lambda parts=parts, params=params: flash.combine_partials(
                        *parts, params, b, t, H, qa.dtype),
                    "total": lambda split=split: flash.absorbed(qa, qs, lat, sh, mask, scale=scale,
                                                                split=split)}
            _emit(receipt, paired(mx, arms, launches=8, study="split", split=split, splits=count,
                                  auto=split == auto_split, partial_write_bytes=b*t*count*H*(C+2)*4,
                                  cache_read_bytes=b*ctx*(C+R)*2, **info))
            del parts
        mx.clear_cache()
    # Bandwidth floor for the decode cache stream: a reduction over the same bytes.
    for ctx in (32768, 131072):
        lat, sh = rand((1, 1, ctx, C)), rand((1, 1, ctx, R))
        mx.eval(lat, sh)
        _emit(receipt, paired(mx, {"stream_sum": lambda lat=lat, sh=sh: (lat.sum(-2), sh.sum(-2))}, launches=8,
                              study="floor", batch=1, context=ctx, rows=0, cache_read_bytes=ctx*(C+R)*2))
    out["passed"] = all(r["finite"] and r["relative_l2"] < 1e-2 for r in receipt["numerics"])
    return out


def crossover(args, mx, np, compare, receipt):
    """Full-layer wall time (one real-weight MLA layer, A/B/B/A order).

    Inside every timed window: q_proj, kv_a_proj + norm, the in-place cache
    update (cache rewound, not rebuilt), absorption or kv_b expansion, the
    attention kernel(s), V un-absorption, LSE window merge where used,
    o_proj. Outside: input creation and the cache prefill."""
    from mlx2.adapters.kimi_linear import REPO, REVISION
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.kimi_linear import ModelArgs, MultiLatentAttention
    config = json.loads((args.model/"config.json").read_text())
    index = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    a = ModelArgs.from_dict(config)
    mla = MultiLatentAttention(a)
    prefix = "model.layers.3.self_attn."
    weights = {}
    for name in {index[k] for k in index if k.startswith(prefix)}:
        for key, value in mx.load(str(args.model/name)).items():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = value
    mla.load_weights(list(weights.items()), strict=True)
    mla.eval()
    mx.eval(mla.parameters())
    c = a.kv_lora_rank
    arms = {  # (mla_flash, auto_rows, flash_rows, expanded_bytes)
        "absorbed": ("all", 1 << 30, 1 << 30, 4 << 30),
        "expanded": ("expanded", 0, 0, 4 << 30),
        "flash_all": ("all", 16, 128, 4 << 30),
    }
    out = {"checkpoint": REPO+"@"+REVISION, "timed_window": " ".join(crossover.__doc__.split())}
    for shape in args.crossover_cells.split(","):
        bs, ctx, rows_s = shape.split("x")
        b, ctx = int(bs[1:]), int(ctx)
        rows_list = [int(r) for r in rows_s.split("/")]
        hidden = mx.random.normal((1, 4096, a.hidden_size)).astype(mx.bfloat16)
        latent, shared = mx.split(mla.kv_a_proj_with_mqa(hidden), [c], -1)
        latent = mla.kv_a_layernorm(latent)
        reps = (ctx+max(rows_list)+4095)//4096
        latent = mx.tile(latent, (b, reps, 1))[:, :ctx+max(rows_list)][:, None]
        shared = mx.tile(shared, (b, reps, 1))[:, :ctx+max(rows_list)][:, None]
        kv_cache = KVCache()
        kv_cache.update_and_fetch(latent, shared)
        mx.eval(kv_cache.keys, kv_cache.values)
        del latent, shared

        def run(arm, x, mask, kv_cache=kv_cache, ctx=ctx):
            mla.policy = "auto"
            mla.flash, mla.auto_rows, mla.flash_rows, mla.expanded_bytes = arm
            kv_cache.offset = ctx
            return mla(x, mask=mask, cache=kv_cache)
        for rows in rows_list:
            info = {"batch": b, "context": ctx, "rows": rows}
            x = mx.random.normal((b, rows, a.hidden_size)).astype(mx.bfloat16)
            mask = "causal" if rows > 1 else None
            outs = {}
            for name, arm in arms.items():
                before = dict(mla.calls)
                outs[name] = run(arm, x, mask)
                mx.eval(outs[name])
                delta = {k: v-before.get(k, 0) for k, v in mla.calls.items()
                         if v != before.get(k, 0) and "calls" not in k and "tokens" not in k}
                receipt.setdefault("counter_deltas", []).append({"arm": name, **info, "delta": delta})
            compare("absorbed_vs_expanded", outs["expanded"], outs["absorbed"], **info)
            del outs
            _emit(receipt, paired(mx, {name: (lambda arm=arm, x=x, mask=mask, run=run: run(arm, x, mask))
                                       for name, arm in arms.items()},
                                  rounds=4, launches=1, study="crossover", **info))
            mx.clear_cache()
        del kv_cache
        mx.clear_cache()
    out["passed"] = all(r["finite"] and r["relative_l2"] < 3e-2 for r in receipt["numerics"])
    return out


def poison(flash, mx, rand, compare, receipt, scale, H, N, R, V, C):
    """NaN in every cache region a row must not read; outputs must be
    bit-identical to the clean run (max_abs exactly 0, finite)."""
    nan = float("nan")

    def tail(z, extra):  # z followed by NaN rows along the token axis
        shape = list(z.shape)
        axis = 1 if z.ndim == 3 else 2
        shape[axis] = extra
        return mx.concatenate([z, mx.full(shape, nan, z.dtype)], axis)

    # 1. Beyond the visible length: the cache buffer continues with NaN.
    t, off = 40, 260
    s = off + t
    q, kv, sh = rand((1, t, H*(N+R)), 0.5), rand((1, s, H*(N+V))), rand((1, 1, s, R))
    clean = flash.expanded(q, kv, sh, "causal", scale=scale)
    kvp, shp = tail(kv, 96)[:, :s], tail(sh, 96)[:, :, :s]
    compare("poison_beyond_length_flash192", clean, flash.expanded(q, kvp, shp, "causal", scale=scale),
            rows=t, width=s)
    # 2. Future keys: NaN from OFF + 32 on; rows 0..31 never see them and
    #    their simdgroups never load them (V rows clamp to the group limit).
    bad = mx.arange(s)[None, :, None] >= off + 32
    kvf = mx.where(bad, mx.array(nan, kv.dtype), kv)
    shf = mx.where(bad[:, None], mx.array(nan, sh.dtype), sh)
    got = flash.expanded(q, kvf, shf, "causal", scale=scale)
    compare("poison_future_keys_flash192", clean[:, :32], got[:, :32], rows=32, width=s)
    receipt["numerics"][-1]["later_rows_nan_expected"] = bool(mx.any(mx.isnan(got[:, 32:])))
    # 2b. B = 2 with the shared key read from its NaN-tailed cache buffer.
    q2, kv2, sh2 = rand((2, t, H*(N+R)), 0.5), rand((2, s, H*(N+V))), rand((2, 1, s, R))
    compare("poison_buffer_tail_flash192_b2", flash.expanded(q2, kv2, sh2, "causal", scale=scale),
            flash.expanded(q2, kv2, tail(sh2, 96), "causal", scale=scale), batch=2, rows=t, width=s)
    # 3. Windows over a NaN-tailed latent buffer.
    lat = rand((1, 1, s, C))
    wkv = rand((H*(N+V), C), 0.04)
    proj = lambda z: z @ wkv.T
    clean = flash.expanded_windows(q, lat, sh, proj, "causal", scale=scale, window=128)
    got = flash.expanded_windows(q, tail(lat, 96)[:, :, :s], shp, proj, "causal", scale=scale, window=128)
    compare("poison_beyond_length_windows", clean, got, rows=t, width=s)
    # 4/5. Latent flash: NaN past the length, and NaN in keys after row 1.
    for variant in flash.ABSORBED_VARIANTS:
        for t in (1, 4):
            qa, qs = rand((1, H, t, C), 0.05), rand((1, H, t, R), 0.5)
            mask = "causal" if t > 1 else None
            clean = flash.absorbed(qa, qs, lat, sh, mask, scale=scale, variant=variant, split=128)
            got = flash.absorbed(qa, qs, tail(lat, 96)[:, :, :s], shp, mask, scale=scale,
                                 variant=variant, split=128)
            compare("poison_beyond_length_latent", clean, got, rows=t, width=s, variant=variant)
        # B = 2 cache buffers read in place (width < capacity, NaN tail).
        lat2, sh2 = rand((2, 1, s, C)), rand((2, 1, s, R))
        qa2, qs2 = rand((2, H, 4, C), 0.05), rand((2, H, 4, R), 0.5)
        clean2 = flash.absorbed(qa2, qs2, lat2, sh2, "causal", scale=scale, variant=variant, split=128)
        got2 = flash.absorbed(qa2, qs2, tail(lat2, 96), tail(sh2, 96), "causal", scale=scale,
                              variant=variant, split=128, width=s)
        compare("poison_buffer_tail_latent_b2", clean2, got2, batch=2, rows=4, width=s, variant=variant)
        width = s
        bad = mx.arange(width)[None, None, :, None] >= width - 4 + 2
        latf = mx.where(bad, mx.array(nan, lat.dtype), lat)
        shf = mx.where(bad, mx.array(nan, sh.dtype), sh)
        got = flash.absorbed(qa, qs, latf, shf, "causal", scale=scale, variant=variant, split=128)
        compare("poison_future_keys_latent", clean[:, :, :2], got[:, :, :2], rows=2, width=width,
                variant=variant)


def layer(args, mx, np, compare, timed, receipt):
    from mlx2.adapters.kimi_linear import REPO, REVISION
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.kimi_linear import ModelArgs, MultiLatentAttention
    config = json.loads((args.model/"config.json").read_text())
    index = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    a = ModelArgs.from_dict(config)
    out = {"checkpoint": REPO+"@"+REVISION}
    mla = MultiLatentAttention(a)
    prefix = "model.layers.3.self_attn."
    weights = {}
    for name in {index[k] for k in index if k.startswith(prefix)}:
        for key, value in mx.load(str(args.model/name)).items():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = value
    mla.load_weights(list(weights.items()), strict=True)
    mla.eval()
    mx.eval(mla.parameters())
    c = a.kv_lora_rank

    grid = {}
    for shape in args.layer_cells.split(","):
        ctx, rows = (int(x) for x in shape.split("x"))
        grid.setdefault(ctx, []).append(rows)
    arms = {  # name: (mla_flash, auto_rows, expanded_bytes)
        "noflash": ("off", 16, 4 << 30),          # current auto: padded D256 / latent_single
        "flash_all": ("all", 16, 4 << 30),        # proposed auto
        "absorbed": ("all", 1 << 30, 4 << 30),    # every row on the latent flash
        "expanded": ("all", 0, 4 << 30),          # every row on flash192
    }
    for context, rows_list in sorted(grid.items()):
        hidden = mx.random.normal((1, 4096, a.hidden_size)).astype(mx.bfloat16)
        latent, shared = mx.split(mla.kv_a_proj_with_mqa(hidden), [c], -1)
        latent = mla.kv_a_layernorm(latent)
        reps = (context+4095)//4096
        latent = mx.tile(latent, (1, reps, 1))[:, :context][:, None]
        shared = mx.tile(shared, (1, reps, 1))[:, :context][:, None]
        mx.eval(latent, shared)
        # One persistent cache per context, rewound before every call (the
        # serving update_and_fetch path, without per-call cache rebuilds).
        cache = KVCache()
        cache.update_and_fetch(mx.concatenate([latent, latent[:, :, :max(rows_list)]], 2),
                               mx.concatenate([shared, shared[:, :, :max(rows_list)]], 2))
        mx.eval(cache.keys, cache.values)

        def run(arm, x, mask, policy="auto"):
            mla.policy = policy
            mla.flash, mla.auto_rows, mla.expanded_bytes = arm
            cache.offset = context
            return mla(x, mask=mask, cache=cache)
        windows = ("all", 16, 512 << 20)  # four 32K windows at 128K
        for rows in sorted(rows_list):
            info = {"context": context, "rows": rows}
            x = mx.random.normal((1, rows, a.hidden_size)).astype(mx.bfloat16)
            mask = "causal" if rows > 1 else None
            explicit = (None if rows == 1 else
                        mx.arange(context, context+rows)[:, None] >= mx.arange(context+rows)[None, :])
            # Ordinary references: expanded (unfused 192/128 SDPA) while its
            # FP32 score block stays near 2 GiB, else the latent_single law.
            ordinary = "expanded" if rows*context <= 512*32768 else "latent_single"
            ref = run(("off", 16, 4 << 30), x, explicit, policy=ordinary)
            mx.eval(ref)
            for name, arm in {**arms, **({"windows": windows} if context >= 131072 and rows > 16 else {})}.items():
                before = dict(mla.calls)
                got = run(arm, x, mask)
                compare(f"{name}_vs_{ordinary}", ref, got, arm=name, **info)
                delta = {k: v-before.get(k, 0) for k, v in mla.calls.items() if v != before.get(k, 0)}
                receipt.setdefault("counter_deltas", []).append({"arm": name, **info, "delta": delta})
                timed(f"layer_{name}", lambda: run(arm, x, mask), repeats=3, arm=name, **info)
            del ref, got
            mx.clear_cache()
        del latent, shared
        mx.clear_cache()
    out["mla_counters"] = dict(mla.calls)
    out["passed"] = all(r["finite"] and r["relative_l2"] < 2e-2 for r in receipt["numerics"])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
