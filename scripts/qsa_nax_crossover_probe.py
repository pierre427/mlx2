"""NAX block-sparse QSA prefill vs the masked SDPA, kernel and layer level.

One process, one model load (the served Flash-Next artifact and policy).

1. Capture: one real-document prompt is prefilled through the full model in
   8192-row chunks with NAX off, recording the hidden input of one QSA
   attention layer (the first full-attention layer by default).
2. Crossover: for each context C and slice width L (L <= C), a fresh cache
   for that one layer is prefilled with the captured rows [0, C - L) (masked),
   then the slice [C - L, C) runs through the layer with NAX forced on and
   forced off (module mode on/off), interleaved, trimming the cache back after
   every call.  Layer ms (indexer + projections + attention core + o_proj)
   per arm, and the kernel-level core (compaction + NAX kernel vs dense mask
   + fused SDPA) on the same q/k/v/selection.  Outputs are compared bitwise
   (bf16 ulp distance) and, where the score tensor fits, against an fp32
   masked reference.
3. Ragged batch: three lanes at different contexts are prefilled one by one,
   merged into a left-padded BatchQSAKVCache, and one slice per lane runs
   batched with NAX on and off.  Each lane's batched NAX output is compared
   with the same lane's B=1 NAX output (per-lane selections and left padding
   must make them identical when the selections agree) and with the masked
   batched output.

  gpuq.sh qsa-nax-a env PYTHONPATH=src MLX_ENABLE_TF32=0 .venv/bin/python \\
      scripts/qsa_nax_crossover_probe.py --i-own-the-gpu --out a.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flash_next_options_sweep import MODEL, Harness, swapouts

CONTEXTS = (2048, 4096, 8192, 12288, 16384, 32768)
SLICES = (512, 2048, 8192)


def ordered_bf16(a):
    """bf16 values as order-preserving integers (ulp distance = difference)."""
    import mlx.core as mx

    bits = np.array(a.astype(mx.float32)).view(np.uint32) >> 16
    bits = bits.astype(np.int64)
    return np.where(bits < 0x8000, bits, 0x8000 - bits)


def compare(a, b):
    import mlx.core as mx

    fa = np.array(a.astype(mx.float32))
    fb = np.array(b.astype(mx.float32))
    ulp = np.abs(ordered_bf16(a) - ordered_bf16(b))
    return {
        "bit_identical": bool((ulp == 0).all()),
        "max_ulp": int(ulp.max()),
        "frac_differs": float((ulp > 0).mean()),
        "frac_gt_1ulp": float((ulp > 1).mean()),
        "max_abs": float(np.abs(fa - fb).max()),
        "rel_l2": float(np.linalg.norm(fa - fb) / max(np.linalg.norm(fb), 1e-30)),
    }


def rel_err(x, ref):
    import mlx.core as mx

    fx = np.array(x.astype(mx.float32))
    fr = np.array(ref.astype(mx.float32))
    return float(np.linalg.norm(fx - fr) / max(np.linalg.norm(fr), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer", type=int, default=None, help="QSA layer index (default: first)")
    ap.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXTS))
    ap.add_argument("--slices", type=int, nargs="+", default=list(SLICES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ragged", type=int, nargs="+", default=[12000, 16384, 20000])
    ap.add_argument("--ragged-slice", type=int, default=2048)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--ref-max-gib", type=float, default=3.0,
                    help="largest fp32 score tensor for the fp32 reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("Metal run: pass --i-own-the-gpu under the GPU lock")

    import mlx.core as mx

    h = Harness("default", model=a.model)
    from mlx2.runtime.models import qwen4_exp as QE
    from mlx2.runtime.models import qwen4_qsa_nax as NAX
    from mlx2.runtime.models.base import create_attention_mask

    swap0 = swapouts()
    model = h.adapter.model
    layers = model.layers
    qsa = [i for i, layer in enumerate(layers) if not layer.is_linear]
    li = qsa[0] if a.layer is None else a.layer
    attn = layers[li].self_attn
    out = {"model": a.model, "mlx": mx.__version__, "layer": li, "qsa_layers": qsa,
           "device": mx.device_info().get("device_name"),
           "geometry": {"heads": attn.num_heads, "kv_heads": attn.num_kv_heads,
                        "head_dim": attn.head_dim, "nax_layout_ok": bool(attn._nax_layout_ok),
                        "block_topk": int(attn.indexer.block_topk)},
           "nax_kernel_available": bool(NAX.nax_kernel_available()),
           "policy": h.adapter.policy.as_dict(), "load_s": h.load_s,
           "crossover": [], "ragged": None}
    print("LOADED", f"{h.load_s:.1f}s layer={li}", out["geometry"], flush=True)

    # -- 1. capture the layer's input rows from a real document ------------
    need = max(max(a.contexts), max(a.ragged) + a.ragged_slice)
    corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    ids = list(h.adapter.tokenizer.encode(corpus))
    if len(ids) < need:
        raise SystemExit(f"corpus has {len(ids)} tokens; need {need}")
    captured = []
    original = QE.Attention.__call__

    def capture(self, x, mask, cache, *, _captured=captured, **kw):
        if self is attn:
            _captured.append(x)
        return original(self, x, mask, cache, **kw)

    QE._QSA_NAX_KERNEL = False
    QE.Attention.__call__ = capture
    t0 = time.perf_counter()
    try:
        cache = model.make_cache()
        for start in range(0, need, 8192):
            chunk = mx.array([ids[start: min(need, start + 8192)]])
            hidden = model.model(chunk, cache)  # trunk only: no vocab-wide logits
            mx.eval(hidden, captured[-1])
    finally:
        QE.Attention.__call__ = original
    X = mx.concatenate(captured, axis=1)
    mx.eval(X)
    captured.clear()
    del cache, hidden
    mx.clear_cache()
    out["capture_s"] = time.perf_counter() - t0
    print("captured", X.shape, f"{out['capture_s']:.1f}s", flush=True)

    def fresh_cache():
        return QSAKVCache(attn.indexer.summary_identity)

    QSAKVCache = QE.QSAKVCache

    def layer_call(x, cache):
        mask = create_attention_mask(x, cache, return_array=True)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        return attn(x, mask, cache)

    def build(cache, rows, start=0):
        QE._QSA_NAX_KERNEL = False
        for s in range(start, rows, 8192):
            y = layer_call(X[:, s: min(rows, s + 8192)], cache)
            mx.eval(y)
        return cache

    stash = {}
    decide = QE.decide_qsa_nax_admission
    nax_fn = QE.nax_qsa_attention

    def spy_decide(selection, **kw):
        stash["selection"] = selection
        return decide(selection, **kw)

    def spy_nax(q, k, v, *args, **kw):
        stash["qkv"] = (q, k, v)
        return nax_fn(q, k, v, *args, **kw)

    def timed(fn, reps):
        times = []
        result = None
        for r in range(reps + 1):
            mx.synchronize()
            t = time.perf_counter()
            result = fn()
            mx.eval(result)
            mx.synchronize()
            if r:
                times.append(1e3 * (time.perf_counter() - t))
        return result, times

    # -- 2. crossover grid ---------------------------------------------------
    for C in a.contexts:
        for L in a.slices:
            if L > C:
                continue
            cache = build(fresh_cache(), C - L)
            x = X[:, C - L: C]
            rec = {"context": C, "slice": L}
            outs, times = {}, {"nax": [], "masked": []}
            for rep in range(a.reps + 1):
                order = ("nax", "masked") if rep % 2 == 0 else ("masked", "nax")
                for arm in order:
                    QE._QSA_NAX_KERNEL = arm == "nax"
                    QE.decide_qsa_nax_admission = spy_decide
                    QE.nax_qsa_attention = spy_nax
                    try:
                        mx.synchronize()
                        t = time.perf_counter()
                        y = layer_call(x, cache)
                        mx.eval(y)
                        mx.synchronize()
                        ms = 1e3 * (time.perf_counter() - t)
                    finally:
                        QE.decide_qsa_nax_admission = decide
                        QE.nax_qsa_attention = nax_fn
                    assert cache.trim(L) == L
                    if rep:
                        times[arm].append(ms)
                    if arm == "nax" and "qkv" not in stash:
                        raise SystemExit(f"NAX did not engage at C={C} L={L}: "
                                         f"{QE.qsa_nax_status()['last_receipt']}")
                    if arm not in outs:
                        outs[arm] = y
                        if arm == "nax":
                            q, k, v = stash["qkv"]
                            sel = stash["selection"]
                            mx.eval(q, k, v)
                            kernel_inputs = (q, k, v, sel)
            rec["selection_kind"] = kernel_inputs[3].kind
            rec["layer_ms"] = {arm: statistics.median(v) for arm, v in times.items()}
            rec["layer_ms_all"] = times
            rec["layer_speedup_pct"] = 100 * (rec["layer_ms"]["masked"] / rec["layer_ms"]["nax"] - 1)
            rec["layer_compare"] = compare(outs["nax"], outs["masked"])
            # Kernel level on the captured q/k/v/selection.
            q, k, v, sel = kernel_inputs

            def run_nax(q=q, k=k, v=v, sel=sel):
                (i, c, n, u, qp, lp, tot) = NAX.compact_blocks_to_kernel_inputs(sel.compact_blocks())
                return NAX.nax_qsa_attention(q, k, v, i, c, n, qp, lp, scale=attn.scale,
                                             u_width=u, total=tot,
                                             n_kv_heads=attn.num_kv_heads).astype(q.dtype)

            def run_masked(q=q, k=k, v=v, sel=sel):
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale,
                                                            mask=sel.dense_mask())

            k_outs, k_times = {}, {"nax": [], "masked": []}
            for rep in range(a.reps):
                for arm, fn in ((("nax", run_nax), ("masked", run_masked)) if rep % 2 == 0
                                else (("masked", run_masked), ("nax", run_nax))):
                    y, ts = timed(fn, 1)
                    k_times[arm].extend(ts)
                    k_outs.setdefault(arm, y)
            rec["kernel_ms"] = {arm: statistics.median(v) for arm, v in k_times.items()}
            rec["kernel_speedup_pct"] = 100 * (rec["kernel_ms"]["masked"] / rec["kernel_ms"]["nax"] - 1)
            rec["kernel_compare"] = compare(k_outs["nax"], k_outs["masked"])
            score_gib = q.shape[1] * L * C * 4 / 2**30
            if score_gib <= a.ref_max_gib:
                f32 = lambda t: t.astype(mx.float32)
                ref = mx.fast.scaled_dot_product_attention(f32(q), f32(k), f32(v),
                                                           scale=attn.scale, mask=sel.dense_mask())
                mx.eval(ref)
                rec["fp32_ref_rel_l2"] = {"nax": rel_err(k_outs["nax"], ref),
                                          "masked": rel_err(k_outs["masked"], ref)}
                del ref
            out["crossover"].append(rec)
            print(f"C={C:6d} L={L:5d} kind={rec['selection_kind']:8s} layer nax={rec['layer_ms']['nax']:.2f}ms "
                  f"masked={rec['layer_ms']['masked']:.2f}ms ({rec['layer_speedup_pct']:+.1f}%) "
                  f"kernel nax={rec['kernel_ms']['nax']:.2f} masked={rec['kernel_ms']['masked']:.2f} "
                  f"({rec['kernel_speedup_pct']:+.1f}%) max_ulp={rec['layer_compare']['max_ulp']} "
                  f"ref={rec.get('fp32_ref_rel_l2')}", flush=True)
            del cache, outs, kernel_inputs, k_outs, fn, run_nax, run_masked
            q = k = v = sel = None
            stash.clear()
            mx.clear_cache()
            swap = swapouts() - swap0
            if swap > a.max_swapout_pages:
                out["aborted"] = f"swapouts +{swap} pages at C={C} L={L}"
                break
        if "aborted" in out:
            break
        Path(a.out).write_text(json.dumps(out, indent=1))

    # -- 3. ragged batch -----------------------------------------------------
    if "aborted" not in out and a.ragged:
        L = a.ragged_slice
        lanes = list(a.ragged)
        caches = [build(fresh_cache(), c) for c in lanes]
        # Each lane's slice: the rows after its own context.
        xs = [X[:, c: c + L] for c in lanes]
        single = []
        for cache, x in zip(caches, xs):
            QE._QSA_NAX_KERNEL = True
            y = layer_call(x, cache)
            mx.eval(y)
            assert cache.trim(L) == L
            single.append(y)
        batch_cache = QE.BatchQSAKVCache.merge(caches)
        xb = mx.concatenate(xs, axis=0)
        res = {"lanes": lanes, "slice": L}
        outs = {}
        QE.qsa_nax_status(reset=True)
        for arm in ("nax", "masked"):
            QE._QSA_NAX_KERNEL = arm == "nax"
            y = layer_call(xb, batch_cache)
            mx.eval(y)
            assert batch_cache.trim(L) == L
            outs[arm] = y
        res["admission"] = QE.qsa_nax_status()["counts"]
        res["per_lane"] = [
            {"context": c,
             "nax_batched_vs_nax_b1": compare(outs["nax"][i: i + 1], single[i]),
             "nax_batched_vs_masked_batched": compare(outs["nax"][i: i + 1],
                                                      outs["masked"][i: i + 1]),
             "finite": bool(mx.all(mx.isfinite(outs["nax"][i])).item())}
            for i, c in enumerate(lanes)
        ]
        out["ragged"] = res
        for lane in res["per_lane"]:
            print("ragged", lane["context"], "nax B>1 vs B=1",
                  lane["nax_batched_vs_nax_b1"]["max_ulp"], "vs masked",
                  lane["nax_batched_vs_masked_batched"]["max_ulp"], "finite", lane["finite"],
                  flush=True)
    QE._QSA_NAX_KERNEL = None
    out["swapouts_delta"] = swapouts() - swap0
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("DONE", a.out, "swap", out["swapouts_delta"], flush=True)


if __name__ == "__main__":
    main()
