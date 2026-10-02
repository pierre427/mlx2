"""Root-cause the Flash-Next row-cost cliff (128 rows cost what 256 rows cost).

One trunk forward + head on fresh caches (as ``measure_fn_mtp_padding.py
--row-cost`` does), at row counts across ``--rows``, under one or more
``--pad-floors`` for the sorted-MoE streaming pad
(``switch_layers._RHS_PAD_MIN_ROWS_PER_EXPERT``; 3 is today's default, 0 is
off).  Arms are interleaved per rep (ABBA order) after a two-pass warm-up.

``--components`` adds a second, synchronised pass that wraps each decoder
sub-module class (PLE, hyper-connection gates, GDN, QSA attention, MoE block,
the routed expert bank, every sorted ``gather_qmm``) with eval + synchronize
and attributes wall time per component (inclusive; the expert bank is also
reported inside the MoE block).  Syncs serialise the pipeline: these are
attributions, not throughput.  The gather_qmm records also note which MLX
kernel the call selects (``rhs`` streaming when M == 1, B >= 16, sorted and
B // E >= 4 after any pad; else per-row ``qmv``; mlx 39400a0d4
``GatherQMM::eval_gpu``).

``--exactness`` compares full-forward logits and every MoE block output at
the given rows across the pad-floor arms (bytes / max_abs_diff on Metal).

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/profile_fn_row_cliff.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --pad-floors 3 0 1 2 --components --out cliff.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ROWS = sorted(set(list(range(96, 329, 8)) + [132, 140, 150, 153, 154, 156, 186, 187, 204, 205, 252]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", nargs="+", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--pad-floors", nargs="+", type=int, default=[3, 0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--components", action="store_true")
    ap.add_argument("--component-rows", nargs="+", type=int, default=[96, 128, 152, 160, 200, 208, 256, 320])
    ap.add_argument("--component-reps", type=int, default=3)
    ap.add_argument("--exactness", nargs="*", type=int, default=None)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    # Construct the adapter before any mlx2.runtime model module is imported.
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import switch_layers as SL

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()

    def check_swap():
        grew = swapouts() - swap0
        if grew > a.max_swapout_pages:
            raise SystemExit(f"aborting: Swapouts grew by {grew} pages")

    layers = model.language_model.model.layers
    moe0 = layers[0].mlp
    bank = moe0.switch_mlp
    experts = None
    for name in ("gate_up_proj", "up_proj", "gate_proj"):
        if name in bank:
            experts = int(bank[name].num_experts)
            break
    topk = int(getattr(moe0, "top_k", getattr(moe0, "num_experts_per_tok", 0)) or 0)
    info = {
        "switch_class": type(bank).__name__, "experts": experts, "top_k": topk,
        "gather_sort_min": SL._GATHER_SORT_MIN_ASSIGNMENTS,
        "pad_floor_default": SL._RHS_PAD_MIN_ROWS_PER_EXPERT,
        "eager_dispatch_max_rows": getattr(Q, "_EAGER_DISPATCH_MAX_ROWS", None),
        "layer_types": collections.Counter("gdn" if l.is_linear else "qsa" for l in layers),
    }
    print("INFO", json.dumps(info, default=str), flush=True)

    text = "\n\n".join((ROOT / "docs" / n).read_text() for n in ("QUALIFICATION.md", "SERVING.md"))
    source = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": text[:40000]}]}))
    assert len(source) > max(a.rows) + 8, len(source)

    def forward(m, keep=False):
        ids = mx.array([source[:m]], mx.uint32)
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.perf_counter()
        hidden, _hyper = model.mtp_backbone(ids, cache=cache)
        logits = model.logits(hidden[:, -1:, :])
        mx.eval(logits)
        ms = 1e3 * (time.perf_counter() - t0)
        del cache, hidden
        return (ms, logits) if keep else ms

    def set_floor(v):
        SL._RHS_PAD_MIN_ROWS_PER_EXPERT = int(v)

    result = {"model": a.model, "info": info, "mlx": mx.__version__}

    # ---- 1. row-cost curve per pad-floor arm -------------------------------
    times = {(f, m): [] for f in a.pad_floors for m in a.rows}
    for f in a.pad_floors:  # warm-up, two passes per arm
        set_floor(f)
        for m in a.rows:
            forward(m)
            forward(m)
    mx.clear_cache()
    for rep in range(a.reps):
        arms = a.pad_floors if rep % 2 == 0 else a.pad_floors[::-1]
        rows = a.rows if rep % 2 == 0 else a.rows[::-1]
        for m in rows:
            for f in arms:
                set_floor(f)
                times[(f, m)].append(forward(m))
        mx.clear_cache()
        check_swap()
        print(f"rep {rep} done", flush=True)
    curve = {}
    for f in a.pad_floors:
        curve[str(f)] = {str(m): {"median_ms": round(statistics.median(times[(f, m)]), 2),
                                  "min_ms": round(min(times[(f, m)]), 2),
                                  "max_ms": round(max(times[(f, m)]), 2)} for m in a.rows}
    result["curve"] = curve
    print("CURVE rows " + " ".join(f"{m:>6}" for m in a.rows), flush=True)
    for f in a.pad_floors:
        print(f"CURVE f={f}  " + " ".join(f"{curve[str(f)][str(m)]['median_ms']:6.0f}" for m in a.rows),
              flush=True)
    set_floor(info["pad_floor_default"])

    # ---- 2. per-component attribution (synchronised) -----------------------
    if a.components:
        acc = collections.defaultdict(float)
        kern = collections.Counter()

        def arrays(value):
            if isinstance(value, mx.array):
                return [value]
            if isinstance(value, (tuple, list)):
                return [x for v in value for x in arrays(v)]
            return []

        def wrap(cls, label):
            orig = cls.__call__

            def timed(self, *args, **kwargs):
                mx.eval(arrays(list(args)))
                mx.synchronize()
                t0 = time.perf_counter()
                out = orig(self, *args, **kwargs)
                mx.eval(arrays(out))
                mx.synchronize()
                acc[label] += time.perf_counter() - t0
                return out

            cls.__call__ = timed
            return (cls, orig)

        qsl_orig = SL.QuantizedSwitchLinear.__call__

        def qsl_timed(self, x, indices, sorted_indices=False):
            n = int(indices.size)
            pad = 0
            if sorted_indices and SL._RHS_PAD_MIN_ROWS_PER_EXPERT and indices.ndim == 1:
                pad = SL._rhs_stream_pad(n, self.num_experts)
            b = n + pad
            rhs = bool(sorted_indices and x.shape[-2] == 1 and b >= 16 and b // self.num_experts >= 4)
            kern["rhs" if rhs else ("qmv_sorted" if sorted_indices else "qmv_unsorted")] += 1
            mx.eval(x, indices)
            mx.synchronize()
            t0 = time.perf_counter()
            out = qsl_orig(self, x, indices, sorted_indices)
            mx.eval(out)
            mx.synchronize()
            acc["expert_gather_qmm"] += time.perf_counter() - t0
            return out

        bank_cls = type(bank)
        patches = [wrap(type(layers[0].mlp), "moe_block"), wrap(bank_cls, "expert_bank")]
        gdn = next(l.linear_attn for l in layers if l.is_linear)
        att = next(l.self_attn for l in layers if not l.is_linear)
        patches += [wrap(type(gdn), "gdn"), wrap(type(att), "qsa_attention"),
                    wrap(Q.GatedResidual, "hyper_connection")]
        ple = next((l.ple for l in layers if l.ple is not None), None)
        if ple is not None:
            patches.append(wrap(type(ple), "ple"))
        SL.QuantizedSwitchLinear.__call__ = qsl_timed
        comp = {}
        try:
            for f in a.pad_floors:
                set_floor(f)
                for m in a.component_rows:
                    forward(m)  # warm-up
                    runs = []
                    for _ in range(a.component_reps):
                        acc.clear()
                        kern.clear()
                        total = forward(m)
                        runs.append({"total_ms": total, **{k: 1e3 * v for k, v in acc.items()},
                                     "kernels": dict(kern)})
                    keys = [k for k in runs[0] if k != "kernels"]
                    row = {k: round(statistics.median(r[k] for r in runs), 1) for k in keys}
                    row["kernels"] = runs[-1]["kernels"]
                    comp[f"{f}:{m}"] = row
                    print(f"COMP f={f} m={m} " + json.dumps(row), flush=True)
                mx.clear_cache()
                check_swap()
        finally:
            for cls, orig in patches:
                cls.__call__ = orig
            SL.QuantizedSwitchLinear.__call__ = qsl_orig
            set_floor(info["pad_floor_default"])
        result["components"] = comp

    # ---- 3. exactness across pad-floor arms --------------------------------
    if a.exactness:
        exact = {}
        captured = []
        moe_cls = type(layers[0].mlp)
        moe_orig = moe_cls.__call__

        def moe_capture(self, *args, **kwargs):
            out = moe_orig(self, *args, **kwargs)
            captured.append(out)
            return out

        moe_cls.__call__ = moe_capture
        try:
            for m in a.exactness:
                ref = None
                for f in a.pad_floors:
                    set_floor(f)
                    captured.clear()
                    _ms, logits = forward(m, keep=True)
                    mx.eval(captured)
                    outs = [logits] + list(captured)
                    if ref is None:
                        ref = (f, outs)
                        continue
                    diffs = [float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item())
                             for x, y in zip(ref[1], outs)]
                    same_bytes = [bool(mx.array_equal(x, y).item()) for x, y in zip(ref[1], outs)]
                    exact[f"{m}:{ref[0]}vs{f}"] = {
                        "logits_max_abs_diff": diffs[0], "logits_identical": same_bytes[0],
                        "argmax_equal": bool((mx.argmax(ref[1][0], -1) == mx.argmax(outs[0], -1)).all().item()),
                        "moe_layers_identical": sum(same_bytes[1:]), "moe_layers": len(diffs) - 1,
                        "moe_max_abs_diff": max(diffs[1:]) if diffs[1:] else None,
                    }
                    print(f"EXACT m={m} {ref[0]}vs{f} " + json.dumps(exact[f'{m}:{ref[0]}vs{f}']), flush=True)
                mx.clear_cache()
        finally:
            moe_cls.__call__ = moe_orig
            set_floor(info["pad_floor_default"])
        result["exactness"] = exact

    result["swapouts_delta_pages"] = swapouts() - swap0
    result["peak_gib"] = mx.get_peak_memory() / 2**30
    json.dump(result, open(a.out, "w"), indent=1, default=str)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
