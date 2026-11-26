#!/usr/bin/env python3
"""GPU microbench: gated block-sparse aMoEba MLP kernels vs the stock MLP (research, unqualified).

Question: does block-skipped execution beat the base model's MLP at realistic open fractions?

Arms, per model layer set x shape x backend (mpp = M5 tensor units / NAX, simd = the M1-M4 path,
which also runs on an M5):
  stock-base     the stock MLP (dense: QuantizedLinear x3 + compiled swiglu; FN: mlx2's SwitchGLU)
  stock-masked   the stock MLP with closed neurons zeroed before down ("zero but compute")
  ours-ident     our kernels, unpacked layout, every gate open (the law's dense base)
  ours-skip@p    our kernels, packed layout, a fraction p of the neurons open, closed blocks skipped
  ours-mask@p    our kernels with SKIP=0 (every block computed, closed neurons zero): the bitwise
                 reference for ours-skip@p
Synthetic partitions at p in --fractions: "aligned" (open neurons are whole original down groups,
the best case for down skipping) and "scattered" (random neurons: gate/up skip after packing,
down groups almost never closed).  FN also runs the real partition (fn-boxes2) with the core only
open and with core + its largest box open.

Reported: p10/p50/p90 ms per layer-set forward (randomized interleaved rounds, see "Timing"),
tokens/s, weight bytes read (device counters) and GB/s, speed against the stock arm of the same
group, and exactness: ours packed all-open == ours-ident (bitwise), ours-skip == ours-mask
(bitwise), max |diff| vs stock (relative to the stock output's max), prefill law vs lane law.

v2 (2026-11-19): every group interleaves its arms with its own stock and ours-ident references;
stock fairness diagnostics (stock-2d: quantized_matmul on the loaded 2-D arrays, which stock-base
reaches through [None][0] views; stock-nn: nn.QuantizedLinear;
stock-gate-proj-only: one projection's qmv); prefill runs the prefill law (ours-ident@lane keeps
the old law as a reference); tile gates time the fused, split and host implementations, one
call per forward.

Run inside one owned GPU window (see --print-window).  --dry-run plans and estimates on the CPU.
--analyze reports the real FN partition's skippable fractions on the CPU (no GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODELS = {
    "fn": {
        "path": Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP",
        "checkpoint": "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@25c2da42",
        "adapter": ("mlx2.adapters.flash_next", "FlashNextAdapter"),
        "layers": (3, 23, 43),
    },
    "27b": {
        "path": Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2",
        "checkpoint": "Qwen3.8-27B-BF16-q8-mlx2@2dd8d304",
        "adapter": ("mlx2.adapters.qwen38_27b", "Qwen3827BAdapter"),
        "layers": (3, 31, 59),
    },
}
FN_PARTITION = (Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba"
                / "fn-boxes2-dd1ef1d7d/partition")
SHAPES = {"decode1": 1, "decode4": 4, "prefill2048": 2048}
BOUNDARY_SHAPES = {"prefill128": 128, "prefill512": 512}     # baseline group only: the law switch point
FN_TOPK = 10


def git_rev() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], capture_output=True,
                          text=True).stdout.strip() or "unknown"


def declaration(model: str) -> dict:
    """The adapter's block-MLP declaration (model-specific names live in the adapter)."""
    import importlib
    mod, cls = MODELS[model]["adapter"]
    return dict(getattr(importlib.import_module(mod), cls).amoeba_block_mlp)


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


def synthetic_owner(rng, experts: int, width: int, frac: float, kind: str, gs: int) -> np.ndarray:
    """Tile 0 (open) holds ``frac`` of each expert's neurons, tile 1 (closed) the rest."""
    own = np.ones((experts, width), np.int32)
    if kind == "aligned":
        groups = width // gs
        n_open = int(round(frac * groups))
        for e in range(experts):
            for g in rng.choice(groups, n_open, replace=False):
                own[e, g * gs:(g + 1) * gs] = 0
    else:
        n_open = int(round(frac * width))
        for e in range(experts):
            own[e, rng.choice(width, n_open, replace=False)] = 0
    return own


def fn_real_partition(layers):
    """(tile names, owner per layer [512, 640], largest box tile) from the FN partition."""
    z = np.load(FN_PARTITION / "tiles.npz")
    meta = json.loads((FN_PARTITION / "partition.json").read_text())
    names = ["core"] + [k for k in z.keys() if k.startswith(("box:", "pair:"))]
    from mlx2.runtime.amoeba_blocks import layout as L
    lidx = {int(l): i for i, l in enumerate(z["layers"])}
    owners = {}
    for layer in layers:
        i = lidx[layer]
        owners[layer] = L.owner_from_masks({n: z[n][i] for n in names}, names).reshape(512, 640)
    sizes = {n: int(z[n][[lidx[l] for l in layers]].sum()) for n in names[1:]}
    top = max(sizes, key=sizes.get)
    return names, owners, top, meta


def analyze(out=None):
    """CPU: per-layer skippable fractions of the real FN partition (all 48 layers)."""
    from mlx2.runtime.amoeba_blocks import layout as L
    z = np.load(FN_PARTITION / "tiles.npz")
    names = ["core"] + [k for k in z.keys() if k.startswith(("box:", "pair:"))]
    meta = json.loads((FN_PARTITION / "partition.json").read_text())
    rows = []
    for i, layer in enumerate(z["layers"]):
        own = L.owner_from_masks({n: z[n][i] for n in names}, names).reshape(512, 640)
        core = own == 0
        g = core.reshape(512, 10, 64)
        lb = L.plan_layer(own, tiles=len(names), group_size=64, chunk=32)
        closed_rows = sum(int(ln) for off, ln, t in lb.chunks if t != 0)
        rows.append({
            "layer": int(layer),
            "core_neuron_share": float(core.mean()),
            "gate_up_rows_skippable_core_only": closed_rows / (512 * 640),
            "down_groups_skippable_core_only": float((~g.any(-1)).mean()),
            "experts_without_core": int((~core.any(-1)).sum()),
            "chunks": int(lb.chunks.shape[0]),
        })
    gu = np.mean([r["gate_up_rows_skippable_core_only"] for r in rows])
    dn = np.mean([r["down_groups_skippable_core_only"] for r in rows])
    summary = {
        "partition": str(FN_PARTITION), "checkpoint": meta.get("checkpoint"), "rule": meta.get("rule"),
        "tiles": len(names), "mean_gate_up_skippable": gu, "mean_down_skippable": dn,
        # gate+up are 2/3 of an expert's weight bytes, down 1/3 (same format)
        "mean_bytes_skippable_core_only_uniform_routing": (2 * gu + dn) / 3,
        "experts_without_core_total": int(sum(r["experts_without_core"] for r in rows)),
        "layers": rows,
    }
    if out:
        Path(out).write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------------------
# Weights and stock arms
# ---------------------------------------------------------------------------


RAW: dict = {}


def load_layer(model: str, layer: int):
    import mlx.core as mx
    from mlx2.runtime.amoeba_blocks.ops import QW
    decl = declaration(model)
    path = MODELS[model]["path"]
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = decl["prefix"].format(layer=layer)
    quant = json.loads((path / "config.json").read_text())["quantization"]
    out = {}
    cache = {}
    for proj in decl["projections"]:
        key = f"{prefix}.{proj}"
        spec = quant.get(key) if isinstance(quant.get(key), dict) else {}
        bits = int(spec.get("bits", quant["bits"]))
        gs = int(spec.get("group_size", quant["group_size"]))
        arrays = []
        for part in ("weight", "scales", "biases"):
            shard = index[f"{key}.{part}"]
            if shard not in cache:
                cache[shard] = mx.load(str(path / shard))
            arrays.append(cache[shard][f"{key}.{part}"])
        w, s, b = arrays
        if w.ndim == 2:
            RAW[(model, layer, proj)] = (w, s, b)       # the loaded 2-D buffers themselves (stock-2d arm)
            w, s, b = w[None], s[None], b[None]
        out[proj] = QW(w, s, b, bits, gs)
    mx.eval(*[a for q in out.values() for a in (q.weight, q.scales, q.biases)])
    return out


def stock_dense(x, q):
    import mlx.core as mx
    from mlx2.runtime.models.activations import swiglu
    g, u, d = (q[n] for n in ("gate_proj", "up_proj", "down_proj"))

    def mm(v, p):
        return mx.quantized_matmul(v, p.weight[0], p.scales[0], p.biases[0], transpose=True,
                                   group_size=p.group_size, bits=p.bits)
    return lambda mask=None: mm(swiglu(mm(x, g), mm(x, u)) * (1 if mask is None else mask), d)


def stock_switch(q):
    """mlx2's own SwitchGLU over the loaded arrays (no random init)."""
    import mlx.nn as nn
    from mlx2.runtime.models import switch_layers as S

    def lin(p):
        m = S.QuantizedSwitchLinear.__new__(S.QuantizedSwitchLinear)
        nn.Module.__init__(m)
        m.weight, m.scales, m.biases = p.weight, p.scales, p.biases
        m.group_size, m.bits, m.mode = p.group_size, p.bits, "affine"
        m.freeze()
        return m

    sg = S.SwitchGLU.__new__(S.SwitchGLU)
    nn.Module.__init__(sg)
    sg.gate_proj, sg.up_proj, sg.down_proj = lin(q["gate_proj"]), lin(q["up_proj"]), lin(q["down_proj"])
    sg.activation = S.SwiGLU()
    sg._sort_policy = "auto"
    sg._sort_counts = {"auto_sorted": 0, "auto_unsorted": 0, "invariant_sorted": 0, "unsorted_decode": 0}
    sg.eval()
    return sg


class MaskedSwitch:
    """Stock SwitchGLU with the activation multiplied by the per-pair open mask (zero but compute)."""

    def __init__(self, sg):
        self.sg = sg

    def __call__(self, x, inds, mask):
        import mlx.core as mx
        from mlx2.runtime.models import switch_layers as S
        sg = self.sg
        xx = mx.expand_dims(x, (-2, -3))
        sort = inds.size >= 64
        idx, inv = inds, None
        if sort:
            xx, idx, inv = S._gather_sort(xx, inds)
            mask = mask.reshape(-1, 1, mask.shape[-1])[mx.argsort(inds.flatten())]
        else:
            mask = mask[..., None, :]
        up = sg.up_proj(xx, idx, sorted_indices=sort)
        gate = sg.gate_proj(xx, idx, sorted_indices=sort)
        y = sg.down_proj(sg.activation(up, gate) * mask, idx, sorted_indices=sort)
        if sort:
            y = S._scatter_unsort(y, inv, inds.shape)
        return y.squeeze(-2)


# ---------------------------------------------------------------------------
# Timing (v2, 2026-11-19)
# ---------------------------------------------------------------------------
#
# The first bench timed each arm once, 3 warm-ups then 20 back-to-back reps, in a fixed order.
# A decode forward is ~0.5-2 ms, so an arm got ~40 ms of GPU work: a new kernel instantiation
# (every new template: NTILES, F, ...) compiled inside the warm-ups with the GPU idle, and the
# timed reps then ran on a GPU still ramping its clock.  Identical work came out 0.5 or 1.9 ms
# depending on whether a compile preceded it.  Now every arm of a group is warmed for at least
# --warm-ms (after its first, compiling call), then the group runs --rounds rounds in a fresh
# random order per round, each arm --reps forwards per round, each forward bracketed by
# mx.synchronize().  Arms of one group share the same clock and cache conditions, and every
# group carries its own stock and ours-ident reference arms, so ratios are within-group.


def pct(ts, q):
    ts = sorted(ts)
    return ts[min(len(ts) - 1, int(q * len(ts)))]


def measure(arms: dict, *, rounds: int, reps: int, warm_ms: float, seed: int, log=None) -> dict:
    """arms: name -> fn (returns arrays to eval).  -> name -> {p10_ms, p50_ms, p90_ms, mean_ms, n,
    round_p50_ms} (or {"error": ...} for an arm whose first call raised; the others still run)."""
    import random

    import mlx.core as mx
    live, out = {}, {}
    for name, fn in arms.items():
        try:
            mx.eval(fn())                    # compile + first run
            mx.synchronize()
            t_end = time.perf_counter() + warm_ms / 1e3
            n = 0
            while n < 2 or time.perf_counter() < t_end:
                mx.eval(fn())
                n += 1
            mx.synchronize()
            live[name] = fn
        except Exception as exc:             # a failing arm must not cost the window
            out[name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            if log:
                log(f"  arm {name} failed: {out[name]['error']}")
    rng = random.Random(seed)
    ts = {n: [] for n in live}
    per_round = {n: [] for n in live}
    for _ in range(rounds):
        order = list(live)
        rng.shuffle(order)
        for name in order:
            fn = live[name]
            rt = []
            for _ in range(reps):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fn())
                mx.synchronize()
                rt.append(time.perf_counter() - t0)
            ts[name] += rt
            per_round[name].append(1e3 * statistics.median(rt))
    for name in live:
        v = ts[name]
        out[name] = {"p10_ms": 1e3 * pct(v, 0.1), "p50_ms": 1e3 * statistics.median(v), "p90_ms": 1e3 * pct(v, 0.9),
                     "mean_ms": 1e3 * statistics.fmean(v), "n": len(v), "round_p50_ms": per_round[name]}
    return out


def maxrel(a, b):
    import mlx.core as mx
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    den = float(mx.max(mx.abs(b32)).item()) or 1.0
    return float(mx.max(mx.abs(a32 - b32)).item()) / den


def seed_of(*parts) -> int:
    """A reproducible round-order seed (hash() of a str changes per process)."""
    import zlib
    return zlib.crc32(repr(parts).encode()) & 0xFFFF


def _ev(v):
    import mlx.core as mx
    mx.eval(v)
    return v


def guarded(fn) -> dict:
    """Run a cell's post-timing checks; an exception becomes the cell's ``check_error`` instead of
    ending the window."""
    try:
        return fn()
    except Exception as exc:
        return {"check_error": f"{type(exc).__name__}: {exc}"[:500]}


def bitwise(xs, ys) -> bool:
    import mlx.core as mx
    return all(mx.array_equal(p, q).item() for p, q in zip(xs, ys))


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------


def stock_nn(q):
    """nn.QuantizedLinear modules over the same arrays (the serving path's module call)."""
    import mlx.nn as nn

    def lin(p):
        m = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(m)
        m.weight, m.scales, m.biases = p.weight[0], p.scales[0], p.biases[0]
        m.group_size, m.bits, m.mode = p.group_size, p.bits, "affine"
        m.freeze()
        return m
    return {n: lin(q[n]) for n in ("gate_proj", "up_proj", "down_proj")}


def run_model(model, a, log, cells):
    import mlx.core as mx
    from mlx2.runtime.amoeba_blocks import layout as L
    from mlx2.runtime.amoeba_blocks import ops as O
    from mlx2.runtime.models.activations import swiglu
    decl = declaration(model)
    moe = decl["kind"] == "switch_glu"
    layers = [int(v) for v in (a.layers_fn if model == "fn" else a.layers_27b).split(",")]
    t0 = time.time()
    weights = {l: load_layer(model, l) for l in layers}
    log(f"{model}: loaded layers {layers} in {time.time() - t0:.0f}s")
    g0 = weights[layers[0]]["gate_proj"]
    E, F, d = g0.experts, g0.n, g0.k
    gs = weights[layers[0]]["down_proj"].group_size
    rng = np.random.default_rng(0)
    shapes = [s for s in SHAPES if s in a.shapes.split(",")]
    bshapes = [s for s in BOUNDARY_SHAPES if s in a.boundary_shapes.split(",") and s not in shapes]
    allshapes = {**SHAPES, **BOUNDARY_SHAPES}
    xs = {s: mx.array(rng.standard_normal((allshapes[s], d)).astype(np.float32)).astype(mx.bfloat16)
          for s in shapes + bshapes}
    inds = {}
    if moe:
        for s, x in xs.items():
            n = int(x.shape[0])
            inds[s] = mx.array(np.stack([rng.choice(E, FN_TOPK, replace=False) for _ in range(n)]).astype(np.int32))
    backends = list(a.active_backends)
    ref_be = backends[0]
    stock = {l: (stock_switch(weights[l]) if moe else None) for l in layers}
    # the loaded 2-D arrays (stock-base calls quantized_matmul on [0] views of their [1, N, K] views)
    flat = {}
    mods = {}
    if not moe:
        for l in layers:
            flat[l] = {n: RAW[(model, l, n)] for n in decl["projections"]}
            mods[l] = stock_nn(weights[l])
    mx.eval(*xs.values(), *inds.values())

    def stock_fn(s, masks=None):
        x = xs[s]
        if moe:
            if masks is None:
                return lambda: [stock[l](x, inds[s]) for l in layers]
            ms = {l: MaskedSwitch(stock[l]) for l in layers}
            return lambda: [ms[l](x, inds[s], masks[l]) for l in layers]
        if masks is None:
            return lambda: [stock_dense(x, weights[l])() for l in layers]
        return lambda: [stock_dense(x, weights[l])(masks[l]) for l in layers]

    def stock_2d(s):
        x = xs[s]

        def mm(v, p, bits, gsz):
            return mx.quantized_matmul(v, p[0], p[1], p[2], transpose=True, group_size=gsz, bits=bits)

        def f():
            out = []
            for l in layers:
                q = weights[l]
                fl = flat[l]
                h = swiglu(mm(x, fl["gate_proj"], q["gate_proj"].bits, q["gate_proj"].group_size),
                           mm(x, fl["up_proj"], q["up_proj"].bits, q["up_proj"].group_size))
                out.append(mm(h, fl["down_proj"], q["down_proj"].bits, q["down_proj"].group_size))
            return out
        return f

    def stock_mod(s):
        x = xs[s]
        return lambda: [mods[l]["down_proj"](swiglu(mods[l]["gate_proj"](x), mods[l]["up_proj"](x))) for l in layers]

    def stock_gate_only(s):
        x = xs[s]
        return lambda: [mods[l]["gate_proj"](x) for l in layers]

    def ours_fn(s, packedw, blocks, topen, backend, skip=True, count=False, law=None):
        x = xs[s]
        n = int(x.shape[0])
        to = mx.contiguous(mx.broadcast_to(topen, (n, topen.shape[1])) if topen.shape[0] == 1 else topen)
        mx.eval(to)

        def f():
            outs = []
            for l in layers:
                g, u = packedw[l]
                dn = weights[l]["down_proj"]
                if moe:
                    y, c = O.block_switch_glu(x, inds[s], g, u, dn, blocks[l], to, skip=skip, count=count,
                                              backend_name=backend, law=law)
                else:
                    y, c = O.block_mlp(x, g, u, dn, blocks[l], to, skip=skip, count=count, backend_name=backend,
                                       law=law)
                outs.append((y, c))
            return outs
        return f

    def ys(outs):
        return [y for y, _ in outs]

    def law_of(s, be):
        n = int(xs[s].shape[0])
        return O.select_law(be, n, d, sorted_rows=(not moe) or n * FN_TOPK >= O.SORT_MIN_PAIRS,
                            mean_rows=n * FN_TOPK / E if moe else None)

    def plan(s):
        pre = s.startswith("prefill")
        return dict(rounds=a.rounds, reps=a.reps_prefill if pre else a.reps, warm_ms=a.warm_ms)

    def emit(base, res, extra=None):
        cell = {"model": model, "checkpoint": MODELS[model]["checkpoint"], **base, **res, **(extra or {})}
        cells.append(cell)
        if "p50_ms" in res:
            log(f"{model} {base['arm']} {base.get('partition', '')} {base['shape']} {base.get('backend', '')} "
                f"{base.get('law', '')}: "
                f"p50 {res['p50_ms']:.3f} ms (p10 {res['p10_ms']:.3f}, p90 {res['p90_ms']:.3f})")
        return cell

    ident = {l: O.DeviceBlocks.from_layout(L.identity_layer(E, F, group_size=gs, chunk=32, tiles=1)) for l in layers}
    unpacked = {l: (weights[l]["gate_proj"], weights[l]["up_proj"]) for l in layers}
    one = mx.ones((1, 1), mx.uint8)
    bytes_dense = len(layers) * O.dense_bytes(*(weights[layers[0]][p] for p in decl["projections"])) if not moe else None

    # ---- group 1 per shape: stock baselines (fairness diagnostics) and ours-ident under both laws.
    # Boundary shapes (--boundary-shapes, baseline only) place the lane/prefill switch point.
    base_out, ident_out = {}, {}
    for s in shapes + bshapes:
        n = int(xs[s].shape[0])
        pre = s.startswith("prefill")
        arms = {"stock-base": stock_fn(s)}
        if not moe:
            arms["stock-2d"] = stock_2d(s)
            arms["stock-nn"] = stock_mod(s)
            if not pre:
                arms["stock-gate-proj-only"] = stock_gate_only(s)
        laws = {}
        for be in backends:
            auto = law_of(s, be)
            arms[f"ours-ident/{be}"] = ours_fn(s, unpacked, ident, one, be)
            laws[f"ours-ident/{be}"] = auto
            other = "lane" if auto == "prefill" else "prefill"
            # the other law, pinned: prefill shapes only, and never simd's lane law past 512 rows (its
            # 8-row tiles re-read the weights ~n/8 times: 1.8 s per 27B forward at 2048 rows)
            if pre and not a.no_lane_prefill and not (be == "simd" and other == "lane" and n > 512):
                try:
                    O.select_law(be, n, d, law=other, sorted_rows=(not moe) or n * FN_TOPK >= O.SORT_MIN_PAIRS)
                    arms[f"ours-ident@{other}/{be}"] = ours_fn(s, unpacked, ident, one, be, law=other)
                    laws[f"ours-ident@{other}/{be}"] = other
                except O.BlockSparseUnsupported:
                    pass
        res = measure(arms, **plan(s), seed=seed_of(model, "baseline", s), log=log)
        try:
            base_out[s] = stock_fn(s)()
            mx.eval(base_out[s])
        except Exception as exc:
            log(f"{model} {s}: stock reference failed: {exc}")
            continue
        outs_by_law = {}
        for name, r in res.items():
            arm, _, be = name.partition("/")
            extra = {"tokens_per_s": n / (r["p50_ms"] / 1e3)} if "p50_ms" in r else {}
            if arm.startswith("stock"):
                if bytes_dense and "p50_ms" in r and arm != "stock-gate-proj-only":
                    extra["gb_per_s_weights"] = bytes_dense / (r["p50_ms"] / 1e3) / 1e9
                if arm not in ("stock-base", "stock-gate-proj-only") and "p50_ms" in r:
                    extra.update(guarded(lambda: {"bitwise_equals_stock_base": bitwise(_ev(arms[name]()), base_out[s])}))
                emit({"arm": arm, "shape": s, "backend": "stock", "group": "baseline"}, r, extra)
                continue
            law = laws[name]
            if "p50_ms" in r:
                def checks():
                    outs = ours_fn(s, unpacked, ident, one, be, count=True, law=law)()
                    mx.eval([y for y, _ in outs])
                    yv = _ev(ys(arms[name]()))
                    outs_by_law[(be, law)] = yv
                    if law == laws[f"ours-ident/{be}"] and s in shapes:
                        ident_out[(s, be)] = yv
                    return {"reads": [O.counter_reads(c) for _, c in outs],
                            "max_rel_vs_stock": max(maxrel(o, b) for o, b in zip(yv, base_out[s]))}
                extra.update(guarded(checks))
                st = res.get("stock-base", {}).get("p50_ms")
                if st:
                    extra["speed_vs_stock"] = st / r["p50_ms"]
            emit({"arm": arm.split("@")[0], "law": law, "pinned": "@" in arm, "shape": s, "backend": be,
                  "group": "baseline", "open_fraction": 1.0}, r, extra)
        for be in backends:
            if (be, "lane") in outs_by_law and (be, "prefill") in outs_by_law:
                cells.append({"model": model, "arm": "law-diff", "shape": s, "backend": be,
                              "max_rel_prefill_vs_lane": max(maxrel(p_, q_) for p_, q_ in
                                                             zip(outs_by_law[(be, "prefill")], outs_by_law[(be, "lane")]))})

    # ---- partitions
    parts = []
    for kind in ("aligned", "scattered"):
        for p in [float(v) for v in a.fractions.split(",")]:
            parts.append((f"{kind}@{p:g}", {l: synthetic_owner(rng, E, F, p, kind, gs) for l in layers}, 2,
                          np.array([[1, 0]], np.uint8), p))
    if model == "fn" and not a.no_gates:
        gate_cells(model, fn_real_partition(layers)[0], d, {k: v for k, v in xs.items() if k in shapes}, rng, a,
                   log, cells, emit)
    if model == "fn" and not a.no_real:
        names, owners, top, _ = fn_real_partition(layers)
        nt = len(names)
        core_only = np.zeros((1, nt), np.uint8)
        core_only[0, 0] = 1
        plus = core_only.copy()
        plus[0, names.index(top)] = 1
        alls = np.ones((1, nt), np.uint8)
        for tag, to in (("real:all", alls), ("real:core", core_only), (f"real:core+{top}", plus)):
            frac = float(np.mean([to[0][owners[l]].mean() for l in layers]))
            parts.append((tag, owners, nt, to, frac))
    for tag, owners, nt, topen_np, frac in parts:
        lbs = {l: L.plan_layer(owners[l], tiles=nt, group_size=gs, chunk=32) for l in layers}
        blocks = {l: O.DeviceBlocks.from_layout(lbs[l]) for l in layers}
        packedw = {l: (O.pack_qweight(weights[l]["gate_proj"], lbs[l]), O.pack_qweight(weights[l]["up_proj"], lbs[l]))
                   for l in layers}
        mx.eval([a_ for pw in packedw.values() for q in pw for a_ in (q.weight, q.scales, q.biases)])
        topen = mx.array(topen_np)
        for s in shapes:
            n = int(xs[s].shape[0])
            arms = {"stock-base": stock_fn(s), f"ours-ident/{ref_be}": ours_fn(s, unpacked, ident, one, ref_be)}
            masks = None
            if tag in ("scattered@0.4", "real:core"):     # stock zero-but-compute arm
                masks = {}
                for l in layers:
                    m_ = mx.array(topen_np[0][owners[l]].astype(np.float32)).astype(mx.bfloat16)  # [E, F]
                    masks[l] = m_[inds[s]] if moe else m_[0]
                arms["stock-masked"] = stock_fn(s, masks)
            for be in backends:
                arms[f"ours-skip/{be}"] = ours_fn(s, packedw, blocks, topen, be)
                if not a.no_mask_arms:
                    arms[f"ours-mask/{be}"] = ours_fn(s, packedw, blocks, topen, be, skip=False)
            res = measure(arms, **plan(s), seed=seed_of(model, tag, s), log=log)
            st = res.get("stock-base", {}).get("p50_ms")
            idm = res.get(f"ours-ident/{ref_be}", {}).get("p50_ms")
            for name in ("stock-base", "stock-masked", f"ours-ident/{ref_be}"):
                if name in res:
                    arm, _, be = name.partition("/")
                    emit({"arm": arm, "partition": tag, "shape": s, "backend": be or "stock", "group": tag,
                          "reference": True}, res[name])
            for be in backends:
                r = res.get(f"ours-skip/{be}", {})
                cell = {"arm": "ours-skip", "partition": tag, "open_fraction": frac, "shape": s, "backend": be,
                        "law": law_of(s, be), "group": tag}
                extra = {}
                if "p50_ms" in r:
                    def checks():
                        outs = ours_fn(s, packedw, blocks, topen, be, count=True)()
                        sk = ys(ours_fn(s, packedw, blocks, topen, be)())
                        ms = ys(ours_fn(s, packedw, blocks, topen, be, skip=False)())
                        mx.eval([y for y, _ in outs], sk, ms)
                        reads = [O.counter_reads(c) for _, c in outs]
                        q0 = weights[layers[0]]
                        byts = [L.bytes_from_reads(rd, d=d, bits=q0["gate_proj"].bits,
                                                   group_size=q0["gate_proj"].group_size,
                                                   down_bits=q0["down_proj"].bits, down_group_size=gs,
                                                   down_cols=O.down_cols(be, law_of(s, be)))["total_bytes"]
                                for rd in reads]
                        extra = {"tokens_per_s": n / (r["p50_ms"] / 1e3), "reads": reads, "bytes_read": int(sum(byts)),
                                 "gb_per_s": sum(byts) / (r["p50_ms"] / 1e3) / 1e9,
                                 "skip_equals_masked_bitwise": bitwise(sk, ms)}
                        if st:
                            extra["speed_vs_stock"] = st / r["p50_ms"]
                        if idm and be == ref_be:
                            extra["speed_vs_ours_ident"] = idm / r["p50_ms"]
                        if tag in ("real:all", "scattered@0.4", "aligned@1", "scattered@1") and (s, be) in ident_out:
                            allopen = ys(ours_fn(s, packedw, blocks, mx.ones((1, nt), mx.uint8), be)())
                            extra["packed_all_open_equals_ident_bitwise"] = bitwise(allopen, ident_out[(s, be)])
                        if frac >= 0.999 or tag == "real:all":
                            extra["max_rel_vs_stock"] = max(maxrel(o, b) for o, b in zip(sk, base_out[s]))
                        rm = res.get(f"ours-mask/{be}", {})
                        if "p50_ms" in rm:
                            extra["ours_mask_p50_ms"] = rm["p50_ms"]
                        return extra
                    extra = guarded(checks)
                emit(cell, r, extra)
        packedw = blocks = None  # release (captured by the arm closures above)
        mx.clear_cache()
    weights = flat = mods = None  # release (captured by the arm closures above)
    for key in [k for k in RAW if k[0] == model]:
        del RAW[key]
    mx.clear_cache()


def gate_cells(model, names, d, xs, rng, a, log, cells, emit):
    """Tile gates on the real FN tile graph and a wide synthetic one (random probes and token prior:
    cost, not decisions).

    One gate call per token per forward gates every layer (tile ids are global), so each arm is
    one call.  Arms: fused kernel, split kernels, the same two on the token law, host token law
    (ids already on the host -> closure -> upload), host probe law (numpy; needs h on the host, a
    sync in a real loop).  ``auto_impl`` records what the default picks for that shape."""
    from mlx2.runtime.amoeba_blocks import layout as L
    real = L.graph_from_partition_names(names)
    # a wide synthetic graph (416 boxes = 13 words, the tile count of the real one): where split's
    # one-threadgroup-per-word parallelism exists
    wb = len(names)
    wide = L.TileGraph([f"b{i}" for i in range(wb)], ["core"] + [f"t{i}" for i in range(1, len(names))],
                       [()] + [(i - 1,) for i in range(1, len(names))], parent=[-1] * wb)
    for gname, graph in (("real", real), ("wide", wide)):
        gate_graph_cells(model, gname, graph, d, xs, rng, a, log, emit)


def gate_graph_cells(model, gname, graph, d, xs, rng, a, log, emit):
    import mlx.core as mx
    from mlx2.runtime.amoeba_blocks import ops as O
    nb = len(graph.boxes)
    tables = O.GateTables.from_graph(graph, rng.standard_normal((nb, d)).astype(np.float32) / np.sqrt(d),
                                     np.zeros(nb, np.float32))
    vocab = 4096
    prior = (rng.random((vocab, nb)) < 0.03).astype(np.uint8)
    host = O.HostGateTables.from_tables(tables, prior)
    prior_dev = mx.array(prior)
    for s, x in xs.items():
        t = int(x.shape[0])
        un = t > 1
        toks = rng.integers(0, vocab, size=t)
        toks_dev = mx.array(toks.astype(np.int32))
        h_host = np.array(x.astype(mx.float32))
        box_dev = prior_dev[toks_dev]
        mx.eval(box_dev)

        # every arm returns (topen, union) and both are evaluated: equal work across arms
        def host_token():
            tp, uu = O.host_tile_gates(host, tokens=toks, union=un)
            return [mx.array(tp)] + ([mx.array(uu)] if un else [])

        def host_probe():
            tp, uu = O.host_tile_gates(host, h=h_host, union=un)
            return [mx.array(tp)] + ([mx.array(uu)] if un else [])

        def gpu(impl, token):
            def f():
                tp, uu = O.tile_gates(x, tables, union=un, impl=impl,
                                      box_open=prior_dev[toks_dev] if token else None)
                return [tp] + ([uu] if un else [])
            return f
        arms = {"fused": gpu("fused", False), "split": gpu("split", False), "fused-token": gpu("fused", True),
                "split-token": gpu("split", True), "host-token": host_token, "host-probe": host_probe}
        res = measure(arms, rounds=a.rounds, reps=a.reps, warm_ms=a.warm_ms, seed=seed_of("gate", gname, s),
                      log=log)
        # host-side cost alone (no upload, no GPU): what a step-overlapped host gate costs
        host_cpu = {}
        for nm, kw in (("host-token", {"tokens": toks}), ("host-probe", {"h": h_host})):
            ts = []
            for _ in range(max(20, a.reps)):
                t0 = time.perf_counter()
                O.host_tile_gates(host, union=un, **kw)
                ts.append(time.perf_counter() - t0)
            host_cpu[nm] = 1e3 * statistics.median(ts)
        f_t, f_u = O.tile_gates(x, tables, union=True, impl="fused")
        s_t, s_u = O.tile_gates(x, tables, union=True, impl="split")
        ft_t, _ = O.tile_gates(x, tables, box_open=box_dev, impl="fused")
        st_t, _ = O.tile_gates(x, tables, box_open=box_dev, impl="split")
        mx.eval(f_t, f_u, s_t, s_u, ft_t, st_t)
        ht, _ = O.host_tile_gates(host, tokens=toks)
        hp, _ = O.host_tile_gates(host, h=h_host)
        checks = {"split_equals_fused_probe_bitwise": bool(mx.array_equal(f_t, s_t).item() and mx.array_equal(f_u, s_u).item()),
                  "split_equals_fused_token_bitwise": bool(mx.array_equal(ft_t, st_t).item()),
                  "host_token_equals_fused_bitwise": bool(np.array_equal(ht, np.array(ft_t))),
                  "host_probe_rows_differing_from_fused": int((hp != np.array(f_t)).any(1).sum())}
        for name, r in res.items():
            extra = {"graph": gname, "boxes": nb, "words": tables.words, "tiles": tables.tiles,
                     "auto_impl": O.gate_impl(t, tables.words), "calls_per_forward": 1, **checks}
            if name in host_cpu:
                extra["host_cpu_only_ms"] = host_cpu[name]
            emit({"arm": f"tile-gates/{name}", "partition": gname, "shape": s,
                  "backend": "host" if name.startswith("host") else "gpu", "group": "gates"}, r, extra)
        log(f"{model} gates {gname} {s}: {checks}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class CellLog(list):
    """Cells, each also appended to a JSONL file as it finishes (a watchdog abort keeps them)."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def append(self, cell):
        super().append(cell)
        with open(self.path, "a") as f:
            f.write(json.dumps(cell, default=str) + "\n")


def artifact_record(model: str) -> dict:
    """The checkpoint this bench reads, checked where mlx2 can recompute the fingerprint."""
    import hashlib
    path = MODELS[model]["path"]
    want = MODELS[model]["checkpoint"]
    rec = {"checkpoint": want, "path": str(path)}
    if model == "fn":
        from mlx2.adapters.flash_next import artifact_identity
        fp = artifact_identity(path)["fingerprint"]
        if not fp.startswith(want.rsplit("@", 1)[1]):
            sys.exit(f"{path}: fingerprint {fp[:12]} is not {want}")
        rec["fingerprint"] = fp
    else:
        # the documented id; also bind what was read (config, index, shard sizes and mtimes)
        h = hashlib.sha256()
        for name in ("config.json", "model.safetensors.index.json"):
            h.update((path / name).read_bytes())
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        for shard in sorted(set(index.values())):
            st = (path / shard).stat()
            h.update(json.dumps([shard, st.st_size, st.st_mtime_ns]).encode())
        rec["bench_files_sha256"] = h.hexdigest()
        rec["fingerprint_source"] = "documented id (docs/research); not recomputed here"
    return rec


# Guessed seconds per 3-layer forward (decode from the 2026-11-19 bench; prefill for the new law
# assumes ~stock, simd ~4x): (model, shape).
_GUESS = {("fn", "decode1"): 0.001, ("fn", "decode4"): 0.0015, ("fn", "prefill2048"): 0.04,
          ("27b", "decode1"): 0.005, ("27b", "decode4"): 0.005, ("27b", "prefill2048"): 0.15}
_SIMD_PREFILL = {"fn": 0.15, "27b": 0.6}
_LANE_PREFILL = {"fn": 0.09, "27b": 0.85}


def estimate(a) -> dict:
    """Rough window length: per arm (warm-up + rounds x reps + ~4 checking forwards)."""
    fr = len(a.fractions.split(","))
    bes = a.backends.split(",")
    total = 0.0
    for model in a.models.split(","):
        parts = 2 * fr + (0 if model != "fn" or a.no_real else 3)
        for s in a.shapes.split(","):
            pre = s.startswith("prefill")
            reps = a.rounds * (a.reps_prefill if pre else a.reps)
            g = _GUESS[(model, s)]
            per = {be: (_SIMD_PREFILL[model] if be == "simd" and pre else g) for be in bes}
            fw = lambda t: a.warm_ms / 1e3 + t * (reps + 5)                              # noqa: E731
            base = fw(g) * (1 if model == "fn" else 3) + sum(fw(per[be]) for be in bes)
            if pre and "mpp" in bes and not a.no_lane_prefill:
                base += fw(_LANE_PREFILL[model])
            part = fw(g) * 2 + sum(fw(per[be]) * (1 if a.no_mask_arms else 2) for be in bes)
            total += base + parts * part
        for s in a.boundary_shapes.split(","):
            if s in BOUNDARY_SHAPES:
                rows = BOUNDARY_SHAPES[s]
                g = _GUESS[(model, "prefill2048")] * rows / 2048 + 0.002
                total += (a.warm_ms / 1e3 + g * (a.rounds * a.reps_prefill + 5)) * (3 + 2 * len(bes) + 2)
        total += 60 + parts * 10                                                           # load + packing
    total += 60                                                                            # gate cells
    return {"estimated_minutes": round(total / 60, 1), "guess_s_per_forward": {f"{k[0]}/{k[1]}": v for k, v in
                                                                              _GUESS.items()},
            "guess_simd_prefill_s": _SIMD_PREFILL}


def print_window(a):
    rev = git_rev()
    runs = f"$HOME/amoeba-runs/amoeba-blocks-bench-{rev}"
    swap = "--swap-growth-gb 2"
    return (f"scripts/amoeba27b_window_q.sh blocksbench {runs} -- \\\n"
            f"  ~/Desktop/mlx2/.venv/bin/python scripts/amoeba_blocks_bench.py --out {runs}/bench {swap}\n"
            f"# daytime: --swap-growth-gb 8 --waiver-note \"Pierre 2026-11-18 approved 8 GB growth for daytime windows\"")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="fresh output directory (refused if nonempty)")
    ap.add_argument("--models", default="fn,27b")
    ap.add_argument("--layers-fn", default=",".join(map(str, MODELS["fn"]["layers"])))
    ap.add_argument("--layers-27b", default=",".join(map(str, MODELS["27b"]["layers"])))
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--backends", default="mpp,simd")
    ap.add_argument("--fractions", default="1.0,0.7,0.4,0.2")
    ap.add_argument("--rounds", type=int, default=4, help="randomized-order rounds per group")
    ap.add_argument("--reps", type=int, default=15, help="forwards per arm per round (decode)")
    ap.add_argument("--reps-prefill", type=int, default=3, help="forwards per arm per round (prefill)")
    ap.add_argument("--warm-ms", type=float, default=250, help="warm-up per arm after its first call")
    ap.add_argument("--no-lane-prefill", action="store_true", help="skip the lane-law prefill reference arm")
    ap.add_argument("--no-mask-arms", action="store_true", help="do not time ours-mask@p (SKIP=0)")
    ap.add_argument("--no-real", action="store_true", help="skip the real FN partition arms")
    ap.add_argument("--no-gates", action="store_true", help="skip the tile-gate cells (FN)")
    ap.add_argument("--boundary-shapes", default=",".join(BOUNDARY_SHAPES),
                    help="extra prefill sizes timed in the baseline group only, both laws pinned")
    ap.add_argument("--memory-gb", type=float, default=80)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--dry-run", action="store_true", help="CPU: print the plan and estimate, no GPU")
    ap.add_argument("--analyze", metavar="JSON", help="CPU: real FN partition skippable fractions, no GPU")
    ap.add_argument("--print-window", action="store_true")
    a = ap.parse_args()
    if a.print_window:
        print(print_window(a))
        return 0
    if a.analyze:
        s = analyze(a.analyze)
        print(json.dumps({k: v for k, v in s.items() if k != "layers"}, indent=1))
        return 0
    if a.dry_run:
        print(json.dumps({"plan": vars(a), **estimate(a), "window": print_window(a)}, indent=1))
        return 0
    if a.swap_growth_gb > 2 and not a.waiver_note:
        sys.exit("--swap-growth-gb above 2 needs --waiver-note (who approved it)")
    if not a.out:
        sys.exit("--out is required")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    import amoeba27b_lib as A
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    A.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb)
    A.setup_mlx(a.memory_gb, 2)
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    info = mx.device_info()
    receipt = {"schema": "mlx2.amoeba-blocks-bench.v2", "mlx2_revision": git_rev(), "mlx": mx.__version__,
               "device": info.get("device_name"), "args": vars(a), "estimate": estimate(a),
               "checkpoints": {m: MODELS[m]["checkpoint"] for m in a.models.split(",")},
               "status": "unqualified research bench; no route selects these kernels",
               "watchdog": {"swap_growth_gb": a.swap_growth_gb, "waiver_note": a.waiver_note or None}}
    from mlx2.runtime.lane import matmul as lane
    a.active_backends = [b for b in a.backends.split(",") if b != "mpp" or lane._tensor_units()]
    receipt["backends_dropped"] = sorted(set(a.backends.split(",")) - set(a.active_backends))
    receipt["artifacts"] = {m: artifact_record(m) for m in a.models.split(",")}
    cells = CellLog(out / "cells.jsonl")
    t0 = time.time()
    try:
        for model in a.models.split(","):
            run_model(model, a, log, cells)
    finally:
        receipt["cells"] = list(cells)
        receipt["seconds"] = round(time.time() - t0, 1)
        receipt["peak_footprint_gb"] = A.WATCH.get("peak_footprint_gb")
        tmp = out / "receipt.json.tmp"
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out / "receipt.json")
        log(f"done: {len(cells)} cells in {receipt['seconds']}s -> {out / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
