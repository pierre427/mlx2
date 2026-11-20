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

Reported: median ms per layer-set forward, tokens/s, weight bytes read (device counters) and
GB/s, and exactness: ours packed all-open == ours-ident (bitwise), ours-skip == ours-mask
(bitwise), max |diff| vs stock (relative to the stock output's max).

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
# Timing
# ---------------------------------------------------------------------------


def timeit(fn, reps, warmup):
    import mlx.core as mx
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_ms": 1e3 * statistics.median(ts), "p10_ms": 1e3 * ts[len(ts) // 10],
            "p90_ms": 1e3 * ts[(9 * len(ts)) // 10]}


def maxrel(a, b):
    import mlx.core as mx
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    den = float(mx.max(mx.abs(b32)).item()) or 1.0
    return float(mx.max(mx.abs(a32 - b32)).item()) / den


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------


def run_model(model, a, log, cells):
    import mlx.core as mx
    from mlx2.runtime.amoeba_blocks import layout as L
    from mlx2.runtime.amoeba_blocks import ops as O
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
    xs = {s: mx.array(rng.standard_normal((n, d)).astype(np.float32)).astype(mx.bfloat16) for s, n in SHAPES.items()
          if s in a.shapes.split(",")}
    inds = {}
    if moe:
        for s, x in xs.items():
            n = int(x.shape[0])
            inds[s] = mx.array(np.stack([rng.choice(E, FN_TOPK, replace=False) for _ in range(n)]).astype(np.int32))
    backends = list(a.active_backends)
    stock = {l: (stock_switch(weights[l]) if moe else None) for l in layers}
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

    def ours_fn(s, packedw, blocks, topen, backend, skip=True, count=False):
        x = xs[s]
        n = int(x.shape[0])
        to = mx.broadcast_to(topen, (n, topen.shape[1])) if topen.shape[0] == 1 else topen
        to = mx.contiguous(to)

        def f():
            outs = []
            for l in layers:
                g, u = packedw[l]
                dn = weights[l]["down_proj"]
                if moe:
                    y, c = O.block_switch_glu(x, inds[s], g, u, dn, blocks[l], to, skip=skip, count=count,
                                              backend_name=backend)
                else:
                    y, c = O.block_mlp(x, g, u, dn, blocks[l], to, skip=skip, count=count, backend_name=backend)
                outs.append((y, c))
            return outs
        return f

    def ys(outs):
        return [y for y, _ in outs]

    base_out = {}
    for s in xs:
        r = timeit(stock_fn(s), a.reps, a.warmup)
        base_out[s] = stock_fn(s)()
        mx.eval(base_out[s])
        cells.append({"model": model, "arm": "stock-base", "shape": s, "backend": "stock", **r,
                      "tokens_per_s": int(xs[s].shape[0]) / (r["median_ms"] / 1e3),
                      "bytes_read_est": len(layers) * O.dense_bytes(*(weights[layers[0]][p] for p in decl["projections"]))
                      if not moe else None})
        log(f"{model} stock-base {s}: {r['median_ms']:.2f} ms")
    # identity layout (law base, all open)
    ident = {l: O.DeviceBlocks.from_layout(L.identity_layer(E, F, group_size=gs, chunk=32, tiles=1)) for l in layers}
    unpacked = {l: (weights[l]["gate_proj"], weights[l]["up_proj"]) for l in layers}
    one = mx.ones((1, 1), mx.uint8)
    ident_out = {}
    for s in xs:
        for be in backends:
            r = timeit(ours_fn(s, unpacked, ident, one, be), reps_for(a, be, s), a.warmup)
            outs = ours_fn(s, unpacked, ident, one, be, count=True)()
            mx.eval([y for y, _ in outs])
            ident_out[(s, be)] = ys(ours_fn(s, unpacked, ident, one, be)())
            mx.eval(ident_out[(s, be)])
            reads = [O.counter_reads(c) for _, c in outs]
            cells.append({"model": model, "arm": "ours-ident", "shape": s, "backend": be, **r,
                          "tokens_per_s": int(xs[s].shape[0]) / (r["median_ms"] / 1e3), "reads": reads,
                          "max_rel_vs_stock": max(maxrel(o, b) for o, b in zip(ident_out[(s, be)], base_out[s]))})
            log(f"{model} ours-ident {s} {be}: {r['median_ms']:.2f} ms")
    # partitions
    parts = []
    for kind in ("aligned", "scattered"):
        for p in [float(v) for v in a.fractions.split(",")]:
            parts.append((f"{kind}@{p:g}", {l: synthetic_owner(rng, E, F, p, kind, gs) for l in layers}, 2,
                          np.array([[1, 0]], np.uint8), p))
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
        # the fused tile-gate kernel on the real tile graph (random probes: cost, not decisions)
        graph = L.graph_from_partition_names(names)
        nb = len(graph.boxes)
        tables = O.GateTables.from_graph(graph, rng.standard_normal((nb, d)).astype(np.float32) / np.sqrt(d),
                                         np.zeros(nb, np.float32))
        for s in xs:
            r = timeit(lambda: [O.tile_gates(xs[s], tables, union=xs[s].shape[0] > 1)[0] for _ in layers],
                       a.reps, a.warmup)
            cells.append({"model": model, "arm": "tile-gates", "shape": s, "backend": "simd", "boxes": nb,
                          "tiles": nt, **r})
            log(f"{model} tile-gates {s}: {r['median_ms']:.3f} ms for {len(layers)} layers")
    for tag, owners, nt, topen_np, frac in parts:
        lbs = {l: L.plan_layer(owners[l], tiles=nt, group_size=gs, chunk=32) for l in layers}
        blocks = {l: O.DeviceBlocks.from_layout(lbs[l]) for l in layers}
        packedw = {l: (O.pack_qweight(weights[l]["gate_proj"], lbs[l]), O.pack_qweight(weights[l]["up_proj"], lbs[l]))
                   for l in layers}
        mx.eval([a_ for pw in packedw.values() for q in pw for a_ in (q.weight, q.scales, q.biases)])
        topen = mx.array(topen_np)
        for s in xs:
            n = int(xs[s].shape[0])
            if tag in ("scattered@0.4", "real:core"):     # one stock zero-but-compute arm per shape
                masks = {}
                for l in layers:
                    m_ = mx.array(topen_np[0][owners[l]].astype(np.float32)).astype(mx.bfloat16)  # [E, F]
                    masks[l] = m_[inds[s]] if moe else m_[0]
                r = timeit(stock_fn(s, masks), a.reps, a.warmup)
                cells.append({"model": model, "arm": "stock-masked", "partition": tag, "shape": s, "backend": "stock",
                              **r, "tokens_per_s": n / (r["median_ms"] / 1e3)})
                log(f"{model} stock-masked {tag} {s}: {r['median_ms']:.2f} ms")
            for be in backends:
                r = timeit(ours_fn(s, packedw, blocks, topen, be), reps_for(a, be, s), a.warmup)
                outs = ours_fn(s, packedw, blocks, topen, be, count=True)()
                sk = ys(ours_fn(s, packedw, blocks, topen, be)())
                ms = ys(ours_fn(s, packedw, blocks, topen, be, skip=False)())
                mx.eval([y for y, _ in outs], sk, ms)
                reads = [O.counter_reads(c) for _, c in outs]
                q0 = weights[layers[0]]
                byts = [L.bytes_from_reads(rd, d=d, bits=q0["gate_proj"].bits, group_size=q0["gate_proj"].group_size,
                                           down_bits=q0["down_proj"].bits, down_group_size=gs,
                                           down_cols=O.down_cols(be))["total_bytes"] for rd in reads]
                cell = {"model": model, "arm": "ours-skip", "partition": tag, "open_fraction": frac, "shape": s,
                        "backend": be, **r, "tokens_per_s": n / (r["median_ms"] / 1e3), "reads": reads,
                        "bytes_read": int(sum(byts)), "gb_per_s": sum(byts) / (r["median_ms"] / 1e3) / 1e9,
                        "skip_equals_masked_bitwise": all(mx.array_equal(p_, q_).item() for p_, q_ in zip(sk, ms))}
                if tag in ("real:all", "scattered@0.4"):
                    # a nontrivial packing with every tile open must give the unpacked layout's bits
                    allopen = ys(ours_fn(s, packedw, blocks, mx.ones((1, nt), mx.uint8), be)())
                    cell["packed_all_open_equals_ident_bitwise"] = all(
                        mx.array_equal(p_, q_).item() for p_, q_ in zip(allopen, ident_out[(s, be)]))
                if tag.endswith("@1") or tag == "real:all":
                    cell["max_rel_vs_stock"] = max(maxrel(o, b) for o, b in zip(sk, base_out[s]))
                if not a.no_mask_arms:
                    rm = timeit(ours_fn(s, packedw, blocks, topen, be, skip=False), reps_for(a, be, s), a.warmup)
                    cell["ours_mask_median_ms"] = rm["median_ms"]
                cells.append(cell)
                log(f"{model} ours-skip {tag} {s} {be}: {r['median_ms']:.2f} ms, {sum(byts) / 1e6:.1f} MB, "
                    f"skip==mask {cell['skip_equals_masked_bitwise']}")
        del packedw, blocks
        mx.clear_cache()
    del weights
    mx.clear_cache()


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


# Guessed seconds per 3-layer forward (no measurement yet): (model, backend, shape).  simd prefill
# re-reads the weights once per 8-row work tile (27B: ~200 GB per forward), hence --slow-reps.
_GUESS = {("fn", "decode1"): 0.004, ("fn", "decode4"): 0.008, ("fn", "prefill2048"): 0.08,
          ("27b", "decode1"): 0.006, ("27b", "decode4"): 0.008, ("27b", "prefill2048"): 0.08}
_SIMD_PREFILL = {"fn": 0.12, "27b": 0.6}


def reps_for(a, backend: str, shape: str) -> int:
    return min(a.reps, a.slow_reps) if backend == "simd" and shape.startswith("prefill") else a.reps


def estimate(a) -> dict:
    """Rough window length from per-forward guesses (cells x timed and checking forwards)."""
    fr = len(a.fractions.split(","))
    total = 0.0
    for model in a.models.split(","):
        parts = 2 * fr + (0 if model != "fn" or a.no_real else 3)
        for s in a.shapes.split(","):
            total += _GUESS[(model, s)] * (a.warmup + a.reps) * 3                 # stock base + 2 masked
            for be in a.backends.split(","):
                per = _SIMD_PREFILL[model] if be == "simd" and s.startswith("prefill") else _GUESS[(model, s)]
                reps = reps_for(a, be, s)
                forwards = a.warmup + reps + 5 + (0 if a.no_mask_arms else a.warmup + reps)
                total += per * forwards * (1 + parts)
        total += 120 + parts * 20                                                  # load + packing
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
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--slow-reps", type=int, default=5, help="reps for simd prefill cells")
    ap.add_argument("--no-mask-arms", action="store_true", help="do not time ours-mask@p (SKIP=0)")
    ap.add_argument("--no-real", action="store_true", help="skip the real FN partition arms")
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
    receipt = {"schema": "mlx2.amoeba-blocks-bench.v1", "mlx2_revision": git_rev(), "mlx": mx.__version__,
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
