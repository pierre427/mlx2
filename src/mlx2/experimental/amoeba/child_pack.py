"""Pack a box's weights out of the donor (child process): the rows its units own, so a box is portable.

For every layer with box units, through the carve host's unit sites (research/amoeba/carve/units.py, the
same structure discovery the carve uses; no model-name branches):
  dense site            unit n:      gate_proj[n, :], up_proj[n, :], down_proj[:, n]
  MoE site, neuron unit e*F + f:     switch gate_proj[e, f, :], up_proj[e, f, :], down_proj[e, :, f]
  MoE site, expert unit e:           the whole expert's gate/up/down
Affine-quantized projections are carried in their stored form (packed words, scales, biases), sliced at
output rows (gate/up) or at whole quantization groups of input columns (down, when a unit is a block of
neurons aligned with the group size): exact, and as small as the donor's own rows. Each slice is checked
against dequantize-then-slice before it is written. Any other projection (unquantized, or columns that do
not align, e.g. a dense site's single-neuron down columns) is dequantized and stored as float16. A projection
layout the packer does not know (a fused gate_up, missing projections) refuses instead of writing a partial
box. Layers are written one at a time (OUT/L<i>.npz; 2026-11-22: an 88K-block Hy4-flash box is ~100 GB as
float16, more than the host's memory headroom and the free disk) with the index OUT/box_weights.json; the
attention, norms and embeddings stay the donor's (referenced by fingerprint, never copied).

    python -m mlx2.experimental.amoeba.child_pack --model M --mask NPZ[:key] --out DIR [--cpu]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def dense_weight(lin, rows=None, expert=None):
    """float32 weight of a (Switch)Linear / Quantized(Switch)Linear, optionally one expert, optionally some rows."""
    import mlx.core as mx
    w = lin.weight
    q = "scales" in lin
    sc = lin.scales if q else None
    bi = lin.get("biases") if q and "biases" in lin else None
    if expert is not None:
        w = w[expert]
        sc = sc[expert] if q else None
        bi = bi[expert] if bi is not None else None
    if rows is not None:
        idx = mx.array(np.asarray(rows, np.int32))
        w = w[idx]
        sc = sc[idx] if q else None
        bi = bi[idx] if bi is not None else None
    if q:
        kw = {"mode": lin.mode} if getattr(lin, "mode", None) else {}
        w = mx.dequantize(w, sc, bi, group_size=lin.group_size, bits=lin.bits, **kw)
    return np.array(w.astype(mx.float32))


def _np(x):
    """numpy view of an mlx array; bfloat16 is carried as its uint16 bits."""
    import mlx.core as mx
    if x.dtype == mx.bfloat16:
        return np.array(x.view(mx.uint16)), "bfloat16"
    return np.array(x), str(x.dtype).split(".")[-1]


def quant_slice(lin, expert=None, rows=None, col_blocks=None, block=None):
    """Stored-form slice of an affine-quantized (Switch)Linear: rows (output dim) or column blocks (input dim,
    each ``block`` wide and group aligned). Returns {"w","s","b": numpy, "meta"} or None when not exact."""
    import mlx.core as mx
    if "scales" not in lin or getattr(lin, "mode", "affine") not in (None, "affine") or "biases" not in lin:
        return None
    gs, bits = int(lin.group_size), int(lin.bits)
    w, sc, bi = lin.weight, lin.scales, lin.biases
    if expert is not None:
        w, sc, bi = w[expert], sc[expert], bi[expert]
    if rows is not None:
        idx = mx.array(np.asarray(rows, np.int32))
        w, sc, bi = w[idx], sc[idx], bi[idx]
    if col_blocks is not None:
        if block % gs or (block * bits) % 32:
            return None
        wpb, spb = block * bits // 32, block // gs
        w = mx.concatenate([w[:, c * wpb:(c + 1) * wpb] for c in col_blocks], axis=-1)
        sc = mx.concatenate([sc[:, c * spb:(c + 1) * spb] for c in col_blocks], axis=-1)
        bi = mx.concatenate([bi[:, c * spb:(c + 1) * spb] for c in col_blocks], axis=-1)
    mx.eval(w, sc, bi)
    (wn, wd), (sn, sd), (bn, bd) = _np(w), _np(sc), _np(bi)
    return {"w": wn, "s": sn, "b": bn,
            "meta": {"format": "affine", "bits": bits, "group_size": gs, "scales_dtype": sd, "biases_dtype": bd}}


def _check_slice(q, lin, expert, rows=None, cols=None):
    """The stored-form slice dequantizes to exactly the dequantized donor's slice."""
    import mlx.core as mx
    m = q["meta"]
    dt = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
    to = lambda a, d: mx.array(a).view(mx.bfloat16) if d == "bfloat16" else mx.array(a).astype(dt[d])
    got = mx.dequantize(mx.array(q["w"]), to(q["s"], m["scales_dtype"]), to(q["b"], m["biases_dtype"]),
                        group_size=m["group_size"], bits=m["bits"])
    want = dense_weight(lin, rows, expert=expert)
    if cols is not None:
        want = want[:, cols]
    if not np.array_equal(np.array(got.astype(mx.float32)), want):
        raise SystemExit("pack: stored-form slice does not dequantize to the donor's rows; refusing")


def _proj(m, name, where):
    p = getattr(m, name, None)
    if p is None:
        raise SystemExit(f"pack: {where} has no {name} (layout {sorted(k for k in m.keys())}); refusing a partial box")
    if "bias" in p:
        raise SystemExit(f"pack: {where} {name} has a linear bias, which the packer does not carry; refusing a "
                         "partial box")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mask", required=True, help="NPZ[:key] [L, max width] mask of the box's units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-units", default="expert")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    a = ap.parse_args(argv)
    from research.amoeba.carve import ops as O
    O.footprint_watchdog(a.watchdog_gb, print)
    O.setup_mlx(a.memory_gb, 2)
    import mlx.core as mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    from research.amoeba.carve.host import Host
    host = Host(a.model, log=lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True), moe_units=a.moe_units)
    path, _, key = a.mask.partition(":")
    z = np.load(path)
    mask = z[key or z.files[0]] > 0
    if mask.shape != (host.L, max(host.widths)):
        raise SystemExit(f"pack: mask {mask.shape} != donor [L, max width] {(host.L, max(host.widths))}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers, total_bytes = {}, 0
    for s in host.sites:
        units = np.nonzero(mask[s.layer, :s.n])[0]
        if not len(units) or s.kind is None:
            continue
        import mlx.core as mx
        where = f"layer {s.layer} ({s.kind})"
        arrays, fmt = {}, {}

        def put(name, parts, lin, axis):
            """parts: list of (quant slice or None, float fallback thunk); one format per projection."""
            if all(q is not None for q, _ in parts):
                for k in ("w", "s", "b"):
                    arrays[f"{name}.{k}"] = np.concatenate([q[k] for q, _ in parts], axis=axis)
                fmt[name] = parts[0][0]["meta"]
            else:
                arrays[name] = np.concatenate([f() for _, f in parts], axis=axis).astype(np.float16)
                fmt[name] = {"format": "float16"}

        if s.kind == "dense":
            m = s.parent
            gl, ul, dl = (_proj(m, n, where) for n in ("gate_proj", "up_proj", "down_proj"))
            for name, lin in (("gate", gl), ("up", ul)):
                q = quant_slice(lin, rows=units)
                if q is not None:
                    _check_slice(q, lin, None, rows=units)
                put(name, [(q, lambda lin=lin: dense_weight(lin, units))], lin, 0)
            put("down", [(None, lambda: dense_weight(dl)[:, units])], dl, 1)
            layout = "dense: gate/up [k, d], down [d, k]"
        else:
            from research.amoeba.carve import units as U
            sw = U._switch(s.module, s.kind)
            gp, up, dp = (_proj(sw, n, where) for n in ("gate_proj", "up_proj", "down_proj"))
            B = getattr(s, "B", 1)
            g, u, dn = [], [], []
            if s.units == "neuron":
                for e in sorted(set((units // s.F).tolist())):
                    blocks = units[units // s.F == e] - e * s.F
                    f = (blocks[:, None] * B + np.arange(B)[None]).ravel() if B > 1 else blocks
                    for lst, lin in ((g, gp), (u, up)):
                        q = quant_slice(lin, expert=e, rows=f)
                        if q is not None and e == int(units[0] // s.F):
                            _check_slice(q, lin, e, rows=f)
                        lst.append((q, lambda lin=lin, e=e, f=f: dense_weight(lin, f, expert=e)))
                    q = quant_slice(dp, expert=e, col_blocks=blocks.tolist(), block=B) if B > 1 else None
                    if q is not None and e == int(units[0] // s.F):
                        _check_slice(q, dp, e, cols=f)
                    dn.append((q, lambda e=e, f=f: dense_weight(dp, expert=e)[:, f]))
                layout = "MoE neuron units e*F+f: gate/up [k, d], down [d, k] (rows in unit-id order)"
            else:
                for e in units.tolist():
                    for lst, lin in ((g, gp), (u, up), (dn, dp)):
                        q = quant_slice(lin, expert=e)
                        lst.append((q, lambda lin=lin, e=e: dense_weight(lin, expert=e)))
                layout = "MoE expert units: gate/up [k*F, d], down [d, k*F] (experts in unit-id order)"
            put("gate", g, gp, 0)
            put("up", u, up, 0)
            put("down", dn, dp, 1)
        arrays["unit_ids"] = units.astype(np.int32)
        f_out = out / f"L{s.layer}.npz"
        np.savez(f_out, **arrays)
        total_bytes += f_out.stat().st_size
        layers[str(s.layer)] = {"kind": s.kind, "units": int(len(units)), "layout": layout, "file": f_out.name,
                                "formats": fmt}
        del arrays
        mx.clear_cache()
        print(time.strftime("%H:%M:%S"), f"packed layer {s.layer}: {len(units)} units, "
              f"{f_out.stat().st_size / 2**30:.2f} GiB, formats {sorted({v['format'] for v in fmt.values()})}",
              flush=True)
    rep = {"schema": "mlx2.amoeba-box-weights.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "mask": a.mask, "moe_units": a.moe_units, "bytes": total_bytes,
           "dtype": "stored affine rows where exact (packed words + scales + biases), else float16 (dequantized); "
                    "per projection in layers[*].formats; bfloat16 scales/biases are stored as their uint16 bits",
           "layers": layers,
           "units": int(sum(v["units"] for v in layers.values())),
           "not_copied": "attention, norms, embeddings, the core: the donor checkpoint's (by fingerprint)"}
    (out / "box_weights.json").write_text(json.dumps(rep, indent=1))
    print(f"packed {rep['units']} units over {len(layers)} layers", flush=True)


if __name__ == "__main__":
    main()
