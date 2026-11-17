"""Pack a box's weights out of the donor (child process): the rows its units own, so a box is portable.

For every layer with box units, through the carve host's unit sites (research/amoeba/carve/units.py, the
same structure discovery the carve uses; no model-name branches):
  dense site            unit n:      gate_proj[n, :], up_proj[n, :], down_proj[:, n]
  MoE site, neuron unit e*F + f:     switch gate_proj[e, f, :], up_proj[e, f, :], down_proj[e, :, f]
  MoE site, expert unit e:           the whole expert's gate/up/down
Quantized projections are dequantized (mx.dequantize with the module's own group size, bits and mode) and
stored as float16. A projection layout the packer does not know (a fused gate_up, missing projections)
refuses instead of writing a partial box. Writes OUT/box_weights.npz and OUT/box_weights.json; the
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
    arrays, layers = {}, {}
    for s in host.sites:
        units = np.nonzero(mask[s.layer, :s.n])[0]
        if not len(units) or s.kind is None:
            continue
        where = f"layer {s.layer} ({s.kind})"
        if s.kind == "dense":
            m = s.parent
            g = dense_weight(_proj(m, "gate_proj", where), units)
            u = dense_weight(_proj(m, "up_proj", where), units)
            dn = dense_weight(_proj(m, "down_proj", where))[:, units]
            layout = "dense: gate/up [k, d], down [d, k]"
        else:
            sw = s.module.switch_glu if s.kind == "gemma4_experts" else s.module.switch_mlp
            gp, up, dp = (_proj(sw, n, where) for n in ("gate_proj", "up_proj", "down_proj"))
            g, u, dn = [], [], []
            if s.units == "neuron":
                for e in sorted(set((units // s.F).tolist())):
                    f = units[units // s.F == e] - e * s.F
                    g.append(dense_weight(gp, f, expert=e))
                    u.append(dense_weight(up, f, expert=e))
                    dn.append(dense_weight(dp, expert=e)[:, f])
                layout = "MoE neuron units e*F+f: gate/up [k, d], down [d, k] (rows in unit-id order)"
            else:
                for e in units.tolist():
                    g.append(dense_weight(gp, expert=e))
                    u.append(dense_weight(up, expert=e))
                    dn.append(dense_weight(dp, expert=e))
                layout = "MoE expert units: gate/up [k*F, d], down [d, k*F] (experts in unit-id order)"
            g, u, dn = np.concatenate(g), np.concatenate(u), np.concatenate(dn, axis=1)
        arrays[f"L{s.layer}.gate"] = g.astype(np.float16)
        arrays[f"L{s.layer}.up"] = u.astype(np.float16)
        arrays[f"L{s.layer}.down"] = dn.astype(np.float16)
        arrays[f"L{s.layer}.unit_ids"] = units.astype(np.int32)
        layers[str(s.layer)] = {"kind": s.kind, "units": int(len(units)), "layout": layout}
    np.savez_compressed(out / "box_weights.npz", **arrays)
    rep = {"schema": "mlx2.amoeba-box-weights.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "mask": a.mask, "moe_units": a.moe_units, "dtype": "float16 (dequantized)", "layers": layers,
           "units": int(sum(v["units"] for v in layers.values())),
           "not_copied": "attention, norms, embeddings, the core: the donor checkpoint's (by fingerprint)"}
    (out / "box_weights.json").write_text(json.dumps(rep, indent=1))
    print(f"packed {rep['units']} units over {len(layers)} layers", flush=True)


if __name__ == "__main__":
    main()
