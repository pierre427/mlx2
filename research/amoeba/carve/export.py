"""Dense masked checkpoint of a carved core, for any donor, verified against the research forward.

A copy of the carve's source checkpoint in which every dropped unit contributes exactly zero:

  dense neuron i at layer l   the up projection's row i is zeroed (for affine-quantized tensors:
                              its scales and biases rows, so the dequantized row is exactly 0),
                              hence act_fn(gate) * up = 0 for that neuron (gated MLPs only: a
                              non-gated MLP is refused, act(0) need not be 0)
  expert e at layer l         the expert's down projection is zeroed, so its routed output is 0
                              and its slot is lost: the drop rule, exactly (no renormalisation)
  neuron (e, f) at layer l    (--moe-units neuron) row f of expert e's up projection is zeroed
                              (stacked ``switch_mlp.up_proj`` [E, F, D]; the up half, row F + f, of a
                              stacked [gate|up] ``gate_up_proj`` [E, 2F, D]; or per-expert
                              ``experts.{e}.up_proj`` / ``w3``), quantized ones through their scales and
                              biases rows, so h[e, f] = act(gate) * 0 = 0: the neuron drop rule, exactly,
                              for gated switch GLUs. The router is untouched; an expert with every
                              neuron dropped still takes its slot, as in the research forward.

A projection that carries its own ``.bias`` has that bias zeroed on the same unit axis as well as its weight
(or scales and biases): a dropped unit contributes exactly nothing (``_zero_keys``).

Only the files holding changed tensors are rewritten; every other file is hard-linked. Tensor keys
are found by suffix in the source's own index (HF names: ``layers.{l}.mlp.up_proj``,
``...experts.{e}.w2`` / ``down_proj``, stacked ``switch_mlp.down_proj``), outside any vision or
audio tower; an ambiguous or missing match refuses the export.

Verification runs in a separate process (one model resident at a time, the 27B rule):
``python -m research.amoeba.carve.export verify --run RUN`` loads the export through its adapter
and compares its logits with the masked research forward saved at export time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

TOWERS = ("vision", "visual", "audio", "mtp.")


def _index(src: Path):
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    files = sorted(src.glob("*.safetensors"))
    if len(files) != 1:
        raise SystemExit("export: no index and not exactly one safetensors file")
    import mlx.core as mx
    return {k: files[0].name for k in mx.load(str(files[0]))}


def _match(keys, pattern):
    rx = re.compile(pattern)
    return [k for k in keys if rx.search(k) and not any(t in k for t in TOWERS)]


def _zero_keys(ks):
    """The tensors to zero for one projection: an affine-quantized one through its scales and biases (the packed
    weight stays), an unquantized one through its weight; the Linear's own ``.bias`` in both cases (a dropped unit
    must contribute exactly nothing, so a bias left in place, or a weight left because a bias matched, is a leak)."""
    q = [k for k in ks if k.endswith((".scales", ".biases"))]
    return (q + [k for k in ks if k.endswith(".bias")]) if q else ks


def plan(host, core, src: Path):
    """[(key, axis, index_list)]: rows (axis 0) or experts (axis 0 of a stacked tensor) to zero,
    or whole per-expert tensors (index None)."""
    keys = list(_index(src))
    edits = []
    for s in host.sites:
        if s.kind is None:
            continue
        l, n = s.layer, s.n
        drop = np.where(np.asarray(core[l][:n]) == 0)[0]
        if not len(drop):
            continue
        if s.is_moe and s.units == "neuron":
            edits += _plan_neurons(keys, s, drop)
            continue
        if s.kind == "dense":
            if not hasattr(s.parent, "up_proj"):
                raise SystemExit(f"export: layer {l} MLP has no up_proj (not gated): zeroing is not exact")
            ks = _match(keys, rf"(^|\.)layers\.{l}\.[A-Za-z_]+\.up_proj\.(weight|scales|biases|bias)$")
            if not ks:
                ks = _match(keys, rf"(^|\.)layers\.{l}\.[A-Za-z_]+\.w3\.(weight|scales|biases|bias)$")
            ks = _zero_keys(ks)
            if not ks or len({k.rsplit(".", 1)[0] for k in ks}) != 1:
                raise SystemExit(f"export: layer {l}: up projection keys ambiguous or missing: {ks[:4]}")
            edits += [(k, "rows", drop.tolist()) for k in ks]
        else:
            st = _match(keys, rf"(^|\.)layers\.{l}\.([A-Za-z_]+\.)*(switch_mlp|switch_glu|experts)\.down_proj\.(weight|scales|biases|bias)$")
            st = _zero_keys(st)
            if st:
                if len({k.rsplit(".", 1)[0] for k in st}) != 1:
                    raise SystemExit(f"export: layer {l}: stacked expert down keys ambiguous: {st[:4]}")
                edits += [(k, "rows", drop.tolist()) for k in st]
                continue
            per = []
            for e in drop.tolist():
                ke = _match(keys, rf"(^|\.)layers\.{l}\.([A-Za-z_]+\.)*experts\.{e}\.(down_proj|w2)\.(weight|scales|biases|bias)$")
                ke = _zero_keys(ke)
                if not ke:
                    raise SystemExit(f"export: layer {l} expert {e}: no down projection key")
                per += [(k, "whole", None) for k in ke]
            edits += per
    return edits


def _plan_neurons(keys, s, drop):
    """Edits zeroing up rows (e, f) of a neuron-unit MoE site (see the module docstring)."""
    l, F = s.layer, s.F
    e_of, f_of = drop // F, drop % F
    sw = s.module.switch_glu if s.kind == "gemma4_experts" else s.module.switch_mlp
    base = rf"(^|\.)layers\.{l}\.([A-Za-z_]+\.)*(switch_mlp|switch_glu|experts)\."
    for proj, off in (("up_proj", 0), ("gate_up_proj", F)):
        st = _match(keys, base + rf"{proj}\.(weight|scales|biases|bias)$")
        st = _zero_keys(st)
        if not st:
            continue
        if len({k.rsplit(".", 1)[0] for k in st}) != 1:
            raise SystemExit(f"export: layer {l}: stacked expert {proj} keys ambiguous: {st[:4]}")
        if proj == "gate_up_proj" and not hasattr(sw, "gate_up_proj"):
            raise SystemExit(f"export: layer {l}: checkpoint stores a [gate|up] tensor but the module splits it; "
                             f"the up half's position is not checked, refusing")
        rows = 2 * F if proj == "gate_up_proj" else F
        Es = {s.E, int(sw.down_proj.weight.shape[0])}       # routed, or routed + a folded shared row
        return [(k, "rows2", (list(zip(e_of.tolist(), (f_of + off).tolist())), Es, rows)) for k in st]
    edits = []
    for e in sorted(set(e_of.tolist())):
        ke = _match(keys, rf"(^|\.)layers\.{l}\.([A-Za-z_]+\.)*experts\.{e}\.(up_proj|w3)\.(weight|scales|biases|bias)$")
        ke = _zero_keys(ke)
        if not ke:
            raise SystemExit(f"export: layer {l} expert {e}: no up projection key (neuron units)")
        edits += [(k, "rows", f_of[e_of == e].tolist()) for k in ke]
    return edits


def export_dense_core(host, src: Path, core, dst: Path, log=print, verify_ids=None, dense_layers=()):
    import mlx.core as mx
    from . import ops as O
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"export: {dst} exists (never overwrite an export)")
    t0 = time.time()
    edits = plan(host, core, src)
    wmap = _index(src)
    by_file = {}
    for k, how, idx in edits:
        by_file.setdefault(wmap[k], []).append((k, how, idx))
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in by_file:
            try:
                os.link(f, tmp / f.name) if f.suffix == ".safetensors" else shutil.copy2(f, tmp / f.name)
            except OSError:                      # another volume: copy
                shutil.copy2(f, tmp / f.name)
    for fname, eds in by_file.items():
        tensors = mx.load(str(src / fname))
        for k, how, idx in eds:
            t = tensors[k]
            if k.endswith(".scales") and t.dtype == mx.uint8:
                raise SystemExit(f"export: {k} holds E8M0 (MX) scales; zeroing them is not 0, refusing")
            if how == "whole":
                tensors[k] = mx.zeros_like(t)
            elif how == "rows2":                 # (expert, row) pairs of a stacked [E, rows, ...] tensor
                pairs, Es, rows = idx
                # a stacked projection's own bias is [E, rows]; every other tensor is [E, rows, ...]
                if t.ndim < (2 if k.endswith(".bias") else 3) or t.shape[0] not in Es or t.shape[1] != rows:
                    raise SystemExit(f"export: {k} has shape {tuple(t.shape)}, expected [E in {sorted(Es)}, {rows}, ...] "
                                     f"(experts, up rows); a transposed or differently stacked tensor is refused")
                keep = np.ones(t.shape[:2], np.float32)
                ee, rr = zip(*pairs)
                keep[list(ee), list(rr)] = 0.0
                shape = t.shape[:2] + (1,) * (t.ndim - 2)
                tensors[k] = (t * mx.array(keep).reshape(shape).astype(t.dtype)) if t.dtype != mx.uint32 else t
            else:
                keep = np.ones((t.shape[0],), np.float32)
                keep[idx] = 0.0
                shape = (t.shape[0],) + (1,) * (t.ndim - 1)
                tensors[k] = (t * mx.array(keep).reshape(shape).astype(t.dtype)) if t.dtype != mx.uint32 else t
        mx.save_safetensors(str(tmp / fname), tensors, metadata={"format": "mlx"})
        del tensors
        mx.clear_cache()
    ref = {}
    if verify_ids is not None:
        host.ctl.reset()
        host.ctl.core = [mx.array(np.asarray(core[l][:n], np.float32)) for l, n in enumerate(host.widths)]
        host.ctl.dense_layers = tuple(dense_layers)
        lg = host.logits_at(mx.array(verify_ids))
        host.ctl.reset()
        mx.eval(lg)
        np.savez(tmp / "carve_verify_reference.npz", ids=np.array(verify_ids, np.int32), logits=np.array(lg))
    prov = {"schema": "mlx2.amoeba-carve-export.v1", "source": str(src), "source_checkpoint": host.checkpoint,
            "source_fingerprint": host.fingerprint, "mlx2_source_revision": O.git_rev(),
            "kept_frac": float(np.mean(np.concatenate([np.asarray(core[l][:n]) for l, n in enumerate(host.widths)]))),
            "edits": len(edits), "rewritten_files": sorted(by_file),
            "zeroing": "dense: up rows; experts: down" + ("; MoE neurons (e, f): expert e's up row f"
                                                         if any(s.units == "neuron" and s.is_moe for s in host.sites) else "")
                       + "; each through the weight (quantized: scales and biases) and the projection's own bias",
            "edited_keys": sorted(k for k, _, _ in edits),
            "status": "research artifact, unqualified; verify with python -m research.amoeba.carve.export verify",
            "s": round(time.time() - t0, 1)}
    O.write_json(tmp / "amoeba_carve_provenance.json", prov)
    tmp.rename(dst)
    log(f"export: {dst} ({len(edits)} tensor edits in {len(by_file)} files, {prov['s']} s)")
    return {**prov, "dir": str(dst)}


def verify(export_dir: Path, log=print, tol_top1=0.99):
    import mlx.core as mx
    from .host import Host
    ref = np.load(export_dir / "carve_verify_reference.npz")
    h = Host(export_dir, log=log)
    h.check_paths(mx.array(ref["ids"]))
    lg = np.array(h.logits_at(mx.array(ref["ids"])))
    r = {"max_abs": float(np.abs(lg - ref["logits"]).max()),
         "top1_agree": float((lg.argmax(-1) == ref["logits"].argmax(-1)).mean()),
         "export_checkpoint": h.checkpoint}
    r["ok"] = r["top1_agree"] >= tol_top1
    log(f"export verify: {json.dumps(r)}")
    return r


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("verify")
    v.add_argument("--run", required=True, help="carve run dir (its carve.json names the export)")
    a = ap.parse_args()
    run = Path(a.run)
    rep = json.loads((run / "carve.json").read_text())
    r = verify(Path(rep["export"]["dir"]))
    rep["export"]["verify"] = r
    from . import ops as O
    O.write_json(run / "carve.json", rep)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
