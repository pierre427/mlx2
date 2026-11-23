"""Route-wide aMoEba neuron masks: serve the core with one or more boxes gated off (experimental, unqualified).

An aMoEba carve is a partition of a model's MLP units (core, boxes, cross-box tiles, remainder). "Stripping" a
topic, for example a language, means serving every unit except that topic's box: nothing is pruned from the
checkpoint, the box is only switched off on this route. This module applies such a switch-off at load time.

Units are the carve's (``research/amoeba/carve/units.py``): a dense MLP layer's unit ``f`` is intermediate neuron
``f``; a routed-expert layer's unit ``e * F + f`` is neuron ``f`` of expert ``e`` (F the expert intermediate width).
A closed unit's activation is exactly zero, the carve's zero ablation: its rows of the gate and up projections are
replaced by exact zeros before the strict load, so ``act(gate) * up`` is ``act(0) * 0 = 0`` for any activation with
``act(0)`` finite, and the down projection receives an exact zero in that position. Every other row keeps its
stored bytes (``mx.where`` on the row mask), so every other neuron's activation is bitwise what the unmasked model
computes. Storage laws:

* unquantized: the weight row is set to 0;
* affine (``scales`` + ``biases``): the row's scales and biases are set to 0, so it dequantizes to ``0 * q + 0``;
* scale-only floating planes (``q2_0``, whose biases are derived as ``-scales`` later): the scales are set to 0;
* microscaling planes (integer scales, e.g. E8M0) have no exact zero row and are refused.

Because the change is in the weights, every kernel route the adapter runs (split or fused gate/up gathers, sorted
or unsorted, prefill, decode, verify) computes the masked model without a per-token cost, and nothing here names a
model: the adapter declares its sites (:class:`MaskSite`) and calls :meth:`NeuronMaskBundle.apply` on its sanitized
weights. Shared experts, attention and the router are never units, so they are never masked.

Bundle (a directory): ``manifest.json`` (schema :data:`SCHEMA`) naming the source checkpoint and its fingerprint,
the boxes switched off (taxonomy ids), the per-layer geometry and the mask file with its sha256; ``mask.npz`` with
one int64 array ``layer_<i>`` of closed unit ids per masked layer. A fingerprint, geometry or digest mismatch
refuses at load (ValueError). The bundle digest joins the adapter's APCv2 namespace (``numerics_tag``), so prefixes
cached by the unmasked model or under another mask are never restored here.

Speculative routes stay exact: every draft is verified by the masked target, so output equals masked ordinary
decode. A drafter trained on the unmasked model (an embedded MTP or DFly-style head) predicts the unmasked
distribution, so acceptance can drop on the masked topic; the adapter records that in its receipt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "mlx2.amoeba-neuron-mask.v1"
ARITHMETIC = "zero-gate-up-rows-v1"
MASK_FILE = "mask.npz"
CLOSED_VALUE = "zero"


@dataclass(frozen=True)
class MaskSite:
    """One maskable layer, declared by the adapter (never inferred here).

    ``stem``: the sanitized weight-name prefix of the MLP (``...mlp``) or routed-expert bank (``...mlp.experts``);
    ``experts``: the routed expert count, or None for a dense MLP; ``width``: neurons per expert (or the dense
    intermediate width); ``projections``: the projections whose output rows are a neuron's gate and up rows."""

    layer: int
    stem: str
    experts: int | None
    width: int
    projections: tuple = ("gate_proj", "up_proj")

    @property
    def kind(self) -> str:
        return "dense" if self.experts is None else "moe"

    @property
    def units(self) -> int:
        return self.width * (self.experts or 1)

    def geometry(self) -> dict:
        return {"layer": self.layer, "kind": self.kind, "experts": self.experts, "width": self.width,
                "units": self.units}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class NeuronMaskBundle:
    """A loaded, digest-checked mask bundle (see the module docstring)."""

    def __init__(self, path: Path, manifest: dict, closed: dict, digest: str):
        self.path, self.manifest, self.closed, self.digest = path, manifest, closed, digest
        self.applied = None

    @classmethod
    def load(cls, path) -> "NeuronMaskBundle":
        path = Path(path).expanduser()
        man_path = path / "manifest.json"
        if not man_path.is_file():
            raise ValueError(f"amoeba_mask: {path} has no manifest.json")
        manifest = json.loads(man_path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"amoeba_mask: schema {manifest.get('schema')!r} is not {SCHEMA}")
        if manifest.get("closed_value", CLOSED_VALUE) != CLOSED_VALUE:
            raise ValueError("amoeba_mask: only zero-closed units are served (a mean stand-in is not implemented)")
        mask = manifest.get("mask") or {}
        mask_path = path / mask.get("file", MASK_FILE)
        if not mask_path.is_file():
            raise ValueError(f"amoeba_mask: mask file {mask_path.name} is missing")
        if _sha256(mask_path) != mask.get("sha256"):
            raise ValueError("amoeba_mask: mask file digest does not match its manifest")
        closed = {}
        with np.load(mask_path) as z:
            for g in manifest.get("layers", ()):
                key = f"layer_{int(g['layer'])}"
                ids = np.asarray(z[key]) if key in z else np.zeros(0, np.int64)
                if ids.ndim != 1 or (ids.size and (ids.min() < 0 or ids.max() >= int(g["units"]))):
                    raise ValueError(f"amoeba_mask: {key} holds unit ids outside 0..{int(g['units']) - 1}")
                ids = np.unique(ids.astype(np.int64))
                if ids.size != int(g.get("closed", ids.size)):
                    raise ValueError(f"amoeba_mask: {key} closes {ids.size} units, the manifest says {g.get('closed')}")
                closed[int(g["layer"])] = ids
            extra = {k for k in z.files if k not in {f"layer_{int(g['layer'])}" for g in manifest.get("layers", ())}}
            if extra:
                raise ValueError(f"amoeba_mask: mask file has layers the manifest does not declare: {sorted(extra)}")
        digest = hashlib.sha256(man_path.read_bytes()).hexdigest()
        return cls(path, manifest, closed, digest)

    # -- checks ------------------------------------------------------------
    def check(self, fingerprint: str, sites) -> None:
        """Refuse (ValueError) unless the bundle was made for this checkpoint and geometry."""
        want = self.manifest.get("source_fingerprint")
        if want != fingerprint:
            raise ValueError(f"amoeba_mask: bundle is for checkpoint fingerprint {want}, this artifact is "
                             f"{fingerprint}; a mask never transfers between checkpoints")
        by_layer = {s.layer: s for s in sites}
        for g in self.manifest.get("layers", ()):
            site = by_layer.get(int(g["layer"]))
            if site is None:
                raise ValueError(f"amoeba_mask: layer {g['layer']} is not a maskable site of this adapter")
            have = site.geometry()
            for key in ("kind", "experts", "width", "units"):
                if g.get(key) != have[key]:
                    raise ValueError(f"amoeba_mask: layer {g['layer']} {key} {g.get(key)!r} != the model's "
                                     f"{have[key]!r}")

    # -- apply -------------------------------------------------------------
    def apply(self, weights: dict, sites, fingerprint: str) -> dict:
        """Zero the closed units' gate/up rows in ``weights`` (sanitized names, before quantize/strict load)."""
        import mlx.core as mx

        self.check(fingerprint, sites)
        by_layer = {s.layer: s for s in sites}
        layers = []
        for layer, ids in sorted(self.closed.items()):
            site = by_layer[layer]
            if ids.size == 0:
                layers.append({"layer": layer, "kind": site.kind, "closed": 0, "units": site.units})
                continue
            keep = np.ones(site.units, dtype=bool)
            keep[ids] = False
            shape = (site.experts, site.width) if site.experts is not None else (site.width,)
            keep_mx = mx.array(keep.reshape(shape))[..., None]
            planes = []
            for proj in site.projections:
                stem = f"{site.stem}.{proj}"
                w, s, b = (weights.get(f"{stem}.{p}") for p in ("weight", "scales", "biases"))
                if w is None:
                    raise ValueError(f"amoeba_mask: {stem}.weight is not in the checkpoint")
                if tuple(w.shape[:-1]) != shape:
                    raise ValueError(f"amoeba_mask: {stem}.weight rows {tuple(w.shape[:-1])} != {shape}")
                if s is None:
                    targets = ("weight",)
                elif not mx.issubdtype(s.dtype, mx.floating):
                    raise ValueError(f"amoeba_mask: {stem} has {s.dtype} scales (microscaling): no exact zero row")
                else:
                    targets = ("scales",) if b is None else ("scales", "biases")
                for part in targets:
                    key = f"{stem}.{part}"
                    a = weights[key]
                    weights[key] = mx.where(keep_mx, a, mx.zeros((), dtype=a.dtype))
                    planes.append(key)
            mx.eval(*(weights[k] for k in planes))
            layers.append({"layer": layer, "kind": site.kind, "closed": int(ids.size), "units": site.units,
                           "planes": planes})
        total = sum(s.units for s in sites)
        closed = sum(x["closed"] for x in layers)
        self.applied = {"layers": layers, "closed_units": closed, "model_units": total,
                        "closed_fraction": closed / total if total else 0.0}
        return self.applied

    # -- identity ----------------------------------------------------------
    def numerics_tag(self) -> str:
        return f"amoeba-mask-{self.digest[:16]}"

    def describe(self) -> dict:
        m = self.manifest
        return {"schema": SCHEMA, "arithmetic": ARITHMETIC, "path": str(self.path), "digest": self.digest,
                "checkpoint": m.get("checkpoint"), "source_fingerprint": m.get("source_fingerprint"),
                "boxes_off": list(m.get("boxes_off", ())), "source": m.get("source"),
                "closed_value": CLOSED_VALUE, **(self.applied or {})}


def write_bundle(out, *, checkpoint: str, fingerprint: str, sites, closed: dict, boxes_off, source: dict) -> Path:
    """Write a bundle: ``closed`` maps layer -> closed unit ids (carve unit order). Refuses a nonempty ``out``."""
    out = Path(out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"amoeba_mask: {out} is not empty (bundles are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    by_layer = {s.layer: s for s in sites}
    arrays, layers = {}, []
    for layer in sorted(closed):
        if layer not in by_layer:
            raise SystemExit(f"amoeba_mask: layer {layer} is not a maskable site")
        ids = np.unique(np.asarray(closed[layer], dtype=np.int64))
        site = by_layer[layer]
        if ids.size and (ids.min() < 0 or ids.max() >= site.units):
            raise SystemExit(f"amoeba_mask: layer {layer} unit ids outside 0..{site.units - 1}")
        arrays[f"layer_{layer}"] = ids
        layers.append({**site.geometry(), "closed": int(ids.size)})
    np.savez_compressed(out / MASK_FILE, **arrays)
    manifest = {"schema": SCHEMA, "arithmetic": ARITHMETIC, "closed_value": CLOSED_VALUE,
                "checkpoint": checkpoint, "source_fingerprint": fingerprint, "boxes_off": list(boxes_off),
                "source": source, "layers": layers,
                "closed_units": int(sum(x["closed"] for x in layers)),
                "model_units": int(sum(s.units for s in sites)),
                "mask": {"file": MASK_FILE, "sha256": _sha256(out / MASK_FILE)},
                "status": {"implemented": True, "qualified": False}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return out


def closed_from_unit_mask(mask: np.ndarray, sites, unit_block: int = 1) -> dict:
    """A carve mask ``[layers, max width]`` (nonzero = unit in the box) -> {layer: closed neuron ids}.

    ``unit_block`` > 1: the carve ran in block units (``--moe-units block64``): a routed-expert unit u is the block of
    neurons u*B .. u*B+B-1 (expert u // (F/B)); dense sites are neuron units in every mode. Refuses a set unit beyond
    a layer's width, or on a layer with no maskable site."""
    m = np.asarray(mask)
    by_layer = {s.layer: s for s in sites}
    closed = {}
    for layer in range(m.shape[0]):
        ids = np.flatnonzero(m[layer])
        if not ids.size:
            continue
        site = by_layer.get(layer)
        if site is None:
            raise SystemExit(f"amoeba_mask: the box holds units on layer {layer}, which has no maskable site")
        B = unit_block if site.experts is not None else 1
        if site.units % B:
            raise SystemExit(f"amoeba_mask: layer {layer}: {B}-neuron units do not tile its {site.units} neurons")
        if ids.max() >= site.units // B:
            raise SystemExit(f"amoeba_mask: layer {layer} box units reach {ids.max()}, width {site.units // B}")
        closed[layer] = (ids[:, None] * B + np.arange(B)[None]).ravel() if B > 1 else ids
    return closed


def main(argv=None):
    """``python -m mlx2.runtime.amoeba_neuron_mask export``: a carve's box mask -> a serving bundle (no weights)."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m mlx2.runtime.amoeba_neuron_mask")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="turn box masks of a carve into a mask bundle for --execution-policy amoeba_mask")
    p.add_argument("--model", required=True, help="the artifact the carve ran on (inspected, no weights loaded)")
    p.add_argument("--mask", action="append", required=True, metavar="NPZ:KEY",
                   help="a [layers, width] unit mask (partition.npz box key or box_mask.npz:box); repeatable, OR-ed")
    p.add_argument("--box", action="append", required=True, help="taxonomy id of a box switched off (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--unit-block", type=int, default=1,
                   help="neurons per carve unit at routed-expert sites (64 for --moe-units block64)")
    a = ap.parse_args(argv)
    from ..adapters.registry import inspect_model

    resolution = inspect_model(a.model)
    adapter = resolution.adapter_type
    if not hasattr(adapter, "amoeba_mask_sites_for"):
        raise SystemExit(f"{adapter.__name__} declares no amoeba mask sites")
    identity, sites = adapter.amoeba_mask_sites_for(a.model)
    union, sources = None, []
    for spec in a.mask:
        file, _, key = spec.rpartition(":")
        if not file:
            raise SystemExit(f"--mask {spec}: expected NPZ:KEY")
        with np.load(Path(file).expanduser()) as z:
            m = np.asarray(z[key]) > 0
        union = m if union is None else np.logical_or(union, m)
        sources.append({"file": str(Path(file).expanduser().resolve()), "key": key, "sha256": _sha256(Path(file))})
    closed = closed_from_unit_mask(union, sites, a.unit_block)
    out = write_bundle(a.out, checkpoint=f"{Path(identity['path']).name}@{identity['fingerprint']}",
                       fingerprint=identity["fingerprint"], sites=sites, closed=closed, boxes_off=a.box,
                       source={"masks": sources, "note": a.note, "unit_block": a.unit_block})
    print(json.dumps(json.loads((out / "manifest.json").read_text())["closed_units"]))


if __name__ == "__main__":
    main()
