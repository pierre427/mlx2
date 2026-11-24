"""Hy4-flash baked aMoEba box: routed experts stored in a few width buckets (experimental, unqualified).

A carved box (``amoeba_neuron_mask`` bundle) closes neuron blocks of routed experts unevenly: in every layer some
experts keep all their blocks, so a stock bank (one width per layer) cannot shrink. The bake groups each layer's
experts into width buckets (``classes``, in blocks of ``block`` neurons, e.g. 24/22/20 of 64): an expert with ``r``
open blocks goes to the narrowest class ``c >= r`` and keeps its ``r`` open blocks plus its first ``c - r`` closed
blocks (zeroed), in block order. Each bucket is a stock ``SwitchGLU`` bank of its width with one extra all-zero
expert at the end; ``bucket_table [E, 2]`` maps a global expert to ``(bucket, slot)``.

Zero law: a closed neuron is the zero mask's (its gate/up scales and biases are 0, so its activation is exactly 0
and its down column contributes exactly 0); a dropped block is such a neuron removed. Dense sites (layer 0) keep
the zero rows in place. On CPU (sequential float32 reduction) the baked model equals the zero-mask model bit for
bit; on Metal the down reduction runs over fewer (zero) columns, so it equals it to float32 reassociation.

Call: small calls (decode, verify) run every bucket on all routed slots, each slot not in the bucket remapped to
the bucket's zero expert (exact zeros; no host sync); large calls (prefill chunks, >= ``SLICE_MIN_ASSIGNMENTS``)
sync once per layer and run each bucket on its own slots only.

Storage: the bake copies every unchanged tensor byte for byte (raw safetensors, no requantization); bucket banks
are row/column block slices of the stock quantized planes (a block is a whole quantization group along the down
projection's input axis: ``block`` must be a multiple of its group size).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

SCHEMA = "mlx2.hy4-expert-buckets.v1"
LAW = "bucketed-zero-mask-v1"
SLICE_MIN_ASSIGNMENTS = 4096
# Small-call path (decode, DFly2 verify): "remap" (default; every bucket bank on every routed slot, others on the
# bucket's zero expert), "sliced" (one host sync per layer, each bank on its own slots, at every size) or "fanout"
# (after the load the bucket banks become one bank of 64-neuron blocks per projection; every routed slot expands to
# its expert's blocks, padded with a zero block, in one gather per projection, no host sync; the down partials are
# summed per block in float32, so it equals the zero mask to float32 reassociation, not bitwise; own layout tag).
# fanout is not helpful as built (2026-11-23 M5 single-layer microbench, Hy4 cut-0.1 bake d1fea001, layers 1 and 21,
# top-8): 0.88-4.6 ms vs 0.44-1.22 ms sliced for T=1-8 and 166-273 ms vs 7-24 ms at T=512 (64-neuron block gathers
# run far below the stock kernels' efficiency); kept as an explicit research option, never a default.
SMALL_PATHS = ("remap", "sliced", "fanout")
SMALL_PATH = os.environ.get("MLX2_HY4_BUCKET_SMALL_PATH", "remap")
FANOUT_MAX_ROWS = 16384          # block rows per fan-out chunk (bounds the float32 partials)
PROJ = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")


# ---------------------------------------------------------------------------------------------------------- layout
def assign_buckets(closed_blocks: np.ndarray, classes) -> dict:
    """``closed_blocks [E, nb]`` (a block whose every neuron is closed) -> the layer's bucket layout.

    Returns ``{"bucket": [E], "slot": [E], "keep": {expert: kept block ids}, "counts": [n_b], "widths": [c_b]}``."""
    E, nb = closed_blocks.shape
    classes = sorted(int(c) for c in classes)
    if not classes or classes[-1] != nb or classes[0] < 1:
        raise ValueError(f"bucket classes {classes} must end at the full width {nb} blocks")
    open_n = nb - closed_blocks.sum(1)
    bucket = np.searchsorted(np.array(classes), open_n, side="left").astype(np.int32)
    # widest first in storage order: bucket 0 is the full-width class
    order = list(range(len(classes)))[::-1]
    remap = {old: new for new, old in enumerate(order)}
    bucket = np.array([remap[int(b)] for b in bucket], dtype=np.int32)
    widths = [classes[o] for o in order]
    slot = np.zeros(E, dtype=np.int32)
    counts = [0] * len(widths)
    keep = {}
    for e in range(E):
        b = int(bucket[e])
        slot[e] = counts[b]
        counts[b] += 1
        opened = np.nonzero(~closed_blocks[e])[0]
        pad = np.nonzero(closed_blocks[e])[0][: widths[b] - len(opened)]
        keep[e] = np.sort(np.concatenate([opened, pad])).astype(np.int64)
        assert len(keep[e]) == widths[b]
    return {"bucket": bucket, "slot": slot, "keep": keep, "counts": counts, "widths": widths}


def layout_digest(layers: dict, *, block: int, mask_digest: str) -> str:
    """Bucket layout + block + the mask bundle's digest (single-neuron closures inside kept blocks included)."""
    h = hashlib.sha256(f"{SCHEMA}:{block}:{mask_digest}".encode())
    for layer in sorted(layers):
        lay = layers[layer]
        h.update(f"{layer}:{lay['widths']}:{lay['counts']}".encode())
        h.update(np.ascontiguousarray(lay["bucket"]).tobytes())
        h.update(np.ascontiguousarray(lay["slot"]).tobytes())
        for e in sorted(lay["keep"]):
            h.update(np.ascontiguousarray(lay["keep"][e]).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- the served bank
def _mlx():
    import mlx.core as mx
    import mlx.nn as nn

    from .switch_layers import SwitchGLU
    return mx, nn, SwitchGLU


def bucketed_switch_glu(input_dims: int, widths, counts, activation, num_experts: int):
    """A ``BucketedSwitchGLU`` (built lazily so this module imports without MLX)."""
    mx, nn, SwitchGLU = _mlx()

    class BucketedSwitchGLU(nn.Module):
        law = LAW

        def __init__(self):
            super().__init__()
            self.buckets = [SwitchGLU(input_dims, int(w), int(n) + 1, activation=activation)
                            for w, n in zip(widths, counts)]
            self.bucket_table = mx.zeros((num_experts, 2), dtype=mx.int32)
            object.__setattr__(self, "_widths", [int(w) for w in widths])
            object.__setattr__(self, "_counts", [int(n) for n in counts])
            object.__setattr__(self, "_calls", {"remap": 0, "sliced": 0, "fanout": 0})
            object.__setattr__(self, "_fanout", None)

        def __call__(self, x, indices):
            if self._fanout is not None:
                return self._fan(x, indices)
            tab = self.bucket_table
            bk = tab[:, 0][indices]
            sl = tab[:, 1][indices]
            if int(indices.size) >= (1 if SMALL_PATH == "sliced" else SLICE_MIN_ASSIGNMENTS):
                try:
                    mx.eval(bk, sl)                 # the one host sync of the sliced path
                except ValueError:                  # inside a graph transformation: no host sync, remap path
                    pass
                else:
                    return self._sliced(x, indices, bk, sl)
            self._calls["remap"] += 1
            y = None
            for i, bank in enumerate(self.buckets):
                idx = mx.where(bk == i, sl, mx.array(self._counts[i], dtype=sl.dtype))   # zero expert otherwise
                yi = bank(x, idx)
                y = yi if y is None else y + yi
            return y

        def _sliced(self, x, indices, bk, sl):
            b = np.array(bk).reshape(-1)
            s = np.array(sl).reshape(-1)
            k = int(indices.shape[-1])
            lead = tuple(indices.shape[:-1])
            D = int(x.shape[-1])
            xf = x.reshape(-1, D)
            tok = np.repeat(np.arange(xf.shape[0]), k)
            out = mx.zeros((b.size, self.buckets[0].down_proj.output_dims), dtype=x.dtype)
            self._calls["sliced"] += 1
            for i, bank in enumerate(self.buckets):
                pos = np.nonzero(b == i)[0]
                if pos.size == 0:
                    continue
                xb = xf[mx.array(tok[pos])]
                yb = bank(xb, mx.array(s[pos].astype(np.int32))[:, None])        # [n, 1, D]
                out[mx.array(pos)] = yb.reshape(pos.size, -1).astype(out.dtype)
            return out.reshape(*lead, k, -1)

        # -- fan-out: one bank of blocks per projection ------------------------------------------------------
        def to_fanout(self, block: int) -> dict:
            """Replace the bucket banks by block banks (same bytes; the bucket planes are released)."""
            from .switch_layers import _gather_sort, _scatter_unsort  # noqa: F401  (import check)

            def _with(mod, arrays):
                new = type(mod).__new__(type(mod))
                dict.update(new, mod)
                new.__dict__.update(mod.__dict__)
                for k, v in arrays.items():
                    new[k] = v
                return new

            def parts(mod):
                return [k for k in ("weight", "scales", "biases") if k in mod and mod[k] is not None]

            banks = list(self.buckets)
            nbs = [w // block for w in self._widths]
            nbmax = max(nbs)
            out = {}
            for proj in ("gate_proj", "up_proj"):
                ref = banks[0][proj]
                arrays = {}
                for k in parts(ref):
                    arrays[k] = mx.concatenate([b[proj][k].reshape(-1, block, b[proj][k].shape[-1]) for b in banks], 0)
                out[proj] = _with(ref, arrays)
            ref = banks[0]["down_proj"]
            arrays = {}
            for k in parts(ref):
                segs = []
                for b, nb in zip(banks, nbs):
                    a = b["down_proj"][k]
                    p = a.shape[-1] // nb
                    segs.append(a.reshape(a.shape[0], a.shape[1], nb, p).transpose(0, 2, 1, 3)
                                .reshape(a.shape[0] * nb, a.shape[1], p))
                arrays[k] = mx.concatenate(segs, 0)
            out["down_proj"] = _with(ref, arrays)
            # block table: expert -> its blocks in order, padded with bucket 0's zero expert's first block
            offs = np.cumsum([0] + [(n + 1) * nb for n, nb in zip(self._counts, nbs)])[:-1]
            zero = int(offs[0] + self._counts[0] * nbs[0])
            tab = np.array(self.bucket_table)
            blocks = np.full((tab.shape[0], nbmax), zero, dtype=np.int32)
            for e, (b, slot) in enumerate(tab):
                start = int(offs[b] + slot * nbs[b])
                blocks[e, : nbs[b]] = np.arange(start, start + nbs[b])
            mx.eval(*[m[k] for m in out.values() for k in parts(m)])
            object.__setattr__(self, "_fanout", {"gate_proj": out["gate_proj"], "up_proj": out["up_proj"],
                                                 "down_proj": out["down_proj"], "table": mx.array(blocks),
                                                 "nbmax": nbmax, "activation": banks[0].activation})
            self.buckets = []                                # release the bucket planes
            return {"blocks": int(out["gate_proj"].num_experts), "nbmax": nbmax}

        def _fan(self, x, indices):
            from .switch_layers import _gather_sort, _scatter_unsort
            f = self._fanout
            self._calls["fanout"] += 1
            k = int(indices.shape[-1])
            lead = tuple(indices.shape[:-1])
            D = int(x.shape[-1])
            xf = x.reshape(-1, D)
            idf = indices.reshape(-1, k)
            n = int(xf.shape[0])
            step = max(1, FANOUT_MAX_ROWS // (k * f["nbmax"]))
            outs = []
            for a in range(0, n, step):
                ids = f["table"][idf[a:a + step]].reshape(-1, k * f["nbmax"])           # [t, k*nb]
                xe = mx.expand_dims(xf[a:a + step], (-2, -3))
                xs, idx, inv = _gather_sort(xe, ids)
                h = f["activation"](f["up_proj"](xs, idx, sorted_indices=True),
                                    f["gate_proj"](xs, idx, sorted_indices=True))
                part = f["down_proj"](h.astype(mx.float32), idx, sorted_indices=True)
                part = _scatter_unsort(part, inv, ids.shape).squeeze(-2)               # [t, k*nb, L]
                t = int(ids.shape[0])
                outs.append(part.reshape(t, k, f["nbmax"], -1).sum(axis=2).astype(x.dtype))
            y = outs[0] if len(outs) == 1 else mx.concatenate(outs, 0)
            return y.reshape(*lead, k, -1)

        def describe(self) -> dict:
            return {"schema": SCHEMA, "law": LAW, "widths_neurons": self._widths, "experts": self._counts,
                    "small_path": "fanout" if self._fanout is not None else SMALL_PATH, "calls": dict(self._calls)}

    return BucketedSwitchGLU()


def install_layer(layer, index: int, spec: dict) -> bool:
    """Swap one decoder layer's routed bank for its ``BucketedSwitchGLU`` (before quantize and the strict load);
    False for a dense layer."""
    from .hy_v4_flash import MoE

    if spec.get("schema") != SCHEMA:
        raise ValueError(f"Hy4-flash expert buckets: unknown schema {spec.get('schema')!r}")
    if not isinstance(layer.mlp, MoE):
        return False
    lay = spec["layers"].get(str(index))
    if lay is None:
        raise ValueError(f"Hy4-flash expert buckets: layer {index} has no bucket layout")
    old = layer.mlp.experts
    E = int(old.gate_proj.num_experts)
    if sum(lay["counts"]) != E:
        raise ValueError(f"Hy4-flash expert buckets: layer {index} buckets hold {sum(lay['counts'])} of {E} experts")
    block = int(spec["block"])
    layer.mlp.experts = bucketed_switch_glu(int(old.gate_proj.input_dims), [int(c) * block for c in lay["widths"]],
                                            lay["counts"], old.activation, E)
    return True


def prepare(model, spec: dict, path: str | None = None) -> dict:
    """After the strict load: switch the small-call path (``SMALL_PATH`` unless ``path``)."""
    path = path or SMALL_PATH
    if path not in SMALL_PATHS:
        raise ValueError(f"MLX2_HY4_BUCKET_SMALL_PATH must be one of {SMALL_PATHS}, got {path!r}")
    if path != "fanout":
        return {"small_path": path}
    import mlx.core as mx
    blocks = 0
    for layer in model.model.layers:
        bank = getattr(getattr(layer, "mlp", None), "experts", None)
        if getattr(bank, "law", None) == LAW:
            blocks += bank.to_fanout(int(spec["block"]))["blocks"]
            mx.clear_cache()
    return {"small_path": "fanout", "blocks": blocks}


def install(model, spec: dict) -> dict:
    """Swap every sparse layer's ``mlp.experts`` for its ``BucketedSwitchGLU`` (before quantize and the strict
    load). ``spec`` is the artifact's ``mlx2_expert_buckets`` config block."""
    done = [i for i, layer in enumerate(model.model.layers) if install_layer(layer, i, spec)]
    return {"schema": SCHEMA, "law": LAW, "layers": len(done), "block": int(spec["block"]),
            "classes": spec.get("classes"), "layout_digest": spec.get("layout_digest")}


# ----------------------------------------------------------------------------------------- raw safetensors bake
_DT = {"U32": np.uint32, "F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I32": np.int32, "U8": np.uint8,
       "U16": np.uint16, "I64": np.int64}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, name, header=None):
    header, base = header or read_header(path)
    m = header[name]
    a, b = m["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        buf = f.read(b - a)
    return np.frombuffer(buf, dtype=_DT[m["dtype"]]).reshape(m["shape"]), m["dtype"]


def write_safetensors(path, tensors, metadata=None, *, copy_from=None):
    """``tensors``: ordered list of ``(name, dtype_str, array | None)``; ``None`` copies that tensor's bytes from
    ``copy_from`` (a source shard path) unchanged. Writes to ``path`` atomically (``.part`` then rename)."""
    src_header, src_base = read_header(copy_from) if copy_from else ({}, 0)
    entries, off = {}, 0
    for name, dt, arr in tensors:
        if arr is None:
            m = src_header[name]
            n = m["data_offsets"][1] - m["data_offsets"][0]
            shape, dt = m["shape"], m["dtype"]
        else:
            arr = np.ascontiguousarray(arr)
            n, shape = arr.nbytes, list(arr.shape)
        entries[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    head = dict(entries)
    if metadata:
        head["__metadata__"] = metadata
    hb = json.dumps(head, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = Path(str(path) + ".part")
    with open(tmp, "wb") as out, (open(copy_from, "rb") if copy_from else open(os.devnull, "rb")) as src:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for name, dt, arr in tensors:
            if arr is None:
                a, b = src_header[name]["data_offsets"]
                src.seek(src_base + a)
                left = b - a
                while left:
                    chunk = src.read(min(left, 64 << 20))
                    out.write(chunk)
                    left -= len(chunk)
            else:
                out.write(np.ascontiguousarray(arr).tobytes())
    os.replace(tmp, path)


def _bucket_planes(planes: dict, lay: dict, block: int, group: int) -> dict:
    """Stock planes ``{proj.part: array}`` -> bucket planes ``{bucket: {proj.part: array}}`` (+ a zero expert)."""
    out = {}
    E = planes["gate_proj.weight"].shape[0]
    for b, (w, n) in enumerate(zip(lay["widths"], lay["counts"])):
        experts = np.array(sorted(e for e in range(E) if lay["bucket"][e] == b), dtype=np.int64)
        assert [int(lay["slot"][e]) for e in experts] == list(range(n))
        bank = {}
        for key, a in planes.items():
            proj, part = key.split(".")
            rows = proj != "down_proj"                       # gate/up: neuron rows (axis 1); down: columns (axis 2)
            axis_len = a.shape[1] if rows else a.shape[2]
            F = planes["gate_proj.weight"].shape[1]
            per = axis_len * block // F                       # storage elements per block along that axis
            if per * F != axis_len * block or per < 1:
                raise ValueError(f"block {block} does not split {key} [{axis_len} per {F} neurons] evenly "
                                 f"(group {group})")
            sel = np.stack([np.concatenate([np.arange(k * per, (k + 1) * per) for k in lay["keep"][e]])
                            for e in experts]) if n else np.zeros((0, w * per), dtype=np.int64)
            if rows:
                picked = np.take_along_axis(a[experts], sel[:, :, None], axis=1) if n else a[:0, : w * per]
            else:
                picked = np.take_along_axis(a[experts], sel[:, None, :], axis=2) if n else a[:0, :, : w * per]
            zero = np.zeros((1, *picked.shape[1:]), dtype=a.dtype)
            bank[key] = np.concatenate([picked, zero], axis=0)
        out[b] = bank
    return out


def _zero_rows(a: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Zero the closed neuron rows (zero-mask law on scales/biases, or weight codes when unscaled)."""
    b = np.array(a, copy=True)
    b[~keep.reshape(b.shape[:-1])] = 0
    return b


def bake(src, out, bundle, *, classes, block=64, watchdog=None, log=print) -> dict:
    """Bake ``bundle`` (an ``amoeba_neuron_mask`` bundle of ``src``) into a new artifact folder ``out``."""
    from ...adapters import hy_v4_flash as A

    src, out = Path(src), Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: bakes are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    if out.stat().st_dev != src.stat().st_dev:
        raise SystemExit(f"{out} is on another volume than {src}: unchanged files are hard links, keep them together")
    artifact = A.inspect_artifact(src)
    from .hy_v4_flash import ModelArgs
    args = ModelArgs.from_hf_config(artifact["config"])
    sites = A.amoeba_mask_sites(args)
    bundle.check(artifact["identity"]["fingerprint"], sites)
    by = {s.layer: s for s in sites}
    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    quant = config.get("quantization") or {}
    group = int(quant.get("group_size", 64))
    t0 = time.time()
    layers_meta, new_wm, written = {}, {}, set()
    # 1. routed expert layers
    for layer in sorted(l for l, s in by.items() if s.experts is not None):
        if watchdog:
            watchdog(f"layer {layer}")
        s = by[layer]
        E, F = s.experts, s.width
        if F % block:
            raise ValueError(f"block {block} does not divide F {F}")
        nb = F // block
        ids = np.asarray(bundle.closed.get(layer, np.zeros(0, dtype=np.int64)))
        closed = np.zeros(E * F, dtype=bool)
        closed[ids] = True
        closed = closed.reshape(E, F)
        closed_blocks = closed.reshape(E, nb, block).all(-1)
        lay = assign_buckets(closed_blocks, classes)
        stem = f"model.layers.{layer}.mlp.switch_mlp"
        names = {f"{stem}.{p}.{q}" for p in PROJ for q in PARTS} & set(wm)
        shards = {wm[n] for n in names}
        if len(shards) != 1:
            raise ValueError(f"layer {layer}: expert planes span {sorted(shards)}")
        shard = src / shards.pop()
        others = [k for k, v in wm.items() if v == shard.name and ".mlp.switch_mlp." not in k]   # expert-only shard
        if others:
            raise ValueError(f"layer {layer}: {shard.name} holds non-expert tensors {others[:3]}")
        header = read_header(shard)
        planes, dtypes = {}, {}
        for n in sorted(names):
            a, dt = read_tensor(shard, n, header)
            key = n[len(stem) + 1:]
            planes[key], dtypes[key] = a, dt
        keep = ~closed                                     # partially closed blocks: zero-mask rows
        for p in ("gate_proj", "up_proj"):
            targets = [q for q in ("scales", "biases") if f"{p}.{q}" in planes] or ["weight"]
            for q in targets:
                planes[f"{p}.{q}"] = _zero_rows(planes[f"{p}.{q}"], keep)
        banks = _bucket_planes(planes, lay, block, group)
        tensors = []
        for b, bank in banks.items():
            for key in sorted(bank):
                tensors.append((f"{stem}.buckets.{b}.{key}", dtypes[key], bank[key]))
        table = np.stack([lay["bucket"], lay["slot"]], axis=1).astype(np.int32)
        tensors.append((f"{stem}.bucket_table", "I32", table))
        name = f"model-experts-L{layer}-buckets.safetensors"
        write_safetensors(out / name, tensors, metadata={"format": "mlx"})
        for t, _, _ in tensors:
            new_wm[t] = name
        written.add(name)
        before = sum(a.nbytes for a in planes.values())
        after = sum(a.nbytes for bank in banks.values() for a in bank.values())
        layers_meta[str(layer)] = {"widths": lay["widths"], "counts": lay["counts"], "bytes_before": int(before),
                                   "bytes_after": int(after), "closed_blocks": int(closed_blocks.sum()),
                                   "_lay": lay}
        log(f"bake layer {layer}: buckets {lay['widths']} x {lay['counts']} experts, "
            f"{before / 2**20:.0f} -> {after / 2**20:.0f} MiB ({time.time() - t0:.0f} s)")
    # 2. dense sites: zero rows in place (shard rewritten byte for byte otherwise)
    dense = [s for s in sites if s.experts is None and len(bundle.closed.get(s.layer, ()))]
    rewrite = {}
    for s in dense:
        keep = np.ones(s.units, dtype=bool)
        keep[np.asarray(bundle.closed[s.layer])] = False
        for p in s.projections:
            q = [x for x in ("scales", "biases") if f"{s.stem}.{p}.{x}" in wm] or ["weight"]
            for part in q:
                rewrite[f"{s.stem}.{p}.{part}"] = keep
    for shard_name in sorted({wm[n] for n in rewrite}):
        if watchdog:
            watchdog(f"dense shard {shard_name}")
        shard = src / shard_name
        header, _ = read_header(shard)
        tensors = []
        for n in [k for k in header if k != "__metadata__"]:
            if n in rewrite:
                a, dt = read_tensor(shard, n, (header, _))
                tensors.append((n, dt, _zero_rows(a, rewrite[n])))
            else:
                tensors.append((n, header[n]["dtype"], None))
        write_safetensors(out / shard_name, tensors, metadata=header.get("__metadata__"), copy_from=shard)
        written.add(shard_name)
        log(f"bake dense zero rows: rewrote {shard_name}")
    # 3. everything else: hard links (read-only, never modified)
    expert_shards = {wm[n] for n in wm if ".mlp.switch_mlp." in n}
    linked = []
    for f in sorted(src.iterdir()):
        if f.name.startswith(".") or f.name in written or f.name in expert_shards or f.name in (
                "config.json", "model.safetensors.index.json"):
            continue
        if f.is_dir():
            os.symlink(f.resolve(), out / f.name)          # e.g. assets/: read-only reference
        else:
            os.link(f, out / f.name)
        linked.append(f.name)
    for k, v in wm.items():
        if ".mlp.switch_mlp." not in k:
            new_wm[k] = v
    digest = layout_digest({int(k): v["_lay"] for k, v in layers_meta.items()}, block=block, mask_digest=bundle.digest)
    spec = {"schema": SCHEMA, "law": LAW, "block": block, "classes": sorted(int(c) for c in classes),
            "layout_digest": digest,
            "layers": {k: {kk: vv for kk, vv in v.items() if kk != "_lay"} for k, v in layers_meta.items()},
            "source": {"checkpoint": f"{src.name}@{artifact['identity']['fingerprint']}",
                       "mask_digest": bundle.digest, "mask_numerics_tag": bundle.numerics_tag(),
                       "mask_path": str(bundle.path), "boxes_off": list(bundle.manifest.get("boxes_off", ()))}}
    config["mlx2_expert_buckets"] = spec
    q = config.get("quantization")
    for qk in ("quantization", "quantization_config"):
        qd = config.get(qk)
        if not isinstance(qd, dict):
            continue
        for k in [k for k in qd if ".mlp.switch_mlp." in k and isinstance(qd[k], dict)]:
            entry = qd.pop(k)
            stem, _, proj = k.rpartition(".")
            layer = stem.split(".")[2]
            for b in range(len(layers_meta[layer]["widths"])):
                qd[f"{stem}.buckets.{b}.{proj}"] = entry
    del q
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "model.safetensors.index.json").write_text(json.dumps({**{k: v for k, v in index.items()
                                                                     if k != "weight_map"},
                                                                  "weight_map": dict(sorted(new_wm.items()))}))
    before = sum(v["bytes_before"] for v in layers_meta.values())
    after = sum(v["bytes_after"] for v in layers_meta.values())
    rec = {"schema": SCHEMA, "source": spec["source"], "classes": spec["classes"], "block": block,
           "layout_digest": digest, "expert_bytes_before": int(before), "expert_bytes_after": int(after),
           "expert_bytes_saved": int(before - after), "linked": linked, "written": sorted(written),
           "seconds": round(time.time() - t0, 1)}
    (out / "mlx2_bake_receipt.json").write_text(json.dumps(rec, indent=1))
    return rec
