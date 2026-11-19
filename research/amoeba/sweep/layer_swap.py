"""Whole-layer weight streaming for the rolling map capture (sweep family interface v1, README.md).

mlx2's loaders stream whole decoder layers only for dense models (``dense_layers`` refuses routed experts) and stream
MoE experts only as a decode-sized per-expert cache (``moe_experts``), which a capture batch would re-read layer by
layer. This module streams a WHOLE layer, every routed expert included, once per chunk, for a model that does not
fit resident, and leaves every number to the model's own code:

- the model is the adapter's own module tree, built by a per-artifact ``spec`` the way the adapter builds it (same
  classes, same quantization and module installers, same policy, same process globals), but with no layer weights:
  each decoder layer holds shape-true placeholders (a broadcast scalar, which allocates nothing) and the rest of the
  model (embeddings, final norm, head) is loaded resident;
- ``begin_layer(l)`` reads layer l's checkpoint tensors (every expert), runs them through the adapter's own
  ``sanitize`` (and the spec's per-layer installers, e.g. the NVFP4 expert-scale fold), checks that they are exactly
  the layer's parameters (names, shapes, dtypes; strict, as the adapter's ``load_weights(strict=True)``) and binds
  them; ``release_layer(l)`` puts the placeholders back and drops every array cache a module derived from them;
- scheduling, call arguments (recomputed from the token ids by the model's own trunk), unit payloads (the carve
  host's unit sites, so the units are the resident capture's: dense ``act``, MoE neurons or experts) and the output
  path are the host family's (host_family.py);
- ``reference_logits`` is the ordinary text-major forward of the same trunk, each layer loaded when the trunk calls
  it and released after (one weight read), so the per-run parity gate compares the layer-major schedule against the
  model's own forward;
- ``prefetch_ranges``: each layer's checkpoint byte ranges (adjacent ranges merged), for the driver's read-ahead.

Anything the spec cannot reproduce refuses (SystemExit) at load, never a zero-filled or silently skipped tensor.

A spec provides (duck typed):
    path, name, fingerprint            the artifact directory, its name and mlx2 fingerprint
    weight_map                         {checkpoint key: shard file name}
    skeleton() -> (model, tokenizer)   the adapter's module tree, nothing loaded
    layer_of(key) -> int | None | "resident" | "skip"
                                       which decoder layer a checkpoint key belongs to
    transform(raw: dict) -> dict       checkpoint keys -> model parameter paths (the adapter's sanitize)
    after_load(model, l) -> None       optional: per-layer installers after layer l is bound
    layer_scope(ids) -> context        optional: the scope the trunk's own forward opens around its layers
    receipt() -> dict                  optional
"""
from __future__ import annotations

import time
from pathlib import Path

from .host_family import HostFamily, _patched


def _flat(module):
    from mlx.utils import tree_flatten
    return dict(tree_flatten(module.parameters()))


def _set(module, arrays):
    from mlx2.runtime.layer_stream import _set_weights
    _set_weights(module, arrays)


def placeholder(shape, dtype):
    """A shape- and dtype-true array that allocates one element (evaluating a broadcast is a view)."""
    import mlx.core as mx
    return mx.broadcast_to(mx.zeros((1,) * len(shape), dtype), tuple(shape))


def _holds_array(v, depth=0):
    import mlx.core as mx
    if isinstance(v, mx.array):
        return True
    if depth < 2 and isinstance(v, (tuple, list)):
        return any(_holds_array(x, depth + 1) for x in v)
    if depth < 2 and isinstance(v, dict):
        return any(_holds_array(x, depth + 1) for x in v.values())
    return False


def drop_derived_caches(module) -> list[str]:
    """Pop every private non-parameter attribute holding arrays (a module's cache of something derived from its
    weights: Kimi's dequantized ``_kimi_dense`` keyed by array ids, the router's ``_mlx2_router_fp32``, Ling's merged
    conv weights, ...). Such a cache must not outlive the weights it was derived from: an id-keyed one could match a
    later array that reuses the id. Returns the names dropped.

    Only ``vars(m)``: an ``_``-prefixed array assigned the ordinary way lives in the module's own dict, where the
    model keeps buffers it needs (RoPE ``_freqs``), so those are never touched; none of the streamed models keeps a
    weight-derived cache there (2026-11-18). The NVFP4 fold's ``expert_output_scale`` is a plain object, kept, and
    recomputed on every load."""
    dropped = []
    for m in module.modules():
        for k in [k for k, v in vars(m).items() if k.startswith("_") and _holds_array(v)]:
            vars(m).pop(k)
            dropped.append(f"{type(m).__name__}.{k}")
    return dropped


def merge_ranges(locs):
    """[(file, begin, end)] sorted and with adjacent or overlapping ranges in one file merged."""
    spans = []
    for f, b, e in sorted(locs):
        if spans and spans[-1][0] == f and b <= spans[-1][2]:
            spans[-1][2] = max(spans[-1][2], e)
        else:
            spans.append([f, b, e])
    return [tuple(s) for s in spans]


class LayerSwapFamily(HostFamily):
    batch_rows = 4096

    def __init__(self, spec, *, moe_units="neuron", log=print):
        import mlx.core as mx
        from research.amoeba.carve.host import Host, find_inner
        t0 = time.time()
        self.spec = spec
        self.log = log
        self.path = Path(spec.path)
        model, tok = spec.skeleton()
        lm, inner = find_inner(model)
        layers = list(inner.layers)
        self._layers = layers
        prefix = self._layer_prefixes(model, layers)
        self._prefix = prefix
        self._keys = {l: [] for l in range(len(layers))}
        resident, skipped = [], []
        for key in spec.weight_map:
            where = spec.layer_of(key)
            if where == "skip":
                skipped.append(key)
            elif where == "resident":
                resident.append(key)
            elif isinstance(where, int) and 0 <= where < len(layers):
                self._keys[where].append(key)
            else:
                raise SystemExit(f"layer swap: checkpoint key {key!r} maps to {where!r}, not a layer of "
                                 f"{len(layers)}, the resident remainder or an explicit skip")
        if any(not ks for ks in self._keys.values()):
            raise SystemExit("layer swap: decoder layers with no checkpoint tensors: "
                             f"{[l for l, ks in self._keys.items() if not ks]}")
        # every layer parameter becomes a placeholder BEFORE anything evaluates the tree (the carve host does)
        self._ph = {}
        for l, layer in enumerate(layers):
            self._ph[l] = {k: placeholder(v.shape, v.dtype) for k, v in _flat(layer).items()}
            _set(layer, self._ph[l])
        want = {k for k in _flat(model) if not any(k.startswith(p) for p in prefix.values())}
        got = spec.transform(self._read(resident))
        if set(got) != want:
            raise SystemExit(f"layer swap: the resident tensors {sorted(set(got) ^ want)[:8]} differ from the model's "
                             "non-layer parameters (missing or unexpected): refusing a partial load")
        self._check_shapes(model, got, "resident")
        _set(model, got)
        mx.eval(list(got.values()))
        del got
        self.loads = 0
        self.load_bytes = 0
        self._loaded = set()
        self._dtypes = {}
        self.skipped_keys = len(skipped)
        host = Host(str(self.path), log=log, model=model, tokenizer=tok, fingerprint=spec.fingerprint,
                    checkpoint=spec.name, moe_units=moe_units)
        host.layers = layers
        super().__init__(host)
        self.streamed_layers = len(layers)
        self.layer_scope = getattr(spec, "layer_scope", None)
        self._ranges = None
        self.load_seconds = round(time.time() - t0, 1)
        log(f"layer swap: {self.checkpoint}: {len(layers)} streamed layers, {len(resident)} resident tensors, "
            f"{len(skipped)} checkpoint tensors not read (spec skip)")

    # ------------------------------------------------------------------ weights

    @staticmethod
    def _layer_prefixes(model, layers):
        """Each layer module's dotted parameter path from the model root ("model.layers.3.")."""
        ids = {id(m): i for i, m in enumerate(layers)}
        out = {}
        for name, m in model.named_modules():
            if id(m) in ids:
                out[ids[id(m)]] = name + "."
        if len(out) != len(layers):
            raise SystemExit("layer swap: cannot locate every decoder layer in the model tree")
        return out

    def _read(self, keys):
        """The checkpoint tensors, lazily (mx.load reads nothing until evaluated); no shard dict outlives the call,
        so an evaluated array is never held by a cached mapping."""
        import mlx.core as mx
        by = {}
        for k in keys:
            by.setdefault(self.spec.weight_map[k], []).append(k)
        out = {}
        for shard, ks in by.items():
            d = mx.load(str(self.path / shard))
            missing = [k for k in ks if k not in d]
            if missing:
                raise SystemExit(f"layer swap: {shard} lacks {missing[:4]} (index and shard disagree)")
            out.update({k: d[k] for k in ks})
            del d
        return out

    @staticmethod
    def _check_shapes(module, got, where):
        """Shapes exactly (load_weights strict); dtypes as load_weights binds them: a float parameter takes the
        checkpoint's float dtype (a BF16 checkpoint into a module built in float32), a packed (integer) one must
        match the module's own packing."""
        import mlx.core as mx
        have = _flat(module)
        bad = [(k, tuple(v.shape), v.dtype, tuple(have[k].shape), have[k].dtype) for k, v in got.items()
               if k in have and (tuple(v.shape) != tuple(have[k].shape)
                                 or (v.dtype != have[k].dtype and not (mx.issubdtype(v.dtype, mx.floating)
                                                                       and mx.issubdtype(have[k].dtype, mx.floating))))]
        if bad:
            raise SystemExit(f"layer swap: {where}: tensor shape or packing differs from the module: {bad[:4]}")

    def begin_layer(self, l):
        import mlx.core as mx
        if l in self._loaded:
            return
        layer, pre = self._layers[l], self._prefix[l]
        got = self.spec.transform(self._read(self._keys[l]))
        if any(not k.startswith(pre) for k in got) or {k[len(pre):] for k in got} != set(self._ph[l]):
            local = {k[len(pre):] for k in got if k.startswith(pre)}
            raise SystemExit(f"layer swap: layer {l}: checkpoint tensors {sorted(local ^ set(self._ph[l]))[:8]} or "
                             f"{[k for k in got if not k.startswith(pre)][:4]} do not match the layer's parameters")
        local = {k[len(pre):]: v for k, v in got.items()}
        del got
        self._check_shapes(layer, local, f"layer {l}")
        dtypes = {k: v.dtype for k, v in local.items()}
        if self._dtypes.setdefault(l, dtypes) != dtypes:          # every chunk binds the same arrays
            raise SystemExit(f"layer swap: layer {l}: tensor dtypes changed between loads")
        _set(layer, local)
        self._loaded.add(l)                     # bound: release_layer restores the placeholders whatever fails next
        try:
            mx.eval(list(local.values()))
            self.load_bytes += int(sum(v.nbytes for v in local.values()))
            del local
            after = getattr(self.spec, "after_load", None)
            if callable(after):
                after(self.h.model, l)
        except BaseException:
            self.release_layer(l)
            raise
        self.loads += 1

    def release_layer(self, l, exc=None):
        import mlx.core as mx
        if l not in self._loaded:
            return
        layer = self._layers[l]
        # placeholders in the dtypes the checkpoint bound (a float module built in float32 holds BF16 weights)
        self._ph[l] = {k: placeholder(v.shape, self._dtypes[l].get(k, v.dtype)) for k, v in self._ph[l].items()}
        _set(layer, self._ph[l])
        drop_derived_caches(layer)
        self._loaded.discard(l)
        mx.clear_cache()

    def _check_plain_outputs(self):
        self.begin_layer(0)
        try:
            super()._check_plain_outputs()
        finally:
            self.release_layer(0)

    # ------------------------------------------------------------------ driver hooks

    def run_layer(self, l, carry):
        if l not in self._loaded:
            raise SystemExit(f"layer swap: run_layer({l}) before begin_layer({l}) (a layer is read once per chunk)")
        return super().run_layer(l, carry)

    def reference_logits(self, ids, layers=None):
        """The ordinary text-major forward: the trunk calls every layer in order; each is loaded when called and
        released once its output is evaluated (one weight read). With ``layers`` the later layers pass through."""
        import mlx.core as mx
        ids = ids if hasattr(ids, "shape") else mx.array(ids)
        n = len(self._layers) if not layers else min(int(layers), len(self._layers))

        def call(i):
            if i >= n:
                return lambda self_, x, *a, **k: x
            base = type(self._layers[i])

            def f(self_, x, *a, _i=i, _b=base, **k):
                self.begin_layer(_i)
                try:
                    out = _b.__call__(self_, x, *a, **k)
                    mx.eval(out)
                finally:
                    self.release_layer(_i)
                return out
            return f
        with _patched(self._layers, call):
            out = self.h.lm(ids)
            mx.eval(out)
        return out

    @property
    def prefetch_ranges(self):
        if self._ranges is None:
            from mlx2.experimental.amoeba.inspection import safetensors_index
            loc = {k: (f, b, e) for k, f, b, e in safetensors_index(self.path)}
            self._ranges = {l: merge_ranges([loc[k] for k in ks]) for l, ks in self._keys.items()}
        return self._ranges

    @prefetch_ranges.setter
    def prefetch_ranges(self, value):          # HostFamily.__init__ assigns {}; the ranges come from the index
        pass

    def stream_counters(self):
        return {"page_ins": self.loads, "page_in_bytes": self.load_bytes, "evictions": None, "bypasses": None,
                "hits": None}

    def stream_receipt(self):
        rec = {"mode": "layer_swap", "streamed_layers": len(self._layers), "layer_loads": self.loads,
               "loaded_bytes": self.load_bytes, "checkpoint_tensors_not_read": self.skipped_keys,
               "resident": "embeddings, final norm, head and any other non-layer parameter"}
        extra = getattr(self.spec, "receipt", None)
        if callable(extra):
            rec["spec"] = extra()
        return rec

    def close(self):
        for l in list(self._loaded):
            self.release_layer(l)
