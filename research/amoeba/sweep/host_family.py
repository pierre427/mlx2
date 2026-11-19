"""Host sweep family: any model the carve host loads, run one layer at a time (SweepFamily interface v1, README.md).

Not registered in FAMILIES: research.amoeba.sweep.capture (the older driver) knows neither its ``neurons`` payload
nor ``begin_layer``; child_rolling loads it by name (``--family host``). Driver extensions beyond README v1, both
optional: ``begin_layer(l)`` (load layer l once before its batches; ``release_layer`` drops it) and the ``neurons``
payload kind.

For every dense adapter that accepts a weight-streaming request (src/mlx2/runtime/streamed_load.py) and has no family of
its own. Nothing numerical is reimplemented: the model is the adapter's, loaded with ``weight_streaming`` in
``dense_layers`` (whole decoder layers) or ``moe_experts`` (routed experts; everything else resident) mode, or
resident for parity runs. The family only schedules it layer-major:

- carry   ``{"h": [B, T, d], "ids": [B, T]}``: the residual stream and the tokens. Each layer's call arguments (the
          mask, per-layer inputs such as PLE) are recomputed from the ids by the model's own trunk, run up to that
          layer with every earlier layer short-circuited, so no KV cache, no APCv2, no reimplemented trunk.
- payload the carve host's unit-site hook payload for the layer (research/amoeba/carve/units.py): dense
          ``{"act"}``; MoE expert units ``{"inds", "w", "rows"}``; MoE neuron units (kind ``"neurons"``)
          ``{"inds", "w", "h", "F"}`` with h the experts' intermediate activations [B, T, k, F].
- final_logits / reference_logits: the model's own output path on the carry, and the ordinary text-major forward
  (of the first ``layers`` blocks when given), for the parity gate.
- ``prefetch_ranges``: each layer's checkpoint byte ranges (streamed tensors only), for the driver's read-ahead.

Layers whose outputs are tuples are carried by their first element; a model whose layers pass state to each other
beyond the residual stream is caught by the parity gate (the driver refuses), never silently mapped.
"""
from __future__ import annotations

import time
from pathlib import Path


class _Stop(BaseException):          # never swallowed by a trunk's ``except Exception`` fallback
    pass


def _hid(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _patched(layers, calls):
    """Context: swap each layer's class for one whose __call__ is calls[i] (None = untouched)."""
    class Ctx:
        def __enter__(self):
            self.saved = []
            for i, layer in enumerate(layers):
                c = calls(i)
                if c is None:
                    continue
                base = type(layer)
                layer.__class__ = type(f"Sweep{base.__name__}", (base,), {"__call__": c})
                self.saved.append((layer, base))

        def __exit__(self, *exc):
            for layer, base in self.saved:
                layer.__class__ = base
    return Ctx()


class HostFamily:
    batch_rows = 4096

    def __init__(self, host, *, stream=None, model_path=None, stream_mode=None):
        self.h = host
        self.stream = stream
        self._held = None
        self._blocks = {}
        for i, layer in enumerate(host.layers):  # streamed layers: the real block, called under one acquire per layer
            if type(layer).__name__ == "StreamedDecoderLayer":
                slot = object.__getattribute__(layer, "_stream_slot")
                self._blocks[i] = (slot.path, slot.block)
        self._check_plain_outputs()
        self.checkpoint, self.fingerprint = host.checkpoint, host.fingerprint
        self.L = host.L
        self.widths = [int(w) for w in host.widths]
        self.kinds = [None if s.kind is None else ("dense" if s.kind == "dense" else
                                                    ("neurons" if s.units == "neuron" else "experts"))
                      for s in host.sites]
        self.tok = host
        self.F = [int(getattr(s, "F", 0) or 0) for s in host.sites]
        self.prefetch_ranges = {}            # the streamed tensors' own checkpoint bytes, per layer
        if stream is not None:
            for i, (path, _) in self._blocks.items():
                self.prefetch_ranges[i] = [(str(loc.path), int(loc.begin), int(loc.end))
                                           for loc in stream.readers[path].specs.values()]

    def _check_plain_outputs(self):
        """Layers must return the residual stream itself: a trunk that unpacks tuples or carries state from layer to
        layer (Gemma4's (h, cache, ...), HySparse2's selected blocks, shared-KV trunks) cannot be short-circuited
        layer by layer, so it refuses here and needs a sweep family of its own."""
        import mlx.core as mx
        seen = {}
        layer = self.h.layers[0]
        base = type(layer)

        def call(self_, x, *a, **k):
            out = base.__call__(self_, x, *a, **k)
            seen["out"] = out
            raise _Stop
        layer.__class__ = type(f"Probe{base.__name__}", (base,), {"__call__": call})
        try:
            self.h.inner(mx.array([[1, 2, 3, 4]]))
        except _Stop:
            pass
        finally:
            layer.__class__ = base
        if isinstance(seen.get("out"), (tuple, list, dict)):
            raise SystemExit("host family: this model's decoder layers return a tuple (state passed between layers); "
                             "layer-major short-circuiting would change the trunk: it needs a sweep family of its own")

    def begin_layer(self, l):
        """Load layer l once for all its batches: the stream manager's acquire held open until release_layer."""
        if l in self._blocks and self.stream is not None:
            path, _ = self._blocks[l]
            self._held = (l, self.stream.acquire(path))
            self._held[1].__enter__()

    # tokenizer duck type (driver calls fam.tok.encode / decode)
    def encode(self, text):
        return self.h.encode(text)

    def decode(self, ids):
        return self.h.decode(ids)

    def _trunk(self, ids):
        import mlx.core as mx
        return self.h.inner(mx.array(ids) if not hasattr(ids, "shape") else ids)

    def embed(self, ids):
        import mlx.core as mx
        rec = {}

        def stop(self_, x, *a, **k):
            rec["x"] = x
            raise _Stop
        with _patched(self.h.layers, lambda i: stop if i == 0 else None):
            try:
                self._trunk(ids)
            except _Stop:
                pass
        mx.eval(rec["x"])
        return {"h": rec["x"], "ids": ids if hasattr(ids, "shape") else mx.array(ids)}

    def layer_args(self, ids, l):
        rec = {}

        def call(i):
            if i > l:
                return None

            def f(self_, x, *a, _i=i, **k):
                if _i == l:
                    rec["a"], rec["k"] = a, k
                    raise _Stop
                return x
            return f
        with _patched(self.h.layers, call):
            try:
                self._trunk(ids)
            except _Stop:
                pass
        if "a" not in rec:
            raise SystemExit(f"host family: the trunk never called layer {l}")
        return rec["a"], rec["k"]

    def run_layer(self, l, carry):
        import mlx.core as mx
        a, k = self.layer_args(carry["ids"], l)
        pay = {}

        def st(l_, kind, p):
            if l_ == l:
                pay.update(p)
        c = self.h.ctl
        c.reset()
        c.stats = st if self.kinds[l] is not None else None
        target = self._blocks[l][1] if (l in self._blocks and self._held and self._held[0] == l) else self.h.layers[l]
        # the scope the trunk's own forward opens around its layers (e.g. the NAX MoE gather's phase), so the layer
        # runs the mechanisms its execution binding names, as in the ordinary forward (qwen35_capture, sweep L7-1)
        # (a family names it as ``layer_scope`` when the trunk opens it from its model module, as Kimi and Nemotron-H
        # open moe_nax_gather.forward_scope)
        scope = getattr(self, "layer_scope", None) or getattr(self.h.inner, "forward_scope", None)
        from contextlib import nullcontext
        try:
            with (scope(carry["ids"], None) if callable(scope) else nullcontext()):
                h = _hid(target(carry["h"], *a, **k))
                mx.eval(h)
        finally:
            c.reset()
        out = None
        if self.kinds[l] == "dense":
            out = {"act": pay["act"]}
        elif self.kinds[l] == "experts":
            out = {"inds": pay["inds"], "w": pay["w"], "rows": pay["rows"]}
        elif self.kinds[l] == "neurons":
            if pay.get("h") is None:
                raise SystemExit(f"host family: layer {l} neuron site gave no intermediate activations")
            out = {"inds": pay["inds"], "w": pay["w"], "h": pay["h"], "F": self.F[l]}
        return {"h": h, "ids": carry["ids"]}, out

    def release_layer(self, l, exc=None):
        if self._held and self._held[0] == l:
            ctx, self._held = self._held[1], None
            if exc is None:
                ctx.__exit__(None, None, None)
            else:                                  # a failed forward: the manager marks itself failed (no reuse)
                try:
                    ctx.__exit__(type(exc), exc, exc.__traceback__)
                except BaseException:
                    pass

    def stream_counters(self):
        st = getattr(self.stream, "stats", None)
        if st is None:
            return None
        return {"page_ins": getattr(st, "page_ins", None), "page_in_bytes": getattr(st, "page_in_bytes", None),
                "evictions": getattr(self.stream, "evictions", None), "bypasses": getattr(self.stream, "bypasses", None),
                "hits": getattr(self.stream, "hits", None)}

    def _output(self, ids, h_last, upto):
        """The model's own output path on the state after layer upto-1: earlier layers short-circuited, layer upto-1
        returns h_last, later layers pass it through."""
        def call(i):
            if i < upto - 1 or i >= upto:
                return lambda self_, x, *a, **k: x
            return lambda self_, x, *a, **k: h_last
        with _patched(self.h.layers, call):
            return self.h.lm(ids)

    def final_logits(self, carry):
        return self._output(carry["ids"], carry["h"], self.L)

    def reference_logits(self, ids, layers=None):
        import mlx.core as mx
        ids = ids if hasattr(ids, "shape") else mx.array(ids)
        if not layers or layers >= len(self.h.layers):
            return self.h.lm(ids)
        with _patched(self.h.layers, lambda i: (lambda self_, x, *a, **k: x) if i >= layers else None):
            return self.h.lm(ids)

    def stream_receipt(self):
        return self.stream.receipt() if self.stream is not None and hasattr(self.stream, "receipt") else None


def load(model_dir, *, stream_mode="dense_layers", ceiling_bytes=None, resident=False, moe_units="neuron", log=print,
         admission_gb=None):
    from mlx2.experimental.amoeba.child_rolling import load_streamed_host
    t0 = time.time()
    host, stream = load_streamed_host(Path(model_dir), stream_mode=stream_mode, ceiling_bytes=ceiling_bytes,
                                      resident=resident, moe_units=moe_units, log=log, admission_gb=admission_gb)
    fam = HostFamily(host, stream=stream, model_path=model_dir, stream_mode=None if resident else stream_mode)
    fam.load_seconds = round(time.time() - t0, 1)
    return fam
