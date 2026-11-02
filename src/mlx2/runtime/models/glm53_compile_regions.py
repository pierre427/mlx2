# SPDX-License-Identifier: MIT
"""Opt-in GLM-5.3 decode layer bodies under ``mx.compile`` (``glm53_compile_regions``).

Policy ``off`` (default) keeps the eager forward. ``v1`` traces, per layer, the
stateless part of one B1 single-token ordinary decode step once and replays
the recorded graph on every later step, so the Python graph construction and
MLX's per-call custom-kernel source assembly (``mx.fast.metal_kernel`` joins
its ~70 KB K-block header into every call) run once per layer instead of once
per step:

``kda``  the whole linear-attention layer: attention HC-in, KDA (projections,
         conv, L2 norms, gates, delta-rule kernel, norm, output projection;
         under ``glm53_kda_fused=decode_v1`` the post-projection body is one
         custom kernel inside the trace),
         HC expand, FFN HC-in, MLP/MoE, HC expand. The recurrent and conv
         state are explicit inputs and outputs; the cache is written outside.
``mla``  the FFN half of a latent-attention layer (attention HC expand, FFN
         HC-in, MLP/MoE, HC expand). Attention HC-in, the latent/indexer cache
         append and the sparse attention stay eager: the append mutates the
         cache planes (glm53_kv_append) and is never traced.

Exactness is a gate, not an assumption: every region must equal the eager
forward bit for bit on the served route (logits and KDA state, recorded by
scripts/research/glm53_compile_regions_probe.py). ``gate_sigmoid`` runs as the
precise Metal kernel inside a traced region (precise_ops), so a fused sigmoid
can never stand in for ``mx.sigmoid``. Route counters (model.calls, the
weights' scheduled_* ledgers, the HC kernel ledgers) are recorded once while
tracing and replayed per step, so receipts count the same launches as eager.

Engages only for B1 decode rows (one token, outside a prompt scope) on the
Metal GPU with every KDA state present; everything else runs eager. The first
admitted step under a policy runs eager too, so every lazily built host memo
holds real arrays before anything is traced. A change of any policy attribute
the trace read keys a fresh warm-up and trace.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from . import deepseek_v4_hc, glm53_hc_fused, glm53_hc_mega, precise_ops

POLICIES = ("off", "v1")
_COUNTERS = {"kda_layers": 0, "mla_ffn_halves": 0, "traces": 0, "declined_steps": 0, "warmup_steps": 0}
# Serializes tracing and the counter replay: a trace records its counter delta
# from the shared ledgers, and the trace flag is per region. (A model forward
# is not reentrant anyway: its caches and ledgers are plain Python state.)
# Known limit: the HC kernel ledgers are process-global, so an eager forward
# of another model running on another thread during a trace would be counted
# into that region's replayed delta (receipt counts only; values unaffected).
_LOCK = threading.RLock()
# Arrival counters per region call (glm53_hc_fused tail_v1, glm53_router
# fused_v2: at most two HC boundaries and one router per region). Passed as a
# fresh host-made input on every call, never captured, so no two replays share one.
COUNTER_SLOTS = 4


def _ledgers(model):
    weights = model.weights
    return [
        ledger for ledger in (
            model.calls,
            getattr(weights, "scheduled_dense_calls", None),
            getattr(weights, "scheduled_grouped_calls", None),
            getattr(weights, "scheduled_expert_calls", None),
            glm53_hc_mega._SCHEDULED,
            glm53_hc_fused._SCHEDULED,
            deepseek_v4_hc._SCHEDULED,
        ) if isinstance(ledger, dict)
    ]


def _snapshot(ledgers):
    return [dict(ledger) for ledger in ledgers]


def _delta(before, ledgers):
    return [
        {key: value - old.get(key, 0) for key, value in ledger.items() if value != old.get(key, 0)}
        for old, ledger in zip(before, ledgers)
    ]


class _Region:
    """One compiled layer body plus the route-counter delta of its trace."""

    __slots__ = ("delta", "fn", "traced")

    def __init__(self, body, model):
        self.delta = None
        self.traced = False

        from .glm53_glue_metal import counter_supply

        def traced(*arrays):
            self.traced = True
            _COUNTERS["traces"] += 1
            ledgers = _ledgers(model)
            before = _snapshot(ledgers)
            token = precise_ops._IN_PRECISE_SPAN.set(True)
            try:
                with counter_supply(arrays[-1]):
                    result = body(*arrays[:-1])
            finally:
                precise_ops._IN_PRECISE_SPAN.reset(token)
            self.delta = _delta(before, ledgers)
            return result

        self.fn = mx.compile(traced)

    def __call__(self, model, *arrays):
        with _LOCK:
            self.traced = False
            counters = mx.array(np.zeros(COUNTER_SLOTS * 16, np.uint32))
            result = self.fn(*arrays, counters)
            if not self.traced:
                for ledger, delta in zip(_ledgers(model), self.delta):
                    for key, value in delta.items():
                        ledger[key] = ledger.get(key, 0) + value
            return result


class CompiledLayers:
    """Per-model cache of compiled layer bodies, keyed by the traced policy."""

    def __init__(self, model):
        self.model = model
        self.key = None
        self.warm_key = None
        self.regions = {}

    def _policy_key(self):
        model, weights = self.model, self.model.weights
        return (
            id(weights), model.math_policy, model.hc_mega, model.hc_fused,
            getattr(model, "kda_group", False), getattr(model, "shared_fold", False),
            getattr(model, "kda_small", "reference"), getattr(model, "router", "reference"),
            getattr(model, "kda_fused", "reference"), getattr(model, "hc_fused_policy", "reference"),
            getattr(weights, "row_policy", None), getattr(weights, "dense_prefill_policy", None),
            getattr(weights, "expert_prefill_policy", None), getattr(weights, "compute_policy", None),
            getattr(weights, "row_token_share", False), getattr(weights, "expert_union", False),
            getattr(weights, "_kblock_exact", None),
            model.args.rms_norm_eps, str(mx.default_device()),
        )

    def _region(self, kind, index):
        key = self._policy_key()
        if key != self.key:
            self.key, self.regions = key, {}
        region = self.regions.get((kind, index))
        if region is None:
            body = self._kda_body(index) if kind == "kda" else self._mla_ffn_body(index)
            region = self.regions[(kind, index)] = _Region(body, self.model)
        return region

    def _ffn_half(self, index, branch, h, post, comb):
        """Attention HC expand, FFN HC-in (one boundary: _mid), MLP/MoE, HC expand."""
        model, prefix = self.model, f"blk.{index}"
        h, x, post, comb = _mid(model, branch, h, post, comb, prefix + ".hc_ffn", prefix + ".ffn_norm.weight")
        return _expand(model._mlp(x, prefix, index), h, post, comb, model.hc_fused)

    def _kda_body(self, index):
        model, prefix = self.model, f"blk.{index}"

        def body(h, conv, recurrent):
            state = SimpleNamespace(conv=conv, recurrent=recurrent, offset=0)
            x, post, comb = model._hc_in(h, prefix + ".hc_attn", prefix + ".attn_norm.weight")
            branch = model._kda(x, state, prefix)
            return self._ffn_half(index, branch, h, post, comb), state.conv, state.recurrent

        return body

    def _mla_ffn_body(self, index):
        model = self.model

        def body(branch, h, post, comb):
            return self._ffn_half(index, branch, h, post, comb)

        return body

    def layer(self, index, layer, cache, h, offset):
        """One layer of a B1 decode step; same values and cache writes as eager."""
        model, prefix = self.model, f"blk.{index}"
        if layer.is_linear:
            h, cache.conv, cache.recurrent = self._region("kda", index)(
                model, h, cache.conv, cache.recurrent)
            cache.offset += 1
            _COUNTERS["kda_layers"] += 1
            return h
        x, post, comb = model._hc_in(h, prefix + ".hc_attn", prefix + ".attn_norm.weight")
        branch = model._mla(x, cache, prefix, frontier=offset)
        _COUNTERS["mla_ffn_halves"] += 1
        return self._region("mla", index)(model, branch, h, post, comb)


class _Warmup:
    """One eager decode step (the body of Glm53GGUFModel.__call__'s layer loop);
    marks the policy warm only once every layer has been built."""

    def __init__(self, layers, key):
        self.layers, self.key = layers, key

    def layer(self, index, layer, cache, h, offset):
        model, prefix = self.layers.model, f"blk.{index}"
        x, post, comb = model._hc_in(h, prefix + ".hc_attn", prefix + ".attn_norm.weight")
        branch = (
            model._kda(x, cache, prefix) if layer.is_linear
            else model._mla(x, cache, prefix, frontier=offset)
        )
        h, x, post, comb = _mid(model, branch, h, post, comb, prefix + ".hc_ffn", prefix + ".ffn_norm.weight")
        h = _expand(model._mlp(x, prefix, index), h, post, comb, model.hc_fused)
        if index == len(model.layers) - 1:
            self.layers.warm_key = self.key
            _COUNTERS["warmup_steps"] += 1
        return h


def _expand(branch, residual, post, comb, fused):
    from .glm53_gguf import hc_expand

    return hc_expand(branch, residual, post, comb, fused=fused)


def _mid(model, branch, h, post, comb, half, norm):
    """Expand then the next half's HC-in: the model's _hc_mid (one fused
    boundary under glm53_hc_fused), or the two steps for minimal models."""
    mid = getattr(model, "_hc_mid", None)
    if mid is not None:
        return mid(branch, h, post, comb, half, norm)
    h = _expand(branch, h, post, comb, model.hc_fused)
    return (h, *model._hc_in(h, half, norm))


def active(model, inputs, cache, *, plain=True):
    """The layer runner of an admitted decode step, or None (eager).

    ``plain`` is False for tap/body-only forwards (AST rows, feature capture):
    those stay eager.
    """
    if getattr(model, "compile_regions", "off") != "v1":
        return None
    from .glm53_gguf import sparse_phase

    admitted = (
        plain
        and inputs.shape == (1, 1)
        and sparse_phase() != "prompt"
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and all(c.conv is not None and c.recurrent is not None
                for c, layer in zip(cache, model.layers) if layer.is_linear)
    )
    if not admitted:
        _COUNTERS["declined_steps"] += 1
        return None
    if model.hc_mega:
        # First-use GPU gate evaluates arrays: never inside a trace.
        glm53_hc_mega.self_check()
        if getattr(model, "hc_fused_policy", "reference") != "reference":
            glm53_hc_fused.self_check()
    layers = model.__dict__.get("_compiled_layers")
    if layers is None:
        layers = model.__dict__["_compiled_layers"] = CompiledLayers(model)
    key = layers._policy_key()
    if layers.warm_key != key:
        # One complete eager decode step per policy first: the layers' lazily
        # built host memos (conv filters, KDA constants, fp32 operands,
        # provider uploads) then hold real arrays, never values created while
        # tracing. An interrupted warm-up leaves the policy cold.
        return _Warmup(layers, key)
    return layers


def release(model):
    """Drop compiled bodies and host memos (they capture weight arrays)."""
    with _LOCK:
        for name in ("_compiled_layers", "_kda_static_memo", "_hc_in_memo", "_mlp_names",
                     "_hc_fused_memo", "_f32_rows_memo"):
            model.__dict__.pop(name, None)


def route_status():
    return {"policy": "v1", "implemented": True, "qualified": False, "selected_by_default": False,
            "counters": dict(_COUNTERS)}
