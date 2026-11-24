# SPDX-License-Identifier: Apache-2.0
"""Row-exact native-MTP verify for Maxwell-V1-Flash (opt-in).

A batched verify forward of ``T`` rows (the current token plus the drafts)
runs each matmul at width ``T``.  MLX picks a different kernel and summation
order for a multi-row product than for decode's one-row product, so the
verify logits of a row can differ from the logits plain decoding computes at
that position by a few ulps, enough to flip a near-tied argmax.  Greedy
output with MTP on then differs from MTP off.

Here every verify row gets the arithmetic of the one-token decode step at the
same position:

``layerwise``
    Layer by layer, the weight-bearing projections (q/k/v/indexer, o_proj,
    the shared expert and dense MLPs) run as one ``[1, T, K]`` strip through
    the shared row-exact ops (``runtime.row_exact.ops``): one weight read,
    each row with the one-row law, gated by a parity canary that falls back
    to one call per row.  Routed experts go through ``ops.routed_experts``
    (the same canary contract).  Everything else (the mHC mixes, norms, rope,
    cache update, attention, the router and the expert combine) runs once
    per row through the decode code itself, so a row reads exactly the keys
    decode would read at its position.  The head is a strip only when its
    canary holds; otherwise one head call per row.
``serial``
    One full decode forward per row.  Exact by construction and as slow as
    ``T`` decode steps: the reference the layerwise form is tested against.

Scope: one lane (``B == 1``).  A multi-lane verify keeps the batched
backbone, which is never labelled row-exact: the decline is counted and the
window records the failure.  Decode, prefill and draft (MTP head) arithmetic
are unchanged.  Original mlx2 code; the window/ops protocol is the shared
``row_exact_verify`` / ``row_exact`` package.
"""

from __future__ import annotations

import mlx.core as mx

from .. import row_exact_verify as REV
from ..row_exact import ops
from ..row_exact.canary import CANARIES, probe_input
from .activations import swiglu
from .base import create_attention_mask
from .cache import KVCache

STATS = {"windows": 0, "rows": 0, "declined_multi_lane": 0, "inexact_windows": 0,
         "head_strip": 0, "head_per_row": 0}
LAST = {"routes": None, "failures": None, "head": None}
_VERDICT_SLOT = "_mlx2_maxwell_row_exact_head_verdicts"


def stats(reset: bool = False) -> dict:
    out = dict(STATS, last_routes=LAST["routes"], last_failures=LAST["failures"], last_head=LAST["head"])
    if reset:
        for key in STATS:
            STATS[key] = 0
        LAST["routes"] = LAST["failures"] = LAST["head"] = None
    return out


def install(model, mode: str) -> None:
    """Bind the verify hooks ``hybrid_speculative`` looks up on the model."""
    if mode not in ("layerwise", "serial"):
        raise ValueError(f"unknown Maxwell row-exact verify mode {mode!r}")
    object.__setattr__(model, "_row_exact_mode", mode)
    object.__setattr__(model, "mtp_verify_backbone", lambda tokens, cache: verify_backbone(model, tokens, cache))
    object.__setattr__(model, "mtp_verify_logits", lambda hidden: verify_logits(model, hidden))


def _rows(x):
    return [x[:, r:r + 1] for r in range(x.shape[1])]


def _cat(parts):
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def verify_backbone(model, tokens, cache):
    """``(post-norm hidden, final streams)`` for the verify rows, as ``mtp_backbone``."""
    if tokens.shape[1] == 1:
        return model.mtp_backbone(tokens, cache)
    window = REV.Window(tokens.shape[1])
    if tokens.shape[0] != 1:
        STATS["declined_multi_lane"] += 1
        window.fail("multi_lane_batched_verify")
        _close(window, counted=False)
        return model.mtp_backbone(tokens, cache)
    STATS["windows"] += 1
    STATS["rows"] += tokens.shape[1]
    with REV.window(window):
        if model._row_exact_mode == "serial":
            outs = [model.mtp_backbone(tokens[:, r:r + 1], cache) for r in range(tokens.shape[1])]
            window.note("blocks", "serial_decode_forward", tokens.shape[1])
            result = _cat([o[0] for o in outs]), _cat([o[1] for o in outs])
        else:
            result = _layerwise(model, tokens, cache, window)
    _close(window)
    return result


def _close(window, counted=True):
    if counted and not window.exact:
        STATS["inexact_windows"] += 1
    LAST["routes"] = {stage: dict(routes) for stage, routes in window.stages.items()}
    LAST["failures"] = dict(window.failures)


def _layerwise(model, tokens, cache, window):
    from .maxwell_moe import _row_padding

    decoder = model.model
    n = len(decoder.layers)
    if cache is None:
        cache = [None] * (2 * n)
    if len(cache) != 2 * n:
        raise ValueError(f"Maxwell cache must hold {n} trunk and {n} indexer planes, got {len(cache)}")
    main, index = cache[:n], cache[n:]
    first = main[0]
    reach = first.offset if type(first) is KVCache else getattr(first, "_idx", 0) if first is not None else 0
    if int(reach) + tokens.shape[1] > decoder.args.max_position_embeddings:
        raise ValueError(f"Maxwell context exceeds {decoder.args.max_position_embeddings} positions")
    xs = _rows(decoder.expand(tokens))
    # Decode computes the mask and the row padding once per step from the
    # first trunk plane before layer 0 writes it.  Layer 0's row r sees that
    # plane in exactly that state, so record them there for every layer.
    masks, paddings = [], []
    for i, (layer, c, ic) in enumerate(zip(decoder.layers, main, index, strict=True)):
        xs = _layer_rows(layer, xs, c, ic, masks, paddings, first_layer=i == 0,
                         dense_limit=decoder.args.dense_context_limit, row_padding=_row_padding)
        window.note("blocks", "layerwise_exact")
    streams = _cat(xs)
    hidden = _cat([decoder.norm(x) for x in xs])
    model.completed_forwards += 1
    return hidden, streams


def _linear(module, strip, stage):
    return ops.linear(module, strip, stage=stage)


def _layer_rows(layer, xs, cache, index_cache, masks, paddings, *, first_layer, dense_limit, row_padding):
    attn = layer.self_attn
    hc = layer.self_attn_hyper_connection
    pre = [hc.pre(x) for x in xs]
    normed = _cat([layer.input_layernorm(a) for a, _, _ in pre])
    if "qkv_proj" in attn:
        y = _linear(attn.qkv_proj, normed, "maxwell_qkv")
        cuts, acc = [], 0
        for size in attn._split[:-1]:
            acc += size
            cuts.append(acc)
        parts = mx.split(y, cuts, axis=-1)
        q, k, v = parts[:3]
        ik = parts[3] if attn.sparse else None
    else:
        q = _linear(attn.q_proj, normed, "maxwell_q")
        k = _linear(attn.k_proj, normed, "maxwell_k")
        v = _linear(attn.v_proj, normed, "maxwell_v")
        ik = _linear(attn.indexer.linear_k_proj, normed, "maxwell_index_k") if attn.sparse else None
    # The indexer queries are needed (and their canary probed) only when a row
    # takes the sparse route, as in decode.
    iq_strip = []

    def iq_row(r):
        if not iq_strip:
            iq_strip.append(_linear(attn.indexer.linear_q_proj, normed, "maxwell_index_q"))
        return iq_strip[0][:, r:r + 1]

    outs = []
    for r in range(len(xs)):
        if first_layer:
            masks.append(create_attention_mask(xs[r], cache))
            paddings.append(row_padding(cache, 1, dense_limit))
        sl = slice(r, r + 1)
        row_iq = (lambda r=r: iq_row(r)) if attn.sparse else None
        outs.append(attn._attend(q[:, sl], k[:, sl], v[:, sl], None if ik is None else ik[:, sl], row_iq,
                                 masks[r], cache, index_cache, paddings[r]))
    o = _linear(attn.o_proj, _cat(outs), "maxwell_o")
    xs = [hc.post(o[:, r:r + 1], x, hp, hr) for r, (x, (_, hp, hr)) in enumerate(zip(xs, pre))]
    hc = layer.mlp_hyper_connection
    pre = [hc.pre(x) for x in xs]
    normed = [layer.post_attention_layernorm(m) for m, _, _ in pre]
    out = _mlp_rows(layer.mlp, normed)
    return [hc.post(out[r], x, hp, hr) for r, (x, (_, hp, hr)) in enumerate(zip(xs, pre))]


def _dense_mlp_rows(mlp, strip, stage):
    gate = _linear(mlp.gate_proj, strip, stage + "_gate")
    up = _linear(mlp.up_proj, strip, stage + "_up")
    act = _cat([swiglu(gate[:, r:r + 1], up[:, r:r + 1]) for r in range(strip.shape[1])])
    return _linear(mlp.down_proj, act, stage + "_down")


def _mlp_rows(mlp, normed):
    """Per-row MLP outputs (``[1, 1, H]`` each) with decode arithmetic."""
    strip = _cat(normed)
    if not hasattr(mlp, "switch_mlp"):
        y = _dense_mlp_rows(mlp, strip, "maxwell_mlp")
        return _rows(y)
    picks = [mlp.gate(x) for x in normed]                 # the router, once per row
    ids = _cat([p[0] for p in picks])
    routed = ops.routed_experts(mlp.switch_mlp, strip, ids, num_experts=mlp.num_experts,
                                key=("maxwell_switch_glu",), stage="maxwell_experts")
    shared = _dense_mlp_rows(mlp.shared_expert, strip, "maxwell_shared")
    out = []
    for r, (x, (_, w)) in enumerate(zip(normed, picks)):
        w = w.astype(mx.float32) * mlp.scale
        mixed = (routed[:, r:r + 1].astype(mx.float32) * w[..., None]).sum(axis=-2)
        out.append(mixed.astype(x.dtype) + shared[:, r:r + 1])
    return out


def verify_logits(model, hidden):
    """Verify-row logits, each bit-equal to that row's own one-row head call."""
    if hidden.ndim != 3 or hidden.shape[1] == 1 or hidden.shape[0] != 1:
        return model.logits_from_hidden(hidden)
    rows = hidden.shape[1]
    if _head_strip_exact(model, rows, hidden.dtype):
        STATS["head_strip"] += 1
        LAST["head"] = "strip"
        return model.logits_from_hidden(hidden)
    STATS["head_per_row"] += 1
    LAST["head"] = "per_row"
    return _cat([model.logits_from_hidden(h) for h in _rows(hidden)])


def _head_strip_exact(model, rows, dtype) -> bool:
    """Whether the head's multi-row call matches its one-row call bit for bit
    on this build, device and weight (probed once per row count).

    Verdicts live on the head module (freed with the model) and are keyed on
    whether the bf16 head kernel is still enabled: a disabled kernel moves
    both forms to other paths, so an earlier verdict says nothing."""
    from . import maxwell_moe as M

    head = model.lm_head
    weight = head["weight"] if "weight" in head else None
    with CANARIES._lock:
        verdicts = head.__dict__.get(_VERDICT_SLOT)
        if verdicts is None:
            verdicts = {}
            object.__setattr__(head, _VERDICT_SLOT, verdicts)
        slot = (rows, dtype, CANARIES.generation, str(mx.default_device()), "head_gemv" in M._KERNEL_DISABLED)
        for stale in [k for k in verdicts if k[2] != CANARIES.generation]:
            del verdicts[stale]
        hit = verdicts.get(slot)
        if hit is not None and hit[0] is weight:
            return hit[1]
        width = model.args.hidden_size

        def probe(probe_rows):
            sample = probe_input(probe_rows, width, dtype, seed=5153)
            batched = model.logits_from_hidden(sample)
            rowwise = _cat([model.logits_from_hidden(sample[:, r:r + 1]) for r in range(probe_rows)])
            return batched, rowwise

        generation = CANARIES.generation
        verdict = CANARIES.exact_private(("maxwell_head", rows, str(dtype)), probe, rows=rows)
        if CANARIES.generation == generation and slot[4] == ("head_gemv" in M._KERNEL_DISABLED):
            verdicts[slot] = (weight, verdict)
        return verdict
