#!/usr/bin/env python3
"""Where does a long Qwen 3.5 / 3.6 / 3.8 prefill spend its time?

Covers the hybrid GDN + full-attention families served by
``runtime/models/qwen38_27b.py`` (Qwen3.8 27B, Qwen3.5 9B) and
``runtime/models/qwen36_35b.py`` (Qwen3.6 35B-A3B, sparse MoE).  The model is
loaded through its adapter with the REAL weights, so absolute numbers are
production numbers (an earlier random-weight profiler for another family ran
about 2x high because its parameters were float32).

PART A -- real chunked prefill.  A ``--context`` token stream (this repo's
``docs/*.md`` tokenized and repeated) is prefilled chunk by chunk exactly as
``BatchGenerator`` does it: ``model(chunk, cache=cache)``, then
``mx.eval([c.state for c in cache])``, then ``mx.clear_cache()``.  Per-chunk
wall ms against the chunk's start offset gives the attention growth.

PART B -- seam breakdown at chosen offsets.  Just before PART A processes the
chunk that starts at a requested offset, the harness:

  1. captures that chunk's input hidden states at one GDN layer and one
     full-attention layer by running ``embed_tokens`` and the layers below the
     target on the same chunk tokens, each against a COPY of its live cache,
     with the masks built the way ``Qwen3_5TextModel.__call__`` builds them.
     Same weights, same inputs, same cache state, same op sequence, so the
     captured activation is the one the real forward is about to compute.
     This is not assumed: while PART A then runs the real chunk, the two
     target layers are wrapped by a pass-through tap that records their
     actual input, and ``capture_max_abs_diff`` reports the difference.
  2. times the REAL layer object against a fresh COPY of its live cache for
     every repeat (see ``copy_cache``): once uninstrumented (the whole-layer
     ground truth) and once seam by seam, re-running the submodules the real
     ``__call__`` runs, in the same order, with ``mx.eval`` of each seam's own
     outputs.  ``mx.eval()`` with no arguments is a no-op, so every barrier
     names its arrays.  The seam pass pays a sync per seam; the ratio
     seam-sum / uninstrumented is reported, not hidden.  The seam mirror is
     checked against the uninstrumented output (``mirror_max_abs_diff``) and
     refuses to run if the layer classes' ``__call__`` are not the ones it
     mirrors.

Cache copies never mutate the prefill state.  A GDN ``ArraysCache`` copy is a
new ``ArraysCache`` holding the same array references; the layer only rebinds
list entries (``cache[0] = ...``, ``cache[1] = ...``), so the live cache's
list is never touched.  A ``KVCache`` copy is a new ``KVCache`` with the same
``keys``/``values`` references and ``offset``; ``update_and_fetch`` either
grows the buffer (a concatenate, which rebinds only the copy's attributes and
leaves the original arrays untouched) or writes positions
``[offset, offset + L)`` of the shared buffer.  Every read of a KVCache is
bounded by its ``offset``, and the real chunk overwrites exactly those
positions next, so the live state (``offset`` and the valid prefix) is
unchanged.  With the production step (256) and chunk (2048) every chunk grows
the buffer, so the shared-buffer write never happens on the GPU path.  The
harness fingerprints each touched live cache before and after PART B and
fails closed if it moved.

Scaling (step 3).  Per-layer-kind seam times are multiplied by the layer
counts read from the model and integrated over every chunk start of the
prompt (linear interpolation between measured offsets, clamped at the ends),
giving whole-prefill shares by category and an integrated total that is
compared against PART A's measured total.

``--cpu-tiny`` runs the whole pipeline on a random-weight tiny model on the
CPU (``--tiny-arch dense`` = Qwen3.8 layout, ``moe`` = Qwen3.6 35B-A3B
layout); its report is ``evidence: false``.  Metal runs refuse without
``--i-own-the-gpu``; ``--dry-run`` prints the plan and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "mlx2.prefill-qwen-profile.v1"

GDN_SEAMS = (
    "input_norm", "gdn_in_proj", "gdn_conv1d_silu", "gdn_qk_norm",
    "gdn_recurrence", "gdn_out_norm_gate", "gdn_out_proj", "residual",
    "post_norm",
)
FA_SEAMS = (
    "input_norm", "fa_qkv_proj", "fa_kv_cache_update", "fa_sdpa", "fa_out_proj",
    "residual", "post_norm",
)
DENSE_MLP_SEAMS = ("mlp",)
MOE_SEAMS = ("moe_router", "moe_experts", "moe_shared", "moe_combine")

SEAM_CATEGORY = {
    "input_norm": "norms",
    "post_norm": "norms",
    "gdn_in_proj": "gdn_projections",
    "gdn_out_proj": "gdn_projections",
    "gdn_recurrence": "gdn_recurrence",
    "gdn_conv1d_silu": "gdn_other",
    "gdn_qk_norm": "gdn_other",
    "gdn_out_norm_gate": "gdn_other",
    "fa_qkv_proj": "attn_projections",
    "fa_out_proj": "attn_projections",
    "fa_sdpa": "attn_core",
    "fa_kv_cache_update": "attn_kv_update",
    "mlp": "mlp_dense",
    "moe_router": "moe_router",
    "moe_experts": "moe_experts",
    "moe_shared": "moe_shared",
    "moe_combine": "moe_shared",
    "residual": "other",
    "embed": "other",
}
CATEGORIES = (
    "gdn_projections", "gdn_recurrence", "gdn_other", "attn_projections",
    "attn_core", "attn_kv_update", "mlp_dense", "moe_router", "moe_experts",
    "moe_shared", "norms", "other",
)

TINY_DEFAULTS = {"context": 256, "chunk": 64}


# ------------------------------------------------------------------ helpers


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def default_offsets(context, chunk):
    raw = (0, context // 4, context // 2, context - chunk)
    return sorted({max(0, min(context - chunk, (o // chunk) * chunk)) for o in raw})


def parse_offsets(text, context, chunk):
    if not text:
        return default_offsets(context, chunk)
    out = []
    for item in text.split(","):
        if not item.strip():
            continue
        o = int(item)
        if o % chunk or not 0 <= o <= context - chunk:
            raise SystemExit(
                f"offset {o} must be a multiple of --chunk {chunk} in [0, {context - chunk}]"
            )
        out.append(o)
    return sorted(set(out))


def interp(x, xs, ys):
    """Piecewise-linear interpolation, clamped at both ends."""
    if len(xs) == 1 or x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def linear_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx_, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx_
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {
        "slope_ms_per_1k_offset": slope,
        "intercept_ms": intercept,
        "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n": n,
    }


def med(xs):
    return statistics.median(xs)


# ------------------------------------------------------------------ models


def tiny_model(arch="dense", seed=7):
    """Random-weight tiny model with a production layer layout (CPU only)."""
    import mlx.core as mx

    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    common = dict(
        hidden_size=64, intermediate_size=64, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=128, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=8192,
    )
    if arch == "dense":
        from mlx2.runtime.models.qwen38_27b import TextModel

        args = TextModelArgs(model_type="qwen3_5", **common)
    elif arch == "moe":
        from mlx2.runtime.models.qwen36_35b import TextModel

        args = TextModelArgs(
            model_type="qwen3_5_moe", num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=32, shared_expert_intermediate_size=32,
            **common,
        )
    else:
        raise ValueError(f"unknown tiny arch {arch!r}")
    mx.random.seed(seed)
    model = TextModel(args)
    model.eval()
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.v_proj.weight = attention.v_proj.weight * 8.0
            attention.o_proj.weight = attention.o_proj.weight * 8.0
    mx.eval(model.parameters())
    return model


def tiny_tokens(context, vocab=128, seed=11):
    rng = random.Random(seed)
    return [rng.randrange(1, vocab) for _ in range(context)]


def corpus_tokens(tokenizer, context):
    paths = sorted((ROOT / "docs").glob("*.md"))
    corpus = "\n\n".join(p.read_text(encoding="utf-8") for p in paths)
    try:
        ids = list(tokenizer.encode(corpus, add_special_tokens=False))
    except TypeError:
        ids = list(tokenizer.encode(corpus))
    if not ids:
        raise SystemExit("docs/*.md tokenized to nothing")
    stream = ids
    while len(stream) < context:
        stream = stream + ids
    return stream[:context], {
        "corpus": "docs/*.md joined by blank lines, repeated",
        "corpus_files": len(paths),
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "corpus_tokens": len(ids),
    }


def inner_model(model):
    inner = model.model
    if not hasattr(inner, "embed_tokens") or not hasattr(inner, "pipeline_layers"):
        raise SystemExit("model.model is not a Qwen3_5TextModel-shaped trunk")
    if getattr(inner, "pipeline_size", 1) != 1:
        raise SystemExit("pipeline-parallel trunks are not profiled")
    return inner


# ------------------------------------------------------------ mirror guard


def mirror_guard(inner):
    """The seam passes re-implement these ``__call__``s; refuse anything else.

    Returns a sha256 of each mirrored source so a report states which code it
    decomposed.
    """
    from mlx2.runtime.models import qwen3_5, qwen3_next, qwen36_35b, qwen38_27b

    allowed = {
        "decoder": {qwen38_27b.DecoderLayer.__call__, qwen36_35b.DecoderLayer.__call__},
        "gdn": {qwen3_5.GatedDeltaNet.__call__},
        "attn": {qwen38_27b.Qwen3NextAttention.__call__},
        "mlp": {qwen3_next.Qwen3NextMLP.__call__, qwen3_next.Qwen3NextSparseMoeBlock.__call__},
        "trunk": {qwen38_27b.Qwen3_5TextModel.__call__},
    }
    if type(inner).__call__ not in allowed["trunk"]:
        raise SystemExit(f"unmirrored trunk {type(inner).__name__}")
    for layer in inner.pipeline_layers:
        if type(layer).__call__ not in allowed["decoder"]:
            raise SystemExit(f"unmirrored decoder layer {type(layer).__name__}")
        branch = layer.linear_attn if layer.is_linear else layer.self_attn
        kind = "gdn" if layer.is_linear else "attn"
        if type(branch).__call__ not in allowed[kind]:
            raise SystemExit(f"unmirrored {kind} {type(branch).__name__}")
        if type(layer.mlp).__call__ not in allowed["mlp"]:
            raise SystemExit(f"unmirrored mlp {type(layer.mlp).__name__}")
    fns = {
        "qwen38_27b.DecoderLayer.__call__": qwen38_27b.DecoderLayer.__call__,
        "qwen36_35b.DecoderLayer.__call__": qwen36_35b.DecoderLayer.__call__,
        "qwen3_5.GatedDeltaNet.__call__": qwen3_5.GatedDeltaNet.__call__,
        "qwen38_27b.Qwen3NextAttention.__call__": qwen38_27b.Qwen3NextAttention.__call__,
        "qwen3_next.Qwen3NextSparseMoeBlock.__call__": qwen3_next.Qwen3NextSparseMoeBlock.__call__,
        "qwen3_next.Qwen3NextMLP.__call__": qwen3_next.Qwen3NextMLP.__call__,
        "qwen38_27b.Qwen3_5TextModel.__call__": qwen38_27b.Qwen3_5TextModel.__call__,
    }
    return {
        k: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()[:16]
        for k, f in fns.items()
    }


# ------------------------------------------------------------------ caches


def copy_cache(c):
    """A new cache object over the live cache's arrays (no data copied).

    See the module docstring for why updating the copy cannot change the
    live cache's valid state.
    """
    from mlx2.runtime.models.cache import ArraysCache, KVCache

    if type(c) is KVCache:
        n = KVCache()
        n.keys, n.values, n.offset = c.keys, c.values, c.offset
        return n
    if type(c) is ArraysCache:
        if c.lengths is not None or c.left_padding is not None or c.speculating:
            raise SystemExit("profiler expects an unpadded, non-speculating B=1 ArraysCache")
        n = ArraysCache(len(c.cache))
        n.cache = list(c.cache)
        return n
    raise SystemExit(f"unsupported cache type {type(c).__name__}")


def cache_fingerprint(mx, c):
    """Offset, array identity and valid-prefix checksums of one live cache."""
    from mlx2.runtime.models.cache import KVCache

    def digest(a):
        if a is None:
            return None
        f = a.astype(mx.float32)
        s, q = mx.sum(f), mx.sum(f * f)
        mx.eval(s, q)
        return (tuple(a.shape), str(a.dtype), float(s.item()), float(q.item()))

    if type(c) is KVCache:
        if c.keys is None:
            return ("kv", c.offset, None, None)
        return (
            "kv", c.offset,
            digest(c.keys[..., : c.offset, :]),
            digest(c.values[..., : c.offset, :]),
        )
    return ("arrays", tuple(id(a) for a in c.cache), tuple(digest(a) for a in c.cache))


# ------------------------------------------------------------- seam passes


def _timed(mx, rec, name, fn):
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    rec[name] = rec.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
    return out


def _mlp_seams(mx, mlp, xn, rec):
    from mlx2.runtime.models import qwen3_next as qn

    if isinstance(mlp, qn.Qwen3NextMLP):
        return _timed(mx, rec, "mlp", lambda: mlp(xn))
    moe = mlp
    if moe.sharding_group is not None:
        raise SystemExit("sharded MoE is not profiled")

    def router():
        gates = moe.gate(xn)
        fused = False
        if moe.moe_router_mode == "fused":
            adm = qn.admit_qwen4_moe_router(
                gates, top_k=moe.top_k, norm_topk_prob=bool(moe.norm_topk_prob)
            )
            if adm.accepted and qn.probe_qwen4_moe_router(gates.dtype):
                (inds, scores) = qn.qwen4_moe_router(gates)
                fused = True
        if not fused and (
            qn._MOE_GATE_COMPILE
            and gates.size // gates.shape[-1] <= qn._MOE_GATE_COMPILE_MAX_TOKENS
        ):
            (inds, scores) = qn._select_experts(gates, moe.top_k, bool(moe.norm_topk_prob))
        elif not fused:
            g = mx.softmax(gates, axis=-1, precise=True)
            k = moe.top_k
            inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(g, inds, axis=-1)
            if moe.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
        return [inds, scores]

    (inds, scores) = _timed(mx, rec, "moe_router", router)

    def experts():
        if moe.shared_folded:
            shared_col = mx.full(inds.shape[:-1] + (1,), moe.num_experts, dtype=inds.dtype)
            rows = moe.switch_mlp(xn, mx.concatenate([inds, shared_col], axis=-1))
            y = (rows[..., : moe.top_k, :] * scores[..., None]).sum(axis=-2)
            return [y, rows[..., moe.top_k, :]]
        if moe.fused_expert_kernel_enabled:
            y = moe.switch_mlp(xn, inds, scores=scores, variant=moe.fused_expert_kernel_mode)
        else:
            y = moe.switch_mlp(xn, inds)
            y = (y * scores[..., None]).sum(axis=-2)
        return [y]

    routed = _timed(mx, rec, "moe_experts", experts)
    y = routed[0]

    def shared():
        out = [qn.gate_sigmoid(moe.shared_expert_gate(xn))]
        if not moe.shared_folded:
            out.append(moe.shared_expert(xn))
        return out

    sh = _timed(mx, rec, "moe_shared", shared)
    gate = sh[0]
    shared_y = routed[1] if moe.shared_folded else sh[1]

    def combine():
        combined = None
        if qn._COMPILE_GLUE:
            combined = qn._run_glue(("moe_combine",), qn._build_moe_combine, y, gate, shared_y)
        return y + gate * shared_y if combined is None else combined

    return _timed(mx, rec, "moe_combine", combine)


def gdn_layer_seams(mx, nn, layer, x, mask, cache, rec):
    """Mirror of DecoderLayer.__call__ + GatedDeltaNet.__call__ (prefill path)."""
    gdn = layer.linear_attn
    if gdn.sharding_group is not None:
        raise SystemExit("sharded GDN is not profiled")
    xn = _timed(mx, rec, "input_norm", lambda: layer.input_layernorm(x))
    (B, S, _) = xn.shape
    (qkv, z, b, a) = _timed(mx, rec, "gdn_in_proj", lambda: list(gdn._input_projections(xn)))
    if gdn._try_fused_decode(qkv, z, b, a, mask, cache) is not None:
        raise SystemExit("a fused decode path engaged during prefill; mirror invalid")

    def conv():
        zz = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=xn.dtype)
        qkv_m = mx.where(mask[..., None], qkv, 0) if mask is not None else qkv
        conv_input = mx.concatenate([conv_state, qkv_m], axis=1)
        cache[0] = mx.contiguous(conv_input[:, -(gdn.conv_kernel_size - 1):, :])
        conv_out = nn.silu(gdn.conv1d(conv_input))
        (q, k, v) = [
            t.reshape(B, S, h, d)
            for (t, h, d) in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        return [q, k, v, zz, cache[0]]

    (q, k, v, zz, _) = _timed(mx, rec, "gdn_conv1d_silu", conv)
    state = cache[1]
    (q, k) = _timed(mx, rec, "gdn_qk_norm", lambda: list(gdn._normalize_qk(q, k)))
    (out, state) = _timed(
        mx, rec, "gdn_recurrence",
        lambda: list(gdn._gated_delta_update(q, k, v, a, b, state, mask, not gdn.training)),
    )
    cache[1] = state
    cache.advance(S)
    o = _timed(mx, rec, "gdn_out_norm_gate", lambda: gdn.norm(out, zz))
    r = _timed(mx, rec, "gdn_out_proj", lambda: gdn.out_proj(o.reshape(B, S, -1)))
    return _finish_layer(mx, layer, x, r, rec)


def fa_layer_seams(mx, layer, x, mask, cache, rec):
    """Mirror of DecoderLayer.__call__ + Qwen3NextAttention.__call__."""
    from mlx2.runtime.models import qwen38_27b as q38

    attn = layer.self_attn
    xn = _timed(mx, rec, "input_norm", lambda: layer.input_layernorm(x))
    (B, L, _) = xn.shape

    def qkv():
        q = attn.q_proj(xn)
        (queries, gate) = mx.split(q.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        (keys, values) = (attn.k_proj(xn), attn.v_proj(xn))
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        return [queries, keys, values, gate]

    (queries, keys, values, gate) = _timed(mx, rec, "fa_qkv_proj", qkv)
    (keys, values) = _timed(
        mx, rec, "fa_kv_cache_update", lambda: list(cache.update_and_fetch(keys, values))
    )
    o = _timed(
        mx, rec, "fa_sdpa",
        lambda: q38.scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=attn.scale, mask=mask
        ),
    )
    r = _timed(
        mx, rec, "fa_out_proj",
        lambda: attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1) * q38.gate_sigmoid(gate)),
    )
    return _finish_layer(mx, layer, x, r, rec)


def _finish_layer(mx, layer, x, r, rec):
    h = _timed(mx, rec, "residual", lambda: x + r)
    hn = _timed(mx, rec, "post_norm", lambda: layer.post_attention_layernorm(h))
    m = _mlp_seams(mx, layer.mlp, hn, rec)
    return _timed(mx, rec, "residual", lambda: h + m)


# ------------------------------------------------------------- PART B probe


def masks_for(inner, h, caches):
    from mlx2.runtime.models.qwen38_27b import create_attention_mask, create_ssm_mask

    fa = create_attention_mask(h, caches[inner.fa_idx]) if inner.fa_idx is not None else None
    ssm = create_ssm_mask(h, caches[inner.ssm_idx]) if inner.ssm_idx is not None else None
    return fa, ssm


def capture_inputs(mx, inner, caches, tokens, targets):
    """Embedding + layers below max(targets) on copies of the live caches."""
    layers = inner.pipeline_layers
    top = max(targets)
    h = inner.embed_tokens(tokens)
    (fa_mask, ssm_mask) = masks_for(inner, h, caches)
    out = {}
    for i in range(top + 1):
        if i in targets:
            out[i] = h
        if i == top:
            break
        layer = layers[i]
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=copy_cache(caches[i]))
    mx.eval(list(out.values()))
    return out, fa_mask, ssm_mask


def time_layer(mx, nn, layer, x, mask, live_cache, warmup, repeats):
    """Uninstrumented and seam-instrumented timings of one real layer."""
    whole = []
    last_whole = None
    for i in range(warmup + repeats):
        c = copy_cache(live_cache)
        t0 = time.perf_counter()
        y = layer(x, mask=mask, cache=c)
        mx.eval(y)
        dt = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            whole.append(dt)
        last_whole = y
    seam_samples = []
    last_seam = None
    for i in range(warmup + repeats):
        c = copy_cache(live_cache)
        rec = {}
        if layer.is_linear:
            y = gdn_layer_seams(mx, nn, layer, x, mask, c, rec)
        else:
            y = fa_layer_seams(mx, layer, x, mask, c, rec)
        if i >= warmup:
            seam_samples.append(rec)
        last_seam = y
    diff = float(mx.max(mx.abs(last_whole.astype(mx.float32) - last_seam.astype(mx.float32))).item())
    seams = {k: med([s[k] for s in seam_samples]) for k in seam_samples[0]}
    seam_sum = sum(seams.values())
    whole_ms = med(whole)
    return {
        "whole_layer_ms": whole_ms,
        "whole_layer_min_ms": min(whole),
        "whole_layer_max_ms": max(whole),
        "seams_ms": seams,
        "seam_share": {k: v / seam_sum for k, v in seams.items()},
        "seam_sum_ms": seam_sum,
        "seam_sum_over_whole": seam_sum / whole_ms,
        "mirror_max_abs_diff": diff,
        "n_samples": repeats,
    }


class _Tap:
    """Pass-through wrapper recording a layer's real input during PART A."""

    def __init__(self, layer, sink, index):
        self.layer, self.sink, self.index = layer, sink, index
        self.is_linear = layer.is_linear

    def __call__(self, x, mask=None, cache=None):
        self.sink[self.index] = x
        return self.layer(x, mask=mask, cache=cache)


def probe(mx, nn, inner, caches, tokens, offset, targets, warmup, repeats):
    touched = list(range(max(targets.values()) + 1))
    before = [cache_fingerprint(mx, caches[i]) for i in touched]
    t0 = time.perf_counter()
    embed = []
    for i in range(warmup + repeats):
        s = time.perf_counter()
        e = inner.embed_tokens(tokens)
        mx.eval(e)
        if i >= warmup:
            embed.append((time.perf_counter() - s) * 1e3)
    captured, fa_mask, ssm_mask = capture_inputs(mx, inner, caches, tokens, set(targets.values()))
    layers = inner.pipeline_layers
    result = {"offset": offset, "embed_ms": med(embed)}
    for kind, idx in targets.items():
        layer = layers[idx]
        mask = ssm_mask if layer.is_linear else fa_mask
        r = time_layer(mx, nn, layer, captured[idx], mask, caches[idx], warmup, repeats)
        r["layer_index"] = idx
        result[kind] = r
        emit("probe_layer", offset=offset, kind=kind, layer=idx,
             whole_ms=round(r["whole_layer_ms"], 3), seam_sum_ms=round(r["seam_sum_ms"], 3),
             seam_over_whole=round(r["seam_sum_over_whole"], 3))
    after = [cache_fingerprint(mx, caches[i]) for i in touched]
    result["cache_unmutated"] = before == after
    result["probe_wall_s"] = time.perf_counter() - t0
    if not result["cache_unmutated"]:
        raise SystemExit(f"PART B mutated the live prefill cache at offset {offset}; run invalid")
    return result, captured


# ----------------------------------------------------------------- scaling


def scale_to_prefill(probes, counts, context, chunk, measured_total_ms):
    offsets = [p["offset"] for p in probes]
    starts = list(range(0, context, chunk))
    cats = {c: 0.0 for c in CATEGORIES}
    seams_total = {}
    est_whole = 0.0
    for s in starts:
        for kind in ("gdn", "fa"):
            n = counts[kind]
            names = probes[0][kind]["seams_ms"].keys()
            for name in names:
                v = n * interp(s, offsets, [p[kind]["seams_ms"][name] for p in probes])
                cats[SEAM_CATEGORY[name]] += v
                key = f"{kind}.{name}"
                seams_total[key] = seams_total.get(key, 0.0) + v
            est_whole += n * interp(s, offsets, [p[kind]["whole_layer_ms"] for p in probes])
        e = interp(s, offsets, [p["embed_ms"] for p in probes])
        cats["other"] += e
        seams_total["embed"] = seams_total.get("embed", 0.0) + e
        est_whole += e
    est_seam = sum(cats.values())
    shares = {c: v / est_seam for c, v in cats.items()}
    per_offset = []
    for p in probes:
        whole = counts["gdn"] * p["gdn"]["whole_layer_ms"] + counts["fa"] * p["fa"]["whole_layer_ms"] + p["embed_ms"]
        seam = counts["gdn"] * p["gdn"]["seam_sum_ms"] + counts["fa"] * p["fa"]["seam_sum_ms"] + p["embed_ms"]
        per_offset.append({"offset": p["offset"], "chunk_estimate_from_whole_layers_ms": whole,
                           "chunk_estimate_from_seams_ms": seam,
                           "attn_core_share_of_chunk": counts["fa"] * p["fa"]["seams_ms"]["fa_sdpa"] / seam})
    return {
        "layer_counts": counts,
        "method": ("per-kind seam ms x layer count, linearly interpolated between measured "
                   "offsets at every chunk start, summed over the prompt; embedding in 'other'"),
        "integrated_seam_total_ms": est_seam,
        "integrated_whole_layer_total_ms": est_whole,
        "measured_part_a_total_ms": measured_total_ms,
        "ratio_whole_layer_estimate_over_measured": est_whole / measured_total_ms,
        "ratio_seam_estimate_over_measured": est_seam / measured_total_ms,
        "category_ms": cats,
        "category_share": shares,
        "category_ms_scaled_to_measured": {c: shares[c] * measured_total_ms for c in CATEGORIES},
        "seam_ms_integrated": seams_total,
        "per_offset_chunk": per_offset,
    }


# -------------------------------------------------------------------- core


def run_profile(model, token_ids, *, context, chunk, offsets, repeats, warmup,
                gdn_layer=None, fa_layer=None, warm_chunk=True):
    """PART A + PART B + scaling. Returns (report_fragment, final_cache)."""
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models.cache import make_prompt_cache

    inner = inner_model(model)
    guard = mirror_guard(inner)
    layers = inner.pipeline_layers
    gdn_layer = inner.ssm_idx if gdn_layer is None else gdn_layer
    fa_layer = inner.fa_idx if fa_layer is None else fa_layer
    if not layers[gdn_layer].is_linear or layers[fa_layer].is_linear:
        raise SystemExit("--gdn-layer must be a linear layer and --fa-layer a full-attention layer")
    targets = {"gdn": gdn_layer, "fa": fa_layer}
    counts = {"gdn": sum(1 for l in layers if l.is_linear),
              "fa": sum(1 for l in layers if not l.is_linear)}
    tokens = mx.array(token_ids[:context], dtype=mx.int32)[None]
    if context % chunk:
        raise SystemExit("--context must be a multiple of --chunk")

    if warm_chunk:
        wc = make_prompt_cache(model)
        model(tokens[:, :chunk], cache=wc)
        mx.eval([c.state for c in wc])
        del wc
        mx.clear_cache()

    cache = make_prompt_cache(model)
    probe_at = set(offsets)
    chunks, probes, capture_diffs = [], [], []
    wall0 = time.perf_counter()
    for start in range(0, context, chunk):
        toks = tokens[:, start:start + chunk]
        taps = {}
        probed = start in probe_at
        if probed:
            p, captured = probe(mx, nn, inner, cache, toks, start, targets, warmup, repeats)
            probes.append(p)
            mx.clear_cache()
            saved = {i: inner.layers[i] for i in targets.values()}
            for i in targets.values():
                inner.layers[i] = _Tap(saved[i], taps, i)
        try:
            t0 = time.perf_counter()
            model(toks, cache=cache)
            mx.eval([c.state for c in cache])
            ms = (time.perf_counter() - t0) * 1e3
        finally:
            if probed:
                for i, layer in saved.items():
                    inner.layers[i] = layer
        if probed:
            d = max(float(mx.max(mx.abs(taps[i].astype(mx.float32) - captured[i].astype(mx.float32))).item())
                    for i in targets.values())
            p["capture_max_abs_diff"] = d
            capture_diffs.append(d)
            del taps, captured
        mx.clear_cache()
        chunks.append({"start": start, "ms": ms, "after_probe": probed})
        emit("chunk", start=start, ms=round(ms, 3), after_probe=probed)
    wall_s = time.perf_counter() - wall0

    total_ms = sum(c["ms"] for c in chunks)
    ms_list = [c["ms"] for c in chunks]
    clean = [c for c in chunks if not c["after_probe"]]
    fit_all = linear_fit([c["start"] / 1024 for c in chunks], ms_list)
    fit_clean = linear_fit([c["start"] / 1024 for c in clean], [c["ms"] for c in clean])
    part_a = {
        "total_prefill_s": total_ms / 1e3,
        "tokens_per_s": context / (total_ms / 1e3),
        "wall_s_including_probes": wall_s,
        "chunk_ms_first": ms_list[0],
        "chunk_ms_median": med(ms_list),
        "chunk_ms_last": ms_list[-1],
        "fit_all_chunks": fit_all,
        "fit_chunks_not_after_probe": fit_clean,
        "chunks": chunks,
        "warm_chunk_before_timing": warm_chunk,
        "note": ("chunk ms = model(chunk) + mx.eval(cache states), the production barrier; "
                 "mx.clear_cache() after each chunk as production does, outside the timer"),
    }
    part_b = {
        "targets": targets,
        "offsets": offsets,
        "repeats": repeats,
        "warmup": warmup,
        "capture": ("embed_tokens + layers below the target on copies of the live caches; "
                    "verified against a pass-through tap on the real chunk"),
        "probes": probes,
    }
    scaling = scale_to_prefill(probes, counts, context, chunk, total_ms)
    checks = {
        "cache_unmutated": all(p["cache_unmutated"] for p in probes),
        "capture_max_abs_diff": max(capture_diffs) if capture_diffs else None,
        "mirror_max_abs_diff": max(max(p["gdn"]["mirror_max_abs_diff"], p["fa"]["mirror_max_abs_diff"])
                                   for p in probes),
        "ratio_whole_layer_estimate_over_measured": scaling["ratio_whole_layer_estimate_over_measured"],
    }
    moe = getattr(layers[0].mlp, "num_experts", None)
    arch = {
        "layers": len(layers),
        "moe": moe is not None,
        "num_experts": moe,
        "top_k": getattr(layers[0].mlp, "top_k", None),
        "moe_shared_folded": getattr(layers[0].mlp, "shared_folded", None),
        "moe_fused_expert_mode": getattr(layers[0].mlp, "fused_expert_kernel_mode", None),
        "moe_router_mode": getattr(layers[0].mlp, "moe_router_mode", None),
        "param_dtypes": sorted({str(v.dtype) for _, v in _flat_params(layers[fa_layer])}),
    }
    return {"architecture": arch, "mirror_source_sha256": guard, "part_a": part_a,
            "part_b": part_b, "scaling": scaling, "checks": checks}, cache


def _flat_params(module):
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


# -------------------------------------------------------------------- CLI


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path)
    src.add_argument("--cpu-tiny", action="store_true")
    p.add_argument("--tiny-arch", choices=("dense", "moe"), default="dense")
    p.add_argument("--context", type=int, default=None, help="default 131072 (256 with --cpu-tiny)")
    p.add_argument("--chunk", type=int, default=None, help="default 2048 (64 with --cpu-tiny)")
    p.add_argument("--offsets", default="")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--gdn-layer", type=int, default=None)
    p.add_argument("--fa-layer", type=int, default=None)
    p.add_argument("--no-warm-chunk", action="store_true")
    p.add_argument("--cache-limit-gb", type=float, default=4.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    return p


def run(argv=None, *, return_cache=False):
    a = build_parser().parse_args(argv)
    a.context = a.context or (TINY_DEFAULTS["context"] if a.cpu_tiny else 131072)
    a.chunk = a.chunk or (TINY_DEFAULTS["chunk"] if a.cpu_tiny else 2048)
    if a.context % a.chunk:
        raise SystemExit("--context must be a multiple of --chunk")
    offsets = parse_offsets(a.offsets, a.context, a.chunk)
    plan = {"schema": SCHEMA, "model": str(a.model) if a.model else f"cpu-tiny-{a.tiny_arch}",
            "context": a.context, "chunk": a.chunk, "offsets": offsets,
            "repeats": a.repeats, "warmup": a.warmup, "cache_limit_gb": a.cache_limit_gb}
    if a.model and (a.model / "config.json").exists():
        cfg = json.loads((a.model / "config.json").read_text())
        t = cfg.get("text_config", cfg)
        plan["config"] = {k: t.get(k) for k in (
            "model_type", "num_hidden_layers", "hidden_size", "full_attention_interval",
            "num_experts", "num_experts_per_tok", "head_dim", "num_attention_heads",
            "num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads")}
        plan["config"]["model_type"] = cfg.get("model_type")
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return plan
    if not a.cpu_tiny and not a.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    mx.set_cache_limit(int(a.cache_limit_gb * 1e9))
    identity, environment, token_meta = {}, None, {}
    if a.cpu_tiny:
        mx.set_default_device(mx.cpu)
        model = tiny_model(a.tiny_arch)
        ids = tiny_tokens(a.context)
        identity = {"model": f"cpu-tiny-{a.tiny_arch}"}
        token_meta = {"corpus": "random ids, seed 11"}
    else:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise SystemExit("this harness measures Metal; no GPU available")
        from mlx2.adapters.registry import resolve_adapter

        # The adapter pins the serving environment before importing any model
        # module, so model code below runs with production gates.
        adapter = resolve_adapter(a.model)(str(a.model))
        model = adapter.model
        environment = getattr(adapter, "environment", None)
        identity = {"model": str(a.model), "adapter": type(adapter).__name__,
                    "fingerprint": getattr(adapter, "identity", {}).get("fingerprint")}
        ids, token_meta = corpus_tokens(adapter.tokenizer, a.context)
    token_meta["token_stream_sha256"] = hashlib.sha256(
        json.dumps(ids[: a.context]).encode()).hexdigest()

    emit("start", **plan)
    body, cache = run_profile(
        model, ids, context=a.context, chunk=a.chunk, offsets=offsets,
        repeats=a.repeats, warmup=a.warmup, gdn_layer=a.gdn_layer,
        fa_layer=a.fa_layer, warm_chunk=not a.no_warm_chunk,
    )
    report = {
        "schema": SCHEMA,
        "evidence": not a.cpu_tiny,
        "qualification": False,
        "plan": plan,
        "identity": identity,
        "serving_environment": environment,
        "tokens": token_meta,
        "device": str(mx.default_device()),
        "mlx_version": getattr(mx, "__version__", None),
        "platform": platform.platform(),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **body,
    }
    s = report["scaling"]
    emit("summary", total_prefill_s=round(report["part_a"]["total_prefill_s"], 3),
         tokens_per_s=round(report["part_a"]["tokens_per_s"], 1),
         ratio_whole_over_measured=round(s["ratio_whole_layer_estimate_over_measured"], 3),
         shares={k: round(v, 4) for k, v in s["category_share"].items()})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return (report, cache) if return_cache else report


def main():
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
