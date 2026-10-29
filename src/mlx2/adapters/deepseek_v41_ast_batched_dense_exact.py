# SPDX-License-Identifier: MIT
"""Candidate batched stock-M1 dense law without Python token-row calls.

MXFP8 uses zero-stride coefficient batch views and x[B,1,K]. Native F32
matmul would collapse contiguous batches back to M, so its activation rows
have a bounded K+1 stride. All original coefficients remain unchanged and
no coefficient bank is copied. WoA uses the unchanged custom M1 dot helper
with independent grid.y rows. Actual installed-MLX GPU parity is required;
this module's name describes the intended law, not an established claim.
Provenance: deepseek-v41-source-ast-m1-batch-axis.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

PROFILE = "source_ast_m1_batch_axis_dense_candidate_v1"


def plan(rows=16):
    if type(rows) is not int or not 1 <= rows <= 16:
        raise ValueError("Source AST M1 batch-axis dense requires rows1..16")
    return dict(profile=PROFILE, physical_rows=rows, native_matrix_rows=1,
        native_batches=rows, python_token_row_calls=0, coefficient_repack_bytes=0,
        coefficient_batch_stride=0, F32_activation_batch_stride="K+1",
        intended_dot_law="stock M1 private row reduction", exact_M1_claim=False,
        explicit_onchip_weight_reuse=False, reuse="one coefficient owner and GPU cache",
        qualified=False, default_selected=False)


def _rows(x, *, ndim=3, width=None):
    from .deepseek_v41_ast_batched_dense import _rows as validate
    return validate(x, ndim=ndim, width=width)


def _record(model, category, rows):
    counts = getattr(model, "source_ast_m1_batch_axis_dense_calls", None)
    if counts is None:
        counts = {}; model.source_ast_m1_batch_axis_dense_calls = counts
    counts[category+"_tiles"] = counts.get(category+"_tiles", 0)+1
    counts[category+"_rows"] = counts.get(category+"_rows", 0)+rows


def _float_batch_matmul(x, weight):
    """One native GEMV batch; padding is activation-only, at most16 floats."""
    import mlx.core as mx
    rows, k = x.shape[1], x.shape[2]
    if weight.ndim != 2 or weight.shape[1] != k:
        raise ValueError("Source AST M1 float coefficient geometry mismatch")
    flat = x.astype(mx.float32).reshape(rows, k)
    storage = mx.concatenate((flat, mx.zeros((rows, 1), mx.float32)), axis=1)
    # Maximum referenced element=(rows-1)*(k+1)+(k-1)<rows*(k+1).
    # Inner matrix stride remainsK; leading batch strideK+1 defeats the native
    # collapse guard (A_batch_stride == M*K) without copying coefficients.
    vectors = mx.as_strided(storage, (rows, 1, k), (k+1, k, 1))
    coefficients = weight.astype(mx.float32)
    batch = mx.broadcast_to(coefficients[None], (rows, *coefficients.shape))
    output = mx.matmul(vectors, mx.swapaxes(batch, -1, -2))
    return output.reshape(1, rows, weight.shape[0])


def _quant_batch_matmul(prepared, q, scales):
    """w.ndim=3 prevents native quantized dispatch flattening B into M."""
    import mlx.core as mx
    rows, k = prepared.shape[1], prepared.shape[2]
    if (q.ndim != 2 or q.dtype != mx.uint32 or q.shape[1]*4 != k
            or scales.shape != (q.shape[0], k//32) or scales.dtype != mx.uint8):
        raise ValueError("Source AST M1 batch-axis requires original MXFP8 group32")
    vectors = prepared.reshape(rows, 1, k)
    packed = mx.broadcast_to(q[None], (rows, *q.shape))
    block_scales = mx.broadcast_to(scales[None], (rows, *scales.shape))
    output = mx.quantized_matmul(vectors, packed, scales=block_scales, mode="mxfp8")
    return output.reshape(1, rows, q.shape[0])


def _raw(model, x, name, *, prepared=None):
    import mlx.core as mx
    weights = model.weights
    weights._ready(); weights._ensure_dense()
    quant = getattr(weights, "_quant", {})
    if name in quant:
        q, scales = quant[name]
        prepared = weights._dense_input(x) if prepared is None else prepared
        result = _quant_batch_matmul(prepared, q, scales)
        counters = weights.scheduled_dense_calls
        counters['source_ast_m1_batch_axis_mxfp8'] = counters.get('source_ast_m1_batch_axis_mxfp8', 0)+1
        return result
    weight = weights.array(name)
    if (name == "output.weight" and weight.dtype == mx.bfloat16
            and getattr(weights, "source_bf16_loads_policy", "reference") == "direct_f32"):
        from ..runtime.gguf.source_ast_m1_dense_rows import project
        return project(x, weight, "head")
    return _float_batch_matmul(x, weight)


def linear_tile(model, x, name, rounded=True):
    from .deepseek_v41_ast_batched_dense import _name, _output
    rows = _rows(x); _name(name)
    if type(rounded) is not bool:raise ValueError("Source AST M1 rounded flag must be boolean")
    raw = _raw(model, x, name)
    value = model.round(raw) if rounded else raw.astype(x.dtype)
    value = _output(value, x)
    _record(model, "linear", rows)
    return value


def paired_linear_tile(model, x, a, b, rounded=True):
    from .deepseek_v41_ast_batched_dense import _name
    rows = _rows(x); _name(a); _name(b)
    if a == b or type(rounded) is not bool:raise ValueError("Source AST M1 pair geometry mismatch")
    weights = model.weights
    reuse = (getattr(weights, "source_dense_qat_policy", "reference") == "fused_fp32"
             and a in getattr(weights, "_quant", {}) and b in weights._quant)
    prepared = weights._dense_input(x) if reuse else None
    first, second = _raw(model, x, a, prepared=prepared), _raw(model, x, b, prepared=prepared)
    if rounded:first, second = model.round(first), model.round(second)
    _record(model, "paired", rows)
    return first, second


def grouped_tile(model, x, name, groups=8):
    import mlx.core as mx
    rows = _rows(x, ndim=4)
    if (groups != 8 or x.shape[2:] != (8, 4096)
            or getattr(model.weights, "source_bf16_loads_policy", "reference") not in ("woa_f32", "direct_f32")):
        raise ValueError("Source AST M1 grouped candidate requires current Source WoA M1 profile")
    from ..runtime.gguf.source_ast_m1_dense_rows import project
    value = project(x, model.weights.array(name), "wo_a")
    _record(model, "grouped", rows)
    return value


def shared_expert_tile(block, x):
    from ..runtime.models.deepseek_v4 import limited_swiglu
    model, weights = block.model, block.model.weights
    rows = _rows(x, width=model.args.hidden_size)
    g, u = block.prefix+"ffn_gate_shexp.weight", block.prefix+"ffn_up_shexp.weight"
    if (getattr(weights, "shared_policy", "reference") == "fused"
            and weights.can_fuse_dense(x, g, u)):
        raise ValueError("Source AST M1 shared batch does not replace the fused shared dot profile")
    gate, up = paired_linear_tile(model, x, g, u, rounded=True)
    middle = model.round(limited_swiglu(gate, up, model.args.swiglu_limit))
    output = linear_tile(model, middle, block.prefix+"ffn_down_shexp.weight", rounded=True)
    _record(model, "shared_expert", rows)
    return output


def router_tile(block, x):
    """Exact-candidate projection followed by the existing router expression."""
    import mlx.core as mx
    from ..runtime.models.deepseek_v4 import route_experts
    model, c = block.model, block.c
    rows = _rows(x, width=c.hidden_size)
    route = route_experts
    if getattr(model, "pointwise_policy", "reference") == "fused":
        from ..runtime.models.deepseek_v41_fused import router, self_check
        self_check(); route = router
    logits = linear_tile(model, x, block.prefix+"ffn_gate_inp.weight", rounded=False)
    ids, scores = route(logits, top_k=c.num_experts_per_tok,
        scale=c.routed_scaling_factor, correction=block.correction, normalize=c.norm_topk_prob)
    if (ids.shape != (1, rows, c.num_experts_per_tok) or scores.shape != ids.shape
            or ids.dtype != mx.uint32 or scores.dtype != mx.float32):
        raise ValueError("Source AST M1 router occurrence geometry changed")
    _record(model, "router", rows)
    return ids, scores


def engram_tile(model, h, layer, rows):
    """Prefetched grouped lookup, M1-law projection and original Engram gate."""
    import mlx.core as mx
    c = model.args
    count = _rows(h, ndim=4, width=c.hidden_size)
    if (h.shape[2] != c.hc_mult or type(layer) is not int
            or not 0 <= layer < c.num_hidden_layers or layer not in c.engram_layer_ids
            or not isinstance(rows, mx.array) or rows.ndim < 2 or not rows.size
            or rows.size % count or not (rows.shape[0] == count or rows.shape[:2] == (1, count))
            or rows.dtype not in (mx.float32, mx.float16, mx.bfloat16)):
        raise ValueError("Source AST M1 Engram decoded row geometry mismatch")
    p = f"blk.{layer}."
    kv = linear_tile(model, rows.reshape(1, count, -1).astype(mx.float32), p+"engram_kv.weight")
    if kv.shape[-1] != (c.hc_mult+1)*c.hidden_size:
        raise ValueError("Source AST M1 Engram key/value extent mismatch")
    key = kv[..., :c.hc_mult*c.hidden_size].reshape(h.shape)
    value = kv[..., c.hc_mult*c.hidden_size:]
    qweight = model.weights.array(p+"engram_q_norm.weight")
    kweight = model.weights.array(p+"engram_k_norm.weight")
    rstd = mx.rsqrt(mx.mean(mx.square(h), -1)+c.rms_norm_eps)*mx.rsqrt(
        mx.mean(mx.square(key), -1)+c.rms_norm_eps)
    dot = mx.sum(h*(qweight*kweight)*key, -1)*rstd*c.hidden_size**-0.5
    mag = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
    gate = mx.sigmoid(mx.where((dot.view(mx.uint32) >> 31) != 0, -mag, mag))
    result = model.round(h+gate[..., None]*value[..., None, :])
    if result.shape != h.shape or result.dtype != mx.float32:
        raise ValueError("Source AST M1 Engram hidden geometry changed")
    _record(model, "engram", count)
    return result


def mixes_tile(block, h, kind):
    """M1 batched HC projection, then unchanged post-normalization/split law.

    In particular, normalize AFTER the projection as Source does. Source's
    coefficient split remains separate from the residual HC expansion policy.
    """
    import mlx.core as mx
    from ..runtime.models.deepseek_v4 import _sinkhorn_mix_fp32
    c, model = block.c, block.model
    rows = _rows(h, ndim=4, width=c.hidden_size)
    if h.shape[2] != c.hc_mult or kind not in ("attn", "ffn"):
        raise ValueError("Source AST M1 HC kind/hidden geometry mismatch")
    fn, scale, base = block.hc[kind]
    flat = h.reshape(*h.shape[:2], -1).astype(mx.float32)
    mixes = _float_batch_matmul(flat, fn)*mx.rsqrt(
        mx.mean(mx.square(flat), -1, keepdims=True)+c.rms_norm_eps)
    _record(model, "mixes", rows)
    if getattr(model, "hc_coeff_policy", "reference") == "source_fused_fp32":
        if c.hc_mult == 4:
            from ..runtime.models.deepseek_v41_source_hc import split
            result = split(mixes, scale, base, iterations=c.hc_sinkhorn_iters, eps=c.hc_eps)
            counts = model.scheduled_hc_calls
            counts["source_fused_fp32"] = counts.get("source_fused_fp32", 0)+1
            return result
        result = _sinkhorn_mix_fp32(mixes, scale, base, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps)
        counts = model.scheduled_hc_calls
        counts["reference_fallback"] = counts.get("reference_fallback", 0)+1
        return result
    if model.hc_policy == "fused_fp32":
        from ..runtime.models.deepseek_v4_hc import split_sinkhorn
        return split_sinkhorn(mixes, scale, base, iterations=c.hc_sinkhorn_iters, eps=c.hc_eps)
    return _sinkhorn_mix_fp32(mixes, scale, base, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps)


def head_tile(model, h, pre):
    import mlx.core as mx
    rows = _rows(h, ndim=4, width=model.args.hidden_size)
    if (h.shape[2] != model.args.hc_mult or not isinstance(pre, mx.array)
            or pre.shape != h.shape[:-1] or pre.dtype != mx.float32):
        raise ValueError("Source AST M1 head HC geometry mismatch")
    hidden = model.norm(model.collapse(h, pre), model.norm_weight)
    logits = linear_tile(model, hidden, "output.weight", rounded=False)
    if logits.shape != (1, rows, model.args.vocab_size):
        raise ValueError("Source AST M1 head must publish whole vocabulary")
    _record(model, "head", rows)
    return logits


def scheduled_receipt(model):
    return dict(profile=PROFILE, scheduled=dict(getattr(model, "source_ast_m1_batch_axis_dense_calls", {})),
        exact_M1_claim=False, observation="scheduled only; actual installed GPU raw gate required")


@lru_cache(maxsize=1)
def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    paths = (Path(__file__), root/"runtime/gguf/source_ast_m1_dense_rows.py",
             root/"runtime/gguf/source_dense.py", root/"adapters/deepseek_v41_source_weights.py",
             root/"runtime/models/deepseek_v41_source_hc.py")
    return hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
