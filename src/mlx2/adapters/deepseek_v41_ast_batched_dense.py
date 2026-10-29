# SPDX-License-Identifier: MIT
"""Explicit native-M AST dense stage, without Python token-row dispatch.

Native M4/M16 products can have a different reduction law from independent M1
products. The owning AST profile must say so and retain ordinary controls.
No coefficient formats, BF16 owner stores, QAT policy or cache state change.
The caller owns all weight scopes, evaluation, live checks and GPU fences.
Provenance: provenance/deepseek-v41-source-ast-batched-dense.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

PROFILE = "source_ast_native_batched_dense_v1"
MAX_ROWS = 16


def plan(rows=16):
    """Pure CPU inventory; this does not compile, select or qualify a route."""
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("Source AST dense tile requires physical rows1..16")
    return dict(profile=PROFILE, physical_rows=rows, python_token_row_calls=0,
                linear_projection_products=1, shared_gate_up_products=2,
                shared_down_products=1, head_projection_products=1,
                input_shape="[1,M,K]", grouped_shape="[1,M,G,K]",
                hidden_shape="[1,M,HC,H]", coefficient_formats="existing adapter formats",
                activation_QAT="existing adapter whole-tile QAT, row-local blocks",
                output_boundaries="existing owner BF16 stores, F32 head logits",
                reduction_law="native row-tile dispatch; distinct from M1 control",
                exact_M1_claim=False, qualified=False, default_selected=False)


def _rows(value, *, ndim=3, width=None):
    import mlx.core as mx
    if (not isinstance(value, mx.array) or value.ndim != ndim or value.shape[0] != 1
            or not 1 <= value.shape[1] <= MAX_ROWS or value.shape[-1] <= 0
            or value.dtype != mx.float32 or (width is not None and value.shape[-1] != width)):
        raise ValueError("Source AST dense requires physical F32 B1 row tile1..16")
    return value.shape[1]


def _name(name):
    if type(name) is not str or not name.endswith(".weight") or not name:
        raise ValueError("Source AST dense requires an exact weight name")


def _output(value, inputs, *, ndim=3, width=None, groups=None):
    import mlx.core as mx
    if (not isinstance(value, mx.array) or value.ndim != ndim or value.shape[:2] != inputs.shape[:2]
            or value.shape[-1] <= 0 or value.dtype != mx.float32
            or (width is not None and value.shape[-1] != width)
            or (groups is not None and value.shape[2] != groups)):
        raise ValueError("Source AST batched dense consumer changed physical row geometry")
    return value


def _record(model, category, rows):
    # Bounded integer-only scheduling observations, never GPU-completion claims.
    counts = getattr(model, "source_ast_batched_dense_calls", None)
    if counts is None:
        counts = {}
        model.source_ast_batched_dense_calls = counts
    counts[category + "_tiles"] = counts.get(category + "_tiles", 0) + 1
    counts[category + "_rows"] = counts.get(category + "_rows", 0) + rows


def linear_tile(model, x, name, rounded=True):
    """One adapter projection over all M rows, including Qa/KV/Qb/WoB.

    Original packed MXFP8/scales and existing row-local input QAT are consumed
    by the adapter. BF16/F32 dense tensors retain their existing representation.
    The existing model owner decides the explicit output BF16 store.
    """
    rows = _rows(x)
    _name(name)
    if type(rounded) is not bool:
        raise ValueError("Source AST dense rounded flag must be boolean")
    value = _output(model.linear(x, name, rounded=rounded), x)
    _record(model, "linear", rows)
    return value


def paired_linear_tile(model, x, a, b, rounded=True):
    """Qa/KV pair: one tile QAT preparation when the existing adapter admits it.

    The fixed two projections are not a loop over token rows. The fallback also
    performs two complete row-tile projections, never M independent M1 calls.
    """
    rows = _rows(x)
    _name(a); _name(b)
    if a == b or type(rounded) is not bool:
        raise ValueError("Source AST dense pair requires distinct names/boolean store")
    if model.can_pair(x, a, b):
        values = model.weights.paired_project(x, a, b, rounded=rounded)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("Source AST dense pair must publish two row tiles")
        first, second = _output(values[0], x), _output(values[1], x)
        _record(model, "paired_QAT_reuse", rows)
    else:
        first = linear_tile(model, x, a, rounded=rounded)
        second = linear_tile(model, x, b, rounded=rounded)
        _record(model, "paired_separate", rows)
    return first, second


def grouped_tile(model, x, name, groups=8):
    """One grouped WoA row tile; return raw F32 like weights.grouped.

    The caller retains its existing model.round owner immediately after this
    stage. M16 fallback is native grouped F32 einsum; existing admitted smaller
    custom shapes remain adapter-owned. Neither is relabelled exact M1 math.
    """
    rows = _rows(x, ndim=4)
    _name(name)
    if type(groups) is not int or groups <= 0 or x.shape[2] != groups:
        raise ValueError("Source AST grouped dense group geometry mismatch")
    value = model.weights.grouped(x, name, groups=groups)
    value = _output(value, x, ndim=4, groups=groups)
    _record(model, "grouped", rows)
    return value


def shared_expert_tile(block, x):
    """One real M-row shared expert, including batched pointwise owner stores."""
    model = block.model
    rows = _rows(x, width=model.args.hidden_size)
    # Existing Block.shared_expert performs two whole-tile gate/up products,
    # BF16 stores, limited SwiGLU and one down product. No per-token call loop.
    value = _output(block.shared_expert(x), x, width=model.args.hidden_size)
    _record(model, "shared_expert", rows)
    return value


def router_tile(block, x):
    """One batched router projection and the original row-local selection law."""
    import mlx.core as mx
    from ..runtime.models.deepseek_v4 import route_experts
    model, c = block.model, block.c
    rows = _rows(x, width=c.hidden_size)
    route = route_experts
    if getattr(model, "pointwise_policy", "reference") == "fused":
        from ..runtime.models.deepseek_v41_fused import router, self_check
        self_check()
        route = router
    logits = linear_tile(model, x, block.prefix+"ffn_gate_inp.weight", rounded=False)
    ids, scores = route(logits, top_k=c.num_experts_per_tok,
                        scale=c.routed_scaling_factor, correction=block.correction,
                        normalize=c.norm_topk_prob)
    if (ids.shape != (1, rows, c.num_experts_per_tok) or scores.shape != ids.shape
            or ids.dtype != mx.uint32 or scores.dtype != mx.float32):
        raise ValueError("Source AST batched router changed occurrence geometry")
    _record(model, "router", rows)
    return ids, scores


def engram_tile(model, h, layer, rows):
    """One prefetched Engram row tile, dense projection and original gate math.

    This does not read the table or mutate hash/cache state. The caller owns the
    already-decoded grouped lookup window until all its consumers complete.
    """
    import mlx.core as mx
    c = model.args
    count = _rows(h, ndim=4, width=c.hidden_size)
    if (h.shape[2] != c.hc_mult or type(layer) is not int
            or not 0 <= layer < c.num_hidden_layers or layer not in c.engram_layer_ids
            or not isinstance(rows, mx.array) or rows.ndim < 2 or not rows.size
            or rows.size % count or not (rows.shape[0] == count or rows.shape[:2] == (1, count))
            or rows.dtype not in (mx.float32, mx.float16, mx.bfloat16)):
        raise ValueError("Source AST Engram decoded row geometry mismatch")
    p = f"blk.{layer}."
    kv = linear_tile(model, rows.reshape(1, count, -1).astype(mx.float32), p+"engram_kv.weight")
    if kv.shape[-1] != (c.hc_mult+1)*c.hidden_size:
        raise ValueError("Source AST Engram projection must publish all key/value channels")
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
        raise ValueError("Source AST Engram gate changed hidden geometry")
    _record(model, "engram", count)
    return result


def head_tile(model, h, pre):
    """One collapse, row-local RMSNorm and complete vocabulary product for M rows."""
    import mlx.core as mx
    rows = _rows(h, ndim=4, width=model.args.hidden_size)
    if (h.shape[2] != model.args.hc_mult or not isinstance(pre, mx.array)
            or pre.shape != h.shape[:-1] or pre.dtype != mx.float32):
        raise ValueError("Source AST head HC/pre geometry mismatch")
    hidden = model.norm(model.collapse(h, pre), model.norm_weight)
    _output(hidden, h, width=model.args.hidden_size)
    logits = linear_tile(model, hidden, "output.weight", rounded=False)
    if logits.shape[-1] != model.args.vocab_size:
        raise ValueError("Source AST head must publish the whole vocabulary")
    _record(model, "head", rows)
    return logits


def scheduled_receipt(model):
    """Small copy without array owners; scheduling is distinct from observed use."""
    return dict(profile=PROFILE, scheduled=dict(getattr(model, "source_ast_batched_dense_calls", {})),
                observation="scheduled calls only; caller must evaluate and record route evidence",
                python_token_row_calls=0, reduction_law="native-M candidate", qualified=False)


@lru_cache(maxsize=1)
def implementation_identity():
    # File hashing is outside hot consumers and owns no device arrays.
    root = Path(__file__).resolve().parents[1]
    paths = (Path(__file__), root/"runtime/models/deepseek_v41.py",
             root/"adapters/deepseek_v41_source_weights.py",
             root/"runtime/models/deepseek_v41_quant.py", root/"runtime/gguf/source_dense.py")
    return hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
