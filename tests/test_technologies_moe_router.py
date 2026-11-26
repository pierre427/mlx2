"""Bit-exact parity: every migrated MoE router vs its verbatim legacy body.

Each ``_legacy_<model>_<op>`` below is the pre-migration code, moved here
verbatim (method bodies become functions over the same attributes). The
tests call the *migrated model code* (module ``__call__`` or function) and
compare with ``mx.array_equal`` (exact) in every dtype the model routes in,
including ties, zeros and masked (-inf) rows. CPU only.
"""
from __future__ import annotations

import math
from dataclasses import replace
import types
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.technologies import moe_router as MR
from mlx2.runtime.technologies.moe_router import RouterSpec, route

mx.set_default_device(mx.cpu)

DTYPES = (mx.bfloat16, mx.float16, mx.float32)
E, K, D = 16, 4, 8


def _cases(dtype, experts=E, seed=0):
    """Logit batches [2, 3, E]: random, heavy ties, zeros, masked, constant, underflow."""
    key = mx.random.key(seed)
    base = mx.random.normal((2, 3, experts), key=key) * 2
    ties = mx.round(base * 1.5) / 1.5
    zeros = mx.zeros_like(base)
    cols = mx.arange(experts)
    masked = mx.where((cols % 3 == 0)[None, None], -mx.inf, base)
    const = mx.full(base.shape, 0.75)
    # every sigmoid underflows to 0 in fp32: the renorm epsilon decides 0 vs NaN
    under = mx.round(base) - 120.0
    return [v.astype(dtype) for v in (base, ties, zeros, masked, const, under)]


def _bias(experts=E, seed=7, dtype=mx.float32):
    b = mx.random.normal((experts,), key=mx.random.key(seed)) * 0.3
    return mx.round(b * 4).astype(dtype) / 4


def _eq(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert mx.array_equal(a, b, equal_nan=True).item(), (a, b)


def _eq_all(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        _eq(g, w)


def _bare(cls, **attrs):
    """A module instance with only the attributes its router body reads."""
    m = cls.__new__(cls)
    nn.Module.__init__(m)
    for name, value in attrs.items():
        object.__setattr__(m, name, value)
    return m


class _Experts:
    """Deterministic expert stand-in that records the indices it saw."""

    def __init__(self, dim=D, experts=E):
        self.table = mx.random.normal((experts, dim), key=mx.random.key(3))
        self.seen = []

    def __call__(self, x, indices):
        self.seen.append(indices)
        return self.table[indices].astype(x.dtype)


def _identity(x):
    return x


# --------------------------------------------------------------- legacy bodies


def _legacy_qwen_softmax(gates, top_k, norm_topk_prob):
    # standard_decoder.MoE / mellum.SparseMoE / qwen3_next (all three copies)
    scores = mx.softmax(gates, axis=-1, precise=True)
    indices = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(scores, indices, axis=-1)
    if norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    return indices, scores


@mx.compile
def _legacy_qwen3_next_select_experts(gates, top_k, norm_topk_prob):
    gates = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return (inds, scores)


def _legacy_gemma4_router(scores, top_k, per_expert_scale):
    indices = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    weights = mx.softmax(mx.take_along_axis(scores, indices, axis=-1), axis=-1)
    return indices, weights * per_expert_scale[indices]


def _legacy_flint_route(gate_out, top_k):
    scores = mx.sigmoid(gate_out.astype(mx.float32))
    indices = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return indices, weights


def _legacy_granite_route(router_out, top_k):
    logits = router_out.astype(mx.float32)
    idx = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
    gates = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), precise=True, axis=-1)
    return idx, gates


@mx.compile
def _legacy_hy_v3_expert_select(gates, expert_bias, top_k, routed_scaling_factor, norm_topk_prob):
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + expert_bias

    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


def _legacy_lfm25_routing(gate_out, x_dtype, top_k, use_bias, bias, norm, scale):
    routing_weights = mx.sigmoid(gate_out)
    selection_scores = routing_weights.astype(mx.float32)
    if use_bias:
        selection_scores = selection_scores + bias
    indices = mx.argpartition(selection_scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(routing_weights, indices, axis=-1)
    if norm:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
    scores = scores * scale
    return indices, scores.astype(x_dtype)


def _legacy_lfm25_decode_reference_route(block, probabilities, activation_dtype=mx.bfloat16):
    selection = probabilities.astype(mx.float32)
    if block.use_expert_bias:
        selection = selection + block.expert_bias
    indices = mx.argpartition(selection, kth=-block.top_k, axis=-1)[..., -block.top_k:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    if block.norm_topk_prob:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
    return indices, (scores * block.routed_scaling_factor).astype(activation_dtype)


def _legacy_deepseek_v4_route_experts(
    logits, *, top_k, scale, correction=None, token_ids=None, table=None, normalize=True
):
    scores = mx.sqrt(nn.softplus(logits.astype(mx.float32)))
    if table is not None:
        if token_ids is None:
            raise ValueError("hash routing requires original token IDs")
        indices = table[token_ids].astype(mx.int32)
    else:
        if correction is None:
            raise ValueError("learned routing requires correction bias")
        indices = mx.argpartition(-(scores + correction), kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return indices.astype(mx.uint32), weights * scale


def _legacy_laguna_router(raw, bias, dtype, top_k, norm, score_func, use_sigmoid, softcap):
    logits = raw.astype(mx.float32)
    if softcap > 0:
        logits = mx.tanh(logits / softcap) * softcap
    if score_func == "sqrtsoftplus":
        scores = mx.sqrt(nn.softplus(logits))
    else:
        scores = mx.sigmoid(logits) if use_sigmoid else mx.softmax(logits, axis=-1)
    corrected = scores + bias.astype(scores.dtype)
    inds = mx.stop_gradient(mx.argpartition(-corrected, kth=top_k - 1, axis=-1)[..., :top_k])
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if norm:
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return inds, weights.astype(dtype)


def _legacy_kimi_linear_gate(logits, expert_bias, top_k, combine, norm, scale):
    scores = mx.sigmoid(logits)
    choice = scores + expert_bias
    ids = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    source = choice if combine == "checkpoint" else scores
    weights = mx.take_along_axis(source, ids, -1)
    if norm:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    return ids, weights * scale


@mx.compile
def _legacy_bailing_group_expert_select(
    gates, expert_bias, top_k, n_group, topk_group, routed_scaling_factor,
    norm_topk_prob, score_function, group_mask_neg_inf=False,
):
    input_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    elif score_function == "softmax":
        scores = mx.softmax(gates.astype(mx.float32), axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE score function: {score_function}")

    original_scores = scores
    if expert_bias is not None:
        scores = scores + expert_bias

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        n_drop = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=n_drop - 1, axis=-2)[..., :n_drop, :]
        fill = -math.inf if group_mask_neg_inf else 0.0
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(fill, dtype=scores.dtype), axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    indices = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(original_scores, indices, axis=-1)
    if top_k > 1 and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return indices, weights.astype(input_type)


@mx.compile
def _legacy_nemotron_group_expert_select(
    gates, e_score_correction_bias, top_k, n_group, topk_group, routed_scaling_factor,
    norm_topk_prob,
):
    orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
    scores = scores + e_score_correction_bias
    if n_group > topk_group:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(-float("inf")), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor
    return inds, scores


def _legacy_xing_router_select(x, weight, e_score_correction_bias, top_k, routed_scaling_factor, norm_topk_prob):
    logits = x.astype(mx.float32) @ weight.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    biased = scores + e_score_correction_bias.astype(mx.float32)
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


def _legacy_raon_router(x, w32, bias, k, renorm, scale):
    scores = mx.sigmoid(x.astype(mx.float32) @ w32.T)
    choice = scores + bias.astype(mx.float32)
    ids = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, ids, -1)
    if renorm:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    return ids, weights * scale if scale != 1.0 else weights


def _legacy_maxwell_router(x, weight, expert_bias, top_k):
    logits = x.astype(mx.float32) @ weight.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    biased = scores + expert_bias.astype(mx.float32)
    ids = mx.stop_gradient(mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k])
    w = mx.take_along_axis(scores, ids, axis=-1)
    return ids, w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)


def _legacy_hy_v4_flash_router(logits, bias, k, norm, scale):
    scores = mx.sigmoid(logits)
    choice = scores + bias.astype(mx.float32)
    indices = mx.stop_gradient(mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k].astype(mx.int32))
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if norm:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return indices, weights * scale


def _legacy_cohere2_route(gate_out, top_k, dtype):
    scores = mx.sigmoid(gate_out.astype(mx.float32))
    inds = mx.stop_gradient(mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k])
    weights = mx.take_along_axis(scores, inds, axis=-1).astype(dtype)
    return inds, weights


def _legacy_inkling2_route(x, weight, bias, top_k, route_scale):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


def _legacy_inkling2_route_logdomain(x, weight, bias, top_k, route_scale):
    logits = x.astype(mx.float32) @ weight.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    picked = mx.take_along_axis(logits, indices, axis=-1)
    log_weights = mx.minimum(picked, 0) - mx.log1p(mx.exp(-mx.abs(picked)))
    log_weights = log_weights - log_weights.max(axis=-1, keepdims=True)
    weights = mx.exp(log_weights)
    weights = weights / weights.sum(axis=-1, keepdims=True) * route_scale
    return indices, weights


def _legacy_glm53_route(logits, bias, routed_topk):
    from mlx2.runtime.models.precise_ops import gate_sigmoid

    scores = gate_sigmoid(logits)
    choice = scores + bias
    ids = mx.argsort(-choice, axis=-1)[..., :routed_topk].astype(mx.uint32)
    selected = mx.take_along_axis(scores, ids, axis=-1)
    selected = selected / selected.sum(-1, keepdims=True) * 2.5
    return ids, selected


def _legacy_mimo_v2_route(w, x, cfg):
    B, T, d = x.shape
    E_, k = cfg.n_routed_experts, cfg.top_k
    logits = x.reshape(-1, d).astype(mx.float32) @ w["mlp.gate.weight"].astype(mx.float32).T
    scores = mx.sigmoid(logits)
    choice = scores + w["mlp.gate.e_score_correction_bias"].astype(mx.float32)[None, :]
    if cfg.n_group > 1:
        g = choice.reshape(-1, cfg.n_group, E_ // cfg.n_group)
        top2 = mx.sort(g, axis=-1)[..., -2:].sum(-1)
        keep_g = mx.argsort(-top2, axis=-1)[:, :cfg.topk_group]
        gmask = mx.zeros(top2.shape, mx.bool_)
        gmask = mx.put_along_axis(gmask, keep_g, mx.array(True), axis=-1)
        emask = mx.repeat(gmask, E_ // cfg.n_group, axis=-1)
        choice = mx.where(emask, choice, mx.array(-mx.inf, mx.float32))
    inds = mx.argsort(-choice, axis=-1)[:, :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if k > 1 and cfg.norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    return inds.astype(mx.int32).reshape(B, T, k), weights.reshape(B, T, k)


def _legacy_kimi_k31_route(w, x, cfg):
    gate = w["block_sparse_moe.gate.weight"]
    logits = x.astype(mx.float32) @ gate.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    choice = scores + w["block_sparse_moe.gate.e_score_correction_bias"].astype(mx.float32)
    k = cfg.top_k
    ids = mx.argsort(-choice, axis=-1)[:, :k].astype(mx.int32)
    chosen = mx.take_along_axis(scores, ids, axis=-1)
    wts = chosen
    if k > 1 and cfg.moe_renormalize:
        wts = chosen / (mx.sum(chosen, -1, keepdims=True) + 1e-20)
    return ids, wts * cfg.routed_scaling_factor, chosen


def _legacy_gpt_oss_route(g, k):
    partitioned_indices = mx.argpartition(g, kth=-k, axis=-1)
    top_k_indices = partitioned_indices[..., -k:]
    top_k_values = mx.take_along_axis(g, top_k_indices, axis=-1)
    return top_k_indices, mx.softmax(top_k_values, axis=-1, precise=True)


# ----------------------------------------------------------------------- tests


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("norm", [True, False])
def test_standard_decoder_and_mellum_moe(dtype, norm):
    from mlx2.runtime.models import mellum, standard_decoder

    for cls in (standard_decoder.MoE, mellum.SparseMoE):
        for logits in _cases(dtype):
            experts = _Experts()
            m = _bare(cls, top_k=K, norm_topk_prob=norm, gate=_identity, switch_mlp=experts)
            got = m(logits)
            idx, w = _legacy_qwen_softmax(logits, K, norm)
            _eq(experts.seen[-1], idx)
            want = mx.sum(experts.table[idx].astype(dtype) * w[..., None], axis=-2)
            _eq(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("norm", [True, False])
def test_qwen3_next_routers(dtype, norm):
    from mlx2.runtime.models import qwen3_next as QN

    block = types.SimpleNamespace(top_k=K, norm_topk_prob=norm)
    for logits in _cases(dtype):
        _eq_all(QN._stock_routing(block, logits), _legacy_qwen_softmax(logits, K, norm))
        _eq_all(
            QN._select_experts(logits, K, norm),
            _legacy_qwen3_next_select_experts(logits, K, norm),
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemma4_router(dtype):
    from mlx2.runtime.models import gemma4

    scale = (mx.arange(E) * 0.1 + 0.5).astype(mx.float32)
    for logits in _cases(dtype):
        hidden = logits  # rms-normalized input, proj = identity
        m = _bare(gemma4.Router, top_k=K, eps=1e-6, proj=_identity,
                  scale=mx.ones((E,), dtype), per_expert_scale=scale, _root_size=E ** -0.5)
        got = m(hidden)
        normed = mx.fast.rms_norm(hidden, mx.ones((E,), dtype) * E ** -0.5, 1e-6)
        _eq_all(got, _legacy_gemma4_router(normed, K, scale))


@pytest.mark.parametrize("dtype", DTYPES)
def test_flint_router(dtype):
    from mlx2.runtime.models import flint

    for logits in _cases(dtype):
        experts = _Experts()
        m = _bare(flint.SparseMoE, top_k=K, gate=_identity, switch_mlp=experts,
                  shared_expert=lambda x: mx.zeros(x.shape[:-1] + (D,), x.dtype),
                  _routed_rows=lambda x, i: None)
        got = m(logits)
        idx, w = _legacy_flint_route(logits, K)
        _eq(experts.seen[-1], idx)
        rows = experts.table[idx].astype(dtype)
        want = mx.sum(rows * w[..., None], axis=-2).astype(dtype) + mx.zeros(
            logits.shape[:-1] + (D,), dtype)
        _eq(got, want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_granitemoe_swa_router(dtype):
    from mlx2.runtime.models import granitemoe_swa as G

    for logits in _cases(dtype):
        experts = _Experts()
        m = _bare(G.MoE, top_k=K, router=_identity, experts=experts)
        got = m(logits)
        idx, gates = _legacy_granite_route(logits, K)
        _eq(experts.seen[-1], idx)
        y = experts.table[idx].astype(dtype)
        _eq(got, (y * gates[..., None]).sum(axis=-2).astype(y.dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("norm,scale,k", [(True, 2.5, K), (False, 1.0, K), (True, 1.0, 1)])
def test_hy_v3_expert_select(dtype, norm, scale, k):
    from mlx2.runtime.models import hy_v3

    bias = _bias()
    for logits in _cases(dtype):
        _eq_all(
            hy_v3.expert_select(logits, bias, k, scale, norm),
            _legacy_hy_v3_expert_select(logits, bias, k, scale, norm),
        )


@pytest.mark.parametrize("dtype", (mx.bfloat16, mx.float32))
@pytest.mark.parametrize("use_bias,norm,scale", [(True, True, 1.0), (False, True, 2.0), (True, False, 1.5)])
def test_lfm25_routing_and_decode_reference(dtype, use_bias, norm, scale):
    from mlx2.runtime.models import lfm25_moe, lfm25_moe_decode

    bias = _bias()
    attrs = dict(top_k=K, norm_topk_prob=norm, use_expert_bias=use_bias,
                 routed_scaling_factor=scale, gate=_identity, num_experts=E)
    if use_bias:
        attrs["expert_bias"] = bias
    m = _bare(lfm25_moe.SparseMoE, **attrs)
    block = types.SimpleNamespace(**attrs)
    for logits in _cases(dtype):
        _eq_all(
            m.routing(logits),
            _legacy_lfm25_routing(logits, logits.dtype, K, use_bias, bias, norm, scale),
        )
        probs = mx.sigmoid(logits)
        _eq_all(
            lfm25_moe_decode.reference_route(block, probs),
            _legacy_lfm25_decode_reference_route(block, probs),
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normalize,scale", [(True, 1.5), (False, 1.0)])
def test_deepseek_v4_route_experts(dtype, normalize, scale):
    from mlx2.runtime.models.deepseek_v4 import route_experts

    corr = _bias()
    table = (mx.arange(50 * K).reshape(50, K) % E).astype(mx.int32)
    ids = mx.array([[1, 7, 49], [0, 3, 3]])
    for logits in _cases(dtype):
        kw = dict(top_k=K, scale=scale, normalize=normalize)
        _eq_all(route_experts(logits, correction=corr, **kw),
                _legacy_deepseek_v4_route_experts(logits, correction=corr, **kw))
        _eq_all(route_experts(logits, token_ids=ids, table=table, **kw),
                _legacy_deepseek_v4_route_experts(logits, token_ids=ids, table=table, **kw))
    with pytest.raises(ValueError, match="correction bias"):
        route_experts(logits, top_k=K, scale=1.0)
    with pytest.raises(ValueError, match="token IDs"):
        route_experts(logits, top_k=K, scale=1.0, table=table)


@pytest.mark.parametrize("dtype", (mx.bfloat16, mx.float32))
@pytest.mark.parametrize("score_func,use_sigmoid,softcap,norm", [
    ("sigmoid", True, 0.0, True), ("sigmoid", True, 30.0, False),
    ("sigmoid", False, 0.0, True), ("sqrtsoftplus", False, 0.0, True),
])
def test_laguna_router(dtype, score_func, use_sigmoid, softcap, norm):
    from mlx2.runtime.models import laguna

    for bias_dtype in (mx.float32, mx.bfloat16):
        bias = _bias(dtype=bias_dtype)
        for logits in _cases(dtype):
            m = _bare(laguna.LagunaTopKRouter, top_k=K, norm_topk_prob=norm,
                      score_func=score_func, use_sigmoid=use_sigmoid, softcap=softcap,
                      proj=_identity, e_score_correction_bias=bias, fused_mode=False)
            _eq_all(m(logits), _legacy_laguna_router(
                logits, bias, dtype, K, norm, score_func, use_sigmoid, softcap))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("combine", ["unbiased", "checkpoint"])
def test_kimi_linear_gate(dtype, combine):
    from mlx2.runtime.models import kimi_linear

    weight = mx.random.normal((E, D), key=mx.random.key(5)).astype(dtype)
    lin = nn.Linear(D, E, bias=False)
    lin.weight = weight
    bias = _bias()
    for x in _cases(dtype, experts=D):
        m = _bare(kimi_linear.Gate, top_k=K, n_group=1, norm_topk_prob=True,
                  routed_scaling_factor=2.5, gate_proj=lin, expert_bias=bias,
                  combine=combine, calls=Counter(), training_indices=False)
        logits = x.astype(mx.float32) @ weight.astype(mx.float32).T
        _eq_all(m(x), _legacy_kimi_linear_gate(logits, bias, K, combine, True, 2.5))


GROUPS = [(1, 1), (4, 2), (4, 4), (8, 3)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_group,topk_group", GROUPS)
@pytest.mark.parametrize("neg_inf", [False, True])
@pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
def test_bailing_group_expert_select(dtype, n_group, topk_group, neg_inf, score_function):
    from mlx2.runtime.models import bailing_hybrid

    for bias in (_bias() - 0.6, None):
        for logits in _cases(dtype):
            args = (logits, bias, K, n_group, topk_group, 2.5, True, score_function, neg_inf)
            _eq_all(bailing_hybrid.group_expert_select(*args),
                    _legacy_bailing_group_expert_select(*args))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_group,topk_group", GROUPS)
@pytest.mark.parametrize("norm,k", [(True, K), (False, K), (True, 1)])
def test_nemotron_group_expert_select(dtype, n_group, topk_group, norm, k):
    from mlx2.runtime.models import nemotron_h

    bias = _bias() - 0.6
    for logits in _cases(dtype):
        args = (logits, bias, k, n_group, topk_group, 5.0, norm)
        _eq_all(nemotron_h.group_expert_select(*args),
                _legacy_nemotron_group_expert_select(*args))


@pytest.mark.parametrize("dtype", DTYPES)
def test_xing_raon_maxwell_hy_v4_routers(dtype):
    from mlx2.runtime.models import hy_v4_flash, maxwell_moe, raon1_mini, xing4_0
    from mlx2.runtime.models import hy_v4_flash_resident as resident

    weight = mx.random.normal((E, D), key=mx.random.key(11)).astype(dtype)
    w32 = weight.astype(mx.float32)
    for bias_dtype in (mx.float32, mx.bfloat16):
        bias = _bias(dtype=bias_dtype)
        for x in _cases(dtype, experts=D):
            for norm, scale in ((True, 2.0), (False, 1.0)):
                _eq_all(xing4_0.router_select(x, weight, bias, K, scale, norm),
                        _legacy_xing_router_select(x, weight, bias, K, scale, norm))
                m = _bare(raon1_mini.Router, weight=weight, e_score_correction_bias=bias,
                          _top_k=K, _renorm=norm, _scale=scale, _w32=None)
                _eq_all(m(x), _legacy_raon_router(x, w32, bias, K, norm, scale))
                h = _bare(hy_v4_flash.Router, top_k=K, scale=scale, norm=norm,
                          weight=weight, e_score_correction_bias=bias)
                logits = resident.f32_matmul(x, weight, kernel=False)
                _eq_all(h(x), _legacy_hy_v4_flash_router(logits, bias, K, norm, scale))
            mw = _bare(maxwell_moe.Router, top_k=K, weight=w32, expert_bias=bias)
            _eq_all(mw(x), _legacy_maxwell_router(x, w32, bias, K))


@pytest.mark.parametrize("dtype", DTYPES)
def test_cohere2_moe_router(dtype):
    from mlx2.runtime.models import cohere2_moe

    for logits in _cases(dtype):
        experts = _Experts()
        m = _bare(cohere2_moe.Cohere2MoeSparseBlock, top_k=K, gate=_identity,
                  switch_mlp=experts)
        got = m(logits)
        idx, w = _legacy_cohere2_route(logits, K, dtype)
        _eq(experts.seen[-1], idx)
        _eq(got, mx.sum(experts.table[idx].astype(dtype) * w[..., None], axis=-2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_inkling2_routes(dtype):
    from mlx2.runtime.models.inkling2 import language, moe_fast

    weight = mx.random.normal((E, D), key=mx.random.key(13)).astype(mx.float32)
    bias = _bias()
    for x in _cases(dtype, experts=D):
        for scale in (8.0, 1.0):
            args = (x, weight, bias, K, scale)
            want = _legacy_inkling2_route(*args)
            _eq_all(language._route(*args), want)
            _eq_all(moe_fast._route(*args), want)
            _eq_all(moe_fast.route_compiled(*args), mx.compile(_legacy_inkling2_route)(*args))
            _eq_all(language._route_logdomain(*args), _legacy_inkling2_route_logdomain(*args))
    # the log-domain guard: every picked logit far below -90 -> sigmoid 0
    x = mx.full((1, 1, D), -40.0)
    args = (x, mx.ones((E, D)), mx.zeros((E,)), K, 8.0)
    _eq_all(language._route_logdomain(*args), _legacy_inkling2_route_logdomain(*args))
    assert mx.all(mx.isfinite(language._route_logdomain(*args)[1])).item()


@pytest.mark.parametrize("dtype", (mx.float32, mx.bfloat16))
def test_glm53_route(dtype):
    from mlx2.runtime.models import glm53_gguf

    bias = _bias()
    fake = types.SimpleNamespace(
        weights=types.SimpleNamespace(array=lambda name: bias),
        router="reference", routed_topk=K, calls=Counter(),
    )
    for logits in _cases(dtype):
        got = glm53_gguf.Glm53GGUFModel._route(fake, logits, "blk.3")
        _eq_all(got, _legacy_glm53_route(logits, bias, K))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_group,topk_group", GROUPS)
@pytest.mark.parametrize("norm,k,scale", [(True, K, 2.5), (False, K, 1.0), (True, 1, 1.0)])
def test_mimo_v2_route(dtype, n_group, topk_group, norm, k, scale):
    from mlx2.runtime.models import mimo_v2

    w = {"mlp.gate.weight": mx.random.normal((E, D), key=mx.random.key(17)).astype(dtype),
         "mlp.gate.e_score_correction_bias": _bias(dtype=mx.bfloat16) - 0.5}
    cfg = types.SimpleNamespace(n_routed_experts=E, top_k=k, n_group=n_group,
                                topk_group=topk_group, norm_topk_prob=norm,
                                routed_scaling_factor=scale)
    for x in _cases(dtype, experts=D):
        _eq_all(mimo_v2.route(w, x, cfg), _legacy_mimo_v2_route(w, x, cfg))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("renorm,k", [(True, K), (False, K), (True, 1)])
def test_kimi_k31_route(dtype, renorm, k):
    from mlx2.runtime.models import kimi_k31

    w = {"block_sparse_moe.gate.weight": mx.random.normal((E, D), key=mx.random.key(19)).astype(dtype),
         "block_sparse_moe.gate.e_score_correction_bias": _bias(dtype=mx.bfloat16)}
    cfg = types.SimpleNamespace(top_k=k, moe_renormalize=renorm, routed_scaling_factor=2.0)
    for x in _cases(dtype, experts=D):
        got = kimi_k31.route(w, x.reshape(-1, D), cfg)
        _eq_all((got.ids, got.weights, got.scores),
                _legacy_kimi_k31_route(w, x.reshape(-1, D), cfg))


@pytest.mark.parametrize("dtype", DTYPES)
def test_gpt_oss_router(dtype):
    from mlx2.runtime.models import gpt_oss

    for logits in _cases(dtype):
        experts = _Experts()
        m = _bare(gpt_oss.MLPBlock, num_experts_per_tok=K, router=_identity,
                  experts=experts, sharding_group=None)
        got = m(logits)
        idx, w = _legacy_gpt_oss_route(logits, K)
        _eq(experts.seen[-1], idx)
        y = experts.table[idx].astype(dtype) * mx.expand_dims(w, axis=-1)
        _eq(got, y.sum(axis=-2))


# ------------------------------------------------------------- spec / dispatch


def test_from_config_derives_standard_hf_fields():
    ds3 = {"num_experts_per_tok": 8, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
           "n_group": 8, "topk_group": 4, "norm_topk_prob": True,
           "routed_scaling_factor": 2.5}
    spec = RouterSpec.from_config(ds3)
    assert (spec.top_k, spec.score_func, spec.use_bias, spec.n_group, spec.topk_group) == (
        8, "sigmoid", True, 8, 4)
    assert spec.norm_topk_prob and spec.routed_scaling_factor == 2.5
    assert spec.topk_method == "argpartition_neg" and spec.norm_eps == 1e-20
    assert spec.grouped and spec.group_score == "top2_sum"
    qwen = RouterSpec.from_config({"num_experts_per_tok": 10, "norm_topk_prob": True})
    assert qwen == RouterSpec(top_k=10, norm_topk_prob=True)
    v2 = RouterSpec.from_config({"num_experts_per_tok": 6, "topk_method": "group_limited_greedy",
                                 "n_group": 8, "topk_group": 3, "scoring_func": "softmax"})
    assert v2.group_score == "max" and not v2.use_bias
    over = RouterSpec.from_config(ds3, norm_eps=0.0, topk_method="argsort_neg")
    assert over.norm_eps == 0.0 and over.topk_method == "argsort_neg"
    with pytest.raises(TypeError):
        RouterSpec.from_config(ds3, not_a_field=1)


def test_spec_validation_and_bias_contract():
    with pytest.raises(ValueError):
        RouterSpec(top_k=0)
    with pytest.raises(ValueError):
        RouterSpec(top_k=2, score_func="sigmoid", softmax_position="after_topk")
    with pytest.raises(ValueError):
        RouterSpec(top_k=2, renorm="logdomain")
    logits = mx.zeros((1, E))
    with pytest.raises(ValueError, match="correction bias"):
        route(logits, RouterSpec(top_k=2, score_func="sigmoid", use_bias=True))
    with pytest.raises(ValueError, match="no correction bias"):
        route(logits, RouterSpec(top_k=2), mx.zeros((E,)))


def test_group_fill_zero_can_pick_a_dropped_expert_neg_inf_cannot():
    # Drift documented in bailing_hybrid: a negative bias pushes kept scores
    # below the 0.0 fill, so a dropped group's expert wins the top-k.
    logits = mx.zeros((1, 8))  # sigmoid 0.5 everywhere
    bias = mx.array([-1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0])
    base = dict(top_k=4, score_func="sigmoid", score_dtype="float32", use_bias=True,
                n_group=2, topk_group=1, topk_method="argpartition_neg")
    zero_ids, _ = route(logits, RouterSpec(group_fill=0.0, **base), bias)
    inf_ids, _ = route(logits, RouterSpec(**base), bias)
    assert sorted(inf_ids.tolist()[0]) == [0, 1, 2, 3]
    assert any(i >= 4 for i in zero_ids.tolist()[0])


def test_norm_eps_option_changes_laguna_law_only_when_chosen():
    # Every picked sigmoid underflows to 0: the legacy Laguna law (no eps)
    # gives 0/0 = NaN weights, the proposed +1e-20 gives zeros.
    logits = mx.random.normal((3, E), key=mx.random.key(23)) - 120.0
    bias = mx.zeros((E,))
    legacy = RouterSpec(top_k=K, score_func="sigmoid", score_dtype="float32", use_bias=True,
                        bias_cast="to_scores", topk_method="argpartition_neg",
                        norm_topk_prob=True)
    eps = replace(legacy, norm_eps=1e-20)
    a, b = route(logits, legacy, bias), route(logits, eps, bias)
    _eq(a[0], b[0])  # same experts
    assert not mx.array_equal(a[1], b[1], equal_nan=True).item()
    assert mx.all(mx.isnan(a[1])).item() and mx.all(b[1] == 0).item()


def test_registry_refuses_on_cpu_and_by_law():
    spec = RouterSpec(top_k=10, norm_topk_prob=True)
    fn, receipt = MR.select_router(spec, logits=mx.zeros((1, 1, 512), mx.bfloat16))
    assert fn is route and receipt.selected == "reference"
    assert "qwen4_moe_router" in receipt.refusals
    sig = RouterSpec(top_k=10, score_func="sigmoid")
    _, receipt = MR.select_router(sig, logits=mx.zeros((1, 256)))
    assert "score_func" in receipt.refusals["qwen4_moe_router"]
    with pytest.raises(RuntimeError):
        MR.select_router(spec, policy="qwen4_moe_router", logits=mx.zeros((1, 1, 512)))
    decl = MR.FUSED_ROUTERS["qwen4_moe_window_router_topk"]
    assert decl.admits(spec) is None
    assert decl.admits(sig) is not None


def test_contract_router_spec_matches_qwen_models():
    from mlx2.runtime.models.qwen_moe_contracts import (
        QWEN36_E256_K8_ROUTING, QWEN4_E512_K10_ROUTING)

    for contract in (QWEN4_E512_K10_ROUTING, QWEN36_E256_K8_ROUTING):
        spec = contract.router_spec()
        assert spec == MR.cached_spec(top_k=contract.top_k, norm_topk_prob=True)
        assert contract.spec_refusal(spec) is None
        assert "top_k" in contract.spec_refusal(replace(spec, top_k=spec.top_k + 1))
        assert MR.FUSED_ROUTERS["qwen4_moe_window_router_topk"].admits(spec) is None


def test_registry_name_and_bounded_refusal_codes():
    from mlx2.runtime.technologies import dispatch

    assert MR.ROUTE.operation == "moe_router.route"
    dispatch.reset_counters()
    MR.select_router(RouterSpec(top_k=10, norm_topk_prob=True),
                     logits=mx.zeros((1, 1, 512), mx.bfloat16))
    MR.select_router(RouterSpec(top_k=8, score_func="sigmoid"), logits=mx.zeros((1, 256)))
    snap = dispatch.counters_snapshot()
    codes = {r["code"] for r in snap["refusals"] if r["technology"] == "moe_router"}
    assert codes and codes <= {"law", "admission", "device", "shape"}, codes
    paths = {r["path"] for r in snap["selections"] if r["technology"] == "moe_router"}
    assert paths == {"reference"}


def test_combine_weights_stay_differentiable():
    spec = RouterSpec(top_k=K, score_func="sigmoid", score_dtype="float32", use_bias=True,
                      topk_method="argpartition_neg", norm_topk_prob=True, norm_eps=1e-20)
    logits = mx.random.normal((3, E), key=mx.random.key(29))
    grad = mx.grad(lambda z: route(z, spec, mx.zeros((E,)))[1].square().sum())(logits)
    assert grad.shape == logits.shape and mx.all(mx.isfinite(grad)).item()


# --------------------------------------------- fast-path laws and signatures

_ALTERNATIVES = {
    "top_k": (3, 5),
    "score_func": ("softmax", "sigmoid", "sqrtsoftplus", "identity"),
    "softmax_position": ("before_topk", "after_topk"),
    "softmax_precise": (True, False),
    "sigmoid_impl": ("mx", "gate_sigmoid"),
    "score_dtype": ("input", "float32"),
    "logit_softcap": (0.0, 30.0),
    "selection_dtype": ("scores", "float32"),
    "use_bias": (True, False),
    "bias_cast": ("promote", "to_scores"),
    "combine_scores": ("unbiased", "biased"),
    "n_group": (1, 4),
    "topk_group": (1, 2),
    "group_score": ("top2_sum", "max"),
    "group_select": ("drop_argpartition", "keep_argsort"),
    "group_fill": (-math.inf, 0.0),
    "topk_method": ("argpartition_high", "argpartition_neg", "argsort_neg"),
    "norm_topk_prob": (True, False),
    "norm_eps": (0.0, 1e-20, 1e-6),
    "norm_min_top_k": (1, 2),
    "renorm": ("linear", "logdomain"),
    "routed_scaling_factor": (1.0, 2.5),
    "index_dtype": ("native", "int32", "uint32"),
    "hash_layer": (False, True),
}


def _one_field_variants(spec):
    from dataclasses import fields as dc_fields

    names = {f.name for f in dc_fields(RouterSpec)}
    assert names == set(_ALTERNATIVES), names ^ set(_ALTERNATIVES)
    for name in sorted(names):
        variants = []
        for value in _ALTERNATIVES[name]:
            if value == getattr(spec, name):
                continue
            try:
                variants.append(replace(spec, **{name: value}))
            except ValueError:
                continue  # not a constructible spec
        yield name, variants


def _captured_spec(module, call):
    """The RouterSpec a model passes to the reference on one call."""
    seen = []
    real = module._route_reference

    def spy(logits, spec, *args, **kwargs):
        seen.append(spec)
        return real(logits, spec, *args, **kwargs)

    module._route_reference = spy
    try:
        call()
    finally:
        module._route_reference = real
    assert seen, "the model never called the reference"
    return seen[-1]


def _model_specs():
    from mlx2.runtime.models import (
        deepseek_v4, glm53_gguf, laguna, lfm25_moe, nemotron_h, qwen3_next)
    from mlx2.runtime.models.inkling2 import language

    bias512 = mx.zeros((512,))
    bias256 = mx.zeros((256,))
    specs = {}
    specs["qwen4_moe_router"] = _captured_spec(
        qwen3_next, lambda: qwen3_next._stock_routing(
            types.SimpleNamespace(top_k=10, norm_topk_prob=True), mx.zeros((1, 1, 512))))
    # group_expert_select is mx.compile'd: a shape no other test uses forces a trace
    specs["nemotron_moe_router"] = _captured_spec(
        nemotron_h, lambda: nemotron_h.group_expert_select(
            mx.zeros((1, 7, 512)), bias512, 22, 1, 1, 5.0, True))
    fake = types.SimpleNamespace(weights=types.SimpleNamespace(array=lambda name: bias256),
                                 router="reference", routed_topk=8, calls=Counter())
    specs["glm53_fused_router"] = _captured_spec(
        glm53_gguf, lambda: glm53_gguf.Glm53GGUFModel._route(fake, mx.zeros((1, 256)), "b"))
    lag = _bare(laguna.LagunaTopKRouter, top_k=8, norm_topk_prob=True, score_func="sigmoid",
                use_sigmoid=True, softcap=0.0, proj=_identity,
                e_score_correction_bias=bias256, fused_mode=False)
    specs["laguna_fused_router"] = _captured_spec(laguna, lambda: lag(mx.zeros((1, 256))))
    specs["deepseek_v41_fused_router"] = _captured_spec(
        deepseek_v4, lambda: deepseek_v4.route_experts(
            mx.zeros((1, 256)), top_k=6, scale=1.5, correction=bias256))
    specs["qwen4_moe_window_router_topk"] = specs["qwen4_moe_router"]
    lfm = _bare(lfm25_moe.SparseMoE, top_k=4, norm_topk_prob=True, use_expert_bias=True,
                routed_scaling_factor=1.0, gate=_identity, expert_bias=mx.zeros((E,)),
                num_experts=E)
    specs["lfm25_moe_decode_route"] = _captured_spec(
        lfm25_moe, lambda: lfm.routing(mx.zeros((1, E))))
    specs["inkling2_moe_decode_kernel"] = _captured_spec(
        language, lambda: language._route(
            mx.zeros((1, D)), mx.zeros((E, D)), mx.zeros((E,)), 8, 8.0))
    return specs


def test_every_fast_path_admits_its_models_spec_and_refuses_every_one_field_change():
    specs = _model_specs()
    paths = {**MR.ROUTER_FAST_PATHS, **MR.FUSED_ROUTERS}
    assert set(specs) == set(paths)
    for name, path in paths.items():
        spec = specs[name]
        assert path.law_refusal(spec) is None, (name, path.law_refusal(spec))
        for field, variants in _one_field_variants(spec):
            # no variant: no constructible spec differs in this field alone
            for variant in variants:
                reason = path.law_refusal(variant)
                if field in path.free:
                    assert reason is None, (name, field, reason)
                else:
                    assert reason is not None and reason.startswith("law: "), (name, field)
                    assert field in reason, (name, field, reason)


def test_softmax_law_covers_every_field():
    from dataclasses import fields as dc_fields

    assert set(MR.SOFTMAX_TOPK_LAW) == {f.name for f in dc_fields(RouterSpec)} - {"top_k"}
    assert MR.SOFTMAX_TOPK_LAW["hash_layer"] is False


@pytest.mark.parametrize("name", sorted(MR.ROUTER_FAST_PATHS))
@pytest.mark.parametrize("dtype", (mx.float32, mx.bfloat16))
def test_fast_path_run_honours_the_reference_signature(name, dtype):
    import inspect
    from dataclasses import replace as dc_replace

    path = MR.ROUTER_FAST_PATHS[name]
    params = inspect.signature(path.run).parameters
    for kw in ("out_dtype", "expert_scale", "hash_table", "token_ids", "prepared", "with_scores"):
        assert kw in params, (name, kw)
    spec = path.law
    experts = 512 if name in ("qwen4_moe_router", "nemotron_moe_router") else 256
    bias = mx.zeros((experts,)) + 0.01 if spec.use_bias else None

    # Stand-in kernel: the reference's (indices, weights) before expert_scale
    # and out_dtype -- what each registered kernel is declared to return.
    def kernel(logits, spec, bias, **_):
        return route(logits, spec, bias, prepared=True)

    stub = dc_replace(path, kernel=kernel)
    logits = (mx.random.normal((2, experts), key=mx.random.key(31)) * 2).astype(dtype)
    scale = mx.arange(experts) % 7 * 0.25 + 0.5
    for kw in ({}, {"out_dtype": mx.bfloat16}, {"expert_scale": scale},
               {"with_scores": True}, {"with_scores": True, "out_dtype": mx.float16}):
        _eq_all(stub.run(logits, spec, bias, **kw), route(logits, spec, bias, **kw))
        prepped = MR.prepare_logits(logits, spec)
        _eq_all(stub.run(prepped, spec, bias, prepared=True, **kw),
                route(prepped, spec, bias, prepared=True, **kw))
    with pytest.raises(ValueError):
        stub.run(logits, spec, bias, hash_table=mx.zeros((4, spec.top_k), mx.int32),
                 token_ids=mx.zeros((2,), mx.int32))
    with pytest.raises(ValueError, match="law"):
        stub.run(logits, replace(spec, logit_softcap=30.0), bias)
    # on CPU the registered path refuses at the admission/device stage, coded
    reason = path.admits(spec, logits=logits, bias=bias)
    assert reason is not None and reason.split(":")[0] in {"admission", "device", "shape"}


def test_after_topk_respects_explicit_index_dtype():
    spec = RouterSpec(top_k=2, softmax_position="after_topk", index_dtype="int32")
    ids, _ = route(mx.array([[2., 3., 1.]]), spec)
    assert ids.dtype == mx.int32


def test_fused_admission_checks_correction_bias_contract(monkeypatch):
    monkeypatch.setattr(MR, "_metal_refusal", lambda: None)
    ds = MR.ROUTER_FAST_PATHS["deepseek_v41_fused_router"]
    logits = mx.zeros((2, 256))
    assert ds.admits(ds.law, logits=logits, bias=mx.zeros((256,))) is None
    for bias in (mx.zeros((2, 256)), mx.zeros((1,))):
        assert ds.admits(ds.law, logits=logits, bias=bias).startswith("shape:")
    qwen = MR.ROUTER_FAST_PATHS["qwen4_moe_router"]
    assert qwen.admits(qwen.law, logits=logits, bias=mx.zeros((256,))).startswith("shape:")
    with pytest.raises(ValueError, match="bias"):
        qwen.run(logits, qwen.law, mx.zeros((256,)))
