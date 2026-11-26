"""CPU parity: every migrated MLA copy against technologies.mla (exact bits).

Each ``_legacy_*`` is the model's pre-migration code, moved here verbatim
(``self`` attributes became parameters). Every test compares with
``mx.array_equal`` (never allclose) in each dtype the model uses.
"""
import math

import mlx.core as mx
import pytest

from mlx2.runtime.models.base import scaled_dot_product_attention
from mlx2.runtime.technologies import mla
from mlx2.runtime.technologies.dispatch import replace

mx.set_default_device(mx.cpu)

DTYPES = (mx.float32, mx.float16, mx.bfloat16)


def _rand(key, shape, dtype, scale=1.0):
    return (mx.random.normal(shape, key=mx.random.key(key)) * scale).astype(dtype)


def _same(a, b):
    mx.eval(a, b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert mx.array_equal(a, b, equal_nan=True).item()


def _ml(i, o, h, key, dtype, quant=False):
    m = mla.QuantizableMultiLinear(i, o, h)
    m.weight = _rand(key, (h, o, i), dtype, 0.1)
    return m


# ------------------------------------------------------------------ legacy

def _legacy_xing_scale(q_head_dim, scaling):
    scale = q_head_dim**-0.5
    if scaling is not None:
        rope_type = scaling.get("type") or scaling.get("rope_type", "default")
        mscale_all_dim = scaling.get("mscale_all_dim", 0)
        factor = scaling.get("factor", 1.0)
        # HF: only non-default rope types with a truthy mscale_all_dim.
        if rope_type != "default" and mscale_all_dim and factor > 1:
            s = 0.1 * mscale_all_dim * math.log(factor) + 1.0
            scale = scale * s * s
    return scale


def _legacy_ling_scale(qk_head_dim, rope_scaling):
    scale = qk_head_dim**-0.5
    if rope_scaling is not None:
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        scaling_factor = rope_scaling.get("factor", 1)
        if mscale_all_dim and scaling_factor > 1:
            mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
            scale *= mscale * mscale
    return scale


def _legacy_xing_absorbed_max_query(kv_lora_rank, qk_nope_head_dim, v_head_dim, cache_len=None):
    d = qk_nope_head_dim + v_head_dim
    if cache_len is None:
        numerator = kv_lora_rank * d
        denom = 2 * kv_lora_rank - d
    else:
        numerator = cache_len * kv_lora_rank * d
        denom = cache_len * (2 * kv_lora_rank - d) + kv_lora_rank * d
    if denom <= 0:
        return 1 << 30
    return max(1, numerator // denom)


def _legacy_xing_absorbed(scale, q_latent, q_pe, kv_latent, k_pe, mask):
    B, H, L, D = q_latent.shape
    q = q_latent.transpose(0, 2, 1, 3).reshape(B, 1, L * H, D)
    q_pe = (q_pe * scale).transpose(0, 2, 1, 3).reshape(B, 1, L * H, q_pe.shape[-1])
    pe_scores = q_pe @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        rows = mx.repeat(mask, H, axis=-2)
        pe_scores = mx.where(rows, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
    output = mx.fast.scaled_dot_product_attention(q, kv_latent, kv_latent, scale=scale, mask=pe_scores)
    return output.reshape(B, L, H, D).transpose(0, 2, 1, 3)


def _legacy_xing_expanded(scale, embed_q, unembed_out, q_nope, q_pe, kv_latent, k_pe, mask):
    pe_scores = (q_pe * scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
    k = embed_q(kv_latent, transpose=False)
    v = unembed_out(kv_latent)
    return mx.fast.scaled_dot_product_attention(q_nope, k, v, scale=scale, mask=pe_scores)


def _legacy_ling_attend_latent(scale, embed_q, unembed_out, absorbed, q_nope, q_pe, kv_latent, k_pe,
                               mask, cache):
    pe_scores = (q_pe * scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        pe_scores = mx.where(
            mask,
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )
    if absorbed:
        q_nope = embed_q(q_nope)
        keys = values = kv_latent
    else:
        keys = embed_q(kv_latent, transpose=False)
        values = unembed_out(kv_latent)
    output = scaled_dot_product_attention(
        q_nope, keys, values, cache=cache, scale=scale, mask=pe_scores,
    )
    if absorbed:
        output = unembed_out(output)
    return output


def _legacy_ling_tail(scale, embed_q, unembed_out, q_nope, q_pe, kv_latent, k_pe, mask, cache):
    pe_scores = (q_pe * scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None and not isinstance(mask, str):
        pe_scores = mx.where(
            mask[..., -1:, :],
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )
    output = scaled_dot_product_attention(
        embed_q(q_nope), kv_latent, kv_latent,
        cache=cache, scale=scale, mask=pe_scores,
    )
    return unembed_out(output).transpose(0, 2, 1, 3)


def _legacy_ling_tree_row(scale, embed_q, unembed_out, q_nope_row, q_pe_row, keys, positional, cache):
    query = embed_q(q_nope_row)
    scores = (q_pe_row * scale) @ positional.swapaxes(-1, -2)
    output = scaled_dot_product_attention(
        query, keys, keys, cache=cache, scale=scale, mask=scores,
    )
    return unembed_out(output)


def _legacy_kimi_latent(q, latent, shared, mask, policy, *, w, n, scale, key_tile, single_score_bytes,
                        rank):
    """kimi_linear.MultiLatentAttention._latent, latent_single / latent_tiled."""
    causal = isinstance(mask, str)
    offset = latent.shape[2] - q.shape[2]

    def visible(r0, r1, k0, k1):
        return mx.arange(offset+r0, offset+r1)[:, None] >= mx.arange(k0, k1)[None, :]
    qn, qs = mx.split(q, [n], -1)
    qn = qn @ w[:, :n]
    if policy == "latent_single" and q.size // q.shape[-1] * latent.shape[2] * 4 <= single_score_bytes:
        score = ((qn @ latent.swapaxes(-1, -2)).astype(mx.float32)
                 + (qs @ shared.swapaxes(-1, -2)).astype(mx.float32)) * scale
        if causal:
            score = mx.where(visible(0, q.shape[2], 0, latent.shape[2]), score, -float("inf"))
        elif mask is not None:
            score = mx.where(mask, score, -float("inf"))
        top = score.max(-1, keepdims=True)
        prob = mx.exp(score - mx.where(mx.isfinite(top), top, 0))
        total = mx.maximum(prob.sum(-1, keepdims=True), 1e-30)
        return ((prob @ latent.astype(mx.float32)) / total).astype(q.dtype) @ w[:, n:].swapaxes(-1, -2)
    pieces = []
    for start in range(0, q.shape[2], 64):
        end = min(start+64, q.shape[2])
        qn_tile, qs_tile = qn[:, :, start:end], qs[:, :, start:end]
        maximum = mx.full((*qn_tile.shape[:-1], 1), -float("inf"), dtype=mx.float32)
        total = mx.zeros_like(maximum)
        accum = mx.zeros((*qn_tile.shape[:-1], rank), dtype=mx.float32)
        for k0 in range(0, latent.shape[2], key_tile):
            k1 = min(k0+key_tile, latent.shape[2])
            score = ((qn_tile @ latent[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)
                     + (qs_tile @ shared[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)) * scale
            if causal:
                score = mx.where(visible(start, end, k0, k1), score, -float("inf"))
            elif mask is not None:
                score = mx.where(mask[..., start:end, k0:k1], score, -float("inf"))
            new_max = mx.maximum(maximum, score.max(-1, keepdims=True))
            safe_max = mx.where(mx.isfinite(new_max), new_max, 0)
            alpha = mx.exp(maximum-safe_max)
            prob = mx.exp(score-safe_max)
            accum = accum * alpha + prob @ latent[:, :, k0:k1].astype(mx.float32)
            total = total * alpha + prob.sum(-1, keepdims=True)
            maximum = new_max
        result = (accum / mx.maximum(total, 1e-30)).astype(q.dtype) @ w[:, n:].swapaxes(-1, -2)
        pieces.append(result)
    return mx.concatenate(pieces, axis=2)


def _legacy_kimi_expanded(q, kv, shared, mask, *, h, n, r, v, scale, d256):
    b = q.shape[0]
    keys, values = mx.split(kv, [n], -1)
    shared = mx.broadcast_to(shared, (b, h, shared.shape[2], r))
    if d256:
        def pad(z, d):
            return mx.concatenate([z, mx.zeros((*z.shape[:-1], d), z.dtype)], -1)
        keys = mx.concatenate([keys, shared, mx.zeros((*keys.shape[:-1], 256-n-r), keys.dtype)], -1)
        return mx.fast.scaled_dot_product_attention(
            pad(q, 256-n-r), keys, pad(values, 256-v), scale=scale, mask=mask,
            force_fused=True)[..., :v]
    keys = mx.concatenate([keys, shared], -1)
    return scaled_dot_product_attention(q, keys, values, cache=None, scale=scale, mask=mask)


def _legacy_hy_mla_attend(q, kv, mask, sinks, scale, rank, budget, bounded):
    N, H, L, _ = q.shape
    T = kv.shape[2]
    sinks = sinks.astype(q.dtype)
    rows = L if isinstance(mask, str) else max(1, budget // max(1, N * H * T))
    if rows >= L:
        return mx.fast.scaled_dot_product_attention(q, kv, kv, scale=scale, mask=mask, sinks=sinks)[..., :rank]
    outs = []
    for a in range(0, L, rows):
        b = min(L, a + rows)
        m = mask if mask is None else mask[..., a:b, :]
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, a:b], kv, kv, scale=scale, mask=m, sinks=sinks)[..., :rank])
        if bounded:
            mx.eval(outs[-1])
    return mx.concatenate(outs, axis=2)


def _legacy_k31_loop(q, kv, k_pe, *, dn, dr, dv, h, offset, rows, out_dtype, keep):
    t = q.shape[0]
    n = kv.shape[0]
    k = mx.concatenate([kv[..., :dn], mx.broadcast_to(k_pe[:, None, :], (n, h, dr))], -1)
    k, v = k.astype(mx.float32), kv[..., dn:]
    q = q.astype(mx.float32) * (dn + dr) ** -0.5
    outs = []
    for r0 in range(0, t, rows):
        r1 = min(t, r0 + rows)
        nk = offset + r1
        scores = mx.einsum("thd,shd->hts", q[r0:r1], k[:nk])
        mask = mx.arange(nk)[None, :] <= mx.arange(offset + r0, offset + r1)[:, None]
        if keep is not None:
            mask = mask & keep(r0, r1)
        probs = mx.softmax(mx.where(mask, scores, -mx.inf), axis=-1).astype(out_dtype)
        outs.append(mx.einsum("hts,shd->thd", probs, v[:nk]))
        mx.eval(outs[-1])
    return mx.concatenate(outs).reshape(t, h * dv)


# ------------------------------------------------------------------ helpers

def _causal(L, S, offset=None):
    offset = S - L if offset is None else offset
    return mx.arange(offset, offset + L)[:, None] >= mx.arange(S)[None, :]


def _masks(L, S):
    """None, causal, a random keep pattern with one fully masked row."""
    yield None
    yield _causal(L, S)[None, None]
    rnd = mx.random.uniform(shape=(L, S), key=mx.random.key(L * 31 + S)) > 0.4
    if L > 1:
        rnd = mx.concatenate([mx.zeros((1, S), mx.bool_), rnd[1:]])
    yield rnd[None, None]


XING = mla.MLASpec(n_heads=4, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                   absorbed_rule="cost_model", absorbed_layout="fold_heads", pe_form="score_bias",
                   sdpa_route="fast")
LING = replace(XING, absorbed_rule="single_row", absorbed_layout="broadcast", sdpa_route="base")


# ------------------------------------------------------------------ spec

SCALINGS = [None, {"type": "yarn", "factor": 40, "mscale_all_dim": 1.0},
            {"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 0.707},
            {"type": "default", "factor": 8, "mscale_all_dim": 1},
            {"rope_type": "linear", "factor": 1.0, "mscale_all_dim": 1},
            {"type": "yarn", "factor": 32, "mscale_all_dim": 0}, {"factor": 7.3, "mscale_all_dim": 0.3}]


@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("dims", [(128, 64), (192, 64), (16, 8)])
def test_softmax_scale_matches_each_model_formula(scaling, dims):
    qk = sum(dims)
    base = {"num_attention_heads": 4, "kv_lora_rank": 32, "qk_nope_head_dim": dims[0],
            "qk_rope_head_dim": dims[1], "v_head_dim": 16, "rope_scaling": scaling}
    xing = mla.MLASpec.from_config(base, mscale_rule="nondefault_rope_type", mscale_order="sequential")
    ling = mla.MLASpec.from_config(base, mscale_rule="any_rope_type", mscale_order="squared")
    plain = mla.MLASpec.from_config(base)
    assert xing.softmax_scale == _legacy_xing_scale(qk, scaling)
    assert ling.softmax_scale == _legacy_ling_scale(qk, scaling)
    assert plain.softmax_scale == qk ** -0.5


def test_from_config_fields_nope_and_overrides():
    cfg = {"num_attention_heads": 32, "kv_lora_rank": 512, "q_lora_rank": None, "qk_nope_head_dim": 128,
           "qk_rope_head_dim": 64, "v_head_dim": 128, "mla_use_nope": True}
    spec = mla.MLASpec.from_config(cfg, softmax_scale=0.5)
    assert (spec.n_heads, spec.kv_lora_rank, spec.q_lora_rank, spec.position) == (32, 512, None, "nope")
    assert spec.softmax_scale == 0.5 and spec.qk_head_dim == 192
    with pytest.raises(TypeError):
        mla.MLASpec.from_config(cfg, not_a_field=1)
    with pytest.raises(ValueError):
        mla.MLASpec.from_config(cfg, pe_form="bogus")


@pytest.mark.parametrize("cache_len", [None, 1, 7, 512, 100000])
@pytest.mark.parametrize("geometry", [(512, 128, 128), (32, 16, 16), (64, 128, 128), (8, 16, 16)])
def test_absorbed_cost_rule_matches_xing(geometry, cache_len):
    assert mla.absorbed_max_query(*geometry, cache_len=cache_len) == \
        _legacy_xing_absorbed_max_query(*geometry, cache_len=cache_len)


def test_use_absorbed_rules():
    assert mla.use_absorbed(XING, 1, 10) and not mla.use_absorbed(XING, 10_000, 10_000)
    assert mla.use_absorbed(XING, 10_000, 10, max_query_override=1 << 30)
    assert not mla.use_absorbed(LING, 2, 10)
    assert mla.use_absorbed(replace(LING, absorbed_verify_rows=16), 16, 10)
    assert mla.use_absorbed(replace(LING, absorbed_rule="always"), 99, 1)
    assert not mla.use_absorbed(replace(LING, absorbed_rule="never"), 1, 1)
    with pytest.raises(ValueError):
        mla.use_absorbed(replace(LING, absorbed_rule="policy"), 1, 1)


def test_multilinear_quantizable_split():
    plain = mla.MultiLinear(8, 4, 2, init="zeros")
    assert not hasattr(plain, "to_quantized") and type(plain).__name__ == "MultiLinear"
    assert mx.array_equal(plain.weight, mx.zeros((2, 4, 8))).item()
    q = mla.QuantizableMultiLinear(64, 32, 2).to_quantized(32, 8)
    assert isinstance(q, mla.QuantizedMultiLinear)
    from mlx2.runtime.models import hy_v4_flash, xing4_0
    assert hy_v4_flash.MultiLinear is mla.MultiLinear
    assert xing4_0.MultiLinear is mla.QuantizableMultiLinear


# ------------------------------------------------------------------ Xing

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L,S", [(1, 1), (1, 9), (3, 9), (4, 4)])
def test_xing_absorbed_and_expanded_parity(dtype, L, S):
    B, H, r, p, dn = 2, XING.n_heads, XING.kv_lora_rank, XING.qk_rope_head_dim, XING.qk_nope_head_dim
    embed_q, unembed_out = _ml(dn, r, H, 1, dtype), _ml(r, 16, H, 2, dtype)
    q_nope, q_pe = _rand(3, (B, H, L, dn), dtype), _rand(4, (B, H, L, p), dtype)
    kv, kpe = _rand(5, (B, 1, S, r), dtype), _rand(6, (B, 1, S, p), dtype)
    scale = _legacy_xing_scale(24, {"type": "yarn", "factor": 40, "mscale_all_dim": 1.0})
    for mask in _masks(L, S):
        _same(mla.absorbed_attention(XING, embed_q(q_nope), q_pe, kv, kpe, mask, scale=scale),
              _legacy_xing_absorbed(scale, embed_q(q_nope), q_pe, kv, kpe, mask))
        k, v = embed_q(kv, transpose=False), unembed_out(kv)
        _same(mla.expanded_attention(XING, q_nope, q_pe, k, v, kpe, mask, scale=scale),
              _legacy_xing_expanded(scale, embed_q, unembed_out, q_nope, q_pe, kv, kpe, mask))


# ------------------------------------------------------------------ Ling

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("absorbed", [True, False])
@pytest.mark.parametrize("L,S", [(1, 1), (1, 12), (5, 12)])
def test_ling_attend_latent_parity(dtype, absorbed, L, S):
    B, H, r, p, dn = 1, LING.n_heads, LING.kv_lora_rank, LING.qk_rope_head_dim, LING.qk_nope_head_dim
    embed_q, unembed_out = _ml(dn, r, H, 11, dtype), _ml(r, 16, H, 12, dtype)
    q_nope, q_pe = _rand(13, (B, H, L, dn), dtype), _rand(14, (B, H, L, p), dtype)
    kv, kpe = _rand(15, (B, 1, S, r), dtype), _rand(16, (B, 1, S, p), dtype)
    scale = _legacy_ling_scale(24, {"factor": 40, "mscale_all_dim": 1.0})
    for mask in _masks(L, S):
        _same(mla.score_bias_attend(LING, q_nope, q_pe, kv, kpe, mask, embed_q=embed_q,
                                    unembed_out=unembed_out, absorbed=absorbed, scale=scale, cache=None),
              _legacy_ling_attend_latent(scale, embed_q, unembed_out, absorbed, q_nope, q_pe, kv, kpe,
                                         mask, None))


@pytest.mark.parametrize("dtype", DTYPES)
def test_ling_tail_and_tree_row_parity(dtype):
    B, H, r, p, dn, S = 1, 4, 32, 8, 16, 10
    embed_q, unembed_out = _ml(dn, r, H, 21, dtype), _ml(r, 16, H, 22, dtype)
    q_nope, q_pe = _rand(23, (B, H, 1, dn), dtype), _rand(24, (B, H, 1, p), dtype)
    kv, kpe = _rand(25, (B, 1, S, r), dtype), _rand(26, (B, 1, S, p), dtype)
    scale = 24 ** -0.5
    for mask in (None, "causal", _causal(4, S)[None, None], next(iter(list(_masks(4, S))[2:]))):
        last = mask[..., -1:, :] if mask is not None and not isinstance(mask, str) else None
        got = unembed_out(mla.absorbed_attention(LING, embed_q(q_nope), q_pe, kv, kpe, last,
                                                 scale=scale, cache=None)).transpose(0, 2, 1, 3)
        _same(got, _legacy_ling_tail(scale, embed_q, unembed_out, q_nope, q_pe, kv, kpe, mask, None))
    got = unembed_out(mla.absorbed_attention(LING, embed_q(q_nope), q_pe, kv, kpe, None,
                                             scale=scale, cache=None))
    _same(got, _legacy_ling_tree_row(scale, embed_q, unembed_out, q_nope, q_pe, kv, kpe, None))


# ------------------------------------------------------------------ Kimi Linear

def _kimi_new(q, latent, shared, mask, policy, *, w, n, scale, key_tile, single_score_bytes):
    qn, qs = mx.split(q, [n], -1)
    qn = qn @ w[:, :n]
    if policy == "latent_single" and mla.latent_single_fits(q, latent, single_score_bytes):
        return mla.latent_softmax_single(qn, qs, latent, shared, mask, scale=scale,
                                         out_dtype=q.dtype) @ w[:, n:].swapaxes(-1, -2)
    return mla.latent_softmax_tiled(qn, qs, latent, shared, mask, scale=scale, out_dtype=q.dtype,
                                    key_tile=key_tile,
                                    finish=lambda result: result @ w[:, n:].swapaxes(-1, -2))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("policy,score_bytes", [("latent_single", 256 << 20), ("latent_single", 16),
                                                ("latent_tiled", 256 << 20)])
@pytest.mark.parametrize("L,S", [(1, 1), (1, 13), (3, 13), (70, 75)])
def test_kimi_latent_laws_parity(dtype, policy, score_bytes, L, S):
    B, H, n, r, v, c = 2, 4, 16, 8, 16, 32
    q = _rand(31, (B, H, L, n + r), dtype)
    w = _rand(32, (H, n + v, c), dtype, 0.1)
    latent, shared = _rand(33, (B, 1, S, c), dtype), _rand(34, (B, 1, S, r), dtype)
    scale = (n + r) ** -0.5
    masks = list(_masks(L, S)) + ["causal"]
    masks.append(mx.zeros((1, 1, L, S), mx.bool_))  # every row padding
    for mask in masks:
        for key_tile in (5, 2048):
            kw = dict(w=w, n=n, scale=scale, key_tile=key_tile, single_score_bytes=score_bytes)
            _same(_kimi_new(q, latent, shared, mask, policy, **kw),
                  _legacy_kimi_latent(q, latent, shared, mask, policy, rank=c, **kw))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L,S", [(1, 6), (6, 6), (3, 9)])
def test_kimi_expanded_concat_parity(dtype, L, S):
    B, H, n, r, v = 1, 4, 16, 8, 16
    q = _rand(41, (B, H, L, n + r), dtype)
    kv = _rand(42, (B, H, S, n + v), dtype)
    shared = _rand(43, (B, 1, S, r), dtype)
    scale = (n + r) ** -0.5
    for mask in [None, "causal", _causal(L, S)[None, None]]:
        _same(mla.expanded_concat_attention(q, kv, shared, mask, scale=scale, nope=n),
              _legacy_kimi_expanded(q, kv, shared, mask, h=H, n=n, r=r, v=v, scale=scale, d256=False))


def test_kimi_expanded_d256_pad_builds_same_operands(monkeypatch):
    """The force_fused D256 kernel is GPU-only: compare the operands it gets."""
    seen = []

    def capture(q, k, v, *, scale, mask, force_fused=False, **_):
        seen.append((q, k, v, scale, mask, force_fused))
        return v
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", capture)
    B, H, n, r, v, L, S = 1, 4, 16, 8, 16, 2, 5
    q = _rand(44, (B, H, L, n + r), mx.bfloat16)
    kv = _rand(45, (B, H, S, n + v), mx.bfloat16)
    shared = _rand(46, (B, 1, S, r), mx.bfloat16)
    got = mla.expanded_concat_attention(q, kv, shared, "causal", scale=0.2, nope=n, pad_to=256)
    want = _legacy_kimi_expanded(q, kv, shared, "causal", h=H, n=n, r=r, v=v, scale=0.2, d256=True)
    _same(got, want)
    (a, b) = seen
    for x, y in zip(a[:3], b[:3]):
        _same(x, y)
    assert a[3:] == b[3:]


# ------------------------------------------------------------------ HY / GLM fused buffer

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("budget,bounded", [(1 << 27, True), (40, True), (40, False), (1, True)])
def test_hy_mla_attend_parity(dtype, budget, bounded):
    N, H, L, T, r, p = 2, 4, 6, 9, 32, 8
    q = _rand(51, (N, H, L, r + p), dtype)
    kv = _rand(52, (N, 1, T, r + p), dtype)
    sinks = _rand(53, (H,), mx.float32)
    for mask in [None, "causal", _causal(L, T)[None, None], list(_masks(L, T))[2]]:
        _same(mla.latent_attention(q, kv, mask, scale=0.17, sinks=sinks, value_dims=r,
                                   score_budget=budget, bounded=bounded),
              _legacy_hy_mla_attend(q, kv, mask, sinks, 0.17, r, budget, bounded))


def test_hy_wrapper_reads_module_globals(monkeypatch):
    from mlx2.runtime.models import hy_v4_flash as hy

    q = _rand(54, (1, 4, 5, 40), mx.bfloat16)
    kv = _rand(55, (1, 1, 7, 40), mx.bfloat16)
    sinks = _rand(56, (4,), mx.float32)
    mask = _causal(5, 7)[None, None]
    monkeypatch.setattr(hy, "SCORE_BUDGET", 30)
    _same(hy.mla_attend(q, kv, mask, sinks, 0.2, 32),
          _legacy_hy_mla_attend(q, kv, mask, sinks, 0.2, 32, 30, hy.BOUNDED_SCORE_BLOCKS))


@pytest.mark.parametrize("dtype", (mx.float32, mx.float16))
def test_glm53_dense_piece_parity(dtype):
    from mlx2.runtime.models import glm53_gguf

    q = _rand(61, (1, 8, 4, 64), dtype)
    keys = _rand(62, (1, 1, 10, 64), dtype)
    for mask in [None, _causal(4, 10)[None, None], list(_masks(4, 10))[2]]:
        _same(mla.latent_attention(q, keys, mask, scale=glm53_gguf.MLA_SPEC.softmax_scale),
              mx.fast.scaled_dot_product_attention(q, keys, keys, scale=256**-0.5, mask=mask))
    assert glm53_gguf.MLA_SPEC.softmax_scale == 256 ** -0.5


# ------------------------------------------------------------------ Kimi K3.1

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset,t,rows", [(0, 5, 2), (3, 4, 16), (0, 1, 1)])
@pytest.mark.parametrize("sparse", [False, True])
def test_k31_expanded_rows_parity(dtype, offset, t, rows, sparse):
    h, dn, dr, dv = 4, 16, 8, 16
    n = offset + t
    q = _rand(71, (t, h, dn + dr), dtype)
    kv = _rand(72, (n, h, dn + dv), dtype)
    k_pe = _rand(73, (n, dr), dtype)
    pattern = mx.random.uniform(shape=(t, n), key=mx.random.key(74)) > 0.3
    keep = (lambda r0, r1: pattern[r0:r1, :offset + r1]) if sparse else None
    got = mla.expanded_rows_softmax(q, kv, k_pe, nope=dn, scale=(dn + dr) ** -0.5, offset=offset,
                                    rows=rows, out_dtype=dtype, keep=keep).reshape(t, h * dv)
    _same(got, _legacy_k31_loop(q, kv, k_pe, dn=dn, dr=dr, dv=dv, h=h, offset=offset, rows=rows,
                                out_dtype=dtype, keep=keep))


def test_k31_spec_scale():
    from mlx2.runtime.models import kimi_k31

    cfg = type("C", (), dict(num_attention_heads=4, q_lora_rank=8, kv_lora_rank=32, qk_nope_head_dim=128,
                             qk_rope_head_dim=64, v_head_dim=128))()
    spec = kimi_k31.mla_spec(cfg)
    assert spec.softmax_scale == (128 + 64) ** -0.5 and spec.position == "nope"
    assert kimi_k31.mla_spec(cfg) is spec  # built once per config


# ------------------------------------------------------------------ registry

def test_registries_refuse_on_cpu_and_geometry():
    kimi = mla.MLASpec(n_heads=32, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
                       v_head_dim=128, pe_form="split_fp32")
    shape = dict(rows=1, batch=1, mask="none", dtype=mx.bfloat16, context=100)
    fn, receipt = mla.ABSORBED_DECODE.select(kimi, device="cpu", **shape)
    assert receipt.selected == "reference" and fn is mla.absorbed_attention
    assert set(receipt.refusals) == {"xing_split_k", "kimi_fused_split_k", "kimi_latent_flash"}
    _, receipt = mla.ABSORBED_DECODE.select(kimi, device="gpu", **shape)
    assert receipt.selected == "kimi_fused_split_k"
    raon = replace(kimi, n_heads=40)
    _, receipt = mla.ABSORBED_DECODE.select(raon, device="gpu", **shape)
    assert receipt.refusals["kimi_latent_flash"].startswith("geometry: n_heads=40")
    _, receipt = mla.ABSORBED_DECODE.select(raon, device="gpu", probed_heads=frozenset({40}), **shape)
    assert receipt.selected == "kimi_latent_flash"  # grouped kernel, the caller's probed heads
    odd = replace(kimi, n_heads=24)
    _, receipt = mla.ABSORBED_DECODE.select(odd, device="gpu", **shape)
    assert receipt.refusals["kimi_latent_flash"].startswith("geometry: n_heads=24")
    with pytest.raises(RuntimeError):
        mla.ABSORBED_DECODE.select(odd, policy="kimi_latent_flash", device="gpu", **shape)
    _, receipt = mla.EXPANDED_PREFILL.select(raon, device="gpu", dtype=mx.bfloat16, mask="causal")
    assert receipt.selected == "kimi_flash192"
    xing = mla.MLASpec(n_heads=16, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
                       v_head_dim=128)
    _, receipt = mla.ABSORBED_DECODE.select(xing, device="gpu", **dict(shape, rows=3))
    assert receipt.selected == "reference" and receipt.refusals["xing_split_k"].startswith("mask:")
    _, receipt = mla.ABSORBED_DECODE.select(xing, device="gpu", **dict(shape, rows=3, mask="array"))
    assert receipt.selected == "xing_split_k"
    from mlx2.runtime.models import glm53_gguf
    _, receipt = mla.LATENT_GATHERED.select(glm53_gguf.MLA_SPEC, device="gpu")
    assert receipt.selected == "glm53_gathered_v1"
    _, receipt = mla.LATENT_GATHERED.select(glm53_gguf.MLA_SPEC, device="cpu")
    assert receipt.selected == "reference"


def test_registry_does_not_import_model_modules():
    import inspect

    registry_src = inspect.getsource(mla).split("# Fast-path registries")[1]
    assert "raon1_mini" not in registry_src and "192 ** -0.5" in registry_src
    assert "_admits_kimi_fused_split_k" not in registry_src


def test_refusal_codes_are_bounded_and_counted():
    from mlx2.runtime.technologies import dispatch

    allowed = {"device", "dtype", "geometry", "shape", "mask", "scale"}
    kimi = mla.MLASpec(n_heads=7, kv_lora_rank=64, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)
    dispatch.reset_counters()
    for device in ("cpu", "gpu"):
        for dtype in (mx.float32, mx.bfloat16):
            for registry in (mla.ABSORBED_DECODE, mla.EXPANDED_PREFILL, mla.LATENT_GATHERED):
                _, receipt = registry.select(kimi, device=device, dtype=dtype, rows=2, batch=1,
                                             mask="array", context=10)
                for reason in receipt.refusals.values():
                    assert dispatch.refusal_code(reason) in allowed, reason
    snap = dispatch.counters_snapshot()
    assert {r["technology"] for r in snap["selections"]} == {"mla"}
    assert {r["operation"] for r in snap["selections"]} == {
        "absorbed_decode", "expanded_prefill", "latent_gathered"}
    assert {r["code"] for r in snap["refusals"]} <= allowed


def test_registered_kernel_entry_points_exist():
    for registry in (mla.ABSORBED_DECODE, mla.EXPANDED_PREFILL, mla.LATENT_GATHERED):
        for path in registry._paths:
            assert callable(getattr(path.run.load(), path.run.__name__))


# ------------------------------------------------------------------ model wiring

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
LING_CONFIG = dict(  # tests/test_ling_adapter_cpu.py tiny_ling
    model_type="bailing_hybrid", vocab_size=128, hidden_size=128,
    intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=3,
    num_attention_heads=2, num_key_value_heads=2, num_experts=4,
    num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
    first_k_dense_replace=1, layer_group_size=2, max_position_embeddings=64,
    rms_norm_eps=1e-6, rope_theta=10000, routed_scaling_factor=1.0,
    head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32,
    qk_nope_head_dim=64, v_head_dim=64, no_kda_lora=True,
    gated_attention_proj_granularity_type="head_wise", eos_token_id=127,
    kda_safe_gate=True, kda_lower_bound=-5.0,
)


@pytest.mark.parametrize("scaling", [None, {"type": "yarn", "factor": 40, "mscale_all_dim": 1.0},
                                     {"factor": 7.3, "mscale_all_dim": 0.3}])
def test_model_modules_build_specs_with_their_legacy_scales(scaling):
    """Real model constructors (tiny configs) carry the legacy softmax scale."""
    import json

    from mlx2.runtime.models import bailing_hybrid, kimi_linear, xing4_0

    xcfg = json.loads((ROOT / "tests/fixtures/xing4_0_tiny/config.json").read_text())
    if scaling is not None:
        xcfg["rope_scaling"] = {**xcfg["rope_scaling"], **scaling}
    xargs = xing4_0.ModelArgs.from_dict(xcfg)
    attn = xing4_0.Xing4_0Attention(xargs)
    want = _legacy_xing_scale(xargs.qk_nope_head_dim + xargs.qk_rope_head_dim, xargs.rope_scaling)
    assert attn.scale == attn.mla_spec.softmax_scale == want
    assert (attn.mla_spec.n_heads, attn.mla_spec.kv_lora_rank) == (xargs.num_attention_heads,
                                                                   xargs.kv_lora_rank)

    largs = bailing_hybrid.ModelArgs.from_dict({**LING_CONFIG, "rope_scaling": scaling})
    ling = bailing_hybrid.MultiLatentAttention(largs)
    assert ling.scale == ling.mla_spec.softmax_scale == _legacy_ling_scale(
        largs.qk_nope_head_dim + largs.qk_rope_head_dim, largs.rope_scaling)

    kargs = kimi_linear.ModelArgs(  # tests/test_kimi_linear.py args()
        model_type="kimi_linear", vocab_size=64, hidden_size=32,
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=1, layer_group_size=4, max_position_embeddings=1048576,
        rms_norm_eps=1e-5, rope_theta=10000., routed_scaling_factor=2.446,
        head_dim=72, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=16,
        v_head_dim=16, rope_scaling=scaling,
        linear_attn_config={"full_attn_layers": [4], "kda_layers": [1, 2, 3], "num_heads": 2,
                            "head_dim": 128, "short_conv_kernel_size": 4})
    kimi = kimi_linear.MultiLatentAttention(kargs)
    n, r = kargs.qk_nope_head_dim, kargs.qk_rope_head_dim
    assert kimi.scale == kimi.mla_spec.softmax_scale == (n + r) ** -0.5
    assert kimi.mla_spec.position == "nope"
