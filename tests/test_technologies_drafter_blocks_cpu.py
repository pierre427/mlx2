"""Exact CPU parity: drafter blocks / MTP fusion vs the copies they replaced.

Each ``_legacy_*`` below is the pre-consolidation code moved here verbatim
(the class or function body as it stood in the named file at d46e7a3ea).
Parity is ``mx.array_equal`` (exact), on tiny random weights, in every dtype
the drafter or head runs in.  Weight-name tests load checkpoint-named tensors
strictly into the migrated modules.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.cohere_eagle import EagleWindowKVCache  # noqa: E402
from mlx2.runtime.technologies import drafter_blocks as db  # noqa: E402

DTYPES = (mx.float32, mx.bfloat16, mx.float16)


def _equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert mx.array_equal(a, b).item(), float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _copy_weights(dst, src):
    """Load ``src``'s parameters into ``dst`` by name, strictly."""
    dst.load_weights(tree_flatten(src.parameters()), strict=True)


# --------------------------------------------------------------- legacy MLPs

class _legacy_dflash_base_Qwen3MLP(nn.Module):
    """runtime/drafters/dflash_base.py (with its ``swiglu``)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        def swiglu(gate, value):
            return nn.silu(gate) * value
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class _legacy_muse_dflash_MLP(nn.Module):
    """runtime/drafters/muse_dflash.py MuseDFlashMLP (laguna_dflash.LagunaDFlashMLP
    and eagle3.Eagle3MLP are the same text)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _legacy_cohere_eagle_MLP(nn.Module):
    """runtime/drafters/cohere_eagle.py CohereEagleMLP."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=True)
        self.up_proj = nn.Linear(dim, hidden, bias=True)
        self.down_proj = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _legacy_inkling2_dspark_DraftMLP(nn.Module):
    """runtime/drafters/inkling2_dspark.py DraftMLP (config -> dims)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        gate = self.gate_proj(x).astype(mx.float32)
        up = self.up_proj(x).astype(mx.float32)
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("legacy, spec", [
    (_legacy_dflash_base_Qwen3MLP, db.GatedMLPSpec()),
    (_legacy_muse_dflash_MLP, db.GatedMLPSpec()),
    (_legacy_cohere_eagle_MLP, db.GatedMLPSpec(bias=True)),
    (_legacy_inkling2_dspark_DraftMLP, db.GatedMLPSpec(gate_law="silu_mul_fp32")),
])
def test_gated_mlp_matches_legacy(legacy, spec, dtype):
    mx.random.seed(1)
    old = legacy(16, 24)
    new = db.GatedMLP(16, 24, spec)
    _copy_weights(new, old)
    old.set_dtype(dtype)
    new.set_dtype(dtype)
    x = (mx.random.normal((3, 5, 16)) * 3).astype(dtype)
    x = mx.concatenate([x, mx.zeros((3, 1, 16), dtype)], axis=1)
    _equal(new(x), old(x))


def test_gated_mlp_spec_refuses_unknown_gate_law():
    with pytest.raises(ValueError):
        db.GatedMLPSpec(gate_law="geglu")


def test_migrated_mlp_classes_keep_names_and_laws():
    from mlx2.runtime.drafters import cohere_eagle, dflash_base, eagle3, inkling2_dspark
    from mlx2.runtime.drafters import laguna_dflash, muse_dflash

    for cls in (dflash_base.Qwen3MLP, muse_dflash.MuseDFlashMLP, laguna_dflash.LagunaDFlashMLP,
                eagle3.Eagle3MLP):
        m = cls(8, 12)
        assert m.mlp_spec == db.GatedMLPSpec()
        assert {k for k, _ in tree_flatten(m.parameters())} == {
            "gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    m = cohere_eagle.CohereEagleMLP(8, 12)
    assert m.mlp_spec.bias and "down_proj.bias" in dict(tree_flatten(m.parameters()))

    class C:
        hidden_size, intermediate_size = 8, 12
    assert inkling2_dspark.DraftMLP(C()).mlp_spec.gate_law == "silu_mul_fp32"


# ------------------------------------------------------ legacy block attention

class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _legacy_muse_dflash_Attention(nn.Module):
    """runtime/drafters/muse_dflash.py MuseDFlashAttention."""

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def _kv(self, x):
        batch, length, _ = x.shape
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        return self.k_norm(k).transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def context_kv(self, context, cache):
        k, v = self._kv(context)
        cache.append(self.rope(k, offset=cache.offset), v)

    def block(self, x, cache):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=cache.offset)
        k, v = self._kv(x)
        k = self.rope(k, offset=cache.offset)
        ctx_k, ctx_v = cache.keys_and_values()
        resident = int(ctx_k.shape[2])
        keys = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2) if resident else k
        values = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2) if resident else v
        if resident:
            rows = mx.arange(length)[:, None]
            columns = mx.arange(resident)[None, :]
            context = columns > rows + resident - self.window
            block = mx.ones((length, length), dtype=mx.bool_)
            mask = mx.concatenate([context, block], axis=-1)
        else:
            mask = None
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class _legacy_laguna_dflash_Attention(nn.Module):
    """runtime/drafters/laguna_dflash.py LagunaDFlashAttention."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        dim = config.hidden_size
        self.qkv_proj = nn.Linear(dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.n_heads, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def split(self, x):
        batch, length, _ = x.shape
        qkv = self.qkv_proj(x)
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q = qkv[..., :q_size].reshape(batch, length, self.n_heads, self.head_dim)
        k = qkv[..., q_size:q_size + kv_size].reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = qkv[..., q_size + kv_size:].reshape(batch, length, self.n_kv_heads, self.head_dim)
        return (
            self.q_norm(q).transpose(0, 2, 1, 3),
            self.k_norm(k).transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
        )

    def context_kv(self, context, cache):
        _, k, v = self.split(context)
        cache.append(self.rope(k, offset=cache.offset), v)

    def block(self, x, cache):
        batch, length, _ = x.shape
        q, k, v = self.split(x)
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        ctx_k, ctx_v = cache.keys_and_values()
        keys = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2) if ctx_k.shape[2] else k
        values = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2) if ctx_v.shape[2] else v
        mask = mx.concatenate(
            [mx.ones((length, ctx_k.shape[2]), dtype=mx.bool_),
             mx.tril(mx.ones((length, length), dtype=mx.bool_))], axis=-1
        )
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
        out = (out.reshape(batch, length, self.n_heads, self.head_dim) * gate[..., None]).reshape(
            batch, length, -1
        )
        return self.o_proj(out)


_BLOCK_CFG = _Cfg(hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                  rms_norm_eps=1e-6, rope_theta=10000.0, sliding_window=5)


def _block_attention(kind):
    from mlx2.runtime.drafters.laguna_dflash import LagunaDFlashAttention
    from mlx2.runtime.drafters.muse_dflash import MuseDFlashAttention

    if kind == "muse":
        return _legacy_muse_dflash_Attention(_BLOCK_CFG), MuseDFlashAttention(_BLOCK_CFG)
    return _legacy_laguna_dflash_Attention(_BLOCK_CFG), LagunaDFlashAttention(_BLOCK_CFG)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["muse", "laguna"])
def test_block_attention_matches_legacy(kind, dtype):
    mx.random.seed(2)
    old, new = _block_attention(kind)
    _copy_weights(new, old)
    old.set_dtype(dtype)
    new.set_dtype(dtype)
    cache_old = EagleWindowKVCache(4, 2, 4)
    cache_new = EagleWindowKVCache(4, 2, 4)
    # Empty context (no mask / zero-width mask), then growing past the ring.
    for step, ctx_len in enumerate((0, 3, 1, 4)):
        x = mx.random.normal((1, 6, 16)).astype(dtype)
        _equal(new.block(x, cache_new), old.block(x, cache_old))
        if ctx_len:
            ctx = mx.random.normal((1, ctx_len, 16)).astype(dtype)
            old.context_kv(ctx, cache_old)
            new.context_kv(ctx, cache_new)
            ko, vo = cache_old.keys_and_values()
            kn, vn = cache_new.keys_and_values()
            _equal(kn, ko)
            _equal(vn, vo)
    # Zeros and tied rows.
    x = mx.zeros((1, 6, 16), dtype)
    _equal(new.block(x, cache_new), old.block(x, cache_old))


def test_block_attention_weight_names_load_strictly():
    _, muse = _block_attention("muse")
    _, laguna = _block_attention("laguna")
    assert {k for k, _ in tree_flatten(muse.parameters())} == {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "q_norm.weight", "k_norm.weight"}
    assert {k for k, _ in tree_flatten(laguna.parameters())} == {
        "qkv_proj.weight", "o_proj.weight", "g_proj.weight", "q_norm.weight", "k_norm.weight"}


# ------------------------------------------------------ legacy chain attention

class _legacy_eagle3_Attention(nn.Module):
    """runtime/drafters/eagle3.py Eagle3Attention."""

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(2 * dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def project(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def attend(self, q, k, v, *, position, prior_k, prior_v):
        length = q.shape[2]
        q = self.rope(q, offset=position)
        k = self.rope(k, offset=position)
        keys = mx.concatenate([prior_k, k.astype(prior_k.dtype)], axis=2) if prior_k.shape[2] else k
        values = mx.concatenate([prior_v, v.astype(prior_v.dtype)], axis=2) if prior_v.shape[2] else v
        start = position - prior_k.shape[2]
        key_pos = mx.arange(start, position + length)
        query_pos = mx.arange(position, position + length)
        mask = key_pos[None, :] <= query_pos[:, None]
        out = mx.fast.scaled_dot_product_attention(
            q, keys.astype(q.dtype), values.astype(q.dtype), scale=self.scale, mask=mask
        )
        return out, k


class _legacy_cohere_eagle_Attention(nn.Module):
    """runtime/drafters/cohere_eagle.py CohereEagleAttention."""

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.window = config.sliding_window
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=True)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=config.rope_theta)

    def project(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def attend(self, q, k, v, *, position, prior_k, prior_v, prior_start):
        length = q.shape[2]
        q = self.rope(q, offset=position)
        k = self.rope(k, offset=position)
        keys = mx.concatenate([prior_k, k.astype(prior_k.dtype)], axis=2) if prior_k.shape[2] else k
        values = mx.concatenate([prior_v, v.astype(prior_v.dtype)], axis=2) if prior_v.shape[2] else v
        key_pos = mx.arange(prior_start, position + length)
        query_pos = mx.arange(position, position + length)
        delta = query_pos[:, None] - key_pos[None, :]
        mask = (delta >= 0) & (delta <= self.window)
        out = mx.fast.scaled_dot_product_attention(
            q, keys.astype(q.dtype), values.astype(q.dtype), scale=self.scale, mask=mask
        )
        return out, k


def _legacy_eagle3_rows(attn, q, k, v, rows, *, commit, chain):
    """The per-row loop of runtime/drafters/eagle3.py Eagle3DraftModel._forward."""
    outputs = []
    for row, caches in enumerate(rows):
        cache = caches[0]
        prior_k, prior_v = cache.keys_and_values()
        local = None if commit else chain[row]
        if local is not None and local[0] is not None:
            prior_k = mx.concatenate([prior_k, local[0]], axis=2) if prior_k.shape[2] else local[0]
            prior_v = mx.concatenate([prior_v, local[1]], axis=2) if prior_v.shape[2] else local[1]
        position = cache.offset + (0 if local is None or local[0] is None else local[0].shape[2])
        out, rotated = attn.attend(
            q[row:row + 1], k[row:row + 1], v[row:row + 1],
            position=position, prior_k=prior_k, prior_v=prior_v,
        )
        outputs.append(out)
        if commit:
            cache.append(rotated, v[row:row + 1])
        elif chain[row][0] is None:
            chain[row] = (rotated, v[row:row + 1])
        else:
            old = chain[row]
            chain[row] = (
                mx.concatenate([old[0], rotated], axis=2),
                mx.concatenate([old[1], v[row:row + 1]], axis=2),
            )
    return mx.concatenate(outputs, axis=0)


def _legacy_cohere_rows(attn, q, k, v, rows, index, *, commit, chain):
    """The per-row loop of runtime/drafters/cohere_eagle.py CohereEagleDraftModel._forward."""
    outputs = []
    for row, caches in enumerate(rows):
        cache = caches[index]
        prior_k, prior_v = cache.keys_and_values()
        local = None if commit else chain[row][index]
        if local is not None and local[0] is not None:
            prior_k = mx.concatenate([prior_k, local[0]], axis=2) if prior_k.shape[2] else local[0]
            prior_v = mx.concatenate([prior_v, local[1]], axis=2) if prior_v.shape[2] else local[1]
        position = cache.offset + (0 if local is None or local[0] is None else local[0].shape[2])
        out, rotated = attn.attend(
            q[row:row + 1], k[row:row + 1], v[row:row + 1],
            position=position, prior_k=prior_k, prior_v=prior_v,
            prior_start=position - prior_k.shape[2],
        )
        outputs.append(out)
        if commit:
            cache.append(rotated, v[row:row + 1])
        else:
            old = chain[row][index]
            if old[0] is None:
                chain[row][index] = (rotated, v[row:row + 1])
            else:
                chain[row][index] = (
                    mx.concatenate([old[0], rotated], axis=2),
                    mx.concatenate([old[1], v[row:row + 1]], axis=2),
                )
    return mx.concatenate(outputs, axis=0)


_CHAIN_CFG = _Cfg(hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                  rms_norm_eps=1e-6, rope_theta=10000.0, sliding_window=3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["eagle3", "cohere"])
def test_chain_attention_and_row_loop_match_legacy(kind, dtype):
    from mlx2.runtime.drafters.cohere_eagle import CohereEagleAttention
    from mlx2.runtime.drafters.eagle3 import Eagle3Attention

    mx.random.seed(3)
    if kind == "eagle3":
        old, new, width = _legacy_eagle3_Attention(_CHAIN_CFG), Eagle3Attention(_CHAIN_CFG), 32
    else:
        old, new, width = _legacy_cohere_eagle_Attention(_CHAIN_CFG), CohereEagleAttention(_CHAIN_CFG), 16
    _copy_weights(new, old)
    old.set_dtype(dtype)
    new.set_dtype(dtype)
    B = 2
    rows_old = [[EagleWindowKVCache(4, 2, 4)] for _ in range(B)]
    rows_new = [[EagleWindowKVCache(4, 2, 4)] for _ in range(B)]
    chain_old = [[(None, None)] if kind == "cohere" else (None, None) for _ in range(B)]
    locals_new = [(None, None)] * B
    for length, commit in ((3, True), (1, False), (2, False), (4, True), (1, False)):
        x = mx.random.normal((B, length, width)).astype(dtype)
        q, k, v = old.project(x)
        qn, kn, vn = new.project(x)
        _equal(qn, q)
        if kind == "eagle3":
            ref = _legacy_eagle3_rows(old, q, k, v, rows_old, commit=commit, chain=chain_old)
        else:
            ref = _legacy_cohere_rows(old, q, k, v, rows_old, 0, commit=commit, chain=chain_old)
        got, new_locals = db.chain_attend_rows(
            new, qn, kn, vn, [r[0] for r in rows_new], locals_new, commit=commit)
        _equal(got, ref)
        if commit:
            # A commit ends the round: round-local KV is discarded.
            chain_old = [[(None, None)] if kind == "cohere" else (None, None) for _ in range(B)]
            locals_new = [(None, None)] * B
        else:
            locals_new = new_locals
            for row in range(B):
                expect = chain_old[row][0] if kind == "cohere" else chain_old[row]
                _equal(locals_new[row][0], expect[0])
                _equal(locals_new[row][1], expect[1])
        for ro, rn in zip(rows_old, rows_new):
            _equal(rn[0].keys_and_values()[0], ro[0].keys_and_values()[0])


def test_chain_attention_weight_names():
    from mlx2.runtime.drafters.cohere_eagle import CohereEagleAttention
    from mlx2.runtime.drafters.eagle3 import Eagle3Attention

    e = dict(tree_flatten(Eagle3Attention(_CHAIN_CFG).parameters()))
    assert set(e) == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}
    assert e["q_proj.weight"].shape == (16, 32)
    c = dict(tree_flatten(CohereEagleAttention(_CHAIN_CFG).parameters()))
    assert set(c) == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "o_proj.bias"}


# ------------------------------------------------------ legacy layer loop / taps

def _legacy_block_layers(layers, x, rows):
    """The layer loop of muse_dflash/laguna_dflash ``_block_logits`` (identical)."""
    for index, layer in enumerate(layers):
        h = layer.input_layernorm(x)
        attn = mx.concatenate(
            [layer.self_attn.block(h[row:row + 1], caches[index]) for row, caches in enumerate(rows)],
            axis=0,
        )
        x = x + attn
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x


def _legacy_laguna_context(aux_hidden_norms, fc, hidden_norm, hidden):
    """runtime/drafters/laguna_dflash.py LagunaDFlashDraftModel._context."""
    parts = mx.split(hidden, len(aux_hidden_norms), axis=-1)
    normed = [norm(part) for norm, part in zip(aux_hidden_norms, parts)]
    return hidden_norm(fc(mx.concatenate(normed, axis=-1)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_tap_injection_matches_legacy(dtype):
    mx.random.seed(4)
    norms = [nn.RMSNorm(8) for _ in range(3)]
    for n in norms:
        n.weight = mx.random.normal((8,))
    fc, hn = nn.Linear(24, 8, bias=False), nn.RMSNorm(8)
    mods = [*norms, fc, hn]
    for m in mods:
        m.set_dtype(dtype)
    hidden = mx.random.normal((2, 5, 24)).astype(dtype)
    spec = db.TapInjectionSpec(per_tap_norm=True)
    _equal(db.inject_target_taps(spec, hidden, fc=fc, hidden_norm=hn, tap_norms=norms),
           _legacy_laguna_context(norms, fc, hn, hidden))
    plain = db.TapInjectionSpec()
    _equal(db.inject_target_taps(plain, hidden, fc=fc, hidden_norm=hn), hn(fc(hidden)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_prenorm_block_layers_match_legacy(dtype):
    from mlx2.runtime.drafters.laguna_dflash import LagunaDFlashLayer

    cfg = _Cfg(**vars(_BLOCK_CFG), intermediate_size=24)
    mx.random.seed(5)
    layers = [LagunaDFlashLayer(cfg) for _ in range(2)]
    for layer in layers:
        layer.set_dtype(dtype)
    rows = [[EagleWindowKVCache(4, 2, 4) for _ in layers] for _ in range(2)]
    ctx = mx.random.normal((1, 3, 16)).astype(dtype)
    for caches in rows:
        for layer, entry in zip(layers, caches):
            layer.self_attn.context_kv(ctx, entry)
    x = mx.random.normal((2, 6, 16)).astype(dtype)
    _equal(db.prenorm_block_layers(layers, x, rows), _legacy_block_layers(layers, x, rows))


# ----------------------------------------------------------- MTP head fusion

def _legacy_bailing_hybrid_fuse(self, embeds, hidden):
    """runtime/models/bailing_hybrid.py MTPLayer.__call__ (fusion line)."""
    return self.eh_proj(mx.concatenate([self.enorm(embeds), self.hnorm(hidden)], axis=-1))


def _legacy_nemotron_h_fuse(first, embeds, hidden):
    """runtime/models/nemotron_h.py mtp_step (fusion lines; embeddings precomputed)."""
    e = first.enorm(embeds)
    h = first.hnorm(hidden)
    return first.eh_proj(mx.concatenate([e, h], axis=-1))


def _legacy_xing4_0_fuse(layer, embeds, hidden):
    """runtime/models/xing4_0.py mtp_step (fusion lines)."""
    e = layer.enorm(embeds)
    h = layer.hnorm(hidden.astype(e.dtype))
    return layer.eh_proj(mx.concatenate([e, h], axis=-1))


def _legacy_hy_v4_flash_DFlyFusion(self, anchors, target_hidden):
    """runtime/models/hy_v4_flash.py DFlyFusion.__call__."""
    fused = self.eh_proj(mx.concatenate([self.enorm(anchors), self.hnorm(target_hidden)], axis=-1))
    return anchors + self.gate * fused


def _legacy_qwen_pre_fc_fuse(mtp, embeds, hidden):
    """runtime/models/qwen38_27b.py / qwen36_35b.py mtp_step and mellum.py MTPModule.fuse."""
    e = mtp.pre_fc_norm_embedding(embeds)
    h = mtp.pre_fc_norm_hidden(hidden)
    return mtp.fc(mx.concatenate([e, h], axis=-1))


def _legacy_inkling_mtp_fuse(self, hidden, embedded):
    """runtime/models/inkling/mtp.py InklingMTPDepth.__call__ (fusion)."""
    return self.input_proj(
        mx.concatenate(
            [self.hidden_norm(hidden), self.embed_norm(embedded)], axis=-1
        )
    )


def _legacy_maxwell_moe_fuse(self, streams, embeddings):
    """runtime/models/maxwell_moe.py MTP.__call__ (fusion)."""
    b, s, nc = streams.shape
    c = nc // self.n
    e = self.e_proj(self.enorm(embeddings))
    h = self.h_proj(self.hnorm(streams.reshape(b, s, self.n, c)))
    return (e[:, :, None, :] + h).reshape(b, s, nc)


def _legacy_qwen4_exp_fuse(self, embeddings, hidden):
    """runtime/models/qwen4_exp.py MTPModule.fuse."""
    embeddings = self.fc_embedding(self.pre_fc_norm_embedding(embeddings))
    hidden = self.pre_fc_norm_hidden(hidden).reshape(
        *hidden.shape[:-1], self.hc_count, self.hidden_size
    )
    hidden = self.fc_hidden(hidden)
    return (embeddings[..., None, :] + hidden).reshape(
        *hidden.shape[:-2], self.hc_count * self.hidden_size
    )


def _head(names, widths, dtype, d=8):
    m = nn.Module()
    for name, kind in names.items():
        if kind == "norm":
            m[name] = nn.RMSNorm(widths.get(name, d), eps=1e-6)
            m[name].weight = mx.random.normal(m[name].weight.shape)
        else:
            m[name] = nn.Linear(*kind, bias=False)
    m.set_dtype(dtype)
    return m


@pytest.mark.parametrize("dtype", DTYPES)
def test_nextn_fusion_matches_legacy_copies(dtype):
    mx.random.seed(6)
    m = _head({"enorm": "norm", "hnorm": "norm", "eh_proj": (16, 8)}, {}, dtype)
    m.gate = mx.random.normal((8,)).astype(dtype)
    e = (mx.random.normal((2, 3, 8)) * 4).astype(dtype)
    h = (mx.random.normal((2, 3, 8)) * 4).astype(dtype)
    got = db.fuse_mtp_module(db.NEXTN_FUSION, m, e, h)
    _equal(got, _legacy_bailing_hybrid_fuse(m, e, h))
    _equal(got, _legacy_nemotron_h_fuse(m, e, h))
    _equal(e + m.gate * got, _legacy_hy_v4_flash_DFlyFusion(m, e, h))
    xing = db.MTPFusionSpec(hidden_cast="embed_dtype")
    for hidden in (h, h.astype(mx.float32)):
        _equal(db.fuse_mtp_module(xing, m, e, hidden), _legacy_xing4_0_fuse(m, e, hidden))


@pytest.mark.parametrize("dtype", DTYPES)
def test_pre_fc_and_hidden_first_fusion_match_legacy(dtype):
    mx.random.seed(7)
    q = _head({"pre_fc_norm_embedding": "norm", "pre_fc_norm_hidden": "norm", "fc": (16, 8)}, {}, dtype)
    e = mx.random.normal((1, 4, 8)).astype(dtype)
    h = mx.random.normal((1, 4, 8)).astype(dtype)
    _equal(db.fuse_mtp_module(db.PRE_FC_NORM_FUSION, q, e, h), _legacy_qwen_pre_fc_fuse(q, e, h))
    ink = _head({"hidden_norm": "norm", "embed_norm": "norm", "input_proj": (16, 8)}, {}, dtype)
    _equal(db.fuse_mtp_module(db.HIDDEN_FIRST_FUSION, ink, e, h), _legacy_inkling_mtp_fuse(ink, h, e))


@pytest.mark.parametrize("n", [1, 3, 4])
@pytest.mark.parametrize("dtype", DTYPES)
def test_stream_sum_fusion_matches_legacy(dtype, n):
    from mlx2.runtime.models.maxwell_moe import RMSNorm as MaxwellRMSNorm
    from mlx2.runtime.models.qwen4_exp import GroupRMSNorm

    mx.random.seed(8)
    c = 8
    mw = nn.Module()
    mw.n = n
    mw.enorm, mw.hnorm = MaxwellRMSNorm(c, 1e-6), MaxwellRMSNorm(c, 1e-6)
    mw.e_proj, mw.h_proj = nn.Linear(c, c, bias=False), nn.Linear(c, c, bias=False)
    mw.set_dtype(dtype)
    streams = mx.random.normal((2, 3, n * c)).astype(dtype)
    emb = mx.random.normal((2, 3, c)).astype(dtype)
    _equal(db.fuse_mtp_module(db.STREAM_SUM_FUSION, mw, emb, streams),
           _legacy_maxwell_moe_fuse(mw, streams, emb))

    q4 = nn.Module()
    q4.hidden_size, q4.hc_count = c, n
    q4.pre_fc_norm_embedding = GroupRMSNorm(c, None, 1e-6)
    q4.pre_fc_norm_hidden = GroupRMSNorm(n * c, c, 1e-6)
    q4.fc_embedding, q4.fc_hidden = nn.Linear(c, c, bias=False), nn.Linear(c, c, bias=False)
    q4.set_dtype(dtype)
    _equal(db.fuse_mtp_module(db.STREAM_SUM_PRE_FC_FUSION, q4, emb, streams),
           _legacy_qwen4_exp_fuse(q4, emb, streams))


def test_mtp_fusion_spec_validation():
    with pytest.raises(ValueError):
        db.MTPFusionSpec(layout="sum")
    with pytest.raises(ValueError):
        db.MTPFusionSpec(concat_order=("embed", "embed"))
    with pytest.raises(ValueError):
        db.MTPFusionSpec(layout="stream_sum")


def test_draft_attention_spec_from_config_and_validation():
    spec = db.DraftAttentionSpec.from_config(_BLOCK_CFG, qkv_layout="packed", head_gate="softplus_fp32",
                                             mask_law="causal_block")
    assert (spec.hidden_size, spec.num_attention_heads, spec.sliding_window) == (16, 4, 5)
    with pytest.raises(ValueError):
        db.DraftAttentionSpec(16, 4, 2, 4, mask_law="causal_window")
    with pytest.raises(ValueError):
        db.DraftAttentionSpec(16, 4, 3, 4, mask_law="causal")
    with pytest.raises(ValueError):
        db.block_mask(db.DraftAttentionSpec(16, 4, 2, 4, mask_law="causal"), 2, 0)


def _legacy_glm53_gguf_mtp_fuse(w_enorm, w_hnorm, eh_proj, embedded, hidden, eps, rms):
    """runtime/models/glm53_gguf_mtp.py Glm53MTPHead.__call__ (fusion lines)."""
    e = rms(embedded, w_enorm, eps)
    h = rms(hidden, w_hnorm, eps)
    return eh_proj(mx.concatenate((e, h), axis=-1))


def _legacy_hy_v4_flash_DFlyHiddenCorrection(self, hidden, previous_embed):
    """runtime/models/hy_v4_flash.py DFlyHiddenCorrection.__call__."""
    x = mx.concatenate([self.hidden_norm(hidden), self.embed_norm(previous_embed)], axis=-1)
    return hidden + self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_functional_nextn_fusion_matches_glm53_gguf(dtype):
    from mlx2.runtime.models.glm53_gguf import rms

    mx.random.seed(9)
    we, wh = mx.random.normal((8,)).astype(dtype), mx.random.normal((8,)).astype(dtype)
    proj = nn.Linear(16, 8, bias=False)
    proj.set_dtype(dtype)
    e = mx.random.normal((1, 2, 8)).astype(dtype)
    h = mx.random.normal((1, 2, 8)).astype(dtype)
    got = db.fuse_mtp(db.NEXTN_FUSION, e, h, embed_norm=lambda v: rms(v, we, 1e-5),
                      hidden_norm=lambda v: rms(v, wh, 1e-5), projection=proj)
    _equal(got, _legacy_glm53_gguf_mtp_fuse(we, wh, proj, e, h, 1e-5, rms))


@pytest.mark.parametrize("dtype", DTYPES)
def test_dfly_fusion_and_hidden_correction_match_legacy(dtype):
    from mlx2.runtime.models.hy_v4_flash import DFlyFusion, DFlyHiddenCorrection

    mx.random.seed(10)
    fusion, corr = DFlyFusion(8, 1e-6), DFlyHiddenCorrection(8, 12, 1e-6)
    fusion.gate = mx.random.normal((8,))
    for m in (fusion, corr):
        m.set_dtype(dtype)
    a = mx.random.normal((2, 3, 8)).astype(dtype)
    t = mx.random.normal((2, 3, 8)).astype(dtype)
    _equal(fusion(a, t), _legacy_hy_v4_flash_DFlyFusion(fusion, a, t))
    _equal(corr(a, t), _legacy_hy_v4_flash_DFlyHiddenCorrection(corr, a, t))
    assert set(dict(tree_flatten(fusion.parameters()))) == {
        "enorm.weight", "hnorm.weight", "eh_proj.weight", "gate"}


@pytest.mark.parametrize("dtype", DTYPES)
def test_model_mtp_heads_load_checkpoint_names_and_match_legacy(dtype):
    """The migrated heads still load their checkpoint tensor names strictly
    and fuse exactly like the legacy text."""
    from mlx2.runtime.models import bailing_hybrid, mellum

    mx.random.seed(11)

    class Mellum(mellum.MTPModule):
        def __init__(self):  # fusion modules only; the decoder layer is not under test
            nn.Module.__init__(self)
            self.fc = nn.Linear(16, 8, bias=False)
            self.pre_fc_norm_embedding = nn.RMSNorm(8, eps=1e-6)
            self.pre_fc_norm_hidden = nn.RMSNorm(8, eps=1e-6)

    class Ling(bailing_hybrid.MTPLayer):
        def __init__(self):
            nn.Module.__init__(self)
            self.enorm, self.hnorm = nn.RMSNorm(8, eps=1e-6), nn.RMSNorm(8, eps=1e-6)
            self.eh_proj = nn.Linear(16, 8, bias=False)

    e = mx.random.normal((1, 3, 8)).astype(dtype)
    h = mx.random.normal((1, 3, 8)).astype(dtype)
    m = Mellum()
    m.load_weights([
        ("fc.weight", mx.random.normal((8, 16))),
        ("pre_fc_norm_embedding.weight", mx.random.normal((8,))),
        ("pre_fc_norm_hidden.weight", mx.random.normal((8,))),
    ], strict=True)
    m.set_dtype(dtype)
    _equal(m.fuse(e, h), _legacy_qwen_pre_fc_fuse(m, e, h))
    ling = Ling()
    ling.load_weights([
        ("eh_proj.weight", mx.random.normal((8, 16))),
        ("enorm.weight", mx.random.normal((8,))),
        ("hnorm.weight", mx.random.normal((8,))),
    ], strict=True)
    ling.set_dtype(dtype)
    _equal(db.fuse_mtp_module(db.NEXTN_FUSION, ling, e, h), _legacy_bailing_hybrid_fuse(ling, e, h))


def _legacy_eagle4_rows(attn, q, k, v, rows, *, commit, chain, sliding_window):
    """The per-row loop of runtime/drafters/eagle4.py Eagle4DraftModel._forward."""
    outputs = []
    for row, caches in enumerate(rows):
        cache = caches[0]
        prior_k, prior_v = cache.keys_and_values()
        local = None if commit else chain[row]
        # The trained window includes the current context position. Keep
        # W committed rows for subsequent tree steps, but a new committed
        # context query reads only W-1 prior rows before adding itself.
        if commit and prior_k.shape[2] >= sliding_window:
            keep = sliding_window - 1
            prior_k = prior_k[:, :, -keep:] if keep else prior_k[:, :, :0]
            prior_v = prior_v[:, :, -keep:] if keep else prior_v[:, :, :0]
        if local is not None and local[0] is not None:
            prior_k = (
                mx.concatenate([prior_k, local[0]], axis=2)
                if prior_k.shape[2]
                else local[0]
            )
            prior_v = (
                mx.concatenate([prior_v, local[1]], axis=2)
                if prior_v.shape[2]
                else local[1]
            )
        position = cache.offset + (
            0 if local is None or local[0] is None else local[0].shape[2]
        )
        out, rotated = attn.attend(
            q[row : row + 1],
            k[row : row + 1],
            v[row : row + 1],
            position=position,
            prior_k=prior_k,
            prior_v=prior_v,
        )
        outputs.append(out)
        if commit:
            cache.append(rotated, v[row : row + 1])
        elif chain[row][0] is None:
            chain[row] = (rotated, v[row : row + 1])
        else:
            old = chain[row]
            chain[row] = (
                mx.concatenate([old[0], rotated], axis=2),
                mx.concatenate([old[1], v[row : row + 1]], axis=2),
            )
    return mx.concatenate(outputs, axis=0)


@pytest.mark.parametrize("window", [1, 3])
@pytest.mark.parametrize("dtype", DTYPES)
def test_eagle4_commit_window_rows_match_legacy(dtype, window):
    """The commit-window trim (W-1 prior rows read, W kept) through
    eagle4's view over the shared row loop, mixed commit/chain steps."""
    from mlx2.runtime.drafters.eagle3 import Eagle3Attention
    from mlx2.runtime.drafters.eagle4 import _CommitWindowView

    mx.random.seed(12)
    old, new = _legacy_eagle3_Attention(_CHAIN_CFG), Eagle3Attention(_CHAIN_CFG)
    _copy_weights(new, old)
    old.set_dtype(dtype)
    new.set_dtype(dtype)
    B = 2
    # The cache holds W rows (eagle4 make_cache sizes it to the window).
    rows_old = [[EagleWindowKVCache(window, 2, 4)] for _ in range(B)]
    rows_new = [[EagleWindowKVCache(window, 2, 4)] for _ in range(B)]
    chain_old = [(None, None) for _ in range(B)]
    chain_new = [(None, None) for _ in range(B)]
    trimmed = 0
    for length, commit in ((1, True), (1, True), (2, False), (1, True), (1, False),
                           (1, False), (1, True), (1, True), (3, False)):
        x = mx.random.normal((B, length, 32)).astype(dtype)
        q, k, v = old.project(x)
        if commit:
            trimmed += int(rows_old[0][0].keys_and_values()[0].shape[2] >= window)
        ref = _legacy_eagle4_rows(old, q, k, v, rows_old, commit=commit, chain=chain_old,
                                  sliding_window=window)
        entries = [r[0] for r in rows_new]
        if commit:
            entries = [_CommitWindowView(e, window) for e in entries]
        got, new_locals = db.chain_attend_rows(new, q, k, v, entries, chain_new, commit=commit)
        _equal(got, ref)
        if commit:
            chain_old = [(None, None) for _ in range(B)]
            chain_new = [(None, None) for _ in range(B)]
        else:
            chain_new = new_locals
            for a, b in zip(chain_new, chain_old):
                _equal(a[0], b[0])
                _equal(a[1], b[1])
        for ro, rn in zip(rows_old, rows_new):
            _equal(rn[0].keys_and_values()[0], ro[0].keys_and_values()[0])
            assert rn[0].offset == ro[0].offset
    assert trimmed >= 2  # the trim branch was taken
