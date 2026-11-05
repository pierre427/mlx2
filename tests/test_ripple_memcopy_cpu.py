"""Ripple hot-path memory-copy fixes are bit-exact against the pre-fix math.

``reference_forward`` is the per-layer hot path exactly as mlx2 cf8247180
computed it (per-layer masks, per-projection RoPE tables, per-layer seed
casts, doubled cos/sin, negated-half concatenation, two FP32 casts of the
gate, per-projection FP32 input casts, transposed reshape before the gate).
Every test compares the current model against it bit for bit, in FP32 and
BF16, on dense and native-NVFP4 modules, through chunked prefill across the
sliding window and decode on both plain and serving-shaped batch caches.
Synthetic miniature tensors; no checkpoint or serving claim.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.generate import _merge_caches
from mlx2.runtime.models import ricursive
from mlx2.runtime.models.base import create_attention_mask, scaled_dot_product_attention
from mlx2.runtime.models.cache import compact_prompt_cache_windows
from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear, shared_fp32_input
from mlx2.runtime.models.ricursive import (
    Model,
    ModelArgs,
    apply_rotary,
    inverse_frequencies,
    rotary,
    rotary_tables,
)

LAMBDAS = [-2.75, -.3, .25, 1.1, 4.0]
TOKENS = mx.array([[1, 2, 4, 8, 3, 10, 9, 5, 6, 7, 11, 12, 13]])


def tiny_args():
    return ModelArgs(hidden_size=16, intermediate_size=32, num_hidden_layers=5,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                     vocab_size=32, sliding_window=4, embed_scale=71.55417527999327,
                     ffn_clamp_threshold=0.2)


def build(dtype, nvfp4, seed=5):
    mx.random.seed(seed)
    model = Model(tiny_args(), nvfp4=nvfp4)
    for layer, value in zip(model.layers, LAMBDAS, strict=True):
        layer.self_attn.value_res_lambda = mx.array([value])
    model.set_dtype(dtype)
    if nvfp4:
        rng = np.random.default_rng(seed)
        for _, module in model.named_modules():
            if isinstance(module, ModelOptNVFP4Linear):
                module.weight = mx.array(rng.integers(0, 2 ** 32, size=module.weight.shape,
                                                      dtype=np.uint64).astype(np.uint32))
                # Finite, positive E4M3 codes.
                module.scales = mx.array(rng.integers(0x30, 0x48, size=module.scales.shape, dtype=np.uint8))
                module.global_scale = mx.array(np.float32(rng.uniform(0.5, 2.0)))
    mx.eval(model.parameters())
    return model


def as_np(x):
    return np.asarray(x.astype(mx.float32))


def assert_bitwise(actual, expected):
    np.testing.assert_array_equal(as_np(actual), as_np(expected))


# --- The pre-fix hot path (mlx2 cf8247180), frozen -------------------------

def reference_rotary(x, offset, theta):
    dims = x.shape[-1]
    inv = inverse_frequencies(dims, theta)
    if isinstance(offset, mx.array) and offset.ndim:
        positions = mx.arange(x.shape[-2], dtype=mx.float32)[None, :] + offset[:, None]
        phase = positions[:, None, :, None] * inv
    else:
        positions = mx.arange(x.shape[-2], dtype=mx.float32) + offset
        phase = positions[:, None] * inv
    phase = mx.concatenate((phase, phase), axis=-1)
    cos = mx.cos(phase).astype(x.dtype).astype(mx.float32)
    sin = mx.sin(phase).astype(x.dtype).astype(mx.float32)
    y = x.astype(mx.float32)
    half = mx.concatenate((-y[..., dims // 2:], y[..., :dims // 2]), axis=-1)
    return (y * cos + half * sin).astype(x.dtype)


def reference_attention(attn, x, seed, cache):
    args = attn.args
    b, length, _ = x.shape
    q = attn.q_norm(attn.q_proj(x).reshape(b, length, args.num_attention_heads, args.head_dim))
    k = attn.k_norm(attn.k_proj(x).reshape(b, length, args.num_key_value_heads, args.head_dim))
    value_input = x
    if args.use_valres:
        blend = mx.sigmoid(attn.value_res_lambda[0]).astype(x.dtype)
        value_input = (seed.astype(mx.float32) + blend.astype(mx.float32)
                       * (x.astype(mx.float32) - seed.astype(mx.float32))).astype(x.dtype)
    v = attn.v_proj(value_input).reshape(b, length, args.num_key_value_heads, args.head_dim)
    q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    mask = create_attention_mask(x, cache, window_size=args.sliding_window if attn.local else None)
    if attn.local:
        offset = cache.offset if cache is not None else 0
        q, k = reference_rotary(q, offset, args.rope_theta), reference_rotary(k, offset, args.rope_theta)
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    out = scaled_dot_product_attention(q, k, v, cache=cache, scale=args.head_dim ** -0.5, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
    if args.use_gated_attention:
        gate = attn.g_proj(x)
        if args.gate_full_precision:
            gate = gate.astype(mx.float32)
        out = out * mx.sigmoid(gate).astype(out.dtype)
    return attn.o_proj(out)


def reference_mlp(mlp, x):
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    if mlp.clamp:
        gate = mx.minimum(gate, mlp.clamp)
        up = mx.clip(up, -mlp.clamp, mlp.clamp)
    silu = (gate.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype)
    return mlp.down_proj(silu * up)


def reference_forward(model, tokens, cache=None):
    backbone = model.model
    h = backbone.embed_tokens(tokens)
    if backbone.args.embed_scale is not None:
        h = ricursive.scale_embeddings(h, backbone.args.embed_scale)
    seed = h
    if cache is None:
        cache = [None] * len(backbone.layers)
    for layer, entry in zip(backbone.layers, cache, strict=True):
        h = h + layer.post_attention_layernorm(
            reference_attention(layer.self_attn, layer.input_layernorm(h), seed, entry))
        h = h + layer.post_feedforward_layernorm(reference_mlp(layer.mlp, layer.pre_feedforward_layernorm(h)))
    return model.lm_head(backbone.norm(h))


# --- Whole hot path ---------------------------------------------------------

@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("chunks", [[13], [3, 5, 1, 4], [1] * 13, [7, 6]])
def test_forward_matches_prefix_hot_path_bitwise(dtype, nvfp4, chunks):
    model = build(dtype, nvfp4)
    assert_bitwise(model(TOKENS), reference_forward(model, TOKENS))
    for batched in (False, True):
        # Plain per-layer caches, and the serving lane's B=1 merged batch
        # caches with the chunk-boundary window compaction prefill performs.
        expected_cache = model.make_cache()
        actual_cache = model.make_cache()
        if batched:
            expected_cache = _merge_caches([expected_cache])
            actual_cache = _merge_caches([actual_cache])
        offset = 0
        for length in chunks:
            piece = TOKENS[:, offset:offset + length]
            assert_bitwise(model(piece, actual_cache), reference_forward(model, piece, expected_cache))
            offset += length
            if batched:
                compact_prompt_cache_windows(actual_cache)
                compact_prompt_cache_windows(expected_cache)
                mx.eval([c.state for c in actual_cache], [c.state for c in expected_cache])
        for token in (15, 16, 17, 18, 19):  # decode past the window wrap
            step = mx.array([[token]])
            assert_bitwise(model(step, actual_cache), reference_forward(model, step, expected_cache))
        for actual, expected in zip(actual_cache, expected_cache, strict=True):
            for a, e in zip(tree_flatten(actual.state), tree_flatten(expected.state), strict=True):
                assert_bitwise(a[1], e[1])


# --- RoPE: one table per forward, split-half rotation ----------------------

@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape,offset", [
    ((1, 80, 1, 128), 1023),               # decode token at the window edge
    ((1, 8, 7, 128), 131071),              # deep position, row count with a SIMD tail
    ((2, 3, 5, 128), mx.array([2, 7])),    # per-lane offsets (batch caches)
    ((1, 4, 7, 4), 3),                     # fixture geometry
])
def test_split_half_rotary_matches_reference_bitwise(dtype, shape, offset):
    mx.random.seed(int(np.prod(shape)))
    x = (mx.random.normal(shape) * 3).astype(dtype)
    assert_bitwise(rotary(x, offset, 10000.0), reference_rotary(x, offset, 10000.0))
    cos, sin = rotary_tables(shape[-2], offset, shape[-1], 10000.0, dtype)
    assert cos.shape[-1] == sin.shape[-1] == shape[-1] // 2
    assert_bitwise(apply_rotary(x, cos, sin), reference_rotary(x, offset, 10000.0))


def test_inverse_frequencies_are_built_once():
    assert inverse_frequencies(128, 10000.0) is inverse_frequencies(128, 10000.0)
    assert inverse_frequencies(128, 10000.0) is not inverse_frequencies(64, 10000.0)


def test_rope_tables_and_masks_are_built_once_per_forward(monkeypatch):
    model = build(mx.bfloat16, False)
    tables, masks = [], []
    real_tables, real_mask = ricursive.rotary_tables, ricursive.create_attention_mask

    def counting_tables(*args, **kwargs):
        tables.append(args)
        return real_tables(*args, **kwargs)

    def counting_mask(*args, **kwargs):
        masks.append(kwargs)
        return real_mask(*args, **kwargs)

    monkeypatch.setattr(ricursive, "rotary_tables", counting_tables)
    monkeypatch.setattr(ricursive, "create_attention_mask", counting_mask)
    cache = _merge_caches([model.make_cache()])
    mx.eval(model(TOKENS, cache))
    compact_prompt_cache_windows(cache)
    mx.eval(model(mx.array([[15]]), cache))
    # Two forwards: one table each (64 sliding layers x q and k would be 128
    # per forward at the real geometry) and one mask per layer kind.
    assert len(tables) == 2
    assert [m["window_size"] for m in masks] == [4, None, 4, None]


# --- Value blend, gate product, NVFP4 shared input --------------------------

def test_value_blend_follows_in_place_parameter_updates():
    model = build(mx.bfloat16, False)
    attn = model.layers[0].self_attn
    tokens = mx.array([[1, 2, 3]])
    before = model(tokens)
    attn.value_res_lambda[0] = 4.0  # in place: same array object
    assert_bitwise(attn.value_blend(mx.bfloat16),
                   mx.sigmoid(mx.array(4.0)).astype(mx.bfloat16).astype(mx.float32))
    after = model(tokens)
    assert not mx.array_equal(before, after)
    assert not any("_value_blend" in name for name, _ in tree_flatten(model.parameters()))

def test_gate_product_reshapes_as_a_view():
    b, length, heads, dim = 1, 7, 4, 4
    out = mx.random.normal((b, heads, length, dim)).astype(mx.bfloat16)
    gate = mx.random.normal((b, length, heads * dim)).astype(mx.bfloat16)
    expected = out.transpose(0, 2, 1, 3).reshape(b, length, -1) * gate
    product = out.transpose(0, 2, 1, 3) * gate.reshape(b, length, heads, dim)
    flat = product.reshape(b, length, -1)
    mx.eval(product, flat)
    assert_bitwise(flat, expected)
    # Buffer identity (numpy cannot view BF16; the layout law is dtype-free).
    product32 = out.astype(mx.float32).transpose(0, 2, 1, 3) * gate.astype(mx.float32).reshape(b, length, heads, dim)
    flat32 = product32.reshape(b, length, -1)
    mx.eval(product32, flat32)
    assert np.asarray(product32).ctypes.data == np.asarray(flat32).ctypes.data
    # The pre-fix order (transposed reshape, then multiply) materializes a copy.
    out32 = out.astype(mx.float32)
    copied = out32.transpose(0, 2, 1, 3).reshape(b, length, -1)
    mx.eval(out32, copied)
    assert np.asarray(copied).ctypes.data != np.asarray(out32).ctypes.data


def test_shared_fp32_input_only_for_sibling_native_projections():
    model = build(mx.bfloat16, True)
    mlp = model.layers[0].mlp
    x = mx.random.normal((1, 3, 16)).astype(mx.bfloat16)
    shared = shared_fp32_input((mlp.gate_proj, mlp.up_proj), x)
    assert shared.dtype == mx.float32
    assert shared_fp32_input((mlp.gate_proj, mlp.up_proj), shared) is None
    assert_bitwise(mlp.gate_proj(shared).astype(x.dtype), mlp.gate_proj(x))
    mlp.up_proj.set_dispatch_policy("fused_b1")
    assert shared_fp32_input((mlp.gate_proj, mlp.up_proj), x) is None
    dense = build(mx.bfloat16, False).layers[0].mlp
    assert shared_fp32_input((dense.gate_proj, dense.up_proj), x) is None
    assert mlp.gate_proj.dispatch_status()["counts"] == {"native": 2}
