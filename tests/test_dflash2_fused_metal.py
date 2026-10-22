# SPDX-License-Identifier: MIT
"""Real Metal parity; synthetic tensors only, no checkpoint qualification.

Coordinator only, under the GPU lock. Stage atol=.0625/rtol=.03; whole-layer
atol=.125/rtol=.05 vs the unfused bf16 chain, intentionally not bit-exact.
The tighter kernel-vs-fp32-law comparison separately catches indexing errors.
"""
import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.drafters import dflash2_fused as fused
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel, _grouped_dynamic_convolve, configure_draft_kernel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def rnd(shape, seed, scale=0.3):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.bfloat16)


def close(got, want, atol=fused.STAGE_ATOL, rtol=fused.STAGE_RTOL):
    assert got.shape == want.shape and got.dtype == want.dtype
    mx.eval(got, want)
    np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), np.asarray(want.astype(mx.float32)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("batch,rows", [(1, 1), (1, 16), (1, 64), (4, 4), (4, 16)])
@pytest.mark.parametrize("kernel,group", [(2, 16), (1, 32), (5, 8)])
@pytest.mark.parametrize("finish", [False, True])
def test_conv(batch, rows, kernel, group, finish):
    h = 5120
    x, dynamic, base = rnd((batch, rows, h), 1), rnd((batch, rows, kernel, h // group), 2), rnd((kernel, h), 3)
    residual = rnd(x.shape, 4) if finish else None
    got = fused.conv(x, dynamic, base, group, residual)
    close(got, fused.conv_reference(x, dynamic, base, group, residual), atol=0.015625, rtol=0.01)
    # The existing implementation's >length tap case has no valid slice;
    # zero-fill semantics there are covered by the fused reference instead.
    if kernel <= rows + 1:
        want = _grouped_dynamic_convolve(x, dynamic, base, group)
        close(got, want if residual is None else residual + want)


@pytest.mark.parametrize("batch,rows", [(1, 1), (1, 16), (1, 64), (4, 4)])
@pytest.mark.parametrize("span", [0, 8, 100])
@pytest.mark.parametrize("traditional", [False, True])
def test_qk_norm_rope(batch, rows, span, traditional):
    q, ck, pk = rnd((batch, rows, 32, 128), 10), rnd((batch, span, 8, 128), 11), rnd((batch, rows, 8, 128), 12)
    qw, kw = rnd((128,), 13, 1), rnd((128,), 14, 1)
    offsets = [1024 + 37 * i for i in range(batch)]
    rope = nn.RoPE(128, traditional=traditional, base=10000000)
    got = fused.qk_rope(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6)
    reference = fused.qk_rope_reference(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6)
    for a, b in zip(got, reference):
        close(a, b, atol=0.03125, rtol=0.01)
    for actual, x, w, delta in zip(got, (q, ck, pk), (qw, kw, kw), (span, 0, span)):
        y = mx.fast.rms_norm(x, w, 1e-6).transpose(0, 2, 1, 3)
        want = y if span == 0 and x is ck else mx.concatenate([
            rope(y[i:i+1], offset=o + delta) for i, o in enumerate(offsets)
        ], axis=0)
        close(actual, want)


@pytest.mark.parametrize("batch,rows", [(1, 16), (4, 4)])
def test_whole_q4_lane_layer(batch, rows):
    """Full reference geometry; one layer, same weights/matmuls in both arms."""
    config = DFlash2Config.from_dict({
        "model_type": "dflash2", "hidden_size": 5120, "intermediate_size": 17408,
        "num_hidden_layers": 1, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 64, "num_target_layers": 1,
        "layer_types": ["sliding_attention"], "sliding_window": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 10000000,
        "dflash_config": {"target_layer_ids": [0], "conv_kernel_size": 2,
                          "conv_group_size": 16, "selector_rank": 8,
                          "selector_top_k": 4, "mask_token_id": 63, "block_size": 8},
    })
    model = DFlash2DraftModel(config)
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    model.set_dtype(mx.bfloat16)
    for i, conv in enumerate((model.layers[0].attention_conv, model.layers[0].mlp_conv)):
        conv.base_kernel = rnd(conv.base_kernel.shape, 30 + i)
    mx.eval(model.parameters())
    configure_draft_kernel(model, "lane")
    x, ctx = rnd((batch, rows, 5120), 40), rnd((batch, 100, 5120), 41)

    def cache():
        entries = []
        for row in range(batch):
            entry = model.make_cache()[0]
            k, v = rnd((1, 8, 64 + row, 128), 50 + row), rnd((1, 8, 64 + row, 128), 60 + row)
            entry.update_and_fetch(k, v)
            entries.append(entry)
        return SimpleNamespace(rows=entries) if batch > 1 else entries[0]

    plain, fast = cache(), cache()
    want = model.layers[0](x, ctx, model.rope, plain)
    mx.eval(want)
    # Same installation as production; configuration/receipts covered on CPU.
    fused.install(model)
    got = model.layers[0](x, ctx, model.rope, fast)
    close(got, want, atol=fused.LAYER_ATOL, rtol=fused.LAYER_RTOL)
    for a, b in zip(fast.rows if batch > 1 else [fast], plain.rows if batch > 1 else [plain]):
        assert a.offset == b.offset and a._idx == b._idx
        for aa, bb in zip(a.state, b.state):
            close(aa, bb)


def test_noncontiguous_projection_and_dynamic_stage_views():
    """Real lane slices need stride-aware reads, without producer copy launches."""
    b, length, span, h = 4, 16, 100, 5120
    x = rnd((b, length, h * 2), 101)[..., ::2]
    dynamic = rnd((b, length, 2, 2, h // 16), 102)[..., 1, :, :]
    base = rnd((2, h * 2), 103)[:, ::2]
    residual = rnd((b, length, h * 2), 104)[..., 1::2]
    close(fused.conv(x, dynamic, base, 16, residual),
          fused.conv_reference(x, dynamic, base, 16, residual), atol=.015625, rtol=.01)
    q = rnd((b, length, 48, 128), 105)[:, :, :32]
    keys = rnd((b, span + length, 16, 128), 106)
    ck, pk = keys[:, :span, :8], keys[:, span:, :8]
    qw, kw = rnd((256,), 107, 1)[::2], rnd((256,), 108, 1)[1::2]
    rope, offsets = nn.RoPE(128, base=10000000), [1024, 777, 300, 0]
    for a, z in zip(fused.qk_rope(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6),
                    fused.qk_rope_reference(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6)):
        close(a, z, atol=.03125, rtol=.01)


@pytest.mark.parametrize("qh,kh,dim", [(2, 1, 8), (8, 2, 64), (6, 3, 48), (4, 2, 256)])
def test_other_supported_head_geometries(qh, kh, dim):
    q, ck, pk = rnd((2, 7, qh, dim), 120), rnd((2, 3, kh, dim), 121), rnd((2, 7, kh, dim), 122)
    qw, kw = rnd((dim,), 123, 1), rnd((dim,), 124, 1)
    rope, offsets = nn.RoPE(dim), [0, 83]
    for a, z in zip(fused.qk_rope(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6),
                    fused.qk_rope_reference(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6)):
        close(a, z, atol=.03125, rtol=.01)


@pytest.mark.parametrize("batch,rows", [(1, 1), (1, 16), (1, 64), (4, 4), (4, 16)])
@pytest.mark.parametrize("hidden_dtype,residual_dtype,weight_dtype", [
    (mx.bfloat16, mx.bfloat16, mx.bfloat16),
    (mx.bfloat16, mx.float32, mx.bfloat16),
    (mx.float32, mx.bfloat16, mx.bfloat16),
    (mx.bfloat16, mx.bfloat16, mx.float32),
])
def test_conv_finish_residual_norm(batch, rows, hidden_dtype, residual_dtype, weight_dtype):
    h = 5120
    x = rnd((batch, rows, h), 140).astype(hidden_dtype)
    dynamic = rnd((batch, rows, 2, 2, h // 16), 141)[..., 1, :, :].astype(hidden_dtype)
    base = rnd((2, h), 142).astype(hidden_dtype)
    residual = rnd(x.shape, 143).astype(residual_dtype)
    weight = rnd((h * 2,), 144, 1)[::2].astype(weight_dtype)
    got = fused.conv_finish_norm(x, dynamic, base, 16, residual, weight, 1e-6)
    reference = fused.conv_finish_norm_reference(x, dynamic, base, 16, residual, weight, 1e-6)
    close(got[0], reference[0], atol=.015625, rtol=.01)
    close(got[1], reference[1], atol=.03125, rtol=.01)
    unfused = residual + _grouped_dynamic_convolve(x, dynamic, base, 16)
    close(got[0], unfused)
    close(got[1], mx.fast.rms_norm(unfused, weight, 1e-6))


@pytest.mark.parametrize("hidden_dtype,dynamic_dtype,base_dtype,residual_dtype", [
    (mx.bfloat16, mx.float32, mx.bfloat16, None),
    (mx.bfloat16, mx.bfloat16, mx.float32, None),
    (mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32),
    (mx.float32, mx.bfloat16, mx.bfloat16, mx.bfloat16),
    (mx.float16, mx.bfloat16, mx.float32, mx.float16),
])
def test_conv_mixed_dtypes(hidden_dtype, dynamic_dtype, base_dtype, residual_dtype):
    x = rnd((1, 16, 5120), 150).astype(hidden_dtype)
    dynamic = rnd((1, 16, 2, 320), 151).astype(dynamic_dtype)
    base = rnd((2, 5120), 152).astype(base_dtype)
    residual = None if residual_dtype is None else rnd(x.shape, 153).astype(residual_dtype)
    got = fused.conv(x, dynamic, base, 16, residual)
    reference = fused.conv_reference(x, dynamic, base, 16, residual)
    close(got, reference, atol=.015625, rtol=.01)
    unfused = _grouped_dynamic_convolve(x, dynamic, base, 16)
    close(got, unfused if residual is None else residual + unfused)
