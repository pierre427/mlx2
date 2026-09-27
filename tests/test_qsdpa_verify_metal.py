"""Quantized verify-block attention kernel: plan, mask contract and routing on
CPU; equivalence on Metal.

The CPU reference (``reference_verify_attention``) runs the kernel's split
ranges, per-row causal limits, left padding, fused-row groups and partial
merge in MLX ops, and is checked against dequantize + SDPA with an explicit
mask. The Metal cases run only with MLX2_RUN_GPU_TESTS=1 (while holding the
GPU lease).
"""

import os

import mlx.core as mx
import pytest

import mlx2.runtime.models.qsdpa_verify_metal as qvm
from mlx2.runtime.models.base import create_causal_mask

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"


def _inputs(B, Hq, Hkv, L, N, D=64, kb=8, vb=8, dtype=mx.float32, seed=0, gs=64):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, L, D)).astype(dtype)
    k = mx.random.normal((B, Hkv, N, D)).astype(dtype)
    v = mx.random.normal((B, Hkv, N, D)).astype(dtype)
    return q, mx.quantize(k, group_size=gs, bits=kb), mx.quantize(v, group_size=gs, bits=vb)


def _dequant_sdpa(q, qk, qv, scale, left_padding=None, gs=64, kb=8, vb=8, dtype=None):
    """The reference: dequantize, then SDPA with the explicit causal mask."""
    dtype = dtype or q.dtype
    B, Hq, L, _ = q.shape
    N = qk[0].shape[2]
    k = mx.dequantize(*qk, group_size=gs, bits=kb).astype(dtype)
    v = mx.dequantize(*qv, group_size=gs, bits=vb).astype(dtype)
    mask = create_causal_mask(L, offset=N - L, left_padding=left_padding)
    if mask.ndim == 2:
        mask = mask[None, None]
    return mx.fast.scaled_dot_product_attention(q.astype(dtype), k, v, scale=scale, mask=mask)


@pytest.fixture
def pretend_gpu(monkeypatch):
    monkeypatch.setattr(qvm.mx, "default_device", lambda: mx.gpu)


# --- plan -----------------------------------------------------------------


def test_fused_groups():
    assert qvm.fused_groups(1, 4) == (1, 8)
    assert qvm.fused_groups(3, 6) == (1, 24)    # 27B, self-MTP K=2
    assert qvm.fused_groups(4, 8) == (1, 32)
    assert qvm.fused_groups(8, 6) == (2, 24)
    assert qvm.fused_groups(3, 12) == (2, 24)   # Flash-Next geometry
    assert qvm.fused_groups(8, 12) == (3, 32)
    for rows in range(1, 9):
        for gqa in (1, 2, 4, 6, 8, 12, 16):
            groups, per = qvm.fused_groups(rows, gqa)
            assert per % 8 == 0 and per <= qvm.MAX_FUSED
            assert groups * per >= rows * gqa > (groups - 1) * per


@pytest.mark.parametrize("n,lp,blocks", [(1000, 0, 7), (1000, 333, 7), (17, 5, 64),
                                         (131072, 0, 128), (4096, 4000, 32), (10, 10, 4)])
def test_split_ranges_cover_visible_keys_once(n, lp, blocks):
    seen = []
    for start, end in qvm.split_ranges(n, lp, blocks):
        assert start >= lp
        seen.extend(range(start, end))
    assert seen == list(range(lp, n))


def test_row_limit_is_causal():
    assert [qvm.row_limit(100, 3, r) for r in range(3)] == [98, 99, 100]
    assert qvm.row_limit(1, 1, 0) == 1


def test_verify_blocks_bounds():
    assert qvm.verify_blocks(131072, 1, 4, 1) == 128
    assert qvm.verify_blocks(131072, 4, 4, 1) == 32
    assert qvm.verify_blocks(300, 1, 4, 1) == 1
    assert qvm.verify_blocks(4096, 1, 2, 2) == 16


# --- CPU reference of the kernel's indexing ----------------------------------


@pytest.mark.parametrize("L", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("Hq,Hkv", [(4, 4), (12, 2), (24, 4), (24, 2)])
@pytest.mark.parametrize("blocks", [1, 5, 16])
def test_reference_matches_masked_sdpa(L, Hq, Hkv, blocks):
    mx.set_default_device(mx.cpu)
    q, qk, qv = _inputs(1, Hq, Hkv, L, 150, seed=L)
    got = qvm.reference_verify_attention(q, qk, qv, scale=0.125, blocks=blocks)
    want = _dequant_sdpa(q, qk, qv, 0.125)
    assert mx.abs(got - want).max().item() < 1e-4


@pytest.mark.parametrize("bits", [(8, 8), (8, 4), (4, 4)])
def test_reference_batched_left_padding(bits):
    mx.set_default_device(mx.cpu)
    lp = mx.array([0, 37, 90, 5])
    q, qk, qv = _inputs(4, 12, 2, 3, 128, kb=bits[0], vb=bits[1], seed=11)
    got = qvm.reference_verify_attention(q, qk, qv, scale=0.125, left_padding=lp, blocks=9,
                                         key_bits=bits[0], value_bits=bits[1])
    want = _dequant_sdpa(q, qk, qv, 0.125, left_padding=lp, kb=bits[0], vb=bits[1])
    assert mx.abs(got - want).max().item() < 1e-4


# --- mask contract ---------------------------------------------------------


def test_causal_left_padding_contract():
    assert qvm.causal_left_padding(None, 1, 1, 10) is None
    assert qvm.causal_left_padding(None, 1, 3, 10) is qvm.NOT_CAUSAL
    assert qvm.causal_left_padding("causal", 2, 3, 10) is None
    assert qvm.causal_left_padding("other", 2, 3, 10) is qvm.NOT_CAUSAL
    lp = mx.array([0, 2])
    mask = create_causal_mask(3, offset=7, left_padding=lp)
    # an unregistered array could mean anything
    assert qvm.causal_left_padding(mask, 2, 3, 10) is qvm.NOT_CAUSAL
    qvm.register_causal_mask(mask, lp)
    assert qvm.causal_left_padding(mask, 2, 3, 10) is lp
    assert qvm.causal_left_padding(mask, 2, 2, 10) is qvm.NOT_CAUSAL  # wrong rows
    assert qvm.causal_left_padding(mask, 3, 3, 10) is qvm.NOT_CAUSAL  # wrong batch
    seg = create_causal_mask(3, offset=7)
    qvm.register_causal_mask(seg, None, kind="segmented")
    assert qvm.registered_mask_kind(seg) == "segmented"
    assert qvm.causal_left_padding(seg, 1, 3, 10) is qvm.NOT_CAUSAL  # segmented only


def test_registry_is_bounded():
    for _ in range(3 * qvm._REGISTRY_LIMIT):
        qvm.register_causal_mask(mx.zeros((1,)), None)
    assert len(qvm._CAUSAL_MASKS) <= qvm._REGISTRY_LIMIT


def test_batch_quantized_cache_registers_its_mask():
    from mlx2.runtime.models.cache import BatchQuantizedKVCache

    cache = BatchQuantizedKVCache([2, 0], group_size=64, bits=8)
    cache.update_and_fetch(mx.zeros((2, 1, 5, 64)), mx.zeros((2, 1, 5, 64)))
    mask = cache.make_mask(3)
    assert qvm.causal_left_padding(mask, 2, 3, 8) is cache.left_padding
    windowed = cache.make_mask(3, window_size=2)
    assert qvm.causal_left_padding(windowed, 2, 3, 8) is qvm.NOT_CAUSAL


# --- gating -----------------------------------------------------------------


def test_supported_shapes(pretend_gpu):
    kw = dict(group_size=64, key_bits=8, value_bits=8)
    q, qk, qv = _inputs(1, 24, 4, 3, 64, D=256, dtype=mx.bfloat16)
    assert qvm.verify_attention_supported(q, qk, qv, **kw)
    q9 = mx.zeros((1, 24, 9, 256), dtype=mx.bfloat16)
    assert not qvm.verify_attention_supported(q9, qk, qv, **kw)          # > 8 rows
    q32, qk32, qv32 = _inputs(1, 24, 4, 3, 64, D=256)
    assert not qvm.verify_attention_supported(q32, qk32, qv32, **kw)     # float32
    q64, qk64, qv64 = _inputs(1, 8, 2, 3, 64, D=64, dtype=mx.bfloat16)
    assert not qvm.verify_attention_supported(q64, qk64, qv64, **kw)     # head dim 64
    assert not qvm.verify_attention_supported(q, qk, qv, group_size=64, key_bits=3, value_bits=8)
    q12, qk12, qv12 = _inputs(1, 24, 2, 8, 64, D=128, kb=4, vb=4, dtype=mx.float16)
    assert qvm.verify_attention_supported(q12, qk12, qv12, group_size=64, key_bits=4, value_bits=4)


def test_supported_is_false_on_cpu():
    q, qk, qv = _inputs(1, 24, 4, 3, 64, D=256, dtype=mx.bfloat16)
    assert not qvm.verify_attention_supported(q, qk, qv, group_size=64, key_bits=8, value_bits=8)


def test_threshold_and_kill_switch(pretend_gpu, monkeypatch):
    kw = dict(group_size=64, key_bits=8, value_bits=8)
    short = _inputs(1, 24, 4, 3, qvm.MIN_CONTEXT - 1, D=256, dtype=mx.bfloat16)
    long = _inputs(1, 24, 4, 3, qvm.MIN_CONTEXT, D=256, dtype=mx.bfloat16)
    assert not qvm.use_verify_kernel(*short, **kw)
    assert qvm.use_verify_kernel(*long, **kw)
    monkeypatch.setenv("MLX2_QSDPA_VERIFY_KERNEL", "0")
    assert not qvm.use_verify_kernel(*long, **kw)


# --- routing (kernel replaced by the CPU reference) ------------------------


@pytest.fixture
def spy_kernel(monkeypatch):
    calls = []

    def fake(queries, q_keys, q_values, *, scale, group_size, key_bits, value_bits,
             left_padding=None, blocks=None):
        calls.append({"rows": queries.shape[2], "left_padding": left_padding})
        return qvm.reference_verify_attention(
            queries, q_keys, q_values, scale=scale, group_size=group_size,
            key_bits=key_bits, value_bits=value_bits, left_padding=left_padding, blocks=3)

    monkeypatch.setattr(qvm, "gqa_quantized_verify_attention", fake)
    monkeypatch.setattr(qvm, "use_verify_kernel", lambda *a, **k: qvm.verify_kernel_enabled())
    return calls


def test_routing_b1_causal(spy_kernel):
    from mlx2.runtime.models.base import quantized_scaled_dot_product_attention

    q, qk, qv = _inputs(1, 12, 2, 3, 100)
    got = quantized_scaled_dot_product_attention(q, qk, qv, scale=0.125, mask="causal",
                                                 group_size=64, bits=8)
    assert spy_kernel == [{"rows": 3, "left_padding": None}]
    assert mx.abs(got - _dequant_sdpa(q, qk, qv, 0.125)).max().item() < 1e-4
    # an unregistered array takes the composed path
    explicit = create_causal_mask(3, offset=97)[None, None]
    quantized_scaled_dot_product_attention(q, qk, qv, scale=0.125, mask=explicit,
                                           group_size=64, bits=8)
    assert len(spy_kernel) == 1


def test_routing_batch_quantized_cache(spy_kernel):
    from mlx2.runtime.models.base import scaled_dot_product_attention
    from mlx2.runtime.models.cache import BatchQuantizedKVCache

    mx.random.seed(4)
    cache = BatchQuantizedKVCache([0, 20, 7], group_size=64, bits=8)
    cache.update_and_fetch(mx.random.normal((3, 2, 60, 64)), mx.random.normal((3, 2, 60, 64)))
    mask = cache.make_mask(3)
    k, v = cache.update_and_fetch(mx.random.normal((3, 2, 3, 64)), mx.random.normal((3, 2, 3, 64)))
    q = mx.random.normal((3, 12, 3, 64))
    got = scaled_dot_product_attention(q, k, v, cache, scale=0.125, mask=mask)
    assert len(spy_kernel) == 1 and spy_kernel[0]["left_padding"] is cache.left_padding
    want = _dequant_sdpa(q, k, v, 0.125, left_padding=cache.left_padding)
    assert mx.abs(got - want).max().item() < 1e-4


def test_routing_segmented_rows_pass_causal(spy_kernel, monkeypatch):
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.segmented_batch_cache import build_segmented_batch_cache_group

    def filled(n, seed):
        mx.random.seed(seed)
        c = KVCache()
        c.update_and_fetch(mx.random.normal((1, 2, n, 64)), mx.random.normal((1, 2, n, 64)))
        return c.to_quantized(group_size=64, bits=8)

    def run():
        (view,) = build_segmented_batch_cache_group([[filled(40, 1)], [filled(25, 2)]])
        mx.random.seed(9)
        keys = mx.random.normal((2, 2, 3, 64))
        values = mx.random.normal((2, 2, 3, 64))
        queries = mx.random.normal((2, 12, 3, 64))
        view.prepare(lengths=[3, 2])
        mask = view.make_mask(3)
        view.update_and_fetch(keys, values)
        return view.bucketed_attention(queries, 0.125, mask)

    fast = run()
    assert [c["rows"] for c in spy_kernel] == [3, 2]
    monkeypatch.setenv("MLX2_QSDPA_VERIFY_KERNEL", "0")
    composed = run()
    assert len(spy_kernel) == 2  # explicit row masks, composed path
    assert mx.abs(fast[0] - composed[0]).max().item() < 1e-4
    assert mx.abs(fast[1, :, :2] - composed[1, :, :2]).max().item() < 1e-4


# --- Metal ------------------------------------------------------------------

metal = pytest.mark.skipif(not GPU, reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


def _strided(Hkv, T, D, kb, vb, B, dtype=mx.bfloat16, seed=3):
    """Quantized caches as views of 256-row-step storage, like a live cache."""
    from mlx2.runtime.models.cache import KVCache

    mx.random.seed(seed)
    parts = []
    for _ in range(B):
        c = KVCache()
        for s in range(0, T, 2048):
            n = min(2048, T - s)
            c.update_and_fetch(mx.random.normal((1, Hkv, n, D)).astype(dtype),
                               mx.random.normal((1, Hkv, n, D)).astype(dtype))
        parts.append(c.to_quantized(group_size=64, key_bits=kb, value_bits=vb).keys_and_values())
    if B == 1:
        return parts[0]
    return tuple(tuple(mx.concatenate([p[i][j] for p in parts], axis=0) for j in range(3))
                 for i in range(2))


def _assert_no_worse_than_reference(got, q, qk, qv, scale, lp, bits):
    """Kernel error vs exact fp32 attention within the dequantize-first path's own.

    The path reference (bf16 dequantize, bf16 SDPA) rounds K/V and its output
    to bf16; the kernel rounds only its output, so its error is bounded by the
    path's or by output rounding (rows with a few visible keys reach 1e-2).
    """
    kb, vb = bits
    exact = _dequant_sdpa(q, qk, qv, scale, left_padding=lp, kb=kb, vb=vb, dtype=mx.float32)
    path = _dequant_sdpa(q, qk, qv, scale, left_padding=lp, kb=kb, vb=vb).astype(mx.float32)
    kernel_err = mx.abs(got.astype(mx.float32) - exact).max().item()
    path_err = mx.abs(path - exact).max().item()
    assert mx.isfinite(got).all().item()
    # two bf16 rounding steps of the largest output (the kernel's own output cast)
    bound = max(1.25 * path_err, 2.0 ** -7 * mx.abs(exact).max().item())
    assert kernel_err <= bound, (kernel_err, path_err)


@metal
@pytest.mark.parametrize("Hq,Hkv", [(24, 4), (24, 2), (16, 2), (8, 8)])
@pytest.mark.parametrize("L", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("bits", [(8, 8), (4, 4)])
def test_kernel_matches_dequant_sdpa(Hq, Hkv, L, bits):
    mx.set_default_device(mx.gpu)
    try:
        qk, qv = _strided(Hkv, 9001, 256, *bits, B=1)
        q = mx.random.normal((1, Hq, L, 256)).astype(mx.bfloat16)
        got = qvm.gqa_quantized_verify_attention(q, qk, qv, scale=0.0625,
                                                 key_bits=bits[0], value_bits=bits[1])
        _assert_no_worse_than_reference(got, q, qk, qv, 0.0625, None, bits)
    finally:
        mx.set_default_device(mx.cpu)


@metal
@pytest.mark.parametrize("D", [128, 256])
def test_kernel_batched_left_padding(D):
    mx.set_default_device(mx.gpu)
    try:
        qk, qv = _strided(4, 6000, D, 8, 8, B=4)
        lp = mx.array([0, 1500, 17, 5999 - 3])
        q = mx.random.normal((4, 24, 3, D)).astype(mx.bfloat16)
        got = qvm.gqa_quantized_verify_attention(q, qk, qv, scale=0.0625, left_padding=lp)
        _assert_no_worse_than_reference(got, q, qk, qv, 0.0625, lp, (8, 8))
    finally:
        mx.set_default_device(mx.cpu)


@metal
def test_routed_path_takes_the_kernel():
    from mlx2.runtime.models import base

    mx.set_default_device(mx.gpu)
    try:
        qk, qv = _strided(4, qvm.MIN_CONTEXT + 77, 256, 8, 8, B=1)
        q = mx.random.normal((1, 24, 3, 256)).astype(mx.bfloat16)
        got = qvm.gqa_quantized_verify_attention(q, qk, qv, scale=0.0625)
        routed = base.quantized_scaled_dot_product_attention(
            q, qk, qv, scale=0.0625, mask="causal", group_size=64, bits=8)
        assert mx.array_equal(routed, got).item()
        os.environ["MLX2_QSDPA_VERIFY_KERNEL"] = "0"
        composed = base.quantized_scaled_dot_product_attention(
            q, qk, qv, scale=0.0625, mask="causal", group_size=64, bits=8)
        assert not mx.array_equal(composed, got).item()
        assert mx.abs(composed.astype(mx.float32) - got.astype(mx.float32)).max().item() < 5e-3
    finally:
        os.environ.pop("MLX2_QSDPA_VERIFY_KERNEL", None)
        mx.set_default_device(mx.cpu)
