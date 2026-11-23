"""mlx2.gptq: GPTQ-style quantization to MLX affine planes (CPU).

Oracles: the packed planes must dequantize through MLX's own kernels to
exactly the values GPTQ committed to, and on correlated calibration inputs
GPTQ must give a lower layer-output error than round-to-nearest (the
algorithm's purpose), for dense weights and for per-expert banks.
"""
import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import gptq as G  # noqa: E402


def correlated_rows(n, I, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, I // 4)).astype(np.float32)
    mix = rng.standard_normal((I // 4, I)).astype(np.float32)
    x = base @ mix + 0.1 * rng.standard_normal((n, I)).astype(np.float32)
    x[:, :8] *= 6.0  # a few high-energy input channels
    return x


def out_error(x, W, Wq):
    return float(np.mean((x @ W.T - x @ Wq.T) ** 2))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_packed_planes_dequantize_exactly(bits):
    rng = np.random.default_rng(1)
    q = rng.integers(0, 2 ** bits, (3, 128))
    packed = G.pack(mx.array(q), bits)
    scales = mx.array(rng.uniform(0.01, 0.2, (3, 2)).astype(np.float32))
    biases = mx.array(rng.uniform(-1, 1, (3, 2)).astype(np.float32))
    got = np.array(mx.dequantize(packed, scales, biases, group_size=64, bits=bits))
    want = q * np.repeat(np.array(scales), 64, axis=1) + np.repeat(np.array(biases), 64, axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bits,act_order", [(2, False), (4, False), (2, True)])
def test_gptq_beats_rtn_on_layer_output_and_planes_match(bits, act_order):
    rng = np.random.default_rng(2)
    O, I, n = 48, 128, 512
    W = (rng.standard_normal((O, I)) * 0.1).astype(np.float32)
    x = correlated_rows(n, I, 3)
    H = mx.array(x.T @ x)[None]
    s = G.GPTQSettings(bits=bits, group_size=32, block=64, act_order=act_order)
    (wq, sc, bi), receipt = G.quantize_bank(mx.array(W), lambda a, b: H, s, dtype=mx.float32)
    Wg = np.array(mx.dequantize(wq, sc, bi, group_size=32, bits=bits))
    q, rs, rb = G.rtn(mx.array(W), bits=bits, group_size=32, dtype=mx.float32)
    Wr = np.array(mx.dequantize(G.pack(q, bits), rs, rb, group_size=32, bits=bits))
    assert out_error(x, W, Wg) < 0.8 * out_error(x, W, Wr)
    assert receipt["method"] == "gptq" and receipt["act_order"] == act_order
    assert wq.dtype == mx.uint32 and wq.shape == (O, I * bits // 32) and sc.shape == (O, I // 32)


def test_expert_bank_uses_each_experts_rows_and_falls_back_for_starved_experts():
    rng = np.random.default_rng(4)
    E, O, I = 6, 16, 64
    W = (rng.standard_normal((E, O, I)) * 0.1).astype(np.float32)
    # One recorded bank call: rows routed to experts, expert 5 never routed.
    x = correlated_rows(300, I, 5)
    ids = rng.integers(0, 5, (300, 2))
    rec = G.InputRecorder(lambda xx, ii, sorted_indices=False: None, bank=True)
    rec(mx.array(x)[:, None, None, :], mx.array(ids))
    h, counts = rec.hessians(E)
    assert counts[5] == 0 and sum(counts) == 600
    # The recorder keeps rows in bf16 (memory); the oracle rounds the same way.
    xb = np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    for e in range(5):
        rows = np.repeat(xb, 2, axis=0)[ids.reshape(-1) == e]
        np.testing.assert_allclose(np.array(h(e, e + 1))[0], rows.T @ rows, rtol=1e-4, atol=1e-1)
    s = G.GPTQSettings(bits=2, group_size=32, block=32, expert_chunk=4, min_rows=8)
    (wq, sc, bi), receipt = G.quantize_bank(mx.array(W), h, s, dtype=mx.float32, rows=counts)
    assert receipt["rtn_fallback_experts"] == 1 and wq.shape == (E, O, I * 2 // 32)
    Wg = np.array(mx.dequantize(wq, sc, bi, group_size=32, bits=2))
    q, rs, rb = G.rtn(mx.array(W[5]), bits=2, group_size=32, dtype=mx.float32)
    np.testing.assert_allclose(Wg[5], np.array(mx.dequantize(G.pack(q, 2), rs, rb, group_size=32, bits=2)))
    for e in range(5):
        rows = np.repeat(x, 2, axis=0)[ids.reshape(-1) == e]
        q, rs, rb = G.rtn(mx.array(W[e]), bits=2, group_size=32, dtype=mx.float32)
        Wr = np.array(mx.dequantize(G.pack(q, 2), rs, rb, group_size=32, bits=2))
        assert out_error(rows, W[e], Wg[e]) <= out_error(rows, W[e], Wr) * 1.001


def test_dead_columns_and_storage_dtype_rounding():
    rng = np.random.default_rng(6)
    W = (rng.standard_normal((8, 64)) * 0.1).astype(np.float32)
    x = correlated_rows(200, 64, 7)
    x[:, 10] = 0.0  # a dead input channel
    s = G.GPTQSettings(bits=4, group_size=32, block=32)
    (wq, sc, bi), _ = G.quantize_bank(mx.array(W), lambda a, b: mx.array(x.T @ x)[None], s, dtype=mx.bfloat16)
    assert sc.dtype == mx.bfloat16 and bi.dtype == mx.bfloat16
    Wg = np.array(mx.dequantize(wq, sc, bi, group_size=32, bits=4).astype(mx.float32))
    assert np.isfinite(Wg).all() and np.all(Wg[:, 10] == Wg[:, 10])


def test_settings_refuse_unpackable_bits_and_bad_groups():
    with pytest.raises(ValueError, match="packs bits"):
        G.GPTQSettings(bits=3).validate(64)
    with pytest.raises(ValueError, match="multiple of group_size"):
        G.GPTQSettings(bits=2, group_size=64).validate(96)


def test_bank_hessians_are_evaluated_in_small_expert_groups(monkeypatch):
    """The per-text update never builds one lazy graph over every expert."""
    import mlx.core as mx
    import numpy as np
    from mlx2 import gptq as G

    E, I = 20, 8
    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((1, 30, 1, 1, I)).astype(np.float32))
    idx = mx.array(rng.integers(0, E, size=(1, 30, 4)).astype(np.uint32))

    def bank(x, indices, sorted_indices=False):
        return x

    whole = G.HessianAccumulator(bank, bank=True, experts=E)
    whole.eval_experts = 10 ** 6
    grouped = G.HessianAccumulator(bank, bank=True, experts=E)
    sizes = []
    real_eval = mx.eval

    def spy(*args):
        sizes.append(len(args[0]) if args and isinstance(args[0], list) else 1)
        return real_eval(*args)

    whole(x, idx)
    monkeypatch.setattr(mx, "eval", spy)
    grouped(x, idx)
    assert sizes and max(sizes) <= grouped.eval_experts
    for e in range(E):
        if whole.H[e] is None:
            assert grouped.H[e] is None
        else:
            np.testing.assert_array_equal(np.array(grouped.H[e]), np.array(whole.H[e]))
    assert grouped.counts == whole.counts


def test_rtn_group_law_is_mlx_quantize():
    """Round-to-nearest uses mx.quantize's affine law: same scales and biases."""
    import mlx.core as mx
    import numpy as np
    from mlx2 import gptq as G

    rng = np.random.default_rng(1)
    as32 = lambda a: np.array(a.astype(mx.float32))  # noqa: E731
    for dtype in (mx.bfloat16, mx.float16):
        for bits in (2, 4, 8):
            W = mx.array((rng.standard_t(4, (64, 256)) * 0.03).astype(np.float32)).astype(dtype)
            _, s, b = mx.quantize(W, group_size=64, bits=bits)
            _, sc, bi = G.rtn(W, bits=bits, group_size=64, dtype=dtype)
            np.testing.assert_array_equal(as32(sc.astype(dtype)), as32(s))
            np.testing.assert_array_equal(as32(bi.astype(dtype)), as32(b))
