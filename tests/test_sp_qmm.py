"""CPU tests for the small-M simdgroup quantized matmul (sp_qmm).

The Metal kernel itself needs a GPU (scripts/bench_sp_qmm.py checks it against
an fp32 reference there). These tests pin what can be checked on CPU:

- an exact emulation of the kernel's lane/fragment mapping on MLX-packed 4- and
  5-bit weights reproduces x @ dequantize(w).T (so the k permutation, the
  128 + q bfloat trick, the 5-bit halfword unpacking and the per-group
  epilogue are right);
- eligibility, instance-scoped routing and removal.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx2.runtime.models import sp_qmm  # noqa: E402


def _lane_map(lane):
    qid = lane >> 2
    return (qid & 4) | ((lane >> 1) & 3), ((qid & 2) << 1) | ((lane & 1) << 1)


def _emulate(x, wq, sc, bi, bits, N, K, M, xl=1):
    """Mirror of the Metal source, one 8x8x8 MMA sum per (tile, group)."""
    G = K // 64
    wbytes = np.array(wq).view(np.uint8).reshape(N, -1)
    sc = np.array(sc.astype(mx.float32))
    bi = np.array(bi.astype(mx.float32))
    Mp = -(-M // 8) * 8
    xp = np.zeros((Mp, K), np.float64)
    xp[:M] = np.array(x.astype(mx.float32))
    out = np.zeros((Mp, N))
    for n0 in range(0, N, 8):
        for mt in range(Mp // 8):
            for g in range(G):
                A = np.zeros((8, 8, 8))
                B = np.zeros((8, 8, 8))
                for lane in range(32):
                    fm, fn = _lane_map(lane)
                    c = fn >> 1
                    row = wbytes[n0 + fm]
                    if bits == 4:
                        w = row[g * 32 + 8 * c: g * 32 + 8 * c + 8].view("<u4")
                        w0, w1 = int(w[0]), int(w[1])
                    else:
                        hb = row[g * 40 + 10 * c: g * 40 + 10 * c + 10]
                        hp = [int(hb[2 * i]) | (int(hb[2 * i + 1]) << 8) for i in range(5)]
                    koff = 16 * (fm >> 1) + (8 if xl else 4) * (fm & 1)
                    for j in range(8):
                        if xl and bits == 4:
                            lo, hi = (w0 >> (4 * j)) & 0xF, (w1 >> (4 * j)) & 0xF
                        elif xl:
                            c0 = hp[0] | hp[1] << 16 | (hp[2] & 0xFF) << 32
                            c1 = (hp[2] >> 8) | hp[3] << 8 | hp[4] << 24
                            lo, hi = (c0 >> 5 * j) & 31, (c1 >> 5 * j) & 31
                        elif bits == 4:
                            pair = ((w0 if j < 4 else w1) >> (4 * (j & 3))) & 0x000F000F
                            lo, hi = pair & 0xFFFF, pair >> 16
                        else:
                            h, i = j >> 2, j & 3
                            chunk = ((hp[0] | hp[1] << 16 | (hp[2] & 0xFF) << 32) if h == 0
                                     else ((hp[2] >> 8) | hp[3] << 8 | hp[4] << 24))
                            lo, hi = (chunk >> 5 * i) & 31, (chunk >> 5 * (i + 4)) & 31
                        A[j, fm, fn], A[j, fm, fn + 1] = 128 + lo, 128 + hi
                        k = g * 64 + koff + (j if xl else (0 if j < 4 else 8) + (j & 3))
                        B[j, fm, fn] = xp[mt * 8 + fn, k]
                        B[j, fm, fn + 1] = xp[mt * 8 + fn + 1, k]
                D = sum(A[j] @ B[j] for j in range(8))
                sx = xp[mt * 8: mt * 8 + 8, g * 64:(g + 1) * 64].sum(-1)[None]
                s = sc[n0:n0 + 8, g][:, None]
                b = bi[n0:n0 + 8, g][:, None]
                out[mt * 8: mt * 8 + 8, n0:n0 + 8] += (s * (D - 128 * sx) + b * sx).T
    return out[:M]


@pytest.mark.parametrize("xl", [0, 1])
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("M", [3, 9])
def test_fragment_mapping_reproduces_matmul(bits, M, xl):
    with mx.stream(mx.cpu):
        mx.random.seed(bits * 10 + M)
        N, K = 16, 192
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        w32 = mx.dequantize(wq, sc.astype(mx.float32), bi.astype(mx.float32),
                            group_size=64, bits=bits)
        ref = np.array(x.astype(mx.float32) @ w32.T)
    got = _emulate(x, wq, sc, bi, bits, N, K, M, xl)
    assert np.abs(got - ref).max() < 1e-5 * max(1.0, np.abs(ref).max())


def test_supports():
    assert sp_qmm.supports(3, 5120, 5120, 4, 64)
    assert sp_qmm.supports(12, 5120, 17408, 5, 64)
    assert not sp_qmm.supports(3, 5120, 5120, 8, 64)
    assert not sp_qmm.supports(3, 5120, 5120, 4, 32)
    assert not sp_qmm.supports(3, 5120, 5120, 4, 64, dtype=mx.float16)
    assert not sp_qmm.supports(33, 5120, 5120, 4, 64)
    assert not sp_qmm.supports(3, 44, 5120, 4, 64)


def test_tiles_fit_shape(monkeypatch):
    monkeypatch.delenv("MLX2_SP_QMM_NF", raising=False)
    monkeypatch.delenv("MLX2_SP_QMM_KS", raising=False)
    nf, ks = sp_qmm.tiles(3, 48, 5120)
    assert 48 % (8 * nf) == 0 and 1 <= ks <= 80
    nf, ks = sp_qmm.tiles(3, 8, 128)
    assert nf == 1 and ks <= 2


def _model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(128, 64, bias=False)
            self.b = nn.Linear(128, 64, bias=False)
            self.c = nn.Linear(128, 64, bias=False)

    with mx.stream(mx.cpu):
        m = M()
        m.set_dtype(mx.bfloat16)  # real checkpoints carry bf16 scales/biases
        nn.quantize(m, group_size=64, bits=4,
                    class_predicate=lambda p, _: p in ("a", "b"))
        m.b.bits = 8  # not eligible after the fact: 8-bit
    return m


def test_apply_routes_only_eligible_small_m(monkeypatch):
    m = _model()
    calls = []

    def fake(x, w, s, b, group_size=64, bits=4, **_):
        calls.append(x.shape)
        return mx.zeros((*x.shape[:-1], w.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(sp_qmm, "qmm", fake)
    h = sp_qmm.apply(m, min_m=2, max_m=16, policy=False)
    assert len(h) == 1 and type(m.a).__name__ == "SpQuantizedLinear"
    assert isinstance(m.a, nn.QuantizedLinear)
    with mx.stream(mx.cpu):
        m.a(mx.zeros((1, 3, 128), dtype=mx.bfloat16))       # routed
        m.a(mx.zeros((1, 1, 128), dtype=mx.bfloat16))       # M = 1: stock
        m.a(mx.zeros((1, 17, 128), dtype=mx.bfloat16))      # M > max: stock
        m.a(mx.zeros((1, 3, 128), dtype=mx.float16))        # fp16: stock
    assert calls == [(1, 3, 128)]
    sp_qmm.remove(h)
    assert type(m.a) is nn.QuantizedLinear
    assert type(m.c) is nn.Linear


def test_five_bit_32bit_unpack_matches_40bit_chunks():
    """The kernel splits each 40-bit chunk into a 32-bit low word and an 8-bit
    high byte; value 6 (bits 30..34) straddles them."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        hp = [int(v) for v in rng.integers(0, 1 << 16, 5)]
        lo = [hp[0] | (hp[1] << 16), (hp[2] >> 8) | (hp[3] << 8) | ((hp[4] << 24) & 0xFFFFFFFF)]
        hi = [hp[2] & 0xFF, hp[4] >> 8]
        chunks = [hp[0] | hp[1] << 16 | (hp[2] & 0xFF) << 32,
                  (hp[2] >> 8) | hp[3] << 8 | hp[4] << 24]
        for h in range(2):
            for v in range(8):
                p = 5 * v
                if p >= 32:
                    got = (hi[h] >> (p - 32)) & 31
                elif p + 5 > 32:
                    got = ((lo[h] >> p) | (hi[h] << (32 - p))) & 31
                else:
                    got = (lo[h] >> p) & 31
                assert got == (chunks[h] >> p) & 31


def test_policy_matches_measured_crossovers():
    # 27B shapes: stock keeps M <= 6 and 4-bit 12..16 (NAX); sp takes the
    # NAX switch rows 8..11 and 5-bit 8..16.
    assert not sp_qmm.prefer(3, 17408, 5120, 4)
    assert not sp_qmm.prefer(6, 17408, 5120, 4)
    assert sp_qmm.prefer(8, 17408, 5120, 4) and sp_qmm.prefer(11, 5120, 17408, 4)
    assert not sp_qmm.prefer(12, 17408, 5120, 4)
    assert sp_qmm.prefer(12, 5120, 17408, 5) and sp_qmm.prefer(16, 5120, 6144, 5)
    assert not sp_qmm.prefer(6, 5120, 6144, 5)
    assert sp_qmm.prefer(8, 248320, 5120, 4) and not sp_qmm.prefer(4, 248320, 5120, 4)
    assert not sp_qmm.prefer(12, 248320, 5120, 4)
    assert not sp_qmm.prefer(8, 48, 5120, 5)
    assert not sp_qmm.prefer(24, 5120, 17408, 5)


def test_serving_policy_is_boolean_and_default_off():
    from mlx2.serving import ServingEngine, sp_qmm_status

    with pytest.raises(ValueError, match="sp_qmm must be boolean"):
        ServingEngine("unused", execution_policy={"sp_qmm": 1})

    class Off:
        sp_qmm_enabled = False

    assert sp_qmm_status(Off()) == {"enabled": False}


def test_env_switch_selects_and_policy_wins(monkeypatch):
    from mlx2.serving import sp_qmm_selected

    monkeypatch.setenv("MLX2_SP_QMM", "1")
    assert sp_qmm_selected(None) is True
    assert sp_qmm_selected({"sp_qmm": False}) is False
    monkeypatch.setenv("MLX2_SP_QMM", "0")
    assert sp_qmm_selected({}) is False
    assert sp_qmm_selected({"sp_qmm": True}) is True


def test_float32_scales_fall_back_to_stock():
    """fp32_head re-types the vocab head's scales/biases to float32; the bf16
    kernel must not read them (Codex review of the 2026-09-26 integration)."""
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models import sp_qmm

    layer = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=4)
    layer.scales = layer.scales.astype(mx.float32)
    layer.biases = layer.biases.astype(mx.float32)
    handle = sp_qmm.apply(layer, min_m=1, max_m=16, policy=False)
    try:
        before = dict(sp_qmm.STATS)
        y = layer(mx.zeros((8, 128), dtype=mx.bfloat16))
        mx.eval(y)
        assert sp_qmm.STATS["routed"] == before.get("routed", 0)
        assert sp_qmm.STATS["stock"] == before.get("stock", 0) + 1
    finally:
        sp_qmm.remove(handle)
    import pytest

    q = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=4)
    with pytest.raises(ValueError, match="bf16 scales"):
        sp_qmm.qmm(mx.zeros((8, 128), dtype=mx.bfloat16), q.weight,
                   q.scales.astype(mx.float32), q.biases, group_size=64, bits=4)
