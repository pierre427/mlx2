"""CPU tests for the NAX sorted gather's affine widths beyond 4/8-bit.

The kernels run only on an M5 GPU (scripts/nax6_canary.py and
qualification/runs/research-20261006/nax-6bit/); here: mlx's packed layout
as the kernel's unpack reads it, the loader-split check per width and the
plan fit, kernel capability (``supports``) versus route admission
(``AFFINE_BITS``), and the per-format counters.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models import switch_layers as sl


@pytest.fixture(autouse=True)
def _reset():
    old_mode, old_bits = nax.MODE, nax.AFFINE_BITS
    nax.status(reset=True)
    with nax.prefill_scope():
        yield
    nax.set_mode(old_mode)
    nax.set_affine_bits(old_bits)
    nax.status(reset=True)


def _qlinear(E=4, out=128, inp=128, bits=6, gs=64, dtype=mx.bfloat16):
    lin = sl.SwitchLinear(inp, out, E, bias=False).to_quantized(group_size=gs, bits=bits)
    lin.scales = lin.scales.astype(dtype)
    lin.biases = lin.biases.astype(dtype)
    return lin


# --- layout -----------------------------------------------------------------


@pytest.mark.parametrize("bits", nax.AFFINE_KERNEL_BITS)
def test_unpack_reference_reads_mlx_packing(bits):
    """The kernel's chunked bit-stream unpack recovers mx.quantize's codes."""
    K, gs = 256, 64
    w = mx.random.normal((3, K), key=mx.random.key(bits)).astype(mx.float32)
    wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
    assert wq.shape == (3, K * bits // 32)
    deq = np.array(mx.dequantize(wq, scales, biases, group_size=gs, bits=bits))
    s = np.repeat(np.array(scales), gs, axis=1)
    b = np.repeat(np.array(biases), gs, axis=1)
    words = np.array(wq).astype(np.uint64)
    for r in range(3):
        q = np.array(nax.unpack_reference(words[r], bits, K), dtype=np.float64)
        assert q.max() < 2**bits
        np.testing.assert_allclose(s[r] * q + b[r], deq[r], rtol=0, atol=1e-6)


def test_unpack_reference_six_bit_straddles_words():
    # Values 0..15 of one 16-value chunk: 6-bit value 5 spans bits 30..35.
    vals = list(range(1, 17))
    stream = sum(v << (6 * j) for j, v in enumerate(vals))
    words = [(stream >> (32 * i)) & 0xFFFFFFFF for i in range(3)]
    assert nax.unpack_reference(words, 6, 16) == vals


# --- loader split / plans ---------------------------------------------------


def test_loader_geometry():
    assert nax._loader_geometry(64, 64) == (128, 32)
    assert nax._loader_geometry(64, 128) == (128, 64)
    assert nax._loader_geometry(96, 64) == (128, 32)  # 192 threads, 128 load
    assert nax._loader_geometry(128, 64) == (256, 16)
    assert nax._loader_geometry(128, 128) == (256, 32)


def _all_plans():
    P = nax.Plan
    return [P(s, bm, bk, gx, pad) for s in (nax._SCHED_SEG, nax._SCHED_DB)
            for bm in nax._TILE_ROWS for bk in (64, 128) for gx in (0, nax._GX)
            for pad in (0, 8192)]


@pytest.mark.parametrize("bits", (2, 4, 6, 8))
def test_geometry_holds_every_plan_for_word_aligned_widths(bits):
    for gs in (32, 64, 128):
        assert all(nax.geometry_ok(p, bits, gs) for p in _all_plans())


@pytest.mark.parametrize("bits", (3, 5))
def test_geometry_odd_widths_need_32_value_runs(bits):
    P, SEG = nax.Plan, nax._SCHED_SEG
    assert not nax.geometry_ok(P(SEG, 128, 64, 0, 0), bits, 64)  # 16 values
    assert nax.geometry_ok(P(SEG, 128, 128, 32, 8192), bits, 64)
    assert nax.geometry_ok(P(SEG, 64, 64, 0, 0), bits, 32)
    fit = nax._fit_plan(P(SEG, 128, 64, 32, 8192), bits, 64)
    assert fit == P(SEG, 64, 64, 32, 0) and nax.geometry_ok(fit, bits, 64)


def test_fit_plan_keeps_six_bit_planner_choices():
    for rows in (2048, 16384, 65536):
        for K, N in ((2048, 1024), (512, 2048)):
            p = nax._plan(rows, 256, K, N)
            assert nax._fit_plan(p, 6, 64) == p


# --- capability vs admission ------------------------------------------------


@pytest.mark.parametrize("bits", nax.AFFINE_KERNEL_BITS)
def test_supports_every_kernel_width(bits):
    lin = _qlinear(bits=bits)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    args = dict(x=x, w=lin.weight, scales=lin.scales, biases=lin.biases,
                indices=idx, group_size=64, mode="affine")
    assert nax.supports(bits=bits, **args)
    # Packed width must match the declared width.
    other = 4 if bits != 4 else 6
    assert not nax.supports(bits=other, **args)


def test_supports_refuses_seven_bit():
    lin = _qlinear(bits=4)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    assert not nax.supports(x, lin.weight, lin.scales, lin.biases,
                            mx.zeros((64,), mx.uint32), 64, 7, "affine")


def test_affine_bits_env_parsing():
    assert nax.affine_bits_from_env({}) == nax.DEFAULT_AFFINE_BITS
    assert nax.affine_bits_from_env({nax.ENV_BITS: "8, 6,4"}) == (4, 6, 8)
    assert nax.affine_bits_from_env({nax.ENV_BITS: " "}) == nax.DEFAULT_AFFINE_BITS
    for bad in ("7", "4,x", ",", "16"):
        with pytest.raises(ValueError):
            nax.affine_bits_from_env({nax.ENV_BITS: bad})
    from mlx2.runtime.models.import_env import PREFIXES

    assert nax.ENV_BITS.startswith(PREFIXES)


def test_set_affine_bits_and_admission():
    old = nax.set_affine_bits([6, 4])
    assert nax.AFFINE_BITS == (4, 6)
    assert nax.set_affine_bits(old) == (4, 6)
    with pytest.raises(ValueError):
        nax.set_affine_bits([7])
    with pytest.raises(ValueError):
        nax.set_affine_bits([])
    nax.set_affine_bits([4, 8])
    assert nax.bits_admitted("affine", 4) and not nax.bits_admitted("affine", 6)
    assert nax.bits_admitted("mxfp4", 4)


def _gpu(monkeypatch):
    monkeypatch.setattr(nax, "_nax_host", True)
    monkeypatch.setattr(nax.mx, "default_device", lambda: nax.mx.gpu)


def test_try_gather_six_bit_not_admitted_is_counted(monkeypatch):
    _gpu(monkeypatch)
    nax.set_affine_bits([4, 8])
    monkeypatch.setattr(nax, "sorted_gather_qmm", lambda *a, **k: pytest.fail("offered"))
    lin = _qlinear(bits=6)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    assert nax.try_gather(x, lin, mx.zeros((64,), mx.uint32)) is None
    st = nax.status()
    assert st["fallbacks"] == {"bits_not_admitted": 1}
    assert st["affine_bits"] == [4, 8] and st["calls_by_format"] == {}


def test_try_gather_six_bit_admitted_counts_format(monkeypatch):
    _gpu(monkeypatch)
    nax.set_affine_bits([4, 6, 8])
    seen = []

    def fake(x, w, s, b, idx, *, group_size, bits, mode, **k):
        seen.append(bits)
        return mx.ones((64, 1, 128))

    monkeypatch.setattr(nax, "sorted_gather_qmm", fake)
    lin = _qlinear(bits=6)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    assert nax.try_gather(x, lin, mx.zeros((64,), mx.uint32)) is not None
    st = nax.status()
    assert seen == [6] and st["calls"]["gather"] == 1
    assert st["calls_by_format"] == {"gather/affine-6": 1}
    nax.status(reset=True)
    assert nax.status()["calls_by_format"] == {}


def test_try_swiglu_split_six_bit_admission(monkeypatch):
    _gpu(monkeypatch)
    gate, up = _qlinear(bits=6), _qlinear(bits=6)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    idx = mx.zeros((64,), mx.uint32)
    rows = (x[:16], mx.zeros((64,), mx.uint32))
    nax.set_affine_bits([4, 8])
    assert nax.try_swiglu_split(gate, up, x, idx, rows) is None
    assert nax.status()["fallbacks"] == {"bits_not_admitted": 1}
    nax.status(reset=True)
    nax.set_affine_bits([4, 6, 8])
    monkeypatch.setattr(nax, "sorted_gather_qmm_swiglu_split",
                        lambda *a, row_map=None, **k: mx.ones((64, 1, 128)))
    assert nax.try_swiglu_split(gate, up, x, idx, rows) is not None
    st = nax.status()
    assert st["calls"]["swiglu_split_map"] == 1
    assert st["calls_by_format"] == {"swiglu_split_map/affine-6": 1}
    assert st["fallbacks"] == {}


def test_try_swiglu_six_bit_not_admitted(monkeypatch):
    _gpu(monkeypatch)
    nax.set_affine_bits([4, 8])
    proj = _qlinear(bits=6)
    x = mx.zeros((64, 1, 128), mx.bfloat16)
    assert nax.try_swiglu(proj, x, mx.zeros((64,), mx.uint32)) is None
    assert nax.status()["fallbacks"] == {"bits_not_admitted": 1}


def test_entry_points_fit_odd_width_plans(monkeypatch):
    """A pinned plan whose loader split cannot hold 3-bit runs is fitted to
    64x64 tiles before the self-test key and the launch."""
    _gpu(monkeypatch)
    launched, keys = [], []
    monkeypatch.setattr(nax, "_launch", lambda *a, **k: launched.append(a[8]) or mx.ones((1,)))
    monkeypatch.setattr(nax, "_checked", lambda key, test: keys.append(key) or True)
    lin = _qlinear(bits=3, out=128, inp=256)
    x = mx.zeros((64, 1, 256), mx.bfloat16)
    P = nax.Plan
    out = nax.sorted_gather_qmm(x, lin.weight, lin.scales, lin.biases,
                                mx.zeros((64,), mx.uint32), group_size=64, bits=3,
                                plan=P(nax._SCHED_SEG, 128, 64, 32, 8192))
    assert out is not None
    assert launched == [P(nax._SCHED_SEG, 64, 64, 32, 0)]
    assert keys[0][4] == launched[0]


def test_default_admits_measured_widths_only():
    # 6-bit measured faster than stock on M5 (research-20261006/nax-6bit);
    # 2/3/5-bit are canary-exact but unmeasured.
    assert nax.DEFAULT_AFFINE_BITS == (4, 6, 8)
    assert set(nax.AFFINE_KERNEL_BITS) - set(nax.DEFAULT_AFFINE_BITS) == {2, 3, 5}
