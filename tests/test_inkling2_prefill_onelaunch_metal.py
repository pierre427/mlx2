"""inkling2_prefill_onelaunch_attention on Metal (GPU; run under the GPU lock).

Checkpoint: synthetic (spec section 4 inputs at the Inkling 2 Mini attention
geometry: 24 query heads over 8 KV heads, head dim 128, E = W = 1024 on
window visits, E = 1024 on global visits). Each test runs on every kernel
form this GPU has (simd always; tensor on M5-class hosts with macOS 26.2+),
each form compared with itself for the bitwise properties:

- N1: relative RMS error against float64 within the dtype bound and no worse
  than the composed path (banded mask + sink row + SDPA) against the same
  reference; finite; correctly rounded at least as often as the composed path;
- N3: chunk widths {37, 256, 512, 2048, N} over N = 6000 positions give
  bitwise equal rows, window (uncropped and window-cropped) and global;
- N4: ring buffers at rotations {0, 1, 511, 1023}, with and without a crop
  start, before and after the wrap, bitwise equal to the chronological copy;
- N5: left padding p in {0, 1, 31, 100} bitwise equal to the sequence alone,
  padding rows exactly 0;
- N6: sink semantics; N7: no-skip and no-fast debug builds bitwise equal;
- N9: peak memory growth of a 2048-row call at 16K keys within output + 64 MiB;
- the first-use self-check passes.

Set ``INKLING2_ONELAUNCH_METAL_RECEIPT`` to a path for a JSON summary (mlx2
commit, device, forms, the N1 numbers).
"""

import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.inkling2 import prefill_onelaunch as P

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {}
H, HKV, D, E, W = 24, 8, 128, 1024, 1024
SCALE = D ** -0.5


def _forms():
    if not mx.metal.is_available():
        return ["simd"]
    return ["simd"] + (["tensor"] if P.tensor_units()[0] else [])


FORMS = _forms()


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def teardown_module(_module):
    path = os.environ.get("INKLING2_ONELAUNCH_METAL_RECEIPT")
    if path:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
        receipt = {"mlx2": head, "mlx_version": mx.__version__,
                   "device": mx.metal.device_info().get("device_name"),
                   "architecture": mx.metal.device_info().get("architecture"),
                   "forms": FORMS, "checkpoint": "synthetic", "results": RESULTS}
        with open(path, "w") as handle:
            json.dump(receipt, handle, indent=1, sort_keys=True)


def _run(form, args, **kw):
    out = P.onelaunch_prefill_attention(*args, form=form, **kw)
    mx.eval(out)
    return out


def _window_kw(window):
    return dict(window=window, scale=SCALE)


# ----- N1 / N2 -----

N1_CASES = [  # (rows, history, window)
    (33, 0, 0), (33, 991, 0), (200, 900, 0), (512, 1500, 0), (2048, 2048, 0),
    (33, 20000, 0), (33, 0, W), (200, 1024, W), (512, 3000, W), (2048, 1000, W),
    (37, 20000, W),
]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("case", N1_CASES)
@pytest.mark.parametrize("form", FORMS)
def test_n1_numerics_against_float64(form, case, dtype):
    rows, history, window = case
    size = history + rows
    args = P.synthetic(31, b=1, hq=H, hkv=HKV, length=rows, size=size, dim=D, extent=E,
                       dtype=dtype)
    kw = dict(offset=history, left_padding=None, **_window_kw(window))
    got = _run(form, args, **kw)
    stock = P.composed(*args, **kw)
    mx.eval(stock)
    ref = P.reference64(*args, **kw)
    err, finite, exact = P.numerics(got, ref, dtype)
    s_err, s_finite, s_exact = P.numerics(stock, ref, dtype)
    agree = P.numerics(got, np.array(stock.astype(mx.float32), dtype=np.float64), dtype)[0]
    RESULTS[f"n1/{form}/{dtype}/{case}"] = dict(err=err, stock_err=s_err, exact=exact,
                                               stock_exact=s_exact, vs_composed=agree)
    assert finite
    assert err <= P.TOLERANCE[dtype], (err, s_err)
    assert err <= s_err * 1.0001 + 1e-7, (err, s_err)
    assert exact >= s_exact, (exact, s_exact)
    assert agree <= P.TOLERANCE[dtype] + s_err  # N2: near the composed path


# ----- N3 chunk-width invariance -----

N = 6000
# 37 and 100 leave partial query blocks (not multiples of 8 or 16 rows):
# the tensor form's edge-checked matmul path. Its bitwise invariance there is
# to be shown on M5 (this test is that evidence).
WIDTHS = (37, 100, 256, 512, 2048, N)


@pytest.fixture(scope="module")
def prompt():
    return P.synthetic(32, b=1, hq=H, hkv=HKV, length=N, size=N, dim=D, extent=E)


def _chunked(form, prompt, width, window, crop):
    q, k, v, rel, sinks = prompt
    outs = []
    for c0 in range(0, N, width):
        c1 = min(N, c0 + width)
        start = max(0, c0 - window + 1) if (window and crop) else 0
        args = (q[:, :, c0:c1], k[:, :, start:c1], v[:, :, start:c1], rel[:, c0:c1],
                sinks[:, c0:c1])
        outs.append(_run(form, args, offset=c0 - start, left_padding=None, abs_base=start,
                         **_window_kw(window)))
    return mx.concatenate(outs, axis=2)


@pytest.mark.parametrize("variant", ["global", "window", "window_cropped"])
@pytest.mark.parametrize("form", FORMS)
def test_n3_chunk_width_invariance_bitwise(form, variant, prompt):
    window = 0 if variant == "global" else W
    crop = variant == "window_cropped"
    base = _chunked(form, prompt, WIDTHS[0], window, crop)
    for width in WIDTHS[1:]:
        got = _chunked(form, prompt, width, window, crop)
        same = bool(mx.array_equal(got, base).item())
        RESULTS[f"n3/{form}/{variant}/{width}"] = same
        assert same, (width, float(mx.abs(got.astype(mx.float32)
                                          - base.astype(mx.float32)).max().item()))


@pytest.mark.parametrize("p", [1, 31, 100])
@pytest.mark.parametrize("form", FORMS)
def test_n3_n5_left_padding_bitwise(form, p, prompt):
    """A left-padded batch row (B = 2, the other row unpadded) reproduces the
    sequence alone bitwise, chunked at two widths; padding rows are 0."""
    n = 1500
    q, k, v, rel, sinks = prompt
    q, k, v, rel, sinks = q[:, :, :n], k[:, :, :n], v[:, :, :n], rel[:, :n], sinks[:, :n]

    def lpad(a, axis):
        shape = list(a.shape)
        shape[axis] = p
        return mx.concatenate([mx.zeros(shape, a.dtype), a], axis=axis)

    def rpad(a, axis):
        shape = list(a.shape)
        shape[axis] = p
        return mx.concatenate([a, mx.zeros(shape, a.dtype)], axis=axis)

    bq = mx.concatenate([rpad(q, 2), lpad(q, 2)], axis=0)
    bk = mx.concatenate([rpad(k, 2), lpad(k, 2)], axis=0)
    bv = mx.concatenate([rpad(v, 2), lpad(v, 2)], axis=0)
    brel = mx.concatenate([rpad(rel, 1), lpad(rel, 1)], axis=0)
    bs = mx.concatenate([rpad(sinks, 1), lpad(sinks, 1)], axis=0)
    pads = mx.array([0, p], dtype=mx.int32)
    for window in (0, W):
        alone = _run(form, (q, k, v, rel, sinks), offset=0, left_padding=None,
                     **_window_kw(window))
        for width in (256, 2048):
            outs = []
            for c0 in range(0, n + p, width):
                c1 = min(n + p, c0 + width)
                outs.append(_run(form, (bq[:, :, c0:c1], bk[:, :, :c1], bv[:, :, :c1],
                                        brel[:, c0:c1], bs[:, c0:c1]),
                                 offset=c0, left_padding=pads, **_window_kw(window)))
            got = mx.concatenate(outs, axis=2)
            assert bool(mx.array_equal(got[1:, :, p:], alone).item()), (window, width)
            assert bool(mx.array_equal(got[:1, :, :n], alone).item()), (window, width)
            assert bool(mx.all(got[1:, :, :p].astype(mx.float32) == 0).item())
            assert bool(mx.all(mx.isfinite(got.astype(mx.float32))).item())


# ----- N4 ring -----

@pytest.mark.parametrize("rot", [0, 1, 511, 1023])
@pytest.mark.parametrize("crop", [0, 200])
@pytest.mark.parametrize("form", FORMS)
def test_n4_ring_before_and_after_wrap_bitwise(form, rot, crop):
    rows = 512
    for width, label in ((1024 + rows, "full"), (1024 + rows + 300, "spare")):
        args = P.synthetic(33, b=1, hq=H, hkv=HKV, length=rows, size=width, dim=D, extent=E)
        q, k, v, rel, sinks = args
        # Before the wrap: rot 0 is the buffer as is. After: the chronological
        # keys start at physical row rot.
        size = width - crop
        kw = dict(offset=size - rows, left_padding=None, abs_base=crop, **_window_kw(W))
        base = _run(form, (q, k[:, :, crop:], v[:, :, crop:], rel, sinks), **kw)
        ring = ((rot + crop) % width, size)
        got = _run(form, (q, mx.roll(k, rot, axis=2), mx.roll(v, rot, axis=2), rel, sinks),
                   ring=ring, **kw)
        assert bool(mx.array_equal(got, base).item()), (label, rot, crop)


# ----- N5(c), N6 -----

@pytest.mark.parametrize("form", FORMS)
def test_n6_sink_semantics(form):
    args = P.synthetic(34, b=1, hq=H, hkv=HKV, length=64, size=1500, dim=D, extent=E)
    q, k, v, rel, sinks = args
    kw = dict(offset=1500 - 64, left_padding=None, **_window_kw(0))
    neg = mx.full(sinks.shape, -mx.inf, mx.float32)
    got = _run(form, (q, k, v, rel, neg), **kw)
    ref = P.reference64(q, k, v, rel, mx.full(sinks.shape, -1e30, mx.float32), **kw)
    err, finite, _ = P.numerics(got, ref, mx.bfloat16)
    assert finite and err <= P.TOLERANCE[mx.bfloat16]
    big = _run(form, (q, k, v, rel, mx.full(sinks.shape, 1e4, mx.float32)), **kw)
    assert float(mx.abs(big.astype(mx.float32)).max().item()) == 0.0
    # A NaN sink makes its row NaN, as in the composed path.
    nan_sinks = mx.where(mx.arange(64)[None, :, None] == 9, mx.nan, sinks.astype(mx.float32))
    nan_out = _run(form, (q, k, v, rel, nan_sinks), **kw).astype(mx.float32)
    assert bool(mx.all(mx.isnan(nan_out[:, :, 9])).item())
    assert bool(mx.all(mx.isfinite(nan_out[:, :, 10:])).item())
    # Every query in the padding: exactly 0.
    empty = _run(form, args, offset=1500 - 64, left_padding=mx.array([1500]),
                 **_window_kw(0))
    assert bool(mx.all(empty.astype(mx.float32) == 0).item())


# ----- N7 debug builds -----

@pytest.mark.parametrize("window,extent", [(0, E), (W, 256)])
@pytest.mark.parametrize("form", FORMS)
def test_n7_skip_and_fast_paths_are_identities(form, window, extent):
    """The bias-free fast path engages on tiles at distance >= E from every
    row of a simdgroup (and inside the window): global with E = 1024 over
    3000 keys, and a window shape with E = 256 < W = 1024 (the Mini's window
    visits have E = W, where the fast path cannot engage)."""
    args = P.synthetic(35, b=2, hq=H, hkv=HKV, length=300, size=3000, dim=D, extent=extent)
    kw = dict(offset=2700, left_padding=mx.array([0, 77]), **_window_kw(window))
    base = _run(form, args, **kw)
    assert bool(mx.array_equal(_run(form, args, no_skip=True, **kw), base).item())
    assert bool(mx.array_equal(_run(form, args, no_fast=True, **kw), base).item())


# ----- N9 memory, self-check -----

@pytest.mark.parametrize("form", FORMS)
def test_n9_memory_is_the_output(form):
    rows, keys = 2048, 16384
    args = P.synthetic(36, b=1, hq=H, hkv=HKV, length=rows, size=keys, dim=D, extent=E)
    mx.eval(args)
    mx.synchronize()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    out = _run(form, args, offset=keys - rows, left_padding=None, **_window_kw(0))
    growth = mx.get_peak_memory() - before
    out_bytes = out.size * out.dtype.size
    RESULTS[f"n9/{form}"] = dict(growth=growth, output=out_bytes)
    assert growth <= out_bytes + (64 << 20), (growth, out_bytes)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("window", [0, W])
@pytest.mark.parametrize("form", FORMS)
def test_first_use_self_check_passes(form, window, dtype):
    P._self_checks.clear()
    reason = P.self_check(form, dtype, dim=D, n_heads=H, n_kv=HKV, extent=E, window=window,
                          sink_dtype=dtype)
    assert reason is None, reason
