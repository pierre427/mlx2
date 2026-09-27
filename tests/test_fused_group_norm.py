"""CPU tests for the fused GroupRMSNorm admission gates and eager reference.

No Metal and no GPU: admission is pure shape/dtype/lever logic, and the eager
reference is what the kernel is measured against, so both belong under CPU test.
The kernel's own numerics are covered by ``probe_fused_group_norm`` and
``scripts/bench_fused_group_norm.py``, which need the GPU queue.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_fused_group_norm as fgn

W = fgn.STREAM_WIDTH
G = fgn.GROUP_SIZE


@pytest.fixture
def enabled():
    previous = fgn.fused_group_norm_enabled()
    fgn.set_fused_group_norm_enabled(True)
    yield
    fgn.set_fused_group_norm_enabled(previous)


def inputs(rows=1024, tokens=1):
    x = mx.zeros((rows, tokens, W), dtype=mx.bfloat16)
    w = mx.ones((W,), dtype=mx.bfloat16)
    return x, w


def admit(x, w, *, eps=fgn.EPS, group_size=G, rows=(1024,)):
    return fgn.admit_fused_group_norm(
        x, w, eps=eps, group_size=group_size, candidate_rows=rows
    )


# tests/conftest.py pins the suite to mx.cpu, so the device gate can never pass
# here. A positive case therefore asserts that every *geometry* check cleared and
# the device is the only thing left objecting -- which is the part this file can
# actually test without the GPU queue.
GEOMETRY_OK = ("eligible", "Metal GPU unavailable")


def test_disabled_by_default():
    assert fgn.fused_group_norm_enabled() is False
    x, w = inputs()
    result = admit(x, w)
    assert result.accepted is False
    assert "not enabled" in result.reason


def test_no_row_count_is_qualified_out_of_the_box():
    """Nothing is selectable until a receipt exists."""
    assert fgn.QUALIFIED_ROW_COUNTS == ()


def test_admitted_only_for_locked_geometry(enabled):
    x, w = inputs()
    assert admit(x, w).reason in GEOMETRY_OK


@pytest.mark.parametrize(
    ("group_size", "eps", "fragment"),
    [
        (1024, fgn.EPS, "group_size"),
        (None, fgn.EPS, "group_size"),
        (G, 1e-05, "eps"),
        (G, 0.0, "eps"),
    ],
)
def test_rejects_unlocked_geometry(enabled, group_size, eps, fragment):
    x, w = inputs()
    result = admit(x, w, eps=eps, group_size=group_size)
    assert result.accepted is False
    assert fragment in result.reason


def test_rejects_wrong_stream_width(enabled):
    x = mx.zeros((1024, 1, W + 2560), dtype=mx.bfloat16)
    w = mx.ones((W + 2560,), dtype=mx.bfloat16)
    result = admit(x, w)
    assert result.accepted is False
    assert "stream width" in result.reason


def test_rejects_wrong_weight_shape(enabled):
    x = mx.zeros((1024, 1, W), dtype=mx.bfloat16)
    w = mx.ones((G,), dtype=mx.bfloat16)
    result = admit(x, w)
    assert result.accepted is False
    assert "weight must be exactly" in result.reason


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_rejects_non_bfloat16(enabled, dtype):
    x = mx.zeros((1024, 1, W), dtype=dtype)
    w = mx.ones((W,), dtype=mx.bfloat16)
    assert "bfloat16" in admit(x, w).reason


def test_rejects_unlisted_row_count(enabled):
    """Row counts outside the candidate list fail closed rather than run."""
    x, w = inputs(rows=7)
    result = admit(x, w)
    assert result.accepted is False
    assert "not qualified" in result.reason


def test_unknown_candidate_row_count_raises(enabled):
    """Asking for an unsupported candidate is a caller error, not a decline."""
    x, w = inputs(rows=7)
    with pytest.raises(ValueError, match="unsupported fused group-norm candidate"):
        admit(x, w, rows=(7,))


def test_row_count_is_the_product_of_leading_axes(enabled):
    """A [2, 512, W] input is 1024 rows and must be judged as such."""
    x = mx.zeros((2, 512, W), dtype=mx.bfloat16)
    w = mx.ones((W,), dtype=mx.bfloat16)
    assert admit(x, w).reason in GEOMETRY_OK
    y = mx.zeros((2, 511, W), dtype=mx.bfloat16)
    result = admit(y, w)
    assert result.accepted is False
    assert "not qualified" in result.reason


def test_calling_without_the_lever_raises(enabled):
    """The public entry point raises; it never silently falls back."""
    fgn.set_fused_group_norm_enabled(False)
    x, w = inputs()
    with pytest.raises(ValueError, match="not eligible"):
        fgn.fused_group_norm(x, w, eps=fgn.EPS, group_size=G,
                             candidate_rows=(1024,))


def test_eager_reference_matches_the_production_arithmetic():
    """The reference the kernel is graded against must be the real math.

    Mirrors GroupRMSNorm.__call__ for group_size=2560 / eps=1e-6 with the
    fast-path gate declined, which is what production does at prefill widths.
    """
    mx.random.seed(3)
    x = (mx.random.normal((1, 64, W)) * 1.7).astype(mx.bfloat16)
    w = (1.0 + 0.05 * mx.random.normal((W,))).astype(mx.bfloat16)

    dtype = x.dtype
    xf = x.astype(mx.float32).reshape(*x.shape[:-1], W // G, G)
    expected = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + fgn.EPS)
    expected = (expected.reshape(*x.shape) * w.astype(mx.float32)).astype(dtype)

    got = fgn.eager_group_norm(x, w)
    mx.eval(expected, got)
    assert got.dtype == mx.bfloat16
    assert mx.array_equal(expected, got).item()


def test_eager_reference_normalizes_each_group_independently():
    """Groups of 2560 are normalized separately, not the whole 10240 row."""
    x = mx.zeros((1, 1, W), dtype=mx.bfloat16)
    # First group constant 1.0, second group constant 2.0, rest zero-ish.
    flat = [1.0] * G + [2.0] * G + [1.0] * (W - 2 * G)
    x = mx.array(flat, dtype=mx.bfloat16).reshape(1, 1, W)
    w = mx.ones((W,), dtype=mx.bfloat16)
    out = fgn.eager_group_norm(x, w).astype(mx.float32)
    mx.eval(out)
    # RMS of a constant group is that constant, so x/rms == 1 for both groups.
    assert float(mx.max(mx.abs(out[0, 0, :G] - 1.0)).item()) < 2e-2
    assert float(mx.max(mx.abs(out[0, 0, G:2 * G] - 1.0)).item()) < 2e-2
