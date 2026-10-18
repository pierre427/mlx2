"""Real-Metal gate: vendored TensorFold mlp_act/gdn_post equal the served eager ops.

The Qwen3.8 tree route runs every MLP activation and GDN norm gate through
TensorFold's ``lane_glue`` (separate projections) or ``lane_fuse`` (stacked
projections) kernels.  Both copied MLX's sigmoid with ``metal::exp``; MLX
0.32.4 serves compiled ``swiglu``/``nn.silu`` with ``metal::precise::exp``
and the bf16 sigmoids round apart at -6.84375, which real MLP gates reach.
Each case feeds every finite bf16 value as the gate (``mlp_act``) or z
(``gdn_post``) and compares the public entry point with the served op the
ordinary decode path runs, bit for bit.  On the fixed ``metal::exp`` kernels
the ``mlp_act`` cases fail (50 of 139,264 sweep elements on MLX 0.32.4).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime.models.activations import swiglu
from mlx2.runtime.models.qwen3_next import _precise_swiglu
from mlx2.runtime.models.served_exp import EXP_SPELLINGS, covering_rows
from mlx2.runtime.tensorfold_qwen38 import lane_fuse, lane_glue

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

NV, DV = 48, 128


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _differing(have, want):
    mx.eval(have, want)
    assert have.shape == want.shape and have.dtype == want.dtype
    width = mx.uint16 if have.dtype.size == 2 else mx.uint32
    same = (have.view(width) == want.view(width)) | (mx.isnan(have) & mx.isnan(want))
    return int((~same).sum().item())


def _finite_sweep(rows, width, seed):
    import numpy as np

    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    finite = np.isfinite((bits.astype(np.uint32) << 16).view(np.float32))
    grid = mx.array(bits[finite]).view(mx.bfloat16)
    if rows * width < int(grid.size):   # a smoke input, too small to cover the grid: a seeded sample
        return grid[mx.random.permutation(int(grid.size), key=mx.random.key(seed))][: rows * width].reshape(1, rows, width)
    return covering_rows(grid, rows, width, seed)[None]


def _mlp_inputs(rows=8, n=17408):
    gate = _finite_sweep(rows, n, 7700)
    up = mx.random.normal((1, rows, n), key=mx.random.key(7800)).astype(mx.bfloat16)
    if rows * n >= 1 << 16:
        assert bool(mx.any(gate == mx.array(-6.84375, dtype=mx.bfloat16)).item())
    return gate, up


def _post_inputs(rows=11):
    y = mx.random.normal((1, rows, NV, DV), key=mx.random.key(7400)).astype(mx.bfloat16)
    z = _finite_sweep(rows, NV * DV, 7500)
    w = (1 + 0.1 * mx.random.normal((DV,), key=mx.random.key(7600))).astype(mx.bfloat16)
    eps = 1e-6
    ref = _precise_swiglu(y, z.reshape(1, rows, NV, DV), mx.fast.rms_norm(y, w, eps))
    return y, z, w, eps, ref.reshape(1, rows, NV * DV)


def test_lane_glue_mlp_act_matches_compiled_swiglu_on_every_bf16_gate():
    gate, up = _mlp_inputs()
    assert _differing(lane_glue.mlp_act(gate, up), swiglu(gate, up)) == 0


def test_lane_fuse_mlp_act_matches_compiled_swiglu_on_every_bf16_gate():
    gate, up = _mlp_inputs()
    gu = mx.concatenate([gate, up], axis=-1)
    assert _differing(lane_fuse.mlp_act(gu), swiglu(gate, up)) == 0


def test_lane_glue_gdn_post_matches_served_norm_gate_on_every_bf16_z():
    y, z, w, eps, ref = _post_inputs()
    assert _differing(lane_glue.gdn_post(y, z, w, eps), ref) == 0


def test_lane_fuse_gdn_post_matches_served_norm_gate_on_every_bf16_z():
    y, z, w, eps, ref = _post_inputs()
    rows = int(y.shape[1])
    b = mx.random.normal((1, rows, NV), key=mx.random.key(7900)).astype(mx.bfloat16)
    zba = mx.concatenate([z, b, b], axis=-1)
    assert _differing(lane_fuse.gdn_post(y, zba, w, eps), ref) == 0


def test_kernels_compile_the_served_spelling_and_report_exact_parity():
    gate, up = _mlp_inputs(rows=1, n=4096)
    mx.eval(lane_glue.mlp_act(gate, up), lane_fuse.mlp_act(mx.concatenate([gate, up], axis=-1)))
    y, z, w, eps, _ = _post_inputs(rows=1)
    mx.eval(lane_glue.gdn_post(y, z, w, eps))
    stats = lane_glue.stats()
    for name in ("mlp_act", "gdn_post"):
        assert stats[name]["exp"] in EXP_SPELLINGS
        assert stats[name]["parity"] == "exact"
        assert lane_glue.refusal(name) is None
        assert lane_fuse.refusal(name) is None
    # The kernel compiles the spelling its probe selected, not a fixed one.
    assert lane_glue.served_spelling("mlp_act") in lane_glue.served_source("mlp_act")
    assert lane_fuse.consumer_stats()["mlp_act"]["parity"] == "exact"


def test_wrong_spelling_is_refused_and_falls_back_to_the_served_op(monkeypatch):
    """A kernel compiled with the other spelling fails the value-sweep canary;
    the entry point then computes the served eager op and counts the fallback."""
    gate_probe, dtype = lane_glue._gate("mlp_act")
    served = gate_probe.spelling(dtype)
    wrong = next(s for s in EXP_SPELLINGS if s != served)
    monkeypatch.setitem(gate_probe._spellings, str(dtype), wrong)
    saved = dict(lane_glue._PARITY)
    try:
        lane_glue._PARITY.pop(("lane_glue", "mlp_act", wrong), None)
        lane_glue._PARITY.pop(("lane_fuse", "mlp_act", wrong), None)
        assert "differs from the served eager op" in lane_glue.refusal("mlp_act")
        assert lane_glue.stats()["mlp_act"]["parity"] != "exact"
        gate, up = _mlp_inputs()
        before = lane_glue.stats()["mlp_act"]["fallback_calls"]
        assert _differing(lane_glue.mlp_act(gate, up), swiglu(gate, up)) == 0
        assert lane_glue.stats()["mlp_act"]["fallback_calls"] == before + 1
        gu = mx.concatenate([gate, up], axis=-1)
        assert _differing(lane_fuse.mlp_act(gu), swiglu(gate, up)) == 0
        assert lane_fuse.consumer_stats()["mlp_act"]["fallback_calls"] >= 1
    finally:
        lane_glue._PARITY.clear()
        lane_glue._PARITY.update(saved)
