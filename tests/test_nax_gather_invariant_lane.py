"""The NAX gather stands aside inside the slice-invariant prefill lane.

Both ports hook the sorted expert gather; the lane pins one kernel
configuration at every row count, and the NAX admission depends on rows, so
the two are kept apart until a GPU run proves the combination invariant.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models import invariant_prefill as inv
from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models import qwen3_next as qn
from mlx2.runtime.models import switch_layers as sl


@pytest.fixture(autouse=True)
def _reset():
    old = nax.MODE
    yield
    nax.set_mode(old)


def _qlinear(E=8, out=128, inp=128):
    return sl.SwitchLinear(inp, out, E, bias=False).to_quantized(group_size=64, bits=4)


def _sorted_call(lin, T=64, k=2, E=8, D=128):
    mx.random.seed(0)
    x = mx.random.normal((T, D)).astype(mx.bfloat16)
    inds = mx.random.randint(0, E, (T, k)).astype(mx.uint32)
    flat = inds.flatten()
    order = mx.argsort(flat)
    xs = mx.expand_dims(x, -2)[order // k]
    return lin(xs, flat[order], sorted_indices=True)


@pytest.mark.parametrize("lane_on,expect_calls", [(False, 1), (True, 0)])
def test_nax_gather_is_not_tried_inside_the_invariant_lane(monkeypatch, lane_on, expect_calls):
    nax.set_mode("gather")
    calls = []
    monkeypatch.setattr(nax, "try_gather", lambda *a, **k: calls.append(1) or None)
    lin = _qlinear()
    with inv.scope(lane_on):
        mx.eval(_sorted_call(lin))
    assert len(calls) == expect_calls


def test_fused_nax_candidates_decline_inside_the_invariant_lane():
    nax.set_mode("fused")
    E, D, H = 8, 128, 128
    mlp = qn.SwitchGLU(D, H, E) if hasattr(qn, "SwitchGLU") else sl.SwitchGLU(D, H, E)
    mlp.gate_proj = _qlinear(E, H, D)
    mlp.up_proj = _qlinear(E, H, D)
    mlp.eval()
    assert qn._nax_split_swiglu_candidate(mlp) is True
    with inv.scope(True):
        assert qn._nax_split_swiglu_candidate(mlp) is False
