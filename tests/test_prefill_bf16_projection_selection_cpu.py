"""BF16 materialized-MLP prefill refuses a swapped projection at selection.

Sweep 2026-10-12 L4-F4: e3c19168e routed ``qwen3_next._prefill_bf16_projection``
through ``quantized_dense_execution``, which admits only an exact
``nn.QuantizedLinear`` (the old code accepted any subclass).  The strictness
is kept; but when lane matmul or row-exact had class-swapped the trunk
projections, the ``staged_bf16`` / ``single_eval_bf16`` research modes only
found out mid-prefill.  The N20 factory now refuses the mode when it is
selected, before any allocation or forward.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import hybrid_packed_prefill_n as N
from mlx2.runtime.lane.installer import LaneQuantizedLinear
from mlx2.runtime.models import qwen3_next as QN


def _q():
    linear = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


def _candidate(swap):
    mlp = SimpleNamespace(gate_proj=_q(), up_proj=_q(), down_proj=_q())
    if swap:
        mlp.up_proj.__class__ = LaneQuantizedLinear
    return SimpleNamespace(trunk=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))


def test_plain_projection_is_admitted_and_runs():
    assert QN.prefill_bf16_projection_refusal(_q()) is None
    x = mx.ones((1, 4, 256), dtype=mx.bfloat16)
    assert QN._prefill_bf16_projection(_q(), x).shape == (1, 4, 128)


def test_swapped_projection_has_a_refusal_and_still_refuses_mid_forward():
    lane = _q()
    lane.__class__ = LaneQuantizedLinear
    assert "ordinary nn.QuantizedLinear" in QN.prefill_bf16_projection_refusal(lane)
    with pytest.raises(ValueError, match="ordinary nn.QuantizedLinear"):
        QN._prefill_bf16_projection(lane, mx.ones((1, 4, 256), dtype=mx.bfloat16))


@pytest.mark.parametrize("mode", ["staged_bf16", "single_eval_bf16"])
def test_n20_selection_refuses_bf16_mode_over_swapped_projection(mode):
    profile = {"gdn_eval_wave_max_segments": 1, "mlp_materialization_mode": mode}
    with pytest.raises(ValueError, match=r"BF16 MLP .*up_proj.*ordinary nn.QuantizedLinear"):
        N.model_memory_bound(_candidate(swap=True), (256,), (1,), profile, mx)
