"""The TensorFold prefill installer leaves HC-owned projections alone.

Sweep 2026-10-12 L4-F3: 086421f7b stopped ``flash_tensorfold_qmv.install``
from class-swapping the projections HC's fused decode kernels own
(``input_mix_weight_down/up``, ``block_inject_weight``), because the swap
makes HC admission decline every call.  ``tensorfold_prefill.install``
(``backend="metal"``) still swapped them to ``TensorFoldPrefillLinear``.
CPU only: installation is a class swap; nothing is dispatched.
"""

import mlx.core as mx
from mlx import nn

from mlx2.runtime.models import flash_tensorfold_qmv as QMV
from mlx2.runtime.models import qwen4_hc_decode as HC
from mlx2.runtime.models import tensorfold_prefill as TP


def _q(n, k):
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


class _HCOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.hc_count = 4
        self.input_mix_weight_down = _q(64, 2048)
        self.input_mix_weight_up = _q(64, 2048)
        self.block_inject_weight = _q(4, 2048)


def _model():
    model = nn.Module()
    model.layer = _HCOwner()
    model.o_proj = _q(2048, 2048)
    return model


def _hc_projections(model):
    return (
        model.layer.input_mix_weight_down,
        model.layer.input_mix_weight_up,
        model.layer.block_inject_weight,
    )


def test_metal_prefill_install_keeps_hc_projections_plain():
    model = _model()
    TP.install(model, backend="metal")

    for projection in _hc_projections(model):
        assert type(projection) is nn.QuantizedLinear
        assert HC._quantized_reason(projection) is None
    # Ordinary projections are still installed.
    assert type(model.o_proj) is TP.TensorFoldPrefillLinear


def test_prefill_install_after_qmv_install_keeps_hc_projections_plain():
    model = _model()
    QMV.install(model)
    TP.install(model, backend="metal")

    for projection in _hc_projections(model):
        assert type(projection) is nn.QuantizedLinear
        assert HC._quantized_reason(projection) is None
