"""The packed mHC kernel params follow the parameter arrays, not their id()."""

import mlx.core as mx
import mlx.nn as nn

from mlx2.runtime.models.xing4_0 import HyperConnection


def _connection():
    hc = HyperConnection.__new__(HyperConnection)
    nn.Module.__init__(hc)
    hc.hc_mult = 2
    hc.iters = 1
    hc.eps = 1e-6
    hc.norm_eps = 1e-6
    hc.clamp_min = -1.0
    hc.clamp_max = 1.0
    hc.hc_base = mx.zeros((8,))
    hc.hc_scale = mx.ones((3,))
    return hc


def test_replacing_parameters_repacks_and_the_cache_pins_them():
    hc = _connection()
    first = hc._kernel_params()
    assert hc._kernel_params() is first
    hc.hc_scale = mx.full((3,), 2.0)
    second = hc._kernel_params()
    assert second is not first
    assert second[4:7].tolist() == [2.0, 2.0, 2.0]
    # The arrays themselves are held, so a freed array's address can never
    # masquerade as the live one.
    assert hc._packed[0] is hc.hc_scale and hc._packed[1] is hc.hc_base
