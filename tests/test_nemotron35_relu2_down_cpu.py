import mlx.core as mx

from mlx2.runtime.models.nemotron_relu2_down import admit
from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear


def layer(experts=1):
    result = QuantizedSwitchLinear(2688, 1024, experts, bias=False)
    result.scales = result.scales.astype(mx.bfloat16)
    result.biases = result.biases.astype(mx.bfloat16)
    return result


def test_admission_is_geometry_locked_without_evaluation():
    x = mx.zeros((1, 1, 1024), dtype=mx.bfloat16)
    indices = mx.zeros((1, 1, 22), dtype=mx.uint32)
    scores = mx.ones((1, 1, 22), dtype=mx.float32)
    assert admit(x, indices, scores, layer(), training=False).accepted


def test_admission_refuses_dtype_width_and_training():
    indices = mx.zeros((1, 1, 22), dtype=mx.uint32)
    scores = mx.ones((1, 1, 22), dtype=mx.float32)
    down = layer()
    assert not admit(mx.zeros((1, 1, 1024)), indices, scores, down, training=False).accepted
    assert not admit(mx.zeros((1, 2, 1024), dtype=mx.bfloat16), indices, scores, down, training=False).accepted
    assert not admit(mx.zeros((1, 1, 1024), dtype=mx.bfloat16), indices, scores, down, training=True).accepted
