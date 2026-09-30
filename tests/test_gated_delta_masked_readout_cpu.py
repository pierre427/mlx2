"""The ops reference must agree with the Metal kernels at masked positions."""

import mlx.core as mx

from mlx2.runtime.models.gated_delta import _gated_delta_step_ops


def test_masked_step_leaves_state_and_emits_zero_readout():
    B, H, Dk, Dv = 2, 2, 4, 3
    q = mx.random.normal((B, H, Dk))
    k = mx.random.normal((B, H, Dk))
    v = mx.random.normal((B, H, Dv))
    g = mx.full((B, H), 0.9)
    beta = mx.full((B, H), 0.5)
    state = mx.random.normal((B, H, Dv, Dk))
    mask = mx.array([True, False])

    y, new_state = _gated_delta_step_ops(q, k, v, g, beta, state, mask)
    y_open, state_open = _gated_delta_step_ops(q, k, v, g, beta, state, None)
    mx.eval(y, new_state, y_open, state_open)

    # Row 0 is live: identical to the unmasked step.
    assert mx.array_equal(y[0], y_open[0])
    assert mx.array_equal(new_state[0], state_open[0])
    # Row 1 is padding: state untouched and, like the kernels, a zero readout.
    assert mx.array_equal(new_state[1], state[1])
    assert float(mx.abs(y[1]).max()) == 0.0
