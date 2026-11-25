"""``inkling2_decode_handover`` (PR sweep 2026-11-25, Rapid-MLX#5953 idea), CPU only.

A one-row decode forward submits the residual stream after every visit. Same
graph and kernels, so outputs are bit-identical; verify rows and prompt
chunks keep their own submission rules. Default off; refused with
``inkling2_native_replay``.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from test_inkling2_prefill_mtplx_cpu import _cpu, _run, _tiny_model  # noqa: F401


def test_defaults_off_and_policy_validation():
    from mlx2.adapters.inkling2 import model_kernel_options, validate_policy

    kernels, rest = validate_policy(None)
    assert kernels["decode_handover"] is False and rest == {}
    assert "decode_handover" not in model_kernel_options(kernels)
    kernels, _ = validate_policy({"inkling2_decode_handover": True})
    assert kernels["decode_handover"] is True
    with pytest.raises(TypeError):
        validate_policy({"inkling2_decode_handover": 1})
    with pytest.raises(ValueError, match="native_replay"):
        validate_policy({"inkling2_decode_handover": True, "inkling2_native_replay": True})


def test_decode_handover_is_bit_identical_and_counted():
    from mlx2.runtime.models.inkling2 import prefill_route

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 80), key=mx.random.key(3))
    # A prompt chunk, a 4-row verify-shaped chunk, then three one-row steps.
    plan = [(0, 72), (72, 76), (76, 77), (77, 78), (78, 79)]
    base = _run(model, tokens, plan)
    assert model.kernel_receipt()["decode_handover_calls"] == 0
    prefill_route.configure_decode(model, decode_handover=True)
    got = _run(model, tokens, plan)
    assert mx.array_equal(got, base)
    visits = len(model.model.layer_indices)
    receipt = model.kernel_receipt()
    # Every visit but the last of each one-row forward; nothing else.
    assert receipt["decode_handover_calls"] == 3 * (visits - 1)
    assert receipt["prefill_handover_calls"] == 0
    prefill_route.configure_decode(model, decode_handover=False)
    again = _run(model, tokens, plan)
    assert mx.array_equal(again, base)
    assert model.kernel_receipt()["decode_handover_calls"] == 3 * (visits - 1)


def test_batched_one_row_steps_hand_over():
    from mlx2.runtime.models.inkling2 import prefill_route

    model = _tiny_model(seed=1)
    tokens = mx.random.randint(0, 500, (2, 10), key=mx.random.key(4))
    plan = [(0, 8), (8, 9), (9, 10)]
    base = _run(model, tokens, plan, pads=[0, 0])
    prefill_route.configure_decode(model, decode_handover=True)
    got = _run(model, tokens, plan, pads=[0, 0])
    assert mx.array_equal(got, base)
    assert model.kernel_receipt()["decode_handover_calls"] == 2 * (len(model.model.layer_indices) - 1)
