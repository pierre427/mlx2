"""Focused correctness and engagement tests for Qwen4 eager dispatch."""

import mlx.core as mx
import pytest
from test_batched_mtp import _tiny_qwen4_model

from mlx2.runtime import round_levers
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models.cache import make_prompt_cache


@pytest.fixture(autouse=True)
def restore_eager_dispatch_runtime():
    original = (
        qwen4_exp._EAGER_DISPATCH,
        qwen4_exp._EAGER_DISPATCH_MAX_ROWS,
        qwen4_exp._EAGER_DISPATCH_STRIDE,
    )
    round_levers.reset_counters()
    yield
    (
        qwen4_exp._EAGER_DISPATCH,
        qwen4_exp._EAGER_DISPATCH_MAX_ROWS,
        qwen4_exp._EAGER_DISPATCH_STRIDE,
    ) = original
    round_levers.reset_counters()


def _forward(model, prompt):
    text = model.language_model.model
    output = text(prompt[None, :], make_prompt_cache(model))
    mx.eval(output)
    return output


def test_eager_dispatch_is_bit_identical_and_stride_bounded():
    mx.random.seed(20260920)
    model = _tiny_qwen4_model()
    prompt = mx.array([1, 2, 3], mx.uint32)
    qwen4_exp._EAGER_DISPATCH = False
    baseline = _forward(model, prompt)
    layer_count = len(model.language_model.model.layers)

    qwen4_exp._EAGER_DISPATCH = True
    qwen4_exp._EAGER_DISPATCH_MAX_ROWS = 64
    for stride, expected in ((1, layer_count), (2, (layer_count + 1) // 2)):
        qwen4_exp._EAGER_DISPATCH_STRIDE = stride
        round_levers.reset_counters()
        eager = _forward(model, prompt)
        assert mx.array_equal(baseline, eager)
        status = qwen4_exp.qwen4_eager_dispatch_status()
        assert status["forwards"] == 1
        assert status["row_declines"] == 0
        assert status["async_evals"] == expected
        assert status["stride"] == stride


def test_eager_dispatch_declines_oversized_forward_without_changing_output():
    mx.random.seed(20260920)
    model = _tiny_qwen4_model()
    prompt = mx.array([1, 2, 3], mx.uint32)

    qwen4_exp._EAGER_DISPATCH = False
    baseline = _forward(model, prompt)
    qwen4_exp._EAGER_DISPATCH = True
    qwen4_exp._EAGER_DISPATCH_MAX_ROWS = 2
    round_levers.reset_counters()
    declined = _forward(model, prompt)

    assert mx.array_equal(baseline, declined)
    status = qwen4_exp.qwen4_eager_dispatch_status()
    assert status["forwards"] == 0
    assert status["row_declines"] == 1
    assert status["async_evals"] == 0
