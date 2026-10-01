"""Per-layer eager dispatch on the shared Qwen3.8/3.6 trunk (CPU, tiny model).

The lever only inserts ``mx.async_eval`` between layers, so logits and cache
state must be bit-identical to the stock forward, it must engage only within
its row bound, and it is selectable only through the adapter policy keys.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.qwen38_27b import eager_dispatch_policy
from mlx2.runtime import round_levers
from route_harness import tiny_qwen38_mtp


def _run(model, stride, tokens):
    model.model.set_eager_dispatch(stride, max_rows=4)
    cache = model.make_cache()
    outputs = [model(tokens, cache=cache)]
    for _ in range(3):
        outputs.append(model(mx.argmax(outputs[-1][:, -1:], axis=-1), cache=cache))
    mx.eval(outputs, [c.state for c in cache])
    return outputs, [v for _, v in tree_flatten([c.state for c in cache])]


def test_eager_dispatch_is_bit_identical_and_counted():
    model, vocab = tiny_qwen38_mtp()
    tokens = mx.array([[3, 17, 5, 9, 11, 2]])  # 6 rows > max_rows: declined
    round_levers.reset_counters()
    stock, stock_state = _run(model, 0, tokens)
    assert round_levers.counters()["eager_dispatch_forwards"] == 0
    eager, eager_state = _run(model, 2, tokens)
    levers = round_levers.counters()
    assert levers["eager_dispatch_row_declines"] == 1
    assert levers["eager_dispatch_forwards"] == 3
    # Four layers at stride 2: layers 1 and 3 (3 is also the last).
    assert levers["eager_async_evals"] == 6
    for a, b in zip(stock, eager):
        assert mx.array_equal(a, b).item()
    assert len(stock_state) == len(eager_state) > 0
    for a, b in zip(stock_state, eager_state):
        assert mx.array_equal(a, b).item()


def test_eager_dispatch_policy_defaults_off_and_validates():
    assert eager_dispatch_policy({}) == (0, 64)
    assert eager_dispatch_policy({"eager_dispatch_stride": 4}) == (4, 64)
    for bad in (
        {"eager_dispatch_stride": -1},
        {"eager_dispatch_stride": True},
        {"eager_dispatch_stride": 2.0},
        {"eager_dispatch_max_rows": 0},
    ):
        with pytest.raises(ValueError):
            eager_dispatch_policy(bad)
    model, _ = tiny_qwen38_mtp()
    assert model.model.eager_dispatch_stride == 0
    with pytest.raises(ValueError):
        model.model.set_eager_dispatch(-2)


def test_selected_stride_is_route_identity_and_needs_observed_use():
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import (
        Qwen3827BAdapter, eager_dispatch_environment, eager_dispatch_policy as policy,
    )
    from mlx2.qualification import required_feature_checks

    # 27B opt-in (neutral end to end); 35B default stride 2 (Pierre, 2026-10-01).
    assert policy({}, Qwen3827BAdapter.default_eager_dispatch_stride) == (0, 64)
    assert policy({}, Qwen3635BA3BAdapter.default_eager_dispatch_stride) == (2, 64)
    assert policy({"eager_dispatch_stride": 0}, 2) == (0, 64)
    on = eager_dispatch_environment({"A": "1"}, (2, 64))
    off = eager_dispatch_environment(on, (0, 64))
    assert on == {"A": "1", "MLX2_EAGER_DISPATCH_STRIDE": "2",
                  "MLX2_EAGER_DISPATCH_MAX_ROWS": "64"}
    assert off == {"A": "1"}
    settings = {"mtp": False, "max_context": 4096, "execution_policy": {}}
    assert "feature_eager_dispatch" in required_feature_checks({**settings, "environment": on})
    assert "feature_eager_dispatch" not in required_feature_checks({**settings, "environment": off})
