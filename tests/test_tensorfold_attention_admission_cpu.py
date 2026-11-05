"""Host admission regressions; no checkpoint or Metal numerical qualification."""

import mlx.core as mx
import pytest

from mlx2.runtime.tensorfold_qwen38 import stream_attention as attention


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _fixture():
    plan = attention.Plan([[-1, 0], [-1, 0, 0]], [4, 7], 4, 2)
    queries = mx.zeros((1, 4, 5, 256), dtype=mx.bfloat16)
    buffer = mx.zeros((1, 2, 12, 256), dtype=mx.bfloat16)
    return plan, queries, (buffer, buffer)


@pytest.mark.parametrize("entry", [attention.position_tree_sdpa, attention.tree_sdpa])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_live_streams_cannot_be_replaced_with_spare_or_truncated_buffers(monkeypatch, entry, count):
    plan, queries, pair = _fixture()
    monkeypatch.setattr(attention, "_kernel", lambda name: pytest.fail("invalid lanes reached Metal dispatch"))
    with pytest.raises(ValueError, match="streams"):
        entry(queries, [pair] * count, 0.0625, plan)


@pytest.mark.parametrize("fault", ["query_dtype", "query_batch", "key_dtype", "key_capacity", "value_heads"])
def test_raw_dispatcher_cannot_bypass_position_law_layout_checks(monkeypatch, fault):
    plan, queries, pair = _fixture()
    keys, values = pair
    if fault == "query_dtype":
        queries = queries.astype(mx.float32)
    elif fault == "query_batch":
        queries = mx.zeros((2, 4, 5, 256), dtype=mx.bfloat16)
    elif fault == "key_dtype":
        keys = keys.astype(mx.float16)
    elif fault == "key_capacity":
        keys = keys[:, :, :8]
    else:
        values = mx.zeros((1, 1, 12, 256), dtype=mx.bfloat16)
    monkeypatch.setattr(attention, "_kernel", lambda name: pytest.fail("invalid layout reached Metal dispatch"))
    with pytest.raises(ValueError):
        attention.tree_sdpa(queries, [pair, (keys, values)], 0.0625, plan)


@pytest.mark.parametrize("heads,kv_heads", [(0, 1), (4, 0), (3, 2), (True, 1)])
def test_plan_refuses_unrepresentable_head_grouping(heads, kv_heads):
    with pytest.raises(ValueError, match="heads"):
        attention.Plan([[-1]], [0], heads, kv_heads)


@pytest.mark.parametrize("parents,starts", [([[]], [0]), ([[-1]], [-1]), ([[-1]], [1.5]),
                                           ([[-2]], [0]), ([[-1, 0.5]], [0]),
                                           ([list(range(-1, attention.MAX_DEPTH))], [0])])
def test_plan_refuses_geometry_that_cannot_fit_its_metadata(parents, starts):
    with pytest.raises(ValueError):
        attention.Plan(parents, starts, 4, 2)


def test_position_kernel_refuses_more_than_sixteen_heads_per_group(monkeypatch):
    plan = attention.Plan([[-1]], [0], 32, 1)
    queries = mx.zeros((1, 32, 1, 256), dtype=mx.bfloat16)
    buffer = mx.zeros((1, 1, 1, 256), dtype=mx.bfloat16)
    monkeypatch.setattr(attention, "_kernel", lambda name: pytest.fail("unsupported group reached Metal dispatch"))
    with pytest.raises(ValueError):
        attention.position_tree_sdpa(queries, [(buffer, buffer)], 0.0625, plan)


def test_valid_ragged_streams_keep_the_kernel_shape_and_binding_contract(monkeypatch):
    plan, queries, pair = _fixture()
    calls = []

    def kernel(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            return [mx.zeros(shape, dtype=dtype) for shape, dtype in
                    zip(kwargs["output_shapes"], kwargs["output_dtypes"], strict=True)]
        return run

    monkeypatch.setattr(attention, "_kernel", kernel)
    result = attention.position_tree_sdpa(queries, [pair, pair], 0.0625, plan)
    assert result.shape == queries.shape and result.dtype == queries.dtype
    assert [name for name, _ in calls] == ["tail", "merge"]
    tail = calls[0][1]
    assert tail["inputs"][1].shape == pair[0].shape
    assert tail["inputs"][2].shape == pair[0].shape
    assert tail["inputs"][3].shape == (16,)  # only an unused slot gets a spare
