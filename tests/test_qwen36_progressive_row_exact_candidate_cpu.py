"""CPU evidence for the default-off Qwen3.6 row-exact candidate.

These tests cover reusable arithmetic and the Qwen3Next attention component;
they deliberately do not qualify or enable progressive verification.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from test_qwen_progressive_external_portability_cpu import tiny_qwen36

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models.base import create_causal_mask
from mlx2.runtime.models.qwen36_progressive_row_exact import (
    CANDIDATE_PROTOCOL,
    install,
)
from mlx2.runtime.models.row_exact_components import (
    RowExactLinearMixin,
    row_exact_subclass,
)
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    execute_progressive_target_rows,
)
from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("width", [2, 4, 8])
def test_shared_dense_projection_component_matches_serial_b1s1(width):
    mx.random.seed(17)
    layer = nn.Linear(13, 19, bias=True)
    values = mx.random.normal((width, 1, 13))
    expected = mx.concatenate(
        [layer(values[index : index + 1]) for index in range(width)], axis=0
    )
    original = type(layer)
    layer.__class__ = row_exact_subclass(RowExactLinearMixin, original)
    record = REV.Window(width)
    with REV.window(record):
        actual = layer(values)
    mx.eval(expected, actual)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert record.exact
    # The shared package's broadcast matvec where its canary holds (the
    # module's own one-row call per row where it does not).
    assert record.stages["projections"] in ({"matvec_rows": 1}, {"per_row_canary": 1}, {"per_row_biased": 1})


def _ordinary_mask(cache):
    return create_causal_mask(
        1,
        offset=cache.offset,
        left_padding=mx.array([0]),
    )


@pytest.mark.parametrize("width", [2, 4, 8])
def test_qwen3next_attention_candidate_matches_serial_b1s1(width):
    model = tiny_qwen36(seed=31)
    candidate = install(model)
    attention = model.layers[3].self_attn
    hidden = mx.random.normal((width, 3, 32))

    bases = []
    for lane in range(width):
        cache = model.make_cache()[3]
        prefix = mx.random.normal((1, 1, 32))
        mx.eval(attention(prefix, _ordinary_mask(cache), cache))
        bases.append(cache)

    serial_outputs = []
    for lane in range(width):
        cache = copy.deepcopy(bases[lane])
        pieces = []
        for position in range(3):
            pieces.append(
                attention(
                    hidden[lane : lane + 1, position : position + 1],
                    _ordinary_mask(cache),
                    cache,
                )
            )
        serial_outputs.append(mx.concatenate(pieces, axis=1))
    expected = mx.concatenate(serial_outputs, axis=0)

    rows = [copy.deepcopy(cache) for cache in bases]
    segmented = SegmentedBatchKVCache(rows)
    segmented.prepare(lengths=[3] * width, right_padding=[0] * width)
    record = REV.Window(width * 3)
    with REV.window(record):
        actual = attention(hidden, segmented.make_mask(3), segmented)
    mx.eval(expected, actual)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert record.exact
    assert record.stages["attention"] == {"serial_b1s1": width * 3}
    assert [row.offset for row in rows] == [base.offset + 3 for base in bases]
    candidate.remove()


def test_qwen3next_attention_candidate_preserves_ragged_lineages():
    model = tiny_qwen36(seed=37)
    candidate = install(model)
    attention = model.layers[3].self_attn
    lengths = [3, 1, 2, 3]
    hidden = mx.random.normal((len(lengths), 3, 32))
    bases = []
    for _ in lengths:
        cache = model.make_cache()[3]
        prefix = mx.random.normal((1, 1, 32))
        mx.eval(attention(prefix, _ordinary_mask(cache), cache))
        bases.append(cache)

    expected = []
    for lane, valid in enumerate(lengths):
        cache = copy.deepcopy(bases[lane])
        pieces = []
        for position in range(valid):
            pieces.append(
                attention(
                    hidden[lane : lane + 1, position : position + 1],
                    _ordinary_mask(cache),
                    cache,
                )
            )
        expected.append(mx.concatenate(pieces, axis=1))

    rows = [copy.deepcopy(cache) for cache in bases]
    segmented = SegmentedBatchKVCache(rows)
    segmented.prepare(lengths=lengths, right_padding=[0, 2, 1, 0])
    record = REV.Window(sum(lengths))
    with REV.window(record):
        actual = attention(hidden, segmented.make_mask(3), segmented)
    mx.eval(actual, *expected)

    for lane, valid in enumerate(lengths):
        np.testing.assert_array_equal(
            np.asarray(actual[lane : lane + 1, :valid]),
            np.asarray(expected[lane]),
        )
    assert record.exact
    assert [row.offset for row in rows] == [
        base.offset + valid for base, valid in zip(bases, lengths)
    ]
    candidate.remove()


def test_qwen36_candidate_executor_is_explicit_and_fail_closed():
    model = tiny_qwen36(seed=41)
    assert not hasattr(model, "progressive_external_target_rows")
    assert not hasattr(model, "progressive_external_verify_protocol")
    candidate = install(model)
    assert callable(model.progressive_external_target_rows)
    assert not hasattr(model, "progressive_external_verify_protocol")

    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
    result = model.progressive_external_target_rows(
        mx.array([[4, 5]]),
        cache,
        (1, 6),
    )
    mx.eval(result.logits, result.features)
    assert result.protocol == CANDIDATE_PROTOCOL
    assert result.row_exact is False
    assert result.detail["candidate"] is True
    assert result.detail["reason"] == "gdn_fused_verify_not_engaged"
    status = candidate.status()
    assert status["qualified"] is False
    assert status["selected"] is False
    assert status["observed_used"] is True
    assert status["counts"]["not_exact"] == 1

    fresh = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=fresh))
    with pytest.raises(RuntimeError, match="fell off its exact path"):
        execute_progressive_target_rows(
            model,
            mx.array([[4, 5]]),
            fresh,
            (1, 6),
            protocol=CANDIDATE_PROTOCOL,
            fidelity=PROGRESSIVE_FIDELITY_EXACT,
        )

    candidate.remove()
    assert not hasattr(model, "progressive_external_target_rows")
    assert not hasattr(model, "progressive_external_verify_protocol")
