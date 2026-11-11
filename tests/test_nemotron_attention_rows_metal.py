"""Metal: the exact verify's strip attention recipe equals one-token decode.

``MLX2_NEMOTRON_EXACT_ATTENTION_ROWS=on`` runs an attention block's norm and
q/k/v/o projections on the whole verify strip and keeps only the cache
append and SDPA per row.  Every verify row's logits, the committed KV rows
and the next decode step must still equal one-token decode bit for bit, and
equal the per-row block recipe (``off``).
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Nemotron exact verify kernels are Metal-only"
)

from mlx2.runtime.models.cache import ArraysCache  # noqa: E402

from test_nemotron_tree_kernels_metal import _metal_model, _prefill  # noqa: E402


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint32 if value.dtype == mx.float32 else mx.uint16))


def _verify(model, cache, tokens, accept):
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    live = copy.deepcopy(cache)
    transaction = HybridVerifyRows([live]).begin([len(tokens)])
    logits, _taps = model.forward_with_taps(mx.array([tokens]), transaction.caches, (0, 3))
    mx.eval(logits)
    transaction.commit([accept])
    return logits, live


def _decode(model, cache, tokens):
    live = copy.deepcopy(cache)
    rows = [model(mx.array([[token]]), live) for token in tokens]
    return mx.concatenate(rows, axis=1), live


@pytest.mark.parametrize("prefix", [37, 300])
@pytest.mark.parametrize("rows", [2, 6, 9])
def test_strip_attention_matches_one_token_decode(monkeypatch, prefix, rows):
    model = _metal_model()
    cache = _prefill(model, prefix)
    tokens = [(11 * index + 5) % 64 for index in range(rows)]
    expected, oracle = _decode(model, cache, tokens)
    monkeypatch.setenv("MLX2_NEMOTRON_EXACT_ATTENTION_ROWS", "off")
    baseline, _ = _verify(model, cache, tokens, rows)
    monkeypatch.setenv("MLX2_NEMOTRON_EXACT_ATTENTION_ROWS", "on")
    logits, live = _verify(model, cache, tokens, rows)
    np.testing.assert_array_equal(_bits(baseline), _bits(expected))
    np.testing.assert_array_equal(_bits(logits), _bits(expected))
    for left, right in zip(live, oracle, strict=True):
        if isinstance(left, ArraysCache):
            for a, e in zip(left.cache, right.cache, strict=True):
                np.testing.assert_array_equal(_bits(a), _bits(e))
        else:
            assert left.offset == right.offset
            for a, e in zip(left.state, right.state, strict=True):
                np.testing.assert_array_equal(
                    _bits(a[..., : left.offset, :]), _bits(e[..., : right.offset, :])
                )
    step = mx.array([[9]])
    np.testing.assert_array_equal(_bits(model(step, live)), _bits(model(step, oracle)))
    stats = model.row_exact_status()
    assert "attention_exact_rows" in str(stats), stats


def test_switch_rejects_unknown_values(monkeypatch):
    from mlx2.runtime.models import nemotron_h

    monkeypatch.setenv("MLX2_NEMOTRON_EXACT_ATTENTION_ROWS", "maybe")
    with pytest.raises(ValueError):
        nemotron_h._exact_attention_rows()


def test_adapter_default_engages_without_the_switch(monkeypatch):
    from mlx2.runtime.models import nemotron_h

    monkeypatch.delenv("MLX2_NEMOTRON_EXACT_ATTENTION_ROWS", raising=False)
    model = _metal_model()
    assert not nemotron_h._exact_attention_rows(model)
    model.__dict__["_mlx2_exact_attention_rows"] = True
    assert nemotron_h._exact_attention_rows(model)
    attention = next(layer for layer in model.layers if layer.block_type == "*")
    assert model._row_exact_recipe(attention).name == "attention_exact_rows"
    monkeypatch.setenv("MLX2_NEMOTRON_EXACT_ATTENTION_ROWS", "off")
    assert model._row_exact_recipe(attention).name == "per_row"
    cache = _prefill(model, 37)
    tokens = [3, 9, 27, 17, 51, 25]
    monkeypatch.delenv("MLX2_NEMOTRON_EXACT_ATTENTION_ROWS")
    expected, _ = _decode(model, cache, tokens)
    logits, _ = _verify(model, cache, tokens, len(tokens))
    np.testing.assert_array_equal(_bits(logits), _bits(expected))
