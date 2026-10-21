"""Tree ownership/PLE/QSA/commit CPU oracles; not Metal numerical evidence."""
import copy

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

from mlx2.adapters.flash_next_tree import FlashNextTargetTreeExecutor
from mlx2.runtime.models.qwen4_tree_gdn import ancestry


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def compare(actual, expected):
    left, right = tree_flatten(actual), tree_flatten(expected)
    assert [name for name, _ in left] == [name for name, _ in right]
    for (_, a), (_, b) in zip(left, right):
        if isinstance(a, mx.array):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-5, atol=3e-5)
        else:
            assert a == b


def states(cache):
    return [(item.state, item.meta_state) for item in cache]


def fixture():
    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 3, 7, 8, 2, 9, 4]]), cache=cache))
    return model, cache


@pytest.mark.parametrize("pattern", [(5, 5, 5), (3, 3, 3, 3, 3), (4, 4, 4, 3)])
def test_full_tree_equals_each_serial_lineage_and_commit(pattern):
    model, base = fixture()
    tokens, parents = [13], [-1]
    for branch, depth in enumerate(pattern):
        parent = 0
        for j in range(depth):
            parents.append(parent)
            parent = len(tokens)
            tokens.append(15 + branch * 7 + j)
    original = copy.deepcopy(states(base))
    executor = FlashNextTargetTreeExecutor(model, cpu_reference=True)
    logits, features, transaction = executor.forward(model, tokens, parents, base, (1,))
    mx.eval(logits, features)
    for path in ancestry(parents):
        cache = copy.deepcopy(base)
        for row in path:
            expected, hidden = model.forward_with_taps(mx.array([[tokens[row]]]), cache, (1,))
        np.testing.assert_allclose(np.asarray(logits[:, path[-1]:path[-1] + 1]),
                                   np.asarray(expected), rtol=2e-5, atol=3e-5)
        np.testing.assert_allclose(np.asarray(features[:, path[-1]:path[-1] + 1]),
                                   np.asarray(hidden), rtol=2e-5, atol=3e-5)
        compare(states(transaction.prepared_cache(path)), states(cache))
    compare(states(base), original)
    compare(states(transaction.prepared_cache([])), original)
    chosen = ancestry(parents)[-1]
    published = transaction.commit_paths([chosen])[0]
    assert transaction.closed
    compare(states(base), original)
    with pytest.raises(ValueError, match="closed"):
        transaction.prepared_cache(chosen)
    # The committed successor continues using the ordinary reference path.
    control = copy.deepcopy(base)
    for row in chosen:
        model(mx.array([[tokens[row]]]), cache=control)
    np.testing.assert_allclose(np.asarray(model(mx.array([[5]]), cache=published)),
                               np.asarray(model(mx.array([[5]]), cache=control)), rtol=2e-5, atol=3e-5)


def test_abort_and_invalid_publication_do_not_touch_base():
    model, base = fixture()
    original = copy.deepcopy(states(base))
    executor = FlashNextTargetTreeExecutor(model, cpu_reference=True)
    _, _, transaction = executor.forward(model, [12, 14, 19], [-1, 0, 0], base, (1,))
    with pytest.raises(ValueError, match="root-connected"):
        transaction.commit_paths([[0, 1, 2]])
    with pytest.raises(ValueError, match="root-connected"):
        transaction.commit_paths([[0, True]])
    assert executor.dispatch_metrics(1, transaction, cached=True).width == 1
    assert not transaction.closed
    transaction.abort()
    with pytest.raises(ValueError, match="matching live"):
        executor.dispatch_metrics(1, transaction, cached=True)
    compare(states(base), original)


@pytest.mark.parametrize("parents", [[], [0], [-1, -1], [-1, 2], [-1, True], [-1] + [0] * 16])
def test_invalid_topology_declines(parents):
    with pytest.raises(ValueError):
        ancestry(parents)


def test_physical_executor_refuses_cpu_and_rev():
    model, base = fixture()
    executor = FlashNextTargetTreeExecutor(model)
    with pytest.raises(ValueError, match="Metal"):
        executor.forward(model, [12, 14], [-1, 0], base, (1,))
    model._progressive_row_exact = True
    with pytest.raises(ValueError, match="REV"):
        executor.forward(model, [12, 14], [-1, 0], base, (1,))
