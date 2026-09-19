"""Actual production classes with NumPy: no MLX import or GPU execution."""

import ast
from collections import Counter, deque
import copy
import gc
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import weakref

ROOT = Path(__file__).parents[1] / "src/mlx2/runtime"


def attention(q, k, v, *, scale, mask=None, sinks=None):
    assert sinks is None
    repeat = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, repeat, axis=1), np.repeat(v, repeat, axis=1)
    scores = q @ k.swapaxes(-1, -2) * scale
    if isinstance(mask, str):
        assert mask == "causal"
        mask = np.arange(k.shape[2])[None, :] <= (
            k.shape[2] - q.shape[2] + np.arange(q.shape[2])[:, None])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)) @ v


def load_cpu():
    mx = SimpleNamespace(**{name: getattr(np, name) for name in (
        "array", "zeros", "arange", "concatenate", "pad", "expand_dims", "roll", "int32")})
    mx.fast = SimpleNamespace(scaled_dot_product_attention=attention)
    namespace = dict(mx=mx, copy=copy, Integral=Integral, deque=deque,
                     _BaseCache=object, _state_checkpoint_max=lambda: 0)
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(n for n in ast.parse((ROOT / "models/base.py").read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == "create_causal_mask")
    nodes.extend(n for n in ast.parse((ROOT / "models/cache.py").read_text()).body
                 if (isinstance(n, ast.ClassDef) and n.name in ("KVCache", "RotatingKVCache"))
                 or (isinstance(n, ast.FunctionDef) and n.name == "create_attention_mask"))
    nodes.extend(n for n in ast.parse((ROOT / "segmented_rotating_kv.py").read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "production-segmented-ring-cpu", "exec"), namespace)
    return namespace


CLASSES = load_cpu()
KVCache, RotatingKVCache, SegmentedKVRows = [CLASSES[name] for name in
    ("KVCache", "RotatingKVCache", "SegmentedKVRows")]


def data(rng, batch, width):
    return (rng.normal(size=(batch, 2, width, 4)).astype(np.float32),
            rng.normal(size=(batch, 2, width, 3)).astype(np.float32))


def setup(lengths, seed=883):
    rng = np.random.default_rng(seed)
    rows, histories = [], []
    for length in lengths:
        row = [RotatingKVCache(max_size=4), KVCache()]
        history = []
        for cache in row:
            k, v = data(rng, 1, length)
            if length:
                # Exercise both initial concatenation and subsequent ring wraps.
                for i in range(length):
                    cache.update_and_fetch(k[..., i:i + 1, :], v[..., i:i + 1, :])
            history.append((k, v))
        rows.append(row)
        histories.append(history)
    return rng, rows, histories


def canonical(cache):
    if cache.keys is None:
        return None
    if type(cache) is RotatingKVCache:
        return cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    return cache.keys_and_values()


def equal_cache(actual, expected):
    assert type(actual) is type(expected)
    assert actual.offset == expected.offset
    a, b = canonical(actual), canonical(expected)
    if a is None or b is None:
        assert a is b
    else:
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def forward(tx, rng, histories=None):
    appends = []
    for layer, view in enumerate(tx.caches):
        window = 4 if layer == 0 else None
        mask = view.make_mask(max(tx.lengths), window_size=window)
        keys, values = data(rng, len(tx.lengths), max(tx.lengths))
        q = rng.normal(size=(len(tx.lengths), 4, max(tx.lengths), 4)).astype(np.float32)
        assert view.update_and_fetch(keys, values) == (None, None)
        actual = view.bucketed_attention(q, 0.5, mask)
        if histories is not None:
            for lane, length in enumerate(tx.lengths):
                if not length:
                    np.testing.assert_array_equal(actual[lane:lane + 1], 0)
                    continue
                old_k, old_v = histories[lane][layer]
                all_k = np.concatenate([old_k, keys[lane:lane + 1, :, :length]], axis=2)
                all_v = np.concatenate([old_v, values[lane:lane + 1, :, :length]], axis=2)
                # Independent full-history oracle, not the cache's own mask.
                positions = old_k.shape[2] + np.arange(length)
                causal = np.arange(all_k.shape[2])[None, :] <= positions[:, None]
                if window:
                    causal &= np.arange(all_k.shape[2])[None, :] > positions[:, None] - window
                expected = attention(q[lane:lane + 1, :, :length], all_k, all_v,
                                     scale=0.5, mask=causal)
                np.testing.assert_allclose(actual[lane:lane + 1, :, :length], expected,
                                           rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(actual[lane:lane + 1, :, length:], 0)
        appends.append((keys, values))
    return appends


@pytest.mark.parametrize("lengths,accepted", [([4, 4, 4], [0, 1, 3]), ([1, 3, 2], [1, 1, 2]),
                                              ([3, 0, 1], [2, 0, 0]), ([4, 4, 4], [4, 4, 4])])
def test_divergent_commit_matches_standalone_prefix_and_full_history_attention(lengths, accepted):
    rng, rows, histories = setup([2, 5, 9])
    before = copy.deepcopy(rows)
    counts = Counter()
    owner = SegmentedKVRows(rows, note=lambda key, amount: counts.update({key: amount}))
    tx = owner.begin(lengths)
    appends = forward(tx, rng, histories)
    assert owner.nbytes >= tx.snapshot_nbytes > 0
    tx.commit(accepted)
    for lane, count in enumerate(accepted):
        for layer, (keys, values) in enumerate(appends):
            if count:
                before[lane][layer].update_and_fetch(keys[lane:lane + 1, :, :count],
                                                     values[lane:lane + 1, :, :count])
            equal_cache(rows[lane][layer], before[lane][layer])
    assert counts["rejected_input_tokens"] == sum(lengths) - sum(accepted)
    assert counts["independent_attention_rows"] == len(rows) * 2
    assert tx.closed and tx.snapshot_nbytes == 0


def test_counterfactual_offset_only_rewind_loses_evicted_ring_tokens():
    rng, rows, histories = setup([9, 6])
    before = copy.deepcopy(rows)
    owner = SegmentedKVRows(rows)
    tx = owner.begin([4, 4])
    appends = forward(tx, rng, histories)
    wrong = copy.deepcopy(rows[0][0])
    wrong.offset = before[0][0].offset + 1
    expected = before[0][0]
    expected.update_and_fetch(appends[0][0][0:1, :, :1], appends[0][1][0:1, :, :1])
    assert not np.array_equal(canonical(wrong)[0], canonical(expected)[0])
    tx.commit([1, 3])
    equal_cache(rows[0][0], expected)


@pytest.mark.parametrize("partial", [False, True])
def test_abort_and_exception_restore_ring_and_global_state(partial):
    rng, rows, _ = setup([0, 8])
    before = copy.deepcopy(rows)
    owner = SegmentedKVRows(rows)
    with pytest.raises(RuntimeError, match="cancel"):
        with owner.begin([3, 2]) as tx:
            if partial:
                tx.caches[0].update_and_fetch(*data(rng, 2, 3))
            else:
                forward(tx, rng)
            raise RuntimeError("cancel")
    for actual, expected in zip(rows, before):
        for a, b in zip(actual, expected):
            equal_cache(a, b)
    assert owner._active is None


def test_membership_extraction_and_next_round_preserve_independent_rows():
    rng, rows, histories = setup([6, 8])
    owner = SegmentedKVRows(rows)
    tx = owner.begin([3, 3])
    for operation in (lambda: owner.filter([0]), lambda: owner.extract(0), lambda: owner.extend(rows)):
        with pytest.raises(RuntimeError, match="leased"):
            operation()
    forward(tx, rng, histories)
    tx.commit([1, 2])
    extracted = owner.extract(1)
    owner.filter([0])
    owner.extend([extracted])
    assert owner.rows[1][0] is not rows[1][0]
    with owner.begin([1, 1]) as second:
        forward(second, rng)
        second.commit([1, 0])
    assert [row[0].offset for row in owner.rows] == [8, 10]
    assert rows[1][0].offset == 10
    with pytest.raises(RuntimeError, match="stale"):
        tx.commit([1, 2])


def test_bad_counts_incomplete_forward_and_stale_cache_fail_before_commit():
    rng, rows, _ = setup([3, 9])
    owner = SegmentedKVRows(rows)
    tx = owner.begin([2, 2])
    with pytest.raises(RuntimeError, match="every target layer"):
        tx.commit([1, 1])
    forward(tx, rng)
    for counts in ([3, 1], [-1, 0], [True, 1], [1]):
        with pytest.raises(ValueError):
            tx.commit(counts)
        assert [row[0].offset for row in rows] == [5, 11]
    rows[0][0].offset += 1
    with pytest.raises(RuntimeError, match="outside"):
        tx.commit([1, 1])
    rows[0][0].offset -= 1
    tx.abort()


def test_cache_geometry_and_ownership_validation():
    _, rows, _ = setup([2, 3])
    with pytest.raises(ValueError, match="independent"):
        SegmentedKVRows([rows[0], rows[0]])
    rows[1][0].max_size = 8
    with pytest.raises(ValueError, match="geometry"):
        SegmentedKVRows(rows)


def test_prompt_mask_can_be_reused_by_same_geometry_layer():
    rng, rows, _ = setup([4, 6])
    rows = [[row[0], copy.deepcopy(row[0])] for row in rows]
    owner = SegmentedKVRows(rows)
    tx = owner.begin([2, 1])
    mask = tx.caches[0].make_mask(2, window_size=4)
    for view in tx.caches:
        view.update_and_fetch(*data(rng, 2, 2))
        view.bucketed_attention(rng.normal(size=(2, 4, 2, 4)), 0.5, mask)
    tx.abort()


def test_randomized_repeated_wrap_reject_and_cancel_rounds():
    for batch in range(1, 5):
        rng, rows, histories = setup(list(range(5, 5 + batch)), seed=810 + batch)
        owner = SegmentedKVRows(rows)
        for round_index in range(20):
            lengths = rng.integers(0, 6, size=batch).tolist()
            lengths[0] = max(1, lengths[0])
            accepted = [int(rng.integers(0, n + 1)) for n in lengths]
            expected = copy.deepcopy(owner.rows)
            tx = owner.begin(lengths)
            appends = forward(tx, rng, histories)
            if round_index % 5 == 0:
                tx.abort()
                accepted = [0] * batch
            else:
                tx.commit(accepted)
            for lane, count in enumerate(accepted):
                for layer, (keys, values) in enumerate(appends):
                    if count:
                        new = (keys[lane:lane + 1, :, :count], values[lane:lane + 1, :, :count])
                        expected[lane][layer].update_and_fetch(*new)
                        histories[lane][layer] = tuple(np.concatenate([old, add], axis=2)
                                                      for old, add in zip(histories[lane][layer], new))
                    equal_cache(owner.rows[lane][layer], expected[lane][layer])


def test_closed_transaction_cannot_pin_departed_authoritative_caches():
    rng, rows, _ = setup([5])
    refs = [weakref.ref(cache) for cache in rows[0]]
    owner = SegmentedKVRows(rows)
    tx = owner.begin([3])
    forward(tx, rng)
    tx.commit([1])
    del owner, rows
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert tx.retained_nbytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        tx.caches[0].make_mask(3)
