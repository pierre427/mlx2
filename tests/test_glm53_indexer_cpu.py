"""GLM-5.3 sparse DSA indexer core (P1): synthetic CPU laws, no checkpoint.

docs/design/glm53-sparse-dsa.md sections 4.1, 5.2 and 6. The Metal cases run
only with MLX2_RUN_GPU_TESTS=1 (under the GPU queue); everything else runs on
the CPU with ``mx.fast.metal_kernel`` forbidden.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import glm53_gguf_indexer as ix
from mlx2.runtime.models.cache import KVCache

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
EPS32 = 2.0**-23


def _forbidden(*args, **kwargs):
    raise AssertionError("GPU forbidden")


@pytest.fixture(autouse=True)
def cpu(request, monkeypatch):
    if request.node.name.startswith("test_gpu_"):
        yield
    else:
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        monkeypatch.setattr(mx.fast, "metal_kernel", _forbidden)
        yield
        mx.set_default_device(previous)


class Weights:
    """The indexer tensors of one DSA layer, random and fixed by seed."""

    def __init__(self, seed=75):
        self.rng = np.random.default_rng(seed)
        self.arrays = {}

    def array(self, name):
        if name not in self.arrays:
            if name.endswith("kpool_ape"):
                value = self.rng.normal(0, 0.1, (4, 128))
            elif name.endswith("k_norm.weight"):
                value = np.ones(128)
            elif name.endswith("k_norm.bias"):
                value = np.zeros(128)
            else:
                raise AssertionError(name)
            self.arrays[name] = mx.array(value.astype(np.float32))
        return self.arrays[name]

    def linear(self, name):
        def call(x):
            if name not in self.arrays:
                out = {"indexer.attn_q_b.weight": 4096, "indexer.proj.weight": 32}.get(
                    name.split(".", 2)[-1], 128
                )
                self.arrays[name] = mx.array(self.rng.normal(0, 0.05, (x.shape[-1], out)).astype(np.float32))
            return x @ self.arrays[name]

        return call


def _today(x, qr, cache, weights, prefix, *, topk=2048, pool=4, projected=None):
    # The pre-P1 pooled_selection body (glm53_gguf_indexer.py at 816f59d18), frozen.
    b, length, _ = x.shape
    if b != 1 or pool != 4:
        raise ValueError("GLM-5.3 packed indexer currently requires unpadded B1/kpool4")
    hd, heads = 128, 32
    projected = projected or {}
    key = projected["key"] if "key" in projected else weights.linear(prefix + ".indexer.attn_k.weight")(x)
    mean = key.mean(-1, keepdims=True)
    key = (key - mean) * mx.rsqrt(
        ((key - mean) ** 2).mean(-1, keepdims=True) + ix.INDEXER_K_NORM_EPS
    )
    key = key * weights.array(prefix + ".indexer.k_norm.weight") + weights.array(
        prefix + ".indexer.k_norm.bias"
    )
    gates = weights.linear(prefix + ".indexer.kpool_gate")(x)
    packed = mx.concatenate((key, gates), axis=-1)[:, None]
    full, _ = cache.update_and_fetch(packed, mx.zeros((b, 1, length, 0), dtype=x.dtype))
    full = full[:, 0]
    total = full.shape[1]
    if total <= topk:
        return None
    q = (projected["q"] if "q" in projected
         else weights.linear(prefix + ".indexer.attn_q_b.weight")(qr)).reshape(b, length, heads, hd)
    score_weights = (projected["proj"] if "proj" in projected
                     else weights.linear(prefix + ".indexer.proj.weight")(x)) * heads**-0.5
    key, gates = mx.split(full, 2, axis=-1)
    pools = (total + pool - 1) // pool
    padding = pools * pool - total
    if padding:
        key = mx.concatenate((key, mx.zeros((b, padding, hd), key.dtype)), axis=1)
        gates = mx.concatenate((gates, mx.zeros((b, padding, hd), gates.dtype)), axis=1)
    key = key.reshape(b, pools, pool, hd)
    gates = gates.reshape(b, pools, pool, hd)
    positions = mx.arange(pools * pool).reshape(pools, pool)
    valid = positions < total
    logits = gates + weights.array(prefix + ".indexer.kpool_ape")[None, None]
    probs = mx.softmax(mx.where(valid[None, ..., None], logits, -1e30), axis=2)
    pooled = (probs * key).sum(axis=2)
    ends = mx.arange(pools) * pool + pool - 1
    count = min(topk // pool, pools)
    results = []
    for start in range(0, length, 128):
        stop = min(length, start + 128)
        pos = total - length + mx.arange(start, stop)
        scores = mx.maximum(
            q[:, start:stop] @ pooled[:, None].swapaxes(-1, -2) * hd**-0.5, 0
        )
        scores = (scores * score_weights[:, start:stop, :, None]).sum(axis=2)
        visible = ends[None, None, :] <= pos[None, :, None]
        scores = mx.where(visible, scores, -1e30)
        selected = mx.argsort(-scores, axis=-1)[..., :count]
        selected_valid = mx.take_along_axis(
            mx.broadcast_to(visible, (b, stop - start, pools)), selected, axis=-1
        )
        ids = selected[..., None] * pool + mx.arange(pool)
        ids = mx.where(selected_valid[..., None], ids, -1).reshape(
            b, stop - start, count * pool
        )
        visible_count = pos + 1
        tail_count = visible_count % pool
        tail_start = visible_count - tail_count
        tail = tail_start[:, None] + mx.arange(pool - 1)
        tail = mx.where(mx.arange(pool - 1)[None, :] < tail_count[:, None], tail, -1)
        ids = mx.concatenate(
            (ids, mx.broadcast_to(tail[None], (b, stop - start, pool - 1))), axis=-1
        )
        if ids.shape[-1] < topk + pool - 1:
            ids = mx.concatenate(
                (ids, mx.full((b, stop - start, topk + pool - 1 - ids.shape[-1]), -1, dtype=ids.dtype)),
                axis=-1,
            )
        results.append(ids.astype(mx.int32))
    return mx.concatenate(results, axis=1)


def _inputs(rng, length):
    x = mx.array(rng.normal(size=(1, length, 4)).astype(np.float32))
    qr = mx.array(rng.normal(size=(1, length, 6)).astype(np.float32))
    return x, qr


def _bits(a):
    return np.array(a).view(np.uint32)


def _raw(rng, pools, gate_scale=1.0):
    raw = rng.normal(size=(4 * pools, 256)).astype(np.float32)
    raw[:, 128:] *= gate_scale
    return raw


def _oracle(raw, ape):
    """float64 pooling from the fp32-rounded logits, and M = sum_j |p_j| |k_j|."""
    rows = raw.reshape(-1, 4, 256)
    key = rows[..., :128].astype(np.float64)
    logits = (rows[..., 128:] + ape[None]).astype(np.float32).astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return (p * key).sum(axis=1), (np.abs(p) * np.abs(key)).sum(axis=1)


# ---- reference law ---------------------------------------------------------


def test_default_pooled_selection_is_todays_law_bitwise():
    rng = np.random.default_rng(31)
    w = Weights()
    ours, theirs = KVCache(), KVCache()
    for length in (2049, 1, 1, 1, 5, 130):
        x, qr = _inputs(rng, length)
        a = ix.pooled_selection(x, qr, ours, w, "blk.3")
        b = _today(x, qr, theirs, w, "blk.3")
        np.testing.assert_array_equal(np.array(a), np.array(b))
        assert a.dtype == mx.int32 and a.shape == (1, length, ix.WIDTH)
    x, qr = _inputs(rng, 7)
    assert ix.pooled_selection(x, qr, KVCache(), w, "blk.3") is None


def test_pool_reference_is_todays_expression_bitwise():
    rng = np.random.default_rng(32)
    raw = mx.array(_raw(rng, 37, 3.0)[None])
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    key, gates = mx.split(raw, 2, axis=-1)
    key, gates = key.reshape(1, 37, 4, 128), gates.reshape(1, 37, 4, 128)
    valid = mx.arange(37 * 4).reshape(37, 4) < 37 * 4
    probs = mx.softmax(mx.where(valid[None, ..., None], gates + ape[None, None], -1e30), axis=2)
    np.testing.assert_array_equal(_bits(ix.pool_reference(raw, ape)), _bits((probs * key).sum(axis=2)))


def test_reference_law_refuses_new_keywords_and_select_from_rows_refuses_reference():
    rng = np.random.default_rng(33)
    w = Weights()
    x, qr = _inputs(rng, 3)
    for kwargs in ({"select": "topk_v1"}, {"pools": ix.Glm53PoolCache()}, {"block": 16}):
        with pytest.raises(ValueError, match="reference pooling"):
            ix.pooled_selection(x, qr, KVCache(), w, "blk.3", **kwargs)
    with pytest.raises(ValueError, match="argsort selects globally"):
        ix.pooled_selection(x, qr, KVCache(), w, "blk.3", pool_law="fixed_order", block=16)
    with pytest.raises(ValueError, match="unknown"):
        ix.pooled_selection(x, qr, KVCache(), w, "blk.3", pool_law="fixed_order", select="topk")
    with pytest.raises(ValueError, match="fixed_order"):
        ix.select_from_rows(x, qr, w, "blk.3", total=4000, raw_rows_fn=None, pools=None,
                            pool_law="reference", select="argsort", block=None)


# ---- pooling laws (6.2) ----------------------------------------------------


@pytest.mark.parametrize("case", ["unit", "large_gates", "cancellation", "huge_gates"])
def test_pooling_laws_within_cancellation_aware_float64_bounds(case):
    rng = np.random.default_rng(34)
    ape = rng.normal(0, 0.1, (4, 128)).astype(np.float32)
    if case == "unit":
        raw = _raw(rng, 64)
    elif case == "large_gates":
        raw = _raw(rng, 64, 30.0)
    elif case == "cancellation":
        raw = _raw(rng, 8)
        raw[:, 128:] = 0.0
        ape[:] = 0.0
        raw[:, :128] = np.tile(np.array([1.0, 2.0**-25, -1.0, 2.0**-25], np.float32), 8)[:, None]
    else:
        raw = _raw(rng, 8)
        raw[:, 128:] = 2.0**24
        ape[:] = 0.0
        ape[1] = 1.0
    expected, magnitude = _oracle(raw, ape)
    fixed = np.array(ix.pool_fixed_order(mx.array(raw), mx.array(ape))).astype(np.float64)
    reference = np.array(ix.pool_reference(mx.array(raw)[None], mx.array(ape))[0]).astype(np.float64)
    assert np.all(np.abs(fixed - expected) <= 8 * EPS32 * magnitude)
    assert np.all(np.abs(reference - expected) <= 8 * EPS32 * magnitude)
    # Between laws: recorded under the 16 eps M bound.
    assert np.all(np.abs(fixed - reference) <= 16 * EPS32 * magnitude)


def test_fixed_order_pool_is_shape_invariant():
    rng = np.random.default_rng(35)
    raw = mx.array(_raw(rng, 70, 5.0))
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    whole = ix.pool_fixed_order(raw, ape)
    parts = mx.concatenate([ix.pool_fixed_order(raw[4 * a: 4 * b], ape) for a, b in ((0, 1), (1, 33), (33, 70))])
    np.testing.assert_array_equal(_bits(whole), _bits(parts))
    np.testing.assert_array_equal(_bits(ix.pool_fixed_order(raw[None], ape)[0]), _bits(whole))
    with pytest.raises(ValueError, match="complete pools"):
        ix.pool_fixed_order(raw[:6], ape)


# ---- incremental pool cache (6.3, 6.5) ---------------------------------------


def test_extend_equals_full_recompute_after_random_appends_and_cuts():
    rng = np.random.default_rng(36)
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    rows = _raw(rng, 120, 4.0)
    cache = ix.Glm53PoolCache()
    total = 0
    for step in range(90):
        if step in (40, 70):  # rewind, then replace the rows after the cut
            total = max(0, total - int(rng.integers(1, 13)))
            cache.cut(total)
            rows[total:] = _raw(rng, 120, 4.0)[total:]
        before = cache.complete
        total = min(len(rows), total + int(rng.integers(1, 10)))
        full = mx.array(rows[:total])
        pooled = cache.extend(lambda a, b, full=full: full[a:b], total, ape)
        assert cache.complete == total // 4 and cache.last_computed == total // 4 - before
        assert cache.pooled.shape[1] % ix.POOL_STEP == 0
        expected = ix.pool_fixed_order(full[: 4 * (total // 4)], ape)
        assert pooled.shape == (1, total // 4, 128)
        np.testing.assert_array_equal(_bits(pooled[0]), _bits(expected))
    assert cache.pooled.shape[1] == 2 * ix.POOL_STEP


def test_extend_cuts_defensively_and_does_no_work_inside_a_pool():
    rng = np.random.default_rng(37)
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    full = mx.array(_raw(rng, 10))
    cache = ix.Glm53PoolCache()
    cache.extend(lambda a, b: full[a:b], 21, ape)
    calls = []
    cache.extend(lambda a, b: calls.append((a, b)) or full[a:b], 23, ape)
    assert calls == [] and cache.last_computed == 0
    # A shorter total re-pools from the cut, never trusting pools past it.
    cache.extend(lambda a, b: calls.append((a, b)) or full[a:b], 13, ape)
    assert cache.complete == 3 and calls == []
    cache.extend(lambda a, b: calls.append((a, b)) or full[a:b], 17, ape)
    assert calls == [(12, 16)]


def test_clone_siblings_complete_the_same_pool_privately():
    rng = np.random.default_rng(38)
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    base = _raw(rng, 3)[:11]
    parent = ix.Glm53PoolCache()
    shared = mx.array(base)
    parent.extend(lambda a, b: shared[a:b], 11, ape)
    left, right = parent.clone(), parent.clone()
    assert left.pooled is not parent.pooled and left.complete == parent.complete == 2
    outs = []
    for child in (left, right):
        rows = mx.array(np.concatenate((base, _raw(rng, 1)[:1])))
        outs.append((child.extend(lambda a, b, r=rows: r[a:b], 12, ape), rows))
    for got, rows in outs:
        np.testing.assert_array_equal(_bits(got[0]), _bits(ix.pool_fixed_order(rows, ape)))
    assert not np.array_equal(_bits(outs[0][0][0, 2]), _bits(outs[1][0][0, 2]))
    assert parent.complete == 2
    np.testing.assert_array_equal(_bits(parent.pooled[0, :2]), _bits(ix.pool_fixed_order(shared[:8], ape)))
    assert not np.array(parent.pooled[0, 2:]).any()  # no child wrote into the parent's buffer


def test_audit_pools_detects_a_stale_pool():
    rng = np.random.default_rng(39)
    ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
    rows = _raw(rng, 5)
    full = mx.array(rows)
    cache = ix.Glm53PoolCache()
    pooled = cache.extend(lambda a, b: full[a:b], 20, ape)
    assert ix.audit_pools(pooled, lambda a, b: full[a:b], 20, ape)
    rows[19] += 1.0
    stale = mx.array(rows)
    with pytest.raises(ValueError, match="differ from recompute"):
        ix.audit_pools(pooled, lambda a, b: stale[a:b], 20, ape)


# ---- selection laws (4.1) ----------------------------------------------------


def _chronological(scores, count):
    order = np.lexsort((np.arange(scores.shape[-1]), -scores.astype(np.float64)))
    return np.sort(order[:count])


def test_stable_topk_equals_deepseek_stable_pool_topk():
    from mlx2.runtime.models.deepseek_v4 import stable_pool_topk

    rng = np.random.default_rng(40)
    for columns, count in ((7, 1), (7, 7), (600, 512), (1500, 512)):
        for values in (rng.normal(size=(3, columns)), rng.integers(0, 4, size=(3, columns)) * 0.25):
            s = mx.array(values.astype(np.float32))
            np.testing.assert_array_equal(np.array(ix.stable_topk(s, count)), np.array(stable_pool_topk(s, count)))
            for row in range(3):
                np.testing.assert_array_equal(
                    np.array(ix.stable_topk(s, count))[row], _chronological(values[row], count)
                )
    with pytest.raises(ValueError, match="fit"):
        ix.stable_topk(mx.zeros((1, 3)), 4)


def test_select_topk_equals_argsort_as_sets_without_ties_and_chronological_with_ties():
    rng = np.random.default_rng(41)
    pools = 700
    positions = mx.array([4 * pools, 4 * pools + 2, 1500, 2051], dtype=mx.int32)
    visible = ix.visible_pools(positions, pools)[None]
    free = mx.array(rng.permutation(4 * pools)[: pools * 4].reshape(4, pools).astype(np.float32)[None])
    a_ids, a_valid = ix.select_pools(free, visible, 512, law="argsort")
    t_ids, t_valid = ix.select_pools(free, visible, 512, law="topk_v1")
    a_ids, a_valid, t_ids, t_valid = (np.array(v)[0] for v in (a_ids, a_valid, t_ids, t_valid))
    for row in range(4):  # invisible pools are -inf ties: compare the valid sets
        assert set(a_ids[row][a_valid[row]]) == set(t_ids[row][t_valid[row]])
        assert a_valid[row].sum() == t_valid[row].sum() == min(512, (int(positions[row]) + 1) // 4)
        assert np.all(np.diff(t_ids[row]) > 0)
    tied = rng.integers(0, 3, size=(1, 4, pools)).astype(np.float32)
    ids, valid = ix.select_pools(mx.array(tied), visible, 512, law="topk_v1")
    vis = np.array(visible)
    for row in range(4):
        masked = np.where(vis[0, row], tied[0, row], -np.inf)
        np.testing.assert_array_equal(np.array(ids)[0, row], _chronological(masked, 512))
        np.testing.assert_array_equal(np.array(valid)[0, row], vis[0, row][np.array(ids)[0, row]])


def test_invisible_and_nonfinite_pools_rank_below_every_finite_visible_score():
    pools = 520
    scores = np.full((1, 1, pools), -3e38, np.float32)  # finite, far below today's -1e30 sentinel
    scores[0, 0, :5] = [np.nan, np.inf, -np.inf, 1.0, -0.0]
    positions = mx.array([4 * 514 + 3], dtype=mx.int32)  # pools 515..519 invisible, no tail
    for law in ("argsort", "topk_v1"):
        ids, valid = ix.select_pools(mx.array(scores), ix.visible_pools(positions, pools)[None], 512, law=law)
        chosen = dict(zip(np.array(ids)[0, 0].tolist(), np.array(valid)[0, 0].tolist()))
        assert all(chosen.get(p) for p in range(3, 515)) and len(chosen) == 512
        assert not any(chosen.get(p, False) for p in (0, 1, 2, 515, 516, 517, 518, 519))
    ids = ix.select_expand_ops(mx.array(scores[0]), positions, 512)
    out = np.array(ids)[0]
    assert set(out[out >= 0].tolist()) == set(range(12, 4 * 515))


def test_empty_selection_failure_contract_is_all_minus_one():
    # Own tail empty ((pos + 1) % 4 == 0) and every visible score nonfinite.
    scores = mx.array(np.full((2, 600), np.nan, np.float32))
    positions = mx.array([2399, 2047], dtype=mx.int32)
    out = np.array(ix.select_expand_ops(scores, positions, 512))
    assert np.all(out == -1)


def test_expand_tail_padding_and_causality():
    pools, count = 600, 3
    pool_ids = mx.array([[[0, 5, 9]] * 4], dtype=mx.int32)
    pool_valid = mx.array([[[True, True, False]] * 4])
    positions = mx.array([39, 40, 41, 42], dtype=mx.int32)
    out = np.array(ix.expand_selection(pool_ids, pool_valid, positions))
    assert out.shape == (1, 4, ix.WIDTH) and out.dtype == np.int32
    np.testing.assert_array_equal(out[0, :, :12], np.tile([0, 1, 2, 3, 20, 21, 22, 23, -1, -1, -1, -1], (4, 1)))
    np.testing.assert_array_equal(out[0, :, 12:15], [[-1, -1, -1], [40, -1, -1], [40, 41, -1], [40, 41, 42]])
    assert np.all(out[0, :, 15:] == -1)
    del pools, count


def test_new_path_pool_boundary_and_per_query_causality():
    """test_indexer_pool_boundary_and_per_query_causality, on the opt-in laws."""
    rng = np.random.default_rng(42)
    w = Weights()
    for select in ("topk_v1", "argsort"):
        cache, pools = KVCache(), ix.Glm53PoolCache()
        kwargs = {"pool_law": "fixed_order", "select": select, "pools": pools}
        x, qr = _inputs(np.random.default_rng(33), 2049)
        a = np.array(ix.pooled_selection(x, qr, cache, w, "blk.3", **kwargs))
        for t in [0, 3, 100, 2047, 2048]:
            actual = a[0, t][a[0, t] >= 0]
            np.testing.assert_array_equal(np.sort(actual), np.arange(t + 1))
        for i in range(9):
            x, qr = _inputs(rng, 1)
            ids = np.array(ix.pooled_selection(x, qr, cache, w, "blk.3", **kwargs))[0, 0]
            pos = 2049 + i
            actual = ids[ids >= 0]
            assert actual.max() <= pos and len(np.unique(actual)) == len(actual)
            assert len(actual) == 2048 + (pos + 1) % 4
            if select == "topk_v1":
                assert np.all(np.diff(actual) > 0)  # chronological


def test_incremental_recompute_and_restore_select_identical_ids():
    w = Weights()
    rng = np.random.default_rng(43)
    steps = [_inputs(rng, n) for n in (2050, 1, 1, 1, 1, 3, 1, 140)]
    runs = {}
    for name, use_pools in (("incremental", True), ("recompute", False)):
        cache, pools, out = KVCache(), ix.Glm53PoolCache() if use_pools else None, []
        for x, qr in steps:
            out.append(np.array(ix.pooled_selection(x, qr, cache, w, "blk.3", pools=pools,
                                                    pool_law="fixed_order", select="topk_v1")))
        runs[name] = (out, cache)
    for left, right in zip(runs["incremental"][0], runs["recompute"][0]):
        np.testing.assert_array_equal(left, right)
    cache = runs["incremental"][1]
    restored = KVCache.from_state(cache.state, cache.meta_state)
    x, qr = _inputs(rng, 1)
    kw = {"pool_law": "fixed_order", "select": "topk_v1"}
    left = ix.pooled_selection(x, qr, cache, w, "blk.3", pools=ix.Glm53PoolCache(), **kw)
    right = ix.pooled_selection(x, qr, restored, w, "blk.3", **kw)
    np.testing.assert_array_equal(np.array(left), np.array(right))


# ---- blocked running top-k (5.2) ---------------------------------------------


def _exact_operands(rng, pools, length, distinct):
    """Operands whose pool scores are exact in fp32, so every shape agrees."""
    q = np.zeros((1, length, 32, 128), np.float32)
    q[..., 0, :8] = rng.integers(-3, 4, size=(1, length, 8))
    rows = rng.integers(-3, 4, size=(distinct, 128)).astype(np.float32)
    pooled = rows[rng.integers(0, distinct, size=pools)][None]
    weights = np.zeros((1, length, 32), np.float32)
    weights[..., 0] = 0.5
    return mx.array(q), mx.array(weights), mx.array(pooled)


@pytest.mark.parametrize("distinct", [5, 600])
def test_blocked_select_equals_global_select(distinct):
    rng = np.random.default_rng(44)
    pools, length = 600, 6
    q, weights, pooled = _exact_operands(rng, pools, length, distinct)
    positions = mx.array([2048, 2050, 2051, 2100, 2398, 2402], dtype=mx.int32)
    scores = ix.pool_scores(q, weights, pooled)
    expected = ix.select_pools(scores, ix.visible_pools(positions, pools)[None], 512, law="topk_v1")
    for block in (16, 100, 512, 4096):
        got = ix.blocked_select(q, weights, pooled, positions, 512, block=block)
        np.testing.assert_array_equal(np.array(got[0]), np.array(expected[0]))
        np.testing.assert_array_equal(np.array(got[1]), np.array(expected[1]))


def test_blocked_select_on_random_operands_equals_global_and_pads():
    rng = np.random.default_rng(45)
    q = mx.array(rng.normal(size=(1, 3, 32, 128)).astype(np.float32))
    weights = mx.array(rng.normal(size=(1, 3, 32)).astype(np.float32))
    pooled = mx.array(rng.normal(size=(1, 600, 128)).astype(np.float32))
    positions = mx.array([2047, 2300, 2403], dtype=mx.int32)
    scores = ix.pool_scores(q, weights, pooled)
    expected = ix.select_pools(scores, ix.visible_pools(positions, 600)[None], 512, law="topk_v1")
    got = ix.blocked_select(q, weights, pooled, positions, 512, block=16)
    np.testing.assert_array_equal(np.array(got[0]), np.array(expected[0]))
    np.testing.assert_array_equal(np.array(got[1]), np.array(expected[1]))
    ids, valid = ix.blocked_select(q, weights, pooled[:, :10], positions, 12, block=4)
    assert np.all(np.array(ids)[..., 10:] == -1) and not np.array(valid)[..., 10:].any()


def test_blocked_prefill_through_select_from_rows_equals_global():
    w = Weights()
    x, qr = _inputs(np.random.default_rng(46), 2400)
    kw = {"pool_law": "fixed_order", "select": "topk_v1"}
    a = ix.pooled_selection(x, qr, KVCache(), w, "blk.3", block=64, **kw)
    b = ix.pooled_selection(x, qr, KVCache(), w, "blk.3", **kw)
    np.testing.assert_array_equal(np.array(a), np.array(b))


# ---- select+expand kernel (4.1) ----------------------------------------------


def _radix_emulation(scores, positions, count):
    """The kernel's algorithm in numpy: ordered keys, 4 x 8-bit radix passes,
    ascending emission, tail and padding."""
    rows, pools = scores.shape
    out = np.full((rows, ix.WIDTH), -1, np.int32)
    neg_inf = np.uint32(0x007FFFFF)
    p = np.arange(pools)
    for r in range(rows):
        bits = scores[r].view(np.uint32).copy()
        bad = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) | (4 * p + 3 > positions[r])
        bits[bits == np.uint32(0x80000000)] = 0
        key = np.where(bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000)).astype(np.uint32)
        key[bad] = neg_inf
        prefix, mask, want = 0, 0, count
        for shift in (24, 16, 8, 0):
            match = (key & np.uint32(mask)) == np.uint32(prefix)
            hist = np.bincount(((key[match] >> np.uint32(shift)) & np.uint32(255)).astype(np.int64), minlength=256)
            above = 0
            for digit in range(255, -1, -1):
                if above + hist[digit] >= want:
                    break
                above += hist[digit]
            prefix |= digit << shift
            mask |= 255 << shift
            want -= above
        tied = key == np.uint32(prefix)
        rank = np.cumsum(tied) - tied
        take = (key > np.uint32(prefix)) | (tied & (rank < want))
        chosen = p[take]
        assert len(chosen) == count
        valid = key[take] != neg_inf
        out[r, : 4 * count] = np.where(valid[:, None], chosen[:, None] * 4 + np.arange(4), -1).reshape(-1)
        tail_count = (positions[r] + 1) % 4
        for i in range(tail_count):
            out[r, 4 * count + i] = positions[r] + 1 - tail_count + i
    return out


def test_select_kernel_algorithm_matches_ops_bitwise_on_gate_cases():
    for scores, positions, count in ix.gate_cases():
        expected = np.array(ix.select_expand_ops(mx.array(scores), mx.array(positions), count))
        np.testing.assert_array_equal(_radix_emulation(scores, positions, count), expected)


def test_select_kernel_spec_builds_and_cpu_refuses_metal(monkeypatch):
    spec = ix.select_kernel_spec()
    assert spec["name"] == "mlx2_glm53_select_expand_v1" and spec["ensure_row_contiguous"]
    assert spec["input_names"] == ["scores", "positions", "meta"] and spec["output_names"] == ["ids"]
    assert ("WIDTH", 2051) in spec["template"] and spec["threadgroup"] == (1024, 1, 1)
    assert "Apple" not in spec["source"] + spec["header"]
    with pytest.raises(ValueError, match="requires the Metal GPU"):
        ix.select_expand(mx.zeros((1, 600)), mx.array([2400], dtype=mx.int32), 512)
    assert ix.select_kernel_ready() is False
    # The kernel policy falls back to the ops law on CPU, counted, no Metal call.
    w = Weights()
    x, qr = _inputs(np.random.default_rng(47), 2050)
    before = ix.route_status()["scheduled_calls"].get("select_kernel_declined", 0)
    kw = {"pool_law": "fixed_order", "pools": None}
    a = ix.pooled_selection(x, qr, KVCache(), w, "blk.3", select="topk_v1_kernel", **kw)
    b = ix.pooled_selection(x, qr, KVCache(), w, "blk.3", select="topk_v1", **kw)
    np.testing.assert_array_equal(np.array(a), np.array(b))
    assert ix.route_status()["scheduled_calls"]["select_kernel_declined"] == before + 1


# ---- pool-score laws (mlx-lm#2046 fusion): same bits as the reference --------


def _score_operands(rng, length, pools, *, ties=True, odd=False):
    """Non-repeated random operands; ``ties`` repeats whole pooled rows (tied
    scores) and zeroes some (ReLU-zero products, signed zeros); about half of
    the pre-ReLU products are negative; head weights take both signs."""
    q = rng.normal(size=(1, length, 32, 128)).astype(np.float32)
    pooled = rng.normal(size=(1, pools, 128)).astype(np.float32)
    if ties and pools > 8:
        pooled[0, 1::7] = pooled[0, 0]
        pooled[0, 2] = 0.0
        pooled[0, 3] = -0.0
    w = (rng.normal(size=(1, length, 32)) * 32**-0.5).astype(np.float32)
    if odd and pools > 8:
        pooled[0, 4, 0] = np.nan
        pooled[0, 5, 0] = np.inf
        pooled[0, 6, 0] = -np.inf
        pooled[0, 7, :3] = 3e38
        q[0, 0, 0, :4] = 1e-38
    return mx.array(q), mx.array(w), mx.array(pooled)


def _numpy_scores(q, w, pooled):
    """The reference law in numpy fp32, heads added left to right (the CPU
    reduce's order): (mm * fp32(128**-0.5)) -> max(., 0) -> * w -> sum."""
    mm = np.array(q @ pooled[:, None].swapaxes(-1, -2))  # MLX's matmul, its order
    scale = np.float32(128**-0.5)
    with np.errstate(invalid="ignore", over="ignore"):
        a = mm * scale
        relu = np.where(np.isnan(a), a, np.where(a > 0, a, np.float32(0)))
        c = relu * np.array(w)[..., None]
        total = np.float32(0) + c[:, :, 0]
        for h in range(1, 32):
            total = total + c[:, :, h]
    return total.astype(np.float32)


SCORE_SHAPES = [(1, 1), (1, 511), (3, 2048), (17, 4103), (128, 4096), (128, 513), (5, 9000)]


@pytest.mark.parametrize("length,pools", SCORE_SHAPES)
def test_fused_pool_scores_equal_reference_bitwise(length, pools):
    rng = np.random.default_rng(5400 + length + pools)
    for ties, odd in ((False, False), (True, False), (True, True)):
        q, w, pooled = _score_operands(rng, length, pools, ties=ties, odd=odd)
        ref = ix.pool_scores(q, w, pooled)
        for law in ("fused_v1", "kernel_v1"):  # kernel_v1 declines to fused_v1 on the CPU
            got = ix.pool_scores(q, w, pooled, law=law)
            assert got.dtype == ref.dtype == mx.float32 and got.shape == ref.shape == (1, length, pools)
            assert np.array_equal(_bits(got), _bits(ref)), (law, ties, odd)
        # Independent oracle of the math and order: for a strided head reduce
        # (P > 1) MLX's CPU sums heads left to right (mlx-lm#2046's order); at
        # P = 1 the heads are contiguous and MLX takes a vectorized row reduce.
        # On Metal the order is col_reduce_looped's simd_sum, which this CPU
        # test cannot reach (the kernel's GPU gate does).
        if pools > 1:
            assert np.array_equal(np.asarray(_numpy_scores(q, w, pooled)).view(np.uint32), _bits(ref))


def test_score_law_counters_and_unknown_law():
    rng = np.random.default_rng(5410)
    q, w, pooled = _score_operands(rng, 2, 600)
    before = dict(ix._SCHEDULED)
    ix.pool_scores(q, w, pooled, law="fused_v1")
    ix.pool_scores(q, w, pooled, law="kernel_v1")
    after = ix._SCHEDULED
    assert after["score_fused"] - before.get("score_fused", 0) == 2
    assert after["score_kernel_declined"] - before.get("score_kernel_declined", 0) == 1
    assert after.get("score_kernel", 0) == before.get("score_kernel", 0)
    with pytest.raises(ValueError, match="unknown GLM-5.3 indexer score law"):
        ix.pool_scores(q, w, pooled, law="head_fold")
    status = ix.route_status()
    assert status["policies"]["score"] == ["reference", "fused_v1", "kernel_v1"]
    assert status["kernels"][ix.SCORE_KERNEL]["selected_by_default"] is False


@pytest.mark.parametrize("distinct", [5, 600])
def test_blocked_chained_topk_under_fused_scores_equals_reference_global(distinct):
    # Heavy ties (distinct=5): the chained running top-k must return the same
    # indices, ties included, under every score law and block size.
    rng = np.random.default_rng(5411)
    pools, length = 600, 6
    q, weights, pooled = _exact_operands(rng, pools, length, distinct)
    positions = mx.array([2048, 2050, 2051, 2100, 2398, 2402], dtype=mx.int32)
    scores = ix.pool_scores(q, weights, pooled)
    expected = ix.select_pools(scores, ix.visible_pools(positions, pools)[None], 512, law="topk_v1")
    for law in ix.SCORE_LAWS:
        for block in (16, 100, 512, 4096):
            got = ix.blocked_select(q, weights, pooled, positions, 512, block=block, score=law)
            np.testing.assert_array_equal(np.array(got[0]), np.array(expected[0]))
            np.testing.assert_array_equal(np.array(got[1]), np.array(expected[1]))


def test_blocked_chained_topk_on_random_tied_operands_under_fused_scores():
    rng = np.random.default_rng(5412)
    q, w, pooled = _score_operands(rng, 3, 700, ties=True)
    positions = mx.array([2047, 2300, 2803], dtype=mx.int32)
    for block in (16, 333):
        ref = ix.blocked_select(q, w, pooled, positions, 512, block=block)
        got = ix.blocked_select(q, w, pooled, positions, 512, block=block, score="fused_v1")
        np.testing.assert_array_equal(np.array(got[0]), np.array(ref[0]))
        np.testing.assert_array_equal(np.array(got[1]), np.array(ref[1]))


@pytest.mark.parametrize("law", ["fused_v1", "kernel_v1"])
def test_selection_ids_under_fused_scores_equal_reference_every_pool_law(law):
    """End to end: reference pooling + argsort, fixed-order + topk_v1 (global
    and blocked prompt, chunks over QUERY_CHUNK) and incremental pools."""
    w = Weights()
    rng = np.random.default_rng(5413)
    for length in (2049, 1, 130, 300):
        x, qr = _inputs(rng, length)
        for kw in ({}, {"pool_law": "fixed_order", "select": "topk_v1"},
                   {"pool_law": "fixed_order", "select": "topk_v1", "block": 64},
                   {"pool_law": "fixed_order", "select": "argsort"}):
            ours, theirs = KVCache(), KVCache()
            prior_x, prior_qr = _inputs(np.random.default_rng(5414), 2100)
            ix.pooled_selection(prior_x, prior_qr, ours, w, "blk.3", **kw)
            ix.pooled_selection(prior_x, prior_qr, theirs, w, "blk.3", **kw)
            a = ix.pooled_selection(x, qr, ours, w, "blk.3", score=law, **kw)
            b = ix.pooled_selection(x, qr, theirs, w, "blk.3", **kw)
            np.testing.assert_array_equal(np.array(a), np.array(b))
    pools_a, pools_b = ix.Glm53PoolCache(), ix.Glm53PoolCache()
    kw = {"pool_law": "fixed_order", "select": "topk_v1"}
    ca, cb = KVCache(), KVCache()
    for length in (2400, 1, 3, 1):
        x, qr = _inputs(rng, length)
        a = ix.pooled_selection(x, qr, ca, w, "blk.3", pools=pools_a, score=law, **kw)
        b = ix.pooled_selection(x, qr, cb, w, "blk.3", pools=pools_b, **kw)
        np.testing.assert_array_equal(np.array(a), np.array(b))


def test_default_reference_law_unchanged_and_score_keyword_reaches_it():
    rng = np.random.default_rng(5415)
    w = Weights()
    ours, fused, theirs = KVCache(), KVCache(), KVCache()
    for length in (2049, 1, 5, 130):
        x, qr = _inputs(rng, length)
        a = ix.pooled_selection(x, qr, ours, w, "blk.3")
        f = ix.pooled_selection(x, qr, fused, w, "blk.3", score="fused_v1")
        b = _today(x, qr, theirs, w, "blk.3")
        np.testing.assert_array_equal(np.array(a), np.array(b))
        np.testing.assert_array_equal(np.array(f), np.array(b))
    with pytest.raises(ValueError, match="unknown GLM-5.3 indexer score law"):
        ix.pooled_selection(x, qr, KVCache(), w, "blk.3", score="nope")


def test_score_kernel_spec_builds_and_cpu_refuses_metal():
    spec = ix.score_kernel_spec()
    assert spec["name"] == ix.SCORE_KERNEL and "#pragma clang fp contract(off)" in spec["source"]
    # fp32(128**-0.5) as an exact bit pattern, not a decimal literal.
    assert f"0x{int(np.float32(128**-0.5).view(np.uint32)):08X}u" in spec["source"]
    assert "simd_sum(v)" in spec["source"] and "c + 0.0f" in spec["source"]
    q, w, pooled = _score_operands(np.random.default_rng(5416), 2, 40)
    mm = q @ pooled[:, None].swapaxes(-1, -2)
    assert not ix.score_kernel_admits(mm, w) and not ix.score_kernel_ready()
    with pytest.raises(ValueError, match="requires the Metal GPU"):
        ix.score_rows(mm, w)
    assert len(ix.score_gate_cases()) == 7
    assert min(c[2].shape[1] for c in ix.score_gate_cases()) == ix.SCORE_KERNEL_MIN_POOLS


def test_score_kernel_admits_refuses_pools_below_the_col_reduce_looped_range(monkeypatch):
    """P == 1 (MLX row-reduces the 32 contiguous heads) and P < 32 (MLX's
    col_reduce_small) sum heads in another order than the kernel's simd_sum."""
    monkeypatch.setattr(ix, "_on_gpu", lambda: True)  # shape rule only; nothing launches
    rng = np.random.default_rng(5417)
    for pools, admitted in ((1, False), (2, False), (7, False), (31, False), (32, True), (4097, True)):
        q, w, pooled = _score_operands(rng, 3, pools, ties=False)
        mm = q @ pooled[:, None].swapaxes(-1, -2)
        assert ix.score_kernel_admits(mm, w) is admitted, pools
    q, w, pooled = _score_operands(rng, 3, 64, ties=False)
    mm = q @ pooled[:, None].swapaxes(-1, -2)
    assert not ix.score_kernel_admits(mm.astype(mx.float16), w)
    assert not ix.score_kernel_admits(mm, w[:, :2])


def test_blocked_last_block_of_one_pool_under_kernel_law_equals_reference():
    # pools = block + 1: the last block holds a single pool (prompts of
    # 16,388-16,391 tokens at SELECT_BLOCK 4096); on CPU kernel_v1 declines.
    rng = np.random.default_rng(5418)
    q, w, pooled = _score_operands(rng, 4, 129, ties=True)
    positions = mx.array([600, 601, 602, 603], dtype=mx.int32)
    ref = ix.blocked_select(q, w, pooled, positions, 64, block=128)
    for law in ("fused_v1", "kernel_v1"):
        got = ix.blocked_select(q, w, pooled, positions, 64, block=128, score=law)
        np.testing.assert_array_equal(np.array(got[0]), np.array(ref[0]))
        np.testing.assert_array_equal(np.array(got[1]), np.array(ref[1]))


# ---- Metal (MLX2_RUN_GPU_TESTS=1 only) ---------------------------------------

metal = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                           reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


@metal
def test_gpu_select_kernel_bitwise_and_pools_extend_bitwise():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        assert ix.select_kernel_self_check()
        assert ix.select_kernel_ready()
        rng = np.random.default_rng(48)
        ape = mx.array(rng.normal(0, 0.1, (4, 128)).astype(np.float32))
        rows = mx.array(_raw(rng, 1100, 4.0))
        cache = ix.Glm53PoolCache()
        for total in (2049, 2050, 2052, 4096, 4399):
            pooled = cache.extend(lambda a, b: rows[a:b], total, ape)
            assert ix.audit_pools(pooled, lambda a, b: rows[a:b], total, ape)
    finally:
        mx.set_default_device(previous)


@metal
def test_gpu_score_laws_bitwise_against_reference():
    """GPU only: fused_v1 by construction, kernel_v1 through its gate, at prompt shapes."""
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        assert ix.score_kernel_self_check() and ix.score_kernel_ready()
        rng = np.random.default_rng(5420)
        for length, pools in ((1, 2048), (128, 4096), (64, 4103), (128, 513), (128, 32), (9, 33),
                              (128, 1), (9, 7), (9, 31)):
            q, w, pooled = _score_operands(rng, length, pools, ties=pools > 8)
            ref = ix.pool_scores(q, w, pooled)
            before = ix._SCHEDULED.get("score_kernel", 0)
            for law in ("fused_v1", "kernel_v1"):
                got = ix.pool_scores(q, w, pooled, law=law)
                assert np.array_equal(_bits(got), _bits(ref)), (law, length, pools)
            launched = pools >= ix.SCORE_KERNEL_MIN_POOLS
            assert ix._SCHEDULED["score_kernel"] == before + int(launched), pools
        # Blocked prompt selection whose last block is one pool (4097 = 4096 + 1).
        q, w, pooled = _score_operands(rng, 128, 4097, ties=True)
        positions = mx.arange(16388 - 128, 16388, dtype=mx.int32)
        ref = ix.blocked_select(q, w, pooled, positions, 512, block=4096)
        got = ix.blocked_select(q, w, pooled, positions, 512, block=4096, score="kernel_v1")
        np.testing.assert_array_equal(np.array(got[0]), np.array(ref[0]))
        np.testing.assert_array_equal(np.array(got[1]), np.array(ref[1]))
    finally:
        mx.set_default_device(previous)
