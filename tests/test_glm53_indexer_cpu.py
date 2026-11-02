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
