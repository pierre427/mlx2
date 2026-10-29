"""Synthetic CPU KV/index contracts, never model qualification or GPU evidence."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v41 as v41
from test_deepseek_v41_cpu import config, Weights


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    def forbidden(*args, **kwargs):
        raise AssertionError("KV CPU oracle reached a Metal kernel")
    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    yield
    mx.set_default_device(old)


def _selector(count, *, policy="reference", ratio=1, layer=20, cap=2048,
              length=3, start=None, nonfinite=False):
    c = SimpleNamespace(index_topk=512, index_n_heads=2, index_head_dim=4,
                        candidate_source_layer_id=20, candidate_block_size=8,
                        candidate_topk_blocks=cap)
    m = SimpleNamespace(index_policy=policy, scheduled_kv_calls={})
    a = object.__new__(v41.Attention)
    a.c, a.model, a.layer, a.ratio = c, m, layer, ratio
    rng = np.random.default_rng(123)
    query = mx.array(rng.normal(size=(1, length, 8)).astype(np.float32))
    weight = mx.full((1, length, 2), -mx.inf if nonfinite else .25, mx.float32)
    calls = []
    def linear(x, name):
        calls.append(name)
        if name == "indexer.attn_q_b":
            return mx.ones(query.shape, mx.float32) if nonfinite else query
        return weight
    a.linear = linear
    a.rotate_quantized = lambda x, *args, **kwargs: x
    key = mx.ones((1, count, 4), mx.float32) if nonfinite else mx.array(rng.normal(size=(1, count, 4)).astype(np.float32))
    source = SimpleNamespace(index_keys=key)
    shared = {}
    if layer > 20:
        shared["candidates"] = mx.ones((1, length, count), mx.bool_)
    x = mx.zeros((1, length, 8), mx.float32)
    qr = mx.zeros((1, length, 4), mx.float32)
    start = max(0, count * ratio - length) if start is None else start
    return a, source, shared, x, qr, start, calls


@pytest.mark.parametrize("count", [0, 1, 2, 63, 64, 511, 512, 513])
@pytest.mark.parametrize("layer,ratio", [(2, 2), (20, 1), (24, 1)])
def test_finite_selector_matches_ids_masks_and_candidates(count, layer, ratio):
    args = dict(ratio=ratio, layer=layer)
    r, rs, ref, x, qr, start, rc = _selector(count, **args)
    f, fs, fast, _, _, _, fc = _selector(count, policy="finite_all_retained", **args)
    r.select(x, qr, rs, start, ref)
    f.select(x, qr, fs, start, fast)
    for key in ("indices", "selected_mask", "candidates"):
        assert (key in ref) == (key in fast)
        if key in ref:
            assert ref[key].dtype == fast[key].dtype
            np.testing.assert_array_equal(np.array(ref[key]), np.array(fast[key]))
    assert fc == ([] if count <= 512 else rc)
    assert fast["_index_generation"] == 1


@pytest.mark.parametrize("start,length,count", [(0, 1, 0), (0, 3, 1), (1, 1, 1), (2, 3, 2), (3, 2, 2)])
def test_ratio_two_odd_partial_causal_visibility(start, length, count):
    a, source, shared, x, qr, _, calls = _selector(count, policy="finite_all_retained", ratio=2, layer=2, length=length, start=start)
    a.select(x, qr, source, start, shared)
    expected = np.arange(count)[None, None] < ((np.arange(start, start+length)+1)//2)[None, :, None]
    np.testing.assert_array_equal(np.array(shared["selected_mask"]), expected)
    assert not calls


@pytest.mark.parametrize("width", [1, 7, 8, 9, 511, 512, 16384, 16385])
def test_candidate_block_shortcut_preserves_partial_block_mask(width):
    lens = np.array([0, max(1, width//2), width], np.int32)[None, :, None]
    rng = np.random.default_rng(width)
    score = rng.normal(size=(1, 3, width)).astype(np.float32)
    score = np.where(np.arange(width)[None, None] < lens, score, -np.inf)
    args = (mx.array(score), mx.array(lens), 2048, 8)
    ref = v41.candidate_blocks(*args)
    fast = v41.candidate_blocks(*args, finite_all_retained=True)
    np.testing.assert_array_equal(np.array(ref), np.array(fast))
    if width <= 16384:
        expected = (np.arange(width)//8*8)[None, None] < lens
        np.testing.assert_array_equal(np.array(fast), expected)


def test_pruning_candidate_source_keeps_score_path_even_when_topk_retains_all():
    a, source, shared, x, qr, start, calls = _selector(17, policy="finite_all_retained", cap=1)
    a.select(x, qr, source, start, shared)
    assert calls == ["indexer.attn_q_b", "indexer.proj"]
    assert "all_retained_index" not in a.model.scheduled_kv_calls


def test_missing_prior_candidates_fails_closed():
    a, source, shared, x, qr, start, _ = _selector(5, policy="finite_all_retained", layer=24)
    del shared["candidates"]
    with pytest.raises(ValueError, match="candidate source"):
        a.select(x, qr, source, start, shared)
    assert "selected_mask" not in shared


def test_reference_keeps_nonfinite_score_behavior_finite_profile_is_explicit():
    a, source, shared, x, qr, start, calls = _selector(4, nonfinite=True)
    a.select(x, qr, source, start, shared)
    assert calls and not np.array(shared["selected_mask"]).any()
    b, bs, fast, _, _, _, fc = _selector(4, nonfinite=True, policy="finite_all_retained")
    b.select(x, qr, bs, start, fast)
    # The finite-only profile deliberately does not inspect scores it omits.
    assert not fc and np.array(fast["selected_mask"]).any()
    assert a.model.index_policy == "reference"


def _owner(count=5, dim=8, *, offset=None):
    pool = mx.arange(count*dim, dtype=mx.float32).reshape(1, count, dim)
    return SimpleNamespace(_pooled=pool, pooled=pool, _pool_count=count,
                           offset=count if offset is None else offset)


def _selection(count=5):
    return mx.arange(count, dtype=mx.int32)[None, None], mx.ones((1, 1, count), mx.bool_)


def test_decode_gather_reuses_only_same_live_owner_selection_and_generation(monkeypatch):
    original = v41._selected_rows
    calls = []
    def gather(pool, ids):
        calls.append((pool, ids))
        return original(pool, ids)
    monkeypatch.setattr(v41, "_selected_rows", gather)
    owner = _owner()
    ids, valid = _selection()
    shared, counters = {"_index_generation": 1}, {}
    first = v41.decode_selected_kv(owner, ids, valid, shared, counters=counters)
    assert v41.decode_selected_kv(owner, ids, valid, shared, counters=counters) is first
    assert len(calls) == 1 and counters == {"decode_gather_build": 1, "decode_gather_hit": 1}
    for change in ("generation", "indices", "mask", "offset", "backing", "owner"):
        if change == "generation": shared["_index_generation"] += 1
        elif change == "indices": ids = mx.asarray(ids, copy=True)
        elif change == "mask": valid = mx.asarray(valid, copy=True)
        elif change == "offset": owner.offset += 1
        elif change == "backing": owner._pooled = owner.pooled = mx.asarray(owner.pooled, copy=True)
        else: owner = _owner()
        before = len(calls)
        v41.decode_selected_kv(owner, ids, valid, shared)
        assert len(calls) == before + 1


def test_gather_growth_cannot_reuse_old_pool_prefix():
    owner = _owner(5)
    ids, valid = _selection()
    shared = {}
    first = v41.decode_selected_kv(owner, ids, valid, shared)
    owner._pooled = owner.pooled = mx.concatenate((owner.pooled, mx.full((1, 1, 8), 999, mx.float32)), 1)
    owner._pool_count = owner.offset = 6
    second = v41.decode_selected_kv(owner, ids, valid, shared)
    assert second is not first
    np.testing.assert_array_equal(np.array(second), np.array(first))
    ids = mx.array([[[5]]], mx.int32)
    valid = mx.ones((1, 1, 1), mx.bool_)
    np.testing.assert_array_equal(np.array(v41.decode_selected_kv(owner, ids, valid, shared)), np.full((1, 1, 1, 8), 999))


def test_failed_gather_removes_stale_entry_and_remains_forward_local(monkeypatch):
    owner = _owner()
    ids, valid = _selection()
    shared = {}
    v41.decode_selected_kv(owner, ids, valid, shared)
    def cancelled(*args): raise RuntimeError("cancelled gather")
    monkeypatch.setattr(v41, "_selected_rows", cancelled)
    shared["_index_generation"] = 9
    with pytest.raises(RuntimeError, match="cancelled"):
        v41.decode_selected_kv(owner, ids, valid, shared)
    assert "_decode_selected_kv" not in shared
    assert not hasattr(owner, "_decode_selected_kv")


@pytest.mark.parametrize("malformed", ["width", "ids_dtype", "mask_dtype", "mask_shape", "too_many"])
def test_decode_gather_refuses_wrong_geometry(malformed):
    owner = _owner()
    ids, valid = _selection()
    if malformed == "width": ids, valid = mx.repeat(ids, 2, 1), mx.repeat(valid, 2, 1)
    elif malformed == "ids_dtype": ids = ids.astype(mx.float32)
    elif malformed == "mask_dtype": valid = valid.astype(mx.int32)
    elif malformed == "mask_shape": valid = valid[:, :, :2]
    else: ids, valid = mx.zeros((1, 1, 6), mx.int32), mx.ones((1, 1, 6), mx.bool_)
    with pytest.raises(ValueError, match="width-one"):
        v41.decode_selected_kv(owner, ids, valid, {})


def test_reused_gather_preserves_stock_attention_bits():
    rng = np.random.default_rng(27)
    owner = _owner()
    ids, valid = _selection()
    q = mx.array(rng.normal(size=(1, 2, 1, 8)).astype(np.float32))
    local = mx.array(rng.normal(size=(1, 1, 4, 8)).astype(np.float32))
    mask, sinks = mx.ones((1, 1, 1, 4), mx.bool_), mx.array([.25, -.75])
    expected = v41.selected_attention(q, local, mask, owner.pooled, ids, valid, sinks)
    gathered = v41.decode_selected_kv(owner, ids, valid, {})
    actual = v41.selected_attention(q, local, mask, owner.pooled, ids, valid, sinks, gathered=gathered)
    np.testing.assert_array_equal(np.array(actual).view(np.uint32), np.array(expected).view(np.uint32))
    with pytest.raises(ValueError, match="reused gather"):
        v41.selected_attention(mx.repeat(q, 2, 2), local, mx.repeat(mask, 2, 2), owner.pooled,
                               mx.repeat(ids, 2, 1), mx.repeat(valid, 2, 1), sinks, gathered=gathered)


@pytest.mark.parametrize("kwargs", [{"index_policy": "bad"}, {"selected_kv_policy": "bad"}])
def test_unknown_policy_refused(kwargs):
    c = config()
    with pytest.raises(ValueError, match="index or selected-KV"):
        v41.DeepSeekV41Model(c, Weights(c), identity="synthetic@kv", **kwargs)


@pytest.mark.parametrize("block_cap", [1, 2048])
def test_forward_exact_state_replay_and_profile_isolation(monkeypatch, block_cap):
    c = config(index_topk=512, candidate_topk_blocks=block_cap)
    ref = v41.DeepSeekV41Model(c, Weights(c), identity="synthetic@kv")
    fast = v41.DeepSeekV41Model(c, Weights(c), identity="synthetic@kv", index_policy="finite_all_retained", selected_kv_policy="decode_reuse")
    assert ref.identity != fast.identity
    rc, fc = ref.make_cache(), fast.make_cache()
    for tokens in ([1, 2, 3], [4], [5, 6], [7]):
        ids = mx.array([tokens], mx.int32)
        a, b = ref(ids, rc), fast(ids, fc)
        np.testing.assert_array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))
        for rs, fs in zip(rc, fc):
            for x, y in zip(rs.state, fs.state):
                if x is not None: np.testing.assert_array_equal(np.array(x), np.array(y))
    snapshot = [s.snapshot() for s in fc]
    a = fast(mx.array([[8]], mx.int32), fc)
    for s, saved in zip(fc, snapshot): s.restore(saved)
    b = fast(mx.array([[8]], mx.int32), fc)
    np.testing.assert_array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))
    with pytest.raises(ValueError, match="identity"):
        fast(mx.array([[8]], mx.int32), rc)
    assert fast.scheduled_kv_calls["all_retained_index"] > 0
    assert fast.scheduled_kv_calls["decode_gather_build"] > 0
    assert not any("gather" in k or "generation" in k for s in fc for k in vars(s))


def test_fast_selection_has_no_explicit_eval_readback_or_scoring(monkeypatch):
    a, source, shared, x, qr, start, _ = _selector(512, policy="finite_all_retained")
    def forbidden(*args, **kwargs): raise AssertionError("unexpected scoring/evaluation")
    monkeypatch.setattr(mx, "eval", forbidden)
    monkeypatch.setattr(mx, "einsum", forbidden)
    monkeypatch.setattr(v41, "stable_pool_topk", forbidden)
    a.select(x, qr, source, start, shared)
    assert shared["indices"].shape == (1, 3, 512)


def test_index_refresh_discards_old_gather_before_any_failure():
    a, source, shared, x, qr, start, _ = _selector(513)
    shared["_decode_selected_kv"] = {"old": True}
    shared["indices"], shared["selected_mask"] = "old", "old"
    def cancelled(*args, **kwargs): raise RuntimeError("cancelled projection")
    a.linear = cancelled
    with pytest.raises(RuntimeError, match="cancelled projection"):
        a.select(x, qr, source, start, shared)
    assert not {"_decode_selected_kv", "indices", "selected_mask"} & shared.keys()


def test_real_cache_growth_511_512_513_invalidates_same_object_and_new_backing():
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    owner = DeepSeekV41LayerCache(1, xp=mx, identity="synthetic@kv-growth", source=True,
                                  history=False, head_dim=8, index_head_dim=4)
    owner.offset = 511
    owner.append_pools(mx.arange(511*8, dtype=mx.float32).reshape(1, 511, 8), mx.zeros((1, 511, 4), mx.float32))
    ids, valid = _selection(511)
    shared = {}
    before = v41.decode_selected_kv(owner, ids, valid, shared)
    backing = owner._pooled
    for frontier in (512, 513):
        owner.offset = frontier
        owner.append_pools(mx.full((1, 1, 8), frontier, mx.float32), mx.zeros((1, 1, 4), mx.float32))
        after = v41.decode_selected_kv(owner, ids, valid, shared)
        assert after is not before
        np.testing.assert_array_equal(np.array(before), np.array(after))
        assert (owner._pooled is backing) == (frontier == 512)
        before = after
    assert owner._capacity == 768


@pytest.mark.parametrize("failure", [None, "enter", "body", "exit"])
def test_execution_scope_cleans_up_and_preserves_forward_counter(failure):
    from contextlib import contextmanager
    events = []
    @contextmanager
    def scope():
        events.append("enter")
        if failure == "enter": raise RuntimeError("enter failed")
        try:
            yield
        finally:
            events.append("exit")
            if failure == "exit": raise RuntimeError("exit failed")
    def body(*args, **kwargs):
        events.append("body")
        if failure == "body": raise RuntimeError("body failed")
        return "result"
    model = SimpleNamespace(_weight_executions=0,
                            weights=SimpleNamespace(execution_scope=scope),
                            _forward_impl=body)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            v41.DeepSeekV41Model._forward(model, None)
    else:
        assert v41.DeepSeekV41Model._forward(model, None) == "result"
    assert model._weight_executions == 0
    assert events == (["enter"] if failure == "enter" else ["enter", "body", "exit"])


def test_no_execution_hook_preserves_native_weight_behavior():
    model = SimpleNamespace(_weight_executions=0, weights=SimpleNamespace(),
                            _forward_impl=lambda *args, **kwargs: "native")
    assert v41.DeepSeekV41Model._forward(model, None) == "native"
    assert model._weight_executions == 0
