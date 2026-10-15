"""Actual FOLD drafter state at cold, leased and durable continuation boundaries."""

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_external_dflash2_cpu import drain, generator, tiny

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.drafters.fold import FOLDKVCache, configure, status
from mlx2.runtime.sample_utils import LaneRNG


def _models():
    target, draft = tiny()
    configure(draft, {"window": 3, "alpha": 2, "sliding_layers": "fold"})
    return target, draft


def _cache_values(caches):
    return [
        tuple(np.asarray(value[..., : cache.offset, :]).copy() for value in cache.state)
        for cache in caches
    ]


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("sample", [False, True])
def test_fold_paired_continuation_survives_apc_lease_and_fresh_disk_manager(
    tmp_path, disk, sample
):
    target, draft = _models()
    source = generator(target, draft)
    config = {"sampling_temp": 0.8 if sample else 0.0}
    source.insert(
        [list(range(1, 14))],
        max_tokens=[5],
        sampling_configs=[config],
        lane_rngs=[LaneRNG(81)],
    )
    _, finished = drain(source)
    end = finished[0]
    assert status(draft)["observed_used"]
    assert all(isinstance(cache, FOLDKVCache) for cache in end.cache_sidecar.state[0])
    expected_planes = _cache_values(end.cache_sidecar.state[0])
    key = APCKey(
        "target+draft",
        revision="fold-revision",
        cache_layout_fingerprint="fold-lifecycle",
    )
    options = {
        "layout_name": "fold-lifecycle",
        "idle_disk_dir": str(tmp_path),
        "idle_disk_seconds": 1,
        "now_fn": lambda: 10.0,
        "persist_dir": str(tmp_path),
        "persist_identity": key,
    }
    apc = APCv2(**options)
    apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
    if disk:
        apc.close(persist_resident=True, time_budget_seconds=10, release_memory=False)
        apc = APCv2(**options)
    hit = apc.lookup(key, end.all_tokens + [end.token])
    assert hit.hit
    hit.sidecar.validate("test", len(end.all_tokens))
    actual_planes = _cache_values(hit.sidecar.state[0])
    for expected, actual in zip(expected_planes, actual_planes):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def continue_from(cache, state):
        batch = generator(target, draft)
        batch.insert(
            [[end.token]],
            max_tokens=[7],
            caches=[cache],
            all_tokens=[end.all_tokens],
            cache_states=[state],
            sampling_configs=[config],
            lane_rngs=[LaneRNG(999)],
            resume_rng=True,
        )
        try:
            tokens, terminal = drain(batch)
            return tokens[0], terminal[0].cache_sidecar
        finally:
            batch.close()

    try:
        expected, expected_state = continue_from(
            copy.deepcopy(end.prompt_cache), end.cache_sidecar
        )
        actual, actual_state = continue_from(hit.cache, hit.sidecar)
        assert actual == expected
        if not sample:
            cold = generator(target, draft)
            try:
                cold.insert([end.all_tokens + [end.token]], max_tokens=[7])
                assert drain(cold)[0][0] == actual
            finally:
                cold.close()
        np.testing.assert_array_equal(
            np.asarray(actual_state.rng_key), np.asarray(expected_state.rng_key)
        )
        assert actual_state.rng_draws == expected_state.rng_draws
        for expected, actual in zip(
            _cache_values(expected_state.state[0]), _cache_values(actual_state.state[0])
        ):
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
        # Both continuations must leave the published parent prefix immutable.
        for expected, actual in zip(
            expected_planes, _cache_values(hit.sidecar.state[0])
        ):
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
    finally:
        if hasattr(hit.cache, "close"):
            hit.cache.close()
        apc.close(release_memory=False)
        source.close()


@pytest.mark.parametrize("stage", ["_propose", "_sidecar"])
def test_fold_post_work_failure_restores_all_three_planes_and_rng(monkeypatch, stage):
    def run(fail):
        target, draft = _models()
        batch = generator(target, draft)
        batch.insert(
            [list(range(1, 12))],
            max_tokens=[4],
            sampling_configs=[{"sampling_temp": 0.8}],
            lane_rngs=[LaneRNG(37)],
        )
        lane = batch.lanes[0]
        while lane.anchor is None:
            batch._prefill(lane)
        if fail:
            before, rng, history = (
                _cache_values(lane.draft_cache),
                lane.rng.snapshot(),
                list(lane.history),
            )
            original = getattr(batch, stage)

            def injected(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("FOLD post-work failure")

            with monkeypatch.context() as patch:
                patch.setattr(batch, stage, injected)
                # A one-token terminal round reaches publication immediately.
                if stage == "_sidecar":
                    lane.maximum = 1
                with pytest.raises(RuntimeError, match="FOLD post-work"):
                    batch._round([lane])
                if stage == "_sidecar":
                    lane.maximum = 4
            assert lane.rng.snapshot() == rng and lane.history == history
            assert not lane.ready
            for expected, actual in zip(before, _cache_values(lane.draft_cache)):
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)
        try:
            return drain(batch)[0][0]
        finally:
            batch.close()

    assert run(True) == run(False)


def test_fold_configuration_refusal_is_atomic():
    _, draft = tiny()
    draft.layers[1].self_attn.sliding_window = 7
    with pytest.raises(ValueError, match="window must cover"):
        configure(draft, {"window": 3, "sliding_layers": "fold"})
    assert not any(
        getattr(layer.self_attn, "fold_policy", None) for layer in draft.layers
    )
    assert getattr(draft, "fold_policy", None) is None


@pytest.mark.parametrize("prefill", [False, True])
def test_fold_admission_reserves_new_third_plane_and_full_context_workspace(prefill):
    target, draft = _models()
    headroom = [10**12]
    batch = generator(target, draft, memory_headroom=lambda: headroom[0])
    batch.insert([list(range(1, 20))], max_tokens=[4])
    lane = batch.lanes[0]
    try:
        # Compare the same actual cache/target state with just the adapter's
        # additional allocation disabled; no model-name branch in admission.
        reserve = draft.extra_memory_reservation
        extra = reserve(lane.draft_cache, append=3, pending_tokens=0, prefill=prefill)
        assert extra > 0
        draft.extra_memory_reservation = lambda *args, **kwargs: 0
        assert batch._admit([lane], 3, prefill=prefill)
        baseline = batch.scheduler_stats["reservation_bytes"]
        draft.extra_memory_reservation = reserve
        headroom[0] = baseline + extra - 1
        assert not batch._admit([lane], 3, prefill=prefill)
        headroom[0] += 1
        assert batch._admit([lane], 3, prefill=prefill)
        assert batch.scheduler_stats["reservation_bytes"] == baseline + extra
    finally:
        batch.close()


def test_fold_pending_context_and_long_context_workspace_are_admitted():
    _, draft = _models()
    cache = draft.make_cache()
    cold = draft.extra_memory_reservation(
        cache, append=3, pending_tokens=0, prefill=False
    )
    pending = draft.extra_memory_reservation(
        cache, append=3, pending_tokens=300, prefill=False
    )
    assert pending > cold
    # Admission must cover actual new backing storage before it is allocated.
    keys = mx.ones((1, 1, 303, 4), mx.float32)
    for entry in cache:
        entry.append_raw_keys(keys)
        entry.update_and_fetch(keys, keys)
    newly_allocated = sum(entry.nbytes for entry in cache)
    assert pending + 2 * 3 * 1 * 4 * 4 * len(cache) >= newly_allocated
    workspace = draft.extra_memory_reservation(
        cache, append=0, pending_tokens=0, prefill=False
    )
    assert workspace > 0
    assert (
        draft.extra_memory_reservation(cache, append=0, pending_tokens=0, prefill=True)
        == 0
    )


@pytest.mark.parametrize("bad", [-1, True, 1.5, float("nan")])
def test_draft_extra_reservation_rejects_malformed_provider_before_allocation(bad):
    target, draft = _models()
    batch = generator(target, draft, memory_headroom=lambda: 10**12)
    batch.insert([[1, 2, 3]], max_tokens=[2])
    lane = batch.lanes[0]
    draft.extra_memory_reservation = lambda *args, **kwargs: bad
    try:
        with pytest.raises(ValueError, match="nonnegative bytes"):
            batch._admit([lane], 2, prefill=True)
        assert not lane.history
        assert all(cache.offset == 0 for cache in lane.draft_cache)
    finally:
        batch.close()


def test_fold_portfolio_reservation_includes_every_provider():
    from mlx2.runtime.drafter_portfolio import DrafterPortfolio, PortfolioProvider

    _, draft = _models()
    second = copy.deepcopy(draft)
    portfolio = DrafterPortfolio(
        (
            PortfolioProvider("first", "dflash2_fold", "a" * 64, draft),
            PortfolioProvider("second", "dflash2_fold", "b" * 64, second),
        )
    )
    cache = portfolio.make_cache()
    kwargs = {"append": 3, "pending_tokens": 17, "prefill": False}
    expected = sum(
        item.draft.extra_memory_reservation(
            cache[portfolio._cache_slices[item.provider_id]], **kwargs
        )
        for item in portfolio.providers
    )
    assert portfolio.extra_memory_reservation(cache, **kwargs) == expected > 0


@pytest.mark.parametrize("mode", ["default", "explicit", "continuation"])
def test_fold_reservation_survives_external_policy_wrappers(mode):
    from test_peer_fold_identity_cpu import _bind

    adapter = _bind(mode, {"window": 3, "sliding_layers": "fold"})
    draft = adapter.draft_model
    assert (
        draft.extra_memory_reservation(
            draft.make_cache(), append=3, pending_tokens=17, prefill=False
        )
        > 0
    )


def test_fold_prefill_context_uses_selected_unfused_key_norm(monkeypatch):
    from mlx import nn

    _, draft = _models()
    selected = {id(layer.self_attn.k_norm) for layer in draft.layers}
    stock = nn.RMSNorm.__call__

    def refuse_selected_fused(norm, values):
        if id(norm) in selected:
            raise RuntimeError("selected FOLD key norm entered fused primitive")
        return stock(norm, values)

    monkeypatch.setattr(nn.RMSNorm, "__call__", refuse_selected_fused)
    cache = draft.make_cache()
    hidden = mx.ones(
        (1, 7, draft.config.hidden_size * len(draft.config.target_layer_ids))
    )
    draft.append_context(hidden, cache)
    mx.eval([entry.state for entry in cache])
    assert all(entry.offset == 7 for entry in cache)


def test_fold_numerical_law_changes_paired_persistence_identity(monkeypatch):
    from test_peer_fold_identity_cpu import _bind

    from mlx2.runtime.drafters import fold

    policy = {"window": 3, "sliding_layers": "fold"}
    current = _bind("default", policy)
    ordinary = _bind("default", None)
    monkeypatch.setattr(fold, "FOLD_NUMERICS_VERSION", "old-inconsistent-append-law")
    previous = _bind("default", policy)
    assert current.identity["fingerprint"] != previous.identity["fingerprint"]
    assert current.layout != previous.layout
    assert ordinary.identity == _bind("default", None).identity


@pytest.mark.parametrize(
    "replacement",
    [
        None,
        False,
        {"window": 3, "sliding_layers": "window"},
        {"window": 3, "alpha": 3, "sliding_layers": "fold"},
    ],
)
def test_fold_live_reconfiguration_is_refused_without_stale_layer_selection(
    replacement,
):
    _, draft = _models()
    policy = draft.fold_policy
    draft.layers[0].self_attn.fold_calls = 17
    with pytest.raises(ValueError, match="fresh drafter"):
        configure(draft, replacement)
    assert draft.fold_policy is policy
    assert all(layer.self_attn.fold_policy is policy for layer in draft.layers)
    assert draft.layers[0].self_attn.fold_calls == 17
    assert configure(draft, policy.as_dict()) is policy
    assert draft.layers[0].self_attn.fold_calls == 17
