"""CPU/static gates for the Gemma 4 exact-prefix strategy transfer."""

import pytest

from mlx2.adapters.gemma4 import Gemma4A4BAdapter, Gemma431BAdapter
from mlx2.runtime.exact_prefix_cascade import (
    longest_first_paths,
    next_cascade_stage,
)
from mlx2.runtime.prefill_plan import prompt_length_prefill_step


def _cache_type(name):
    return type(name, (), {"__module__": "mlx2.runtime.models.cache"})


KVCache = _cache_type("KVCache")
RotatingKVCache = _cache_type("RotatingKVCache")


def _cache_geometry(adapter, prefix_tokens):
    layers = adapter.exact_prefix_cascade_contract()["shared_prefix_reuse"]
    caches = []
    for index in range(
        layers["full_attention_layers"] + layers["sliding_attention_layers"]
    ):
        if index % 6 == 5:
            cache = KVCache()
        else:
            cache = RotatingKVCache()
            cache.max_size = 1024
            cache.keep = 0
        cache.offset = prefix_tokens
        caches.append(cache)
    return caches


def test_longest_first_is_stable_and_deduplicates_complete_paths():
    paths = [(1,), (2, 3, 4), (5, 6), (2, 3, 4), (7, 8)]
    assert longest_first_paths(paths) == (
        (2, 3, 4),
        (5, 6),
        (7, 8),
        (1,),
    )


def test_accepted_prefix_prunes_impossible_siblings_and_returns_only_suffix():
    paths = [(1, 2, 3, 4), (1, 2, 9), (1, 7, 8, 9), (6, 7)]
    first = next_cascade_stage(paths)
    assert first.path == (1, 2, 3, 4)
    assert first.suffix == first.path

    second = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert second.path == (1, 2, 9)
    assert second.suffix == (9,)
    assert second.viable_indices == (2,)
    assert set(second.pruned_indices) == {1, 3}
    assert next_cascade_stage(paths, (9,)) is None


@pytest.mark.parametrize(
    ("adapter_type", "variant", "layers", "full", "moe", "prefill"),
    [
        (Gemma4A4BAdapter, "26b-a4b", 30, 5, "adapter_owned", 2048),
        (Gemma431BAdapter, "31b", 60, 10, "not_present", 512),
    ],
)
def test_variant_contract_keeps_geometry_and_prefill_adapter_owned(
    adapter_type, variant, layers, full, moe, prefill
):
    adapter = object.__new__(adapter_type)
    contract = adapter.exact_prefix_cascade_contract()
    assert contract["variant"] == variant
    assert contract["verification_order"] == "longest_first"
    assert contract["invalid_sibling_pruning"] is True
    assert contract["accepted_prefix_state"] == (
        "exact_apcv2_restore_or_canonical_replay"
    )
    reuse = contract["shared_prefix_reuse"]
    assert reuse["authority"] == "apcv2"
    assert reuse["recompute_common_tokens"] is False
    assert reuse["full_attention_layers"] == full
    assert reuse["sliding_attention_layers"] == layers - full
    assert contract["moe_tensor_math"] == moe
    assert contract["transactional_multirow_state_reuse"] is False
    assert contract["qualified"] is contract["selected"] is False
    assert contract["observed_used"] is False
    assert adapter.prefill_step_default() == prefill


@pytest.mark.parametrize("adapter_type", [Gemma4A4BAdapter, Gemma431BAdapter])
def test_exact_cache_geometry_reuses_common_prefix_without_recompute(adapter_type):
    adapter = object.__new__(adapter_type)
    caches = _cache_geometry(adapter, 4096)
    decision = adapter.exact_shared_prefix_geometry(caches, 4096)
    assert decision == {
        "eligible": True,
        "authority": "apcv2_exact_restore",
        "prefix_tokens": 4096,
        "recompute_common_tokens": False,
        "suffix_only": True,
        "publishable": False,
    }

    caches[0].offset -= 1
    refused = adapter.exact_shared_prefix_geometry(caches, 4096)
    assert refused["reason"] == "cache_logical_offset_mismatch"


def test_exact_cache_geometry_refuses_wrong_plane_window_and_layer_count():
    adapter = object.__new__(Gemma4A4BAdapter)
    caches = _cache_geometry(adapter, 64)
    assert adapter.exact_shared_prefix_geometry(caches[:-1], 64)["reason"] == (
        "cache_layer_count_mismatch"
    )
    caches = _cache_geometry(adapter, 64)
    caches[0] = KVCache()
    caches[0].offset = 64
    assert adapter.exact_shared_prefix_geometry(caches, 64)["reason"] == (
        "cache_plane_type_mismatch"
    )
    caches = _cache_geometry(adapter, 64)
    caches[0].max_size = 512
    assert adapter.exact_shared_prefix_geometry(caches, 64)["reason"] == (
        "sliding_cache_geometry_mismatch"
    )


def test_adapter_planner_preserves_common_prefix_boundary():
    adapter = object.__new__(Gemma431BAdapter)
    stage = adapter.plan_exact_prefix_cascade(
        [(1, 2, 3), (1, 7), (1, 2, 9)], (1, 2), attempted=(0,)
    )
    assert stage.path == (1, 2, 9)
    assert stage.accepted_prefix == (1, 2)
    assert stage.suffix == (9,)


@pytest.mark.parametrize("paths", [[], [()], [(1, -1)], [(True, 2)]])
def test_invalid_cascade_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        longest_first_paths(paths)


def test_generic_autoscale_is_only_the_declining_adapter_fallback():
    assert prompt_length_prefill_step(32768) == 512
    assert prompt_length_prefill_step(32769) == 2048
    assert prompt_length_prefill_step(65537) == 8192
    assert Gemma4A4BAdapter.default_prefill_step == 2048
    assert Gemma431BAdapter.default_prefill_step == 512
