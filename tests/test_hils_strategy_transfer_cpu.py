"""Host-only gates for the HiLS exact-prefix strategy transfer."""

import pytest

from mlx2.adapters.olmo_hils import LIVE_CACHE_LAYOUT, OlmoHiLSAdapter
from mlx2.runtime.exact_prefix_cascade import (
    longest_first_paths,
    next_cascade_stage,
)


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


def _cache_type(name):
    return type(name, (), {"__module__": "mlx2.runtime.models.olmo_hils"})


HiLSCache = _cache_type("HiLSCache")
SWABandCache = _cache_type("SWABandCache")


def _adapter():
    adapter = object.__new__(OlmoHiLSAdapter)
    adapter.identity = {"fingerprint": "a" * 64}
    adapter.config = {
        "num_hidden_layers": 32,
        "chunk_size": 64,
        "hils_sliding_window": 512,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "full_attn_interleave": 4,
    }
    return adapter


def _caches(position=63):
    inserted = position + position // 63
    pooled = inserted // 64
    caches = []
    for index in range(32):
        hils = index % 4 == 3
        cache = (HiLSCache if hils else SWABandCache)()
        cache.offset = position
        cache.ins_offset = inserted
        cache.chunk_size = 64
        cache.start_pos = 0
        cache.keys = _Tensor((1, 32, inserted, 128))
        cache.values = _Tensor((1, 32, inserted, 128))
        if hils:
            cache.num_pooled_chunks = pooled
            cache.lmk_k = _Tensor((1, pooled, 32, 128)) if pooled else None
            cache.prior_b = _Tensor((1, pooled, 32)) if pooled else None
        else:
            cache.window = 512
        caches.append(cache)
    return caches


def _binding(prefix=(10, 11), **updates):
    value = {
        "execution_domain": "ordinary_s1",
        "cache_layout": LIVE_CACHE_LAYOUT,
        "state_revision": "a" * 64,
        "checkpoint_kind": "live_authoritative_exact",
        "checkpoint_position": 63,
        "transcript_tail": prefix,
        "rng_state": "authoritative_after_prefix",
        "batch_size": 1,
        "caches": _caches(),
    }
    value.update(updates)
    return value


def test_hils_prefill_default_is_one_integral_landmark_window():
    assert _adapter().prefill_step_default() == 504
    assert 504 + 504 // 63 == 512


def test_shared_planner_is_stable_longest_first_and_prefix_bounded():
    paths = longest_first_paths([(1,), (1, 2, 3, 4), (1, 2, 9), (1, 7)])
    assert paths == ((1, 2, 3, 4), (1, 2, 9), (1, 7), (1,))
    stage = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert stage.path == (1, 2, 9)
    assert stage.accepted_prefix == (1, 2)
    assert stage.suffix == (9,)
    assert set(stage.pruned_indices) == {2, 3}


def test_adapter_plans_suffix_only_from_bound_live_ordinary_state():
    adapter = _adapter()
    paths = [(10, 11, 99, 98), (10, 11, 12), (10, 88, 77)]
    stage = adapter.plan_exact_prefix_cascade(
        paths, (10, 11), attempted=(0,), state_binding=_binding()
    )
    assert stage.path == (10, 11, 12)
    assert stage.suffix == (12,)
    assert stage.pruned_indices == (2,)
    geometry = adapter.exact_shared_prefix_geometry(_caches(), 63)
    assert geometry == {
        "eligible": True,
        "authority": "live_authoritative_ordinary_s1",
        "cache_layout": LIVE_CACHE_LAYOUT,
        "checkpoint_position": 63,
        "inserted_position": 64,
        "recompute_common_tokens": False,
        "suffix_only": True,
        "rollback_authorized": False,
        "multirow_verify": False,
        "publishable": False,
    }


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"execution_domain": "multirow"}, "execution domain"),
        ({"cache_layout": "other"}, "cache layout"),
        ({"state_revision": "b" * 64}, "state revision"),
        ({"checkpoint_kind": "restored"}, "checkpoint kind"),
        ({"checkpoint_position": 1}, "checkpoint position"),
        ({"transcript_tail": (10, 9)}, "transcript tail"),
        ({"rng_state": "unknown"}, "RNG state"),
        ({"batch_size": 2}, "batch size"),
    ],
)
def test_shared_prefix_binding_fails_closed(updates, match):
    with pytest.raises(ValueError, match=match):
        _adapter().plan_exact_prefix_cascade(
            [(10, 11, 99), (10, 11, 12)],
            (10, 11),
            attempted=(0,),
            state_binding=_binding(**updates),
        )


def test_cache_geometry_rejects_inserted_pool_and_swa_drift():
    adapter = _adapter()
    caches = _caches()
    caches[0].ins_offset -= 1
    assert adapter.exact_shared_prefix_geometry(caches, 63)["reason"] == (
        "inserted_coordinate_mismatch"
    )
    caches = _caches()
    caches[3].num_pooled_chunks = 0
    assert adapter.exact_shared_prefix_geometry(caches, 63)["reason"] == (
        "landmark_pool_geometry_mismatch"
    )
    caches = _caches()
    caches[0].window = 256
    assert adapter.exact_shared_prefix_geometry(caches, 63)["reason"] == (
        "sliding_window_geometry_mismatch"
    )


def test_hils_contract_never_promotes_the_planner_to_a_serving_route():
    contract = _adapter().exact_prefix_cascade_contract()
    assert contract["implemented"] is True
    assert contract["qualified"] is False
    assert contract["selected"] is False
    assert contract["observed_used"] is False
    assert contract["transactional_multirow_state_reuse"] is False
    assert contract["apcv2_publication"] is False
    diagnostics = _adapter().diagnostics()
    assert diagnostics["route"] == "ordinary"
    assert diagnostics["prefix_candidate_verification"] == contract
