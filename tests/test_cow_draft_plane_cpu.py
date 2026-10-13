"""L2-03 (sweep 2026-10-12): an APCv2 COW branch keeps MTP-draft clones on the
inherited draft_mtp plane (e3c19168e let the nested cache contract override it,
so draft writes were counted as target attention_kv / gdn_recurrent)."""

import mlx.core as mx


def _kv(n):
    from mlx2.runtime.models.cache import KVCache

    cache = KVCache()
    k = mx.ones((1, 1, n, 4))
    cache.update_and_fetch(k, k)
    mx.eval(cache.state)
    return cache


def test_l2_03_cow_draft_clone_keeps_inherited_draft_mtp_plane():
    from mlx2.runtime.cow_cache import freeze_prompt_cache

    frozen, _sidecar = freeze_prompt_cache(
        [_kv(4)], key="k", tokens=range(4), cache_type="assistant",
        sidecar=[[_kv(3)]])
    owner = frozen.cow_owner
    branch = frozen.branch()
    draft_clone = branch.cow_sidecar[0][0]
    target_clone = branch[0]
    assert sorted(draft_clone._cow_segment_tokens) == ["draft_mtp"]
    assert sorted(target_clone._cow_segment_tokens) == ["attention_kv"]
    k = mx.ones((1, 1, 1, 4))
    draft_clone.update_and_fetch(k, k)  # a draft-only write
    planes = owner.telemetry.snapshot()["planes"]
    assert planes["draft_mtp"]["materializations"] == 1
    assert planes.get("attention_kv", {}).get("materializations", 0) == 0
