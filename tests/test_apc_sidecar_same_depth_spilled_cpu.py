"""A spilled draft checkpoint at ``len - 1`` beats a plain hit at ``len - 1``.

Maxwell-V1-Flash-oQ8e@da677814 at 16K under a 2 GiB APCv2 cap
(specround-phase0-rerun-20261126-maxwell-e23b79e61-a1): pressure spilled the
prompt-boundary entry, which carries the MTP sidecar, to the idle disk tier.
The warm repeat then matched the finished turn still resident, a sidecar-less
entry that trims to the same ``len - 1`` depth.  ``plain_wins`` let it win, so
the resident-only admission probe returned the plain hit, never asked
admission to restore the boundary, and the self-MTP route decoded the whole
answer as a target-only plain fallback.  A sidecar at the same depth resumes
the draft route at no extra prefill, so only a strictly deeper plain landing
may win.  When the restore has no room, the plain hit is still served: it
beats a cold prefill.
"""

import tempfile

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import KVCache


def _kv(length, seed=0):
    cache = KVCache()
    value = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def _sidecar(covered):
    return MTPAPCSidecar(
        ([_kv(covered - 1, seed=7)], mx.ones((1, 1, 4))), covered_tokens=covered
    )


def _boundary_spilled_behind_finished_turn(directory, *, spill=True):
    now = [0.0]
    apc = APCv2(
        max_size=8, layout_name="same-depth-spill", idle_disk_seconds=1,
        idle_disk_dir=directory, now_fn=lambda: now[0],
    )
    key = APCKey("m")
    prompt = list(range(200))
    boundary = len(prompt) - 1
    assert apc.store(
        key, prompt[:boundary], [_kv(boundary)], sidecar=_sidecar(boundary)
    ).stored
    if spill:
        now[0] = 100.0
        assert apc.spill_idle_entries(now=100.0) == 1
        assert not apc._trie.get(key, prompt[:boundary]).prompt_cache
    # The finished turn: prompt plus answer, stored without draft state.
    turn = prompt + list(range(1000, 1060))
    assert apc.store(key, turn[:-1], [_kv(len(turn) - 1)]).stored
    return apc, key, prompt, boundary


def test_resident_probe_defers_to_same_depth_disk_sidecar():
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt, boundary = _boundary_spilled_behind_finished_turn(directory)
        probe = apc.lookup(
            key, prompt, require_sidecar=True, allow_disk_restore=False
        )
        assert probe.cache is None
        assert probe.miss_reason == "disk_restore_requires_admission"
        apc.discard_lookup_credit(probe, "disk_restore_admission")
        hit = apc.lookup(key, prompt, require_sidecar=True)
        assert hit.sidecar is not None
        assert hit.cached_tokens == boundary
        hit.cache.close()


def test_restoring_lookup_serves_same_depth_disk_sidecar():
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt, boundary = _boundary_spilled_behind_finished_turn(directory)
        hit = apc.lookup(key, prompt, require_sidecar=True)
        assert hit.sidecar is not None and hit.cached_tokens == boundary
        hit.cache.close()


def test_multi_turn_deeper_disk_sidecar_beats_shallow_resident_ancestor():
    """Turn 1's boundary (100) is resident with a sidecar, a plain checkpoint
    (150) hides it from the trie candidates, turn 2's boundary (199) was
    spilled and turn 2's finished entry is resident.  The ancestor search
    returns the shallow resident ancestor while deferring the deeper disk
    one; only the deeper one may decide that the plain landing loses."""
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = APCv2(
            max_size=16, layout_name="same-depth-multi-turn", idle_disk_seconds=1,
            idle_disk_dir=directory, now_fn=lambda: now[0],
        )
        key = APCKey("m")
        prompt = list(range(200))
        boundary = len(prompt) - 1
        assert apc.store(
            key, prompt[:boundary], [_kv(boundary)], sidecar=_sidecar(boundary)
        ).stored
        now[0] = 100.0
        assert apc.spill_idle_entries(now=100.0) == 1
        assert apc.store(key, prompt[:100], [_kv(100)], sidecar=_sidecar(100)).stored
        turn = prompt + list(range(1000, 1060))
        assert apc.store(key, turn[:-1], [_kv(len(turn) - 1)]).stored
        assert apc.store(key, prompt[:150], [_kv(150)]).stored
        probe = apc.lookup(key, prompt, require_sidecar=True, allow_disk_restore=False)
        # Not the plain landing at 199: the shallow ancestor, with the deeper
        # disk checkpoint reported so admission restores it.
        assert probe.sidecar is not None and probe.cached_tokens == 100
        assert probe.deeper_disk_tokens == boundary
        probe.cache.close()
        apc.discard_lookup_credit(probe, "disk_restore_admission")
        hit = apc.lookup(key, prompt, require_sidecar=True)
        assert hit.sidecar is not None and hit.cached_tokens == boundary
        hit.cache.close()


def test_plain_hit_served_when_restore_has_no_budget(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt, boundary = _boundary_spilled_behind_finished_turn(directory)
        monkeypatch.setattr(apc, "_restore_entry_locked", lambda *a, **k: None)
        hit = apc.lookup(key, prompt, require_sidecar=True)
        assert hit.cache is not None and hit.sidecar is None
        assert hit.cached_tokens == boundary
        hit.cache.close()


def test_plain_hit_still_served_without_sidecar_anywhere():
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt, boundary = _boundary_spilled_behind_finished_turn(directory)
        apc._drop_entry_locked(
            key, prompt[:boundary], apc._trie.get(key, prompt[:boundary]),
            reason="test",
        )
        hit = apc.lookup(key, prompt, require_sidecar=True, allow_disk_restore=False)
        assert hit.cache is not None and hit.sidecar is None
        assert hit.cached_tokens == boundary
        hit.cache.close()


def test_plain_route_lookup_is_unchanged():
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt, boundary = _boundary_spilled_behind_finished_turn(directory)
        hit = apc.lookup(key, prompt, allow_disk_restore=False)
        assert hit.cache is not None and hit.sidecar is None
        assert hit.cached_tokens == boundary
        hit.cache.close()
