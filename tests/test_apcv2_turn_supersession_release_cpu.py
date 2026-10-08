"""The finished-turn supersession index must not outlive the turns it names.

d13979194 indexed every finished turn (default retention role) under its
prompt boundary in ``APCv2._turn_supersessions`` with a strong reference, on
either retention policy.  When the next agent round extends that turn, the
round's prompt boundary subsumes it as a disposable prefix: it leaves the trie
and the resident byte count, but the index kept the entry, its frozen cache
objects and their MLX buffers alive until the boundary itself was dropped --
including after the boundary was spilled to disk.  Under the default LRU
policy nothing reads the index, and under either policy a hot boundary's
record grew by one turn per regeneration, evicted turns included.
"""

import gc
import weakref

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache


def _state(length, seed=0, width=1):
    cache = KVCache()
    if width == 1:
        values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(
            1, 1, length, 1
        )
    else:
        values = mx.random.normal((1, 1, length, width), key=mx.random.key(seed))
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return [cache]


@pytest.mark.parametrize("policy", [None, "value"])
def test_a_subsumed_finished_turn_releases_its_state(policy):
    key = APCKey("turn-prune")
    apc = APCv2(max_size=64, layout_name="turn-prune-v1", retention_policy=policy)
    prompt = list(range(100, 110))
    turn = prompt + [1, 2, 3, 4]
    apc.store(key, prompt, _state(10), retention_role="committed_prompt_boundary")
    apc.store(key, turn, _state(14, seed=5))
    with apc._apc_lock:
        stored = apc._trie.get(key, turn)
    entry = weakref.ref(stored)
    plane = weakref.ref(stored.prompt_cache[0])
    del stored
    # The next round of a plain (sidecar-less, no session id) conversation
    # extends the finished turn; its prompt boundary subsumes the turn.
    apc.store(
        key, turn + [7, 8], _state(16, seed=9),
        retention_role="committed_prompt_boundary",
    )
    with apc._apc_lock, pytest.raises(KeyError):
        apc._trie.get(key, turn)
    with apc._apc_lock:
        assert apc._trie.get(key, prompt).prompt_cache  # the boundary stays
    gc.collect()
    assert plane() is None, "a pruned finished turn's KV is still reachable"
    assert entry() is None
    apc.clear(release_memory=False)


@pytest.mark.parametrize("policy", [None, "value"])
def test_regenerations_of_a_hot_prompt_do_not_grow_the_index(policy):
    key = APCKey("supersede-regen")
    apc = APCv2(max_size=4, layout_name="supersede-regen-v1", retention_policy=policy)
    prompt = list(range(1000, 1040))
    boundary = prompt[:-1]
    apc.store(
        key, boundary, _state(len(boundary)), retention_role="committed_prompt_boundary"
    )
    evicted = []
    for i in range(64):  # sampled regenerations: a distinct reply each time
        turn = prompt + [5000 + i] * 8
        apc.store(key, turn, _state(len(turn), seed=i))
        with apc._apc_lock:
            evicted.append(weakref.ref(apc._trie.get(key, turn)))
        hit = apc.lookup(key, prompt, allow_disk_restore=False)  # boundary stays hot
        if hit.cache is not None:
            hit.cache.close()
        del hit
    with apc._apc_lock:
        assert apc._trie.search(key, boundary).exact is not None
        records = len(apc._turn_supersessions.get((key, tuple(boundary)), {}))
    assert records <= apc.max_size, f"{records} turn records for a 4-entry cache"
    gc.collect()
    live = sum(ref() is not None for ref in evicted)
    assert live <= len(apc), f"{live} turn entries alive with {len(apc)} resident"
    apc.clear(release_memory=False)


@pytest.mark.parametrize("disk", [False, True])
def test_agent_rounds_keep_resident_memory_within_apc_accounting(disk, tmp_path):
    key = APCKey("agent-rounds")
    apc = APCv2(
        max_size=512, max_bytes=1 << 34, layout_name="agent-rounds-v1",
        idle_disk_dir=str(tmp_path) if disk else None,
        idle_disk_seconds=1 if disk else 0,
    )
    gc.collect()
    base = mx.get_active_memory()
    prompt = list(range(1000, 1200))
    for rnd in range(4):
        # The round's request leases its prefix (restoring a spilled turn),
        # as admission does, and releases it once its lane is built.
        hit = apc.lookup(key, prompt)
        if hit.cache is not None:
            hit.cache.close()
        del hit
        apc.store(
            key, prompt, _state(len(prompt), seed=10 * rnd, width=256),
            retention_role="committed_prompt_boundary",
        )
        turn = prompt + list(range(20000 + 300 * rnd, 20000 + 300 * rnd + 100))
        apc.store(key, turn, _state(len(turn), seed=10 * rnd + 1, width=256))
        if disk:
            apc.spill_idle_entries(now=apc._now() + 10_000 * (rnd + 1))
        prompt = turn + list(range(40000 + 100 * rnd, 40000 + 100 * rnd + 50))
    gc.collect()
    unaccounted = mx.get_active_memory() - base - apc.nbytes
    # One round's turn is ~1 MiB here; the leak held three pruned turns.
    assert unaccounted < (256 << 10), unaccounted
    apc.clear(release_memory=False)
