"""APCv2 ``pressure_drop_superseded``: skip disk writes a later turn covers.

On the Flash-Next service (2026-11-01) memory-admission evictions wrote every
agent turn to the disk tier as a whole new ~1 GB snapshot, almost all of it
the previous turn's: 1.33 TB written and 182 GB read back in 5.6 h.  With the
option on, a pressure victim whose path holds a stored descendant is dropped
instead.  CPU only.
"""

from __future__ import annotations

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import ArraysCache, KVCache

KEY = APCKey("pressure-drop-superseded")


def _kv(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _recurrent(length):
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32)
    cache.lengths = mx.array([length], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [length])
    mx.eval(cache.state)
    return cache


def _hybrid(length, seed=0):
    return [_recurrent(length), _kv(length, seed)]


def _sidecar(covered, seed=0):
    return MTPAPCSidecar(
        ([_kv(covered - 1, seed=seed)], mx.ones((1, 1, 4))), covered_tokens=covered
    )


def _paths(apc):
    with apc._apc_lock:
        return {
            tuple(tokens): (bool(entry.prompt_cache), bool(getattr(entry, "_apc_disk", None)))
            for _key, tokens, entry in apc._entry_records_locked()
        }


def _close(hit):
    close = getattr(hit.cache, "close", None)
    if callable(close):
        close()


def _apc(tmp_path, *, budget, drop):
    return APCv2(
        max_size=64, max_bytes=budget, layout_name="pressure-drop",
        idle_disk_seconds=180, idle_disk_dir=str(tmp_path),
        pressure_drop_superseded=drop,
    )


def _agent_loop(apc, turns=6, step=16):
    """Conversation B's leaf, then conversation A growing one turn at a time."""
    other = [9000 + i for i in range(32)]
    apc.store(KEY, other, _hybrid(32, seed=99))
    conversation = list(range(32))
    chain = []
    for turn in range(turns):
        tokens = conversation + [100 + turn * step + i for i in range(step * turn)]
        conversation = tokens
        chain.append(list(tokens))
        apc.store(KEY, tokens, _hybrid(len(tokens), seed=turn))
    return other, chain


def _budget():
    return 2 * sum(cache.nbytes for cache in _hybrid(32 + 16 * 15)) + 1


def test_off_by_default_spills_every_pressure_victim(tmp_path):
    apc = APCv2(
        max_size=64, max_bytes=_budget(), layout_name="pressure-drop",
        idle_disk_seconds=180, idle_disk_dir=str(tmp_path),
    )
    _agent_loop(apc)
    disk = apc.apc_stats["idle_disk"]
    assert disk["pressure_drop_superseded"] is False
    assert disk["pressure_superseded_drops"] == 0
    assert disk["pressure_spills"] >= 3
    apc.close()


def test_superseded_turns_are_dropped_unwritten_and_leaves_still_spill(tmp_path):
    baseline = _apc(tmp_path / "off", budget=_budget(), drop=False)
    _agent_loop(baseline)
    written_off = baseline.apc_stats["idle_disk"]["bytes_written"]
    baseline.close()

    apc = _apc(tmp_path / "on", budget=_budget(), drop=True)
    other, chain = _agent_loop(apc)
    disk = apc.apc_stats["idle_disk"]
    assert disk["pressure_superseded_drops"] >= 2
    assert disk["pressure_superseded_drop_bytes"] > 0
    assert disk["bytes_written"] < written_off
    paths = _paths(apc)
    # Every dropped entry was an ancestor of a stored turn; the other
    # conversation's leaf was spilled, not dropped.
    assert paths[tuple(other)] == (False, True)
    for tokens in chain[:-2]:
        assert tuple(tokens) not in paths
    # The next agent round still resumes from the newest turn, and the other
    # conversation still restores from disk at full depth.
    hit = apc.lookup(KEY, chain[-1] + [7])
    assert hit.cached_tokens == len(chain[-1])
    _close(hit)
    hit = apc.lookup(KEY, other + [7])
    assert hit.cached_tokens == len(other)
    _close(hit)
    assert apc.apc_stats["idle_disk"]["superseded_drop_regrets"] == 0
    apc.close()


def test_a_prompt_diverging_below_a_dropped_turn_counts_a_regret(tmp_path):
    apc = _apc(tmp_path, budget=_budget(), drop=True)
    _other, chain = _agent_loop(apc)
    dropped = chain[0]
    assert tuple(dropped) not in _paths(apc)
    hit = apc.lookup(KEY, dropped + [424242])
    _close(hit)
    disk = apc.apc_stats["idle_disk"]
    assert disk["superseded_drop_regrets"] == 1
    assert disk["superseded_drop_regret_tokens"] == len(dropped) - hit.cached_tokens
    # An admission probe without disk restore is not counted again.
    _close(apc.lookup(KEY, dropped + [424242], allow_disk_restore=False))
    assert apc.apc_stats["idle_disk"]["superseded_drop_regrets"] == 1
    apc.close()


def test_entries_a_descendant_cannot_replace_are_still_spilled(tmp_path):
    one = sum(cache.nbytes for cache in _hybrid(64))
    apc = _apc(tmp_path, budget=one + one // 2, drop=True)
    base = list(range(32))
    # A next-turn boundary, an interior checkpoint and a sidecar-carrying turn
    # each sit below a deeper sidecar-less finished turn.
    apc.store(KEY, base, _hybrid(32, seed=1), retention_role="committed_prompt_boundary",
              next_turn_boundary=True)
    apc.store(KEY, base + [1], _hybrid(33, seed=2), retention_role="interior_checkpoint")
    apc.store(KEY, base + [1, 2], _hybrid(34, seed=3), sidecar=_sidecar(34, seed=4))
    apc.store(KEY, base + [1, 2, 3] + list(range(500, 530)), _hybrid(63, seed=5))
    paths = _paths(apc)
    disk = apc.apc_stats["idle_disk"]
    assert disk["pressure_superseded_drops"] == 0
    for tokens in (base, base + [1], base + [1, 2]):
        assert paths[tuple(tokens)] == (False, True)
    apc.close()


def test_selection_takes_the_policy_unless_the_environment_names_one():
    import pytest

    from mlx2.serving import apc_pressure_drop_superseded_selection as select

    assert select(False) == {"enabled": False, "source": "policy"}
    assert select(True) == {"enabled": True, "source": "policy"}
    assert select(True, "0") == {"enabled": False, "source": "environment"}
    assert select(False, "1") == {"enabled": True, "source": "environment"}
    with pytest.raises(ValueError):
        select("yes")
    with pytest.raises(ValueError):
        select(True, "true")
