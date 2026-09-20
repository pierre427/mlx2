# SPDX-License-Identifier: Apache-2.0
"""The shard-streaming loader's ordering invariant, on CPU.

The bug this pins: evicting a file's UBC mirror only after every shard has
been materialised makes peak memory ``weights + a full mirror of the same
bytes``. The fix evicts per shard -- but eviction must *follow* that shard's
materialisation. Evicting a file whose tensors are still lazy throws away a
cache the next ``mx.eval`` has to refill from disk, which is a pessimisation
dressed up as a saving, so the ordering is the property worth pinning.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime import ubc_evict as ubc


@pytest.fixture
def shards(tmp_path):
    """Three small safetensors shards, two tensors each."""
    files = []
    for index in range(3):
        target = tmp_path / f"shard-{index}.safetensors"
        mx.save_safetensors(
            str(target),
            {
                f"block.{index}.weight": mx.full((64, 64), float(index)),
                f"block.{index}.bias": mx.full((64,), float(index)),
            },
        )
        files.append(target)
    return files


@pytest.fixture
def trace(monkeypatch):
    """Record eviction calls and whether each path's tensors were live."""
    events: list[tuple[str, str]] = []
    real_eval = mx.eval

    def spy_eval(*args, **kwargs):
        events.append(("eval", ""))
        return real_eval(*args, **kwargs)

    def spy_evict(paths):
        for path in paths:
            events.append(("evict", str(path)))
        return 0

    monkeypatch.setattr(mx, "eval", spy_eval)
    monkeypatch.setattr(ubc, "ubc_evict_paths", spy_evict)
    return events


def test_every_eviction_follows_a_materialisation(shards, trace):
    """Eviction never precedes the eval that fills the cache it discards."""
    ubc.load_shards_evicting(shards)

    assert [kind for kind, _ in trace] == ["eval", "evict"] * 3, trace
    for position, (kind, _) in enumerate(trace):
        if kind == "evict":
            assert trace[position - 1][0] == "eval", (
                f"eviction at {position} precedes its materialisation: {trace}"
            )


def test_each_shard_is_evicted_before_the_next_is_read(shards, trace):
    """The mirror is bounded by one shard, not by the whole checkpoint."""
    ubc.load_shards_evicting(shards)

    evicted = [path for kind, path in trace if kind == "evict"]
    assert evicted == [str(f) for f in shards]


def test_values_match_a_plain_load(shards):
    """Streaming changes when bytes are read, never what they are."""
    streamed = ubc.load_shards_evicting(shards)

    plain: dict = {}
    for shard in shards:
        plain.update(mx.load(str(shard)))

    assert sorted(streamed) == sorted(plain)
    for key, value in plain.items():
        assert mx.array_equal(streamed[key], value), key


def test_sanitize_runs_before_materialisation(shards, trace):
    """Dropped tensors are never read: that is the point of the hook."""
    seen: list[int] = []

    def sanitize(shard):
        seen.append(len(shard))
        return {k: v for k, v in shard.items() if not k.endswith(".bias")}

    weights = ubc.load_shards_evicting(shards, sanitize=sanitize)

    assert seen == [2, 2, 2]
    assert all(not k.endswith(".bias") for k in weights)
    assert len(weights) == 3
    # Pruning does not stop the file being evictable: nothing is left lazy.
    assert [kind for kind, _ in trace] == ["eval", "evict"] * 3


def test_kept_lazy_tensors_defer_their_file_s_eviction(shards, trace):
    """A file with unread tensors must not be evicted yet."""
    deferred: list[str] = []

    weights = ubc.load_shards_evicting(
        shards,
        keep_lazy=lambda name: name.endswith(".bias"),
        deferred=deferred,
    )

    assert deferred == [str(f) for f in shards]
    assert [kind for kind, _ in trace] == ["eval", "eval", "eval"]
    assert len(weights) == 6


def test_keep_lazy_without_a_deferred_list_is_refused(shards):
    """Silently dropping the held paths would leak the mirror we came to kill."""
    with pytest.raises(ValueError, match="deferred"):
        ubc.load_shards_evicting(shards, keep_lazy=lambda name: True)


def test_a_fully_lazy_shard_is_not_evaluated_at_all(shards, trace):
    """No tensors to materialise means no eval and no eviction."""
    deferred: list[str] = []

    ubc.load_shards_evicting(
        shards, keep_lazy=lambda name: True, deferred=deferred
    )

    assert trace == []
    assert deferred == [str(f) for f in shards]
