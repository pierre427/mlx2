"""Value retention keeps finished turns; pressure honours prefetch pins.

Two follow-ups to the 2026-10-07 reply-retention fix
(tests/test_apcv2_reply_retention_cpu.py):

* Under the opt-in value policy a finished turn started at a lower prior than
  a prompt boundary (0.3 vs 0.5) and was credited only for its reply tokens
  over its own prompt boundary, so it was the first eviction, before any
  stale boundary of another session.  The next agent round then re-prefilled
  the reply.  A finished turn now starts with a boundary's prior, and the
  boundary directly beneath it (which now only serves regenerations) drops to
  a regenerate prior, so pressure removes that boundary first.
* Pressure eviction ignored resident pins taken by a session prefetch.
  A live resident pin now takes an entry out of pressure eviction in either
  policy, like a lease.  A publication that cannot fit beside pinned entries
  is not cached until the pins expire (prefetch pins last 30 s by default and
  are capped per tenant).
"""

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache


def _state(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return [cache]


def _bytes_for(length):
    probe = APCv2(max_size=8, layout_name="probe")
    probe.store(APCKey("p"), list(range(length)), _state(length))
    nbytes = int(probe.nbytes)
    probe.clear(release_memory=False)
    return nbytes


def test_value_policy_keeps_the_finished_turn_over_old_boundaries():
    now = [0.0]
    key = APCKey("value-turn")
    old = [[10 * (i + 1)] * 8 for i in range(4)]
    prompt = list(range(100, 110))
    reply = list(range(110, 116))
    budget = sum(_bytes_for(len(t)) for t in old[1:]) + _bytes_for(16)
    apc = APCv2(max_size=64, max_bytes=budget, layout_name="value-turn-v1",
                now_fn=lambda: now[0], wall_time_fn=lambda: 1000.0 + now[0],
                retention_policy="value")
    for i, tokens in enumerate(old):
        now[0] = float(i)
        apc.store(key, tokens, _state(8, seed=i),
                  retention_role="committed_prompt_boundary")
    now[0] = 10.0
    apc.store(key, prompt, _state(len(prompt), seed=50),
              retention_role="committed_prompt_boundary")
    now[0] = 11.0
    apc.store(key, prompt + reply, _state(16, seed=60))
    hit = apc.lookup(key, prompt + reply + [200, 201])
    assert hit.hit and hit.cached_tokens == 16, (hit.cached_tokens, hit.retention_role)
    hit.cache.close()
    # The superseded boundary went first; it only served regenerations.
    with pytest.raises(KeyError):
        apc._trie.get(key, prompt)
    apc.clear(release_memory=False)


def test_a_regenerated_boundary_earns_its_value_back():
    now = [0.0]
    key = APCKey("value-regen")
    apc = APCv2(max_size=64, layout_name="value-regen-v1", now_fn=lambda: now[0],
                retention_policy="value")
    prompt = list(range(100, 110))
    apc.store(key, prompt, _state(10), retention_role="committed_prompt_boundary")
    now[0] = 1.0
    apc.store(key, prompt + [7, 8], _state(12, seed=5))
    boundary = apc._trie.get(key, prompt)
    assert apc._boundary_superseded_locked(key, prompt, boundary)
    value = lambda: apc._entry_retention_value_locked(  # noqa: E731
        key, prompt, boundary, now[0]
    )
    demoted = value()
    now[0] = 2.0
    hit = apc.lookup(key, prompt + [9])  # a regenerate diverges below the turn
    assert hit.cached_tokens == 10
    hit.cache.close()
    assert value() > demoted
    apc.clear(release_memory=False)


@pytest.mark.parametrize("policy", [None, "value"])
def test_pressure_keeps_a_prefetch_pinned_entry(policy):
    now = [0.0]
    key = APCKey("pins")
    budget = 3 * _bytes_for(8)
    apc = APCv2(max_size=64, max_bytes=budget, layout_name="pins-v1",
                now_fn=lambda: now[0], wall_time_fn=lambda: 1000.0 + now[0],
                retention_policy=policy)
    for i in range(3):
        now[0] = float(i)
        apc.store(key, [10 * (i + 1)] * 8, _state(8, seed=i),
                  retention_role="committed_prompt_boundary")
    pinned = apc._trie.get(key, [10] * 8)  # the oldest, first in line
    pinned._apc_resident_pin_expiries = {("tenant", "session"): 1000.0 + 3600}
    now[0] = 5.0
    apc.store(key, [99] * 8, _state(8, seed=9), retention_role="committed_prompt_boundary")
    assert apc._trie.get(key, [10] * 8).prompt_cache  # pin honoured
    with pytest.raises(KeyError):
        apc._trie.get(key, [20] * 8)  # the next oldest went instead
    # An expired pin no longer protects it.
    now[0] = 4000.0
    apc.store(key, [77] * 8, _state(8, seed=7), retention_role="committed_prompt_boundary")
    with pytest.raises(KeyError):
        apc._trie.get(key, [10] * 8)
    apc.clear(release_memory=False)


def test_live_pins_outrank_a_new_publication_until_they_expire():
    """With only pinned entries resident, the new state is not cached (the
    request itself is unaffected); once the pins expire it is."""
    wall = [1000.0]
    key = APCKey("pins-only")
    apc = APCv2(max_size=64, max_bytes=2 * _bytes_for(8), layout_name="pins-only-v1",
                wall_time_fn=lambda: wall[0])
    for i in range(2):
        apc.store(key, [10 * (i + 1)] * 8, _state(8, seed=i))
        apc._trie.get(key, [10 * (i + 1)] * 8)._apc_resident_pin_expiries = {
            ("t", str(i)): 1030.0
        }
    assert not apc.store(key, [99] * 8, _state(8, seed=9)).stored
    assert apc._trie.get(key, [10] * 8).prompt_cache
    wall[0] = 1031.0
    assert apc.store(key, [99] * 8, _state(8, seed=9)).stored
    with apc._apc_lock:
        assert apc._n_bytes <= apc.max_bytes
    apc.clear(release_memory=False)


def _superseded(apc, key, tokens):
    with apc._apc_lock:
        return apc._boundary_superseded_locked(key, tokens, apc._trie.get(key, tokens))


def test_supersession_follows_the_turn_that_backs_it():
    """Codex review: no stale state when the turn is rejected, dropped or the
    boundary is republished; found past an interior checkpoint."""
    key = APCKey("supersede")
    apc = APCv2(max_size=64, layout_name="supersede-v1", retention_policy="value")
    prompt = list(range(100, 110))
    apc.store(key, prompt, _state(10), retention_role="committed_prompt_boundary")
    apc.store(key, prompt + [1, 2], _state(12, seed=3), retention_role="interior_checkpoint")
    turn = prompt + [1, 2, 3, 4]
    apc.store(key, turn, _state(14, seed=5))
    assert _superseded(apc, key, prompt)  # found past the interior checkpoint
    apc.store(key, prompt, _state(10, seed=9), retention_role="committed_prompt_boundary")
    assert _superseded(apc, key, prompt)  # a republished boundary stays superseded
    with apc._apc_lock:
        apc._drop_entry_locked(key, turn, apc._trie.get(key, turn))
    assert not _superseded(apc, key, prompt)  # the turn is gone
    apc.clear(release_memory=False)


def test_a_rejected_turn_does_not_supersede():
    key = APCKey("supersede-reject")
    apc = APCv2(max_size=64, max_bytes=_bytes_for(10), layout_name="supersede-reject-v1",
                wall_time_fn=lambda: 1000.0, retention_policy="value")
    prompt = list(range(100, 110))
    apc.store(key, prompt, _state(10), retention_role="committed_prompt_boundary")
    apc._trie.get(key, prompt)._apc_resident_pin_expiries = {("t", "s"): 2000.0}
    assert not apc.store(key, prompt + [1, 2, 3], _state(13, seed=4)).stored
    assert not _superseded(apc, key, prompt)
    apc.clear(release_memory=False)


def test_a_boundary_shared_by_another_session_is_not_demoted():
    key = APCKey("supersede-shared")
    apc = APCv2(max_size=64, layout_name="supersede-shared-v1", retention_policy="value")
    prompt = list(range(100, 110))
    for tag in (("t", "a"), ("t", "b")):
        apc.store(key, prompt, _state(10), retention_role="committed_prompt_boundary",
                  session_tag=tag)
    apc.store(key, prompt + [1, 2], _state(12, seed=4), session_tag=("t", "a"))
    assert not _superseded(apc, key, prompt)  # session b still resumes from it
    apc.store(key, prompt + [7, 8], _state(12, seed=6), session_tag=("t", "b"))
    assert _superseded(apc, key, prompt)
    apc.clear(release_memory=False)


def test_count_pool_does_not_spill_a_pinned_entry(tmp_path):
    """Codex review: count enforcement reached pinned entries after the new
    publication was rejected and spilled a successful prefetch."""
    key = APCKey("pins-count")
    apc = APCv2(max_size=1, layout_name="pins-count-v1", idle_disk_seconds=3600,
                idle_disk_dir=str(tmp_path), wall_time_fn=lambda: 1000.0)
    apc.store(key, [10] * 8, _state(8), retention_role="committed_prompt_boundary")
    apc._trie.get(key, [10] * 8)._apc_resident_pin_expiries = {("t", "s"): 2000.0}
    apc.store(key, [20] * 8, _state(8, seed=2), retention_role="committed_prompt_boundary")
    assert apc._trie.get(key, [10] * 8).prompt_cache
    assert apc.apc_stats["idle_disk"]["pressure_spills"] <= 1  # only the new entry
    apc.clear(release_memory=False)
