"""A finished turn's entry must survive the count pool long enough to be reused.

Agent tool rounds send the previous prompt plus the previous reply plus a tool
result.  On a hybrid target the only exact resume point past the prompt is the
entry published when the reply finished.  The count pool ranks it (default,
rank 1) below every prompt boundary (rank 2), so once ``max_size`` boundaries
from unrelated requests are resident, each new finished-turn entry was spilled
to disk inside its own ``store``.  The next round's first lookup does not
restore from disk, so it fell back to the prompt boundary and re-prefilled
the whole reply (qualification/runs/intake-probes-20261007/apcdiag*).
"""

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache


def _state(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return [cache]


def test_finished_turn_entry_survives_a_pool_full_of_old_boundaries(tmp_path):
    now = [0.0]
    apc = APCv2(max_size=4, layout_name="reply-retention-v1", idle_disk_seconds=3600,
                idle_disk_dir=str(tmp_path), now_fn=lambda: now[0])
    key = APCKey("reply-retention")
    for i in range(4):  # unrelated, older requests fill the pool
        now[0] = float(i)
        apc.store(key, [10 * (i + 1)] * 8, _state(8, seed=i),
                  retention_role="committed_prompt_boundary")
    prompt = list(range(100, 110))
    reply = list(range(110, 116))
    now[0] = 10.0
    apc.store(key, prompt, _state(len(prompt), seed=50),
              retention_role="committed_prompt_boundary")
    now[0] = 11.0
    apc.store(key, prompt + reply, _state(len(prompt + reply), seed=60))

    hit = apc.lookup(key, prompt + reply + [200, 201], allow_disk_restore=False)
    assert hit.hit and hit.cached_tokens == len(prompt + reply), (
        hit.cached_tokens, hit.retention_role)
    hit.cache.close()
    apc.clear(release_memory=False)
