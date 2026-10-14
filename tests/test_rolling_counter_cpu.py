"""Windowed event counters (10 s / 1 min / 5 min / 30 min) and their APCv2 use."""

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.rolling_counter import RollingCounter

mx.set_default_device(mx.cpu)


def test_events_fall_out_of_each_window_at_its_edge():
    now = [0.0]
    counter = RollingCounter(clock=lambda: now[0])
    counter.add(bytes=100, seconds=0.5, idle=1)
    now[0] = 9.0
    counter.add(bytes=10, seconds=0.25, pressure=1)
    now[0] = 15.0
    snap = counter.snapshot()
    assert snap["10s"] == {"count": 1, "bytes": 10.0, "seconds": 0.25, "pressure": 1.0}
    assert snap["1m"]["count"] == 2 and snap["1m"]["bytes"] == 110.0
    assert snap["1m"]["idle"] == 1.0 and snap["1m"]["pressure"] == 1.0
    now[0] = 400.0
    snap = counter.snapshot()
    assert snap["5m"] == {"count": 0}
    assert snap["30m"]["count"] == 2
    now[0] = 2000.0
    assert counter.snapshot()["30m"] == {"count": 0}


def test_event_cap_drops_the_oldest_and_says_so():
    now = [0.0]
    counter = RollingCounter(clock=lambda: now[0], max_events=3)
    for step in range(5):
        now[0] = float(step)
        counter.add(bytes=1)
    snap = counter.snapshot()
    assert snap["10s"]["count"] == 3
    assert snap["events_dropped_at_cap"] == 2


def _state(length):
    cache = KVCache()
    values = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def test_apcv2_status_reports_recent_spills_and_restores(tmp_path):
    now = [0.0]
    apc = APCv2(
        max_size=4,
        layout_name="rolling-spill-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        now_fn=lambda: now[0],
    )
    key = APCKey("m")
    apc.store(key, [1, 2, 3], [_state(3)])
    recent = apc.apc_stats["idle_disk"]["recent"]
    assert recent["spills"]["1m"] == {"count": 0}
    now[0] = 180
    assert apc.spill_idle_entries() == 1
    spills = apc.apc_stats["idle_disk"]["recent"]["spills"]
    assert spills["10s"]["count"] == 1 and spills["10s"]["idle"] == 1.0
    assert spills["10s"]["bytes"] > 0 and spills["10s"]["seconds"] >= 0
    assert spills["10s"]["background"] == 0.0
    hit = apc.lookup(key, [1, 2, 3, 4])
    assert hit.cached_tokens == 3
    restores = apc.apc_stats["idle_disk"]["recent"]["restores"]
    assert restores["10s"]["count"] == 1 and restores["10s"]["bytes"] > 0
    assert apc.apc_stats["idle_disk"]["restores"] == 1
