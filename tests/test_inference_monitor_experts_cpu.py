"""Expert swap stats for mlx2.top: snapshot, window rates, rendering (CPU)."""
from mlx2.inference_monitor import expert_stream_reading, expert_stream_snapshot
from mlx2.top import _expert_lines


def status(hits, misses, read, wait, evict, live=70e9, drop=None, approx=()):
    s = {"execution": {"expert_stream": {"counters": {
        "v41_expert_cache_hits": hits, "v41_expert_cache_misses": misses, "v41_expert_bytes_read": read,
        "v41_expert_demand_wait_ns": wait, "v41_expert_evictions": evict, "v41_expert_q1_pool_hits": 5,
        "flag": True}, "live_bytes": live, "adaptive_residency": {"effective_cache_target_bytes": 75e9}},
        "source_approximate_operations": {"selected": list(approx)}}}
    if drop is not None:
        s["execution"]["source_approximate_operations"]["route_drop"] = {"stats": {"dropped_layer_rows": drop}}
    return s


def test_absent_outside_expert_mode():
    assert expert_stream_snapshot({"execution": {}}) is None
    assert expert_stream_reading(None, None, 1.0) is None
    assert _expert_lines(None) == []


def test_window_rates_and_rendering():
    before = status(100, 50, 1e9, 2e9, 10)
    after = status(190, 60, 1.5e9, 3e9, 18, drop=42, approx=["route_drop_min_v1"])
    r = expert_stream_reading(after, (10.0, expert_stream_snapshot(before)), 20.0)
    assert r.seconds == 10 and r.misses_delta == 10 and r.hits_delta == 90 and r.evictions_delta == 8
    assert abs(r.hit_rate_window - 0.9) < 1e-9 and r.rate(r.bytes_delta) == 5e7
    names = [n for n, _, _ in r.reduced]
    assert "v41_expert_q1_pool_hits" in names and "route_drop_dropped_layer_rows" in names
    text = "\n".join(_expert_lines(r))
    assert "swap-ins" in text and "swap-outs" in text and "hit 90.0% (window)" in text
    assert "route_drop_min_v1 (unqualified)" in text and "q1_pool_hits" in text


def test_counter_reset_never_goes_negative():
    r = expert_stream_reading(status(5, 1, 0, 0, 0), (0.0, expert_stream_snapshot(status(100, 50, 1e9, 2e9, 10))), 5.0)
    assert r.misses_delta == 0 and r.hits_delta == 0
