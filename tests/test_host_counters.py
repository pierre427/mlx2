from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from mlx2.host_counters import HostCounters, count_add, count_max, counter_snapshot


def test_concurrent_writers_and_snapshots_preserve_exact_counts():
    counters = HostCounters()

    def writer(index):
        for value in range(1000):
            count_add(counters, "shared")
            count_add(counters, f"writer_{index}")
            count_max(counters, "maximum", index * 1000 + value)
            if value % 100 == 0:
                snapshot = counter_snapshot(counters)
                assert snapshot["shared"] >= snapshot[f"writer_{index}"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(writer, range(8)))
    snapshot = counter_snapshot(counters)
    assert snapshot["shared"] == 8000
    assert snapshot["maximum"] == 7999
    assert [snapshot[f"writer_{i}"] for i in range(8)] == [1000] * 8


def test_counter_missing_key_and_legacy_fixture_contract():
    counters = HostCounters({"existing": 3})
    assert counters["missing"] == 0
    assert counters.get("missing", 7) == 7
    assert "missing" not in counters
    assert counters.setdefault("existing", 99) == 3
    assert counters.setdefault("new", 4) == 4
    legacy = Counter()
    count_add(legacy, "a", 2)
    count_max(legacy, "a", 8)
    assert counter_snapshot(legacy) == {"a": 8}


def test_counter_update_is_additive_for_mapping_iterable_kwargs_and_self():
    counts = HostCounters({"a": 2, "b": 1})
    counts.update({"a": 3}, b=2)
    counts.update(iter(("a", "c", "c")), c=1)
    assert counts.snapshot() == {"a": 6, "b": 3, "c": 3}
    counts.update(counts)
    assert counts.snapshot() == {"a": 12, "b": 6, "c": 6}
    counts.update()
    assert counts["a"] == 12
