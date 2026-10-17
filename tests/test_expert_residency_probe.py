from mlx2.runtime.expert_residency_probe import replay_hybrid, replay_per_projection_lru


def test_hybrid_shares_capacity_without_removing_layer_floor():
    sequence = [(0, [0]), (0, [1]), (0, [2]), (1, [7]), (0, [0]), (1, [7])]
    current = replay_per_projection_lru(
        sequence, capacities={0: 2, 1: 2}, decode_start=4, terminal_start=3
    )
    hybrid = replay_hybrid(
        sequence, row_bytes={0: 1, 1: 1}, total_budget_bytes=4,
        floor_entries=1, decode_start=4, terminal_start=3,
    )
    assert hybrid["decode_first_8_page_ins"] < current["decode_first_8_page_ins"]
    assert hybrid["minimum_floor_bytes"] == 2


def test_scan_resistance_keeps_intermediate_prefill_out_of_residency():
    sequence = [(0, [1]), (0, [2]), (0, [9]), (0, [9])]
    hybrid = replay_hybrid(
        sequence, row_bytes={0: 1}, total_budget_bytes=2, floor_entries=1,
        decode_start=3, terminal_start=2, scan_resistant=False,
    )
    scan = replay_hybrid(
        sequence, row_bytes={0: 1}, total_budget_bytes=2, floor_entries=1,
        decode_start=3, terminal_start=2, scan_resistant=True,
    )
    assert scan["decode_first_8_page_ins"] == 0
    assert scan["phase"]["intermediate_prefill"]["misses"] == 2
    assert hybrid["decode_first_8_page_ins"] == 0


def test_elastic_shrink_records_transition_eviction():
    report = replay_hybrid(
        [(0, [1]), (0, [2]), (0, [3]), (0, [1])],
        row_bytes={0: 10}, total_budget_bytes=40, floor_entries=1,
        decode_start=3, terminal_start=2, budget_schedule=[(3, 20)],
    )
    assert report["transition_evictions"] >= 1


class _HostReader:
    num_experts = 16
    expert_bytes = 1

    def check_sources(self):
        pass

    def outstanding_bytes(self):
        return 0

    def iter_many(self, experts):
        for expert in experts:
            yield (expert, expert)

    def materialize(self, payload):
        return (payload,)


def _live_terminal_hot(sequence, *, serving_start, terminal_start, decode_start, capacity):
    from mlx2.runtime import weight_stream as ws

    stats = ws.StreamStats()
    cache = ws.ExpertLRU(_HostReader(), capacity_experts=capacity, stats=stats,
                         prefill_admission="terminal_hot")
    decode_misses = 0
    for index, (_, values) in enumerate(sequence):
        before = stats.misses
        if serving_start <= index < decode_start:
            phase = "terminal" if index >= terminal_start else "intermediate"
            with ws.expert_prefill_admission(phase, ("request",)):
                cache.acquire(values)
        else:
            cache.acquire(values)
        if index >= decode_start:
            decode_misses += stats.misses - before
    return decode_misses


def test_scan_resistant_replay_matches_live_terminal_hot_admission():
    # The replay broke frequency ties by expert id and counted load-phase
    # calls; the live policy keeps router order on ties and counts only the
    # request's prefill calls (sweep 2026-10-16 NEMO-MAIN-3).
    cases = [
        # (sequence, serving_start, terminal_start, decode_start)
        ([(0, [2]), (0, [2]), (0, [5]), (0, [5, 2]), (0, [5])], 2, 3, 4),
        ([(0, [3, 1]), (0, [1])], 0, 0, 1),
    ]
    for (sequence, serving_start, terminal_start, decode_start) in cases:
        replay = replay_hybrid(
            sequence, row_bytes={0: 1}, total_budget_bytes=1, floor_entries=1,
            decode_start=decode_start, terminal_start=terminal_start,
            serving_start=serving_start, scan_resistant=True,
        )
        live = _live_terminal_hot(
            sequence, serving_start=serving_start, terminal_start=terminal_start,
            decode_start=decode_start, capacity=1,
        )
        assert replay["phase"]["decode"]["misses"] == live, sequence
