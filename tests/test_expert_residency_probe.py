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
