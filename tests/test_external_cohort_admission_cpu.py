"""CPU scheduler/admission probes with synthetic Muse/DFlash weights."""

from contextlib import closing

from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

from test_dflash_pair_select import generator, tiny


def _batch(width):
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    return generator(
        model, draft, completion_batch_size=width, ready_drain="all",
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        dynamic_tree_overflow="cohort",
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=HybridTreeExecutor.create(),
        cost_per_commit_handoff=False, memory_headroom=lambda: 10**12,
    )


def _prefill(batch, prompts):
    uids = batch.insert(prompts, max_tokens=[100] * len(prompts))
    for lane in batch.lanes.values():
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ready.clear()
    return uids


def test_newer_admissible_lane_progresses_past_denied_oldest_cohort(monkeypatch):
    with closing(_batch(5)) as batch:
        uids = _prefill(batch, [[1, 2] * 32] * 4 + [[3, 4]])
        lanes = [batch.lanes[uid] for uid in uids]
        for lane in lanes[:4]:
            lane.decode_service_epoch = 0
        lanes[4].decode_service_epoch = 1
        batch._decode_service_epoch = 1
        # Price real unequal cache states through the actual admission path.
        assert batch._admit([lanes[4]], 4)
        short_required = batch.scheduler_stats["reservation_bytes"]
        batch.memory_headroom = lambda: short_required
        assert all(not batch._admit([lane], 4) for lane in lanes[:4])
        assert batch._admit([lanes[4]], 4)
        calls = []
        monkeypatch.setattr(
            batch, "_round_at_batch_route",
            lambda group, **kwargs: calls.append(tuple(lane.uid for lane in group)),
        )
        rngs = [lane.rng.snapshot() for lane in lanes]
        for _ in range(10):
            batch.next()
            assert calls[-1:] == [(uids[4],)]
            assert set(batch.scheduler_waiting_uids()) == set(uids[:4])
        assert calls == [(uids[4],)] * 10
        assert batch._schedule_cursor >= 10
        assert [lane.rng.snapshot() for lane in lanes] == rngs


def test_provider_admission_stops_at_physical_limit_and_rotates(monkeypatch):
    with closing(_batch(8)) as batch:
        uids = _prefill(batch, [[1, 2]] * 8)
        admitted_widths, calls = [], []
        admit = batch._admit

        def capture(lanes, append, **kwargs):
            admitted_widths.append(len(lanes))
            return admit(lanes, append, **kwargs)

        monkeypatch.setattr(batch, "_admit", capture)
        monkeypatch.setattr(
            batch, "_round_at_batch_route",
            lambda group, **kwargs: calls.append(tuple(lane.uid for lane in group)),
        )
        for _ in range(4):
            batch.next()
        assert len(calls) == 4 and all(len(group) == 4 for group in calls)
        assert set(calls[0]) | set(calls[1]) == set(uids)
        assert set(calls[2]) | set(calls[3]) == set(uids)
        assert max(admitted_widths) == 4
