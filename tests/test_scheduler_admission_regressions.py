"""CPU ordinary scheduling and exact shared-state admission regressions."""

from collections import deque
from types import SimpleNamespace

import pytest
import mlx.core as mx

from mlx2.runtime.batch_admission import AdmissionState, LinearStateCost, StateBudget
from mlx2.runtime.generate import BatchGenerator, MTPGenerationBatch
from test_batched_mtp import _tiny_qwen4_model


def _batch(**kwargs):
    mx.random.seed(20261022)
    return BatchGenerator(
        _tiny_qwen4_model(), completion_batch_size=4, prefill_batch_size=2,
        prefill_step_size=4, **kwargs,
    )


def test_budgeted_admission_beside_live_ordinary_decode():
    batch = _batch(kv_budget_bytes=10**9, kv_cost=(0, 1, 1))
    try:
        first = batch.insert([[1]], max_tokens=[10])[0]
        batch.next()
        assert batch._generation_batch.uids == [first]
        second = batch.insert([[2, 3, 4]], max_tokens=[3])[0]
        batch.next()
        assert second in batch._prompt_batch.uids
        states = batch._final_extent_states()
        state = next(state for state in states if state.uid == first)
        assert state.completed_units == len(batch._generation_batch.tokens[0])
        assert state.projected_units >= state.completed_units
    finally:
        batch.close()


def test_memory_admission_checks_padding_selected_rows():
    batch = _batch(kv_budget_bytes=100, kv_cost=(0, 1, 1), prefill_batch_window=4)
    try:
        uids = batch.insert([[1] * 25, [2] * 2, [3] * 25], max_tokens=[1, 1, 1000])
        assert batch._select_prefill_indices(2) == [0, 2]
        prompts, _ = batch.next()
        assert {response.uid for response in prompts} == {uids[0]}
        assert [row[0] for row in batch._unprocessed_sequences] == uids[1:]
    finally:
        batch.close()


def test_prefilling_lane_reserves_its_completion_slot_until_promotion():
    batch = _batch()
    try:
        first = batch.insert([[1], [2], [3]], max_tokens=[64] * 3)
        for _ in range(4):
            batch.next()
            if len(batch._generation_batch) == 3:
                break
        assert batch._generation_batch.uids == first
        fourth = batch.insert([[4] * 9], max_tokens=[64])[0]
        batch.next()
        batch.next()
        assert batch._prompt_batch.uids == [fourth]
        assert batch._currently_processing[0][0] == [[4]]
        fifth = batch.insert([[5]], max_tokens=[1])[0]
        batch.next()
        assert len(batch._generation_batch) == 4
        assert fourth in batch._generation_batch.uids
        assert [row[0] for row in batch._unprocessed_sequences] == [fifth]
        assert not batch._prompt_batch.uids
    finally:
        batch.close()


def test_mtp_prefix_lengths_do_not_materialize_token_arrays():
    class Prefix:
        shape = (1, 128000)
        ndim = 2

        def tolist(self):
            pytest.fail("state admission must not materialize an MTP prefix")

    batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
    batch.state = SimpleNamespace(lanes=[SimpleNamespace(token_prefix=Prefix())])
    assert batch.prefix_lengths == [128000]


@pytest.mark.parametrize("step", [None, 4])
def test_incremental_projection_matches_shared_width_and_capped_extents(step):
    cost = LinearStateCost(5, 2, max_units=20, allocation_step_units=step)
    active = [AdmissionState(0, 7)]
    candidates = [AdmissionState(1, 0), AdmissionState(2, 15), AdmissionState(3, 100)]
    charges = list(cost.cohort_prefix_bytes(active, iter(candidates)))
    expected = ([19, 24, 59, 104] if step is None else [25, 50, 123, 204])
    assert charges == expected


def test_admission_snapshots_resident_bytes_once_and_keeps_unselected_caches():
    class Resident:
        def __init__(self, size):
            self.size, self.reads = size, 0

        @property
        def nbytes(self):
            self.reads += 1
            return self.size

    batch = _batch()
    try:
        resident = [Resident(20), Resident(30), Resident(40)]
        batch._unprocessed_sequences = deque(
            (uid, [], 1, [cache], []) for uid, cache in enumerate(resident, 1)
        )
        active_reads = []
        batch._final_extent_states = lambda: active_reads.append(1) or [AdmissionState(0, 10)]
        batch.state_budget = StateBudget(110, LinearStateCost(0, 1, allocation_step_units=1))
        states = [AdmissionState(1, 20), AdmissionState(2, 30, metadata={"unverified_bytes": 30})]
        # First: 2*20 + unselected 30+40 ==110. Second: 3*30 +
        # unverifiable selected 30 + unselected40 ==160, so only one fits.
        assert batch._admit_states(states) == 1
        assert active_reads == [1]
        assert [cache.reads for cache in resident] == [1, 1, 1]
        assert len(batch._unprocessed_sequences) == 3
    finally:
        batch._unprocessed_sequences.clear()
        batch.close()
