"""CPU lifecycle oracles for the explicitly declared ordinary decode frontier."""
import pytest

from mlx2.runtime.generate import BatchGenerator, PromptProcessingBatch
from test_batched_mtp import _tiny_qwen4_model


def _batch(**kwargs):
    return BatchGenerator(_tiny_qwen4_model(), completion_batch_size=4,
                          prefill_batch_size=2, prefill_step_size=4, **kwargs)


def _insert(batch, lengths=(5, 9, 17, 21), *, declared=True, size=4):
    return batch.insert(
        [[1 + i] * length for i, length in enumerate(lengths)],
        max_tokens=[3] * len(lengths),
        self_mtp_configs=[{"batch_cohort": {"tenant_id": "t", "id": "cold", "size": size}}
                          if declared else {} for _ in lengths],
    )


def _hold(batch):
    for _ in range(30):
        _, responses = batch.next()
        assert not responses
        if batch._ordinary_held is not None:
            return
    pytest.fail("no held frontier")


@pytest.mark.parametrize("lengths", [(5, 9, 17, 21), (9, 5, 21, 17), (1, 1, 1, 1)])
@pytest.mark.parametrize("decode_first", [False, True])
def test_declared_first_token_and_all_decode_have_full_producing_width(lengths, decode_first):
    batch = _batch(decode_first=decode_first)
    uids = _insert(batch, lengths)
    completed_prompts = set()
    responses = []
    try:
        for _ in range(80):
            prompts, output = batch.next()
            assert len(batch._prompt_batch) <= 2
            completed_prompts.update(p.uid for p in prompts if p.end_of_prompt)
            if output:
                assert completed_prompts == set(uids)
            responses.extend(output)
            if sum(r.finish_reason is not None for r in responses) == 4:
                break
        assert len(responses) == 12
        assert {r.execution_width for r in responses} == {4}
        assert batch._ordinary_held is None
        assert batch._ordinary_cohort_active is None
        assert not batch._ordinary_cohorts
    finally:
        batch.close()


def test_undeclared_rows_still_generate_before_long_tail_prefills():
    batch = _batch()
    uids = _insert(batch, (1, 1, 29, 29), declared=False)
    try:
        for _ in range(10):
            _, output = batch.next()
            if output:
                assert {r.uid for r in output} == set(uids[:2])
                assert {r.execution_width for r in output} == {2}
                assert batch._ordinary_held is None
                break
        else:
            pytest.fail("progressive outputs did not arrive")
    finally:
        batch.close()


@pytest.mark.parametrize("member", [0, 1, 2, 3])
def test_removal_from_held_prompt_or_queue_fails_all_survivors(member):
    batch = _batch()
    uids = _insert(batch)
    try:
        _hold(batch)
        # At this boundary lane 0 is held, lane 1 prefilling, tail queued.
        held_uid = batch._ordinary_held[0].uids[0]
        removed = held_uid if member == 0 else uids[member]
        caches = batch.remove([removed], return_prompt_caches=True)
        assert removed in caches
        failures = batch.take_atomic_cohort_failures()
        assert len(failures) == 1
        assert set(failures[0]["uids"]) == set(uids) - {removed}
        assert not batch._find_uids(uids)
        assert not batch._ordinary_cohorts
        assert batch._ordinary_held is None
        assert batch._ordinary_cohort_active is None
        assert batch.next() == ([], [])
        assert not batch.take_atomic_cohort_failures()
    finally:
        batch.close()


def test_incomplete_and_overcapacity_groups_fail_before_any_forward():
    for lengths, size in [((5, 9), 4), ((5,) * 5, 5)]:
        batch = _batch()
        try:
            uids = _insert(batch, lengths, size=size)
            assert batch.next() == ([], [])
            assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(uids)
            assert not batch._find_uids(uids)
        finally:
            batch.close()


def test_frontier_reserves_full_budget_and_accounts_held_cache():
    batch = _batch(kv_budget_bytes=10**9, kv_cost=(0, 1, 1))
    uids = _insert(batch)
    try:
        _hold(batch)
        held = batch._ordinary_held[0]
        assert set(held.uids) <= set(batch.scheduler_waiting_uids())
        states = batch._final_extent_states()
        assert set(held.uids) <= {s.uid for s in states}
        actual = sum(c.nbytes for c in held.prompt_cache)
        assert batch._cohort_committed([]) >= actual > 0
        batch.kv_budget_bytes = 1
        batch.next()
        assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(uids)
        assert not batch._find_uids(uids)
        assert batch._ordinary_held is None
    finally:
        batch.close()


def test_memory_failure_releases_held_cohort_and_preserves_exception(monkeypatch):
    batch = _batch()
    uids = _insert(batch)
    try:
        _hold(batch)
        def fail(*args, **kwargs):
            raise RuntimeError("device out of memory fixture")
        monkeypatch.setattr(PromptProcessingBatch, "prompt", fail)
        with pytest.raises(RuntimeError, match="out of memory"):
            batch.next()
        assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(uids)
        assert not batch._find_uids(uids)
        assert batch._ordinary_held is None
    finally:
        batch.close()


def test_close_releases_held_frontier():
    batch = _batch()
    uids = _insert(batch)
    _hold(batch)
    held = batch._ordinary_held[0]
    batch.close()
    assert not batch._find_uids(uids)
    assert not held.prompt_cache
    assert batch._ordinary_held is None
    assert batch._ordinary_cohort_active is None
    assert not batch._ordinary_cohorts


def test_full_group_budget_rejection_precedes_prefill():
    batch = _batch(kv_budget_bytes=1, kv_cost=(0, 1, 1))
    try:
        uids = _insert(batch)
        assert batch.next() == ([], [])
        assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(uids)
        assert batch._prompt_tokens_counter == 0
        assert not batch._find_uids(uids)
    finally:
        batch.close()


def test_unrelated_queued_work_survives_cancelled_frontier():
    batch = _batch()
    uids = _insert(batch)
    other = _insert(batch, (1,), declared=False)[0]
    try:
        _hold(batch)
        batch.remove([uids[0]])
        assert set(batch.take_atomic_cohort_failures()[0]["uids"]) == set(uids[1:])
        output = []
        for _ in range(10):
            _, rows = batch.next()
            output.extend(rows)
        assert len(output) == 3
        assert {r.uid for r in output} == {other}
        assert {r.execution_width for r in output} == {1}
    finally:
        batch.close()


def test_declared_group_waits_for_existing_ordinary_decode_to_drain():
    batch = _batch()
    first = _insert(batch, (1,), declared=False)[0]
    uids = _insert(batch)
    output = []
    try:
        for _ in range(80):
            _, rows = batch.next()
            output.extend(rows)
            if sum(r.finish_reason is not None for r in output) == 5:
                break
        assert [r.uid for r in output[:3]] == [first] * 3
        assert {r.uid for r in output[3:]} == set(uids)
        assert {r.execution_width for r in output[3:]} == {4}
    finally:
        batch.close()


def test_post_release_removal_keeps_surviving_decode_lanes():
    batch = _batch()
    uids = _insert(batch, (1, 1, 1, 1))
    try:
        for _ in range(10):
            batch.next()
            if len(batch._generation_batch) == 4:
                break
        assert batch._ordinary_cohort_active is None
        batch.remove([uids[0]])
        assert not batch.take_atomic_cohort_failures()
        output = []
        for _ in range(10):
            _, rows = batch.next()
            output.extend(rows)
        assert len(output) == 9
        assert {r.uid for r in output} == set(uids[1:])
        assert {r.execution_width for r in output} == {3, 4}
    finally:
        batch.close()
