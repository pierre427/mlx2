"""Partial cohort construction and post-commit rollback on CPU tensors."""

import pytest
from test_dflash_pair_select import drain, generator, tiny
from test_peer_survey_multiround_sampling_cpu import _known_target

from mlx2.runtime.sample_utils import LaneRNG


@pytest.mark.parametrize("owner", ["draft", "target"])
def test_later_cache_allocation_failure_publishes_no_partial_cohort(monkeypatch, owner):
    model, draft = tiny()
    batch = generator(model, draft)
    existing = batch.insert([[1, 2]], max_tokens=[2])[0]
    before_uid = batch.next_uid
    before_lanes = dict(batch.lanes)
    before_stats = dict(batch.scheduler_stats)
    target = draft if owner == "draft" else model
    original, calls = target.make_cache, 0

    def allocate():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected second cache allocation")
        return original()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(target, "make_cache", allocate)
            with pytest.raises(MemoryError, match="second cache"):
                batch.insert([[3, 4], [5, 6]], max_tokens=[2, 2])
        assert batch.lanes == before_lanes
        assert batch.next_uid == before_uid
        assert batch.scheduler_stats == before_stats
        retried = batch.insert([[3, 4], [5, 6]], max_tokens=[2, 2])
        assert retried == [before_uid, before_uid + 1]
        output, _ = drain(batch)
        assert set(output) == {existing, *retried}
        assert all(len(tokens) == 2 for tokens in output.values())
    finally:
        batch.close()


def test_later_request_validation_does_not_consume_uid_or_publish_first_request():
    model, draft = tiny()
    batch = generator(model, draft)
    try:
        for options, message in (({"max_tokens": [2]}, "one entry"),
                                 ({"max_tokens": [2, 0]}, "positive budget")):
            with pytest.raises(ValueError, match=message):
                batch.insert([[1, 2], [3, 4]], **options)
            assert not batch.lanes
            assert batch.next_uid == 0
        assert batch.insert([[1, 2]], max_tokens=[2]) == [0]
    finally:
        batch.close()


def _run_with_failure(monkeypatch, stage=None):
    batch = _known_target(monkeypatch, width=3)
    output = {}
    failures, calls = 0, 0
    batch.insert([[1, 2]] * 3, max_tokens=[4, 8, 9],
                 lane_rngs=[LaneRNG(101), LaneRNG(102), LaneRNG(103)],
                 sampling_configs=[{"sampling_temp": 1.0, "emit_logprobs": False}] * 3)
    for lane in batch.lanes.values():
        while lane.anchor is None:
            batch._prefill(lane)
    if stage is not None:
        original = getattr(batch, stage)

        def fail_after_real_work(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 3:
                raise RuntimeError("injected post-work failure")
            return result

        monkeypatch.setattr(batch, stage, fail_after_real_work)
    try:
        for _ in range(40):
            # Snapshot the actual lane stream; a failed round must not
            # consume randomness or expose a partly published response.
            before = {uid: (lane.rng.snapshot(), tuple(lane.history), lane.generated,
                            tuple(lane.ready)) for uid, lane in batch.lanes.items()}
            try:
                batch._round(list(batch.lanes.values()))
            except RuntimeError as exc:
                assert str(exc) == "injected post-work failure"
                failures += 1
                for uid, (rng, history, generated, ready) in before.items():
                    lane = batch.lanes[uid]
                    assert lane.rng.snapshot() == rng
                    assert tuple(lane.history) == history
                    assert lane.generated == generated
                    assert tuple(lane.ready) == ready
                continue
            done = []
            for lane in batch.lanes.values():
                while lane.ready:
                    response = lane.ready.popleft()
                    output.setdefault(response.uid, []).append(response.token)
                    if response.finish_reason:
                        done.append(response.uid)
            if done:
                batch.remove(done)
            if not batch.lanes:
                break
        assert not batch.lanes
        assert failures == int(stage is not None)
        return output
    finally:
        batch.close()


@pytest.mark.parametrize("stage", ["_propose", "_commit", "_sidecar"])
def test_multiround_retry_keeps_sampled_output_and_lane_ownership(monkeypatch, stage):
    assert _run_with_failure(monkeypatch, stage) == _run_with_failure(monkeypatch)
