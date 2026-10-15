"""Cross-review of exact ordinary checkpoint control and owner boundaries."""

import mlx.core as mx
import pytest
from route_harness import patch_host
from test_exact_decode_checkpoint_cpu import _batch, _model, _next_output, _run_server

from mlx2.batch_metrics import BatchRuntimeMetrics
from mlx2.runtime.generate import BatchGenerator, StopSequenceMatcher


class _StatefulSequence:
    def __init__(self):
        self.calls = []

    def __call__(self, tokens, logits):
        self.calls.append(tokens.tolist())
        target = (5, 7, 9, 11)[min(len(self.calls) - 1, 3)]
        return mx.where(mx.arange(logits.shape[-1]) == target, 0.0, -100.0)[None]


def test_stop_prefix_and_stateful_processor_survive_checkpoint():
    model = _model()

    def run(park):
        processor = _StatefulSequence()
        batch = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                               prefill_step_size=8)
        uid, = batch.insert([[1, 2, 3]], max_tokens=[8],
                            logits_processors=[[processor]],
                            stop_matchers=[StopSequenceMatcher([[5, 7, 9]])])
        checkpoint = None
        try:
            rows = [_next_output(batch), _next_output(batch)]
            assert [row.token for row in rows] == [5, 7]
            if park:
                state = batch._generation_batch._matcher_states[0]
                calls = list(processor.calls)
                checkpoint = batch.suspend_ordinary_checkpoint(uid, ("law", "request"))
                batch.resume_ordinary_checkpoint(checkpoint, ("law", "request"))
                assert batch._generation_batch.logits_processors[0][0] is processor
                assert batch._generation_batch._matcher_states[0] == state
                assert processor.calls == calls
            rows.append(_next_output(batch))
            assert rows[-1].finish_reason == "stop"
            assert rows[-1].token == 9 and not len(batch._generation_batch)
            return rows, processor.calls
        finally:
            if checkpoint is not None:
                checkpoint.close()
            batch.close()

    reference, reference_calls = run(False)
    restored, restored_calls = run(True)
    assert restored_calls == reference_calls
    assert all(mx.array_equal(a.logprobs, b.logprobs).item()
               for a, b in zip(reference, restored))


def test_restore_geometry_refusal_preserves_checkpoint_for_exact_retry():
    model = _model()
    batch, uid = _batch(model)
    checkpoint = None
    try:
        _next_output(batch)
        checkpoint = batch.suspend_ordinary_checkpoint(uid, ("law", "request"))
        geometry = checkpoint.geometry
        checkpoint.geometry = ()
        with pytest.raises(ValueError, match="geometry"):
            batch.resume_ordinary_checkpoint(checkpoint, ("law", "request"))
        assert not checkpoint.closed and not checkpoint.consumed
        assert checkpoint.control is not None and not checkpoint.control.prompt_cache
        assert not len(batch._generation_batch)
        checkpoint.geometry = geometry
        assert batch.resume_ordinary_checkpoint(checkpoint, ("law", "request")) == uid
        assert _next_output(batch).uid == uid
    finally:
        if checkpoint is not None:
            checkpoint.close()
        batch.close()


def test_foreign_executor_refusal_preserves_origin_and_uid_allocation():
    model = _model()
    source, uid = _batch(model)
    target = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1)
    checkpoint = None
    try:
        _next_output(source)
        checkpoint = source.suspend_ordinary_checkpoint(uid, ("law", "request"))
        with pytest.raises(ValueError, match="executor"):
            target.resume_ordinary_checkpoint(checkpoint, ("law", "request"))
        assert not checkpoint.closed and not checkpoint.consumed
        assert not len(target._generation_batch)
        source.resume_ordinary_checkpoint(checkpoint, ("law", "request"))
        newcomer, = source.insert([[7, 8, 9]], max_tokens=[2])
        assert newcomer != uid
        assert source._find_uids((uid, newcomer)) == {uid: (2, 0), newcomer: (0, 0)}
    finally:
        if checkpoint is not None:
            checkpoint.close()
        target.close()
        source.close()


def test_resume_metrics_failure_does_not_end_an_owned_live_lane(monkeypatch):
    patch_host(monkeypatch)
    model = _model()
    expected, _, _, _ = _run_server(model, sample=True)
    original = BatchRuntimeMetrics.lane_attached
    attachments = 0

    def attached(metrics, *args, **kwargs):
        nonlocal attachments
        attachments += 1
        if attachments == 2:
            raise RuntimeError("injected resume telemetry failure")
        return original(metrics, *args, **kwargs)

    monkeypatch.setattr(BatchRuntimeMetrics, "lane_attached", attached)
    actual, receipt, counts, job = _run_server(model, sample=True, fault=True)
    assert attachments == 2
    assert actual == expected and len(actual) == 14
    assert counts["exact_decode_resumes"] == 1
    assert receipt["preemption"]["events"][0]["restored_without_prefill"] is True
    assert job.exact_decode_checkpoint is None
