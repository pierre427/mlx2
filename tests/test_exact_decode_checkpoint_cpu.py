"""Real ordinary executor and serving checkpoint continuation, CPU only."""
import threading
import time

import mlx.core as mx
import pytest
from route_harness import patch_host
from test_suspend_replay import collect, make_adapter

from mlx2 import serving
from mlx2.runtime import decode_checkpoint as dc
from mlx2.runtime import io_ownership
from mlx2.runtime.generate import BatchGenerator, GenerationBatch
from mlx2.runtime.sample_utils import LaneRNG, draw_key


def _model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel
    args = TextModelArgs(model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, vocab_size=128, full_attention_interval=1,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=3, mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=256)
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _batch(model, *, sample=False):
    rng = LaneRNG(11)
    sampler = (lambda values: mx.random.categorical(values, key=draw_key(rng))) if sample else None
    batch = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=8)
    uid = batch.insert([list(range(1, 19))], max_tokens=[14],
                       samplers=[sampler], lane_rngs=[rng if sample else None])[0]
    return batch, uid


def _next_output(batch):
    for _ in range(64):
        _prompts, outputs = batch.next()
        if outputs:
            assert len(outputs) == 1
            return outputs[0]
    raise AssertionError("executor stalled")


@pytest.mark.parametrize("sample", [False, True])
def test_executor_checkpoint_preserves_pending_token_logprobs_and_rng(sample):
    model = _model()
    reference, _ = _batch(model, sample=sample)
    expected = [_next_output(reference) for _ in range(14)]
    batch, uid = _batch(model, sample=sample)
    rows = [_next_output(batch) for _ in range(4)]
    control = batch._generation_batch
    rng = control.lane_rngs[0]
    draws = None if rng is None else rng.draws
    pending = control._next_tokens.tolist()
    checkpoint = batch.suspend_ordinary_checkpoint(uid, ("revision", "request"))
    try:
        assert checkpoint.control is control and not control.prompt_cache
        assert len(batch._generation_batch) == 0
        assert checkpoint.control._next_tokens.tolist() == pending
        assert batch.resume_ordinary_checkpoint(checkpoint, ("revision", "request")) == uid
        assert batch._generation_batch is control
        assert rng is None or control.lane_rngs[0].draws == draws
        rows.extend(_next_output(batch) for _ in range(10))
        assert [row.token for row in rows] == [row.token for row in expected]
        assert all(mx.array_equal(a.logprobs, b.logprobs).item() for a, b in zip(rows, expected))
        with pytest.raises(ValueError, match="no longer owned"):
            checkpoint.restore(model, ("revision", "request"))
    finally:
        checkpoint.close()
        batch.close()
        reference.close()


def _run_server(model, *, fault=False, sample=False, exact=True, pressure=None):
    engine = serving.ServingEngine("tiny", adapter_factory=make_adapter(model),
        qualification_mode=True, mtp=False, max_lanes=1, prefill_step=8,
        execution_policy={"memory_preemption": {"enabled": True,
            "exact_decode_checkpoint": exact}})
    try:
        assert engine.ready.wait(10), engine.error
        if pressure is not None:
            engine.memory_pressure_level = pressure
        request = {"tokens": list(range(1, 19)), "max_tokens": 14,
                       "temperature": 0.8 if sample else 0, "seed": 11}
        if fault:
            request["mlx_fault"] = {"kind": "memory_preempt", "after_tokens": 4}
        job = engine.submit(request)
        tokens, receipt = collect(job)
        return tokens, receipt, dict(engine.counts), job
    finally:
        engine.close()


@pytest.mark.parametrize("sample", [False, True])
def test_real_serving_fault_resumes_without_reprefill_and_default_stays_blocked(monkeypatch, sample):
    patch_host(monkeypatch)
    model = _model()
    expected, _, _, _ = _run_server(model, sample=sample)
    actual, receipt, counts, job = _run_server(model, sample=sample, fault=True)
    assert actual == expected and len(actual) == 14
    assert counts["exact_decode_checkpoints"] == counts["exact_decode_resumes"] == 1
    assert counts["preempted_replays"] == 0
    assert job.exact_decode_checkpoint is None
    event, = receipt["preemption"]["events"]
    assert event["mechanism"] == "exact_decode_checkpoint"
    assert event["restored_without_prefill"] is True
    blocked, receipt, counts, _ = _run_server(model, sample=sample, fault=True, exact=False)
    assert blocked == expected
    assert counts.get("exact_decode_resumes", 0) == 0
    assert receipt["preemption"]["fault_unfired"] == "decode_state_not_reconstructible"


@pytest.mark.parametrize("failure", ["empty", "detach"])
def test_checkpoint_capture_transaction_preserves_running_lane(monkeypatch, failure):
    model = _model()
    reference, _ = _batch(model)
    expected = [_next_output(reference).token for _ in range(14)]
    batch, uid = _batch(model)
    rows = [_next_output(batch).token for _ in range(4)]
    control = batch._generation_batch
    with monkeypatch.context() as patch:
        if failure == "empty":
            def fail_empty(*args, **kwargs):
                raise RuntimeError("injected empty construction")
            patch.setattr(GenerationBatch, "empty", fail_empty)
        else:
            remove = batch.remove
            def fail_remove(*args, **kwargs):
                remove(*args, **kwargs)
                raise RuntimeError("injected detach ledger failure")
            patch.setattr(batch, "remove", fail_remove)
        with pytest.raises(RuntimeError, match="injected"):
            batch.suspend_ordinary_checkpoint(uid, ("revision", "request"))
    try:
        assert batch._generation_batch is control and control.prompt_cache
        rows.extend(_next_output(batch).token for _ in range(10))
        assert rows == expected
    finally:
        batch.close()
        reference.close()


def _wait(predicate):
    until = time.monotonic() + 3
    while not predicate() and time.monotonic() < until:
        time.sleep(0.005)
    assert predicate()


def _checkpoint_io_drained():
    with io_ownership._READ_POOLS_LOCK:
        return all(
            owner != dc.CHECKPOINT_IO_OWNER
            for owner, _workers in io_ownership._QUARANTINED_READ_POOLS.values()
        )


def test_write_timeout_retains_original_kv_and_late_file_closes(monkeypatch):
    model = _model()
    batch, uid = _batch(model)
    _next_output(batch)
    control = batch._generation_batch
    cache = control.prompt_cache
    entered, release = threading.Event(), threading.Event()
    late_handles = []
    original = dc._write_payload
    def blocked(payload):
        entered.set()
        assert release.wait(3)
        owner = original(payload)
        late_handles.append(owner.handle)
        return owner
    monkeypatch.setattr(dc, "_write_payload", blocked)
    monkeypatch.setattr(dc, "CHECKPOINT_IO_SECONDS", 0.03)
    try:
        with pytest.raises(dc.ReadTimeout):
            batch.suspend_ordinary_checkpoint(uid, ("revision", "request"))
        assert entered.is_set() and batch._generation_batch is control
        assert control.prompt_cache is cache
        with pytest.raises(dc.ReadTimeout, match="have not drained"):
            dc.ReadPool(1, "replacement", owner=dc.CHECKPOINT_IO_OWNER)
        _next_output(batch)
    finally:
        release.set()
        _wait(lambda: bool(late_handles) and late_handles[0].closed)
        _wait(_checkpoint_io_drained)
        batch.close()


def test_restore_timeout_close_retains_descriptor_until_worker_returns(monkeypatch):
    model = _model()
    batch, uid = _batch(model)
    _next_output(batch)
    owner = batch.suspend_ordinary_checkpoint(uid, ("revision", "request"))
    handle = owner.file.handle
    entered, release = threading.Event(), threading.Event()
    original = dc._read_payload
    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(dc, "_read_payload", blocked)
    monkeypatch.setattr(dc, "CHECKPOINT_IO_SECONDS", 0.03)
    try:
        with pytest.raises(dc.ReadTimeout):
            batch.resume_ordinary_checkpoint(owner, ("revision", "request"))
        assert entered.is_set() and not len(batch._generation_batch)
        owner.close()
        owner.close()
        assert not handle.closed
        assert owner.control is owner.owner_executor is None
    finally:
        release.set()
        _wait(lambda: handle.closed)
        _wait(_checkpoint_io_drained)
        batch.close()


def _engine(model):
    return serving.ServingEngine("tiny", adapter_factory=make_adapter(model),
        qualification_mode=True, mtp=False, max_lanes=1, prefill_step=8,
        execution_policy={"memory_preemption": {"enabled": True,
            "exact_decode_checkpoint": True}})


def _terminal(job):
    while True:
        event = job.events.get(timeout=10)
        if "error" in event or "finish_reason" in event:
            return event


@pytest.mark.parametrize("failure", ["corrupt", "detach_rollback"])
def test_serving_restore_failure_closes_owner_and_keeps_worker_usable(monkeypatch, failure):
    patch_host(monkeypatch)
    engine = _engine(_model())
    owners = []
    original_capture = dc.OrdinaryDecodeCheckpoint.capture.__func__
    original_suspend = BatchGenerator.suspend_ordinary_checkpoint
    def capture(cls, *args):
        owner = original_capture(cls, *args)
        owners.append(owner)
        return owner
    try:
        assert engine.ready.wait(10), engine.error
        with monkeypatch.context() as patch:
            patch.setattr(dc.OrdinaryDecodeCheckpoint, "capture", classmethod(capture))
            if failure == "corrupt":
                original_read = dc._read_payload
                patch.setattr(dc, "_read_payload", lambda *args: original_read(*args)[:-1] + b"x")
            else:
                def fail_suspend(self, uid, binding):
                    original_remove = self.remove
                    def fail_remove(*args, **kwargs):
                        original_remove(*args, **kwargs)
                        raise RuntimeError("injected ledger detach")
                    self.remove = fail_remove
                    try:
                        return original_suspend(self, uid, binding)
                    finally:
                        self.remove = original_remove
                patch.setattr(BatchGenerator, "suspend_ordinary_checkpoint", fail_suspend)
                patch.setattr(dc, "_read_payload", lambda *args: (_ for _ in ()).throw(
                    RuntimeError("injected rollback read")))
            job = engine.submit({"tokens": list(range(1, 19)), "max_tokens": 14,
                "temperature": 0, "mlx_fault": {"kind": "memory_preempt", "after_tokens": 4}})
            terminal = _terminal(job)
            assert "error" in terminal
            assert terminal["status"] == 503
        assert owners and all(owner.closed and owner.control is None for owner in owners)
        assert job.exact_decode_checkpoint is None
        healthy = engine.submit({"tokens": list(range(20, 38)), "max_tokens": 3, "temperature": 0})
        assert len(collect(healthy)[0]) == 3
    finally:
        engine.close()


def test_cancellation_after_capture_closes_parked_checkpoint(monkeypatch):
    patch_host(monkeypatch)
    captured, release = threading.Event(), threading.Event()
    owners = []
    original = BatchGenerator.suspend_ordinary_checkpoint
    def pause(self, *args):
        owner = original(self, *args)
        owners.append(owner)
        captured.set()
        assert release.wait(3)
        return owner
    monkeypatch.setattr(BatchGenerator, "suspend_ordinary_checkpoint", pause)
    engine = _engine(_model())
    try:
        assert engine.ready.wait(10), engine.error
        job = engine.submit({"tokens": list(range(1, 19)), "max_tokens": 14,
            "temperature": 0, "mlx_fault": {"kind": "memory_preempt", "after_tokens": 4}})
        assert captured.wait(3)
        job.cancelled.set()
        release.set()
        assert _terminal(job)["error"] == "cancelled"
        assert owners[0].closed and owners[0].control is None
        assert job.exact_decode_checkpoint is None
        assert engine.counts.get("exact_decode_resumes", 0) == 0
    finally:
        release.set()
        engine.close()


def test_capture_staging_decline_preserves_uninterrupted_output(monkeypatch):
    patch_host(monkeypatch)
    model = _model()
    expected, _, _, _ = _run_server(model)
    original = BatchGenerator.ordinary_checkpoint_staging_bytes
    admission = serving.ensure_admission_headroom
    awaiting_capture = []
    def staging(batch):
        awaiting_capture.append(True)
        return original(batch)
    def headroom(*args, **kwargs):
        if awaiting_capture:
            awaiting_capture.pop()
            return False
        return admission(*args, **kwargs)
    monkeypatch.setattr(BatchGenerator, "ordinary_checkpoint_staging_bytes", staging)
    monkeypatch.setattr(serving, "ensure_admission_headroom", headroom)
    actual, receipt, counts, job = _run_server(model, fault=True)
    assert actual == expected
    assert receipt["preemption"]["fault_unfired"] == "exact_checkpoint_staging_unavailable"
    assert counts.get("exact_decode_checkpoints", 0) == 0
    assert job.exact_decode_checkpoint is None


def test_initial_attachment_metric_failure_cannot_orphan_running_lane(monkeypatch):
    from mlx2.batch_metrics import BatchRuntimeMetrics
    patch_host(monkeypatch)
    model = _model()
    expected, _, _, _ = _run_server(model)
    def failed(*args, **kwargs):
        raise RuntimeError("injected initial attachment metric")
    monkeypatch.setattr(BatchRuntimeMetrics, "lane_attached", failed)
    actual, _receipt, counts, job = _run_server(model)
    assert actual == expected and len(actual) == 14
    assert counts["lane_attached_metric_failures"] == 1
    assert job.events.empty()


def test_real_single_lane_pressure_parks_and_resumes_without_fault_injection(monkeypatch):
    from mlx2.runtime.os_memory import PressureLevel
    patch_host(monkeypatch)
    model = _model()
    expected, _, _, _ = _run_server(model, sample=True)
    state = {"outputs": 0, "critical": False}
    step = BatchGenerator.next
    suspend = BatchGenerator.suspend_ordinary_checkpoint
    def next_step(batch):
        prompts, outputs = step(batch)
        state["outputs"] += len(outputs)
        if state["outputs"] == 4:
            state["critical"] = True
        return prompts, outputs
    def park(batch, *args):
        owner = suspend(batch, *args)
        state["critical"] = False
        return owner
    monkeypatch.setattr(BatchGenerator, "next", next_step)
    monkeypatch.setattr(BatchGenerator, "suspend_ordinary_checkpoint", park)
    actual, receipt, counts, job = _run_server(model, sample=True, pressure=lambda:
        PressureLevel.CRITICAL if state["critical"] else PressureLevel.NORMAL)
    assert actual == expected and len(actual) == 14
    assert counts["exact_decode_checkpoints"] == counts["exact_decode_resumes"] == 1
    assert receipt["preemption"]["events"][0]["trigger"] == "pressure"
    assert receipt["preemption"]["exact_decode_route"] == {
        "implemented": True, "selected": True, "qualified": False,
        "observed_used": True,
        "scope": "same_executor_ordinary_b1_sdpa_capability_cache_v2",
    }
    assert job.exact_decode_checkpoint is None
