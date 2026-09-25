"""A Metal out-of-memory in one generation step must not kill the server.

Series 2026-09-24, Qwen3.8-27B-CRACK-MLX-4bit --ordinary on a 36 GiB M3 Pro
(16K context, 8 GiB cache): the qualifier's 16K priming prefill failed in
``PromptBatch.prompt`` with ``kIOGPUCommandBufferCallbackErrorOutOfMemory``,
the generation worker exited, and /health answered 503 from then on.
"""

import pytest

import mlx.core as mx

from mlx2 import serving
from mlx2.runtime import generate
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

OOM = RuntimeError(
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)."
)
GIB = 1 << 30


def test_allocator_limit_follows_the_admission_reserve():
    # M3 Pro: advisory 28.08, service 3.0, driver 1.003 -> 24.08 GiB, where
    # MLX's default only trimmed its cache at 0.95 x 28.08 = 26.7 GiB.
    assert serving.admission_memory_limit_bytes(28.08, 3.0, 1.003) == int(24.077 * GIB)
    assert serving.admission_memory_limit_bytes(112.0, 16.0, 4.0) == 92 * GIB
    assert serving.admission_memory_limit_bytes(None, 3.0, 1.0) is None
    assert serving.admission_memory_limit_bytes(4.0, 3.0, 1.0) is None


def test_only_metal_memory_failures_are_recovered():
    assert serving.is_device_out_of_memory(OOM)
    assert not serving.is_device_out_of_memory(RuntimeError("shape mismatch"))
    assert not serving.is_device_out_of_memory(ValueError(str(OOM)))


def _inject_one_oom(monkeypatch):
    original = generate.BatchGenerator.next
    state = {"armed": True}

    def next_(self):
        if state["armed"] and getattr(self, "_prompt_batch", None) is not None:
            state["armed"] = False
            raise OOM
        return original(self)

    monkeypatch.setattr(generate.BatchGenerator, "next", next_)
    return state


def test_oom_fails_only_in_flight_requests_and_the_server_keeps_serving(monkeypatch):
    patch_host(monkeypatch)
    state = _inject_one_oom(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    try:
        prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(40)]
        body = {"tokens": prompt, "max_tokens": 4, "temperature": 0}
        failed = collect(engine.submit(dict(body)))
        assert not state["armed"]
        assert failed.get("status") == 503 and "out of memory" in failed["error"]
        assert engine.error is None
        for _ in range(2):
            served = collect(engine.submit(dict(body)))
            assert "error" not in served, served
        assert engine.counts["device_oom_recoveries"] == 1
        assert engine.counts["device_oom_failed_requests"] == 1
    finally:
        engine.close()


def test_a_device_that_cannot_run_after_oom_stops_the_worker(monkeypatch):
    patch_host(monkeypatch)
    _inject_one_oom(monkeypatch)

    real_clear = mx.clear_cache
    lost = {"once": True}

    def clear_cache():
        if lost["once"]:
            lost["once"] = False
            raise RuntimeError("device lost")
        real_clear()

    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    try:
        prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(40)]
        failed = collect(engine.submit({"tokens": prompt, "max_tokens": 4,
                                        "temperature": 0}))
        assert failed.get("status") == 503
        engine.thread.join(30)
        assert engine.error is not None and "Insufficient Memory" in engine.error
        assert engine.counts["device_oom_unrecoverable"] == 1
    finally:
        monkeypatch.undo()
        engine.close()
