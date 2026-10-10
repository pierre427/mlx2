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


@pytest.mark.parametrize(
    "text",
    [
        # MLX's allocator when Metal returns no buffer (backend/metal/allocator.cpp).
        "[malloc] Unable to allocate 1073741824 bytes.",
        # MLX's allocator above a strict memory limit.
        "[metal::malloc] Resource limit (68719476736) exceeded.",
    ],
)
def test_allocator_failures_are_memory_faults(text):
    assert serving.is_device_out_of_memory(RuntimeError(text))
    assert serving.device_fault_kind(RuntimeError(text)) == "out_of_memory"
    assert not serving.is_device_out_of_memory(ValueError(text))


@pytest.mark.parametrize(
    "text",
    [
        # Not MLX's wording (a 2026-10-09 repro invented it): a generic
        # substring must not turn an unrelated step failure into recovery.
        "[metal::malloc] Out of memory",
        "tokenizer: out of memory for vocab table",
        "unable to allocate a lane slot for request r1",
        "resource limit (lanes) reached",
    ],
)
def test_generic_memory_wording_is_not_a_device_fault(text):
    assert not serving.is_device_out_of_memory(RuntimeError(text))
    assert serving.device_fault_kind(RuntimeError(text)) is None


def test_a_buffer_size_limit_is_not_a_memory_fault():
    # A single buffer past Metal's maximum size is a geometry error that
    # abandoning lanes cannot cure.
    text = (
        "[metal::malloc] Attempting to allocate 137438953472 bytes which is "
        "greater than the maximum allowed buffer size of 103079215104 bytes."
    )
    assert not serving.is_device_out_of_memory(RuntimeError(text))


GPU_TIMEOUT = RuntimeError(
    "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
    "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)."
)


def test_a_gpu_watchdog_timeout_is_a_recoverable_device_fault():
    """jundot/omlx#4149: a deep prefill chunk's command buffer was stopped by
    the Metal watchdog.  It abandons that buffer's lanes, exactly like OOM,
    so it takes the same recovery instead of killing the worker."""
    assert serving.device_fault_kind(OOM) == "out_of_memory"
    assert serving.device_fault_kind(GPU_TIMEOUT) == "gpu_timeout"
    assert serving.device_fault_kind(RuntimeError(serving.SIMULATED_GPU_TIMEOUT)) == "gpu_timeout"
    assert not serving.is_device_out_of_memory(GPU_TIMEOUT)
    assert serving.device_fault_kind(RuntimeError("shape mismatch")) is None
    assert serving.device_fault_kind(ValueError(str(GPU_TIMEOUT))) is None


def test_gpu_timeout_mid_step_fails_in_flight_requests_and_keeps_serving(monkeypatch):
    patch_host(monkeypatch)
    original = generate.BatchGenerator.next
    state = {"armed": True}

    def next_(self):
        if state["armed"] and getattr(self, "_prompt_batch", None) is not None:
            state["armed"] = False
            raise GPU_TIMEOUT
        return original(self)

    monkeypatch.setattr(generate.BatchGenerator, "next", next_)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    try:
        prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(40)]
        body = {"tokens": prompt, "max_tokens": 4, "temperature": 0}
        failed = collect(engine.submit(dict(body)))
        assert not state["armed"]
        assert failed.get("status") == 503 and "device GPU timeout" in failed["error"]
        assert engine.error is None
        served = collect(engine.submit(dict(body)))
        assert "error" not in served, served
        assert engine.counts["device_gpu_timeout_events"] == 1
        assert engine.counts["device_gpu_timeout_failed_requests"] == 1
        assert engine.counts["device_gpu_timeout_recoveries"] == 1
        assert engine.counts["device_oom_events"] == 0
    finally:
        engine.close()


@pytest.mark.parametrize("after_tokens", [0, 2])
def test_injected_gpu_timeout_fault_is_recovered(monkeypatch, after_tokens):
    """``mlx_fault: gpu_timeout`` raises Metal's watchdog error from one real
    generation step, so a qualification harness can observe the recovery on
    a served route.  0 fires in prefill, N after N completion tokens."""
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    try:
        prompt = [(5 * i + 3) % (vocab - 2) + 1 for i in range(40)]
        body = {"tokens": prompt, "max_tokens": 6, "temperature": 0}
        reference = collect(engine.submit(dict(body)))
        failed = collect(engine.submit(dict(
            body, mlx_fault={"kind": "gpu_timeout", "after_tokens": after_tokens}
        )))
        assert failed.get("status") == 503
        assert "kIOGPUCommandBufferCallbackErrorTimeout" in failed["error"]
        assert len(failed["tokens"]) == after_tokens
        again = collect(engine.submit(dict(body)))
        assert again["tokens"] == reference["tokens"]
        assert engine.error is None
        assert engine.counts["device_gpu_timeout_recoveries"] == 1
        assert engine.batch_metrics._counters["fault_gpu_timeout"] == 1
    finally:
        engine.close()


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


def _inject_oom_holding(monkeypatch, nbytes):
    """Fail one prefill step with an OOM whose failed frame pins ``nbytes``."""
    original = generate.BatchGenerator.next
    state = {"armed": True}

    def next_(self):
        if state["armed"] and getattr(self, "_prompt_batch", None) is not None:
            state["armed"] = False
            pinned = mx.zeros((nbytes // 4,), dtype=mx.float32)
            mx.eval(pinned)
            raise OOM
        return original(self)

    monkeypatch.setattr(generate.BatchGenerator, "next", next_)
    return state


def test_recovery_releases_what_the_failed_step_pinned(monkeypatch):
    """Series 2026-09-25, Qwen3.8-27B-CRACK on the M3 (040210e1): recovery kept
    the server up, but the failed step's memory stayed held and every later
    request answered 429 for 70 minutes with 0.2 GB free.  The failed frames'
    locals outlive the step through the exception's traceback; recovery must
    release them before it measures and rebuilds."""
    patch_host(monkeypatch)
    monkeypatch.setattr(serving.ServingEngine, "DEVICE_OOM_RELEASE_TOLERANCE_GIB", 32 / 1024)
    _inject_oom_holding(monkeypatch, 256 << 20)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    try:
        prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(40)]
        body = {"tokens": prompt, "max_tokens": 4, "temperature": 0}
        failed = collect(engine.submit(dict(body)))
        assert failed.get("status") == 503
        served = collect(engine.submit(dict(body)))
        assert "error" not in served, served
        assert engine.error is None
        assert engine.counts["device_oom_recoveries"] == 1
        assert engine.counts["device_oom_memory_not_released"] == 0
    finally:
        engine.close()


def test_memory_that_stays_held_after_oom_stops_the_worker(monkeypatch):
    """If device memory does not come back, serving on would only 429."""
    patch_host(monkeypatch)
    monkeypatch.setattr(serving.ServingEngine, "DEVICE_OOM_RELEASE_TOLERANCE_GIB", 32 / 1024)
    leaked = []
    original = generate.BatchGenerator.next
    state = {"armed": True}

    def next_(self):
        if state["armed"] and getattr(self, "_prompt_batch", None) is not None:
            state["armed"] = False
            leaked.append(mx.zeros(((256 << 20) // 4,), dtype=mx.float32))
            mx.eval(leaked[-1])
            raise OOM
        return original(self)

    monkeypatch.setattr(generate.BatchGenerator, "next", next_)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    try:
        prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(40)]
        failed = collect(engine.submit({"tokens": prompt, "max_tokens": 4,
                                        "temperature": 0}))
        assert failed.get("status") == 503
        engine.thread.join(30)
        assert engine.error is not None and "Insufficient Memory" in engine.error
        assert engine.counts["device_oom_memory_not_released"] == 1
    finally:
        leaked.clear()
        engine.close()


def test_qwen38_prefill_transient_is_charged_at_admission():
    """The 16K priming prefill that hit Metal OOM on the M3 now needs room for
    its chunk transient, so admission evicts or refuses instead."""
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C

    budget = Qwen38CacheBudget(
        attention_layers=16, recurrent_layers=48, mtp_layers=0, kv_heads=4,
        head_dim=256, recurrent_heads=48, recurrent_key_heads=16,
        recurrent_key_dim=128, recurrent_value_dim=128, conv_kernel=4,
    )
    controller = C(host_memory_gib=36.0, advisory_gib=28.08,
                   cache_estimator=budget.project,
                   transient_gib_per_lane=budget.transient_gib_per_lane,
                   saturation_lane_cap=4, verification_row_cap=4)
    cold = serving.prefill_transient_gib(
        budget, context_tokens=16128, uncached_tokens=16128, prefill_step=2048
    )
    # 2.0 GiB per full chunk plus one bf16 K/V copy at 16,128 tokens.
    assert cold == pytest.approx(2.0 + 16 * 16128 * 2 * 4 * 256 * 2 / GIB)
    assert serving.prefill_transient_gib(
        budget, context_tokens=16128, uncached_tokens=1, prefill_step=2048
    ) == 0.0
    short = serving.prefill_transient_gib(
        budget, context_tokens=111, uncached_tokens=111, prefill_step=2048
    )
    assert short < 0.15
    required = serving.lane_admission_required_gib(
        controller, context_tokens=16192, draft_depth=0, cache_gib=0.0,
        prefill_gib=cold,
    )
    assert required == pytest.approx(10.34, abs=0.01)
    # Idle M3 before the priming: about 8.3 GiB measured plus the 2.37 GiB
    # APCv2 could evict.  Admission evicts to fit; with less evictable it
    # refuses -- it no longer admits into the prefill that failed.
    state = {"free": 8.3, "evictable": 2.37}

    def evict():
        if state["evictable"] <= 0:
            return False
        state["free"] += state["evictable"]
        state["evictable"] = 0.0
        return True

    admitted, _floor, _required = serving.admit_lane_headroom(
        controller, context_tokens=16192, draft_depth=0, cache_gib=0.0,
        prefill_gib=cold, headroom=lambda: int(state["free"] * GIB),
        reclaim=lambda: False, evict=evict,
        evictable=lambda: int(state["evictable"] * GIB),
    )
    assert admitted and state["evictable"] == 0.0
    admitted, _floor, _required = serving.admit_lane_headroom(
        controller, context_tokens=16192, draft_depth=0, cache_gib=0.0,
        prefill_gib=cold, headroom=lambda: int(9.0 * GIB),
        reclaim=lambda: False, evict=lambda: False, evictable=lambda: 0,
    )
    assert not admitted


def test_memory_the_server_holds_on_purpose_is_not_a_leak(monkeypatch):
    """Multi-LoRA slot tensors and the media feature cache are allocated after
    the load-time baseline; recovery counted them as unreleased memory and
    stopped a worker that had recovered."""
    from types import SimpleNamespace

    patch_host(monkeypatch)
    monkeypatch.setattr(serving.ServingEngine, "DEVICE_OOM_RELEASE_TOLERANCE_GIB", 32 / 1024)
    held = []
    original = generate.BatchGenerator.next
    state = {"armed": True}

    def next_(self):
        if state["armed"] and getattr(self, "_prompt_batch", None) is not None:
            state["armed"] = False
            held.append(mx.zeros(((256 << 20) // 4,), dtype=mx.float32))
            mx.eval(held[-1])
            raise OOM
        return original(self)

    monkeypatch.setattr(generate.BatchGenerator, "next", next_)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2)
    try:
        # The held bytes belong to a cache the server keeps by design.
        engine.adapter.media_feature_cache = SimpleNamespace(bytes=256 << 20)
        prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(40)]
        body = {"tokens": prompt, "max_tokens": 4, "temperature": 0}
        failed = collect(engine.submit(dict(body)))
        assert failed.get("status") == 503
        served = collect(engine.submit(dict(body)))
        assert "error" not in served, served
        assert engine.error is None
        assert engine.counts["device_oom_memory_not_released"] == 0
    finally:
        held.clear()
        engine.close()


def test_multi_lora_reports_its_slot_bytes():
    from types import SimpleNamespace
    import threading

    from mlx2.runtime.multi_lora import MultiLoRAManager

    manager = object.__new__(MultiLoRAManager)
    manager.lock = threading.Lock()
    manager.wrapped = {
        "a": SimpleNamespace(lora_a=mx.zeros((3, 8, 4)), lora_b=mx.zeros((3, 4, 16))),
    }
    assert manager.slot_nbytes() == (3 * 8 * 4 + 3 * 4 * 16) * 4
