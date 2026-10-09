"""A Metal device fault during a fused GDN probe or dispatch is not a decline.

Serving recovers device faults (out-of-memory, GPU timeout) by abandoning
the lanes in the failed buffer.  A probe that swallowed one and cached the
decline disabled that fused route for the rest of the process (sweep
2026-10-02 G3).  CPU only: kernels are injected.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_fused_gdn_verify as V
from mlx2.runtime.models import qwen36_35b as Q
from mlx2.runtime.models.served_exp import is_device_fault

OOM = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


def test_fault_text_is_a_device_fault():
    assert is_device_fault(RuntimeError(OOM))


def test_qwen36_probe_reraises_and_probes_again(monkeypatch):
    calls = []

    def kernel(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(OOM)
        return (mx.zeros((1,)),)

    Q.probe_qwen36_gdn.cache_clear()
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(Q, "qwen4_fused_gdn_decode", kernel)
    try:
        with pytest.raises(RuntimeError, match="Insufficient Memory"):
            Q.probe_qwen36_gdn(mx.bfloat16, mx.float32)
        assert Q.probe_qwen36_gdn(mx.bfloat16, mx.float32) is not None
    finally:
        Q.probe_qwen36_gdn.cache_clear()


def test_qwen36_probe_still_declines_kernel_refusals(monkeypatch):
    Q.probe_qwen36_gdn.cache_clear()
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: True)

    def refuse(*a, **k):
        raise RuntimeError("Unable to build metal library")

    monkeypatch.setattr(Q, "qwen4_fused_gdn_decode", refuse)
    try:
        assert Q.probe_qwen36_gdn(mx.bfloat16, mx.float32) is None
    finally:
        Q.probe_qwen36_gdn.cache_clear()


def test_batch_verify_probe_reraises_and_probes_again(monkeypatch):
    calls = []
    monkeypatch.setattr(V, "probe_qwen4_fused_gdn_verify", lambda *a, **k: 32)

    def kernel(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(OOM)
        return (mx.zeros((1,)),) * 5

    monkeypatch.setattr(V, "qwen4_fused_gdn_batch_verify", kernel)
    monkeypatch.setattr(V, "_PROBED_BATCH", {})
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        V.probe_qwen4_fused_gdn_batch_verify(mx.bfloat16, 3, compact=False)
    assert V.probe_qwen4_fused_gdn_batch_verify(mx.bfloat16, 3, compact=False) == 32
    assert len(calls) == 2


def _stub():
    from test_qwen36_decode_wins import _gdn_stub

    return _gdn_stub()


@pytest.mark.parametrize("mode", ["batch_decode", "batch_verify"])
def test_qwen36_dispatch_reraises_device_fault(monkeypatch, mode):
    layer, cache, o = _stub()
    monkeypatch.setattr(Q, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(Q, "served_silu_refusal", lambda: None)

    def probe(*a, **k):
        raise RuntimeError(OOM)

    monkeypatch.setattr(Q, "probe_qwen36_gdn", probe)
    getattr(layer, f"set_fused_gdn_{mode}_mode")("row_exact")
    if mode == "batch_decode":
        cache.speculating = False
        o = {k: v[:, :1] for k, v in o.items() if k in ("qkv", "z", "b", "a")}
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        layer._try_fused_decode(o["qkv"], o["z"], o["b"], o["a"], None, cache)


# Follow-up (sweep 2026-10-08): the fp32 decode probe, which every other GDN
# probe starts from, and the B=1 verify, catch-up and replay probes still
# cached ``None`` on a device fault, so one fault turned off every fused GDN
# route (the fp16-state probe included) until restart; and Flash-Next's
# dispatch sites turned a re-raised probe fault into a stock fallback, so it
# still never reached serving's recovery.
from mlx2.runtime.models import qwen4_exp as E  # noqa: E402
from mlx2.runtime.models import qwen4_fused_gdn as F  # noqa: E402
from mlx2.runtime.models import qwen4_fused_gdn_prefill as P  # noqa: E402


def _faulting_once(n_outputs, calls):
    def kernel(*_a, **k):
        calls.append(k.get("threadgroup_y"))
        if len(calls) == 1:
            raise RuntimeError(OOM)
        return tuple(mx.zeros((1,)) for _ in range(n_outputs))

    return kernel


@pytest.fixture
def fp32_probe(monkeypatch):
    monkeypatch.setattr(F, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(F, "_PROBED_THREADGROUP_Y", None)
    monkeypatch.setattr(F, "_PROBED_ST16", {})
    monkeypatch.setattr(F, "fused_gdn_runtime_supported", lambda: True)


def test_fp32_decode_probe_reraises_and_probes_again(monkeypatch, fp32_probe):
    calls = []
    monkeypatch.setattr(F, "qwen4_fused_gdn_decode", _faulting_once(3, calls))
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        F.probe_qwen4_fused_gdn_decode(mx.bfloat16)
    assert F._PROBE_COMPLETE is False
    assert F.probe_qwen4_fused_gdn_decode(mx.bfloat16) == 32
    assert calls == [32, 32]


def test_fp32_decode_probe_still_steps_past_kernel_refusals(monkeypatch, fp32_probe):
    def kernel(*_a, threadgroup_y, **_k):
        if threadgroup_y == 32:
            raise RuntimeError("Unable to build metal library")
        return (mx.zeros((1,)),) * 3

    monkeypatch.setattr(F, "qwen4_fused_gdn_decode", kernel)
    assert F.probe_qwen4_fused_gdn_decode(mx.bfloat16) == 16
    assert F._PROBE_COMPLETE is True


def test_fp16_probe_is_not_poisoned_by_a_fault_in_its_fp32_start(
    monkeypatch, fp32_probe
):
    calls = []
    monkeypatch.setattr(F, "qwen4_fused_gdn_decode", _faulting_once(3, calls))
    monkeypatch.setattr(
        F, "qwen4_fused_gdn_batch_decode", lambda *a, **k: (mx.zeros((1,)),) * 3
    )
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        F._probe_st16_decode(mx.bfloat16)
    assert "decode" not in F._PROBED_ST16
    assert F._probe_st16_decode(mx.bfloat16) == 32


@pytest.mark.parametrize(
    "probe_name, cache_name, kernel_name, n_outputs",
    [
        ("probe_qwen4_fused_gdn_verify", "_PROBED_STEPS", "qwen4_fused_gdn_verify", 5),
        ("probe_qwen4_fused_gdn_catchup", "_PROBED_CATCHUP_STEPS",
         "qwen4_fused_gdn_catchup", 3),
        ("probe_qwen4_fused_gdn_replay_verify", "_PROBED_REPLAY_STEPS",
         "qwen4_fused_gdn_replay_verify", 6),
    ],
)
def test_b1_multitoken_probes_reraise_and_probe_again(
    monkeypatch, probe_name, cache_name, kernel_name, n_outputs
):
    calls = []
    monkeypatch.setattr(V, cache_name, {})
    monkeypatch.setattr(V, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(V, "probe_qwen4_fused_gdn_decode", lambda *a, **k: 32)
    monkeypatch.setattr(V, kernel_name, _faulting_once(n_outputs, calls))
    monkeypatch.setattr(V, "qwen4_fused_gdn_reconstruct", lambda *a, **k: mx.zeros((1,)))
    probe = getattr(V, probe_name)
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        probe(mx.bfloat16, 3)
    assert getattr(V, cache_name) == {}
    assert probe(mx.bfloat16, 3) == 32
    assert calls == [32, 32]


class _Cache(list):
    speculating = False

    def rollback_spans(self, steps, mask):
        return ()

    def record_rollback(self, *args, **kwargs):
        pass

    def advance(self, steps):
        pass


def _accept(*args, **kwargs):
    return F.FusedGdnAdmission(True, "eligible")


def _fault(*args, **kwargs):
    raise RuntimeError(OOM)


def _inputs(rows, width):
    return (mx.zeros((rows, width, 8)), mx.zeros((rows, width, 8)),
            mx.zeros((rows, width, 4)), mx.zeros((rows, width, 4)))


@pytest.fixture
def flash_next_layer(monkeypatch):
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

    model, _vocab = tiny_qwen4_mtp()
    layer = next(
        layer.linear_attn
        for layer in model.language_model.model.layers
        if getattr(layer, "linear_attn", None) is not None
    )
    layer.set_fused_gdn_decode_mode("fused")
    layer.set_fused_gdn_verify_mode("fused")
    layer.set_fused_gdn_prefill_mode("fused")
    for name in (
        "admit_qwen4_fused_gdn_decode", "admit_qwen4_fused_gdn_verify",
        "admit_qwen4_fused_gdn_batch_decode", "admit_qwen4_fused_gdn_batch_verify",
    ):
        monkeypatch.setattr(E, name, _accept)
    monkeypatch.setattr(P, "admit_qwen4_fused_gdn_prefill", _accept)
    monkeypatch.setattr(E, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(P, "runtime_supported", lambda: True)
    monkeypatch.setattr(E, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(E, "batch_verify_row_steps", lambda *a: (3, 3))
    for name in (
        "probe_qwen4_fused_gdn_decode", "probe_qwen4_fused_gdn_verify",
        "probe_qwen4_fused_gdn_replay_verify", "probe_qwen4_fused_gdn_catchup",
        "probe_qwen4_fused_gdn_batch_verify",
    ):
        monkeypatch.setattr(E, name, _fault)
    monkeypatch.setattr(P, "qwen4_gdn_prefill_prework", _fault)
    return layer


@pytest.mark.parametrize(
    "route, rows, width",
    [
        ("decode", 1, 1),
        ("batch_decode", 2, 1),
        ("verify", 1, 3),
        ("catchup", 1, 3),
        ("batch_verify", 2, 3),
        ("prefill", 1, 64),
    ],
)
def test_flash_next_dispatch_reraises_device_fault(flash_next_layer, route, rows, width):
    layer = flash_next_layer
    state = [mx.zeros((rows, 3, 8)), mx.zeros((rows, 4, 8, 8))]
    cache = _Cache(state)
    call = {
        "decode": layer._try_fused_decode,
        "batch_decode": layer._try_fused_batch_decode,
        "verify": layer._try_fused_verify,
        "catchup": lambda *a: layer._try_fused_verify(*a, catchup=True),
        "batch_verify": layer._try_fused_batch_verify,
        "prefill": layer._try_fused_prefill,
    }[route]
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        call(*_inputs(rows, width), None, cache)
    assert layer.fused_gdn_decode_fallbacks == 0
    assert layer.fused_gdn_prefill_fallbacks == 0


def test_flash_next_dispatch_still_falls_back_on_kernel_refusal(
    flash_next_layer, monkeypatch
):
    def refuse(*a, **k):
        raise RuntimeError("Unable to build metal library")

    monkeypatch.setattr(E, "probe_qwen4_fused_gdn_decode", refuse)
    layer = flash_next_layer
    cache = _Cache([mx.zeros((1, 3, 8)), mx.zeros((1, 4, 8, 8))])
    assert layer._try_fused_decode(*_inputs(1, 1), None, cache) is None
    assert layer.fused_gdn_decode_last_fallback == (
        "Metal kernel dispatch failed: RuntimeError"
    )
