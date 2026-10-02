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
