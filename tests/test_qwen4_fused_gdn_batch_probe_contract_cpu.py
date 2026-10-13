"""The snapshot batched-verify probe carries the Qwen4 fused-GDN contract.

Sweep 2026-10-12 L4-F1: since e3c19168e a missing contract refuses, and the
snapshot (``compact=False``) arm of ``probe_qwen4_fused_gdn_batch_verify``
passed none to the B=1 verify probe or the batch build, so row-exact batched
verify with snapshot rollback always declined ("Metal kernel probe
declined").  CPU only: the runtime check, decode probe and kernel builders are
stubbed, so no Metal work is dispatched.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_fused_gdn_verify as V
from mlx2.runtime.models.qwen_gdn_contracts import QWEN4_GDN_CONTRACT


@pytest.fixture
def stubbed(monkeypatch):
    calls = []

    def build(*_args, **kwargs):
        calls.append(kwargs.get("contract"))
        if kwargs.get("contract") is None and "architecture" not in kwargs:
            # The real builders refuse a missing contract this way.
            raise ValueError("missing fused GDN contract")
        return tuple(mx.zeros((1,)) for _ in range(6))

    monkeypatch.setattr(V, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(V, "probe_qwen4_fused_gdn_decode", lambda *a, **k: 32)
    monkeypatch.setattr(V, "qwen4_fused_gdn_verify", build)
    monkeypatch.setattr(V, "qwen4_fused_gdn_batch_verify", build)
    monkeypatch.setattr(V, "qwen4_fused_gdn_replay_verify", build)
    monkeypatch.setattr(
        V,
        "qwen4_fused_gdn_batch_replay_verify",
        lambda *a, **k: tuple(mx.zeros((1,)) for _ in range(6)),
    )
    monkeypatch.setattr(V, "qwen4_fused_gdn_reconstruct", lambda *a, **k: mx.zeros((1,)))
    monkeypatch.setattr(V, "_PROBED_STEPS", {})
    monkeypatch.setattr(V, "_PROBED_BATCH", {})
    monkeypatch.setattr(V, "_PROBED_REPLAY_STEPS", {})
    monkeypatch.setattr(V, "_PROBED_DYNAMIC_REPLAY_STEPS", {})
    return calls


@pytest.mark.parametrize("compact", [True, False])
def test_batch_verify_probe_accepts_both_rollback_forms(stubbed, compact):
    assert V.probe_qwen4_fused_gdn_batch_verify(mx.bfloat16, 4, compact=compact) == 32


def test_snapshot_probe_threads_the_contract_to_probe_and_build(stubbed):
    assert V.probe_qwen4_fused_gdn_batch_verify(
        mx.bfloat16, 5, compact=False, contract=QWEN4_GDN_CONTRACT
    ) == 32
    assert stubbed and all(c is QWEN4_GDN_CONTRACT for c in stubbed)


def test_qwen4_layer_passes_its_contract_to_the_batch_probe():
    import inspect

    from mlx2.runtime.models import qwen4_exp

    source = inspect.getsource(qwen4_exp)
    start = source.index("threadgroup_y = probe_qwen4_fused_gdn_batch_verify(")
    call = source[start : source.index(")", start)]
    assert "contract=QWEN4_GDN_CONTRACT" in call
