"""Synthetic provider transactions; no target checkpoint or GPU execution."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v41 import execution_policy
from mlx2.adapters.deepseek_v41_residency import DeepSeekV41Residency
from mlx2.runtime.model_residency import ModelResidency

mx.set_default_device(mx.cpu)


class Weights:
    def __init__(self):
        self.live = 10
        self.changed = self.fail_activation = False
        self._expected_stat = {"size": 10}
    def verify_source(self):
        if self.changed: raise ValueError("source changed")
    def resident_bytes(self): return self.live
    def resume_reservation_bytes(self): return 10
    def dense_receipt(self): return {"live": self.live}
    def bind_dense_synchronizer(self, sync): self.sync = sync
    def park(self): self.live = 0
    def activate(self):
        self.live = 10
        if self.fail_activation: raise MemoryError("partial activation")


class Model:
    identity = "synthetic@0123456789abcdef"
    def release_weight_references(self): self.released = True
    def restore_weight_references(self): self.released = False
    def preload(self): return self
    def eval(self): return self


def provider(monkeypatch):
    from mlx2.adapters import deepseek_v41_residency as module
    monkeypatch.setattr(module, "resume_headroom", lambda workspace: {"available_payload_bytes":1 << 40})
    adapter = SimpleNamespace(model=Model(), weights=Weights(), weight_stream=None, engram_tables={},
                              identity={"fingerprint":"0123456789abcdef", "weight_sha256":"abcdef", "path":"/synthetic"},
                              profile="cpu", layout="exact", kernel_policy={}, weight_streaming_request=None,
                              artifact_inspector=lambda path: {"identity":{"fingerprint":"0123456789abcdef"},"weight_stat":{"size":10}})
    value = DeepSeekV41Residency(adapter)
    controller = ModelResidency(value,identity={"checkpoint":"synthetic@0123456789abcdef"},
                                synchronize=lambda:None,measure=lambda:{"active":adapter.weights.live})
    return adapter,value,controller


def test_in_place_parking_retains_binding_and_restores_exact_provider_identity(monkeypatch):
    a,p,c = provider(monkeypatch)
    original = a.model
    binding = p.binding()
    assert c.park()["state"] == "parked" and p.resident_bytes() == 0
    assert c.activate()["state"] == "active" and p.resident_bytes() == 10
    assert a.model is original and p.binding() == binding
    c.park();c.activate()
    assert c.generation == 4
    a.weights = a.model = None
    assert p.resident_bytes() == 0


def test_source_and_policy_changes_refuse_resume_before_allocation(monkeypatch):
    a,p,c = provider(monkeypatch)
    c.park();a.weights.changed = True
    with pytest.raises(ValueError,match="source changed"):c.activate()
    assert p.resident_bytes() == 0
    a.weights.changed = False;a.kernel_policy["changed"] = True
    with pytest.raises(ValueError,match="binding changed"):c.activate()
    assert p.resident_bytes() == 0


def test_low_headroom_and_partial_allocation_remain_parked(monkeypatch):
    from mlx2.adapters import deepseek_v41_residency as module
    a,p,c = provider(monkeypatch)
    c.park()
    with monkeypatch.context() as patch:
        patch.setattr(module,"resume_headroom",lambda workspace:{"available_payload_bytes":0})
        with pytest.raises(MemoryError,match="headroom"):c.activate()
    assert p.resident_bytes() == 0 and c.state == "parked"
    a.weights.fail_activation = True
    with pytest.raises(MemoryError,match="partial activation"):c.activate()
    assert p.resident_bytes() == 0 and c.state == p.state == "parked"
    a.weights.fail_activation = False
    assert c.activate()["state"] == "active"


@pytest.mark.parametrize("value",[True,0,128,16385,"1024"])
def test_dense_budget_requires_bounded_integer_mib(value):
    with pytest.raises(ValueError,match="integer MiB"):
        execution_policy({"deepseek_dense_storage":"paged","deepseek_dense_cache_mib":value})


def test_dense_paging_is_explicit_and_has_bound_budget():
    assert execution_policy({})["deepseek_dense_storage"] == "resident"
    assert execution_policy({"deepseek_dense_storage":"paged"})["deepseek_dense_cache_mib"] == 1024
    with pytest.raises(ValueError,match="requires paged"):
        execution_policy({"deepseek_dense_cache_mib":1024})


def test_retirement_tracks_host_ownership_after_adapter_drops_weights(monkeypatch):
    a,p,c = provider(monkeypatch)
    counters = {"weight_read_inflight":1,"io_finalizer_inflight":0}
    pool = SimpleNamespace(counters=lambda:dict(counters),
                           retirement_receipt=lambda:{"ownership_drained":not any(counters.values())})
    a.weights.dense_pages = SimpleNamespace(pool=pool)
    retained = DeepSeekV41Residency(a)
    a.weights = a.model = None
    assert retained.resident_bytes() == 0
    assert not retained.retirement_ready()
    assert retained.retirement_receipt()["retained_bytes"] > 0
    counters.update(weight_read_inflight=0,io_finalizer_inflight=1)
    assert not retained.retirement_ready()
    counters["io_finalizer_inflight"] = 0
    assert retained.retirement_ready()
    assert retained.retirement_receipt()["retained_bytes"] == 0
