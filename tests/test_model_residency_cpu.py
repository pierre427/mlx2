"""Synthetic CPU ownership/residency contracts; no released-model qualification."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import threading
import time

import pytest

from mlx2.runtime.model_residency import ModelResidency, ResidencyBudget


@pytest.fixture(autouse=True)
def evidence(record_property):
    record_property("checkpoint", "synthetic-residency@"+hashlib.sha256(b"residency-contract-v1").hexdigest())
    record_property("source_revision", subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    record_property("evidence_kind", "synthetic_cpu_not_model_qualification")


class Provider:
    def __init__(self, path, payload=b"native-exact-bytes"):
        self.path = path
        path.write_bytes(payload)
        self.payload = payload
        self.required = len(payload)
        self.calls = []
        self.fail_admit = self.fail_load = self.fail_park = False

    def binding(self):
        return {"source": hashlib.sha256(self.path.read_bytes()).hexdigest(), "profile": "exact"}

    def resident_bytes(self): return len(self.payload)
    def reservation_bytes(self): return self.required
    def receipt(self): return {"logical_payload_bytes": len(self.payload)}
    def retirement_receipt(self): return {"ownership_drained":True,"retained_bytes":0}

    def park(self):
        self.calls.append(("park", threading.get_ident()))
        if self.fail_park: raise RuntimeError("park failure")
        self.payload = b""
        return self.receipt()

    def admit(self):
        self.calls.append(("admit", threading.get_ident()))
        if self.fail_admit: raise MemoryError("fresh host headroom exhausted")

    def activate(self):
        self.calls.append(("activate", threading.get_ident()))
        self.payload = self.path.read_bytes()
        if self.fail_load: raise MemoryError("partial allocation failure")
        return self.receipt()


def controller(provider, *, budget=None):
    return ModelResidency(provider, identity={"artifact":"fixture", "runtime":"exact"},
                          synchronize=lambda: None,
                          measure=lambda: {"active_bytes":1000+provider.resident_bytes()*2, "cached_bytes":0},
                          budget=budget)


def test_fifo_protects_active_owner_and_does_not_bypass_waiter():
    pool=ResidencyBudget(12)
    a=pool.request("a",8); assert pool.try_grant(a)
    b=pool.request("b",7); c=pool.request("c",4)
    assert not pool.try_grant(b) and not pool.try_grant(c)
    assert pool.receipt()["reserved_bytes"]==8
    pool.release(a)
    assert pool.try_grant(b) and pool.try_grant(c)
    assert pool.receipt()["reserved_bytes"]==11
    pool.release(b);pool.release(c)
    assert pool.receipt()["reserved_bytes"]==0


def test_expired_waiters_and_cancellation_are_bounded():
    clock=[0.0];pool=ResidencyBudget(10,max_waiters=2,clock=lambda:clock[0])
    head=pool.request("head",10,timeout=1)
    tail=pool.request("tail",10,timeout=2)
    with pytest.raises(RuntimeError,match="full"):pool.request("overflow",1)
    clock[0]=1.5
    assert pool.try_grant(tail) and head.closed
    with pytest.raises(ValueError,match="closed"):pool.try_grant(head)
    pool.release(tail)
    expiry=pool.request("expiry",1,timeout=1);clock[0]=3
    with pytest.raises(TimeoutError):pool.try_grant(expiry)
    assert pool.receipt()["waiting_owners"]==0
    cancel=pool.request("cancel",2);pool.release(cancel);pool.release(cancel)


@pytest.mark.parametrize("amount", [0,True,-1,11,1.5])
def test_bad_reservation_bounds_are_refused(amount):
    with pytest.raises(ValueError):ResidencyBudget(10).request("owner",amount)


def test_exact_roundtrip_keeps_logical_and_physical_receipts_distinct(tmp_path):
    provider=Provider(tmp_path/"weights");expected=provider.payload
    pool=ResidencyBudget(100);model=controller(provider,budget=pool)
    parked=model.park();assert model.state=="parked" and provider.payload==b""
    assert parked["last"]["logical_payload_bytes_before"]==len(expected)
    assert parked["last"]["memory_before"]["active_bytes"]-parked["last"]["memory_after"]["active_bytes"]==2*len(expected)
    assert pool.receipt()["reserved_bytes"]==0
    restored=model.activate()
    assert provider.payload==expected and restored["generation"]==2
    assert [c[0] for c in provider.calls]==["park","admit","activate"]
    assert restored["binding"]==parked["binding"]
    provider.park();model.retire();assert pool.receipt()["reserved_bytes"]==0


def test_source_change_refuses_before_release_or_reload(tmp_path):
    provider=Provider(tmp_path/"weights");model=controller(provider)
    provider.path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="source/profile"):model.park()
    assert provider.payload and not provider.calls
    provider.path.write_bytes(provider.payload);model.park()
    provider.path.write_bytes(b"changed again")
    with pytest.raises(ValueError,match="source/profile"):model.activate()
    assert model.state=="parked" and not provider.payload


@pytest.mark.parametrize("failure",["fail_admit","fail_load"])
def test_activation_failure_cleans_partial_payload_and_reservation(tmp_path,failure):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    model=controller(provider,budget=pool);model.park();setattr(provider,failure,True)
    with pytest.raises(MemoryError):model.activate()
    assert model.state=="parked" and not provider.payload
    assert pool.receipt()["reserved_bytes"]==0
    setattr(provider,failure,False);model.activate();assert provider.payload


def test_uncertain_activation_cleanup_retains_capacity_and_closes_route(tmp_path):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    model=controller(provider,budget=pool);model.park()
    provider.fail_load=provider.fail_park=True
    with pytest.raises(MemoryError):model.activate()
    assert model.state=="failed" and pool.receipt()["reserved_bytes"]==provider.required
    with pytest.raises(RuntimeError,match="reload"):model.activate()
    with pytest.raises(RuntimeError,match="retains"):model.retire()


@pytest.mark.parametrize("reading",[None,{"active_bytes":-1}])
def test_measurement_failure_after_grant_releases_capacity(tmp_path,reading):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    model=controller(provider,budget=pool);model.park();model.measure=lambda:reading
    with pytest.raises(ValueError):model.activate()
    assert model.state=="parked" and not provider.payload
    assert pool.receipt()["reserved_bytes"]==0
    assert model.last["status"]=="failed"


def test_initial_reservation_failure_cancels_pending_ticket(tmp_path,monkeypatch):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    monkeypatch.setattr(pool,"try_grant",lambda _:(_ for _ in ()).throw(RuntimeError("grant failed")))
    with pytest.raises(RuntimeError,match="grant failed"):controller(provider,budget=pool)
    assert pool.receipt()["waiting_owners"]==0


@pytest.mark.parametrize("budgeted",[False,True])
def test_teardown_keeps_capacity_until_quarantined_owner_drains(tmp_path,budgeted):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100) if budgeted else None
    model=controller(provider,budget=pool)
    provider.payload=b""  # Adapter detached tensor owners; read staging remains.
    provider.retirement_receipt=lambda:{"ownership_drained":False,"retained_bytes":7}
    with pytest.raises(RuntimeError,match="asynchronous ownership"):model.retire()
    assert model.state=="failed" and model.last["operation"]=="retire"
    if pool is not None:
        assert pool.receipt()["reserved_bytes"]==provider.required
    provider.retirement_receipt=lambda:{"ownership_drained":True,"retained_bytes":0}
    result=model.retire()
    assert result["state"]=="closed" and result["last"]["status"]=="completed"
    if pool is not None:
        assert pool.receipt()["reserved_bytes"]==0


@pytest.mark.parametrize("proof",[None,{}, {"ownership_drained":1,"retained_bytes":0},
                                  {"ownership_drained":True,"retained_bytes":-1},
                                  {"ownership_drained":True,"retained_bytes":7},
                                  {"ownership_drained":False,"retained_bytes":0}])
def test_missing_or_uncertain_retirement_proof_keeps_capacity(tmp_path,proof):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    model=controller(provider,budget=pool);provider.payload=b""
    provider.retirement_receipt=lambda:proof
    with pytest.raises((ValueError,RuntimeError)):model.retire()
    assert model.state=="failed" and pool.receipt()["reserved_bytes"]==provider.required


def test_retirement_inspection_failure_keeps_capacity(tmp_path):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    model=controller(provider,budget=pool);provider.payload=b""
    def fail():
        raise RuntimeError("ownership inspection failed")
    provider.retirement_receipt=fail
    with pytest.raises(RuntimeError,match="inspection failed"):model.retire()
    assert pool.receipt()["reserved_bytes"]==provider.required


def test_provider_without_retirement_contract_refused_before_reservation(tmp_path):
    provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    provider.retirement_receipt=None
    with pytest.raises(ValueError,match="complete model residency provider"):
        controller(provider,budget=pool)
    assert pool.receipt()["reserved_bytes"]==pool.receipt()["waiting_owners"]==0


def test_two_providers_share_capacity_and_cancel_pending_activation(tmp_path):
    a,b=Provider(tmp_path/"a",b"a"*8),Provider(tmp_path/"b",b"b"*8)
    pool=ResidencyBudget(16);ma,mb=controller(a,budget=pool),controller(b,budget=pool)
    ma.park();mb.park()
    foreign=pool.request("workspace",12);assert pool.try_grant(foreign)
    assert ma.activate()["status"]=="waiting_for_budget"
    assert mb.activate()["status"]=="waiting_for_budget"
    mb.cancel_activation()
    assert pool.receipt()["waiting_owners"]==1
    pool.release(foreign);ma.activate();assert a.payload and not b.payload
    mb.activate();assert pool.receipt()["reserved_bytes"]==16


def test_foreign_evaluation_thread_cannot_park(tmp_path):
    provider=Provider(tmp_path/"weights");model=controller(provider)
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError,match="evaluation owner"):
            executor.submit(model.park).result()
    assert provider.payload and not provider.calls


@pytest.mark.parametrize("export_epoch",[False,True])
def test_live_worker_parks_and_reactivates_on_owner_with_exact_apcv2(tmp_path,monkeypatch,export_epoch):
    import mlx.core as mx
    from route_harness import collect,make_engine,patch_host,tiny_qwen38_mtp
    from mlx2.serving import AdmissionClosed
    mx.set_default_device(mx.cpu);patch_host(monkeypatch)
    provider=Provider(tmp_path/"provider-bytes")
    class AdapterMixin:
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.model_residency=provider
        def close(self):
            provider.park();super().close()
    model,vocab=tiny_qwen38_mtp()
    engine=make_engine(model,vocab,adapter_mixin=AdapterMixin,cache_dir=str(tmp_path/"apc"),
                       apc_persist_dir=str(tmp_path/"apc") if export_epoch else None)
    try:
        request={"tokens":[5,6,7,8],"max_tokens":3,"temperature":0}
        first=collect(engine.submit(dict(request)))
        assert first.get("error") is None
        engine.park_model(drain_timeout_seconds=5)
        assert engine.wait_for_quiesce(10)
        parked=engine.service_state();assert parked["state"]=="parked",parked
        assert parked["last_transition"]["result"]["suspend"]["resident_entries_after"]==0
        if export_epoch:
            receipt=engine.export_suspension("parked-exact-epoch")
            assert receipt["clock_frozen"] and receipt["entries"]>0
        with pytest.raises(AdmissionClosed):engine.submit(dict(request))
        assert engine.resume()["state"]=="activating"
        assert engine.wait_for_quiesce(10)
        assert engine.service_state()["state"]=="serving"
        assert not getattr(engine.apc,"_suspension_clock",None)
        second=collect(engine.submit(dict(request)))
        assert second["tokens"]==first["tokens"]
        assert engine.apc.apc_stats["idle_disk"]["restores"]>=1
        assert all(owner==engine.thread.ident for _,owner in provider.calls)
        assert engine.error is None
    finally:
        engine.close()


def test_activation_source_failure_keeps_live_worker_closed(tmp_path,monkeypatch):
    from test_serving_quiesce import _engine
    engine=_engine();provider=Provider(tmp_path/"weights")
    engine._model_residency=controller(provider);engine._model_residency.park()
    engine._service_state="parked";provider.path.write_bytes(b"replacement")
    engine.activate_model();engine._worker_activate_model(None)
    state=engine.service_state()
    assert state["state"]=="parked" and state["last_transition"]["result"]["status"]=="activation_failed"
    assert not provider.payload and engine.wait_for_quiesce(0)


def test_cancelled_activation_never_reopens_admission(tmp_path):
    from test_serving_quiesce import _engine
    engine=_engine();provider=Provider(tmp_path/"weights")
    engine._model_residency=controller(provider);engine._model_residency.park();engine._service_state="parked"
    engine.activate_model();engine.cancel_model_activation();engine._worker_activate_model(None)
    assert engine.service_state()["state"]=="parked" and not provider.payload
    assert engine.service_state()["last_transition"]["result"]["status"]=="cancelled"


def test_resume_waiting_for_owned_park_queues_activation(tmp_path):
    from test_serving_quiesce import _engine
    engine=_engine();provider=Provider(tmp_path/"weights")
    engine._model_residency=controller(provider);engine._model_residency.park()
    engine._service_state="draining";engine._quiesce_worker_owned=True
    engine._quiesce_complete.clear()
    def complete_park(timeout=None):
        engine._service_state="parked"
        engine._quiesce_worker_owned=False
        engine._quiesce_complete.set()
        return True
    engine._quiesce_complete.wait=complete_park
    assert engine.resume()["state"]=="activating"
    assert not provider.payload


def test_apc_thaw_failure_reparks_allocated_weights(tmp_path):
    from types import SimpleNamespace
    from test_serving_quiesce import _engine
    engine=_engine();provider=Provider(tmp_path/"weights");pool=ResidencyBudget(100)
    model=engine._model_residency=controller(provider,budget=pool);model.park()
    engine._service_state="parked"
    engine._parked_apc_suspend={"failures":0,"resident_entries_after":0}
    def fail_thaw():
        raise RuntimeError("APCv2 clock restoration failed")
    engine.apc=SimpleNamespace(_suspension_clock=True,thaw_suspension=fail_thaw)
    engine.activate_model();engine._worker_activate_model(None)
    state=engine.service_state()
    assert state["state"]=="parked" and model.state=="parked" and not provider.payload
    assert pool.receipt()["reserved_bytes"]==0
    assert state["last_transition"]["result"]["suspend"]==engine._parked_apc_suspend


def test_residency_http_contract(admin_server):
    from test_server_quiesce import _post
    engine,base=admin_server
    engine.park_model=lambda **kw: {"state":"draining","options":kw}
    engine.activate_model=lambda: {"state":"activating"}
    engine.cancel_model_activation=lambda: {"state":"parked"}
    for route,state in (("park","draining"),("activate","activating"),("cancel","parked")):
        with _post(base,"/v1/admin/residency/"+route,b"{}") as response:
            assert response.status==202 and json.load(response)["state"]==state


from test_server_quiesce import admin_server  # noqa: E402,F401 - shared HTTP fixture
