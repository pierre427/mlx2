"""Exact-shape research bootstrap is distinct from serving qualification."""

import json
from pathlib import Path

import pytest

from mlx2.runtime.paged_atomic_owner import PagedAtomicRequestOwner
from mlx2.runtime.paged_pack_price import load_research_calibration
from mlx2.runtime.paged_pack_scheduler import PrefillOffer, PrefillOption
from mlx2.runtime.paged_request_route import RouteCapability, coordinate_paged_request
from mlx2.runtime.paged_request_transaction import CandidateRequest

RAW = (Path(__file__).resolve().parents[1] /
       "qualification/runs/varlen-live-q1-20261003/attempt-3-calibration.json")

pytestmark = pytest.mark.skipif(
    not RAW.is_file(),
    reason="private source-bound research calibration is not in the public export",
)


def _price():
    data = json.loads(RAW.read_text())
    return load_research_calibration(RAW, live_identity=data["identity"],
                                     context_tokens=(63,))


def _coordinate(price, *, rows=63, context=(63,), permit_bootstrap=True,
                identity=None, offers=None):
    owner = PagedAtomicRequestOwner("r1", {"kv": ["base"]},
                                    supported_planes=("kv",), enabled=True)
    request = CandidateRequest(1, "r1", rows, ("kv",))
    if offers is None:
        offers = (PrefillOffer(1, (PrefillOption(rows, 0),)),)
    called = []

    def run(candidate):
        called.append(True)
        candidate.stage("kv", [f"row-{i}" for i in range(rows)])
        return rows

    receipt = coordinate_paged_request(
        request, owner, RouteCapability(True, False, ("kv",), price.profile_id),
        (), offers, price=price,
        live_identity=dict(price.identity) if identity is None else identity,
        context_tokens=context, row_capacity=64, free_pages=64,
        run_candidate=run, permit_candidate=True,
        permit_research_calibration=permit_bootstrap)
    return receipt, called, owner


def test_exact_calibration_loads_as_research_only_and_admits_one_private_prompt():
    price = _price()
    assert price.cold_bound_ms == 269.384166
    assert price.q1_bound_ms == 162.489083
    receipt, called, owner = _coordinate(price)
    assert receipt.selected and receipt.published and receipt.observed_used
    assert receipt.qualified is False and called == [True]
    assert owner.snapshot().generation == 1


@pytest.mark.parametrize("change,reason", [
    ({"permit_bootstrap": False}, "research_calibration_shape_refused"),
    ({"rows": 64}, "research_calibration_shape_refused"),
    ({"context": (64,)}, "research_calibration_shape_refused"),
    ({"offers": ()}, "research_calibration_shape_refused"),
    ({"identity": {"host": "drift"}}, "source_bound_price_missing"),
])
def test_bootstrap_refusals_never_run_candidate(change, reason):
    receipt, called, owner = _coordinate(_price(), **change)
    assert receipt.reason == reason and not receipt.selected
    assert not receipt.published and not called and owner.snapshot().generation == 0


def test_tampered_calibration_and_stale_source_refuse(tmp_path):
    data = json.loads(RAW.read_text())
    stale = dict(data["identity"])
    stale["source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="identity"):
        load_research_calibration(RAW, live_identity=stale, context_tokens=(63,))
    with pytest.raises(ValueError, match="context"):
        load_research_calibration(RAW, live_identity=data["identity"],
                                  context_tokens=(64,))
    data["cases"][0]["native_q1_step_ms"][0] = 0.1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        load_research_calibration(path, live_identity=data["identity"],
                                  context_tokens=(63,))


def test_explicit_serving_bootstrap_receipt_and_cold_policy(monkeypatch, tmp_path):
    import sys
    from threading import RLock
    from types import ModuleType, SimpleNamespace

    from mlx2 import serving
    from mlx2.runtime import paged_native_batch_lifecycle, paged_price_identity

    native = ModuleType("_paged_kv_native")
    native.__file__ = str(tmp_path / "native.so")
    monkeypatch.setitem(sys.modules, "_paged_kv_native", native)
    monkeypatch.setattr(paged_price_identity, "cached_live_price_identity",
                        lambda *_, **__: dict(_price().identity))
    probe = SimpleNamespace(native_probe=SimpleNamespace(published=True),
                            reason="accepted")
    monkeypatch.setattr(paged_native_batch_lifecycle, "probe_queued_native_qwen3",
                        lambda *_, **__: probe)
    monkeypatch.setattr(paged_native_batch_lifecycle,
                        "prepare_queued_native_first_response", lambda *_, **__: object())
    owner = SimpleNamespace(close=lambda: None, reap_retired=lambda: None,
                            reap_quarantine=lambda: None)
    candidate = SimpleNamespace(backend=SimpleNamespace(
        writer=SimpleNamespace(pool=SimpleNamespace(free_count=64))))
    adapter = SimpleNamespace(
        identity={"path": str(tmp_path), "fingerprint": "r1"},
        create_native_paged_qwen3_request=lambda **_: (owner, candidate))
    passed = []

    def install(*args, **kwargs):
        passed.append(kwargs)
        return {"selected": True, "qualified": False,
                "observed_used": False, "reason": "native_installed"}

    batch = SimpleNamespace(install_native_queued=install)
    job = SimpleNamespace(uid=1, preempted=False, cached_tokens=0,
        request={"paged_native_qwen3": True,
                                   "skip_writing_prefix_cache": True,
                                   "temperature": 0},
                          cancelled=SimpleNamespace(is_set=lambda: False))
    result = serving.install_explicit_native_qwen3_request(
        batch, adapter, job, prompt_tokens=[1] * 63, maximum=2,
        lifecycle_lock=RLock(), price_path=RAW,
        manifest_path="manifest", mlx_wheel_path="wheel")
    assert result["selected"] and not result["qualified"]
    assert not result["observed_used"]
    assert result["price_provenance"] == "research_calibrated"
    assert passed[0]["price_provenance"] == "research_calibrated"
    assert passed[0]["price_evidence_sha256"] == _price().evidence_sha256
    for prompt, maximum in (([1] * 64, 2), ([1] * 63, 3)):
        with pytest.raises(ValueError, match="cold context63 B1"):
            serving.install_explicit_native_qwen3_request(
                batch, adapter, job, prompt_tokens=prompt, maximum=maximum,
                lifecycle_lock=RLock(), price_path=RAW,
                manifest_path="manifest", mlx_wheel_path="wheel")
    job.cached_tokens = 1
    with pytest.raises(ValueError, match="nonempty token prompt"):
        serving.install_explicit_native_qwen3_request(
            batch, adapter, job, prompt_tokens=[1] * 63, maximum=2,
            lifecycle_lock=RLock(), price_path=RAW,
            manifest_path="manifest", mlx_wheel_path="wheel")
    job.cached_tokens = 0
    job.request["top_p"] = 0.9
    with pytest.raises(ValueError, match="exact greedy sampling"):
        serving.install_explicit_native_qwen3_request(
            batch, adapter, job, prompt_tokens=[1] * 63, maximum=2,
            lifecycle_lock=RLock(), price_path=RAW,
            manifest_path="manifest", mlx_wheel_path="wheel")
    job.request.pop("top_p")
    job.request["skip_writing_prefix_cache"] = False
    with pytest.raises(ValueError, match="explicit no-write request"):
        serving.install_explicit_native_qwen3_request(
            batch, adapter, job, prompt_tokens=[1] * 63, maximum=2,
            lifecycle_lock=RLock(), price_path=RAW,
            manifest_path="manifest", mlx_wheel_path="wheel")


def test_explicit_route_refuses_missing_native_binary_before_price(monkeypatch):
    import sys
    from threading import RLock
    from types import SimpleNamespace

    from mlx2 import serving

    monkeypatch.setitem(sys.modules, "_paged_kv_native", None)
    job = SimpleNamespace(uid=1, preempted=False, cached_tokens=0,
                          request={"paged_native_qwen3": True,
                                   "skip_writing_prefix_cache": True,
                                   "temperature": 0},
                          cancelled=SimpleNamespace(is_set=lambda: False))
    with pytest.raises(ValueError, match="kernel is unavailable"):
        serving.install_explicit_native_qwen3_request(
            object(), object(), job, prompt_tokens=[1] * 63, maximum=2,
            lifecycle_lock=RLock(), price_path=RAW,
            manifest_path="manifest", mlx_wheel_path="wheel")
