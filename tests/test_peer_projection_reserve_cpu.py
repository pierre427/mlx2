"""Projection scratch must consume the same budget as cache and serving lanes."""

from types import SimpleNamespace as NS

import mlx.core as mx
import pytest
from test_int8_prefill import _Model
from test_int8_prefill_q8_inplace import _Model as _Q8Model
from test_int8_prefill_q45_inplace import _Model as _Q45Model

from mlx2.cache_sizing import clamp_host_default_cache_bytes
from mlx2.runtime import int8_prefill as ip
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
from mlx2.serving import ServingEngine


def test_real_handle_reserves_missing_copies_then_only_transient_after_warmup(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    handle = ip.apply(_Model(bits=4), {"enabled": True, "hadamard_rotation": True})
    try:
        missing = handle.resident_estimate_bytes()
        scratch = handle.rotation_build_transient_bytes_max()
        assert missing > 0 and scratch > 0
        assert handle.unfilled_reserve_bytes() == missing + scratch
        # Test actual owner accounting with CPU arrays matching the stored
        # representation; the M5 rotation kernel is not executed here.
        monkeypatch.setattr(handle, "_build", lambda bound: (
            mx.zeros((bound.spec.n, bound.spec.k), dtype=mx.int8),
            mx.ones((bound.spec.n,), dtype=mx.float32),
        ))
        handle.warmup()
        assert handle.weight_bytes() == missing
        assert handle.unfilled_reserve_bytes() == scratch
        handle.evict_weights()
        assert handle.unfilled_reserve_bytes() == missing + scratch
    finally:
        ip.remove(handle)


def test_same_live_projection_reserve_reaches_http_lane_and_apc_budgets():
    pending = [2 << 30]
    engine = ServingEngine.__new__(ServingEngine)
    engine.int8_prefill_handle = NS(unfilled_reserve_bytes=lambda: pending[0])
    engine._hard_reserve_gib = 20.0
    controller = SelfMTPLaneAdmissionController(
        workspace_reserve_gib=engine.projection_workspace_reserve_gib,
    )
    assert engine.hard_reserve_gib == controller.hard_reserve_gib == 22.0
    effective, detail = clamp_host_default_cache_bytes(
        10 << 30, floor_bytes=0, admission_limit_bytes=30 << 30,
        resident_bytes=20 << 30, stream_reserve_bytes=0,
        workspace_reserve_bytes=pending[0], lane_need_bytes=1 << 30, lanes=1,
    )
    assert effective == 7 << 30
    assert detail["workspace_reserve_bytes"] == 2 << 30
    pending[0] = 1 << 30
    assert engine.hard_reserve_gib == controller.hard_reserve_gib == 21.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "2", None])
def test_invalid_live_reserve_fails_closed(value):
    with pytest.raises(ValueError, match="workspace"):
        SelfMTPLaneAdmissionController(workspace_reserve_gib=lambda: value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "2", None])
def test_http_reserve_rejects_invalid_live_provider_after_valid_startup(value):
    pending = [0]
    engine = ServingEngine.__new__(ServingEngine)
    engine.int8_prefill_handle = NS(unfilled_reserve_bytes=lambda: pending[0])
    engine._hard_reserve_gib = 20.0
    controller = SelfMTPLaneAdmissionController(
        workspace_reserve_gib=engine.projection_workspace_reserve_gib,
    )
    assert engine.hard_reserve_gib == controller.hard_reserve_gib == 20.0
    pending[0] = value
    with pytest.raises(ValueError, match="workspace"):
        _ = engine.hard_reserve_gib
    with pytest.raises(ValueError, match="workspace"):
        _ = controller.hard_reserve_gib


def test_present_handle_without_accounting_provider_is_not_treated_as_disabled():
    engine = ServingEngine.__new__(ServingEngine)
    assert engine.projection_workspace_reserve_gib() == 0
    engine.int8_prefill_handle = NS()
    with pytest.raises(ValueError, match="provider"):
        engine.projection_workspace_reserve_gib()


@pytest.mark.parametrize("cache", ["none", "resident", "ttl"])
@pytest.mark.parametrize("family", ["q8", "q45"])
def test_actual_inplace_metadata_accounts_only_unfilled_or_transient_bytes(
    monkeypatch, cache, family,
):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    monkeypatch.setattr(ip.Int8PrefillHandle, "_start_reaper", lambda _: None)
    model = _Q8Model() if family == "q8" else _Q45Model()
    handle = ip.apply(model, {"enabled": True, "scope": "mlp", "cache": cache,
                              f"{family}_inplace": True})
    try:
        assert handle.weight_bytes() == handle.resident_estimate_bytes() == 0
        expected = handle.metadata_copy_bytes()
        assert expected > 0 and handle.unfilled_reserve_bytes() == expected
        if cache == "none":
            # Actual CPU metadata transpose/copy is returned to the caller,
            # never charged as a persistent owner. Its worst-case copy stays
            # reserved before and after each module invocation.
            for bound in handle._bound.values():
                st, bt = handle._q8_meta(bound)
                mx.eval(st, bt)
                assert st.nbytes + bt.nbytes <= expected
                assert handle.metadata_bytes() == 0
                assert handle.unfilled_reserve_bytes() == expected
        else:
            handle.warmup()
            assert handle.metadata_bytes() == expected
            assert handle.unfilled_reserve_bytes() == 0
            handle.evict_weights()
            assert handle.metadata_bytes() == 0
            assert handle.unfilled_reserve_bytes() == expected
    finally:
        ip.remove(handle)


def test_startup_refuses_weight_build_before_allocating_and_removes_install(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    model = _Model(bits=4)
    adapter = NS(model=model)
    handle = ip.apply(model, {"enabled": True, "hadamard_rotation": True})
    monkeypatch.setattr(ip, "apply_for_adapter", lambda *_: handle)
    monkeypatch.setattr(handle, "warmup", lambda: pytest.fail("headroom refusal came too late"))
    with pytest.raises(ip.Int8PrefillError, match="headroom"):
        ip.bind_for_serving(adapter, handle.policy, max_lanes=1,
                            config={}, speculation="off", memory_headroom=lambda: 0)
    assert not any(ip._STATE_ATTR in module.__dict__ for _, module in model.named_modules())
