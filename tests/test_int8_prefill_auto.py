"""``--int8-prefill auto`` and int8 prefill over lane-matmul projections.

CPU only: the two Q8 kernels are replaced by the CPU mirrors of
test_int8_prefill_q8_inplace, and device support is faked where a test needs
the resolver past its device gate.
"""

import json

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import int8_prefill as ip
from mlx2.runtime.lane import installer as lane

from test_int8_prefill_q8_inplace import _Model, fake_device, mirrored  # noqa: F401  (fixtures)


class _Adapter:
    def __init__(self, model, scopes=("mlp", "all")):
        self.model = model
        self._scopes = scopes

    def int8_prefill_supported(self):
        return self._scopes


@pytest.fixture
def device_ok(monkeypatch):
    monkeypatch.setattr(ip, "device_support", lambda: (True, "fake-M5"))
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")


AUTO_POLICY = ip.Int8PrefillPolicy(
    enabled=True, scope="all", q8_inplace=True, act_scale="group64", q8_only=True
)


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


def test_q8_only_requires_q8_inplace_and_binds_revision():
    with pytest.raises(ValueError, match="q8_only requires q8_inplace"):
        ip.Int8PrefillPolicy(enabled=True, q8_only=True)
    with pytest.raises(ValueError):
        ip.Int8PrefillPolicy.from_value({"enabled": True, "q8_inplace": True, "q8_only": 1})
    plain = ip.Int8PrefillPolicy(enabled=True, scope="all", q8_inplace=True)
    assert "q8_only" not in plain.as_dict()
    assert AUTO_POLICY.as_dict()["q8_only"] is True
    assert AUTO_POLICY.revision != plain.revision
    assert ip.Int8PrefillPolicy.from_value(AUTO_POLICY.as_dict()) == AUTO_POLICY


def test_auto_policy_is_group64_q8_only_at_the_widest_declared_scope():
    assert ip.auto_policy(_Adapter(_Model())) == AUTO_POLICY
    mlp = ip.auto_policy(_Adapter(_Model(), scopes=("mlp",)))
    assert mlp.scope == "mlp" and mlp.q8_only and mlp.act_scale == "group64"
    assert ip.auto_policy(_Adapter(_Model(), scopes=())) is None


def test_cli_auto_is_the_server_default_and_takes_no_q8_flag():
    from mlx2.server import build_parser

    assert ip.cli_value("auto") == "auto"
    with pytest.raises(ValueError, match="'auto' always uses group64"):
        ip.cli_value("auto", "per_row")
    parser = build_parser()
    assert parser.parse_args(["--model", "m"]).int8_prefill == "auto"
    assert parser.parse_args(["--model", "m", "--int8-prefill", "off"]).int8_prefill == "off"


# --------------------------------------------------------------------------
# resolve_auto
# --------------------------------------------------------------------------


def _resolve(adapter, **kw):
    kw.setdefault("qualification_mode", False)
    kw.setdefault("qualified_settings", None)
    return ip.resolve_auto(adapter, **kw)


Q8_ONLY = dict(mlp_bits=8, attn_bits=8)
# Q4/Q5 in place, group64, inplace_only: what auto picks for a checkpoint with
# no Q8 projection (Qwen3.8-27B-oQ4e).
AUTO_Q45 = ip.Int8PrefillPolicy(
    enabled=True, scope="all", q45_inplace=True, act_scale="group64", inplace_only=True
)
AUTO_MIXED = ip.Int8PrefillPolicy(
    enabled=True, scope="all", q8_inplace=True, q45_inplace=True,
    act_scale="group64", inplace_only=True,
)


def test_auto_q8_only_policy_keeps_its_qualified_revision():
    # Qwen3.8-27B-MLX-8bit was qualified with this revision on 320f7c61
    # (qualification/runs/qualify-27b-8bit-int8auto-20261009).  Extending auto
    # to Q4/Q5 must not move it, or that record stops matching.
    assert ip.auto_policy(_Adapter(_Model(**Q8_ONLY))) == AUTO_POLICY
    assert AUTO_POLICY.revision == (
        "46f0d38f7a6683afbebcdcc236c49d825846cab70d6dd2e12d39d841ccfb1c4e"
    )


@pytest.mark.parametrize(
    "bits, census, expected",
    [
        (Q8_ONLY, {"q8": 10, "q45": 0, "other_quantized": 0, "linear": 0}, "AUTO_POLICY"),
        (dict(mlp_bits=4, attn_bits=4), {"q8": 0, "q45": 10, "other_quantized": 0, "linear": 0},
         "AUTO_Q45"),
        (dict(mlp_bits=8, attn_bits=4), {"q8": 6, "q45": 4, "other_quantized": 0, "linear": 0},
         "AUTO_MIXED"),
    ],
)
def test_auto_picks_in_place_kernels_from_the_census(device_ok, bits, census, expected):
    policy, receipt = _resolve(_Adapter(_Model(**bits)), qualification_mode=True)
    assert receipt["census"] == census
    assert policy == globals()[expected]
    assert receipt["resolved"] == "on" and receipt["revision"] == policy.revision
    assert policy.act_scale == "group64"


def test_auto_resolves_on_in_qualification_mode(device_ok):
    policy, receipt = _resolve(_Adapter(_Model(**Q8_ONLY)), qualification_mode=True)
    assert policy == AUTO_POLICY
    assert receipt["resolved"] == "on" and receipt["evidence"] == "qualification mode"
    assert receipt["revision"] == AUTO_POLICY.revision
    assert receipt["census"] == {"q8": 10, "q45": 0, "other_quantized": 0, "linear": 0}


def test_auto_follows_the_qualification_record(device_ok):
    adapter = _Adapter(_Model(**Q8_ONLY))
    recorded = {"int8_prefill": {**AUTO_POLICY.as_dict(), "revision": AUTO_POLICY.revision}}
    policy, receipt = _resolve(adapter, qualified_settings=recorded)
    assert policy == AUTO_POLICY and receipt["evidence"] == "qualification record"
    policy, receipt = _resolve(adapter, qualified_settings={"mtp": False})
    assert policy is None and "does not include int8 prefill" in receipt["reason"]
    stale = {"int8_prefill": {"enabled": True, "revision": "0" * 64}}
    policy, receipt = _resolve(adapter, qualified_settings=stale)
    assert policy is None and "different int8 prefill revision" in receipt["reason"]


def test_auto_is_off_without_evidence(device_ok):
    policy, receipt = _resolve(_Adapter(_Model()))
    assert policy is None and receipt["resolved"] == "off"
    assert receipt["reason"].startswith("no qualification evidence")


@pytest.mark.parametrize(
    "adapter, kwargs, reason",
    [
        (_Adapter(_Model(), scopes=()), {}, "does not declare"),
        (_Adapter(_Model()), {"conflicts": ["sp_qmm"]}, "conflicts with sp_qmm"),
        (_Adapter(_Model(mlp_bits=6, attn_bits=6)), {}, "no 8-bit or 4/5-bit gs64"),
    ],
)
def test_auto_declines_with_a_reason(device_ok, adapter, kwargs, reason):
    policy, receipt = _resolve(adapter, qualification_mode=True, **kwargs)
    assert policy is None and reason in receipt["reason"]


def test_auto_declines_off_an_m5_device():
    # The suite runs on the CPU device: the real device gate refuses.
    policy, receipt = _resolve(_Adapter(_Model()), qualification_mode=True)
    assert policy is None and receipt["reason"].startswith("device:")


def test_shared_projection_is_counted_wrapped_and_restored_once(device_ok):
    model = _Model(layers=1)
    shared = model.layers[0].mlp.gate_proj
    model.layers[0].mlp.up_proj = shared
    census = ip.checkpoint_census(_Adapter(model), "mlp")
    assert census["q8"] == 2
    for _ in range(2):
        handle = ip.apply(model, {**AUTO_POLICY.as_dict(), "scope": "mlp"})
        try:
            assert handle.q8_module_count() == 2
            assert type(shared).__mro__[1] is nn.QuantizedLinear
            assert model.layers[0].mlp.up_proj is shared
        finally:
            ip.remove(handle)
        assert type(shared) is nn.QuantizedLinear


def test_shared_projection_cannot_cross_selected_scope(device_ok):
    model = _Model(layers=1)
    shared = model.layers[0].mlp.gate_proj
    # An in-place wrapper on gate_proj would also change this excluded head.
    model.lm_head = shared
    with pytest.raises(ip.Int8PrefillError, match="alias.*scope"):
        ip.apply(model, {**AUTO_POLICY.as_dict(), "scope": "mlp"})
    assert type(shared) is nn.QuantizedLinear
    assert not any(hasattr(type(module), "_mlx2_int8_prefill_base")
                   for _, module in model.named_modules())
    policy, receipt = _resolve(_Adapter(model, scopes=("mlp",)), qualification_mode=True)
    assert policy is None and "outside the selected scope" in receipt["reason"]


# --------------------------------------------------------------------------
# q8_only installs
# --------------------------------------------------------------------------


def test_q8_only_leaves_non_q8_projections_on_stock(mirrored):  # noqa: F811
    model = _Model(mlp_bits=8, attn_bits=4)
    handle = ip.apply(model, AUTO_POLICY)
    try:
        assert handle.q8_module_count() == 6
        assert len(handle.modules) == 6
        for layer in model.layers:
            assert type(layer.self_attn.q_proj) is nn.QuantizedLinear
            assert hasattr(type(layer.mlp.gate_proj), "_mlx2_int8_prefill_base")
        assert handle.status()["skipped"] == 4
        assert all(
            reason.startswith("q8_only:")
            for path, reason in handle._skipped.items()
            if "self_attn" in path
        )
        assert handle.weight_copy_bytes() == 0
    finally:
        ip.remove(handle)


# --------------------------------------------------------------------------
# composition with lane matmul
# --------------------------------------------------------------------------


def _lane_swap(module):
    """What ``lane.install`` does to a projection, minus the device prepare."""
    object.__setattr__(module, "_lane_prepared", None)
    module.__class__ = lane._SWAP[type(module)]
    object.__setattr__(module, "_lane_min_rows", 1)
    object.__setattr__(module, "_lane_max_rows", 128)
    object.__setattr__(module, "_lane_chunk_above_max", True)


def test_int8_wraps_lane_projections_and_routes_by_rows(mirrored):  # noqa: F811
    model = _Model(mlp_bits=8, attn_bits=8, layers=1)
    for _name, module in list(model.named_modules()):
        if type(module) is nn.QuantizedLinear:
            _lane_swap(module)
    gate = model.layers[0].mlp.gate_proj
    assert type(gate) is lane.LaneQuantizedLinear
    spec, reason = ip.module_spec("layers.0.mlp.gate_proj", gate)
    assert spec is not None and spec.kind == "quantized", reason
    handle = ip.apply(model, AUTO_POLICY)
    try:
        assert handle.q8_module_count() == 5  # q, o, gate, up, down
        assert type(gate).__mro__[1] is lane.LaneQuantizedLinear
        assert lane.installed(gate) is False  # nothing prepared on the CPU swap
        object.__setattr__(gate, "_lane_prepared", object())
        assert lane.installed(gate) is True  # seen through the int8 wrapper
        object.__setattr__(gate, "_lane_prepared", None)
        before_lane = lane.STATS["stock_disabled"]
        before_gemm = mirrored["gemm"]
        small = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
        mx.eval(gate(small))
        assert lane.STATS["stock_disabled"] == before_lane + 1  # lane's call ran
        assert mirrored["gemm"] == before_gemm
        big = mx.random.normal((1, 512, 256)).astype(mx.bfloat16)
        mx.eval(gate(big))
        assert mirrored["gemm"] == before_gemm + 1  # int8 took the prefill rows
        assert lane.STATS["stock_disabled"] == before_lane + 1
        with pytest.raises(ValueError, match="remove int8 prefill first"):
            lane.uninstall(model)
        with pytest.raises(ValueError, match="remove int8 prefill first"):
            lane.install(model)
    finally:
        ip.remove(handle)
    assert type(gate) is lane.LaneQuantizedLinear  # int8 restores the lane class
    assert lane.uninstall(model) == 5
    assert type(gate) is nn.QuantizedLinear


# --------------------------------------------------------------------------
# serving engine
# --------------------------------------------------------------------------


def _q8_serving_model():
    from test_int8_prefill import _serving_model

    model = _serving_model(hidden=256, inter=512)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0,
    )
    mx.eval(model.parameters())
    return model


def _engine(model, **kwargs):
    from test_approximate_kv_serving import make_adapter

    from mlx2.serving import ServingEngine

    adapter = make_adapter(model, operations=None)
    adapter.int8_prefill_supported = lambda self: ("mlp", "all")
    return ServingEngine("tiny", adapter_factory=adapter, mtp=False, **kwargs)


@pytest.fixture
def serving_host(monkeypatch):
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def test_engine_auto_binds_in_qualification_mode(serving_host, device_ok):
    engine = _engine(_q8_serving_model(), qualification_mode=True, int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        recorded = status["settings"]["int8_prefill"]
        assert recorded["revision"] == AUTO_POLICY.revision
        assert recorded["q8_only"] is True and recorded["q8_modules"] > 0
        auto = status["int8_prefill"]["auto"]
        assert auto["resolved"] == "on" and auto["evidence"] == "qualification mode"
    finally:
        engine.close()


def test_engine_auto_without_evidence_serves_stock_and_records_nothing(
    serving_host, device_ok
):
    engine = _engine(_q8_serving_model(), int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert "int8_prefill" not in status["settings"]
        assert status["int8_prefill"]["enabled"] is False
        auto = status["int8_prefill"]["auto"]
        assert auto["resolved"] == "off"
        assert auto["reason"].startswith("no qualification evidence")
    finally:
        engine.close()


def test_engine_auto_declines_a_bind_refusal(serving_host, device_ok, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ip.Int8PrefillError("decode block reaches the row threshold")

    monkeypatch.setattr(ip, "validate_decode_row_bound", refuse)
    engine = _engine(_q8_serving_model(), qualification_mode=True, int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert "int8_prefill" not in status["settings"]
        auto = status["int8_prefill"]["auto"]
        assert auto["resolved"] == "off" and "bind refused" in auto["reason"]
        assert "evidence" not in auto
    finally:
        engine.close()


def test_engine_explicit_mode_still_refuses_a_bind_failure(serving_host, device_ok, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ip.Int8PrefillError("decode block reaches the row threshold")

    monkeypatch.setattr(ip, "validate_decode_row_bound", refuse)
    engine = _engine(
        _q8_serving_model(), qualification_mode=True, int8_prefill=AUTO_POLICY.as_dict()
    )
    try:
        engine.thread.join(timeout=60)
        assert not engine.thread.is_alive()
        assert engine.error and "row threshold" in engine.error
    finally:
        engine.close()


_BIND_FAULT = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


def test_engine_auto_declines_a_bind_time_device_fault(serving_host, monkeypatch):
    """The probe re-raises a device fault instead of caching it; ``auto``
    then serves stock prefill and names the fault, as it does for a refusal."""
    monkeypatch.setattr(ip, "device_support", lambda: (True, "fake-M5"))

    def fault():
        raise RuntimeError(_BIND_FAULT)

    monkeypatch.setattr(ip, "require_supported_device", fault)
    engine = _engine(_q8_serving_model(), qualification_mode=True, int8_prefill="auto")
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert "int8_prefill" not in status["settings"]
        auto = status["int8_prefill"]["auto"]
        assert auto["resolved"] == "off"
        assert "device fault (out_of_memory)" in auto["reason"]
        assert "Insufficient Memory" in auto["reason"]
        assert "evidence" not in auto
    finally:
        engine.close()


def test_engine_explicit_mode_fails_on_a_bind_time_device_fault(serving_host, monkeypatch):
    monkeypatch.setattr(ip, "device_support", lambda: (True, "fake-M5"))

    def fault():
        raise RuntimeError(_BIND_FAULT)

    monkeypatch.setattr(ip, "require_supported_device", fault)
    engine = _engine(
        _q8_serving_model(), qualification_mode=True, int8_prefill=AUTO_POLICY.as_dict()
    )
    try:
        engine.thread.join(timeout=60)
        assert not engine.thread.is_alive()
        assert engine.error and "Insufficient Memory" in engine.error
    finally:
        engine.close()


def test_engine_auto_reads_the_qualification_record(serving_host, device_ok, tmp_path):
    record = tmp_path / "q.json"
    record.write_text(json.dumps({"settings": {"mtp": False}}))
    engine = _engine(_q8_serving_model(), int8_prefill="auto", qualification=str(record))
    try:
        engine.thread.join(timeout=60)
        assert not engine.thread.is_alive() and engine.error
        auto = engine.int8_prefill_auto
        # The record has no int8 prefill: auto stays off (the record itself is
        # not a valid qualification, so the route then fails to load).
        assert auto["resolved"] == "off"
        assert "does not include int8 prefill" in auto["reason"]
    finally:
        engine.close()
