"""Bit-exact (batch-invariant) verify mode: policy, fail-closed receipts, wiring.

The kernels live in the mlx fork (``mx.metal.set_qmv_bitexact``) and are
GPU-qualified by ``python/tests/test_quantized.py::test_qmv_bitexact_across_m``
and ``scripts/gpu_check_verify_bitexact.py``.  These CPU tests pin the serving
contract: default off, fail closed without the capability, receipts claim
``verify_bitexact: true`` only with a live mode and an advanced route counter.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime import verify_bitexact as vb

ROOT = Path(__file__).resolve().parents[1]


class FakeMetal:
    """Host-side stand-in for the fork's mx.metal bit-exact API."""

    def __init__(self, *, initial=False, max_m=64, auto_dispatch=True, refuse=False):
        self.mode = initial
        self.max_m = max_m
        self.count = 0
        self.auto_dispatch = auto_dispatch
        self.refuse = refuse
        self.set_calls = []

    def is_available(self):
        return True

    def set_qmv_bitexact(self, enabled):
        self.set_calls.append(bool(enabled))
        previous = self.mode
        if not self.refuse:
            self.mode = bool(enabled)
        return previous

    def qmv_bitexact(self):
        return self.mode

    def qmv_bitexact_max_m(self):
        return self.max_m

    def qmv_bitexact_dispatches(self):
        # With auto_dispatch, every read while the mode is on observes one more
        # routed dispatch (the stand-in for matmuls encoded between reads).
        if self.auto_dispatch and self.mode:
            self.count += 1
        return self.count


class FakeMx:
    def __init__(self, metal):
        self.metal = metal


class BareMetal:
    def is_available(self):
        return True


def test_policy_parsing_default_off():
    assert not vb.VerifyBitexactPolicy.from_value(None).enabled
    assert not vb.VerifyBitexactPolicy.from_value(False).enabled
    assert not vb.VerifyBitexactPolicy.from_value("off").enabled
    assert vb.VerifyBitexactPolicy.from_value(True).enabled
    assert vb.VerifyBitexactPolicy.from_value("on").enabled
    assert vb.VerifyBitexactPolicy.from_value({"enabled": True}).enabled
    with pytest.raises(ValueError):
        vb.VerifyBitexactPolicy.from_value("sometimes")


def test_capability_probe():
    missing = vb.capability(FakeMx(BareMetal()))
    assert missing["available"] is False
    assert missing["reason"] == "mlx_without_bitexact_qmv"
    assert set(missing["missing_api"]) == set(vb.REQUIRED_API)
    ok = vb.capability(FakeMx(FakeMetal(max_m=32)))
    assert ok == {"available": True, "reason": None, "missing_api": [], "max_m": 32}
    # The real installed mlx never makes the probe raise.
    real = vb.capability()
    assert isinstance(real["available"], bool)


def test_bind_off_is_a_no_op():
    metal = FakeMetal()
    handle, settings = vb.bind_for_serving(
        vb.VerifyBitexactPolicy(False), mx_module=FakeMx(metal)
    )
    assert handle is None and settings is None
    assert metal.set_calls == []


def test_bind_fails_closed_without_capability():
    with pytest.raises(vb.VerifyBitexactUnavailable, match="bit-exact qmv mode"):
        vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(BareMetal()))


def test_bind_fails_closed_when_mlx_refuses_the_mode():
    metal = FakeMetal(refuse=True)
    with pytest.raises(vb.VerifyBitexactUnavailable, match="refused"):
        vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))


def test_bind_activates_and_remove_restores_previous():
    metal = FakeMetal(initial=False, max_m=48)
    handle, settings = vb.bind_for_serving(
        vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal)
    )
    assert settings == {
        "enabled": True,
        "schema": vb.SCHEMA,
        "scope": vb.SCOPE,
        "max_m": 48,
    }
    assert metal.mode is True and handle.active
    handle.remove()
    assert metal.mode is False and not handle.active


def test_receipt_true_only_when_mechanism_ran():
    """Mechanism assertion: the claim requires an advanced route counter."""
    metal = FakeMetal()
    handle, _ = vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))
    start = handle.begin_request(explicit=True)
    receipt = handle.request_receipt(start)
    assert receipt["verify_bitexact"] is True
    assert receipt["reason"] is None
    assert receipt["requested"] is True
    assert receipt["dispatches_during_request"] > 0
    assert receipt["scope"] == "quantized_matmul"
    assert "joined_slab_sdpa" in receipt["residual_width_dependence"]
    assert handle.counts["receipts_true"] == 1


def test_receipt_fails_closed_without_dispatches():
    """Regression: a live mode with no routed matmul must not claim bit-exact."""
    metal = FakeMetal(auto_dispatch=False)
    handle, _ = vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))
    receipt = handle.request_receipt(handle.begin_request(explicit=False))
    assert receipt["verify_bitexact"] is False
    assert receipt["reason"] == "no_bitexact_dispatch_observed"
    assert handle.counts["receipts_false"] == 1


def test_receipt_fails_closed_when_mode_changes_mid_request():
    metal = FakeMetal()
    handle, _ = vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))
    start = handle.begin_request(explicit=False)
    handle.remove()
    assert handle.request_receipt(start)["reason"] == "mode_not_active"
    handle.activate()
    assert handle.request_receipt(start)["reason"] == "mode_changed_during_request"
    # Someone else switched mlx's global mode off behind the handle's back.
    start = handle.begin_request(explicit=False)
    metal.mode = False
    assert handle.request_receipt(start)["reason"] == "mlx_mode_off"
    assert handle.request_receipt(None)["reason"] == "no_request_snapshot"


def test_request_check_fails_closed():
    vb.check_request({"verify_bitexact": False}, None)
    vb.check_request({}, None)
    with pytest.raises(vb.VerifyBitexactUnavailable, match="--verify-bitexact"):
        vb.check_request({"verify_bitexact": True}, None)
    metal = FakeMetal()
    handle, _ = vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))
    vb.check_request({"verify_bitexact": True}, handle)
    handle.remove()
    with pytest.raises(vb.VerifyBitexactUnavailable):
        vb.check_request({"verify_bitexact": True}, handle)


def test_receipt_fields_default_off():
    from mlx2.serving import verify_bitexact_receipt_fields

    assert verify_bitexact_receipt_fields(None, None) == {"verify_bitexact": False}
    metal = FakeMetal()
    handle, _ = vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))
    fields = verify_bitexact_receipt_fields(handle, handle.begin_request(explicit=False))
    assert fields["verify_bitexact"] is True
    assert fields["verify_bitexact_detail"]["schema"] == vb.SCHEMA


def test_validate_request_field():
    from mlx2.server import validate_request

    body = {"messages": [{"role": "user", "content": "hi"}], "verify_bitexact": True}
    assert validate_request(body)["verify_bitexact"] is True
    with pytest.raises(ValueError, match="verify_bitexact must be boolean"):
        validate_request({**body, "verify_bitexact": "yes"})


def test_server_flag_maps_to_engine_kwargs():
    from mlx2.server import build_parser, serving_engine_kwargs

    parser = build_parser()
    base = ["--model", "m"]
    args = parser.parse_args(base)
    assert args.verify_bitexact is False
    kwargs = serving_engine_kwargs(
        args, None, native_mtp=True, approximate_kv=None, max_request_bytes=1 << 20
    )
    assert kwargs["verify_bitexact"] is False
    args = parser.parse_args(base + ["--verify-bitexact"])
    kwargs = serving_engine_kwargs(
        args, None, native_mtp=True, approximate_kv=None, max_request_bytes=1 << 20
    )
    assert kwargs["verify_bitexact"] is True


def test_qualification_requires_observed_route():
    from mlx2.qualification import required_feature_checks

    assert "feature_verify_bitexact" not in required_feature_checks({})
    assert "feature_verify_bitexact" in required_feature_checks(
        {"verify_bitexact": {"enabled": True}}
    )
    spec = importlib.util.spec_from_file_location(
        "qualify_serving_rm09", ROOT / "scripts" / "qualify_serving.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    observed = module.feature_observations(
        {"verify_bitexact": {"active": True, "dispatches": 17}}
    )
    assert observed["verify_bitexact"] == 17
    # An inactive mode never counts, however large the process counter.
    assert module.feature_observations(
        {"verify_bitexact": {"active": False, "dispatches": 17}}
    )["verify_bitexact"] == 0
    assert module.feature_observations({})["verify_bitexact"] == 0


def test_prometheus_exports_counters():
    from mlx2 import prometheus

    class Engine:
        pass

    metal = FakeMetal()
    handle, _ = vb.bind_for_serving(vb.VerifyBitexactPolicy(True), mx_module=FakeMx(metal))
    handle.request_receipt(handle.begin_request(explicit=False))
    engine = Engine()
    engine.verify_bitexact_handle = handle
    builder = prometheus.PrometheusBuilder()
    prometheus._add_verify_bitexact(builder, engine)
    text = builder.render()
    assert "mlx2_verify_bitexact_enabled 1" in text
    assert 'mlx2_verify_bitexact_receipts_total{claim="true"} 1' in text
    off = Engine()
    builder = prometheus.PrometheusBuilder()
    prometheus._add_verify_bitexact(builder, off)
    assert "mlx2_verify_bitexact_enabled 0" in builder.render()


# -- engine wiring (tiny CPU model) -------------------------------------------


def _tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        full_attention_interval=4,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=256,
    )
    mx.random.seed(3)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.fixture
def serving_host(monkeypatch):
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def _engine(**kwargs):
    from test_approximate_kv_serving import make_adapter

    from mlx2.serving import ServingEngine

    return ServingEngine(
        "tiny",
        adapter_factory=make_adapter(_tiny_model(), operations=None),
        qualification_mode=True,
        mtp=False,
        **kwargs,
    )


def _run(engine, request):
    job = engine.submit(request)
    while True:
        event = job.events.get(timeout=60)
        if "error" in event:
            raise AssertionError(event)
        if "finish_reason" in event:
            return event["receipt"]


def test_engine_default_off_records_false_and_rejects_requests(serving_host):
    engine = _engine()
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert "verify_bitexact" not in status["settings"]
        assert status["verify_bitexact"]["active"] is False
        receipt = _run(engine, {"tokens": [1, 2, 3, 4], "max_tokens": 2, "temperature": 0})
        assert receipt["verify_bitexact"] is False
        assert "verify_bitexact_detail" not in receipt
        with pytest.raises(ValueError, match="--verify-bitexact"):
            engine.submit(
                {"tokens": [1, 2, 3], "max_tokens": 2, "verify_bitexact": True}
            )
    finally:
        engine.close()


def test_engine_on_binds_settings_and_receipts(serving_host, monkeypatch):
    metal = FakeMetal()
    monkeypatch.setattr(vb, "_metal", lambda mx_module=None: metal)
    engine = _engine(verify_bitexact=True)
    try:
        assert engine.ready.wait(60), engine.error
        status = engine.status()
        assert status["settings"]["verify_bitexact"]["enabled"] is True
        assert "verify-bitexact" in str(status["profile"])
        receipt = _run(
            engine,
            {"tokens": [1, 2, 3, 4], "max_tokens": 2, "temperature": 0, "verify_bitexact": True},
        )
        assert receipt["verify_bitexact"] is True
        assert receipt["verify_bitexact_detail"]["requested"] is True
        assert engine.verify_bitexact_handle.counts["requests_explicit"] == 1
    finally:
        engine.close()
    # Worker shutdown restores mlx's previous global mode.
    assert metal.mode is False


def test_engine_on_without_route_evidence_fails_closed(serving_host, monkeypatch):
    metal = FakeMetal(auto_dispatch=False)
    monkeypatch.setattr(vb, "_metal", lambda mx_module=None: metal)
    engine = _engine(verify_bitexact=True)
    try:
        assert engine.ready.wait(60), engine.error
        receipt = _run(engine, {"tokens": [1, 2, 3, 4], "max_tokens": 2, "temperature": 0})
        assert receipt["verify_bitexact"] is False
        assert receipt["verify_bitexact_detail"]["reason"] == "no_bitexact_dispatch_observed"
    finally:
        engine.close()


def test_engine_on_fails_closed_on_stock_mlx(serving_host, monkeypatch):
    monkeypatch.setattr(vb, "_metal", lambda mx_module=None: BareMetal())
    engine = _engine(verify_bitexact=True)
    try:
        engine.thread.join(60)
        assert not engine.ready.is_set()
        assert "bit-exact qmv mode" in engine.error
    finally:
        engine.close()


# -- GPU-gated scripts ----------------------------------------------------------


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"rm09_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["bench_qmv_bitexact", "gpu_check_verify_bitexact"])
def test_gpu_scripts_refuse_without_gpu_flag_and_dry_run(name, monkeypatch, capsys):
    module = _load_script(name)
    monkeypatch.setattr(sys, "argv", [name, "--out", "/dev/null"])
    with pytest.raises(SystemExit) as refused:
        module.main()
    assert refused.value.code == 2
    assert "--i-own-the-gpu" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [name, "--dry-run"])
    assert module.main() == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert lines
    if name == "gpu_check_verify_bitexact":
        arms = [json.loads(l) for l in lines]
        assert {a["arm"] for a in arms} == {"bitexact", "control"}
        assert all(("--verify-bitexact" in a["cmd"]) == (a["arm"] == "bitexact") for a in arms)
