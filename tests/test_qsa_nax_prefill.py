"""NAX block-sparse QSA prefill: admission, policy knobs, diagnostics, and the
qualifier requirement (qsa-nax-prefill 2026-10-02).

Before this change the kernel's admission counters were recorded but never
surfaced, the auto crossover (16384) and single-lane rule were fixed module
constants, and no qualification required the kernel to run.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.qualification import required_feature_checks, selected_not_observed_features

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _sel(*, batch=1, length=512, width=16384, kind="explicit"):
    return SimpleNamespace(kind=kind, batch=batch, length=length, physical_width=width)


def _decide(selection, **kw):
    from mlx2.runtime.models import qwen4_exp as QE

    kw.setdefault("training", False)
    kw.setdefault("layout_ok", True)
    kw.setdefault("device_supported", True)
    kw.setdefault("kernel_available", True)
    return QE.decide_qsa_nax_admission(selection, **kw)


# --- admission matrix -------------------------------------------------------

@pytest.mark.parametrize(
    "selection, kwargs, reason",
    [
        (_sel(), {}, "engaged_auto"),
        (_sel(width=16383), {}, "auto_context_below_crossover"),
        (_sel(length=63), {}, "query_below_min"),
        (_sel(batch=2), {}, "auto_batch_gt_one"),
        (_sel(kind="implicit_all"), {}, "selection_implicit_all"),
        (_sel(), {"cache_layout_ok": False}, "unsupported_cache_layout"),
        (_sel(), {"training": True}, "training"),
        (_sel(), {"layout_ok": False}, "unsupported_layout"),
        (_sel(), {"device_supported": False}, "unsupported_device"),
        (_sel(), {"kernel_available": False}, "kernel_unavailable"),
        # Policy crossover.
        (_sel(width=8192), {"min_physical_kv": 8192}, "engaged_auto"),
        (_sel(width=8191), {"min_physical_kv": 8192}, "auto_context_below_crossover"),
        (_sel(width=16384), {"min_physical_kv": 32768}, "auto_context_below_crossover"),
        # Batched admission: ragged lanes share the crossover on the padded width.
        (_sel(batch=3), {"batched": True}, "engaged_auto_batched"),
        (_sel(batch=3, width=4096), {"batched": True}, "auto_context_below_crossover"),
        (_sel(batch=1), {"batched": True}, "engaged_auto"),
        (_sel(batch=2), {"batched": False}, "auto_batch_gt_one"),
        # The slice-invariant lane never runs the kernel: a decline, not an engagement.
        (_sel(), {"invariant_lane": True}, "invariant_prefill"),
        (_sel(width=100), {"invariant_lane": True}, "auto_context_below_crossover"),
    ],
)
def test_admission_matrix(selection, kwargs, reason):
    decision = _decide(selection, **kwargs)
    assert decision.reason == reason
    assert decision.engage is reason.startswith("engaged_")


def test_module_selections_drive_the_auto_defaults(monkeypatch):
    from mlx2.runtime.models import qwen4_exp as QE

    monkeypatch.setattr(QE, "_QSA_NAX_AUTO_BATCHED", True)
    monkeypatch.setattr(QE, "_QSA_NAX_AUTO_MIN_PHYSICAL_KV", 4096)
    assert _decide(_sel(batch=4, width=4096)).reason == "engaged_auto_batched"
    assert _decide(_sel(batch=4, width=4095)).reason == "auto_context_below_crossover"


def test_explicit_on_ignores_crossover_and_batch(monkeypatch):
    from mlx2.runtime.models import qwen4_exp as QE

    monkeypatch.setattr(QE, "_QSA_NAX_KERNEL", True)
    assert _decide(_sel(batch=4, width=128), batched=False).reason == "engaged_on"
    monkeypatch.setattr(QE, "_QSA_NAX_KERNEL", False)
    assert _decide(_sel()).reason == "explicit_off"


# --- status and diagnostics -------------------------------------------------

def test_status_counts_engagements_and_reasons(monkeypatch):
    from mlx2.runtime.models import qwen4_exp as QE

    monkeypatch.setattr(QE, "_QSA_NAX_DEVICE_SUPPORTED", False)
    monkeypatch.setattr(QE, "_QSA_NAX_DEVICE_NAME", "Apple M3 Max")
    QE.qsa_nax_status(reset=True)
    try:
        for selection, kw in ((_sel(), {}), (_sel(batch=2, width=20000), {"batched": True}),
                              (_sel(width=2048), {})):
            QE._record_qsa_nax_admission(selection, _decide(selection, **kw))
        status = QE.qsa_nax_status()
        assert status["attempts"] == 3
        assert status["engagements"] == 2
        assert status["fallbacks"] == 1
        assert status["counts"] == {
            "engaged_auto": 1, "engaged_auto_batched": 1, "auto_context_below_crossover": 1,
        }
        assert status["last_receipt"] == {
            "engaged": False, "reason": "auto_context_below_crossover",
            "batch": 1, "query_width": 512, "physical_kv": 2048,
        }
        assert status["mode"] == "auto"
        assert status["min_query_width"] == 64
        assert status["min_physical_kv"] == QE._QSA_NAX_AUTO_MIN_PHYSICAL_KV
        assert status["batched"] is False
        assert status["nax_host"] is False
        assert status["device_name"] == "Apple M3 Max"
        assert "kernel_available" in status
        QE.qsa_nax_status(reset=True)
        assert QE.qsa_nax_status()["attempts"] == 0
        assert QE.qsa_nax_status()["last_receipt"] is None
    finally:
        QE.qsa_nax_status(reset=True)


def test_flash_next_diagnostics_expose_qsa_nax_prefill():
    import mlx.nn as nn

    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.runtime.models import qwen4_exp as QE

    adapter = object.__new__(FlashNextAdapter)
    adapter.model = nn.Module()
    adapter._tables = []
    adapter._diagnostic_modules = ()
    adapter.policy = FlashNextPolicy()
    QE.qsa_nax_status(reset=True)
    try:
        QE._record_qsa_nax_admission(_sel(), _decide(_sel()))
        status = adapter.diagnostics()["qsa_nax_prefill"]
        assert status["engagements"] == 1
        assert status["counts"] == {"engaged_auto": 1}
        assert {"nax_host", "device_name", "min_physical_kv", "batched"} <= set(status)
    finally:
        QE.qsa_nax_status(reset=True)


# --- policy knobs -----------------------------------------------------------

def test_policy_knobs_default_and_receipt_neutral():
    # The crossover default is 8192 since 2026-10-02 (it was 16384, the
    # module default); the policy pins it because it differs from the module.
    default = FlashNextPolicy()
    assert default.qsa_nax_min_physical_kv == 8192
    assert default.qsa_nax_batched is False
    assert "qsa_nax_min_physical_kv" not in default.as_dict()
    assert "qsa_nax_batched" not in default.as_dict()
    env = default.environment()
    assert env["MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV"] == "8192"
    assert "MLX_QWEN4_QSA_NAX_BATCHED" not in env
    assert FlashNextPolicy.from_mapping(default.as_dict()) == default
    # An explicit default reads back as the default receipt.
    assert FlashNextPolicy(qsa_nax_min_physical_kv=8192).as_dict() == default.as_dict()
    # The old value is an explicit choice, and leaves the module default.
    old = FlashNextPolicy(qsa_nax_min_physical_kv=16384)
    assert old.as_dict()["qsa_nax_min_physical_kv"] == 16384
    assert "MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV" not in old.environment()


def test_policy_knobs_round_trip_and_reach_the_environment():
    chosen = FlashNextPolicy.from_mapping(
        {"qsa_nax_min_physical_kv": 12288, "qsa_nax_batched": True}
    )
    receipt = chosen.as_dict()
    assert receipt["qsa_nax_min_physical_kv"] == 12288
    assert receipt["qsa_nax_batched"] is True
    assert FlashNextPolicy.from_mapping(receipt) == chosen
    env = chosen.environment()
    assert env["MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV"] == "12288"
    assert env["MLX_QWEN4_QSA_NAX_BATCHED"] == "1"
    # Every variable the policy sets is one the import guard tracks.
    from mlx2.runtime.models.import_env import PREFIXES

    for name in ("MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV", "MLX_QWEN4_QSA_NAX_BATCHED"):
        assert name.startswith(PREFIXES)


@pytest.mark.parametrize(
    "bad",
    [{"qsa_nax_min_physical_kv": -1}, {"qsa_nax_min_physical_kv": "16384"},
     {"qsa_nax_min_physical_kv": True}, {"qsa_nax_batched": 1},
     {"qsa_nax_batched": "yes"}],
)
def test_policy_knobs_validate(bad):
    # The field exists (an unknown field would raise for another reason).
    good = {"qsa_nax_min_physical_kv": 0, "qsa_nax_batched": True}
    (name,) = bad
    FlashNextPolicy.from_mapping({name: good[name]})
    with pytest.raises(ValueError, match=name):
        FlashNextPolicy.from_mapping(bad)


def test_non_default_knobs_enter_the_apc_execution_identity():
    from mlx2.runtime.apc_numerics import execution_numerics_identity

    assert execution_numerics_identity({}) is None
    assert execution_numerics_identity(
        {"MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV": "16384", "MLX_QWEN4_QSA_NAX_BATCHED": "0"}
    ) is None
    low = execution_numerics_identity({"MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV": "8192"})
    high = execution_numerics_identity({"MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV": "32768"})
    assert low and high and low != high
    assert execution_numerics_identity({"MLX_QWEN4_QSA_NAX_BATCHED": "1"})
    env = FlashNextPolicy(qsa_nax_min_physical_kv=8192, qsa_nax_batched=True).environment()
    assert execution_numerics_identity(env) is not None


# --- qualifier requirement --------------------------------------------------

GEOMETRY = {"budget": 2048, "compress_ratio": 4, "head_dim": 128, "n_heads": 4}


def _settings(policy=None, **overrides):
    policy = policy or FlashNextPolicy()
    settings = {"environment": policy.environment(), "max_context": 262144,
                "max_lanes": 4, "mtp": False, "speculation": "ordinary",
                "prefill_step": 8192, "qsa_indexer": dict(GEOMETRY)}
    settings.update(overrides)
    return settings


def test_flash_next_route_requires_the_nax_prefill_kernel():
    assert "feature_qsa_nax_prefill" in required_feature_checks(_settings())
    # Probe floor exactly crossover + one 64-row slice.
    # The default crossover is 8192 (2026-10-02).
    edge = 8192 + 64 + 256
    assert "feature_qsa_nax_prefill" in required_feature_checks(_settings(max_context=edge))
    short = _settings(max_context=edge - 1)
    assert "feature_qsa_nax_prefill" not in required_feature_checks(short)
    reason = selected_not_observed_features(short)["feature_qsa_nax_prefill"]
    assert "8192" in reason and "64-row" in reason


def test_requirement_follows_the_crossover_in_force():
    lowered = FlashNextPolicy(qsa_nax_min_physical_kv=4096)
    assert "feature_qsa_nax_prefill" in required_feature_checks(
        _settings(lowered, max_context=8192))
    assert "feature_qsa_nax_prefill" not in required_feature_checks(
        _settings(max_context=8192))
    raised = FlashNextPolicy(qsa_nax_min_physical_kv=65536)
    reason = selected_not_observed_features(
        _settings(raised, max_context=32768))["feature_qsa_nax_prefill"]
    assert "65536" in reason


def test_explicit_on_and_global_shortcircuit_move_the_floor():
    env = {**FlashNextPolicy().environment(), "MLX_QWEN4_QSA_NAX_KERNEL": "1"}
    assert "feature_qsa_nax_prefill" in required_feature_checks(
        _settings(environment=env, max_context=4096))
    # Under the global dense short-circuit the indexer is implicit up to its
    # top-k (512 blocks = 2048 tokens): explicit from 2052.
    sc = {**env, "MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT": "1"}
    assert "feature_qsa_nax_prefill" in required_feature_checks(
        _settings(environment=sc, max_context=2052 + 64 + 256))
    assert "feature_qsa_nax_prefill" not in required_feature_checks(
        _settings(environment=sc, max_context=2052 + 63 + 256))


def test_routes_that_cannot_engage_say_why():
    from mlx2.runtime.prefill_plan import execution_identity

    invariant = _settings(prefill_execution=execution_identity(
        invariant={"schema": "invariant-prefill-v1", "law": {"x": 1}}))
    assert "invariant_prefill" in selected_not_observed_features(
        invariant)["feature_qsa_nax_prefill"]
    approx = _settings(approximate_kv={"enabled": True})
    assert "quantized" in selected_not_observed_features(approx)["feature_qsa_nax_prefill"]
    tiny = _settings(prefill_step=32)
    assert "prefill_step 32" in selected_not_observed_features(
        tiny)["feature_qsa_nax_prefill"]
    for settings in (invariant, approx, tiny):
        assert "feature_qsa_nax_prefill" not in required_feature_checks(settings)


def test_unselected_routes_neither_require_nor_label_it():
    off = {**FlashNextPolicy().environment(), "MLX_QWEN4_QSA_NAX_KERNEL": "off"}
    for settings in (_settings(environment=off),
                     {"environment": {}, "mtp": False, "max_lanes": 4, "max_context": 262144}):
        assert "feature_qsa_nax_prefill" not in required_feature_checks(settings)
        assert "feature_qsa_nax_prefill" not in selected_not_observed_features(settings)


@pytest.fixture(scope="module")
def qualify():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("qualify_serving")
    finally:
        sys.path.remove(str(SCRIPTS))


def _nax(engagements=0, *, nax_host=True, device="Apple M5 Max", counts=None):
    return {"qsa_nax_prefill": {
        "mode": "auto", "nax_host": nax_host, "device_name": device,
        "engagements": engagements, "attempts": engagements, "counts": counts or {},
    }}


def test_observation_is_an_engagement_delta(qualify):
    from mlx2.qualification import qsa_nax_prefill_engagement

    observed = qualify.feature_observations({"execution": _nax(40)},
                                            initial={"execution": _nax(8)})
    assert observed["qsa_nax_prefill"] == 32
    assert qsa_nax_prefill_engagement(_nax(8), _nax(8)) == 0
    assert qsa_nax_prefill_engagement({}, {}) == 0
    assert qualify.unobservable_features({"qsa_nax_prefill"}) == []


def test_host_gated_not_observed():
    from mlx2.qualification import (
        HOST_GATED_FEATURES,
        host_gate_record,
        host_gated_not_observed,
    )

    assert "qsa_nax_prefill" in HOST_GATED_FEATURES
    m3 = _nax(nax_host=False, device="Apple M3 Max", counts={"unsupported_device": 9})
    assert "not an M5" in host_gated_not_observed(m3, m3)["qsa_nax_prefill"]
    refused = host_gated_not_observed(
        _nax(counts={"kernel_unavailable": 4}), _nax())
    assert "kernel_unavailable" in refused["qsa_nax_prefill"]
    # Engaged, or idle on an admitting host: no waiver (a failed check stays failed).
    assert "qsa_nax_prefill" not in host_gated_not_observed(_nax(3), _nax())
    assert "qsa_nax_prefill" not in host_gated_not_observed(_nax(), _nax())
    assert host_gate_record("qsa_nax_prefill", m3) == {
        "nax_host": False, "device_name": "Apple M3 Max"}


def test_loader_binds_the_waiver_to_the_serving_host(tmp_path, monkeypatch):
    import json

    import mlx2.qualification as Q
    from mlx2.adapters.qwen import QWEN4_FLASH_NEXT

    settings = _settings(max_context=32768, environment={})
    assert "feature_qsa_nax_prefill" in Q.required_feature_checks(settings)
    entry = {"status": "selected, not observed", "host_gated": True,
             "reason": "host is not an M5 (NAX) Metal device",
             "host_gate": {"nax_host": False, "device_name": "Apple M3 Max"}}
    others = {c: {"passed": True} for c in Q.required_feature_checks(settings)
              if c != "feature_qsa_nax_prefill"}
    record = {
        "passed": True, "runtime": {"source": "abc"}, "artifact": "weights",
        "settings": settings,
        "qualification_harness": Q.APPROVED_QUALIFICATION_HARNESS,
        "checks": {**{c: {"passed": True} for c in Q.REQUIRED_CHECKS | {"structured_output"}},
                   **others},
        "selected_not_observed": {"feature_qsa_nax_prefill": entry},
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))

    def load():
        return Q.load_qualified_route(path, runtime=record["runtime"], artifact="weights",
                                      settings=settings, descriptor=QWEN4_FLASH_NEXT,
                                      name="nax")

    monkeypatch.setitem(Q.HOST_GATE_PROBES, "qsa_nax_prefill",
                        lambda: {"nax_host": False, "device_name": "Apple M3 Max"})
    assert load().profile
    monkeypatch.setitem(Q.HOST_GATE_PROBES, "qsa_nax_prefill",
                        lambda: {"nax_host": True, "device_name": "Apple M5 Max"})
    with pytest.raises(ValueError, match="qsa_nax_prefill"):
        load()
    record.pop("selected_not_observed")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="qsa_nax_prefill"):
        load()


def test_host_gate_probe_shape():
    from mlx2.qualification import HOST_GATE_PROBES

    gate = HOST_GATE_PROBES["qsa_nax_prefill"]()
    assert set(gate) == {"nax_host", "device_name"}
    assert isinstance(gate["nax_host"], bool) and isinstance(gate["device_name"], str)
