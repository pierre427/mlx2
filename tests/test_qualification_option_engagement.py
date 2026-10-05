"""Selected options must show engagement (flip lane, 2026-10-02 options sweeps).

The 2026-10-02 sweeps found routes that qualified while a selected mechanism
was either unobservable (Qwen3.6 reports its decode wins under
``decode_wins``, not the Flash-Next ``moe`` shape) or never required at all
(decode-first, invariant prefill, the slice floor, fp32 head logits, fp16
GDN state, MLX's GDN core, the adaptive sorted-MoE pad, the NAX gather on
Qwen3.6).  Each requirement here follows the H1 pattern: a counter delta,
or "selected, not observed" with a reason where the probe cannot reach it.
The Qwen3.6 and 27B cases replay the sweeps' own status snapshots.
"""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.qualification import (
    APPROVED_QUALIFICATION_HARNESS,
    REQUIRED_CHECKS,
    contention_gated_not_observed,
    load_qualified_route,
    required_feature_checks,
    selected_not_observed_features,
    unqualifiable_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
Q36 = ROOT / "qualification/runs/options-sweep-qwen36-20261002/smoke"
B27 = ROOT / "qualification/runs/options-sweep-27b-20261002/smoke"


@pytest.fixture(scope="module")
def qualify():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("qualify_serving")
    finally:
        sys.path.remove(str(SCRIPTS))


def _q36(stage):
    initial = json.loads((Q36 / stage / "status-initial.json").read_text())
    final = json.loads((Q36 / stage / "status-final.json").read_text())
    return initial, final


def _b27(arm):
    """The 27B smoke's final status (the receipt keeps no initial one)."""
    return json.loads((B27 / arm / "qualification.json").read_text())["final_status"]


def _features(names):
    return {"feature_" + name for name in names}


# -- Qwen3.6 decode wins ---------------------------------------------------


def test_qwen36_decode_wins_is_no_longer_an_unqualifiable_candidate():
    assert unqualifiable_candidate({"environment": {"MLX_QWEN36_DECODE_WINS": "1"}}) is None
    # The other Qwen3.6 candidate keeps its gate.
    assert unqualifiable_candidate(
        {"environment": {"MLX_QWEN36_MOE_ROUTED_CANDIDATE": "1"}}
    )


Q36_ORDINARY = {
    "moe_routed_decode", "moe_topk_fold",
    "qwen36_fused_gdn_decode", "qwen36_fused_gdn_batch_decode",
}
Q36_MTP = Q36_ORDINARY | {"qwen36_fused_gdn_verify", "qwen36_fused_gdn_batch_verify"}


@pytest.mark.parametrize("stage,expected", [
    ("dw_nowin--ordinary", Q36_ORDINARY),
    ("dw_nowin--mtp", Q36_MTP),
])
def test_qwen36_decode_wins_are_required_and_observed(qualify, stage, expected):
    initial, final = _q36(stage)
    required = required_feature_checks(final["settings"])
    assert _features(expected) <= required
    observed = qualify.feature_observations(final, initial=initial)
    assert {name: observed[name] for name in expected} == {
        name: observed[name] for name in expected if observed[name] > 0
    }
    # Every decode slice the smoke ran is observable: nothing unobservable.
    assert not qualify.unobservable_features(
        {name.removeprefix("feature_") for name in required}
    )


def test_qwen36_routed_decode_reads_the_one_token_share_of_decode_wins(qualify):
    initial, final = _q36("dw_nowin--ordinary")
    observed = qualify.feature_observations(final, initial=initial)
    assert observed["moe_routed_decode"] == 37680
    assert observed["moe_topk_fold"] == 37680
    # Window calls are not one-token routed decode.
    final["execution"]["decode_wins"]["moe"]["window_calls"] = 37680
    observed = qualify.feature_observations(final, initial=initial)
    assert observed["moe_routed_decode"] == 0
    assert observed["qwen36_moe_window"] == 37680


def test_qwen36_gdn_slice_reachability_follows_the_route():
    _, final = _q36("dw_nowin--ordinary")
    ordinary = final["settings"]
    assert "feature_qwen36_fused_gdn_verify" not in required_feature_checks(ordinary)
    skipped = selected_not_observed_features(ordinary)
    assert "native MTP" in skipped["feature_qwen36_fused_gdn_verify"]
    assert "native MTP" in skipped["feature_qwen36_fused_gdn_batch_verify"]
    _, final = _q36("dw_nowin--mtp")
    # The 2026-10-02 default handoff width 1: batched verify never forms,
    # the 4-lane batch hands off to batched decode.
    mtp = {**final["settings"],
           "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 1}}
    required = required_feature_checks(mtp)
    assert {"feature_qwen36_fused_gdn_verify",
            "feature_qwen36_fused_gdn_batch_decode"} <= required
    assert "feature_qwen36_fused_gdn_batch_verify" not in required
    assert "width 1" in selected_not_observed_features(mtp)[
        "feature_qwen36_fused_gdn_batch_verify"]


def test_qwen36_moe_window_is_required_when_selected(qualify):
    _, final = _q36("dw_all--ordinary")
    assert "feature_qwen36_moe_window" in required_feature_checks(final["settings"])


# -- NAX gather on any model ----------------------------------------------


def test_nax_gather_is_required_from_a_settings_selection():
    _, final = _q36("nax_gather--ordinary")
    settings = final["settings"]
    # The sweep's receipt could not tell: the selection was nowhere.
    assert "feature_moe_nax_gather" not in required_feature_checks(settings)
    carried = {**settings, "moe_nax_gather": "gather"}
    assert "feature_moe_nax_gather" in required_feature_checks(carried)
    assert "feature_moe_nax_gather" not in required_feature_checks(
        {**settings, "moe_nax_gather": "off"})


def test_serving_records_an_inherited_nax_selection(monkeypatch):
    from mlx2 import serving

    fake = SimpleNamespace(MODE="fused")
    monkeypatch.setitem(sys.modules, serving._MOE_NAX_GATHER_MODULE, fake)
    assert serving._moe_nax_gather_setting({}) == {"moe_nax_gather": "fused"}
    # A profile that pins it (Flash-Next) already carries it.
    assert serving._moe_nax_gather_setting({"MLX2_MOE_NAX_GATHER": "fused"}) == {}
    fake.MODE = "off"
    assert serving._moe_nax_gather_setting({}) == {}


def test_serving_status_adds_counters_the_adapter_does_not_report(monkeypatch):
    from mlx2 import serving

    nax = SimpleNamespace(MODE="gather", calls={"gather": 0},
                          status=lambda: {"mode": "gather", "calls": {"gather": 3}})
    switch = SimpleNamespace(moe_pad_status=lambda: {"policy": "adaptive",
                                                     "choices": {"cost_qmv": 2}})
    gdn = SimpleNamespace(gdn_core_status=lambda: {"enabled": True, "calls": 5})
    monkeypatch.setitem(sys.modules, serving._MOE_NAX_GATHER_MODULE, nax)
    monkeypatch.setitem(sys.modules, serving._SWITCH_LAYERS_MODULE, switch)
    monkeypatch.setitem(sys.modules, serving._GATED_DELTA_MODULE, gdn)
    adapter = SimpleNamespace(diagnostics=lambda: {"decode_wins": {}})
    execution = serving._execution_diagnostics(adapter)
    assert execution["moe_nax_gather"]["calls"] == {"gather": 3}
    assert execution["moe_pad"]["policy"] == "adaptive"
    assert execution["gdn_core"] == {"enabled": True, "calls": 5}
    # The adapter's own report wins; defaults add nothing.
    own = SimpleNamespace(diagnostics=lambda: {"moe_nax_gather": {"own": True}})
    assert serving._execution_diagnostics(own)["moe_nax_gather"] == {"own": True}
    nax.MODE = "off"
    switch.moe_pad_status = lambda: {"policy": "floor", "choices": {}}
    gdn.gdn_core_status = lambda: {"enabled": False, "calls": 0}
    assert serving._execution_diagnostics(adapter) == {"decode_wins": {}}


# -- Adaptive sorted-MoE pad ----------------------------------------------


def test_adaptive_pad_is_required_and_observed(qualify):
    _, final = _q36("rhs_pad_adaptive--ordinary")
    settings = final["settings"]
    assert settings["moe_rhs_pad"]["policy"] == "adaptive"
    assert "feature_moe_rhs_pad" in required_feature_checks(settings)
    # The Flash-Next profile pin alone also selects it.
    assert "feature_moe_rhs_pad" in required_feature_checks(
        {"environment": {"MLX2_MOE_RHS_PAD_POLICY": "adaptive"}})
    assert "feature_moe_rhs_pad" not in required_feature_checks(
        {"environment": {"MLX2_MOE_RHS_PAD_POLICY": "floor"}})
    # A zero floor turns padding off whatever the policy variable says.
    assert "feature_moe_rhs_pad" not in required_feature_checks(
        {"environment": {"MLX2_MOE_RHS_PAD_POLICY": "adaptive"},
         "moe_rhs_pad": {"policy": "off", "min_rows_per_expert": 0}})
    before = {"execution": {"moe_pad": {"choices": {"cost_qmv": 4}}}}
    after = {"execution": {"moe_pad": {"choices": {"cost_qmv": 10, "cost_rhs": 2}}}}
    assert qualify.feature_observations(after, initial=before)["moe_rhs_pad"] == 8
    assert qualify.feature_observations(before, initial=before)["moe_rhs_pad"] == 0


def test_invariant_lane_suppresses_the_pad_requirement():
    from mlx2.runtime.prefill_plan import execution_identity

    settings = {"moe_rhs_pad": {"policy": "adaptive", "min_rows_per_expert": 3},
                "prefill_execution": execution_identity(
                    invariant={"schema": "invariant-prefill-v1", "law": {"x": 1}})}
    assert "feature_moe_rhs_pad" not in required_feature_checks(settings)
    assert "invariant_prefill" in selected_not_observed_features(settings)[
        "feature_moe_rhs_pad"]


# -- Serving policies (27B sweep) -----------------------------------------


def test_decode_first_is_required_where_prompt_work_meets_decode(qualify):
    final = _b27("decode_first_order")
    settings = final["settings"]
    assert "feature_decode_first" in required_feature_checks(settings)
    assert qualify.feature_observations(final)["decode_first"] == 583
    one_lane = {**settings, "max_lanes": 1}
    assert "feature_decode_first" not in required_feature_checks(one_lane)
    assert "max_lanes 1" in selected_not_observed_features(one_lane)["feature_decode_first"]
    off = {**settings, "decode_first": None}
    assert "feature_decode_first" not in selected_not_observed_features(off)
    assert "feature_decode_first" not in required_feature_checks(off)


def test_invariant_prefill_is_required_and_observed(qualify):
    final = _b27("invariant_mtp_lane_off")
    assert "feature_invariant_prefill" in required_feature_checks(final["settings"])
    assert qualify.feature_observations(final)["invariant_prefill"] == 67
    assert "feature_invariant_prefill" not in required_feature_checks(
        _b27("default_mtp")["settings"])


def test_fp32_head_logits_is_required_and_observed(qualify):
    final = _b27("fp32_head_mtp")
    assert "feature_fp32_head_logits" in required_feature_checks(final["settings"])
    assert qualify.feature_observations(final)["fp32_head_logits"] == 1
    # Flash-Next records it in its adapter policy.
    assert "feature_fp32_head_logits" in required_feature_checks(
        {"adapter_policy": {"fp32_head_logits": True}})
    # Selected but not installed: not observed.
    final["execution"].pop("fp32_head_logits")
    assert qualify.feature_observations(final)["fp32_head_logits"] == 0


def test_fp16_gdn_state_is_required_and_observed(qualify):
    final = _b27("gdn_state_fp16_mtp")
    assert "feature_gdn_state_fp16" in required_feature_checks(final["settings"])
    assert qualify.feature_observations(final)["gdn_state_fp16"] == 43584
    assert "feature_gdn_state_fp16" in required_feature_checks(
        {"adapter_policy": {"gdn_state_dtype": "float16"}})
    assert "feature_gdn_state_fp16" not in required_feature_checks(
        _b27("default_mtp")["settings"])


def test_gdn_core_is_required_and_counted(qualify):
    final = _b27("gdn_core_mtp")
    assert "feature_gdn_core" in required_feature_checks(final["settings"])
    # The sweep had no counter: nothing observable.
    assert qualify.feature_observations(final)["gdn_core"] == 0
    before = {"execution": {"gdn_core": {"enabled": True, "calls": 10}}}
    after = {"execution": {"gdn_core": {"enabled": True, "calls": 58}}}
    assert qualify.feature_observations(after, initial=before)["gdn_core"] == 48


def test_gdn_core_counter_counts_core_dispatches(monkeypatch):
    import mlx.core as mx

    from mlx2.runtime.models import gated_delta as gd

    calls = []
    monkeypatch.setattr(gd, "_can_use_core_gated_delta", lambda *a: True)
    monkeypatch.setattr(gd, "_core_gated_delta_update",
                        lambda *a, **k: calls.append(1) or ("y", "state"))
    monkeypatch.setattr(gd, "_ENABLE_GDN_CORE", True)
    monkeypatch.setattr(gd.mx, "default_device", lambda: gd.mx.gpu)
    monkeypatch.setattr(gd.mx.metal, "is_available", lambda: True)
    before = gd.gdn_core_status()["calls"]
    q = k = mx.zeros((1, 32, 16, 128))
    v = mx.zeros((1, 32, 32, 128))
    a = b = mx.zeros((1, 32, 32))
    gd.gated_delta_update(q, k, v, a, b, mx.zeros((32,)), mx.zeros((32,)))
    status = gd.gdn_core_status()
    assert calls and status["calls"] == before + 1
    assert status["enabled"] is True


def test_slice_floor_is_contention_gated(qualify):
    final = _b27("slice_floor_1024")
    settings = final["settings"]
    assert "feature_decode_fairness_slice_floor" in required_feature_checks(settings)
    assert qualify.feature_observations(final)["decode_fairness_slice_floor"] == 6
    assert contention_gated_not_observed(final) == {}
    # Constructed, never lifted: selected, not observed.
    idle = {"scheduler": {"decode_fairness_slice_floor_lifts": 6}}
    assert "slice floor" in contention_gated_not_observed(idle, idle)[
        "decode_fairness_slice_floor"]
    # Never constructed (no counter at all) stays a failure.
    assert contention_gated_not_observed({"scheduler": {}}) == {}
    one_lane = {**settings, "max_lanes": 1}
    assert "feature_decode_fairness_slice_floor" not in required_feature_checks(one_lane)
    assert "max_lanes 1" in selected_not_observed_features(one_lane)[
        "feature_decode_fairness_slice_floor"]


def _slice_floor_record(tmp_path, selected_not_observed=None, checks=None):
    from mlx2.adapters.qwen import QWEN4_FLASH_NEXT

    settings = {"mtp": False, "max_context": 32768, "max_lanes": 4,
                "decode_time_fairness": {"enabled": True, "slice_floor": 1024}}
    record = {
        "passed": True, "runtime": {"source": "abc"}, "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {**{c: {"passed": True}
                      for c in REQUIRED_CHECKS | {"structured_output"}},
                   **(checks or {})},
    }
    if selected_not_observed is not None:
        record["selected_not_observed"] = selected_not_observed
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    return lambda: load_qualified_route(
        path, runtime=record["runtime"], artifact="weights", settings=settings,
        descriptor=QWEN4_FLASH_NEXT, name="slice-floor")


def test_loader_accepts_a_contention_gated_entry_but_not_a_bare_omission(tmp_path):
    with pytest.raises(ValueError, match="decode_fairness_slice_floor"):
        _slice_floor_record(tmp_path)()
    entry = {"status": "selected, not observed", "reason": "no contended slice",
             "contention_gated": True}
    assert _slice_floor_record(
        tmp_path, {"feature_decode_fairness_slice_floor": entry})().profile
    with pytest.raises(ValueError, match="decode_fairness_slice_floor"):
        _slice_floor_record(tmp_path, {"feature_decode_fairness_slice_floor": {
            **entry, "contention_gated": False}})()
    assert _slice_floor_record(
        tmp_path, checks={"feature_decode_fairness_slice_floor": {"passed": True}})().profile


# -- Copy-draft strong span 14 --------------------------------------------


def test_copy_draft_strong_span_does_not_change_qualifier_requirements():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    def settings(strong):
        return {"environment": FlashNextPolicy().environment(), "max_context": 32768,
                "max_lanes": 4, "mtp": True, "speculation": "self_mtp",
                "execution_policy": {"num_draft": 2},
                "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 3},
                "self_mtp_copy_draft": {
                    "enabled": True, "max_span": 7, "min_match": 8,
                    "strong_match": 32, "strong_max_span": strong, "initial_span": 7}}

    assert required_feature_checks(settings(14)) == required_feature_checks(settings(16))
    assert selected_not_observed_features(settings(14)) == selected_not_observed_features(
        settings(16))
    assert "feature_self_mtp_copy_draft" in required_feature_checks(settings(14))
