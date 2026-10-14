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
FN = ROOT / "qualification/runs/options-sweep-flashnext-20261002/smoke"


def _private_json(path):
    if not path.is_file():
        pytest.skip(f"private qualification receipt is not exported: {path.name}")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def qualify():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("qualify_serving")
    finally:
        sys.path.remove(str(SCRIPTS))


def _q36(stage):
    initial = _private_json(Q36 / stage / "status-initial.json")
    final = _private_json(Q36 / stage / "status-final.json")
    return initial, final


def _b27(arm):
    """The 27B smoke's final status (the receipt keeps no initial one)."""
    return _private_json(B27 / arm / "qualification.json")["final_status"]


def _features(names):
    return {"feature_" + name for name in names}


# -- Proposal composition --------------------------------------------------


def _composition_response(
    policy,
    *,
    trust_record=None,
    post_trust_record=None,
    audited=False,
    periodic_audit=False,
    trusted=False,
    frozen_override=None,
):
    from mlx2.runtime.proposal_composition import ProposalCompositionPolicy

    normalized = ProposalCompositionPolicy.from_value(policy).as_dict()
    frozen = dict(normalized)
    frozen["trusted_pld"] = {
        "enabled": normalized["trusted_pld"],
        "min_score": normalized["trusted_pld_min_score"],
        "min_verified_tokens": normalized["trusted_pld_min_verified_tokens"],
        "min_match": normalized["trusted_pld_min_match"],
        "max_span": normalized["trusted_pld_max_span"],
        "audit_interval": normalized["trusted_pld_audit_interval"],
    }
    frozen.update(frozen_override or {})
    trust_key = "prompt_lookup:match_4:span_4"
    return {
        "mlx2": {
            "speculation": {
                "draft_settings": {"proposal_composition": frozen},
                "proposal_composition": {
                    "selected": True,
                    # External source use was historically omitted from this
                    # convenience bit. Qualification reads committed source
                    # evidence instead of trusting it.
                    "observed_used": False,
                    "ranking": {
                        "order": "score_then_span_then_source",
                        "authority": "lagged_committed_verification_only",
                        "score_counts": {
                            "prompt_lookup:match_4": {
                                "weighted_accepted_tokens": 4.0,
                                "weighted_proposed_tokens": 4.0,
                            },
                            "native_mtp": {
                                "weighted_accepted_tokens": 2.0,
                                "weighted_proposed_tokens": 3.0,
                            },
                            "external": {
                                "weighted_accepted_tokens": 1.0,
                                "weighted_proposed_tokens": 2.0,
                            },
                        },
                    },
                    "trusted_pld": {
                        "scope": "request_lane",
                        "sources": (
                            {} if trust_record is None else {trust_key: trust_record}
                        ),
                    },
                    "current_trust_key": trust_key,
                    "current_score_key": "prompt_lookup:match_4",
                    "current_source": "prompt_lookup",
                    "current_audit": audited,
                    "current_periodic_audit": periodic_audit,
                    "current_trusted": trusted,
                    "postcommit_learning": {
                        "scope": "request_lane",
                        "observed_proposed_tokens": 4,
                        "observed_accepted_tokens": 4,
                        "trust_key": trust_key,
                        "score_key": "prompt_lookup:match_4",
                        "audited": audited,
                        "periodic_audit": periodic_audit,
                        "trusted": trusted,
                        "trusted_pld": {
                            "scope": "request_lane",
                            "sources": (
                                {}
                                if post_trust_record is None
                                else {trust_key: post_trust_record}
                            ),
                        },
                    },
                    "adapter_diagnostics": {"scope": "process_global"},
                    "committed_verification": {
                        source: {
                            "verified_rounds": 1,
                            "proposed_tokens": 4,
                            "accepted_tokens": 3,
                        }
                        for source in ("prompt_lookup", "native_mtp", "external")
                    },
                },
            }
        }
    }


def test_proposal_composition_checks_follow_selected_sources_only():
    settings = {
        "speculation": "external_draft",
        "execution_policy": {"proposal_composition": {"prompt_lookup": True}},
    }
    required = required_feature_checks(settings)
    assert {
        "feature_proposal_composition",
        "feature_proposal_composition_pld",
        "feature_proposal_composition_external_source",
        "feature_proposal_composition_lane_ranking",
    } <= required
    assert "feature_proposal_composition_native_mtp" not in required
    assert not {name for name in required if name.startswith("feature_trusted_pld_")}

    settings["execution_policy"]["proposal_composition"] = {
        "prompt_lookup": True,
        "native_mtp": True,
        "trusted_pld": True,
    }
    required = required_feature_checks(settings)
    assert "feature_proposal_composition_native_mtp" in required
    assert {
        "feature_trusted_pld_calibration",
        "feature_trusted_pld_blind",
        "feature_trusted_pld_periodic_audit",
        "feature_trusted_pld_revocation",
        "feature_trusted_pld_recovery",
    } <= required

    settings["execution_policy"]["proposal_composition"] = False
    required = required_feature_checks(settings)
    assert not {
        name
        for name in required
        if "proposal_composition" in name or "trusted_pld" in name
    }


def test_composition_observation_requires_bound_lane_local_receipts(qualify):
    policy = {
        "prompt_lookup": True,
        "native_mtp": True,
        "trusted_pld": True,
        "trusted_pld_min_verified_tokens": 4,
        "trusted_pld_audit_interval": 2,
    }
    settings = {
        "speculation": "external_draft",
        "execution_policy": {"proposal_composition": policy},
    }
    cold = {
        "blind_rounds": 0,
        "blind_tokens": 0,
        "blind_since_audit": 0,
        "audit_rounds": 0,
        "audit_proposed_tokens": 0,
        "audit_accepted_tokens": 0,
        "audit_failures": 0,
        "periodic_audit_rounds": 0,
        "revoked": False,
        "exact_accepted_tokens": 0,
        "exact_proposed_tokens": 0,
    }
    calibrated = {
        **cold,
        "audit_rounds": 1,
        "audit_proposed_tokens": 4,
        "audit_accepted_tokens": 4,
        "exact_accepted_tokens": 4,
        "exact_proposed_tokens": 4,
    }
    blind = {
        **calibrated,
        "blind_rounds": 1,
        "blind_tokens": 4,
        "blind_since_audit": 1,
    }
    periodic = {
        **blind,
        "blind_since_audit": 0,
        "audit_rounds": 2,
        "audit_proposed_tokens": 8,
        "audit_accepted_tokens": 8,
        "periodic_audit_rounds": 1,
        "exact_accepted_tokens": 8,
        "exact_proposed_tokens": 8,
    }
    revoked = {
        **periodic,
        "blind_since_audit": 0,
        "audit_rounds": 3,
        "audit_proposed_tokens": 12,
        "audit_accepted_tokens": 11,
        "audit_failures": 1,
        "revoked": True,
        "exact_accepted_tokens": 0,
        "exact_proposed_tokens": 0,
    }
    recovered = {
        **revoked,
        "audit_rounds": 4,
        "audit_proposed_tokens": 16,
        "audit_accepted_tokens": 15,
        "revoked": False,
        "exact_accepted_tokens": 4,
        "exact_proposed_tokens": 4,
    }
    responses = [
        _composition_response(
            policy,
            trust_record=cold,
            post_trust_record=calibrated,
            audited=True,
        ),
        _composition_response(
            policy,
            trust_record=calibrated,
            post_trust_record=blind,
            trusted=True,
        ),
        _composition_response(
            policy,
            trust_record=blind,
            post_trust_record=periodic,
            audited=True,
            periodic_audit=True,
        ),
        _composition_response(
            policy,
            trust_record=revoked,
            post_trust_record=recovered,
            audited=True,
        ),
    ]
    evidence = qualify.proposal_composition_evidence(responses, settings)
    observed = qualify.feature_observations(
        {"settings": settings},
        proposal_evidence=evidence,
    )
    assert evidence["source_bound_receipts"] == 4
    assert observed["proposal_composition"] > 0
    assert observed["proposal_composition_pld"] > 0
    assert observed["proposal_composition_native_mtp"] > 0
    assert observed["proposal_composition_external_source"] > 0
    assert observed["proposal_composition_lane_ranking"] > 0
    assert all(
        observed[name] > 0
        for name in (
            "trusted_pld_calibration",
            "trusted_pld_blind",
            "trusted_pld_periodic_audit",
            "trusted_pld_revocation",
            "trusted_pld_recovery",
        )
    )

    mismatched = [_composition_response(
        policy,
        trust_record=recovered,
        post_trust_record=recovered,
        frozen_override={"lookback": 999},
    )]
    evidence = qualify.proposal_composition_evidence(mismatched, settings)
    observed = qualify.feature_observations(
        {"settings": settings}, proposal_evidence=evidence
    )
    assert evidence["selected_receipts"] == 1
    assert evidence["source_bound_receipts"] == 0
    assert observed["proposal_composition"] == 0
    assert observed["proposal_composition_external_source"] == 0


def test_trusted_pld_periodic_gate_requires_a_post_blind_audit_transition(qualify):
    policy = {
        "prompt_lookup": True,
        "trusted_pld": True,
        "trusted_pld_min_verified_tokens": 4,
        "trusted_pld_audit_interval": 2,
    }
    settings = {
        "speculation": "external_draft",
        "execution_policy": {"proposal_composition": policy},
    }
    # Multiple calibration audits plus a blind copy are not evidence that the
    # configured post-blind cadence ever forced an exact audit.
    before = {
        "blind_rounds": 0,
        "blind_tokens": 0,
        "blind_since_audit": 0,
        "audit_rounds": 4,
        "audit_proposed_tokens": 16,
        "audit_accepted_tokens": 16,
        "audit_failures": 0,
        "periodic_audit_rounds": 0,
        "revoked": False,
        "exact_accepted_tokens": 16,
        "exact_proposed_tokens": 16,
    }
    after = {
        **before,
        "blind_rounds": 1,
        "blind_tokens": 4,
        "blind_since_audit": 1,
    }
    evidence = qualify.proposal_composition_evidence(
        [
            _composition_response(
                policy,
                trust_record=before,
                post_trust_record=after,
                trusted=True,
            )
        ],
        settings,
    )
    assert evidence["trusted_pld"]["calibration"] > 0
    assert evidence["trusted_pld"]["blind"] > 0
    assert evidence["trusted_pld"]["periodic_audit"] == 0


def test_trusted_pld_gate_accepts_terminal_carry_of_fresh_postcommit_state(qualify):
    policy = {
        "prompt_lookup": True,
        "trusted_pld": True,
        "trusted_pld_min_verified_tokens": 4,
        "trusted_pld_audit_interval": 2,
    }
    settings = {
        "speculation": "external_draft",
        "execution_policy": {"proposal_composition": policy},
    }
    before = {
        "blind_rounds": 1,
        "blind_tokens": 4,
        "blind_since_audit": 1,
        "audit_rounds": 1,
        "periodic_audit_rounds": 0,
        "audit_proposed_tokens": 4,
        "audit_accepted_tokens": 4,
        "audit_failures": 0,
        "revoked": False,
        "exact_accepted_tokens": 4,
        "exact_proposed_tokens": 4,
    }
    after = {
        **before,
        "blind_since_audit": 0,
        "audit_rounds": 2,
        "periodic_audit_rounds": 1,
        "audit_proposed_tokens": 8,
        "audit_accepted_tokens": 8,
        "exact_accepted_tokens": 8,
        "exact_proposed_tokens": 8,
    }
    terminal = _composition_response(
        policy,
        trust_record=before,
        post_trust_record=after,
        audited=True,
        periodic_audit=True,
    )
    composition = terminal["mlx2"]["speculation"]["proposal_composition"]
    composition.update(
        current_source="ordinary",
        current_trust_key=None,
        current_score_key=None,
        current_audit=False,
        current_periodic_audit=False,
    )
    composition["trusted_pld"]["sources"] = {
        "prompt_lookup:match_4:span_4": after
    }
    evidence = qualify.proposal_composition_evidence([terminal], settings)
    assert evidence["trusted_pld"]["periodic_audit"] > 0


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


def test_qwen36_independent_moe_window_reachability():
    base = {
        "environment": {
            "MLX_QWEN36_MOE_WINDOW_BATCH_DECODE": "1",
            "MLX_QWEN36_MOE_WINDOW_VERIFY": "1",
        },
        "max_lanes": 4,
        "mtp": True,
        "speculation": "self_mtp",
        "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 1},
    }
    assert {
        "feature_moe_window_batch_decode",
        "feature_moe_window_verify",
    } <= required_feature_checks(base)
    ordinary_b1 = {**base, "max_lanes": 1, "mtp": False, "speculation": None}
    skipped = selected_not_observed_features(ordinary_b1)
    assert "no batched decode" in skipped["feature_moe_window_batch_decode"]
    assert "no native MTP verify" in skipped["feature_moe_window_verify"]


def test_qwen36_independent_moe_window_observations_are_run_deltas(qualify):
    initial = {
        "execution": {
            "decode_wins": {
                "moe": {
                    "window_calls_by_consumer": {
                        "batch_decode": 11,
                        "verify": 13,
                        "row_exact": 17,
                    }
                }
            }
        }
    }
    final = json.loads(json.dumps(initial))
    calls = final["execution"]["decode_wins"]["moe"][
        "window_calls_by_consumer"
    ]
    calls["batch_decode"] = 18
    calls["verify"] = 24
    observed = qualify.feature_observations(final, initial=initial)
    assert observed["moe_window_batch_decode"] == 7
    assert observed["moe_window_verify"] == 11


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


def test_serving_does_not_inherit_nax_selection_from_process_state(monkeypatch):
    from mlx2 import serving

    monkeypatch.setenv("MLX2_MOE_NAX_GATHER", "fused")
    assert not hasattr(serving, "_moe_nax_gather_mode")
    assert not hasattr(serving, "_moe_nax_gather_setting")


def test_serving_status_uses_only_route_owned_diagnostics(monkeypatch):
    from mlx2 import serving

    nax = SimpleNamespace(MODE="gather", calls={"gather": 0},
                          status=lambda: {"mode": "gather", "calls": {"gather": 3}})
    switch = SimpleNamespace(moe_pad_status=lambda: {"policy": "adaptive",
                                                     "choices": {"cost_qmv": 2}})
    gdn = SimpleNamespace(gdn_core_status=lambda: {"enabled": True, "calls": 5})
    adapter = SimpleNamespace(diagnostics=lambda: {"decode_wins": {}})
    assert serving._execution_diagnostics(adapter) == {"decode_wins": {}}
    # The adapter's own report wins; defaults add nothing.
    own = SimpleNamespace(diagnostics=lambda: {
        "moe_nax_gather": nax.status(), "moe_pad": switch.moe_pad_status(),
        "gdn_core": gdn.gdn_core_status(),
    })
    execution = serving._execution_diagnostics(own)
    assert execution["moe_nax_gather"]["calls"] == {"gather": 3}
    assert execution["moe_pad"]["policy"] == "adaptive"
    assert execution["gdn_core"] == {"enabled": True, "calls": 5}


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


def test_cache_capsules_and_persistent_blocks_require_real_engagement(qualify):
    settings = {
        "cache_capsules": {"enabled": True},
        "persistent_block_bytes": 4 << 20,
    }
    assert {
        "feature_cache_capsules",
        "feature_persistent_blocks",
    } <= required_feature_checks(settings)

    initial = {
        "cache_capsules": {"requests": 2, "primary_successes": 2},
        "counts": {"cache_capsule_prepared": 1},
        "apcv2": {
            "idle_disk": {
                "block_bytes": 4 << 20,
                "parks": 4,
                "restores": 3,
                "resumes": 3,
                "bytes_written": 100,
            }
        },
    }
    final = {
        "settings": settings,
        "cache_capsules": {"requests": 5, "primary_successes": 5},
        "counts": {"cache_capsule_prepared": 4},
        "apcv2": {
            "idle_disk": {
                "block_bytes": 4 << 20,
                "parks": 5,
                "restores": 4,
                "resumes": 4,
                "bytes_written": 200,
            }
        },
    }
    observed = qualify.feature_observations(final, initial=initial)
    assert observed["cache_capsules"] == 3
    assert observed["persistent_blocks"] == 1
    fallback_only = {
        **final,
        "counts": {"cache_capsule_prepared": initial["counts"]["cache_capsule_prepared"]},
    }
    assert qualify.feature_observations(fallback_only, initial=initial)[
        "cache_capsules"
    ] == 0
    wrong_block_width = {
        **final,
        "apcv2": {
            "idle_disk": {
                **final["apcv2"]["idle_disk"],
                "block_bytes": 8 << 20,
            }
        },
    }
    assert qualify.feature_observations(wrong_block_width, initial=initial)[
        "persistent_blocks"
    ] == 0
    assert qualify.feature_observations(initial, initial=initial)["cache_capsules"] == 0
    assert qualify.feature_observations(initial, initial=initial)["persistent_blocks"] == 0


def test_cache_capsule_output_probe_is_exact(qualify):
    reference = {
        "choices": [{"message": {"content": "\nCAPSULE_READY\n"}}]
    }
    fanout = {
        "choices": [
            {"message": {"content": "CAPSULE_READY"}},
            {"message": {"content": " CAPSULE_READY"}},
            {"message": {"content": "CAPSULE_READY\n"}},
        ]
    }
    assert qualify.cache_capsule_probe_passes(reference, fanout)
    fanout["choices"][2]["message"]["content"] = "CAPSULE_READY extra"
    assert not qualify.cache_capsule_probe_passes(reference, fanout)


@pytest.mark.parametrize(
    "policy,feature",
    [
        ({"ple_early_dispatch": True}, "feature_ple_early_dispatch"),
        (
            {"qsa_stage1_direct_selector": "direct8"},
            "feature_qsa_stage1_direct_selector",
        ),
        ({"ple_read_policy": "serial"}, "feature_ple_read_policy"),
        ({"tensorfold_qmv_rows": True}, "feature_tensorfold_qmv_rows"),
        (
            {"qsa_batch_decode_sparse": "indexed"},
            "feature_qsa_batch_decode_sparse",
        ),
        ({"qsa_nax_batched": True}, "feature_qsa_nax_batched"),
    ],
)
def test_flash_next_default_off_options_require_engagement(policy, feature):
    settings = {
        "adapter_policy": policy,
        "max_context": 65536,
        "max_lanes": 4,
        "mtp": False,
    }
    assert feature in required_feature_checks(settings)


def test_flash_next_mtp_only_options_require_their_exact_paths():
    settings = {
        "adapter_policy": {
            "row_exact_verify": True,
            "row_exact_window_kernels": True,
            "moe_window_row_exact": True,
            "moe_window_verify": True,
            "mtp_draft_vocab": True,
        },
        "max_context": 65536,
        "max_lanes": 4,
        "mtp": True,
        "speculation": "self_mtp",
    }
    assert {
        "feature_row_exact_verify",
        "feature_row_exact_attn_window",
        "feature_moe_window_row_exact",
        "feature_moe_window_verify",
        "feature_mtp_draft_vocab",
    } <= required_feature_checks(settings)


def test_flash_next_default_off_option_observations_are_run_deltas(qualify):
    initial = {
        "execution": {
            "tensorfold_longctx": {
                "ple_early_dispatch": {"dispatches": 2, "device_ids": 2}
            },
            "qsa_stage1": {
                "candidates": {
                    "direct_selector_configured": "direct8",
                    "runtime_counts": {"direct8_topk_dispatches": 3},
                }
            },
            "ple_tables": [
                {
                    "read_policy": {
                        "configured": "serial",
                        "counts": {"foreground_serial_calls": 4},
                    }
                }
            ],
            "tensorfold_qmv": {"counters": {"kernel_calls": 5}},
            "row_exact_verify": {
                "windows_row_exact": 6,
                "windows_not_exact": 0,
                "failures": {},
                "stages": {"attention": {"window_dense": 7}},
            },
            "moe": {
                "moe_window": {
                    "calls": {"row_exact": 8, "batch_decode": 9, "verify": 10}
                }
            },
            "qsa_batch_decode_sparse": {"engagements": 11},
            "qsa_nax_prefill": {"counts": {"engaged_auto_batched": 12}},
            "mtp_draft_vocab": {"runtime_enabled": True, "reduced_head_calls": 13},
        }
    }
    final = json.loads(json.dumps(initial))
    execution = final["execution"]
    execution["tensorfold_longctx"]["ple_early_dispatch"] = {
        "dispatches": 4,
        "device_ids": 5,
    }
    execution["qsa_stage1"]["candidates"]["runtime_counts"][
        "direct8_topk_dispatches"
    ] = 7
    execution["ple_tables"][0]["read_policy"]["counts"][
        "foreground_serial_calls"
    ] = 9
    execution["tensorfold_qmv"]["counters"]["kernel_calls"] = 11
    execution["row_exact_verify"]["windows_row_exact"] = 14
    execution["row_exact_verify"]["stages"]["attention"]["window_dense"] = 15
    execution["moe"]["moe_window"]["calls"] = {
        "row_exact": 18,
        "batch_decode": 20,
        "verify": 22,
    }
    execution["qsa_batch_decode_sparse"]["engagements"] = 24
    execution["qsa_nax_prefill"]["counts"]["engaged_auto_batched"] = 26
    execution["mtp_draft_vocab"]["reduced_head_calls"] = 28
    observed = qualify.feature_observations(final, initial=initial)
    assert {
        name: observed[name]
        for name in (
            "ple_early_dispatch",
            "qsa_stage1_direct_selector",
            "ple_read_policy",
            "tensorfold_qmv_rows",
            "row_exact_verify",
            "row_exact_attn_window",
            "moe_window_row_exact",
            "moe_window_batch_decode",
            "moe_window_verify",
            "qsa_batch_decode_sparse",
            "qsa_nax_batched",
            "mtp_draft_vocab",
        )
    } == {
        "ple_early_dispatch": 2,
        "qsa_stage1_direct_selector": 4,
        "ple_read_policy": 5,
        "tensorfold_qmv_rows": 6,
        "row_exact_verify": 8,
        "row_exact_attn_window": 8,
        "moe_window_row_exact": 10,
        "moe_window_batch_decode": 11,
        "moe_window_verify": 12,
        "qsa_batch_decode_sparse": 13,
        "qsa_nax_batched": 14,
        "mtp_draft_vocab": 15,
    }


def test_row_exact_observation_fails_closed_on_any_fallback(qualify):
    initial = {
        "execution": {
            "row_exact_verify": {
                "windows_row_exact": 2,
                "windows_not_exact": 0,
                "failures": {},
            }
        }
    }
    final = {
        "execution": {
            "row_exact_verify": {
                "windows_row_exact": 5,
                "windows_not_exact": 1,
                "failures": {"attention_preselected": 1},
            }
        }
    }
    assert qualify.feature_observations(final, initial=initial)["row_exact_verify"] == 0


@pytest.mark.parametrize("route", ["ordinary", "mtp2"])
def test_historical_sparse_qsa_smoke_no_longer_passes_without_engagement(
    qualify, route
):
    final = _private_json(FN / f"qsa_sparse_indexed--{route}" / "status-final.json")
    assert "feature_qsa_batch_decode_sparse" in required_feature_checks(
        final["settings"]
    )
    assert qualify.feature_observations(final)["qsa_batch_decode_sparse"] == 0


def test_historical_row_exact_smoke_no_longer_hides_fallbacks(qualify):
    final = _private_json(FN / "row_exact_verify--mtp2" / "status-final.json")
    required = required_feature_checks(final["settings"])
    assert {
        "feature_row_exact_verify",
        "feature_row_exact_attn_window",
        "feature_moe_window_row_exact",
    } <= required
    observed = qualify.feature_observations(final)
    assert observed["row_exact_verify"] == 0
    assert observed["row_exact_attn_window"] > 0
    assert observed["moe_window_row_exact"] > 0


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
                   # The base fairness policy the floor rides on is required
                   # too; this helper isolates the floor.
                   "feature_decode_fairness": {"passed": True},
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


def test_low_bit_lane_requires_its_own_engagement(qualify):
    """Codex 2026-10-14 MEDIUM 4: the generic row_exact_verify windows could
    qualify row_exact_low_bit_lane with zero lane_exact engagement."""
    settings = {
        "adapter_policy": {"row_exact_verify": True, "row_exact_low_bit_lane": True},
        "max_context": 65536, "max_lanes": 4, "mtp": True,
    }
    assert "feature_row_exact_low_bit_lane" in required_feature_checks(settings)

    def snapshot(lane_exact, per_row=0, refused=None, count=3, intended=3):
        return {"execution": {"row_exact_verify": {
            "stages": {"projections": {"lane_exact": lane_exact, "per_row": per_row}},
            "low_bit_lane": {"count": count, "intended": intended, "refused": refused or {}},
        }}}

    initial = snapshot(2)
    assert qualify.feature_observations(snapshot(9), initial=initial)["row_exact_low_bit_lane"] == 7
    for final in (
        snapshot(2),                          # nothing engaged this run
        snapshot(9, per_row=1),               # some verify rows still per row
        snapshot(9, refused={"x": 1}),        # a layer was refused
        snapshot(9, count=2),                 # not every declined layer prepared
    ):
        assert qualify.feature_observations(final, initial=initial)["row_exact_low_bit_lane"] == 0
