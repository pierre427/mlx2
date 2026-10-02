"""The qualifier must require every default-on mechanism to engage (sweep H1).

The e8861bb5 Flash-Next qualification passed with fused GDN batch decode at
0 calls: nothing required the default-on kernels (HC decode, fused GDN batch
decode/verify, attention fused rows, MoE top-k fold, QSA fused scores, MoE
routed decode; the 27B fused_gdn) to run.  A route that structurally cannot
engage one is recorded as "selected, not observed", not as a pass.
"""

import importlib
import sys
from pathlib import Path

import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.qualification import required_feature_checks, selected_not_observed_features

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DEFAULT_ON = {
    "feature_hc_decode",
    "feature_attn_fused_rows",
    "feature_moe_topk_fold",
    "feature_moe_routed_decode",
}
LONG = {"feature_attn_fused_rows_qsa_mask", "feature_qsa_fused_scores"}


@pytest.fixture(scope="module")
def qualify():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("qualify_serving")
    finally:
        sys.path.remove(str(SCRIPTS))


def _settings(**overrides):
    settings = {"environment": FlashNextPolicy().environment(), "max_context": 262144,
                "max_lanes": 4, "mtp": False, "speculation": "ordinary"}
    settings.update(overrides)
    return settings


def test_ordinary_route_requires_every_default_on_mechanism():
    required = required_feature_checks(_settings())
    assert DEFAULT_ON | LONG | {"feature_fused_gdn_batch_decode"} <= required
    assert "feature_fused_gdn_batch_verify" not in required
    assert "feature_fused_gdn_batch_verify" in selected_not_observed_features(_settings())


def test_mtp_route_with_handoff_requires_batch_decode_and_verify():
    settings = _settings(mtp=True, speculation="self_mtp",
                         mtp_ordinary_handoff={"enabled": True, "max_mtp_width": 3})
    required = required_feature_checks(settings)
    assert DEFAULT_ON | LONG <= required
    assert {"feature_fused_gdn_batch_decode", "feature_fused_gdn_batch_verify"} <= required


@pytest.mark.parametrize("handoff", [None, {"enabled": True, "max_mtp_width": 4}])
def test_mtp_route_that_cannot_batch_decode_says_so(handoff):
    settings = _settings(mtp=True, speculation="self_mtp")
    if handoff:
        settings["mtp_ordinary_handoff"] = handoff
    required = required_feature_checks(settings)
    assert "feature_fused_gdn_batch_decode" not in required
    assert "feature_fused_gdn_batch_verify" in required
    reason = selected_not_observed_features(settings)["feature_fused_gdn_batch_decode"]
    assert "native MTP" in reason


def test_short_context_labels_the_long_context_kernels():
    settings = _settings(max_context=8192)
    required = required_feature_checks(settings)
    assert not LONG & required
    assert LONG <= set(selected_not_observed_features(settings))


def test_index_q_is_recorded_but_optional():
    assert "feature_attn_fused_rows_index_q" not in required_feature_checks(_settings())
    assert "feature_attn_fused_rows_index_q" in selected_not_observed_features(_settings())


def test_unselected_mechanisms_are_not_required():
    policy = FlashNextPolicy(hc_decode_kernels=False, fused_gdn_batch_decode="off",
                             fused_gdn_batch_verify="off", attn_fused_rows=False,
                             moe_topk_fold="off", qsa_fused_scores=False,
                             moe_routed_decode="off")
    settings = _settings(environment=policy.environment())
    assert not (DEFAULT_ON | LONG) & required_feature_checks(settings)
    assert selected_not_observed_features(settings) == {}


def test_27b_fused_gdn_is_required_when_selected(qualify):
    settings = {"environment": {"MLX2_QWEN38_FUSED_GDN": "1"}, "mtp": True,
                "max_context": 131072, "max_lanes": 4}
    assert "feature_qwen38_fused_gdn" in required_feature_checks(settings)
    final = {"execution": {"fused_gdn": {"enabled": True, "decode_calls": 500,
                                         "batch_decode_calls": 3, "fallbacks": 0}}}
    initial = {"execution": {"fused_gdn": {"enabled": True, "decode_calls": 0,
                                           "batch_decode_calls": 0}}}
    assert qualify.feature_observations(final, initial=initial)["qwen38_fused_gdn"] == 503
    stale = {"execution": {"fused_gdn": {"enabled": True, "decode_calls": 500,
                                         "batch_decode_calls": 3}}}
    assert qualify.feature_observations(final, initial=stale)["qwen38_fused_gdn"] == 0


def test_every_required_default_on_feature_is_observable(qualify):
    for settings in (_settings(), _settings(mtp=True, speculation="self_mtp",
                     mtp_ordinary_handoff={"enabled": True, "max_mtp_width": 3})):
        required = {n.removeprefix("feature_") for n in required_feature_checks(settings)}
        assert qualify.unobservable_features(required) == []
    assert "attn_fused_rows_index_q" in qualify.feature_observations({})


def _status(**execution):
    return {"settings": {"environment": FlashNextPolicy().environment()},
            "execution": execution}


def test_observations_count_engagement_not_selection(qualify):
    selected_idle = _status(
        hc_decode={"enabled": True, "broken": False, "errors": 0, "calls": 0},
        fused_gdn={"batch_decode": {"calls": 0}, "batch_verify": {"calls": 0}},
        attn_fused_rows={"enabled": True, "kernel_failures": [], "counts": {}},
        moe={"routed_decode": {"calls": 0}, "moe_window": {"topk_calls": {"launch": 0}}},
        tensorfold_longctx={"qsa_fused_scores": {"enabled": True, "counts": {}}},
    )
    observed = qualify.feature_observations(selected_idle, initial=selected_idle)
    for name in ("hc_decode", "fused_gdn_batch_decode", "fused_gdn_batch_verify",
                 "attn_fused_rows", "moe_topk_fold", "moe_routed_decode", "qsa_fused_scores"):
        assert observed[name] == 0, name
    engaged = _status(
        hc_decode={"enabled": True, "broken": False, "errors": 0, "calls": 10},
        fused_gdn={"batch_decode": {"calls": 4}, "batch_verify": {"calls": 5}},
        attn_fused_rows={"enabled": True, "kernel_failures": [], "counts": {
            "projection_grouped": 7, "prep_rows": 9, "sdpa_1pass_rows": 3,
            "sdpa_2pass_rows": 2, "mask_rows": 6}},
        moe={"routed_decode": {"calls": 8, "down_calls": 8, "shared_fold_calls": 6},
             "moe_window": {"topk_calls": {"launch": 8, "fold": 0}}},
        tensorfold_longctx={"qsa_fused_scores": {"enabled": True, "counts": {"engaged": 11}}},
    )
    observed = qualify.feature_observations(engaged, initial=selected_idle)
    assert observed["hc_decode"] == 10
    assert observed["fused_gdn_batch_decode"] == 4
    assert observed["fused_gdn_batch_verify"] == 5
    assert observed["attn_fused_rows"] == 5
    assert observed["attn_fused_rows_qsa_mask"] == 6
    assert observed["moe_topk_fold"] == 8
    assert observed["moe_routed_decode"] == 6
    assert observed["qsa_fused_scores"] == 11
    broken = _status(hc_decode={"enabled": True, "broken": True, "errors": 1, "calls": 10})
    assert qualify.feature_observations(broken)["hc_decode"] == 0


def test_e8861bb5_mtp_receipt_shape_records_batch_decode_as_not_observed():
    """The served e8861bb5 MTP settings: handoff off, so batch decode is
    labelled, while every other default-on kernel is required."""
    env = FlashNextPolicy().environment()
    settings = {"environment": env, "max_context": 32768, "max_lanes": 4, "mtp": True}
    required = required_feature_checks(settings)
    assert {"feature_hc_decode", "feature_attn_fused_rows", "feature_moe_topk_fold",
            "feature_moe_routed_decode", "feature_qsa_fused_scores",
            "feature_fused_gdn_batch_verify"} <= required
    assert "feature_fused_gdn_batch_decode" in selected_not_observed_features(settings)
