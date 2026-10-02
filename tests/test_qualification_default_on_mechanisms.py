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
    "feature_moe_nax_gather",
}
# Reached once the indexer selects explicitly (past the indexer budget).
LONG = {"feature_attn_fused_rows_qsa_mask", "feature_attn_fused_rows_index_q",
        "feature_qsa_fused_scores"}


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
    # Probe floor 2304 - 256 = 2048 tokens: 512 blocks, exactly the indexer's
    # top-k, so every fused-rows call selects all blocks implicitly.
    settings = _settings(max_context=2304)
    required = required_feature_checks(settings)
    assert not LONG & required
    assert LONG <= set(selected_not_observed_features(settings))
    # One more block closes past the budget: the indexer selects explicitly.
    assert LONG <= required_feature_checks(_settings(max_context=2308))


def test_unselected_mechanisms_are_not_required():
    policy = FlashNextPolicy(hc_decode_kernels=False, fused_gdn_batch_decode="off",
                             fused_gdn_batch_verify="off", attn_fused_rows=False,
                             moe_topk_fold="off", qsa_fused_scores=False,
                             moe_routed_decode="off", moe_nax_gather="off")
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


# --- sweep 1002 review items 2-4 --------------------------------------------


def test_ordinary_8k_route_requires_the_indexer_kernels():
    """Both served artifacts have indexer_budget 2048, so the 8K near-limit
    probe (prompt > 7936 tokens) runs the explicit selection on every fused
    call; the old 16,384 indexed-attention threshold exempted all three."""
    for mtp in (False, True):
        settings = _settings(max_context=8192, mtp=mtp,
                             speculation="self_mtp" if mtp else "ordinary")
        assert LONG <= required_feature_checks(settings), mtp


def test_indexed_min_context_does_not_move_the_requirements():
    for value in (1024, 16384, 1 << 20):
        env = FlashNextPolicy(indexed_min_context=value).environment()
        assert LONG <= required_feature_checks(_settings(environment=env, max_context=8192))
        assert not LONG & required_feature_checks(
            _settings(environment=env, max_context=2304))


def test_requirements_follow_the_recorded_indexer_budget():
    geometry = {"budget": 16384, "compress_ratio": 4, "head_dim": 128, "n_heads": 4}
    settings = _settings(max_context=8192, qsa_indexer=geometry)
    assert not LONG & required_feature_checks(settings)
    assert LONG <= required_feature_checks(_settings(max_context=32768, qsa_indexer=geometry))


def test_scorer_admission_limits_are_labelled():
    narrow = {"budget": 2048, "compress_ratio": 4, "head_dim": 64, "n_heads": 4}
    settings = _settings(qsa_indexer=narrow)
    assert "feature_qsa_fused_scores" not in required_feature_checks(settings)
    assert "head_dim 64" in selected_not_observed_features(settings)["feature_qsa_fused_scores"]


def test_scorer_without_fused_rows_runs_from_the_first_wide_selection():
    """Without the fused rows no dense short-circuit applies: the indexer
    selects explicitly from the first block, and the scorer engages past
    head_dim (128) blocks."""
    env = FlashNextPolicy(attn_fused_rows=False).environment()
    assert "feature_qsa_fused_scores" in required_feature_checks(
        _settings(environment=env, max_context=1024))  # 768 tokens, 192 blocks
    assert "feature_qsa_fused_scores" not in required_feature_checks(
        _settings(environment=env, max_context=768))  # 512 tokens, 128 blocks


def test_index_q_is_required_wherever_the_fused_indexer_runs():
    required = required_feature_checks(_settings())
    assert {"feature_attn_fused_rows", "feature_attn_fused_rows_index_q"} <= required
    assert "feature_attn_fused_rows_index_q" not in selected_not_observed_features(_settings())


def _fold(**policy):
    return FlashNextPolicy(moe_topk_fold="fold", **policy).environment()


def test_fold_with_no_reachable_consumer_is_labelled_not_required():
    """Codex review item 4: fold, routed decode off and only the verify window
    on an ordinary route -- no call can fold."""
    env = _fold(moe_routed_decode="off", moe_window_verify=True)
    settings = _settings(environment=env)
    assert "feature_moe_topk_fold" not in required_feature_checks(settings)
    assert "feature_moe_topk_fold" in selected_not_observed_features(settings)


def test_fold_is_required_where_a_call_can_fold():
    # One-token calls under routed gate_up_down.
    env = _fold(moe_routed_decode="gate_up_down")
    assert "feature_moe_topk_fold" in required_feature_checks(_settings(environment=env))
    # MTP verify windows of num_draft + 1 = 3 rows.
    env = _fold(moe_routed_decode="off", moe_window_verify=True)
    mtp = _settings(environment=env, mtp=True, speculation="self_mtp",
                    execution_policy={"num_draft": 2})
    assert "feature_moe_topk_fold" in required_feature_checks(mtp)
    # Four-row verify windows take the launch instead.
    wide = dict(mtp, execution_policy={"num_draft": 3})
    assert "feature_moe_topk_fold" not in required_feature_checks(wide)
    # Batched one-token decode windows on an ordinary route.
    env = _fold(moe_routed_decode="off", moe_window_batch_decode=True)
    assert "feature_moe_topk_fold" in required_feature_checks(_settings(environment=env))
    assert "feature_moe_topk_fold" not in required_feature_checks(
        _settings(environment=env, max_lanes=1))


def test_launch_mode_stays_required():
    assert "feature_moe_topk_fold" in required_feature_checks(_settings())


SERVED = [Path.home() / "mlx-models" / name for name in (
    "Qwen3.8-Flash-Next-MLX-4bit-MTP", "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP")]


@pytest.mark.parametrize("artifact", SERVED, ids=lambda p: p.name)
def test_served_artifacts_have_indexer_budget_2048(artifact):
    import json

    from mlx2.adapters.flash_next import qsa_indexer_geometry

    if not (artifact / "config.json").exists():
        pytest.skip("artifact not present")
    config = json.loads((artifact / "config.json").read_text())
    geometry = qsa_indexer_geometry(config.get("text_config", config))
    assert geometry == {"budget": 2048, "compress_ratio": 4, "head_dim": 128, "n_heads": 4}


def test_serving_settings_carry_the_indexer_geometry(monkeypatch):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    geometry = {"budget": 4096, "compress_ratio": 8, "head_dim": 128, "n_heads": 4}

    class Mixin:
        qsa_indexer = geometry

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=True, adapter_mixin=Mixin)
    try:
        assert engine.status()["settings"]["qsa_indexer"] == geometry
    finally:
        engine.close()


def test_mirrored_dispatch_constants_match_the_runtime(qualify):
    from mlx2 import qualification as Q
    from mlx2.runtime.models import qwen4_moe_window, qwen4_qsa_scores
    from mlx2.runtime.models.qwen4_exp import TextModelArgs

    assert Q.QUALIFIER_NEAR_LIMIT_HEADROOM == qualify.LONG_CONTEXT_HEADROOM
    assert Q.QSA_FUSED_SCORES_HEAD_DIM == qwen4_qsa_scores._HEAD_DIM
    assert Q.QSA_FUSED_SCORES_MAX_MATRIX_ROWS == qwen4_qsa_scores.MAX_MATRIX_ROWS
    assert Q.MOE_TOPK_FOLD_MAX_ROWS == qwen4_moe_window.topk_fold_max_rows()
    defaults = TextModelArgs.__dataclass_fields__
    assert Q.QSA_INDEXER_DEFAULT == {
        "budget": defaults["indexer_budget"].default,
        "compress_ratio": defaults["indexer_compress_ratio"].default,
        "head_dim": defaults["indexer_head_dim"].default,
        "n_heads": defaults["indexer_n_heads"].default,
    }


# NAX segmented MoE gather (Flash-Next default "fused" since 2026-10-02):
# required where selected; host-gated, so a non-M5 host or a refused canary
# is "selected, not observed" -- neither a pass nor a failure.

def _nax(calls=None, fallbacks=None, nax_host=True, mode="fused", verified=None):
    return {"moe_nax_gather": {
        "mode": mode, "nax_host": nax_host,
        "calls": {"gather": 0, "swiglu": 0, "swiglu_map": 0, "swiglu_split": 0,
                  "swiglu_split_map": 0, **(calls or {})},
        "fallbacks": dict(fallbacks or {}), "not_candidates": 0,
        "last_fallback": None, "verified": dict(verified or {}),
    }}


def test_nax_gather_is_required_where_selected_only():
    assert "feature_moe_nax_gather" in required_feature_checks(_settings())
    for mode in ("gather", "fused"):
        env = FlashNextPolicy(moe_nax_gather=mode).environment()
        assert "feature_moe_nax_gather" in required_feature_checks(
            _settings(environment=env))
    off = FlashNextPolicy(moe_nax_gather="off").environment()
    assert "feature_moe_nax_gather" not in required_feature_checks(
        _settings(environment=off))
    # Other models never select it (module default off, no profile pin).
    assert "feature_moe_nax_gather" not in required_feature_checks(
        {"environment": {}, "mtp": False, "max_lanes": 4, "max_context": 8192})


def test_nax_gather_observation_counts_both_fused_halves(qualify):
    from mlx2.qualification import moe_nax_gather_engagement

    idle = _nax()
    ran = _nax(calls={"gather": 48, "swiglu_split_map": 48})
    assert qualify.feature_observations({"execution": ran},
                                        initial={"execution": idle})["moe_nax_gather"] == 48
    assert qualify.feature_observations({"execution": idle},
                                        initial={"execution": idle})["moe_nax_gather"] == 0
    # Only at load (before the initial status): no engagement this run.
    assert moe_nax_gather_engagement(ran, ran) == 0
    # Fused mode with the down gather but no gate/up launch is not engaged.
    assert moe_nax_gather_engagement(_nax(calls={"gather": 48}), idle) == 0
    assert moe_nax_gather_engagement(
        _nax(mode="gather", calls={"gather": 12}), _nax(mode="gather")) == 12
    assert moe_nax_gather_engagement({}, {}) == 0


def test_nax_gather_host_gate_and_canary_are_selected_not_observed():
    from mlx2.qualification import host_gated_not_observed

    idle = _nax()
    assert host_gated_not_observed(_nax(calls={"gather": 4, "swiglu_split_map": 4}),
                                   idle) == {}
    m3 = host_gated_not_observed(_nax(nax_host=False, fallbacks={"not_nax_host": 9}),
                                 _nax(nax_host=False))
    assert "not an M5" in m3["moe_nax_gather"]
    canary = host_gated_not_observed(
        _nax(fallbacks={"kernel_unverified": 3, "swiglu_declined": 3}), idle)
    assert "canary" in canary["moe_nax_gather"]
    assert "kernel_unverified" in canary["moe_nax_gather"]
    failed = host_gated_not_observed(_nax(verified={"gather 4-bit": False}), idle)
    assert "canary" in failed["moe_nax_gather"]
    # On an M5 with a passing canary, no engagement stays a failed check.
    assert host_gated_not_observed(idle, idle) == {}
    assert host_gated_not_observed({}, {}) == {}


def _nax_record(tmp_path, checks, selected_not_observed=None):
    import json

    from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
    from mlx2.qualification import (
        APPROVED_QUALIFICATION_HARNESS,
        REQUIRED_CHECKS,
        load_qualified_route,
    )

    settings = {"mtp": False, "max_context": 32768,
                "environment": {"MLX2_MOE_NAX_GATHER": "fused"}}
    record = {
        "passed": True, "runtime": {"source": "abc"}, "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {**{c: {"passed": True}
                      for c in REQUIRED_CHECKS | {"structured_output"}}, **checks},
    }
    if selected_not_observed is not None:
        record["selected_not_observed"] = selected_not_observed
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    return lambda: load_qualified_route(
        path, runtime=record["runtime"], artifact="weights", settings=settings,
        descriptor=QWEN4_FLASH_NEXT, name="nax")


M3_GATE = {"nax_host": False, "device_name": "Apple M3 Max"}
M5_GATE = {"nax_host": True, "device_name": "Apple M5 Max"}


@pytest.fixture
def serving_host(monkeypatch):
    """Pin the loader's re-evaluated host gate (default: a non-NAX M3)."""
    import mlx2.qualification as Q

    def pin(gate):
        monkeypatch.setitem(Q.HOST_GATE_PROBES, "moe_nax_gather", lambda: dict(gate))

    pin(M3_GATE)
    return pin


def test_loader_accepts_host_gated_not_observed_but_not_a_bare_omission(
    tmp_path, serving_host
):
    entry = {"status": "selected, not observed", "host_gated": True,
             "reason": "host is not an M5 (NAX) Metal device",
             "host_gate": dict(M3_GATE)}
    assert _nax_record(tmp_path, {"feature_moe_nax_gather": {"passed": True}})().profile
    assert _nax_record(tmp_path, {}, {"feature_moe_nax_gather": entry})().profile
    with pytest.raises(ValueError, match="moe_nax_gather"):
        _nax_record(tmp_path, {})()
    for bad in ({**entry, "host_gated": False}, {**entry, "reason": ""},
                {**entry, "status": "passed"}):
        with pytest.raises(ValueError, match="moe_nax_gather"):
            _nax_record(tmp_path, {}, {"feature_moe_nax_gather": bad})()
    # A recorded failure is never excused by a not-observed entry.
    with pytest.raises(ValueError):
        _nax_record(tmp_path, {"feature_moe_nax_gather": {"passed": False}},
                    {"feature_moe_nax_gather": entry})()


# Codex port review 2026-10-02 item 2: the waiver is bound to the host gate it
# was recorded under and re-evaluated on the serving host.

def test_host_gate_waiver_is_refused_where_the_serving_host_admits_nax(
    tmp_path, serving_host
):
    m3_entry = {"status": "selected, not observed", "host_gated": True,
                "reason": "host is not an M5 (NAX) Metal device",
                "host_gate": dict(M3_GATE)}
    load = _nax_record(tmp_path, {}, {"feature_moe_nax_gather": m3_entry})
    assert load().profile  # served on another non-NAX host: stock path, as recorded
    serving_host(M5_GATE)
    with pytest.raises(ValueError, match="moe_nax_gather"):
        load()  # an M3 receipt never exercised the kernel an M5 will run
    # An entry without its host gate (the pre-fix receipt shape) is refused.
    bare = {k: v for k, v in m3_entry.items() if k != "host_gate"}
    serving_host(M3_GATE)
    with pytest.raises(ValueError, match="moe_nax_gather"):
        _nax_record(tmp_path, {}, {"feature_moe_nax_gather": bare})()
    for gate in ({"nax_host": "no", "device_name": "x"}, {"nax_host": False},
                 {"nax_host": False, "device_name": ""}):
        with pytest.raises(ValueError, match="moe_nax_gather"):
            _nax_record(tmp_path, {}, {"feature_moe_nax_gather": {
                **m3_entry, "host_gate": gate}})()


def test_canary_waiver_binds_the_device_it_was_refused_on(tmp_path, serving_host):
    canary = {"status": "selected, not observed", "host_gated": True,
              "reason": "bitwise canary refused the NAX kernel (kernel_unverified)",
              "host_gate": dict(M5_GATE)}
    load = _nax_record(tmp_path, {}, {"feature_moe_nax_gather": canary})
    serving_host(M5_GATE)
    assert load().profile
    serving_host({"nax_host": True, "device_name": "Apple M5 Pro"})
    with pytest.raises(ValueError, match="moe_nax_gather"):
        load()
    serving_host(M3_GATE)  # no NAX here: the stock path runs, as the receipt saw
    assert load().profile


def test_host_gate_record_comes_from_the_served_status():
    from mlx2.qualification import host_gate_record

    status = _nax(nax_host=False)
    status["moe_nax_gather"]["device_name"] = "Apple M3 Max"
    assert host_gate_record("moe_nax_gather", status) == M3_GATE
    assert host_gate_record("moe_nax_gather", {}) is None


def test_qualify_serving_binds_the_host_gate(qualify):
    import inspect

    source = inspect.getsource(qualify.main)
    assert '"host_gate": host_gate_record(' in source


def test_status_reports_the_device_the_gate_was_evaluated_on(monkeypatch):
    from mlx2.runtime.models import moe_nax_gather as nax

    monkeypatch.setattr(nax, "_nax_host", None)
    monkeypatch.setattr(nax, "_device_name", None)
    gate = nax.host_gate()
    assert set(gate) == {"nax_host", "device_name"}
    assert isinstance(gate["nax_host"], bool) and isinstance(gate["device_name"], str)
    assert nax.status()["device_name"] == gate["device_name"]


# Codex port review 2026-10-02 item 3: the invariant prefill lane suppresses
# the NAX gather, so a route with both selected cannot require its engagement.

def _invariant_settings(**policy):
    from mlx2.runtime.prefill_plan import execution_identity

    pol = FlashNextPolicy(invariant_prefill=True, **policy)
    return _settings(
        environment=pol.environment(),
        prefill_execution=execution_identity(
            invariant={"schema": "invariant-prefill-v1", "law": {"x": 1}}
        ),
    )


def test_invariant_lane_suppresses_the_nax_requirement():
    combined = _invariant_settings()
    assert FlashNextPolicy(invariant_prefill=True).moe_nax_gather == "fused"
    assert "feature_moe_nax_gather" not in required_feature_checks(combined)
    reason = selected_not_observed_features(combined)["feature_moe_nax_gather"]
    assert reason.startswith("selected, suppressed by invariant_prefill")
    # Every other default-on requirement is unchanged by the lane.
    assert required_feature_checks(combined) == (
        required_feature_checks(_settings()) - {"feature_moe_nax_gather"}
    )
    # NAX off under the lane: nothing selected, nothing recorded.
    off = _invariant_settings(moe_nax_gather="off")
    assert "feature_moe_nax_gather" not in selected_not_observed_features(off)
    # Without the lane NAX stays required.
    assert "feature_moe_nax_gather" in required_feature_checks(_settings())


def test_loader_qualifies_the_combined_route_without_nax_engagement(tmp_path):
    import json

    from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
    from mlx2.qualification import (
        APPROVED_QUALIFICATION_HARNESS,
        REQUIRED_CHECKS,
        load_qualified_route,
    )
    from mlx2.runtime.prefill_plan import execution_identity

    settings = {"mtp": False, "max_context": 32768,
                "environment": {"MLX2_MOE_NAX_GATHER": "fused"},
                "prefill_execution": execution_identity(
                    invariant={"schema": "invariant-prefill-v1", "law": {"x": 1}})}
    record = {
        "passed": True, "runtime": {"source": "abc"}, "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {c: {"passed": True} for c in REQUIRED_CHECKS | {"structured_output"}},
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    assert load_qualified_route(
        path, runtime=record["runtime"], artifact="weights", settings=settings,
        descriptor=QWEN4_FLASH_NEXT, name="nax+invariant").profile
