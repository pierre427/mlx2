"""Config/qualification lane fixes from the 2026-10-06 sweep (CQ-*)."""

import pytest


# CQ-1 / G2-03 ------------------------------------------------------------
def test_nemotron_does_not_inherit_flash_next_handoff_width():
    # The handoff width is a per-model measurement.  Both Nemotron adapters
    # subclass FlashNextAdapter and have none of their own.
    from mlx2.adapters import nemotron3_super as ns
    from mlx2.adapters import nemotron35_lightning as nl
    from mlx2.adapters.registry import AdapterResolution

    for adapter, descriptor in (
        (ns.Nemotron3SuperAdapter, ns.DESCRIPTOR),
        (nl.Nemotron35LightningAdapter, nl.descriptor_for(has_mtp=True)),
    ):
        resolution = AdapterResolution(adapter, descriptor, {})
        assert resolution.default_execution_policy("native_mtp") == {}
        assert resolution.default_mtp_ordinary_handoff is None, adapter.__name__


# CQ-8 -------------------------------------------------------------------
def _flash_next_resolution():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.registry import AdapterResolution

    return AdapterResolution(FlashNextAdapter, FlashNextAdapter.descriptor, {})


@pytest.mark.parametrize("max_lanes", [1, 3])
def test_handoff_default_steps_aside_when_lanes_cannot_exceed_width(max_lanes):
    from mlx2.qualification import required_feature_checks
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    skipped = {}
    resolved = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"),
        _flash_next_resolution(), max_lanes=max_lanes, skipped=skipped,
    ) or {}
    assert "mtp_ordinary_handoff" not in resolved
    assert "prefill_scheduling" not in resolved
    assert "max_mtp_width 3" in skipped["mtp_ordinary_handoff"]
    assert "max_bypass" in skipped["prefill_scheduling"]
    settings = {"environment": {}, "mtp": True, "speculation": "self_mtp",
                "max_lanes": max_lanes,
                "mtp_ordinary_handoff": resolved.get("mtp_ordinary_handoff") or {}}
    assert "feature_mtp_ordinary_handoff" not in required_feature_checks(settings)


def test_handoff_default_still_selected_above_its_width():
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    skipped = {}
    resolved = resolve_execution_policy_defaults(
        None, RouteSelection("native_mtp", "adapter_default"),
        _flash_next_resolution(), max_lanes=4, skipped=skipped,
    )
    assert resolved["mtp_ordinary_handoff"] == {"enabled": True, "max_mtp_width": 3}
    assert skipped == {}


def test_explicit_unreachable_handoff_is_selected_not_observed():
    # An operator may still select it; the qualifier records it as selected,
    # not observed rather than demanding evidence no run can produce.
    from mlx2.qualification import required_feature_checks, selected_not_observed_features

    settings = {"environment": {}, "mtp": True, "speculation": "self_mtp", "max_lanes": 2,
                "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 4}}
    assert "feature_mtp_ordinary_handoff" not in required_feature_checks(settings)
    assert "feature_mtp_ordinary_handoff" in selected_not_observed_features(settings)
    settings["max_lanes"] = 5
    assert "feature_mtp_ordinary_handoff" in required_feature_checks(settings)


def test_skipped_route_defaults_reach_settings_as_provenance(monkeypatch):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    from mlx2.qualification import PROVENANCE_ONLY_SETTINGS
    from mlx2.server import build_parser, serving_engine_kwargs

    args = build_parser().parse_args(["--model", "x"])
    args.skipped_route_defaults = {"mtp_ordinary_handoff": "why"}
    kwargs = serving_engine_kwargs(
        args, None, native_mtp=True, approximate_kv=None, max_request_bytes=1 << 20)
    assert kwargs["skipped_route_defaults"] == {"mtp_ordinary_handoff": "why"}
    assert "skipped_route_defaults" in PROVENANCE_ONLY_SETTINGS

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=True,
                         skipped_route_defaults={"mtp_ordinary_handoff": "why"})
    try:
        assert engine.status()["settings"]["skipped_route_defaults"] == {
            "mtp_ordinary_handoff": "why"}
    finally:
        engine.close()


# CQ-7 -------------------------------------------------------------------
def test_cli_help_matches_adapter_defaults_and_enforcement():
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.server import build_parser

    actions = {a.dest: a for a in build_parser()._actions}
    default = NorthMiniCodeAdapter.THINKING_GUARD_DEFAULTS["thinking_steer_alpha"]
    assert f"North-Mini-Code: {default}" in actions["thinking_steer_alpha"].help
    # Adaptive depth is exact and runs unqualified (serving.py); the help
    # must not claim a receipt is required.
    assert "requires a matching qualification receipt" not in actions["adaptive_mtp_depth"].help


# CQ-9 -------------------------------------------------------------------
def _media_adapters():
    from mlx2.adapters.gemma4 import Gemma431BAdapter, Gemma4A4BAdapter
    from mlx2.adapters.lfm25_vl import LFM25VLAdapter
    from mlx2.adapters.mlx_vlm import Gemma3nAdapter, MiniCPMOAdapter
    from mlx2.adapters.qwen25_vl import Qwen25VLCandidateAdapter
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    return (Gemma431BAdapter, Gemma4A4BAdapter, Gemma3nAdapter, MiniCPMOAdapter,
            LFM25VLAdapter, Qwen25VLCandidateAdapter, SmolVLM2CandidateAdapter)


def test_media_adapters_refuse_an_explicit_tf32_like_the_standard_decoder(
    monkeypatch, tmp_path
):
    # MLX latches TF32 at the first fp32 dispatch; a receipt that records no
    # TF32 value must not describe a TF32 process.  Refused before any
    # artifact is read, so no weights are needed.
    from mlx2.process_env import ProcessNumericsConflict

    import mlx2.adapters.gemma4 as gemma4

    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    for adapter in _media_adapters():
        # Gemma 4 reads its config for the variant before the shared mlx-vlm
        # constructor; stand that in so the refusal is what the test reaches.
        monkeypatch.setattr(gemma4, "inspect_gemma4_artifact",
                            lambda _path, a=adapter: {"variant": a.descriptor.variant})
        with pytest.raises(ProcessNumericsConflict, match="MLX_ENABLE_TF32"):
            adapter(str(tmp_path))


# CQ-3 -------------------------------------------------------------------
@pytest.mark.parametrize("explicit", [
    {"external_varlen_prefill": True},
    {"varlen_dense_mlp": {"enabled": True}},
    {"external_varlen_prefill": True, "varlen_dense_mlp": True},
])
def test_explicit_external_varlen_is_not_contradicted_by_adapter_defaults(explicit):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, descriptor_for
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    resolution = AdapterResolution(Qwen3827BAdapter, descriptor_for(has_mtp=True), {})
    operator = {"draft_model": "/x", **Qwen3827BAdapter.default_external_route_binding,
                **explicit}
    resolved = resolve_execution_policy_defaults(
        operator, RouteSelection("external_draft", "explicit_flag"), resolution)
    # The varlen pair is the operator's: no adapter value fills either key.
    for key in ("external_varlen_prefill", "varlen_dense_mlp"):
        assert resolved.get(key) == explicit.get(key), (key, resolved)
    # The rest of the artifact-bound tree defaults still apply.
    assert resolved["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"


# CQ-2 -------------------------------------------------------------------
def _tensorfold_qmv_engine(tmp_path, lane_matmul):
    import json
    import time

    import route_harness

    from mlx2.serving import ServingEngine

    (tmp_path / "config.json").write_text(json.dumps({"num_experts": 8}))
    model, vocab = route_harness.tiny_qwen38_mtp()

    class TensorfoldQmvMixin:
        tensorfold_qmv = {"installed": 1}

    engine = ServingEngine(
        str(tmp_path),
        adapter_factory=route_harness.make_adapter(
            model, vocab, adapter_mixin=TensorfoldQmvMixin),
        qualification_mode=True, mtp=False, max_lanes=1, max_inflight=4,
        prefill_step=16, lane_matmul=lane_matmul,
    )
    for _ in range(1200):
        if engine.error or engine.ready.is_set():
            break
        time.sleep(0.1)
    return engine


def test_lane_auto_steps_aside_for_an_explicit_tensorfold_qmv(tmp_path):
    # --lane-matmul auto is the CLI default; an explicitly selected mechanism
    # that owns the projections must not be refused by it.
    engine = _tensorfold_qmv_engine(tmp_path, "auto")
    try:
        assert not engine.error, engine.error
        status = engine.status()
        assert status["settings"]["lane_matmul_stepped_aside"] == "auto->off (tensorfold_qmv)"
        assert status["lane_matmul"]["installed"] is False
    finally:
        engine.close()


def test_explicit_lane_mode_still_refuses_tensorfold_qmv(tmp_path):
    engine = _tensorfold_qmv_engine(tmp_path, "crossover")
    try:
        assert "cannot own the same projections" in str(engine.error)
    finally:
        engine.close()


# CQ-4 -------------------------------------------------------------------
def _flash_next_settings(**extra):
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    env = {
        "MLX_QWEN4_FUSED_GDN_PREFILL": "1", "MLX_QWEN4_MOE_WEIGHTED_SUM": "1",
        **FlashNextPolicy().environment(),
    }
    return {"environment": env, "mtp": True, "speculation": "self_mtp", "max_lanes": 16,
            "max_context": 262144, "execution_policy": {"num_draft": 2},
            "decode_time_fairness": {"enabled": True, "fair_share": 0.5,
                                     "stall_target_ms": 500.0}, **extra}


def test_default_on_mechanisms_require_engagement():
    from mlx2.qualification import required_feature_checks

    features = required_feature_checks(_flash_next_settings())
    for name in ("fused_gdn_prefill", "moe_weighted_sum", "decode_fairness"):
        assert "feature_" + name in features, name
    lane = {"environment": {}, "mtp": False, "speculation": "ordinary", "max_lanes": 16,
            "lane_matmul": {"mode": "crossover", "law_id": "lane-matmul-v1",
                            "covered": {"q4": 256}}}
    assert "feature_lane_matmul" in required_feature_checks(lane)
    assert "feature_lane_matmul" not in required_feature_checks(
        {**lane, "lane_matmul": {"mode": "off", "covered": {}}})


def test_base_fairness_is_contention_gated_and_needs_lanes():
    from mlx2.qualification import (
        CONTENTION_GATED_FEATURES,
        contention_gated_not_observed,
        selected_not_observed_features,
    )

    one_lane = _flash_next_settings(max_lanes=1)
    assert "feature_decode_fairness" in selected_not_observed_features(one_lane)
    assert "decode_fairness" in CONTENTION_GATED_FEATURES
    initial = {"scheduler": {"decode_fairness_prefill_chunks": 4}}
    assert "decode_fairness" in contention_gated_not_observed(initial, initial)
    final = {"scheduler": {"decode_fairness_prefill_chunks": 6}}
    assert "decode_fairness" not in contention_gated_not_observed(final, initial)


def _load_qualifier():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_serving.py"
    spec = importlib.util.spec_from_file_location("qualify_serving_cq4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_observations_are_run_deltas():
    observe = _load_qualifier().feature_observations
    initial = {
        "execution": {"fused_gdn": {"prefill_calls": 5, "verify_calls": 9,
                                    "replay_rollback_calls": 2},
                      "moe": {"weighted_sum": {"calls": 40}},
                      "round_levers": {"eager_async_evals": 3}},
        "lane_matmul": {"counts": {"lane_calls": 100}},
        "scheduler": {"decode_fairness_prefill_chunks": 1, "self_mtp_copy_rounds": 7,
                      "pld_retrieval_cycles": 4},
    }
    observed = observe(initial, initial=initial)
    # A counter that only moved before the run (load, an earlier run) is not
    # engagement during this run.
    for name in ("fused_gdn_prefill", "moe_weighted_sum", "lane_matmul", "decode_fairness",
                 "self_mtp_copy_draft", "eager_dispatch", "fused_gdn_verify",
                 "fused_gdn_replay_rollback", "prompt_lookup"):
        assert observed[name] == 0, name
    final = {
        "execution": {"fused_gdn": {"prefill_calls": 8, "verify_calls": 9,
                                    "replay_rollback_calls": 2},
                      "moe": {"weighted_sum": {"calls": 52}},
                      "round_levers": {"eager_async_evals": 3}},
        "lane_matmul": {"counts": {"lane_calls": 130}},
        "scheduler": {"decode_fairness_prefill_chunks": 3, "self_mtp_copy_rounds": 9,
                      "pld_retrieval_cycles": 4},
    }
    observed = observe(final, initial=initial)
    assert observed["fused_gdn_prefill"] == 3
    assert observed["moe_weighted_sum"] == 12
    assert observed["lane_matmul"] == 30
    assert observed["decode_fairness"] == 2
    assert observed["self_mtp_copy_draft"] == 2


def test_flash_next_diagnostics_export_moe_weighted_sum():
    import inspect
    from types import SimpleNamespace

    from mlx2.adapters import flash_next

    assert "moe_weighted_sum_status(moe_modules)" in inspect.getsource(
        flash_next.FlashNextAdapter.diagnostics)
    switch = SimpleNamespace(moe_weighted_sum=True, moe_weighted_sum_calls=6,
                             moe_weighted_sum_fallbacks=1,
                             moe_weighted_sum_last_fallback="Metal runtime unavailable")
    off = SimpleNamespace(switch_mlp=SimpleNamespace(moe_weighted_sum=False))
    block = SimpleNamespace(switch_mlp=switch)
    assert flash_next.moe_weighted_sum_status([off]) is None
    assert flash_next.moe_weighted_sum_status([block, block, off]) == {
        "layers": 2, "calls": 12, "fallbacks": 2,
        "last_fallback": "Metal runtime unavailable"}


# CQ-12 ------------------------------------------------------------------
def test_lane_matmul_and_tensorfold_tree_have_one_projection_owner():
    # Under --lane-matmul auto (dense M5) the mlx2 lane installer owns the
    # 27B projections, and the default external route runs the vendored
    # TensorFold tree.  That executor carries its own lane kernels (lane_qmm
    # patches QuantizedLinear.__call__ and tiles weights; lane_fuse restacks
    # gate/up, k/v, in_proj_z/b/a), but mlx2 never installs them: they stay
    # disabled, lane_fuse never builds or replaces member weights, and every
    # tree projection is an ordinary module call (the mlx2 lane wrapper when
    # covered).  One owner, so the two defaults compose.
    import re
    from pathlib import Path
    from types import SimpleNamespace

    import mlx.core as mx

    from mlx2.runtime.tensorfold_qwen38 import lane_attention, lane_fuse, lane_qmm

    assert (lane_qmm.enabled, lane_fuse.enabled, lane_attention.enabled) == (
        False, False, False)
    src = Path(__file__).resolve().parents[1] / "src" / "mlx2"
    vendored = src / "runtime" / "tensorfold_qwen38"
    flips = re.compile(
        r"lane_qmm\.install|lane_attention\.install|lane_fuse\.build\(|"
        r"lane_(?:qmm|fuse|attention)\.enabled\s*=")
    offenders = [
        str(path.relative_to(src)) for path in src.rglob("*.py")
        if vendored not in path.parents and flips.search(path.read_text())
    ]
    assert offenders == []
    weight = mx.zeros((8, 8), dtype=mx.uint32)
    parent = SimpleNamespace(k_proj={"weight": weight}, v_proj={"weight": weight})
    assert lane_fuse.attn_kv(parent, mx.zeros((1, 15, 64), dtype=mx.bfloat16)) is None
    assert "_lane_fuse_groups" not in vars(parent)


# CQ-6 -------------------------------------------------------------------
def test_engine_and_cli_lane_defaults_agree():
    # Scripts that build ServingEngine directly must measure the law the
    # server serves (--lane-matmul auto).
    import inspect

    from mlx2.server import build_parser
    from mlx2.serving import ServingEngine

    engine_default = inspect.signature(ServingEngine.__init__).parameters["lane_matmul"].default
    assert engine_default == build_parser().parse_args(["--model", "x"]).lane_matmul == "auto"


def test_lane_auto_without_a_backend_does_not_inspect_the_model(monkeypatch):
    # Off the GPU auto installs nothing, so an adapter without a tensor model
    # (or one the detector cannot walk) still starts.
    import mlx2.runtime.lane as lane
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    assert lane.available() is False
    model, vocab = tiny_qwen38_mtp()
    import mlx2.runtime.lane.policy as policy

    monkeypatch.setattr(policy, "detect", lambda *a, **k: pytest.fail("auto inspected the model"))
    engine = make_engine(model, vocab, mtp=False)
    try:
        status = engine.status()["lane_matmul"]
        assert status["requested"] == "auto" and status["installed"] is False
    finally:
        engine.close()


# CQ-11 ------------------------------------------------------------------
def test_default_qwen38_tree_route_can_qualify_with_observed_tree_rounds():
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.qualification import required_feature_checks, unqualifiable_candidate

    default = Qwen3827BAdapter.default_route_execution_policy["external_draft"]
    settings = {"environment": {}, "mtp": False, "speculation": "external_draft",
                "max_lanes": 4, "execution_policy": {
                    "batch_size_route": default["batch_size_route"],
                    "pairwise_selection": default["pairwise_selection"]}}
    assert unqualifiable_candidate(settings) is None
    assert "feature_external_tree" in required_feature_checks(settings)

    observe = _load_qualifier().feature_observations
    initial = {"scheduler": {"external_tree_rounds": 3, "external_tensorfold_target_rounds": 3}}
    assert observe(initial, initial=initial)["external_tree"] == 0
    final = {"scheduler": {"external_tree_rounds": 9, "external_tensorfold_target_rounds": 7}}
    assert observe(final, initial=initial)["external_tree"] == 4
    # A chain-only external route does not require it.
    chain = {**settings, "execution_policy": {"pairwise_selection": "host"}}
    assert "feature_external_tree" not in required_feature_checks(chain)


# CQ-5 -------------------------------------------------------------------
@pytest.mark.parametrize("module_name, class_name, descriptor_name", [
    ("lfm25_vl", "LFM25VLAdapter", "LFM25_VL"),
    ("smolvlm2", "SmolVLM2CandidateAdapter", "DESCRIPTOR"),
    ("qwen25_vl", "Qwen25VLCandidateAdapter", "DESCRIPTOR"),
])
def test_unqualified_media_adapters_still_resolve(monkeypatch, module_name, class_name,
                                                  descriptor_name):
    # AGENTS.md: an unqualified model may still be loaded and served,
    # labelled unqualified; it is not refused.
    import importlib

    from mlx2.adapters import registry

    module = importlib.import_module(f"mlx2.adapters.{module_name}")
    adapter = getattr(module, class_name)
    monkeypatch.setattr(
        registry, "inspect_model",
        lambda _path: registry.AdapterResolution(adapter, getattr(module, descriptor_name), {}))
    assert registry.resolve_adapter("/unused") is adapter


# CQ-13 ------------------------------------------------------------------
def test_behaviour_switches_reach_settings_and_bit_changers_the_apc_key(monkeypatch):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    from mlx2.runtime.apc_numerics import execution_numerics_identity
    from mlx2.runtime.env_switches import serving_env_switches

    assert serving_env_switches({"MLX2_STRUCTURED_WORKERS": "0"}) == {}
    assert serving_env_switches({"MLX2_DECODE_MASK": "none"}) == {"MLX2_DECODE_MASK": "none"}
    # Bit-changing switches enter the APCv2 numerics identity; the default
    # (absent, or the explicit default) does not.
    for env in ({"MLX2_MIXED_PREFILL_DECODE": "1"}, {"MLX2_NORTH_NORM": "legacy_layernorm"},
                {"MLX2_XING_MHC_KERNEL": "0"}):
        assert execution_numerics_identity(env) is not None, env
    assert execution_numerics_identity({"MLX2_XING_MHC_KERNEL": "1"}) is None
    assert execution_numerics_identity({"MLX2_MIXED_PREFILL_DECODE": "0"}) is None

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False)
    try:
        default_settings = engine.status()["settings"]
    finally:
        engine.close()
    assert "process_env" not in default_settings
    monkeypatch.setenv("MLX2_GREEDY_BATCH_SAMPLER", "0")
    engine = make_engine(model, vocab, mtp=False)
    try:
        assert engine.status()["settings"]["process_env"] == {"MLX2_GREEDY_BATCH_SAMPLER": "0"}
    finally:
        engine.close()
