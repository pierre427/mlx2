"""Host-only contracts for the diagnostic, never loads model weights."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts/measure_flashnext_peak_memory.py"
    spec = importlib.util.spec_from_file_location("memory_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_precedes_loading(probe, monkeypatch):
    monkeypatch.setattr(probe, "served_config", lambda *_: pytest.fail("read/load before guard"))
    with pytest.raises(SystemExit) as error:
        probe.main([])
    assert error.value.code == 2


def test_served_resolved_defaults_override_current_defaults(probe, tmp_path):
    initial = {"execution": {"policy": {"num_draft": 2, "hc_decode_kernels": True}},
               "settings": {"host_memory_signals": {"enabled": True},
                            "self_mtp_copy_draft": {"enabled": True, "max_span": 7},
                            "prefill_scheduling": {"order": "srpt"},
                            "apc_interior_checkpoints": {"count": 4}}}
    ladder = tmp_path / "ladder.json"
    policy = tmp_path / "policy.json"
    ladder.write_text(json.dumps({"initial": initial}))
    policy.write_text(json.dumps({"num_draft": 2, "adaptive_mtp_depth": False}))
    observed, resolved = probe.served_config(ladder, policy)
    assert observed == initial
    assert resolved["qsa_fused_scores"] is False
    assert resolved["fused_gdn_batch_verify"] == "off"
    assert resolved["apc_interior_checkpoints"] == {"count": 4}
    assert resolved["self_mtp_copy_draft"]["max_span"] == 7


def test_cleanup_drops_derived_buffers_preserves_source_weights(probe):
    weight = object()
    attention = SimpleNamespace(weight=weight, _qsa_fused_cache=(weight, object()))
    hc_module = SimpleNamespace(weight=weight, _hc_decode_plan=(weight, object()))
    expert = SimpleNamespace(weight=weight, _mlx2_expert_views=(weight, object()))
    ple = SimpleNamespace(weight=weight, _ple_compile_cache={"graph": object()})
    modules = [attention, hc_module, expert, ple]
    model = SimpleNamespace(named_modules=lambda: enumerate(modules))
    hc = SimpleNamespace(_COMPILED={"graph": object()})
    counts = probe.clear_derived(model, hc)
    assert counts == {"_qsa_fused_cache": 1, "_hc_decode_plan": 1,
                      "_mlx2_expert_views": 1, "_ple_compile_cache": 1}
    assert attention._qsa_fused_cache is None
    assert not hc._COMPILED
    assert all(module.weight is weight for module in modules)
    assert not hasattr(expert, "_mlx2_expert_views")


def test_request_calibration_and_prefix_identity(probe):
    # Character tokenizer tests exact sizing and distinct prefix headers.
    tokenizer = SimpleNamespace(encode=lambda text, **_: list(text))
    engine = SimpleNamespace(adapter=SimpleNamespace(tokenizer=tokenizer))
    reqs = probe.requests(engine, 160, 4, 2, 128)
    assert all(len(r["prompt"]) == 160 for r in reqs)
    assert len({r["prompt"] for r in reqs}) == 4
    assert reqs == probe.requests(engine, 160, 4, 2, 128)
    assert reqs != probe.requests(engine, 160, 4, 3, 128)


def test_unknown_arms_and_widths_fail_closed(probe):
    for args in (["--dry-run", "--arms", "baseline,typo"],
                 ["--dry-run", "--widths", "16"],
                 ["--dry-run", "--cycles", "0"]):
        with pytest.raises(SystemExit):
            probe.parse_args(args)


def test_default_served_config_reads_committed_run_without_git(probe, monkeypatch):
    # Snapshot clones (QUALIFICATION.md Step 1) carry only origin/qualify/*;
    # the run directory committed on main must be enough on its own.
    def no_git(args, **_):
        pytest.fail(f"consulted repository refs: {args}")
    monkeypatch.setattr(probe.subprocess, "check_output", no_git)
    initial, policy = probe.served_config()
    assert initial["settings"]["prefill_step"] == 8192
    assert policy["qsa_fused_scores"] is False


def test_default_served_config_falls_back_to_remote_ref(probe, monkeypatch, tmp_path):
    shown = []
    def git_show(args, **_):
        shown.append(args[-1])
        if not args[-1].startswith("origin/"):
            raise probe.subprocess.CalledProcessError(128, args)
        name = args[-1].rsplit("/", 1)[-1]
        return json.dumps({"initial": {"execution": {"policy": {}}, "settings": {}}}
                          if name == "ladder-short.json" else {"num_draft": 2})
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe.subprocess, "check_output", git_show)
    _, policy = probe.served_config()
    assert policy["num_draft"] == 2
    assert shown[:2] == [f"{probe.REF}:{probe.RUN}/results/ladder-flash-next-uncensored-mtp2/ladder-short.json",
                         f"origin/{probe.REF}:{probe.RUN}/results/ladder-flash-next-uncensored-mtp2/ladder-short.json"]


def test_resolved_policy_fields_validate_without_model_load(probe):
    from mlx2.adapters.flash_next_policy import FlashNextPolicy
    initial, policy = probe.served_config()
    adapter_keys = set(FlashNextPolicy.__dataclass_fields__)
    FlashNextPolicy.from_mapping({k: v for k, v in policy.items() if k in adapter_keys})
    assert initial["settings"]["prefill_step"] == 8192


def test_runtime_constructor_validates_without_model_load(probe):
    from mlx2.serving import ServingEngine
    _, policy = probe.served_config()
    ServingEngine.validate_arguments(
        "/nonexistent/model", mtp=True, execution_policy=policy,
        max_lanes=4, cache_bytes=16 * probe.GIB, max_context=65536,
    )


def test_live_switch_restores_after_exception(probe, monkeypatch):
    from mlx2 import memory
    import sys
    import mlx2.runtime.models as models_pkg
    # Host doubles exercise scope/restoration without constructing Metal kernels.
    def toggle(getter, setter, field="_ENABLED"):
        module = SimpleNamespace(**{field: True})
        setattr(module, getter, lambda: getattr(module, field))
        setattr(module, setter, lambda value: setattr(module, field, value))
        return module
    hc = toggle("hc_decode_enabled", "set_hc_decode_enabled")
    attn = toggle("enabled", "set_enabled")
    qsa = toggle("enabled", "set_enabled")
    routed = toggle("unused_getter", "set_expert_views", "_VIEWS")
    for name, module in (("qwen4_hc_decode", hc), ("qwen4_attn_rows", attn),
                         ("qwen4_qsa_scores", qsa), ("qwen4_routed_decode", routed),
                         ("qwen4_exp", SimpleNamespace())):
        monkeypatch.setitem(sys.modules, "mlx2.runtime.models." + name, module)
        # `from mlx2.runtime.models import X` returns the package attribute once
        # an earlier test imported the real module; patch it too.
        monkeypatch.setattr(models_pkg, name, module, raising=False)
    model = SimpleNamespace(named_modules=lambda: [])
    original = memory.host_term_reserve_credit_bytes
    with ExitStack() as stack:
        probe.arm_switches(stack, "q1_credit_off", model, None)
        assert memory.host_term_reserve_credit_bytes(128 << 30, 112 << 30) == 0
    assert memory.host_term_reserve_credit_bytes is original
    for arm, get_value in (("hc_off", hc.hc_decode_enabled),
                           ("attn_rows_off", attn.enabled),
                           ("qsa_scores_off", qsa.enabled),
                           ("expert_views_off", lambda: routed._VIEWS)):
        before = get_value()
        with pytest.raises(RuntimeError), ExitStack() as stack:
            probe.arm_switches(stack, arm, model, None)
            assert get_value() is False
            raise RuntimeError("simulated arm failure")
        assert get_value() == before


def test_summary_separates_peak_pool_and_failed_arms(probe, tmp_path):
    def end(arm, peak, active, pool, kind="arm_end"):
        state = {"memory": {"metal_peak_bytes": peak * probe.GIB,
                            "metal_active_bytes": active * probe.GIB,
                            "metal_buffer_cache_bytes": pool * probe.GIB},
                 "apcv2": {"idle_disk": {"resident_bytes": probe.GIB}}}
        after = {**state, "memory": {**state["memory"], "metal_buffer_cache_bytes": 0}}
        return {"kind": kind, "round": 0, "width": 4, "arm": arm,
                "before_clear": state, "after_clear": after}
    records = [end("baseline", 89, 80, 3), end("q1_credit_off", 79, 76, 2),
               end("hc_off", 90, 80, 3, "arm_failed_end"),
               {"kind": "arm_error", "arm": "hc_off", "error": "429"}]
    for arm in ("baseline", "q1_credit_off"):
        records.append({"kind": "sample", "round": 0, "width": 4, "cycle": 0,
                        "phase": "cold", "arm": arm, "outputs": [{"sha256": "same"}]})
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    result = probe.summarize(path)
    assert len(result["arms"]) == 2
    q1 = next(r for r in result["arms"] if r["arm"] == "q1_credit_off")
    assert q1["peak_delta_gib_vs_baseline"] == -10
    assert q1["before_clear"]["metal_buffer_cache_gib"] == 2
    assert q1["after_clear"]["metal_buffer_cache_gib"] == 0
    assert q1["output_matches"] == q1["output_comparisons"] == 1
    assert result["errors"][0]["error"] == "429"
