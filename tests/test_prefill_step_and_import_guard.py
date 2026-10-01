"""Adapter-owned prefill chunk default and the import-order profile guard."""

import sys
import types

import pytest


def _engine_settings(monkeypatch, adapter_step, *, prefill_identity=None, expect_error=None, **overrides):
    from types import SimpleNamespace as NS

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    class APC:
        def __init__(self, **_kw):
            self.apc_stats = {}

        @staticmethod
        def key(*_a, **kw):
            seen.setdefault("cache_semantics", []).append(kw["semantic_fingerprint"])
            return "key"

        def spill_idle_entries(self):
            pass

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_a, **_kw):
            pass

        def next(self):
            return [], []

        def close(self):
            pass

    seen = {}

    class Adapter:
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=10, eos_token_ids=[])

        def __init__(self, _path):
            self.prefill_execution_identity = prefill_identity

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, *, max_lanes, prefill_step):
            seen["prefill_step"] = prefill_step
            return {"num_draft": 0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    if adapter_step == "decline":
        Adapter.prefill_step_default = lambda self: None
    elif adapter_step is not None:
        Adapter.prefill_step_default = lambda self: adapter_step
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine("fake", adapter_factory=Adapter, qualification_mode=expect_error is None, mtp=False,
                                   max_lanes=2, max_inflight=2, **overrides)
    try:
        if expect_error is not None:
            engine.thread.join(5)
            assert not engine.ready.is_set()
            assert expect_error in engine.error
            return {}, seen, engine
        assert engine.ready.wait(5), engine.error
        settings = engine.status()["settings"]
    finally:
        engine.close()
    return settings, seen, engine


@pytest.mark.parametrize("adapter_step,overrides,step,source", [
    (None, {}, 2048, "default"),
    (8192, {}, 8192, "adapter"),
    (8192, {"prefill_step": 512}, 512, "engine_argument"),
])
def test_prefill_step_prefers_operator_then_adapter(monkeypatch, adapter_step, overrides, step, source):
    settings, seen, engine = _engine_settings(monkeypatch, adapter_step, **overrides)
    assert engine.prefill_step == step and seen["prefill_step"] == step
    assert settings["prefill_step"] == step and settings["prefill_step_source"] == source


def test_flash_next_policy_owns_the_prefill_step():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    assert FlashNextPolicy().prefill_step == 8192
    assert FlashNextPolicy.from_mapping({"prefill_step": 2048}).prefill_step == 2048
    with pytest.raises(ValueError):
        FlashNextPolicy(prefill_step=0)


def test_late_profile_fails_closed(monkeypatch):
    from mlx2.runtime.models import import_env

    name = "mlx2_test_fake_tensor_module"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(import_env, "_SNAPSHOTS", {})
    import_env.snapshot(name, {"MLX_QWEN4_FUSED_GDN_DECODE": "0", "PATH": "x"})
    import_env.assert_profile_applied("test", {"MLX_QWEN4_FUSED_GDN_DECODE": "0", "HOME": "y"})
    with pytest.raises(import_env.ImportOrderError, match="MLX_QWEN4_FUSED_GDN_DECODE"):
        import_env.assert_profile_applied("test", {"MLX_QWEN4_FUSED_GDN_DECODE": "1"})
    with pytest.raises(import_env.ImportOrderError, match="MLX_GDN_PACKED"):
        import_env.assert_profile_applied(
            "test", {"MLX_QWEN4_FUSED_GDN_DECODE": "0", "MLX_GDN_PACKED": "1"})


def test_snapshot_of_an_unimported_module_is_ignored(monkeypatch):
    from mlx2.runtime.models import import_env

    monkeypatch.setattr(import_env, "_SNAPSHOTS", {"mlx2_never_imported": {"MLX_QWEN4_X": "1"}})
    import_env.assert_profile_applied("test", {})


def test_flash_next_profile_enables_fused_gdn_prefill(monkeypatch, tmp_path):
    from mlx2.adapters import flash_next
    from mlx2.runtime.models import import_env

    import os

    monkeypatch.setattr(import_env, "_SNAPSHOTS", {})
    # configure_environment edits os.environ; keep the edit inside this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    profile = flash_next.configure_environment(tmp_path)
    assert profile["MLX_QWEN4_FUSED_GDN_PREFILL"] == "1"
    # Bit-exact against the eager MoE tail on Metal (recon-20261001/l4-moe-wsum).
    assert profile["MLX_QWEN4_MOE_WEIGHTED_SUM"] == "1"


def test_flash_next_subclasses_without_a_policy_defer_to_the_engine(monkeypatch):
    # Qwen3.8/3.6 dense and Nemotron inherit FlashNextAdapter but carry no
    # FlashNextPolicy: the inherited prefill_step_default must not raise
    # (it crashed every such model's generation worker at startup).
    from mlx2.adapters.flash_next import FlashNextAdapter

    class Subclass(FlashNextAdapter):
        def __init__(self):  # no FlashNextPolicy, like Qwen3827BAdapter
            pass

    assert Subclass().prefill_step_default() is None


def test_engine_keeps_its_default_when_the_adapter_declines(monkeypatch):
    settings, seen, engine = _engine_settings(monkeypatch, "decline")
    assert engine.prefill_step == 2048 and seen["prefill_step"] == 2048
    assert settings["prefill_step_source"] == "default"
def test_serving_wires_weights_once_for_every_route(monkeypatch):
    """The wired limit is raised after the adapter loads, not by one generator."""
    from mlx2.runtime import weight_residency

    calls = []
    receipt = {"wired": True, "wired_limit_bytes": 123}
    monkeypatch.setattr(
        weight_residency, "wire_serving_weights", lambda: calls.append(1) or receipt
    )
    _settings, _seen, engine = _engine_settings(monkeypatch, None)
    assert calls == [1]
    assert engine.status()["weight_residency"] == receipt


def test_wire_serving_weights_sets_the_working_set_limit(monkeypatch):
    from mlx2.runtime import weight_residency

    mx = weight_residency.mx
    seen = []
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1 << 30}
    )
    monkeypatch.setattr(mx, "set_wired_limit", lambda limit: seen.append(limit) or 0)
    receipt = weight_residency.wire_serving_weights()
    assert seen == [1 << 30]
    assert receipt["wired"] and receipt["wired_limit_bytes"] == 1 << 30

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert weight_residency.wire_serving_weights()["wired"] is False


def test_prefill_identity_reaches_settings_and_both_cache_namespaces(monkeypatch):
    from mlx2.runtime.prefill_plan import apc_prefill_fingerprint

    identity = {
        "version": 1,
        "scan": {"chunk_size": 8, "segment_max_rows": 512, "layers": 48},
    }
    settings, seen, _ = _engine_settings(monkeypatch, 512, prefill_identity=identity)
    assert settings["prefill_execution"] == identity
    assert len(seen["cache_semantics"]) == 2
    assert (
        seen["cache_semantics"]
        == [apc_prefill_fingerprint("text-token-v1", identity)] * 2
    )
    plain, plain_seen, _ = _engine_settings(monkeypatch, 512)
    assert "prefill_execution" not in plain
    assert plain_seen["cache_semantics"] == ["text-token-v1"] * 2


def test_prefill_candidate_fails_closed_without_qualification(monkeypatch):
    _engine_settings(monkeypatch, 512, prefill_identity={"scan": {"chunk_size": 8}},
                     expect_error="prefill candidates require qualification")
