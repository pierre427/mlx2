"""Adapter-owned prefill chunk default and the import-order profile guard."""

import sys
import types

import pytest


def _engine_settings(monkeypatch, adapter_step, **overrides):
    from types import SimpleNamespace as NS

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    class APC:
        def __init__(self, **_kw):
            self.apc_stats = {}

        def key(self, *_a, **_kw):
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
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, *, max_lanes, prefill_step):
            seen["prefill_step"] = prefill_step
            return {"num_draft": 0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    if adapter_step is not None:
        Adapter.prefill_step_default = lambda self: adapter_step
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine("fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
                                   max_lanes=2, max_inflight=2, **overrides)
    try:
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
