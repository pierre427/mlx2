"""Qwen adapters refuse a stale import-time latch (sweep 2026-10-02 G2/P2/P3).

G2/P2: only Flash-Next called ``assert_profile_applied``.  The Qwen3.8 27B
(and the 9B that inherits its constructor), Qwen3.6 35B and Qwen3.5 122B
adapters pinned a profile into ``os.environ`` after a model module may already
have latched other values at import, and ran the stale route under a receipt
naming the new one.  Qwen3.6 B=1 fused GDN decode is now applied through its
live setter, so a second Qwen3.6 load in one process takes effect.

P3: an explicit ``MLX_ENABLE_TF32`` that enables TF32 was silently overwritten
with "0" by every adapter profile; the profiles now refuse it.
"""

import os
import sys
from types import ModuleType

import pytest

from mlx2 import process_env
from mlx2.runtime.models import import_env

FAKE = "mlx2.runtime.models._fake_latched_for_test"


class _Loaded(Exception):
    """Raised by the stubbed weight load: the guard let construction through."""


def _latched(monkeypatch, seen):
    """One snapshotted module that saw ``seen``; no other snapshot interferes."""
    monkeypatch.setattr(import_env, "_SNAPSHOTS", {FAKE: dict(seen)})
    monkeypatch.setitem(sys.modules, FAKE, ModuleType(FAKE))


def _artifact(tmp_path):
    return {
        "has_mtp": False,
        "identity": {"path": str(tmp_path), "fingerprint": "f", "files": []},
        "config": {"text_config": {"mtp_num_hidden_layers": 0}},
    }


def _qwen36(monkeypatch, tmp_path, policy=None):
    from mlx2.adapters import qwen36_35b

    monkeypatch.setattr(qwen36_35b, "inspect_artifact", lambda _p: _artifact(tmp_path))

    def load(self, *_a, **_k):
        raise _Loaded

    monkeypatch.setattr(qwen36_35b.Qwen3635BA3BAdapter, "_load_weights", load)
    return qwen36_35b.Qwen3635BA3BAdapter(str(tmp_path), execution_policy=policy)


def _qwen122(monkeypatch, tmp_path):
    from mlx2.adapters import qwen35_122b
    from mlx2.runtime.models import qwen35_122b as tensors

    monkeypatch.setattr(qwen35_122b, "inspect_artifact", lambda _p: _artifact(tmp_path))

    def model(_args):
        raise _Loaded

    monkeypatch.setattr(tensors, "Model", model)
    monkeypatch.setattr(tensors.ModelArgs, "from_dict", classmethod(lambda cls, c: c))
    return qwen35_122b.Qwen35122BA10BAdapter(str(tmp_path))


def _qwen27(monkeypatch, tmp_path, policy=None):
    from mlx2.adapters import qwen38_27b
    from mlx2.runtime.models import qwen38_27b as tensors

    monkeypatch.setattr(
        qwen38_27b.Qwen3827BAdapter,
        "artifact_inspector",
        staticmethod(lambda _p: _artifact(tmp_path)),
    )

    def model(_args):
        raise _Loaded

    monkeypatch.setattr(tensors, "Model", model)
    monkeypatch.setattr(tensors.ModelArgs, "from_dict", classmethod(lambda cls, c: c))
    return qwen38_27b.Qwen3827BAdapter(str(tmp_path), execution_policy=policy)


BUILDERS = {"qwen38_27b": _qwen27, "qwen36_35b": _qwen36, "qwen35_122b": _qwen122}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_stale_gdn_core_latch_refuses(monkeypatch, tmp_path, name):
    # A module imported under MLX_GDN_CORE=1 keeps the MLX core prefill kernel
    # whatever the profile pins now ("0" on all three).
    _latched(monkeypatch, {"MLX_GDN_CORE": "1"})
    with pytest.raises(import_env.ImportOrderError, match="MLX_GDN_CORE"):
        BUILDERS[name](monkeypatch, tmp_path)


@pytest.mark.parametrize("name", ["qwen36_35b", "qwen35_122b"])
def test_stale_moe_fused_gate_up_latch_refuses(monkeypatch, tmp_path, name):
    _latched(monkeypatch, {"MLX_QWEN4_MOE_FUSED_GATE_UP": "1"})
    with pytest.raises(import_env.ImportOrderError, match="MOE_FUSED_GATE_UP"):
        BUILDERS[name](monkeypatch, tmp_path)


def test_qwen122_refuses_stale_fused_gdn_decode_latch(monkeypatch, tmp_path):
    # 122B builds Qwen3.6 GDN layers but does not re-apply the decode switch.
    monkeypatch.delenv("MLX_QWEN36_FUSED_GDN_DECODE", raising=False)
    _latched(monkeypatch, {"MLX_QWEN36_FUSED_GDN_DECODE": "1"})
    with pytest.raises(import_env.ImportOrderError, match="FUSED_GDN_DECODE"):
        _qwen122(monkeypatch, tmp_path)


def test_qwen36_second_load_with_other_live_selection_is_allowed(monkeypatch, tmp_path):
    # First load in this process selected fused B=1 decode and the window;
    # the second selects nothing.  Those switches are re-applied by live
    # setters, so the stale import value is not a reason to refuse.
    from mlx2.adapters.qwen36_35b import configure_environment

    first = configure_environment(
        {"fused_gdn_decode": True, "moe_window": True, "fused_gdn_verify": True}
    )
    _latched(monkeypatch, {k: v for k, v in first.items() if k.startswith(import_env.PREFIXES)})
    with pytest.raises(_Loaded):
        _qwen36(monkeypatch, tmp_path, {"fused_gdn_decode": False})


def test_matching_profile_constructs(monkeypatch, tmp_path):
    from mlx2.adapters.qwen38_27b import configure_environment

    seen = configure_environment()
    _latched(monkeypatch, {k: v for k, v in seen.items() if k.startswith(import_env.PREFIXES)})
    with pytest.raises(_Loaded):
        _qwen27(monkeypatch, tmp_path)


@pytest.mark.parametrize("latched,selected,expected", [
    (True, "0", "stock"),   # policy kill switch beats a stale "1" latch
    (False, "1", "fused"),  # a later selection takes effect without re-import
])
def test_qwen36_fused_gdn_decode_applied_live(monkeypatch, latched, selected, expected):
    import mlx.core as mx

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.runtime.models import qwen36_35b as tensors
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(tensors, "_FUSED_GDN_DECODE", latched)
    layer = tensors.GatedDeltaNet(TextModelArgs(
        hidden_size=64, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, num_attention_heads=4,
    ))
    assert layer.fused_gdn_decode_mode == ("fused" if latched else "stock")

    class Model:
        def named_modules(self):
            return [("layer", layer)]

    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.model = Model()
    adapter._kernels = {}
    adapter.environment = {"MLX_QWEN36_FUSED_GDN_DECODE": selected}
    adapter._select_decode_wins()
    assert layer.fused_gdn_decode_mode == expected


# --- P3: explicit TF32 ------------------------------------------------------


@pytest.mark.parametrize("module", ["qwen36_35b", "qwen38_27b"])
def test_explicit_tf32_is_refused_not_overwritten(monkeypatch, module):
    import importlib

    configure = importlib.import_module(f"mlx2.adapters.{module}").configure_environment
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(process_env.ProcessNumericsConflict, match="MLX_ENABLE_TF32"):
        configure()
    assert os.environ["MLX_ENABLE_TF32"] == "1"


def test_explicit_zero_and_default_are_accepted(monkeypatch):
    from mlx2.adapters.qwen38_27b import configure_environment

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    assert configure_environment()["MLX_ENABLE_TF32"] == "0"


def test_guard_refuses_tf32_explicit_at_package_import(monkeypatch):
    # A profile that already overwrote the operator's "1" (Flash-Next and the
    # other adapters that spread PROCESS_NUMERICS) is caught by the guard.
    monkeypatch.setattr(import_env, "_SNAPSHOTS", {})
    monkeypatch.setattr(process_env, "_EXPLICIT_AT_IMPORT", "1")
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    with pytest.raises(process_env.ProcessNumericsConflict, match="MLX_ENABLE_TF32"):
        import_env.assert_profile_applied("the Flash-Next adapter")
