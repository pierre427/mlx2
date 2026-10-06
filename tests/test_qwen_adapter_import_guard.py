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
    expected = "MOE_FUSED_GATE_UP" if name == "qwen36_35b" else "import-time flags"
    with pytest.raises(import_env.ImportOrderError, match=expected):
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


def test_qwen27_varlen_dense_mlp_policy_reaches_model_construction(
    monkeypatch, tmp_path
):
    from mlx2.adapters.qwen38_27b import configure_environment

    seen = configure_environment()
    _latched(
        monkeypatch,
        {key: value for key, value in seen.items() if key.startswith(import_env.PREFIXES)},
    )
    with pytest.raises(_Loaded):
        _qwen27(monkeypatch, tmp_path, {"varlen_dense_mlp": True})


def test_qwen27_external_draft_preserves_target_varlen_policy(
    monkeypatch, tmp_path
):
    from mlx2.adapters import qwen38_27b
    from mlx2.adapters.qwen38_27b import configure_environment
    from mlx2.runtime.models.varlen_dense_mlp import VarlenDenseMLPPolicy

    seen = configure_environment()
    _latched(
        monkeypatch,
        {key: value for key, value in seen.items() if key.startswith(import_env.PREFIXES)},
    )
    parsed = []
    original = VarlenDenseMLPPolicy.from_value.__func__

    def record(cls, value):
        parsed.append(value)
        return original(cls, value)

    monkeypatch.setattr(VarlenDenseMLPPolicy, "from_value", classmethod(record))
    monkeypatch.setattr(qwen38_27b, "inspect_external_policy", lambda *_a: {})
    with pytest.raises(_Loaded):
        _qwen27(
            monkeypatch,
            tmp_path,
            {"draft_model": "draft", "varlen_dense_mlp": True},
        )
    assert parsed == [True]


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


def test_standard_decoder_refuses_explicit_tf32(monkeypatch, tmp_path):
    # The standard decoder pins no profile, so an explicit TF32 used to run
    # unrecorded; it is refused like every pinned profile (flip 2026-10-02).
    from mlx2.adapters import standard_decoder

    def inspect(_path):
        raise _Loaded

    monkeypatch.setattr(standard_decoder, "inspect_artifact", inspect)
    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(process_env.ProcessNumericsConflict, match="standard decoder"):
        standard_decoder.StandardDecoderAdapter(str(tmp_path))
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(process_env, "_EXPLICIT_AT_IMPORT", None)
    with pytest.raises(_Loaded):
        standard_decoder.StandardDecoderAdapter(str(tmp_path))


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


# --- review item 5: routed decode / top-k reload ----------------------------


def test_qwen36_reload_turning_routed_decode_and_topk_off_is_allowed(monkeypatch, tmp_path):
    """First load selected routed decode and the top-k launch; the second
    turns both off.  ``_select_decode_wins`` re-applies both through the MoE
    blocks' live setters, so the latched import values are not a refusal."""
    from mlx2.adapters.qwen36_35b import configure_environment

    first = configure_environment(
        {"moe_routed_decode": "gate_up_down", "moe_topk_fold": "launch"}
    )
    assert first["MLX_QWEN4_MOE_ROUTED_DECODE"] == "gate_up_down"
    _latched(monkeypatch, {k: v for k, v in first.items() if k.startswith(import_env.PREFIXES)})
    with pytest.raises(_Loaded):
        _qwen36(monkeypatch, tmp_path, {"moe_routed_decode": "off", "moe_topk_fold": "off"})


@pytest.mark.parametrize("latched,kernels,expected", [
    (("gate_up_down", "launch"), {}, ("off", "off")),
    (("off", "off"), {"moe_routed_decode": "gate_up_down_shared", "moe_topk_fold": "launch"},
     ("gate_up_down_shared", "launch")),
])
def test_qwen36_routed_decode_and_topk_applied_live(monkeypatch, latched, kernels, expected):
    """Every Qwen3.6 MoE block (decoder and MTP layers build the same class)
    takes the selection from the live setters, whatever it latched."""
    import mlx.core as mx

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.runtime.models import qwen3_next
    from mlx2.runtime.models.qwen36_moe_decode import Qwen36SparseMoeBlock
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(qwen3_next, "_MOE_ROUTED_DECODE", latched[0])
    monkeypatch.setattr(qwen3_next, "_MOE_TOPK_MODE", latched[1])
    monkeypatch.setattr(qwen3_next, "_MOE_SHARED_IN_GATHER", False)
    args = TextModelArgs(
        hidden_size=64, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=64, norm_topk_prob=True,
    )
    blocks = [Qwen36SparseMoeBlock(args), Qwen36SparseMoeBlock(args)]
    assert blocks[0].switch_mlp.routed_decode_mode == latched[0]
    assert blocks[0].moe_topk_mode == latched[1]

    class Model:
        def named_modules(self):
            return [(f"b{i}", b) for i, b in enumerate(blocks)]

    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.model = Model()
    adapter._kernels = kernels
    adapter.environment = {}
    adapter._select_decode_wins()
    for block in blocks:
        assert (block.switch_mlp.routed_decode_mode, block.moe_topk_mode) == expected


# --- review item 6: TF32 enabled after package import ----------------------

PROFILE_MODULES = [
    "agnes_3_flash", "flash_next", "gpt_oss", "hy_v3", "laguna_xs21",
    "muse_glimmer", "nemotron3_super", "north_mini_code", "qwen35_9b",
    "qwen36_27b", "qwen36_35b", "qwen38_27b", "xing",
]


def _configure(module, tmp_path):
    import importlib

    configure = importlib.import_module(f"mlx2.adapters.{module}").configure_environment
    return configure(tmp_path) if module == "flash_next" else configure()


@pytest.mark.parametrize("module", PROFILE_MODULES)
def test_tf32_set_after_package_import_is_refused_by_every_profile(monkeypatch, tmp_path, module):
    """Codex review item 6: ``mlx2`` imported first (no explicit value, so
    nothing recorded at import), then MLX_ENABLE_TF32=1 and possibly an fp32
    dispatch.  The profile must refuse before it writes "0" over the value,
    or the later guard sees only its own "0"."""
    monkeypatch.setattr(process_env, "_EXPLICIT_AT_IMPORT", None)
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(process_env.ProcessNumericsConflict, match="MLX_ENABLE_TF32"):
        _configure(module, tmp_path)
    assert os.environ["MLX_ENABLE_TF32"] == "1"


@pytest.mark.parametrize("module,cls", [
    ("olmo_hils", "OlmoHiLSAdapter"), ("granite_swa", "GraniteSWAAdapter"),
])
def test_tf32_set_after_package_import_is_refused_by_inline_profiles(
    monkeypatch, tmp_path, module, cls
):
    import importlib

    mod = importlib.import_module(f"mlx2.adapters.{module}")
    monkeypatch.setattr(mod, "inspect_artifact", lambda _p: {
        "identity": {"path": str(tmp_path), "fingerprint": "f", "files": []}, "config": {},
    })
    monkeypatch.setattr(process_env, "_EXPLICIT_AT_IMPORT", None)
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(process_env.ProcessNumericsConflict, match="MLX_ENABLE_TF32"):
        getattr(mod, cls)(str(tmp_path))
    assert os.environ["MLX_ENABLE_TF32"] == "1"


def test_every_profile_that_spreads_process_numerics_is_covered():
    """A new adapter that spreads PROCESS_NUMERICS must refuse explicit TF32."""
    from pathlib import Path

    adapters = Path(__file__).resolve().parents[1] / "src" / "mlx2" / "adapters"
    spreading = {
        p.stem for p in adapters.glob("*.py") if "**PROCESS_NUMERICS" in p.read_text()
    }
    refusing = {
        p.stem for p in adapters.glob("*.py") if "require_process_numerics(" in p.read_text()
    }
    assert spreading <= refusing, sorted(spreading - refusing)
    assert spreading == set(PROFILE_MODULES) | {"olmo_hils", "granite_swa"}


def test_qwen36_routed_decode_default_steps_aside_for_the_candidate(monkeypatch, tmp_path):
    # The default decode wins (flip 2026-10-02) must not turn an explicit
    # historical-candidate selection into an exclusivity refusal.
    from mlx2.adapters import qwen36_35b

    seen = {}

    def load(self, *_a, **_k):
        seen.update(self._kernels)
        raise _Loaded

    monkeypatch.setattr(qwen36_35b, "inspect_artifact", lambda _p: _artifact(tmp_path))
    monkeypatch.setattr(qwen36_35b.Qwen3635BA3BAdapter, "_load_weights", load)
    monkeypatch.setattr(import_env, "_SNAPSHOTS", {})
    with pytest.raises(_Loaded):
        qwen36_35b.Qwen3635BA3BAdapter(
            str(tmp_path), execution_policy={"moe_routed_candidate": True}
        )
    assert seen["moe_routed_candidate"] is True
    assert "moe_routed_decode" not in seen
    assert seen["moe_topk_fold"] == "launch"
    with pytest.raises(ValueError, match="exclusive"):
        _qwen36(monkeypatch, tmp_path, {
            "moe_routed_candidate": True, "moe_routed_decode": "gate_up_down_shared"})


@pytest.mark.parametrize("selected", ["1", "0"])
def test_qwen36_fused_gdn_diagnostics_follow_the_live_mode(monkeypatch, selected):
    # An operator's MLX_QWEN36_FUSED_GDN_DECODE (policy silent) selects the
    # fused decode live; its counters must be reported, or the qualifier
    # fails the route closed.  The decode-wins block carries no hard-coded
    # qualification label: the route receipt decides that.
    import mlx.core as mx

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.models import qwen36_35b as tensors
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(tensors, "_FUSED_GDN_DECODE", False)
    layer = tensors.GatedDeltaNet(TextModelArgs(
        hidden_size=64, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, num_attention_heads=4,
    ))

    class Model:
        def named_modules(self):
            return [("layer", layer)]

    monkeypatch.setattr(Qwen3827BAdapter, "diagnostics", lambda self: {})
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.model = Model()
    adapter._kernels = {"fused_gdn_verify": True}
    adapter.layout = "fixture-layout"
    adapter.environment = {"MLX_QWEN36_FUSED_GDN_DECODE": selected}
    adapter._select_decode_wins()
    monkeypatch.setattr(tensors, "qwen36_decode_wins_stats", lambda _m: {"calls": 0})
    result = adapter.diagnostics()
    if selected == "1":
        assert result["fused_gdn_decode"]["mode"] == "fused"
    else:
        assert "fused_gdn_decode" not in result
    assert result["decode_wins"] == {"calls": 0}
