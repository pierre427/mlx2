"""Adapter profile transitions keep operator switches (sweep 2026-10-12 L1-F2).

678fa8fb2 made every profile clear the shared selector namespaces
(``process_env.INHERITED_MODEL_PROFILE_PREFIXES``) so that a previous model's
route choice, for example a Qwen TensorFold ``MLX2_QWEN_TARGET_EXECUTION``,
cannot leak into a later Muse load.  It also deleted what the operator had set
in the process environment before any profile applied: the Qwen3.8 tree gate
kill switches, the documented ``SERVING_ENV_SWITCHES`` and the cohort
qualifier's gates.  These tests pin both halves: operator values survive
every profile; values that appear after the first profile do not.
"""

import importlib.util
import os
import pathlib
import sys
from types import ModuleType

import pytest

from mlx2 import process_env
from mlx2.runtime.models import import_env

ROOT = pathlib.Path(__file__).resolve().parents[1]

OPERATOR = {
    # Documented tree15 gate kill switch (default-on route, "0 turns one off").
    "MLX2_TREE_SINGLE_FENCE": "0",
    # Documented cohort coalescing knob for the Qwen3.8 TensorFold route.
    "MLX2_TENSORFOLD_COHORT_LIMIT": "2",
    # env_switches.SERVING_ENV_SWITCHES members.
    "MLX2_XING_MHC_KERNEL": "0",
    "MLX2_FUSED_SDPA_MIN_L": "0",
    # Documented standard-decoder external-draft opt-in (SERVING.md).
    "MLX_LM_EXTERNAL_ROUND_COW": "1",
}


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch):
    """Each test is a process whose first profile has not applied yet."""
    monkeypatch.setattr(process_env, "_OPERATOR_SELECTORS", None, raising=False)
    monkeypatch.setattr(import_env, "_SNAPSHOTS", {})


@pytest.fixture
def operator_env(monkeypatch):
    for name, value in OPERATOR.items():
        monkeypatch.setenv(name, value)


def test_qwen38_profile_keeps_operator_tree_kill_switch(operator_env):
    from mlx2.adapters import qwen38_27b
    from mlx2.runtime.external_speculative import _tree_gates

    qwen38_27b.configure_environment()
    assert os.environ.get("MLX2_TREE_SINGLE_FENCE") == "0", "operator kill switch deleted"
    gates = _tree_gates("tree15", "provider", default_on=True)
    assert gates["single_fence"] is False, "default-on gate re-enabled despite explicit 0"
    assert os.environ.get("MLX2_TENSORFOLD_COHORT_LIMIT") == "2"


@pytest.mark.parametrize(
    ("module", "switch"),
    [("xing", "MLX2_XING_MHC_KERNEL"), ("gpt_oss", "MLX2_FUSED_SDPA_MIN_L")],
)
def test_profiles_keep_documented_serving_switches(operator_env, module, switch):
    from mlx2.runtime.env_switches import serving_env_switches

    importlib.import_module(f"mlx2.adapters.{module}").configure_environment()
    switches = serving_env_switches()
    assert switches.get("MLX2_XING_MHC_KERNEL") == "0"
    assert switches.get("MLX2_FUSED_SDPA_MIN_L") == "0"
    assert switches.get(switch) == "0"


class _Applied(Exception):
    pass


def test_standard_decoder_keeps_serving_switches_and_round_cow(
    operator_env, monkeypatch, tmp_path
):
    from mlx2.adapters import standard_decoder
    from mlx2.runtime.env_switches import serving_env_switches

    monkeypatch.setattr(standard_decoder, "inspect_artifact", lambda _path: {
        "config": {"model_type": "llama"},
        "identity": {"path": str(tmp_path), "fingerprint": "f", "files": []},
        "sampling_defaults": None,
    })
    seen = {}

    def applied(_owner):
        seen.update(os.environ)
        raise _Applied

    monkeypatch.setattr(standard_decoder, "assert_process_profile_applied", applied)
    with pytest.raises(_Applied):
        standard_decoder.StandardDecoderAdapter(str(tmp_path))
    assert seen.get("MLX_LM_EXTERNAL_ROUND_COW") == "1"
    assert serving_env_switches(seen).get("MLX2_FUSED_SDPA_MIN_L") == "0"
    assert serving_env_switches(seen).get("MLX2_XING_MHC_KERNEL") == "0"


def test_operator_value_latched_at_import_is_not_an_import_order_error(operator_env, monkeypatch):
    """Second failure mode: a module imported under the operator value."""
    from mlx2.adapters import gpt_oss

    fake = "mlx2.runtime.models._operator_latch_probe"
    monkeypatch.setitem(sys.modules, fake, ModuleType(fake))
    import_env.snapshot(fake)  # imported under the operator environment
    assert import_env._SNAPSHOTS[fake]["MLX2_FUSED_SDPA_MIN_L"] == "0"
    gpt_oss.configure_environment()
    assert os.environ["MLX2_FUSED_SDPA_MIN_L"] == "0"


def test_qwen38_profile_keeps_cohort_qualifier_gates(monkeypatch):
    """scripts/qualify_qwen38_tensorfold_cohort.py launches the server with these."""
    spec = importlib.util.spec_from_file_location(
        "qualify_qwen38_tensorfold_cohort",
        ROOT / "scripts" / "qualify_qwen38_tensorfold_cohort.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, value in mod.COMPOSED_GATES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MLX2_TENSORFOLD_COHORT_LIMIT", "4")
    from mlx2.adapters import qwen38_27b

    qwen38_27b.configure_environment()
    expected = {**mod.COMPOSED_GATES, "MLX2_TENSORFOLD_COHORT_LIMIT": "4"}
    lost = sorted(k for k, v in expected.items() if os.environ.get(k) != v)
    assert not lost, f"profile deleted qualifier gates: {lost}"


def test_operator_kill_switch_survives_an_intermediate_muse_load(operator_env):
    """Muse clears the tree namespace for itself; the next Qwen3.8 profile
    must see the operator's kill switch again, not a fail-open default."""
    from mlx2.adapters import muse_glimmer, qwen38_27b

    muse_glimmer.configure_environment()
    assert "MLX2_TREE_SINGLE_FENCE" not in os.environ
    assert "MLX2_TENSORFOLD_COHORT_LIMIT" not in os.environ
    qwen38_27b.configure_environment()
    assert os.environ.get("MLX2_TREE_SINGLE_FENCE") == "0"
    assert os.environ.get("MLX2_TENSORFOLD_COHORT_LIMIT") == "2"


def test_previous_qwen_tensorfold_selection_does_not_leak_into_muse(monkeypatch):
    """The 678fa8fb2 leak stays closed whether or not the operator set it."""
    from mlx2.adapters import muse_glimmer, qwen38_27b

    qwen38_27b.configure_environment()  # first profile: latches the operator set
    stale = {
        "MLX2_DFLASH_TOPOLOGY": "tree15",
        "MLX2_QWEN_TARGET_EXECUTION": "tensorfold",
        "MLX2_TENSORFOLD_CACHE_EXECUTOR": "1",
        "MLX2_TENSORFOLD_COHORT_LIMIT": "4",
        "MLX2_TREE_SINGLE_FENCE": "1",
    }
    for name, value in stale.items():
        monkeypatch.setenv(name, value)
    profile = muse_glimmer.configure_environment()
    assert profile["MLX2_QWEN_TARGET_EXECUTION"] == "reference"
    assert os.environ["MLX2_QWEN_TARGET_EXECUTION"] == "reference"
    for name in stale.keys() - {"MLX2_QWEN_TARGET_EXECUTION"}:
        assert name not in os.environ, name


@pytest.mark.parametrize("module", ["gpt_oss", "xing", "qwen38_27b"])
def test_selectors_set_after_the_first_profile_are_still_cleared(monkeypatch, module):
    """Only the operator's pre-profile environment is preserved; a selector a
    previous model or arm left behind is still cleared by the next profile."""
    for name in ("MLX2_DFLASH_TOPOLOGY", "MLX2_FUSED_SDPA_MIN_L", "MLX2_EAGER_DISPATCH_STRIDE"):
        monkeypatch.delenv(name, raising=False)
    first = importlib.import_module("mlx2.adapters.gpt_oss")
    first.configure_environment()
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.setenv("MLX2_FUSED_SDPA_MIN_L", "0")
    monkeypatch.setenv("MLX2_EAGER_DISPATCH_STRIDE", "3")
    importlib.import_module(f"mlx2.adapters.{module}").configure_environment()
    for name in ("MLX2_DFLASH_TOPOLOGY", "MLX2_FUSED_SDPA_MIN_L", "MLX2_EAGER_DISPATCH_STRIDE"):
        assert name not in os.environ, (module, name)


def test_operator_value_a_profile_overwrote_is_restored_for_the_next_profile(monkeypatch):
    from mlx2.adapters import gpt_oss, qwen38_27b

    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP", raising=False)
    monkeypatch.setenv("MLX_LM_EXTERNAL_ROUND_COW", "1")
    qwen38_27b.configure_environment()  # owns MLX_LM_: clears it for itself
    assert "MLX_LM_EXTERNAL_ROUND_COW" not in os.environ
    gpt_oss.configure_environment()
    assert os.environ.get("MLX_LM_EXTERNAL_ROUND_COW") == "1"
    # Qwen3.8's own MLX_LM_ pins are not operator values: gpt-oss drops them.
    assert "MLX_LM_SEGMENTED_SELF_MTP" not in os.environ
