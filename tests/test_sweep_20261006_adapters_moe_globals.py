"""Regression tests from the 2026-10-06 sweep: MoE process-global claims and
profile environment rollback across adapters (LEAD-01, LEAD-02, G3-01/04)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "src" / "mlx2" / "adapters"
MODELS = ROOT / "src" / "mlx2" / "runtime" / "models"
XING_FIXTURE = ROOT / "tests" / "fixtures" / "xing4_0_tiny"
QWEN36_35B = Path.home() / "mlx-models" / "Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"

# adapter module -> (adapter class, runtime model module it builds)
STOCK_MOE_ADAPTERS = {
    "gpt_oss": ("GptOssAdapter", "gpt_oss"),
    "granite_swa": ("GraniteSWAAdapter", "granitemoe_swa"),
    "hy_v3": ("HYV3Adapter", "hy_v3"),
    "laguna_xs21": ("LagunaXS21Adapter", "laguna"),
    "laguna_s21": ("LagunaS21Adapter", "laguna"),
    "nemotron3_super": ("Nemotron3SuperAdapter", "nemotron_h"),
    "nemotron35_lightning": ("Nemotron35LightningAdapter", "nemotron_h"),
    "xing": ("XingAdapter", "xing4_0"),
}
FAKE_ARTIFACT = {
    "identity": {"path": "/nonexistent-mlx2-artifact", "fingerprint": "x", "files": []},
    "config": {}, "reap": False, "has_mtp": False, "weight_map": {},
    "mtp_path": None, "qualification": "pending",
}


@pytest.fixture
def fresh_globals(monkeypatch):
    from mlx2.adapters import process_globals
    from mlx2.runtime.models import moe_nax_gather, switch_layers

    monkeypatch.setattr(process_globals, "_HOLDERS", {})
    monkeypatch.setattr(process_globals, "_PENDING", {})
    monkeypatch.setattr(moe_nax_gather, "MODE", "off")
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "floor")


@pytest.mark.parametrize("adapter", sorted(STOCK_MOE_ADAPTERS))
def test_switch_layers_adapter_claims_the_moe_globals(adapter):
    _, model = STOCK_MOE_ADAPTERS[adapter]
    assert "switch_layers import" in (MODELS / f"{model}.py").read_text()
    assert "claim_stock_moe(self" in (ADAPTERS / f"{adapter}.py").read_text()


@pytest.mark.parametrize("adapter", sorted(STOCK_MOE_ADAPTERS))
def test_failed_load_rolls_back_claim_and_environment(adapter, monkeypatch, fresh_globals):
    """The claim lands before tensors load, and a failed load leaves neither
    the claim, the pad policy nor the pinned profile behind."""
    from mlx2.adapters import process_globals
    from mlx2.runtime.models import switch_layers

    name, _ = STOCK_MOE_ADAPTERS[adapter]
    module = importlib.import_module(f"mlx2.adapters.{adapter}")
    monkeypatch.setattr(module, "inspect_artifact", lambda *a, **k: dict(FAKE_ARTIFACT))
    real = process_globals.claim_stock_moe
    seen = {}

    class Stop(Exception):
        pass

    def claim_then_fail(holder, owner):
        real(holder, owner)
        seen.update(owner=owner, policy=switch_layers._RHS_PAD_POLICY,
                    live=process_globals._PENDING.copy())
        raise Stop

    monkeypatch.setattr(process_globals, "claim_stock_moe", claim_then_fail)
    switch_layers._RHS_PAD_POLICY = "always"  # an inherited foreign value
    os.environ["MLX_LM_OTHER_ADAPTER_PIN"] = "1"
    before = dict(os.environ)
    with pytest.raises(Stop):
        getattr(module, name)("/nonexistent-mlx2-artifact")
    assert seen["policy"] == "floor" and seen["live"]
    assert switch_layers._RHS_PAD_POLICY == "always"
    assert process_globals.live_selections() == [] and not process_globals._PENDING
    assert dict(os.environ) == before


def _tiny_xing_adapter(monkeypatch):
    from mlx2.adapters import xing, xing_tokenizer

    config = json.loads((XING_FIXTURE / "config.json").read_text())
    config.setdefault("max_position_embeddings", 4096)
    artifact = {
        "config": config, "weight_map": {"w": "weights.safetensors"},
        "has_mtp": int(config.get("num_nextn_predict_layers", 0)) > 0,
        "identity": {"path": str(XING_FIXTURE), "fingerprint": "tiny", "files": []},
        "qualification": "pending",
    }
    monkeypatch.setattr(xing, "inspect_artifact", lambda path: artifact)
    monkeypatch.setattr(
        xing_tokenizer, "load_tokenizer",
        lambda path, allow_reference_fallback=False: (object(), {"tiny": True}),
    )
    monkeypatch.setattr(
        xing_tokenizer, "make_tokenizer_wrapper",
        lambda tokenizer, eos_token_ids=None: object(),
    )
    return xing.XingAdapter(str(XING_FIXTURE))


@pytest.mark.skipif(not XING_FIXTURE.is_dir(), reason="Xing tiny fixture absent")
def test_live_xing_refuses_a_foreign_pad_policy_until_closed(monkeypatch, fresh_globals):
    from mlx2.adapters.process_globals import (
        MOE_RHS_PAD_POLICY, ProcessGlobalConflict, claim, live_selections,
    )
    from mlx2.runtime.models import switch_layers

    adapter = _tiny_xing_adapter(monkeypatch)
    foreign = type("FlashNextLike", (), {})()
    selection = {MOE_RHS_PAD_POLICY: ("always", switch_layers.set_pad_policy)}
    try:
        assert [owner for owner, _ in live_selections()] == ["the Xing4.0 adapter"]
        with pytest.raises(ProcessGlobalConflict):
            claim(foreign, "flash-next-like", selection)
    finally:
        adapter.close()
    assert live_selections() == []
    claim(foreign, "flash-next-like", selection).rollback()


@pytest.mark.parametrize(
    "adapter,name",
    [("agnes_3_flash", "Agnes3FlashAdapter"), ("muse_glimmer", "MuseGlimmerAdapter"),
     ("olmo_hils", "OlmoHiLSAdapter")],
)
def test_failed_non_moe_load_restores_environment(adapter, name, monkeypatch):
    module = importlib.import_module(f"mlx2.adapters.{adapter}")
    artifact = dict(FAKE_ARTIFACT)
    # Muse's inspector returns the identity itself.
    monkeypatch.setattr(
        module, "inspect_artifact",
        lambda *a, **k: artifact["identity"] if adapter == "muse_glimmer" else artifact,
    )
    os.environ["MLX_LM_OTHER_ADAPTER_PIN"] = "1"
    before = dict(os.environ)
    with pytest.raises(Exception):
        getattr(module, name)("/nonexistent-mlx2-artifact")
    assert dict(os.environ) == before


def test_profile_wipe_keeps_operator_checkpoint_knobs(monkeypatch):
    from mlx2.adapters.hy_v3 import configure_environment

    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "512")
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "2")
    monkeypatch.setenv("MLX_LM_SOME_EXPERIMENT", "1")
    configure_environment()
    assert os.environ["MLX_LM_STATE_CHECKPOINT_STRIDE"] == "512"
    assert os.environ["MLX_LM_STATE_CHECKPOINT_MAX"] == "2"
    assert "MLX_LM_SOME_EXPERIMENT" not in os.environ


@pytest.mark.skipif(not QWEN36_35B.is_dir(), reason="Qwen3.6-35B artifact absent")
def test_qwen36_35b_claims_the_rhs_pad_policy():
    # Fresh interpreter: the adapter's import-order guard needs model modules
    # unimported.  The real constructor runs (metadata only) up to its claim.
    script = textwrap.dedent(f"""
        import json
        from mlx2.adapters import process_globals as pg
        from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
        captured = {{}}
        class Stop(Exception):
            pass
        def fake_claim(holder, owner, selections):
            captured.update({{k: v[0] for k, v in selections.items()}})
            raise Stop
        pg.claim = fake_claim
        try:
            Qwen3635BA3BAdapter({str(QWEN36_35B)!r})
        except Stop:
            pass
        print(json.dumps(captured))
    """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    ).stdout.strip().splitlines()[-1]
    assert json.loads(out)["moe_rhs_pad_policy"] == "floor"


# Dense: its imports of invariant_prefill reach switch_layers only for the
# MoE helpers it never builds.
_DENSE_EXEMPT = {"qwen38_27b"}


def test_every_adapter_building_switch_layers_claims_the_globals():
    """Discovery guard: an adapter whose runtime model imports mlx2
    switch_layers must claim the MoE process globals itself."""
    import re

    users = {
        path.stem for path in MODELS.glob("*.py")
        if "switch_layers import" in path.read_text()
    }
    missing = []
    for path in sorted(ADAPTERS.glob("*.py")):
        source = path.read_text()
        built = set(re.findall(r"runtime\.models\.(\w+) import", source)) | set(
            re.findall(r"runtime\.models import (\w+)", source)
        )
        if built & users and path.stem not in _DENSE_EXEMPT and not (
            "claim_stock_moe(self" in source or "MOE_RHS_PAD_POLICY" in source
        ):
            missing.append(path.stem)
    assert missing == []
