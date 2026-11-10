"""The native Gemma 4 A4B builds mlx2 SwitchGLU experts, so it must hold the
stock MoE process-global claim (NAX gather off, rhs pad floor) while it runs,
roll it back on a failed load, and release it on close.  The dense 31B and the
mlx-vlm reference backend build no mlx2 experts and claim nothing."""

from __future__ import annotations

import pytest

from mlx2.adapters import gemma4 as g4
from mlx2.adapters import mlx_vlm, process_globals
from mlx2.runtime.models import gemma4 as native
from mlx2.runtime.models import moe_nax_gather, switch_layers


class _Backend:
    @staticmethod
    def load_processor(model_path, *, eos_token_ids=None):
        return "processor"

    @staticmethod
    def load(model_path, **kwargs):
        return "vlm-model", "processor"


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(process_globals, "_HOLDERS", {})
    monkeypatch.setattr(process_globals, "_PENDING", {})
    monkeypatch.setattr(moe_nax_gather, "MODE", "off")
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "floor")
    monkeypatch.setenv("MLX2_MOE_NAX_GATHER", "off")
    monkeypatch.setenv("MLX2_MOE_RHS_PAD_POLICY", "floor")
    monkeypatch.setattr(native, "load_config", lambda path: {})
    monkeypatch.setattr(native, "load", lambda path: "native-model")
    monkeypatch.setattr(mlx_vlm, "_load_model", lambda load, path, **kw: load(path))
    state = {"fail": False}

    def base_init(self, model_path, *, execution_policy=None):
        # Only the part of the shared constructor that loads tensors.
        self.identity = {"config": {}}
        self._vlm_backend = _Backend()
        self.model, self.processor = self._load_model_and_processor(model_path)
        if state["fail"]:
            raise RuntimeError("load failed after the claim")

    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter, "__init__", base_init)
    return state


def _claimed():
    return [owner for owner, _ in process_globals.live_selections()]


def test_native_a4b_claims_and_releases_the_stock_moe_globals(harness, monkeypatch):
    monkeypatch.setenv("MLX2_GEMMA4_BACKEND", "native")
    adapter = g4.Gemma4A4BAdapter("/fake")
    assert adapter.tensor_backend == "native"
    assert _claimed() == ["the Gemma4A4BAdapter adapter"]
    adapter.close()
    assert _claimed() == []


def test_native_a4b_failed_load_rolls_the_claim_back(harness, monkeypatch):
    monkeypatch.setenv("MLX2_GEMMA4_BACKEND", "native")
    harness["fail"] = True
    with pytest.raises(RuntimeError, match="after the claim"):
        g4.Gemma4A4BAdapter("/fake")
    assert _claimed() == [] and process_globals._PENDING == {}


@pytest.mark.parametrize(
    "cls, backend",
    [(g4.Gemma431BAdapter, "native"), (g4.Gemma4A4BAdapter, "mlx_vlm")],
)
def test_no_mlx2_experts_means_no_claim(harness, monkeypatch, cls, backend):
    monkeypatch.setenv("MLX2_GEMMA4_BACKEND", backend)
    adapter = cls("/fake")
    assert adapter.tensor_backend == backend
    assert _claimed() == []
    adapter.close()
