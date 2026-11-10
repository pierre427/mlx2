"""Bug sweep 2026-11-06, Ripple lane: CPU regressions (no weights, no Metal)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from mlx2.adapters.ripple import CACHE_LAYOUT, apc_layout, resolved_options

ROOT = Path(__file__).resolve().parents[1]


def test_affine_default_namespace_is_unchanged():
    # All three affine defaults on: metal_glue shares the metal_rmsnorm law.
    assert apc_layout(CACHE_LAYOUT, resolved_options("affine", {})) == (
        CACHE_LAYOUT + "+gqa_folded_attention+metal_rmsnorm")
    assert apc_layout(CACHE_LAYOUT, {}) == CACHE_LAYOUT
    assert apc_layout(CACHE_LAYOUT, {"fused_elementwise": True}) == CACHE_LAYOUT


def test_metal_glue_without_metal_rmsnorm_gets_its_own_namespace():
    # RIPPLE-1: glue alone reorders the Q/K norm FP32 sum while block norms
    # stay eager; it shared the eager namespace and could warm-hit a prefix
    # cached under a different arithmetic law.
    eager = apc_layout(CACHE_LAYOUT, {})
    rmsnorm = apc_layout(CACHE_LAYOUT, {"metal_rmsnorm": True})
    glue = apc_layout(CACHE_LAYOUT, {"metal_glue": True})
    assert len({eager, rmsnorm, glue}) == 3
    explicit_off = resolved_options("affine", {"metal_rmsnorm": False})
    assert apc_layout(CACHE_LAYOUT, explicit_off) == CACHE_LAYOUT + "+gqa_folded_attention+metal_glue"
    assert apc_layout(CACHE_LAYOUT, {"metal_rmsnorm": True, "metal_glue": True}) == rmsnorm


def _script(name):
    spec = importlib.util.spec_from_file_location(f"sweep1106_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "function"), [
    ("probe_ripple_serving", "source_revision"),
    ("convert_ripple_oq4e", "checkout_revision"),
])
def test_receipt_revision_is_the_scripts_checkout_not_the_cwd(tmp_path, monkeypatch, name, function):
    # RIPPLE-2: ``git rev-parse HEAD`` ran in the caller's cwd, so a receipt
    # named whatever checkout the operator stood in (or failed outside one).
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    module = _script(name)
    monkeypatch.chdir(tmp_path)
    assert getattr(module, function)() == expected
