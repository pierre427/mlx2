"""The sorted-MoE pad policy is route identity (sweep 1002 review item 1).

``MLX2_MOE_RHS_PAD_POLICY=adaptive`` (or ``always``, or another
``MLX2_MOE_RHS_PAD_MIN_ROWS`` floor) runs other expert kernels than the
default floor, with another reduction order, yet left the adapter
environment, adapter policy and serving settings identical to the default
route's.  A floor receipt therefore qualified the adaptive route.  The
effective law now enters the serving settings when it is not the default.
"""

import json

import pytest

from mlx2.qualification import APPROVED_QUALIFICATION_HARNESS, load_qualified_route
from mlx2.runtime import apc_numerics
from mlx2.runtime.models import switch_layers


def _settings(monkeypatch, policy="floor", floor=3):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    # The policy is latched at switch_layers import; the identity must report
    # what actually runs, so set the latched values (and the environment).
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", policy)
    monkeypatch.setattr(switch_layers, "_RHS_PAD_MIN_ROWS_PER_EXPERT", floor)
    monkeypatch.setenv("MLX2_MOE_RHS_PAD_POLICY", policy)
    monkeypatch.setenv("MLX2_MOE_RHS_PAD_MIN_ROWS", str(floor))
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=True)
    try:
        return engine.adapter.descriptor, engine.status()["settings"]
    finally:
        engine.close()


def test_default_floor_adds_no_setting(monkeypatch):
    _descriptor, settings = _settings(monkeypatch)
    assert "moe_rhs_pad" not in settings


@pytest.mark.parametrize(
    "policy,floor,expected",
    [
        ("adaptive", 3, {"policy": "adaptive", "min_rows_per_expert": 3}),
        ("always", 3, {"policy": "always", "min_rows_per_expert": 3}),
        ("floor", 2, {"policy": "floor", "min_rows_per_expert": 2}),
        ("adaptive", 0, {"policy": "off", "min_rows_per_expert": 0}),
    ],
)
def test_effective_pad_law_enters_the_settings(monkeypatch, policy, floor, expected):
    _descriptor, settings = _settings(monkeypatch, policy, floor)
    assert settings["moe_rhs_pad"] == expected


def test_a_floor_receipt_does_not_qualify_an_adaptive_route(monkeypatch, tmp_path):
    _descriptor, floor_settings = _settings(monkeypatch)
    descriptor, adaptive_settings = _settings(monkeypatch, "adaptive")
    assert floor_settings != adaptive_settings
    receipt = tmp_path / "floor.json"
    receipt.write_text(json.dumps({
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "runtime": "r", "artifact": "a", "settings": floor_settings,
        "passed": True, "checks": {},
    }))
    with pytest.raises(ValueError, match="does not match serving settings"):
        load_qualified_route(receipt, runtime="r", artifact="a", settings=adaptive_settings,
                             descriptor=descriptor, name="tiny")


def test_identity_reads_the_latched_policy_not_a_later_environment(monkeypatch):
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "adaptive")
    monkeypatch.setattr(switch_layers, "_RHS_PAD_MIN_ROWS_PER_EXPERT", 3)
    monkeypatch.delenv("MLX2_MOE_RHS_PAD_POLICY", raising=False)
    assert apc_numerics.moe_rhs_pad_identity() == {
        "policy": "adaptive", "min_rows_per_expert": 3,
    }
    assert apc_numerics.execution_numerics_identity() == {
        "version": 1, "MLX2_MOE_RHS_PAD_POLICY": "adaptive",
    }
