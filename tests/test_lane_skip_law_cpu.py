"""A --lane-policy skip moves projections back to stock: it is part of the law.

Two serves of one artifact, one keeping the MLP on stock kernels via
``skip``, ran 8-32-row calls (prefill tails, batched decode) under different
arithmetic yet shared one law_id, so one APCv2 namespace in memory, on idle
disk and in a persist dir rescanned across restarts.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import installer, policy
from mlx2.runtime.lane.installer import apc_lane_fingerprint


def _q(k, n, seed):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    m = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4)
    return m


class _Attn(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (
            _q(128, 128, s + i) for i in range(4))


class _MLP(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.gate_proj, self.up_proj = _q(128, 256, s), _q(128, 256, s + 1)
        self.down_proj = _q(256, 128, s + 2)


class _Block(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.self_attn, self.mlp = _Attn(s), _MLP(s + 10)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Block(100 * i) for i in range(2)]


def _serve(overrides):
    model = _Toy()
    receipt = lane.apply_policy(
        model, policy.resolve(policy.detect(model), overrides=overrides))
    lane.uninstall(model)
    return receipt


def test_lane_skip_patterns_bind_the_law_and_the_apc_namespace(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)
    default = _serve(None)
    no_mlp = _serve({"skip": ["*.mlp.*"]})
    assert no_mlp["refused"].get("skipped") == 6
    assert default["covered"] != no_mlp["covered"]
    # Different projections run the lane arithmetic: a different law.
    assert default["law_id"] != no_mlp["law_id"]
    assert apc_lane_fingerprint("base", default) != apc_lane_fingerprint("base", no_mlp)
    # A pattern that matches nothing changes nothing.
    assert _serve({"skip": ["nothing.here"]})["law_id"] == default["law_id"]


def _apply(model, overrides):
    return lane.apply_policy(model, policy.resolve(policy.detect(model), overrides=overrides))


# Skipping one member of the default gate/up stack splits it.
GATE_ONLY = {"skip": ["*.gate_proj"]}


def _layout(model):
    """Which projections run the lane law, on which stack and preparation."""
    return {name: (type(m), id(installer._group(m)), id(installer._prepared(m)),
                   m.__dict__.get("_lane_min_rows"))
            for name, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)}


@pytest.mark.parametrize("first,second", [(None, GATE_ONLY), (GATE_ONLY, None)],
                         ids=["narrow", "widen"])
def test_a_reinstall_that_changes_coverage_is_refused_and_changes_nothing(
        monkeypatch, first, second):
    # Review r1/r2: a repeat install that re-covered or restored only the
    # skipped member left its sibling on the old gate/up stack (a different
    # split-K) under the fresh-install law and APCv2 namespace.
    monkeypatch.setattr(installer, "available", lambda: True)
    model = _Toy()
    try:
        installed = _apply(model, first)
        mlp = model.layers[0].mlp
        assert (installer._group(mlp.up_proj) is not None) is (first is None)
        before = _layout(model)
        with pytest.raises(ValueError, match="coverage cannot change"):
            _apply(model, second)
        assert _layout(model) == before
        # The law the route was installed under still describes the model.
        again = _apply(model, first)
        assert again["law_id"] == installed["law_id"] == _serve(first)["law_id"]
        assert _layout(model) == before
    finally:
        lane.uninstall(model)


@pytest.mark.parametrize("overrides", [None, GATE_ONLY], ids=["default", "skip"])
def test_a_same_coverage_reinstall_is_accepted(monkeypatch, overrides):
    monkeypatch.setattr(installer, "available", lambda: True)
    model = _Toy()
    try:
        first = _apply(model, overrides)
        before = _layout(model)
        again = _apply(model, overrides)
        assert again["law_id"] == first["law_id"] == _serve(overrides)["law_id"]
        assert _layout(model) == before
    finally:
        lane.uninstall(model)
