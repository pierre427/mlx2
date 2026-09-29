"""Lane matmul tunables: detection, resolution order, validation, stats, metrics."""

import json

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import installer, policy


def _q(k, n, bits, gs=64, seed=0):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    m = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
    m.weight, m.scales, m.biases = mx.quantize(w, group_size=gs, bits=bits)
    return m


class SwitchGLU(nn.Module):          # name-detected mixture-of-experts block
    def __init__(self):
        super().__init__()
        self.gate = _q(128, 64, 4, seed=9)


class _Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _q(128, 64, 4, seed=1)
        self.k_proj = _q(128, 64, 4, seed=2)
        self.down_proj = _q(128, 64, 5, seed=3)
        self.lm_head = _q(128, 64, 8, seed=4)
        self.dense = nn.Linear(128, 64, bias=False)
        self.dense.weight = self.dense.weight.astype(mx.bfloat16)


class _MoE(_Dense):
    def __init__(self):
        super().__init__()
        self.experts = SwitchGLU()


def test_detect_reports_formats_and_moe():
    dense = policy.detect(_Dense())
    assert dense == {"moe": False, "formats": {"q4": 2, "q5": 1, "q8": 1, "bf16": 1}}
    assert policy.detect(_MoE())["moe"] is True


@pytest.mark.parametrize("config", [
    {"num_experts": 128},
    {"text_config": {"n_routed_experts": 256}},
    {"num_local_experts": 8},
])
def test_config_declared_experts_mark_moe(config):
    assert policy.detect(_Dense(), config)["moe"] is True


def test_zero_experts_is_dense():
    assert policy.detect(_Dense(), {"num_experts": 0, "text_config": {}})["moe"] is False


def test_dense_defaults_come_from_the_builtin_table():
    resolved = policy.resolve(policy.detect(_Dense()))
    assert resolved["mode"] == "crossover"
    assert resolved["min_rows"]["q4"] == 8 and resolved["min_rows"]["bf16"] == 16
    assert resolved["sources"]["mode"] == "builtin"
    assert resolved["sources"]["min_rows.q4"] == "builtin"


def test_moe_defaults_off_and_cli_or_override_can_enable():
    detected = policy.detect(_MoE())
    off = policy.resolve(detected)
    assert off["mode"] == "off" and off["sources"]["mode"] == "builtin.moe"
    forced = policy.resolve(detected, mode="crossover")
    assert forced["mode"] == "crossover" and forced["sources"]["mode"] == "cli"
    opted = policy.resolve(detected, overrides={"moe": {"mode": "crossover", "min_rows": {"q4": 16}}})
    assert opted["mode"] == "crossover" and opted["min_rows"]["q4"] == 16
    assert opted["sources"]["min_rows.q4"] == "override.moe"
    explicit = policy.resolve(detected, overrides={"mode": "exact"})
    assert explicit["mode"] == "exact" and explicit["sources"]["mode"] == "override"


def test_family_defaults_then_overrides_win(monkeypatch):
    monkeypatch.setitem(policy.FAMILY_DEFAULTS, "fam", {"min_rows": {"q4": 6}, "max_rows": 16})
    detected = policy.detect(_Dense())
    fam = policy.resolve(detected, family="fam")
    assert fam["min_rows"]["q4"] == 6 and fam["max_rows"] == 16
    assert fam["sources"]["min_rows.q4"] == "family:fam"
    user = policy.resolve(detected, family="fam", overrides={"min_rows": {"q4": 4}})
    assert user["min_rows"]["q4"] == 4 and user["sources"]["min_rows.q4"] == "override"
    assert user["max_rows"] == 16


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"bogus": 1}, "unknown lane policy keys"),
        ({"mode": "fast"}, "mode must be"),
        ({"min_rows": {"q7": 4}}, "min_rows keys"),
        ({"min_rows": {"q4": 0}}, "positive integer"),
        ({"max_rows": 500}, "max_rows"),
        ({"grouping": "yes"}, "grouping"),
        ({"skip": "lm_head"}, "skip"),
        ({"moe": {"mode": "sometimes"}}, "mode must be"),
    ],
)
def test_invalid_overrides_fail_closed(override, message):
    with pytest.raises(ValueError, match=message):
        policy.resolve(policy.detect(_Dense()), overrides=override)


def test_overrides_load_inline_or_from_a_file(tmp_path):
    assert policy.load_overrides('{"max_rows": 16}') == {"max_rows": 16}
    path = tmp_path / "lane.json"
    path.write_text(json.dumps({"skip": ["lm_head"]}))
    assert policy.load_overrides(str(path)) == {"skip": ["lm_head"]}
    with pytest.raises(TypeError):
        policy.load_overrides("[1, 2]")


def test_apply_policy_sets_per_format_thresholds_and_skips(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)
    model = _Dense()
    resolved = policy.resolve(policy.detect(model), overrides={"skip": ["lm_head"]})
    receipt = lane.apply_policy(model, resolved)
    try:
        assert model.q_proj._lane_min_rows == 8 and model.down_proj._lane_min_rows == 8
        assert model.dense._lane_min_rows == 16
        assert not lane.installed(model.lm_head)
        assert receipt["refused"]["skipped"] == 1
        assert receipt["groups"] == {"affine-q4-g64x2": 1}
        assert "q4:8" in receipt["law_id"] and receipt["policy"]["sources"]["skip"] == "override"
    finally:
        lane.uninstall(model)
    exact = lane.apply_policy(model, policy.resolve(policy.detect(model), mode="exact"))
    try:
        assert model.q_proj._lane_min_rows == 1 and model.dense._lane_min_rows == 1
        assert exact["policy"]["mode"] == "exact"
    finally:
        lane.uninstall(model)
    assert lane.apply_policy(model, policy.resolve(policy.detect(model), mode="off")) is None


def test_unavailable_device_keeps_stock_modules_and_weights(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: False)
    model = _Dense()
    modules = (model.q_proj, model.k_proj)
    weights = (model.q_proj.weight, model.k_proj.weight)
    result = lane.apply_policy(model, policy.resolve(policy.detect(model)))
    assert result["available"] is False
    assert result["covered"] == {}
    assert result["refused"]["device_unsupported"] == 5
    assert model.q_proj is modules[0] and model.k_proj is modules[1]
    assert model.q_proj.weight is weights[0] and model.k_proj.weight is weights[1]
    assert not lane.installed(model.q_proj)


def test_call_counters_track_paths(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)
    model = _Dense()
    lane.apply_policy(model, policy.resolve(policy.detect(model), overrides={"min_rows": {"q4": 4}}))
    monkeypatch.setattr(installer, "available", lambda: True)
    monkeypatch.setattr(installer, "lane_matmul",
                        lambda x, lw: mx.zeros((*x.shape[:-1], lw.n), dtype=x.dtype))
    installer.STATS.clear()
    try:
        x8 = mx.zeros((8, 128), dtype=mx.bfloat16)
        model.q_proj(x8)
        model.k_proj(x8)                       # sibling: reuses the group launch
        model.q_proj(mx.zeros((2, 128), dtype=mx.bfloat16))
        model.q_proj(mx.zeros((40, 128), dtype=mx.bfloat16))
        counts = lane.stats()
        assert counts["lane_calls"] == 2 and counts["lane_rows"] == 16
        assert counts["lane_launches"] == 1 and counts["group_launches"] == 1
        assert counts["group_reuses"] == 1
        assert counts["stock_below_min_rows"] == 1 and counts["stock_above_max_rows"] == 1
        assert counts["rows_8-15"] == 2 and counts["rows_1-3"] == 1
    finally:
        installer.STATS.clear()
        lane.uninstall(model)


def test_status_and_metrics_expose_lane_state(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)
    from test_prometheus import FakeEngine, assert_valid_prometheus_text

    from mlx2.prometheus import render_engine_metrics
    from mlx2.serving import lane_matmul_status

    model = _Dense()
    engine = FakeEngine()
    engine.lane_matmul = "auto"
    engine.lane_matmul_receipt = lane.apply_policy(model, policy.resolve(policy.detect(model)))
    installer.STATS.clear()
    installer.STATS.update({"lane_calls": 5, "lane_rows": 40, "stock_below_min_rows": 7,
                            "rows_8-15": 5, "rows_1-3": 7, "lane_launches": 3,
                            "group_launches": 1, "group_reuses": 2})
    try:
        status = lane_matmul_status(engine)
        assert status["installed"] and status["mode"] == "crossover"
        assert status["covered"]["affine-q4-g64"] == 2 and status["counts"]["lane_calls"] == 5
        text = render_engine_metrics(engine)
        assert_valid_prometheus_text(text)
        assert 'mlx2_lane_matmul_enabled{mode="crossover"} 1' in text
        assert 'mlx2_lane_matmul_covered_projections{format="affine-q4-g64"} 2' in text
        assert 'mlx2_lane_matmul_calls_total{path="lane"} 5' in text
        assert 'mlx2_lane_matmul_calls_total{path="stock_below_min_rows"} 7' in text
        assert 'mlx2_lane_matmul_calls_by_rows_total{rows="8-15"} 5' in text
        assert "mlx2_lane_matmul_group_reuses_total 2" in text
    finally:
        installer.STATS.clear()
        lane.uninstall(model)
