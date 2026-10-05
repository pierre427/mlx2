"""CPU tests for adapter-declared lane projection groups.

A declared group stacks projections that one ``__call__`` feeds the same
input tensor object.  The static test pins that property at each call site;
the tiny-model tests show the stacked launch engaging (one launch, every
other member a reuse, no partial launches) with outputs matching stock.  The
lane kernel itself is replaced by a dequantize-and-matmul reference here, as
in test_lane_matmul.py; Metal timing and parity need the GPU gate.
"""

import ast
import inspect
import json
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import installer, policy
from mlx2.runtime.lane.installer import ProjectionGroup
from mlx2.runtime.models import muse_glimmer, olmo_hils, xing4_0

MODULES = (muse_glimmer, olmo_hils, xing4_0)
FIXTURE = Path(__file__).parent / "fixtures" / "xing4_0_tiny"


@pytest.fixture(autouse=True)
def _cpu_lane(monkeypatch):
    monkeypatch.setattr(installer, "available", lambda: True)

    def reference(x, lw):
        if lw.bits == installer.UNQUANTIZED_BITS:
            y = x @ lw.weight.T
        else:
            w = mx.dequantize(lw.weight, lw.scale_bias[..., 0].T, lw.scale_bias[..., 1].T,
                              group_size=lw.group_size, bits=lw.bits)
            y = (x.astype(mx.float32) @ w.T.astype(mx.float32)).astype(x.dtype)
        return y if lw.bias is None else y + lw.bias

    monkeypatch.setattr(installer, "lane_matmul", reference)
    installer.STATS.clear()
    yield
    installer.STATS.clear()
    lane.set_enabled(True)


# -- static: every member reads the same, never-rebound input -----------------

def _self_path(node):
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return node.attr
        base = _self_path(node.value)
        return None if base is None else f"{base}.{node.attr}"
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and type(node.slice.value) is int:
        base = _self_path(node.value)
        return None if base is None else f"{base}.{node.slice.value}"
    return None


def _declared():
    return [spec for module in MODULES for spec in module.lane_projection_groups()]


def _check_call_site(spec):
    assert "__call__" in spec.parent.__dict__, "the call site must be the class's own"
    fn = ast.parse(textwrap.dedent(inspect.getsource(spec.parent.__call__))).body[0]
    params = {a.arg for a in fn.args.args[1:]}
    inputs = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and _self_path(node.func) in spec.members:
            assert not node.keywords and len(node.args) == 1, ast.unparse(node)
            assert isinstance(node.args[0], ast.Name), ast.unparse(node)
            inputs.setdefault(_self_path(node.func), set()).add(node.args[0].id)
    assert set(inputs) == set(spec.members), "every member is called"
    names = set().union(*inputs.values())
    assert len(names) == 1 and names <= params, inputs
    (name,) = names
    rebinds = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name
               and not isinstance(n.ctx, ast.Load)]
    assert not rebinds, f"{name} is rebound before or between member calls"


@pytest.mark.parametrize("spec", _declared(), ids=lambda spec: spec.name)
def test_declared_call_site_feeds_one_unrebound_input(spec):
    _check_call_site(spec)


class _GateReadsOutput(nn.Module):
    def __call__(self, x):
        out = self.q_proj(x) + self.k_proj(x)
        return out * self.gate_proj(out)


class _InputRebound(nn.Module):
    def __call__(self, x):
        q = self.q_proj(x)
        x = x * 2
        return q + self.gate_proj(x)


@pytest.mark.parametrize("parent", [_GateReadsOutput, _InputRebound])
def test_the_call_site_check_rejects_a_group_that_does_not_share_its_input(parent):
    with pytest.raises(AssertionError):
        _check_call_site(ProjectionGroup("bad", parent, ("q_proj", "gate_proj")))


def test_adapters_offer_their_model_groups_without_loading():
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.adapters.olmo_hils import OlmoHiLSAdapter
    from mlx2.adapters.xing import XingAdapter

    for adapter, module in ((MuseGlimmerAdapter, muse_glimmer),
                            (OlmoHiLSAdapter, olmo_hils), (XingAdapter, xing4_0)):
        assert adapter.lane_projection_groups() == module.lane_projection_groups()


# -- tiny real models ---------------------------------------------------------


def test_default_sibling_group_skips_array_attributes():
    class MixedSiblings(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(128, 64, bias=False)
            self.up_proj = mx.array([1])

    model = MixedSiblings()
    array = model.up_proj
    assert not hasattr(array, "__dict__")
    receipt = installer.install(model, min_rows=1)
    assert receipt["groups"] == {}
    assert model.up_proj is array
    installer.uninstall(model)

def _muse(bits=4):
    args = muse_glimmer.ModelArgs(hidden_size=128, num_hidden_layers=2, intermediate_size=128,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                                  vocab_size=64, sliding_window=16)
    mx.random.seed(11)
    model = muse_glimmer.Model(args)
    model.set_dtype(mx.bfloat16)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model


def _hils(bits=6):
    args = olmo_hils.ModelArgs(model_type="olmo_hils", hidden_size=128, num_hidden_layers=4,
                               intermediate_size=128, num_attention_heads=2, rms_norm_eps=1e-6,
                               vocab_size=64, max_position_embeddings=512, sliding_window=16,
                               rope_theta=10000.0, chunk_size=8, hils_topk=2, lmk_q_lora_dim=64)
    mx.random.seed(12)
    model = olmo_hils.Model(args)
    model.set_dtype(mx.bfloat16)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model


def _xing():
    model = xing4_0.Model(xing4_0.ModelArgs.from_dict(
        json.loads((FIXTURE / "config.json").read_text())))
    weights = model.sanitize(mx.load(str(FIXTURE / "weights.safetensors")))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    # CPU gather_mm (the experts) is float32-only: only the projections the
    # group covers take bf16, the lane's unquantized format.
    for _name, module in model.named_modules():
        if type(module) is xing4_0.Xing4_0Attention:
            module.q_a_proj.set_dtype(mx.bfloat16)
            module.kv_a_proj_with_mqa.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _stock_then_lane(model, run, declared):
    """Stock output, then the lane with default groups, then with ``declared``.

    Returns (stock, default-grouped lane output, receipt, declared output,
    counters of the declared run).  Grouping only regroups columns of the
    same per-member arithmetic, so the two lane outputs must agree closely.
    """
    stock = run(model)
    lane.install(model, min_rows=1)
    baseline = run(model)
    receipt = lane.install(model, min_rows=1, declared=declared)
    installer.STATS.clear()
    lane_out = run(model)
    return stock, baseline, receipt, lane_out, lane.stats()


def _close(a, b, tol=2e-2):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return bool(mx.all(mx.abs(a - b) <= tol * (1 + mx.abs(b))))


def _events(counts, name):
    return tuple(counts.get(f"declared_{e}:{name}", 0) for e in ("launches", "reuses", "partial"))


def test_muse_output_gate_stacks_with_qkv_and_reuses_one_launch():
    model = _muse()
    tokens = mx.array([[1, 5, 9, 13, 17, 21]])
    stock, baseline, receipt, out, counts = _stock_then_lane(
        model, lambda m: m(tokens), muse_glimmer.lane_projection_groups())
    entry = receipt["declared_groups"]["muse-attn-qkv-gate"]
    assert entry["formed"] == {"affine-q4-g64": 2} and entry["refused"] == {}
    assert receipt["groups"]["affine-q4-g64x4"] == 2              # q/k/v/gate per layer
    assert receipt["groups"]["affine-q4-g64x2"] == 2              # MLP gate/up unchanged
    assert "+declared[muse-attn-qkv-gate@" in receipt["law_id"]
    assert _events(counts, "muse-attn-qkv-gate") == (2, 6, 0)
    attn = model.model.layers[0].self_attn
    assert installer._group(attn.gate_proj) is installer._group(attn.q_proj)
    assert installer._group(model.model.layers[0].mlp.gate_proj) is not installer._group(attn.q_proj)
    assert _close(out, baseline)
    lane.set_enabled(False)
    assert mx.array_equal(model(tokens), stock)                   # stock law untouched
    lane.uninstall(model)


def test_hils_landmark_down_projection_joins_only_hils_layers():
    model = _hils()
    tokens = mx.array([[3, 1, 4, 1, 5]])
    _stock, baseline, receipt, out, counts = _stock_then_lane(
        model, lambda m: m(tokens), olmo_hils.lane_projection_groups())
    hils = [layer for layer in model.model.layers if layer.is_hils]
    swa = [layer for layer in model.model.layers if not layer.is_hils]
    assert hils and swa
    entry = receipt["declared_groups"]["hils-attn-qkv-lmkq"]
    assert entry["formed"] == {"affine-q6-g64": len(hils)} and entry["refused"] == {}
    for layer in hils:
        attn = layer.self_attn
        group = installer._group(attn.q_proj)
        assert installer._group(attn.lmk_q_proj[0]) is group and group.size == 4
        assert installer._group(attn.lmk_q_proj[1]) is None       # reads the rank-64 result
    for layer in swa:
        assert installer._group(layer.self_attn.q_proj).declared is None
        assert installer._group(layer.self_attn.q_proj).size == 3
    assert _events(counts, "hils-attn-qkv-lmkq") == (len(hils), 3 * len(hils), 0)
    assert _close(out, baseline)
    lane.uninstall(model)


def test_xing_query_down_projection_pairs_with_latent_kv():
    model = _xing()
    tokens = mx.array([[2, 7, 1, 8, 2, 8]])
    _stock, baseline, receipt, out, counts = _stock_then_lane(
        model, lambda m: m(tokens), xing4_0.lane_projection_groups())
    layers = model.model.layers
    # The trunk layers plus the fixture's MTP layer, which has the same attention.
    attentions = sum(type(m) is xing4_0.Xing4_0Attention for _n, m in model.named_modules())
    assert attentions == len(layers) + 1
    formed = receipt["declared_groups"]["xing-mla-qa-kva"]
    assert formed["formed"] == {"unquantized": attentions} and formed["refused"] == {}
    # Every Xing4.0 artifact has a query LoRA: the q_proj variant never forms.
    assert receipt["declared_groups"]["xing-mla-q-kva"]["formed"] == {}
    assert receipt["declared_groups"]["xing-mla-q-kva"]["refused"] == {"member missing": attentions}
    attn = layers[0].self_attn
    assert installer._group(attn.q_a_proj) is installer._group(attn.kv_a_proj_with_mqa)
    assert "+declared[xing-mla-qa-kva@" in receipt["law_id"]
    assert _events(counts, "xing-mla-qa-kva") == (len(layers), len(layers), 0)
    assert _close(out, baseline)
    lane.uninstall(model)


def test_xing_full_query_variant_forms_without_a_query_lora():
    config = json.loads((FIXTURE / "config.json").read_text())
    config["q_lora_rank"] = None
    attn = xing4_0.Xing4_0Attention(xing4_0.ModelArgs.from_dict(config))
    attn.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 5, config["hidden_size"]), key=mx.random.key(4)).astype(mx.bfloat16)
    _stock, baseline, receipt, out, counts = _stock_then_lane(
        attn, lambda m: m(x), xing4_0.lane_projection_groups())
    assert receipt["declared_groups"]["xing-mla-q-kva"]["formed"] == {"unquantized": 1}
    assert _events(counts, "xing-mla-q-kva") == (1, 1, 0)
    assert _close(out, baseline)


# -- fail-closed behaviour and identity ---------------------------------------

def _q(k, n, bits, seed):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    module = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
    module.weight, module.scales, module.biases = mx.quantize(w, group_size=64, bits=bits)
    return module


class _Gated(nn.Module):
    def __init__(self, gate_bits=4):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (_q(128, 64, 4, s) for s in (1, 2, 3))
        self.gate_proj = _q(128, 64, gate_bits, 4)

    def __call__(self, x, gate_input=None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return q + k + v, self.gate_proj(x if gate_input is None else gate_input)


GATED = ProjectionGroup("test-qkv-gate", _Gated, ("q_proj", "k_proj", "v_proj", "gate_proj"))


def test_off_by_default_keeps_the_law_and_groups_byte_identical():
    resolved = policy.resolve(policy.detect(_Gated()))
    assert resolved["declared_groups"] is False
    assert resolved["sources"]["declared_groups"] == "builtin"
    plain = lane.apply_policy(_Gated(), resolved)
    offered = lane.apply_policy(_Gated(), resolved, declared=(GATED,))
    assert offered["law_id"] == plain["law_id"] and offered["groups"] == plain["groups"]
    assert "declared" not in offered["law_id"]
    assert "declared_groups" not in plain
    assert offered["declared_groups"]["test-qkv-gate"]["refused"] == {"not selected": 1}
    selected = lane.apply_policy(
        _Gated(), policy.resolve(policy.detect(_Gated()), overrides={"declared_groups": True}),
        declared=(GATED,))
    assert selected["law_id"] != plain["law_id"]
    assert selected["policy"]["declared_groups"] is True
    assert selected["groups"] == {"affine-q4-g64x4": 1}


def test_mixed_formats_refuse_the_whole_group_and_fall_back_to_defaults():
    model = _Gated(gate_bits=8)
    receipt = lane.install(model, min_rows=1, declared=(GATED,))
    assert receipt["declared_groups"]["test-qkv-gate"]["refused"] == {"mixed formats": 1}
    assert receipt["groups"] == {"affine-q4-g64x3": 1}            # the default q/k/v group
    assert "declared" not in receipt["law_id"]
    assert installer._group(model.gate_proj) is None


def test_selected_but_undeliverable_fails_closed_and_restores_stock():
    selected = policy.resolve(policy.detect(_Gated()), overrides={"declared_groups": True})
    with pytest.raises(ValueError, match="declares none"):
        lane.apply_policy(_Gated(), selected)
    model = _Gated(gate_bits=8)
    with pytest.raises(ValueError, match="none formed"):
        lane.apply_policy(model, selected, declared=(GATED,))
    assert type(model.q_proj) is nn.QuantizedLinear and not lane.installed(model.q_proj)
    with pytest.raises(ValueError, match="requires grouping"):
        policy.resolve(policy.detect(_Gated()),
                       overrides={"declared_groups": True, "grouping": False})
    with pytest.raises(ValueError, match="boolean"):
        policy.resolve(policy.detect(_Gated()), overrides={"declared_groups": 1})
    with pytest.raises(ValueError, match="require grouping"):
        lane.install(_Gated(), declared=(GATED,), groups=())
    with pytest.raises(ValueError, match="distinct"):
        ProjectionGroup("bad", _Gated, ("q_proj", "q_proj"))


def test_a_member_fed_another_tensor_is_counted_partial_and_stays_correct():
    model = _Gated()
    x = mx.random.normal((1, 4, 128), key=mx.random.key(9)).astype(mx.bfloat16)
    other = x + 0                                                 # equal values, new object
    lane.install(model, min_rows=1)
    baseline = model(x, other)
    lane.install(model, min_rows=1, declared=(GATED,))
    installer.STATS.clear()
    first = model(x, other)
    model(x)                                                      # next call: detects it
    # q launches (k, v reuse); gate's other tensor relaunches, so that first
    # launch fed 3 of 4 members; the next call's launch shows gate's fed 1.
    assert _events(lane.stats(), "test-qkv-gate") == (3, 5, 2)
    for got, want in zip(first, baseline, strict=True):
        assert _close(got, want)


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("declaration,groups,error", [
    ((object(),), installer.DEFAULT_GROUPS, TypeError),
    ((GATED, GATED), installer.DEFAULT_GROUPS, ValueError),
    ((GATED,), (), ValueError),
])
def test_invalid_declarations_preserve_existing_projection_topology(installed, declaration, groups, error):
    model = _Gated()
    if installed:
        lane.install(model, min_rows=1, declared=(GATED,))
    before = [(type(module), installer._group(module), installer._prepared(module))
              for module in (model.q_proj, model.k_proj, model.v_proj, model.gate_proj)]
    with pytest.raises(error):
        lane.install(model, min_rows=4, declared=declaration, groups=groups)
    for module, (kind, group, prepared) in zip(
        (model.q_proj, model.k_proj, model.v_proj, model.gate_proj), before, strict=True
    ):
        assert type(module) is kind
        assert installer._group(module) is group
        assert installer._prepared(module) is prepared
        if installed:
            assert module._lane_min_rows == 1


def test_the_declared_spec_is_part_of_the_law_and_reinstall_dissolves_it():
    renamed = ProjectionGroup("test-qkv-gate", _Gated, ("gate_proj", "q_proj", "k_proj", "v_proj"))
    model = _Gated()
    first = lane.install(model, min_rows=1, declared=(GATED,))
    again = lane.install(model, min_rows=1, declared=(GATED,))
    assert again["law_id"] == first["law_id"]
    assert again["declared_groups"]["test-qkv-gate"]["formed"] == {"affine-q4-g64": 1}
    assert lane.install(_Gated(), min_rows=1, declared=(renamed,))["law_id"] != first["law_id"]
    plain = lane.install(model, min_rows=1)
    assert plain["law_id"] == "lane-matmul-v1+grouped"
    assert installer._group(model.gate_proj) is None
    assert installer._group(model.q_proj).declared is None


def test_status_and_metrics_expose_declared_groups():
    from test_prometheus import FakeEngine, assert_valid_prometheus_text

    from mlx2.prometheus import render_engine_metrics
    from mlx2.serving import lane_matmul_status

    model = _Gated()
    engine = FakeEngine()
    engine.lane_matmul = "auto"
    engine.lane_matmul_receipt = lane.apply_policy(
        model, policy.resolve(policy.detect(model), overrides={"declared_groups": True}),
        declared=(GATED,))
    x = mx.zeros((1, 8, 128), dtype=mx.bfloat16)
    installer.STATS.clear()
    model(x)
    status = lane_matmul_status(engine)
    assert status["declared_groups"]["test-qkv-gate"]["formed"] == {"affine-q4-g64": 1}
    text = render_engine_metrics(engine)
    assert_valid_prometheus_text(text)
    assert ('mlx2_lane_matmul_declared_group_events_total'
            '{event="launches",group="test-qkv-gate"} 1') in text
    assert ('mlx2_lane_matmul_declared_group_events_total'
            '{event="reuses",group="test-qkv-gate"} 3') in text
