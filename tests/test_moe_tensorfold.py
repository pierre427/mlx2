import gc

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.runtime.models import moe_tensorfold, switch_layers
from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

DIMS, HIDDEN, EXPERTS = 64, 64, 8

# (label, quantize kwargs or None, input dims, dtype)
LAYOUTS = [
    ("dense-f32", None, DIMS, mx.float32),
    ("affine-q4-g32", {"group_size": 32, "bits": 4}, DIMS, mx.bfloat16),
    ("affine-q4-g64", {"group_size": 64, "bits": 4}, DIMS, mx.bfloat16),
    ("affine-q8-g64", {"group_size": 64, "bits": 8}, DIMS, mx.float16),
    ("mxfp4-g32", {"group_size": 32, "bits": 4, "mode": "mxfp4"}, DIMS, mx.bfloat16),
    # K % 32 != 0 takes the nvfp4 dequantize + gather_mm tail policy.  The
    # CPU gather_mm (stock or coalesced) is float32-only, hence the dtypes.
    ("nvfp4-g16-dense-tail", {"group_size": 16, "bits": 4, "mode": "nvfp4"}, 48, mx.float32),
]


class Model(nn.Module):
    def __init__(self, *, dims=DIMS, quant=None, dtype=mx.float32, bias=False, layers=1):
        super().__init__()
        self.layers = [SwitchGLU(dims, HIDDEN, EXPERTS, bias=bias) for _ in range(layers)]
        for block in self.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(block, name)
                proj.update({k: v.astype(dtype) for k, v in proj.parameters().items()})
                if quant is not None:
                    setattr(block, name, proj.to_quantized(**quant))
        self.eval()

    def __call__(self, x, indices):
        for block in self.layers:
            x = block(x, indices)
        return x

    @property
    def experts(self):
        return self.layers[0]


def _np(a):
    return np.array(a, copy=False)


def _routing(batch, top_k, seed=0):
    rng = np.random.default_rng(seed)
    rows = [rng.choice(EXPERTS, size=top_k, replace=False) for _ in range(batch)]
    return mx.array(np.stack(rows).astype(np.uint32))


@pytest.fixture(autouse=True)
def cpu_and_clean_stats():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    moe_tensorfold.ENABLED[0] = True
    moe_tensorfold.STATS.clear()
    try:
        yield
    finally:
        mx.set_default_device(previous)
        moe_tensorfold.ENABLED[0] = True
        moe_tensorfold.STATS.clear()


@pytest.mark.parametrize("label,quant,dims,dtype", LAYOUTS, ids=[l[0] for l in LAYOUTS])
@pytest.mark.parametrize("batch,top_k", [(1, 4), (2, 3), (5, 4), (12, 8)])
def test_gate_up_coalescing_matches_stock_bitwise(label, quant, dims, dtype, batch, top_k):
    mx.random.seed(7)
    model = Model(dims=dims, quant=quant, dtype=dtype)
    x = mx.random.normal((batch, dims)).astype(dtype)
    indices = _routing(batch, top_k)
    expected = model(x, indices)
    mx.eval(expected)

    receipt = moe_tensorfold.install(model)
    actual = model(x, indices)
    mx.eval(actual)

    assert receipt["installed"] == 1 and receipt["selected"] is True
    assert receipt["qualified"] is False and receipt["refused"] == {}
    assert list(receipt["covered"]) == [model.experts.__dict__["_tensorfold_gate_up"].format]
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual, expected).item()
    sorted_call = batch * top_k >= switch_layers._GATHER_SORT_MIN_ASSIGNMENTS
    assert moe_tensorfold.stats() == {
        "calls": 1,
        "assignments": batch * top_k,
        "sorted_calls" if sorted_call else "unsorted_calls": 1,
    }


def test_linear_bias_and_leading_batch_axes():
    mx.random.seed(3)
    model = Model(bias=True)
    for block in model.layers:
        for name in ("gate_proj", "up_proj"):
            proj = getattr(block, name)
            proj.bias = mx.random.normal(proj.bias.shape)
    x = mx.random.normal((2, 3, DIMS))
    indices = mx.array(np.random.default_rng(1).integers(0, EXPERTS, (2, 3, 2)).astype(np.uint32))
    for idx in (indices, mx.concatenate([indices] * 5, axis=-1) % EXPERTS):
        expected = model(x, idx)
        moe_tensorfold.install(model)
        actual = model(x, idx)
        assert mx.array_equal(actual, expected).item()
        moe_tensorfold.uninstall(model)


def test_views_are_contiguous_zero_copy_and_nothing_is_resident_twice():
    mx.random.seed(5)
    model = Model(quant={"group_size": 64, "bits": 4}, dtype=mx.float16, layers=4)
    mx.eval(model.parameters())
    gc.collect()
    before = mx.get_active_memory()
    moe_tensorfold.install(model)
    gc.collect()
    after = mx.get_active_memory()
    bank = sum(v.nbytes for block in model.layers
               for proj in (block.gate_proj, block.up_proj) for v in proj.parameters().values())
    assert after - before < bank // 8

    block = model.experts
    group = block.__dict__["_tensorfold_gate_up"]
    for name in ("weight", "scales", "biases"):
        table = _np(group.projection[name])
        for half in (block.gate_proj, block.up_proj):
            view = _np(half[name])
            assert np.shares_memory(table, view)
            assert view.flags["C_CONTIGUOUS"]
        assert table.shape[0] == 2 * EXPERTS


def test_parameter_tree_is_unchanged():
    mx.random.seed(11)
    model = Model(quant={"group_size": 32, "bits": 4})
    before = {k: (v.shape, v.dtype) for k, v in tree_flatten(model.parameters())}
    values = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    trainable = [k for k, _ in tree_flatten(model.trainable_parameters())]
    moe_tensorfold.install(model)
    after = {k: (v.shape, v.dtype) for k, v in tree_flatten(model.parameters())}
    assert after == before
    assert [k for k, _ in tree_flatten(model.trainable_parameters())] == trainable
    for k, v in tree_flatten(model.parameters()):
        assert np.array_equal(np.array(v), values[k])
    assert not any(k for k, _ in model.named_modules() if "tensorfold" in k)


def test_disable_is_stock_and_uninstall_restores_independent_banks():
    mx.random.seed(9)
    model = Model()
    x = mx.random.normal((3, DIMS))
    indices = mx.array([[0, 1], [2, 3], [4, 5]], dtype=mx.uint32)
    expected = model(x, indices)
    mx.eval(expected)

    moe_tensorfold.install(model)
    table = _np(model.experts.__dict__["_tensorfold_gate_up"].projection["weight"])
    moe_tensorfold.set_enabled(False)
    assert mx.array_equal(model(x, indices), expected).item()
    assert moe_tensorfold.stats() == {}
    moe_tensorfold.set_enabled(True)

    assert moe_tensorfold.uninstall(model) == 1
    assert model.experts.__dict__.get("_tensorfold_gate_up") is None
    for half in (model.experts.gate_proj, model.experts.up_proj):
        assert not np.shares_memory(table, _np(half["weight"]))
    assert mx.array_equal(model(x, indices), expected).item()
    assert moe_tensorfold.stats() == {}
    assert moe_tensorfold.uninstall(model) == 0


def test_repeat_install_is_idempotent():
    model = Model(layers=2)
    first = moe_tensorfold.install(model)
    group = model.experts.__dict__["_tensorfold_gate_up"]
    second = moe_tensorfold.install(model)
    assert first["covered"] == {"dense-float32": 2}
    assert second["covered"] == {"already": 2} and second["installed"] == 2
    assert model.experts.__dict__["_tensorfold_gate_up"] is group


def test_rebound_weights_fall_back_to_stock_and_drop_the_table():
    mx.random.seed(13)
    model = Model()
    x = mx.random.normal((2, DIMS))
    indices = mx.array([[0, 7], [3, 4]], dtype=mx.uint32)
    moe_tensorfold.install(model)
    fresh = {k: mx.random.normal(v.shape) for k, v in tree_flatten(model.parameters())}
    model.load_weights(list(fresh.items()))
    reference = Model()
    reference.load_weights(list(fresh.items()))

    assert mx.array_equal(model(x, indices), reference(x, indices)).item()
    assert moe_tensorfold.stats() == {"stale_dropped": 1}
    assert model.experts.__dict__.get("_tensorfold_gate_up") is None

    receipt = moe_tensorfold.install(model)
    assert receipt["covered"] == {"dense-float32": 1}
    assert mx.array_equal(model(x, indices), reference(x, indices)).item()


def test_quantize_after_install_is_detected():
    mx.random.seed(17)
    model = Model()
    x = mx.random.normal((2, DIMS))
    indices = mx.array([[1, 2], [5, 6]], dtype=mx.uint32)
    moe_tensorfold.install(model)
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda _p, m: hasattr(m, "to_quantized"))
    assert type(model.experts.gate_proj) is QuantizedSwitchLinear
    expected = model(x, indices)  # first call drops the stale dense table
    assert moe_tensorfold.stats() == {"stale_dropped": 1}
    moe_tensorfold.install(model)
    assert mx.array_equal(model(x, indices), expected).item()
    assert moe_tensorfold.stats()["calls"] == 1


def test_training_uses_the_parameter_tree():
    mx.random.seed(19)
    model = Model()
    x = mx.random.normal((2, DIMS))
    indices = mx.array([[0, 1], [2, 3]], dtype=mx.uint32)

    def loss(m):
        return m(x, indices).sum()

    stock = nn.value_and_grad(model, loss)(model)[1]
    moe_tensorfold.install(model)
    model.train()
    coalesced = nn.value_and_grad(model, loss)(model)[1]
    gate = coalesced["layers"][0]["gate_proj"]["weight"]
    assert mx.abs(gate).sum().item() > 0
    assert mx.array_equal(gate, stock["layers"][0]["gate_proj"]["weight"]).item()
    assert moe_tensorfold.stats().get("calls", 0) == 0
    assert moe_tensorfold.stats()["fallback_training"] >= 1


def test_doubled_rows_stay_out_of_the_sorted_tail_window(monkeypatch):
    mx.random.seed(23)
    model = Model(quant={"group_size": 64, "bits": 4}, dtype=mx.bfloat16)
    x = mx.random.normal((5, DIMS)).astype(mx.bfloat16)
    indices = _routing(5, 4)  # 20 sorted rows; stock stays unpadded below 32
    monkeypatch.setattr(switch_layers, "_SORTED_GATHER_TAIL_ROWS", 32)
    expected = model(x, indices)
    mx.eval(expected)
    moe_tensorfold.install(model)
    shapes = []
    original = QuantizedSwitchLinear.__call__

    def spy(self, x, idx, sorted_indices=False):
        shapes.append(tuple(idx.shape))
        return original(self, x, idx, sorted_indices=sorted_indices)

    monkeypatch.setattr(QuantizedSwitchLinear, "__call__", spy)
    actual = model(x, indices)
    assert mx.array_equal(actual, expected).item()
    assert shapes == [(2, 32), (20,)]  # padded coalesced gate/up, then stock down
    assert moe_tensorfold.stats()["tail_padded_calls"] == 1


def test_quantized_group_declines_sorted_gather_qmm(monkeypatch):
    model = Model(quant={"group_size": 64, "bits": 4}, dtype=mx.bfloat16)
    moe_tensorfold.install(model)
    seen = []
    original = QuantizedSwitchLinear.__call__

    def spy(self, x, idx, sorted_indices=False):
        seen.append(sorted_indices)
        return original(self, x, idx, sorted_indices=sorted_indices)

    monkeypatch.setattr(QuantizedSwitchLinear, "__call__", spy)
    x = mx.zeros((3, DIMS), dtype=mx.bfloat16)
    indices = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]] * 3)
    mx.eval(model(x, indices))
    # Coalesced gate/up is the first call; down_proj keeps its stock policy.
    assert seen == [False, True]


def _pair(model):
    return model.experts.gate_proj, model.experts.up_proj


def test_mismatched_projection_types_fail_closed():
    model = Model()
    model.experts.up_proj = model.experts.up_proj.to_quantized(32, 4)
    receipt = moe_tensorfold.install(model)
    assert receipt["covered"] == {} and receipt["installed"] == 0
    assert receipt["selected"] is False
    assert receipt["refused"] == {"gate/up projection types differ": 1}
    assert model.experts.__dict__.get("_tensorfold_gate_up") is None


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: setattr(m.experts.up_proj, "weight", m.experts.up_proj.weight.astype(mx.float16)),
     "gate/up weight dtypes differ"),
    (lambda m: setattr(m.experts, "up_proj", m.experts.up_proj.to_quantized(32, 8))
     or setattr(m.experts, "gate_proj", m.experts.gate_proj.to_quantized(32, 4)),
     "gate/up quantization bits differs"),
    (lambda m: setattr(m.experts.gate_proj, "bias", mx.zeros((EXPERTS, HIDDEN))),
     "gate/up parameter sets differ"),
    (lambda m: [setattr(p, "lora_a", mx.zeros((4, DIMS))) for p in _pair(m)],
     "gate/up carry state outside the SwitchLinear contract"),
    (lambda m: setattr(m.experts, "up_proj", nn.Linear(DIMS, HIDDEN))
     or setattr(m.experts, "gate_proj", nn.Linear(DIMS, HIDDEN)),
     "gate/up are not generic SwitchLinear projections"),
])
def test_structural_refusals(mutate, reason):
    model = Model()
    mutate(model)
    weights = {k: v for k, v in tree_flatten(model.parameters())}
    receipt = moe_tensorfold.install(model)
    assert receipt["refused"] == {reason: 1} and receipt["installed"] == 0
    after = {k: v for k, v in tree_flatten(model.parameters())}
    assert all(after[k] is v for k, v in weights.items())


def test_subclass_is_refused():
    class Custom(SwitchGLU):
        pass

    model = Model()
    model.layers[0] = Custom(DIMS, HIDDEN, EXPERTS)
    receipt = moe_tensorfold.install(model)
    assert receipt["refused"] == {"SwitchGLU subclass owns its forward": 1}


def test_apc_fingerprint_binds_the_law_only_when_installed():
    assert moe_tensorfold.apc_fingerprint("base", None) == "base"
    assert moe_tensorfold.apc_fingerprint("base", {"installed": 0, "law_id": "x"}) == "base"
    receipt = moe_tensorfold.install(Model())
    assert moe_tensorfold.apc_fingerprint("base", receipt) == (
        "base", "moe-gate-up-tensorfold", moe_tensorfold.LAW_ID)
