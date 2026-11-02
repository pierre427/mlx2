"""AST verify levers in glm53_layer_tree: compiled tree bodies, fused tree
routing and the serial HC law for tree rows.

CPU tests cover TreeRegions' warm-up/trace/key/release mechanics. The Metal
tests (MLX2_RUN_GPU_TESTS=1) check bit for bit that hc_in_rows equals each
row's one-token Glm53GGUFModel._hc_in under glm53_hc=mega_v1 and that the
fused tree router equals the per-row op chain.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import glm53_layer_tree as tree
from mlx2.runtime.models import glm53_compile_regions as regions
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
DIM = 4096
metal = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                           reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def bits(a):
    return np.array(a).view(np.uint32)


def _policy_model(**extra):
    weights = SimpleNamespace(scheduled_dense_calls={}, scheduled_grouped_calls={},
                              scheduled_expert_calls={})
    return SimpleNamespace(weights=weights, math_policy="ordinary", hc_mega=False, hc_fused=False,
                           args=SimpleNamespace(rms_norm_eps=1e-5), calls={}, **extra)


def test_tree_regions_warm_only_after_success_key_and_release():
    model = _policy_model(tree_router="reference")
    regs = tree.TreeRegions(model)
    calls = []

    def failing():
        def body(x):
            raise RuntimeError("interrupted")
        return body

    with pytest.raises(RuntimeError):
        regs.call("kda", 0, [-1], failing, mx.zeros((1,)))
    assert ("kda", 0, (-1,)) not in regs.warm  # a failed warm-up stays cold

    def build():
        def body(x):
            calls.append(1)
            return x * 2
        return body

    x = mx.array([1.0, 2.0])
    assert np.array_equal(np.array(regs.call("kda", 0, [-1], build, x)), [2.0, 4.0])
    assert ("kda", 0, (-1,)) in regs.warm and regs.counters["tree_compile_warmups"] == 1
    assert np.array_equal(np.array(regs.call("kda", 0, [-1], build, x)), [2.0, 4.0])
    assert regs.counters["tree_compile_traces"] == 1
    # The tree router is part of the key: a change drops every region.
    model.tree_router = "fused_v1"
    regs.call("kda", 0, [-1], build, x)
    assert regs.regions == {} and regs.counters["tree_compile_warmups"] == 2
    model.__dict__["_tree_regions"] = regs
    regions.release(model)
    assert "_tree_regions" not in model.__dict__


def test_tree_regions_decline_without_policy():
    model = _policy_model(tree_compile="off", layers=[])
    model.hc_fused = True
    assert tree.tree_regions(model, []) is None


class _HcWeights:
    def __init__(self, rng, prefix):
        self.values = {
            prefix + "_fn": mx.array(rng.normal(0, 0.02, (24, 4 * DIM)).astype(np.float32)),
            prefix + "_scale": mx.array(rng.normal(1, 0.1, (3,)).astype(np.float32)),
            prefix + "_base": mx.array(rng.normal(0, 0.1, (24,)).astype(np.float32)),
            "norm": mx.array(rng.normal(1, 0.1, (DIM,)).astype(np.float32)),
        }
        self.tensors = {prefix + "_fn": SimpleNamespace(ggml_type=0)}

    def array(self, name):
        return self.values[name]

    def linear(self, name):
        return lambda x: x @ self.values[name].T


@metal
def test_hc_in_rows_equals_each_rows_serial_mega_hc_in(gpu):
    rng = np.random.default_rng(5)
    prefix = "blk.1.hc_attn"
    model = object.__new__(Glm53GGUFModel)
    model.weights = _HcWeights(rng, prefix)
    model.args = SimpleNamespace(rms_norm_eps=1e-5)
    model.hc_mega, model.hc_fused, model.hc_fused_policy = True, True, "reference"
    model.calls = {}
    # Four distinct HC streams per row (the case the old tree law got wrong).
    h = mx.array(rng.normal(0, 1, (1, 3, 4, DIM)).astype(np.float32))
    x, post, comb = tree.hc_in_rows(model, h, prefix, "norm", 1e-5)
    for row in range(3):
        sx, spost, scomb = model._hc_in(h[:, row : row + 1], prefix, "norm")
        assert np.array_equal(bits(x[:, row : row + 1]), bits(sx))
        assert np.array_equal(bits(post[:, row : row + 1]), bits(spost))
        assert np.array_equal(bits(comb[:, row : row + 1]), bits(scomb))


@metal
def test_fused_tree_routing_equals_the_per_row_op_chain(gpu):
    from mlx2.runtime.models.precise_ops import gate_sigmoid

    rng = np.random.default_rng(9)
    prefix = "blk.4"
    weights = SimpleNamespace(
        values={prefix + ".ffn_gate_inp.weight": mx.array(rng.normal(0, 0.05, (144, 64)).astype(np.float32)),
                prefix + ".exp_probs_b.bias": mx.array(rng.normal(0, 0.01, (144,)).astype(np.float32))},
        tensors={prefix + ".ffn_gate_inp.weight": SimpleNamespace(ggml_type=0)})
    weights.array = lambda name: weights.values[name]
    weights.linear = lambda name: (lambda x: x @ weights.values[name].T)
    model = SimpleNamespace(weights=weights, calls={})
    x = mx.array(rng.normal(0, 1, (1, 4, 64)).astype(np.float32))
    ids, scores = tree._fused_rows_routing(model, x, prefix)
    for row in range(4):
        p = gate_sigmoid(weights.linear(prefix + ".ffn_gate_inp.weight")(x[:, row : row + 1]))
        chosen = mx.argsort(-(p + weights.array(prefix + ".exp_probs_b.bias")), axis=-1)[..., :8]
        w = mx.take_along_axis(p, chosen.astype(mx.uint32), axis=-1)
        w = w / w.sum(-1, keepdims=True) * 2.5
        assert np.array_equal(np.array(ids[:, row : row + 1]), np.array(chosen.astype(mx.uint32)))
        assert np.array_equal(bits(scores[:, row : row + 1]), bits(w))


def _kq(shape, rng, ggml_type=12):
    """A random Q4_K (or Q3_K = 11) packed matrix; scales may be any half."""
    from mlx2.runtime.gguf.k_blocks import SIZES

    count, block = SIZES[ggml_type]
    row_bytes = shape[-1] // count * block
    rows = int(np.prod(shape[:-1]))
    raw = rng.integers(0, 256, rows * row_bytes, dtype=np.uint8)
    # Keep every fp16 scale finite and small: clear the high exponent bits.
    raw[1::2] &= 0x3B
    return SimpleNamespace(shape=shape, ggml_type=ggml_type, row_bytes=row_bytes,
                           data=mx.array(raw))


@metal
@pytest.mark.parametrize("tokens", [2, 3, 5])
def test_union_fold_equals_the_shared_fold_bit_for_bit(gpu, tokens):
    from mlx2.runtime.gguf.kblock_metal import kblock_moe_shared
    from mlx2.runtime.gguf.kblock_union_shared import kblock_moe_union_shared

    rng = np.random.default_rng(tokens)
    experts, m, k = 6, 256, 512
    gate, up = _kq((experts, m, k), rng), _kq((experts, m, k), rng)
    down = _kq((experts, k, m), rng)
    gate_sh, up_sh, down_sh = _kq((m, k), rng), _kq((m, k), rng), _kq((k, m), rng)
    x = mx.array(rng.normal(0, 1, (tokens, k)).astype(np.float32))
    ids = mx.array(np.stack([rng.choice(experts, 4, replace=False) for _ in range(tokens)]).astype(np.uint32))
    scores = mx.array(rng.uniform(0, 1, (tokens, 4)).astype(np.float32))
    want = kblock_moe_shared(x, ids, scores, gate, up, down, gate_sh, up_sh, down_sh)
    got = kblock_moe_union_shared(x, ids, scores, gate, up, down, gate_sh, up_sh, down_sh)
    assert np.array_equal(bits(got), bits(want))
    for row in range(tokens):
        one = kblock_moe_shared(x[row:row + 1], ids[row:row + 1], scores[row:row + 1],
                                gate, up, down, gate_sh, up_sh, down_sh)
        assert np.array_equal(bits(got[row:row + 1]), bits(one))


def test_adapter_refuses_tree_levers_that_cannot_engage():
    from mlx2.adapters.glm53_gguf import _check_tree_levers

    base = {"glm53_ast_target": "layered", "glm53_shared_fold": "kblock", "glm53_row_kernel": "kblock_rows",
            "glm53_tree_batch": "kblock_v1", "glm53_hc": "mega_v1", "glm53_tree_moe": "union_fold_v1",
            "glm53_tree_compile": "v1", "glm53_tree_router": "fused_v1"}
    _check_tree_levers(base)
    for key, value in (("glm53_ast_target", "serial"), ("glm53_shared_fold", "off"),
                       ("glm53_row_kernel", "reference"), ("glm53_tree_batch", "off")):
        with pytest.raises(ValueError):
            _check_tree_levers({**base, key: value})
    with pytest.raises(ValueError):
        _check_tree_levers({**base, "glm53_tree_moe": "fold", "glm53_hc": "reference"})
