"""CPU tests for the lane matmul: format coverage, decode, geometry, install.

The Metal kernels themselves run only on an M5 GPU (scripts/lane_matmul_gate.py);
here the generated decode is executed in Python against MLX's own
dequantization, and installation is checked to fall back to stock on CPU.
"""

import re

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import matmul as lm


def _quantized(k, n, bits, gs, seed=0):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    module = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
    module.weight, module.scales, module.biases = mx.quantize(w, group_size=gs, bits=bits)
    return module


def _reference_ints(module):
    """Integer weights recovered from MLX's own dequantization."""
    wq, s, b = module.weight, module.scales, module.biases
    deq = mx.dequantize(wq, s, b, group_size=module.group_size, bits=module.bits)
    s_full = mx.repeat(s, module.group_size, axis=1).astype(mx.float32)
    b_full = mx.repeat(b, module.group_size, axis=1).astype(mx.float32)
    safe = mx.where(s_full == 0, 1, s_full)
    return np.rint(np.asarray((deq.astype(mx.float32) - b_full) / safe)).astype(np.int64)


def _run_generated_decode(source, words):
    """Execute the generated ``dst[q] = ...`` lines with Python integers."""
    env = {f"w{i}": int(w) for i, w in enumerate(words)}
    out = {}
    for q, expr in re.findall(r"dst\[(\d+)\] = (.+?);", source):
        out[int(q)] = eval(expr.replace("u)", ")").replace("u ", " "), {}, env) & 0xFFFFFFFF
    return [out[q] for q in sorted(out)]


@pytest.mark.parametrize("bits", [2, 3, 5, 6])
@pytest.mark.parametrize("gs", [32, 64, 128])
def test_generated_decode_matches_mlx_packing(bits, gs):
    k, n = 2 * gs, 8
    module = _quantized(k, n, bits, gs)
    ints = _reference_ints(module)
    packed = np.asarray(module.weight).astype(np.uint32)
    words_per_group = gs * bits // 32
    source = lm._unpack_group_source(bits, gs)
    for row in range(n):
        for g in range(k // gs):
            words = packed[row, g * words_per_group:(g + 1) * words_per_group]
            decoded = _run_generated_decode(source, words)
            values = [(word >> (8 * e)) & 0xFF for word in decoded for e in range(4)]
            assert values == list(ints[row, g * gs:(g + 1) * gs])


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8, 16])
def test_every_format_has_a_complete_kernel_source(bits):
    for gs in ((64,) if bits == 16 else lm.GROUP_SIZES):
        source = lm.main_source(bits, gs)
        assert "@@" not in source
        assert ("s[f][j]" in source) == (bits != 16)


def test_split_k_depends_only_on_shape_and_format():
    assert lm.split_k(5120, 17408, 64, 4) == lm.split_k(5120, 17408, 64, 4)
    for bits in (2, 3, 5, 6):
        for gs in lm.GROUP_SIZES:
            sk = lm.split_k(5120, 17408, gs, bits)
            assert sk * lm.NT * gs <= 16 * 1024
    assert lm.split_k(17408, 5120, 64, 4) in (1, 2, 4, 8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bits": 7, "group_size": 64, "mode": "affine", "n": 64, "k": 128, "weight_dtype": mx.uint32,
              "scales_dtype": mx.bfloat16}, "7-bit"),
        ({"bits": 4, "group_size": 48, "mode": "affine", "n": 64, "k": 96, "weight_dtype": mx.uint32,
              "scales_dtype": mx.bfloat16}, "group size"),
        ({"bits": 4, "group_size": 32, "mode": "mxfp4", "n": 64, "k": 128, "weight_dtype": mx.uint32,
              "scales_dtype": mx.bfloat16}, "not affine"),
        ({"bits": 4, "group_size": 64, "mode": "affine", "n": 66, "k": 128, "weight_dtype": mx.uint32,
              "scales_dtype": mx.bfloat16}, "multiple of 4"),
        ({"bits": 16, "group_size": 64, "mode": "none", "n": 64, "k": 100, "weight_dtype": mx.bfloat16},
         "multiple of 64"),
        ({"bits": 16, "group_size": 64, "mode": "none", "n": 64, "k": 128, "weight_dtype": mx.float32},
         "bf16 or fp16"),
    ],
)
def test_check_geometry_refuses_unsupported(kwargs, message):
    with pytest.raises(lane.LaneUnsupported, match=message):
        lm.check_geometry(**kwargs)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_prepare_binds_scale_bias_pairs(bits):
    module = _quantized(256, 64, bits, 64)
    lw = lane.prepare(module)
    assert (lw.bits, lw.group_size, lw.n, lw.k) == (bits, 64, 64, 256)
    assert lw.scale_bias.shape == (4, 64, 2)
    assert mx.array_equal(lw.scale_bias[..., 0], module.scales.T)
    assert mx.array_equal(lw.scale_bias[..., 1], module.biases.T)
    assert lw.weight is module.weight


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q4 = _quantized(128, 64, 4, 64, seed=1)
        self.q5 = _quantized(128, 64, 5, 32, seed=2)
        self.dense = nn.Linear(128, 64, bias=True)
        self.dense.weight = self.dense.weight.astype(mx.bfloat16)
        self.dense.bias = self.dense.bias.astype(mx.bfloat16)
        self.odd = nn.Linear(100, 64)          # K not a multiple of 64: stays stock


def test_install_covers_formats_falls_back_on_cpu_and_uninstalls():
    model = _Tiny()
    x = mx.random.normal((3, 128), key=mx.random.key(3)).astype(mx.bfloat16)
    before = [model.q4(x), model.q5(x), model.dense(x)]
    receipt = lane.install(model, max_rows=16)
    assert receipt["covered"] == {"affine-q4-g64": 1, "affine-q5-g32": 1, "unquantized": 1}
    assert sum(receipt["refused"].values()) == 1
    assert receipt["available"] is False
    assert lane.installed(model.q4) and lane.installed(model.dense)
    assert not lane.installed(model.odd)
    after = [model.q4(x), model.q5(x), model.dense(x)]
    for old, new in zip(before, after, strict=True):
        assert mx.array_equal(old, new)          # CPU: stock arithmetic unchanged
    assert lane.install(model)["covered"] == {"already": 3}
    assert lane.uninstall(model) == 3
    assert type(model.q4) is nn.QuantizedLinear and type(model.dense) is nn.Linear


def test_lane_matmul_refuses_bad_calls():
    lw = lane.prepare(_quantized(128, 64, 4, 64))
    with pytest.raises(lane.LaneUnsupported, match="width"):
        lane.lane_matmul(mx.zeros((1, 64), dtype=mx.bfloat16), lw)
    with pytest.raises(lane.LaneUnsupported, match="rows"):
        lane.lane_matmul(mx.zeros((lane.MAX_ROWS + 1, 128), dtype=mx.bfloat16), lw)
    with pytest.raises(lane.LaneUnsupported, match="bf16 or fp16"):
        lane.lane_matmul(mx.zeros((1, 128), dtype=mx.float32), lw)
