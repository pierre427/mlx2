"""Fused elementwise regions in the Ripple forward: bit-identical to eager.

CPU evidence only; Metal JIT kernels may round transcendentals differently,
so scripts/research/ripple_fused_gate.py repeats the comparison on the GPU
before the option is used there.
"""
import re
from collections import Counter

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import ricursive
from test_ripple_memcopy_cpu import assert_bitwise, build


def run(model, chunks, decode, *, batch=1):
    cache = model.make_cache()
    tokens = mx.array([[(3 * i + 1) % 31 + 1 for i in range(sum(chunks))]] * batch)
    outputs, start = [], 0
    for size in chunks:
        outputs.append(model(tokens[:, start:start + size], cache=cache))
        start += size
    step = tokens[:, -1:]
    for _ in range(decode):
        logits = model(step, cache=cache)
        outputs.append(logits)
        step = mx.argmax(logits[:, -1:], axis=-1)
    mx.eval(outputs)
    return outputs


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("chunks", [[13], [3, 5, 1, 4], [7, 6]])
def test_fused_regions_match_eager_bitwise(dtype, nvfp4, chunks):
    eager = build(dtype, nvfp4)
    fused = build(dtype, nvfp4)
    fused.set_fused_elementwise(True)
    for actual, expected in zip(run(fused, chunks, 6), run(eager, chunks, 6), strict=True):
        assert_bitwise(actual, expected)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_fused_regions_match_eager_bitwise_batched(dtype):
    eager = build(dtype, False)
    fused = build(dtype, False)
    fused.set_fused_elementwise(True)
    for actual, expected in zip(run(fused, [5, 4], 5, batch=3), run(eager, [5, 4], 5, batch=3), strict=True):
        assert_bitwise(actual, expected)


def test_toggle_reaches_every_layer_and_back():
    model = build(mx.bfloat16, False)
    model.set_fused_elementwise(True)
    flags = [m.fused for _, m in model.named_modules()
             if isinstance(m, (ricursive.RMSNorm, ricursive.Attention, ricursive.MLP, ricursive.DecoderLayer))]
    assert flags and all(flags) and model.fused_elementwise
    model.set_fused_elementwise(False)
    assert not any(m.fused for _, m in model.named_modules() if hasattr(m, "fused"))


def decode_graph_ops(model, path):
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5, 6]]), cache=cache)
    mx.eval([c.state for c in cache])
    out = model(mx.array([[7]]), cache=cache)
    mx.export_to_dot(str(path), out)
    views = {"Broadcast", "Reshape", "Transpose", "Slice", "Flatten", "Unflatten",
             "ExpandDims", "Squeeze", "AsStrided"}
    labels = Counter(re.findall(r'label ="([^"]+)"', path.read_text()))
    return Counter({k: v for k, v in labels.items() if k not in views})


def test_fused_decode_graph_has_far_fewer_nodes(tmp_path):
    eager = build(mx.bfloat16, False)
    fused = build(mx.bfloat16, False)
    fused.set_fused_elementwise(True)
    before = decode_graph_ops(eager, tmp_path / "eager.dot")
    after = decode_graph_ops(fused, tmp_path / "fused.dot")
    # Compute kernels (views excluded): 563 -> 372 for this 5-layer model.
    assert any(name.startswith("Compiled") for name in after)
    assert sum(after.values()) < 0.7 * sum(before.values())
    # Transcendentals stay outside fused regions (precise rounding).
    assert not any(name.startswith("Compiled") and ("Rsqrt" in name or "Sigmoid" in name) for name in after)
