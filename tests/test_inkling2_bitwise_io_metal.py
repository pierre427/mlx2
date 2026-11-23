"""Inkling 2 bit-identical attention I/O changes on Metal (GPU; run under the GPU lock).

- ``inkling2_sink_slot`` spare slot (``language.SINK_SLOT_SPARE``) against the
  reference concatenation, model level, bitwise.
- split-K ring read (``splitk.RING_DIRECT``): the kernels on a physical ring
  buffer with a start row against the same kernels on the reordered copy,
  bitwise, wrapped and unwrapped, at several start rows, both kernels (the
  packed ``part`` kernel and the L == 1 ``direct`` kernel past 2048 keys);
  and the model with the switch on and off.
- split-K direct-kernel 8-byte BF16 loads (``splitk.VECTOR_LOADS``) against
  the scalar loads, bitwise, on contiguous, cache-view, misaligned and
  odd-stride K/V and on a ring.

Set ``INKLING2_BITWISE_IO_METAL_RECEIPT`` to a path to get a JSON summary.
"""

import json
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

RESULTS = {}


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def teardown_module(_module):
    path = os.environ.get("INKLING2_BITWISE_IO_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


def _model(geometry, bf16, seed=0):
    from mlx2.runtime.models.inkling2 import Model, TextConfig
    from test_inkling2_cpu import TINY

    config = dict(TINY)
    if geometry == "wide":
        config.update(hidden_size=512, head_dim=128, swa_head_dim=128, d_rel=16,
                      intermediate_size=128,
                      loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=64))
    mx.random.seed(seed)
    model = Model(TextConfig.from_dict(config))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
        value = noise + (1.0 if key.endswith("norm.weight") else 0.0)
        weights.append((key, value.astype(mx.bfloat16) if bf16 else value))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _run(model, tokens, plan, cache=None):
    cache = model.make_cache() if cache is None else cache
    outs = []
    for a, b in plan:
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


def _gap(got, base):
    return float(mx.abs(got.astype(mx.float32) - base.astype(mx.float32)).max().item())


# ----- sink slot spare step -----


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("step", [8, 256])
def test_sink_slot_spare_bitwise_on_gpu(step, rows, bf16, geometry, monkeypatch):
    """Exact-fill chunks (one and two cache steps), decode across growths:
    the spare-step slot against the reference concatenation (sink slot off),
    with split-K and sink SDPA off so every global visit takes the slot."""
    from mlx2.runtime.models.cache import KVCache

    monkeypatch.setattr(KVCache, "step", step)
    model = _model(geometry, bf16)
    n = 4 * step + 6
    tokens = mx.random.randint(0, 500, (rows, n))
    plan = ([(0, step), (step, 2 * step), (2 * step, 4 * step)]
            + [(i, i + 1) for i in range(4 * step, n)])
    base = _run(model, tokens, plan)
    model.configure_kernels(sink_slot=True)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()
    model.configure_kernels()
    RESULTS[f"sink_slot/{geometry}/bf16={bf16}/rows={rows}/step={step}"] = {
        "bitwise": bool(mx.array_equal(got, base)), "max_abs": _gap(got, base),
        "grows": receipt["sink_slot_grows"], "calls": receipt["sink_slot_calls"],
        "fallback_reasons": receipt["sink_slot_fallback_reasons"]}
    assert receipt["sink_slot_grows"] > 0
    assert set(receipt["sink_slot_fallback_reasons"]) == {"cache_type"}
    assert mx.array_equal(got, base)


# ----- split-K ring read -----

H, HK, D, EXTENT = 24, 8, 128, 1024  # Inkling-2-Mini geometry


def _ring_inputs(b, length, width, dtype, seed, pad):
    mx.random.seed(seed)
    q = (mx.random.normal((b, length, H, D)) * 0.3).astype(dtype).transpose(0, 2, 1, 3)
    k = (mx.random.normal((b, HK, width, D)) * 0.3).astype(dtype)
    v = mx.random.normal((b, HK, width, D)).astype(dtype)
    rel = (mx.random.normal((b, length, H, EXTENT)) * 2).astype(dtype)
    sinks = (mx.random.normal((b, length, H)) * 2).astype(dtype)
    left = None if pad is None else mx.array(pad, dtype=mx.int32)
    return q, k, v, rel, sinks, left


RING_CASES = [
    # (width, size, window): a wrapped 1024 ring (Mini sliding layers, decode
    # and verify), a window-cropped read of a wider ring (size = window +
    # L - 1), an unwrapped partial ring, and a 4096 ring for the direct
    # kernel (L == 1 past 2048 keys, window 0).
    (1024, 1024, 1024), (1031, 1024 + 7, 1024), (1024, 600, 1024), (4096, 4096, 0),
    (4096, 3000, 0),
]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("length", [1, 2, 8, 16])
@pytest.mark.parametrize("width,size,window", RING_CASES)
def test_splitk_ring_read_bitwise_vs_reordered_copy(width, size, window, length, b, dtype):
    from mlx2.runtime.models.inkling2 import splitk

    if size < length:
        pytest.skip("fewer keys than queries")
    pad = None if b == 1 else [0, 37]
    q, k, v, rel, sinks, left = _ring_inputs(b, length, width, dtype, width + size + length, pad)
    kw = dict(offset=size - length, left_padding=left, window=window, scale=D ** -0.5)
    for base in sorted({0, 1, 7, 129, width // 2, width - size, width - 1}):
        if base < 0 or base >= width:
            continue
        ring = (base, size)
        with mx.stream(mx.gpu):
            kc, vc = splitk.ring_gather(k, ring), splitk.ring_gather(v, ring)
            mx.eval(kc, vc)
            got = splitk.gqa_splitk_attention(q, k, v, rel, sinks, ring=ring, **kw)
            ref = splitk.gqa_splitk_attention(q, kc, vc, rel, sinks, **kw)
            mx.eval(got, ref)
        key = f"ring/{dtype}/b={b}/L={length}/w={width}/s={size}/win={window}/base={base}"
        RESULTS[key] = {"bitwise": bool(mx.array_equal(got, ref)), "max_abs": _gap(got, ref)}
        assert mx.array_equal(got, ref), key


def test_splitk_ring_read_bitwise_vs_cropped_view():
    """The model's crop of a chronological copy is a strided view (base
    pointer offset); the ring read of the same keys matches it bitwise."""
    from mlx2.runtime.models.inkling2 import splitk

    width, length, window = 1024, 8, 1024
    q, k, v, rel, sinks, _ = _ring_inputs(1, length, width, mx.bfloat16, 5, None)
    for rot in (0, 3, 517, 1023):
        chrono_k = splitk.ring_gather(k, (rot, width))
        chrono_v = splitk.ring_gather(v, (rot, width))
        for start in (0, 1, 9):
            kw = dict(offset=width - start - length, left_padding=None, window=window,
                      scale=D ** -0.5)
            with mx.stream(mx.gpu):
                ref = splitk.gqa_splitk_attention(q, chrono_k[:, :, start:],
                                                  chrono_v[:, :, start:], rel, sinks, **kw)
                got = splitk.gqa_splitk_attention(
                    q, k, v, rel, sinks, ring=((rot + start) % width, width - start), **kw)
                mx.eval(got, ref)
            RESULTS[f"ring_view/rot={rot}/start={start}"] = bool(mx.array_equal(got, ref))
            assert mx.array_equal(got, ref), (rot, start)


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", ["b1", "b2", "b2_padded"])
def test_ring_direct_model_bitwise_on_gpu(rows, bf16, geometry, monkeypatch):
    """Split-K on: the sliding visits read the wrapped rings in place
    (``RING_DIRECT``) vs the reordered copy (switch off), logits bitwise
    over prefill 9, verify 8 and decode past the wrap."""
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache
    from mlx2.runtime.models.inkling2 import splitk

    model = _model(geometry, bf16)
    b = 1 if rows == "b1" else 2
    tokens = mx.random.randint(0, 500, (b, 40))
    plan = [(0, 9), (9, 17)] + [(i, i + 1) for i in range(17, 40)]

    def cache():
        planes = model.make_cache()
        if rows == "b2_padded":
            for i in range(0, len(planes), 2):
                planes[i] = (BatchKVCache([0, 3]) if type(planes[i]) is KVCache
                             else BatchRotatingKVCache(8, [0, 3]))
        return planes

    model.configure_kernels(verify_splitk=True)
    try:
        monkeypatch.setattr(splitk, "RING_DIRECT", False)
        base = _run(model, tokens, plan, cache())
        monkeypatch.setattr(splitk, "RING_DIRECT", True)
        got = _run(model, tokens, plan, cache())
        receipt = model.kernel_receipt()
    finally:
        model.configure_kernels()
    RESULTS[f"ring_model/{geometry}/bf16={bf16}/{rows}"] = {
        "bitwise": bool(mx.array_equal(got, base)), "max_abs": _gap(got, base),
        "ring": receipt["verify_splitk_ring"], "declines": receipt["verify_splitk_declines"]}
    assert receipt["verify_splitk_ring"].get("calls", 0) == 40 * len(plan)
    assert "copies" not in receipt["verify_splitk_ring"]
    assert mx.array_equal(got, base)


# ----- split-K direct kernel: vector vs scalar loads -----


def _direct_layout(layout, size, seed):
    mx.random.seed(seed)
    if layout == "contiguous":
        k = (mx.random.normal((1, HK, size, D)) * 0.3).astype(mx.bfloat16)
        v = mx.random.normal((1, HK, size, D)).astype(mx.bfloat16)
    elif layout == "cache_view":
        k = (mx.random.normal((1, HK, size + 200, D)) * 0.3).astype(mx.bfloat16)[:, :, :size]
        v = mx.random.normal((1, HK, size + 200, D)).astype(mx.bfloat16)[:, :, :size]
    elif layout == "misaligned_base":
        # Rows 8-byte misaligned (2 elements off): the guard picks scalar
        # loads in both arms.
        k = (mx.random.normal((1, HK, size, D + 8)) * 0.3).astype(mx.bfloat16)[..., 2:D + 2]
        v = mx.random.normal((1, HK, size, D + 8)).astype(mx.bfloat16)[..., 2:D + 2]
    elif layout == "row_aligned_view":
        # Base 8 bytes off (4 elements): 8-byte loads still legal.
        k = (mx.random.normal((1, HK, size, D + 4)) * 0.3).astype(mx.bfloat16)[..., 4:D + 4]
        v = mx.random.normal((1, HK, size, D + 4)).astype(mx.bfloat16)[..., 4:D + 4]
    else:  # odd_stride
        k = (mx.random.normal((1, HK, size, D + 2)) * 0.3).astype(mx.bfloat16)[..., 1:D + 1]
        v = mx.random.normal((1, HK, size, D + 2)).astype(mx.bfloat16)[..., :D]
    q = (mx.random.normal((1, 1, H, D)) * 0.3).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    rel = (mx.random.normal((1, 1, H, EXTENT)) * 2).astype(mx.bfloat16)
    sinks = (mx.random.normal((1, 1, H)) * 2).astype(mx.bfloat16)
    return q, k, v, rel, sinks


@pytest.mark.parametrize("layout", ["contiguous", "cache_view", "misaligned_base",
                                    "row_aligned_view", "odd_stride"])
@pytest.mark.parametrize("size", [2048, 5000, 40960])
def test_direct_vector_loads_bitwise_vs_scalar(size, layout, monkeypatch):
    """L == 1 past 2048 keys (the direct kernel): 8-byte loads vs the scalar
    loads, the same bits; also through a ring read."""
    from mlx2.runtime.models.inkling2 import splitk

    q, k, v, rel, sinks = _direct_layout(layout, size, seed=size)
    kw = dict(offset=size - 1, left_padding=None, window=0, scale=D ** -0.5)
    out = {}
    for vec in (False, True):
        monkeypatch.setattr(splitk, "VECTOR_LOADS", vec)
        with mx.stream(mx.gpu):
            out[vec] = splitk.gqa_splitk_attention(q, k, v, rel, sinks, **kw)
            mx.eval(out[vec])
    RESULTS[f"vec/{layout}/{size}"] = {"bitwise": bool(mx.array_equal(out[True], out[False])),
                                       "max_abs": _gap(out[True], out[False])}
    assert mx.array_equal(out[True], out[False])
    if layout == "contiguous":
        ring = (size // 3, size)
        for vec in (False, True):
            monkeypatch.setattr(splitk, "VECTOR_LOADS", vec)
            with mx.stream(mx.gpu):
                out[vec] = splitk.gqa_splitk_attention(q, k, v, rel, sinks, ring=ring, **kw)
                mx.eval(out[vec])
        assert mx.array_equal(out[True], out[False])


@pytest.mark.parametrize("bf16", [False, True])
def test_direct_vector_loads_model_bitwise(bf16, monkeypatch):
    """Wide model (D = 128), split-K on, decode past 2048 global keys so the
    direct kernel runs on the global visits: vector vs scalar loads bitwise."""
    from mlx2.runtime.models.inkling2 import splitk

    model = _model("wide", bf16)
    tokens = mx.random.randint(0, 500, (1, 2060))
    plan = [(0, 1024), (1024, 2048)] + [(i, i + 1) for i in range(2048, 2060)]
    model.configure_kernels(verify_splitk=True)
    try:
        monkeypatch.setattr(splitk, "VECTOR_LOADS", False)
        base = _run(model, tokens, plan)
        monkeypatch.setattr(splitk, "VECTOR_LOADS", True)
        got = _run(model, tokens, plan)
    finally:
        model.configure_kernels()
    RESULTS[f"vec_model/bf16={bf16}"] = {"bitwise": bool(mx.array_equal(got, base)),
                                         "max_abs": _gap(got, base)}
    assert mx.array_equal(got, base)
