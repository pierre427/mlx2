"""Inkling 2 opt-in prefill/head/compile options on Metal (GPU; run under the
GPU lock).

Random tiny decoders at two geometries: the CPU test's (hidden 128, head 32)
and a wider one closer to the Mini's kernel shapes (hidden 512, head 128, 6
query heads over 2 KV heads). Each option is checked off (bitwise reference)
and on (bounded gap, bitwise recorded) over chunked prefill + decode, B1 and
B2 with left padding, FP32 and BF16, with its engagement counter. Set
``INKLING2_NONBITWISE_METAL_RECEIPT`` to a path to get a JSON summary.
"""

import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {}
PLAN = [(0, 20), (20, 60)] + [(i, i + 1) for i in range(60, 70)]
LONG_PLAN = [(0, 300), (300, 812)] + [(i, i + 1) for i in range(812, 816)]


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
    path = os.environ.get("INKLING2_NONBITWISE_METAL_RECEIPT")
    if path:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
        receipt = {"mlx2": head, "mlx_version": mx.__version__,
                   "device": mx.metal.device_info().get("device_name"),
                   "checkpoint": "random tiny decoders (no artifact)", "results": RESULTS}
        with open(path, "w") as handle:
            json.dump(receipt, handle, indent=1, sort_keys=True)


def _model(geometry, bf16, seed=0):
    from test_inkling2_cpu import TINY

    from mlx2.runtime.models.inkling2 import Model, TextConfig

    config = dict(TINY)
    if geometry == "wide":
        config.update(hidden_size=512, head_dim=128, swa_head_dim=128, d_rel=16,
                      num_attention_heads=6, num_key_value_heads=2,
                      swa_num_attention_heads=6, swa_num_key_value_heads=2,
                      intermediate_size=128, rel_extent=64,
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


def _run(model, tokens, plan, pads=None):
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

    cache = model.make_cache()
    if pads is not None:
        window = model.config.sliding_window_size
        for i in range(0, len(cache), 2):
            cache[i] = (BatchKVCache(list(pads)) if type(cache[i]) is KVCache
                        else BatchRotatingKVCache(window, list(pads)))
    outs = []
    for a, b in plan:
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


def _gap(got, base):
    diff = mx.abs(got.astype(mx.float32) - base.astype(mx.float32))
    return {"max_abs": float(diff.max().item()),
            "argmax_equal": float((mx.argmax(got, -1) == mx.argmax(base, -1)).mean().item()),
            "bitwise": bool(mx.array_equal(got, base))}


def _ab(model, tokens, plan, pads, option_kwargs, base_kwargs=None):
    base_kwargs = base_kwargs or {}
    model.configure_kernels(**base_kwargs)
    base = _run(model, tokens, plan, pads)
    model.configure_kernels(**base_kwargs, **option_kwargs)
    got = _run(model, tokens, plan, pads)
    receipt = model.kernel_receipt()
    model.configure_kernels(**base_kwargs)
    again = _run(model, tokens, plan, pads)
    return base, got, receipt, bool(mx.array_equal(again, base))


SHAPES = [("tiny", False, None), ("tiny", True, None), ("tiny", True, (0, 5)),
          ("wide", False, None), ("wide", True, None), ("wide", True, (0, 5))]


def _shape_id(shape):
    geometry, bf16, pads = shape
    return f"{geometry}-{'bf16' if bf16 else 'fp32'}-{'b2pad' if pads else 'b1'}"


# ----- inkling2_prefill_mask_bf16_budget (item 1a) -----


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_mask_bf16_budget(monkeypatch, shape):
    from mlx2.runtime.models.inkling2 import language

    geometry, bf16, pads = shape
    model = _model(geometry, bf16)
    tokens = mx.random.randint(0, 500, (2 if pads else 1, 70), key=mx.random.key(1))
    monkeypatch.setattr(language, "_MASK_BUDGET_BYTES", 1 << 14)  # several slices
    base, got, receipt, off_exact = _ab(model, tokens, PLAN, pads,
                                        {"prefill_mask_bf16_budget": True})
    result = {**_gap(got, base), "calls": receipt["prefill_mask_bf16_budget_calls"]}
    RESULTS[f"mask_bf16_budget/{_shape_id(shape)}"] = result
    assert off_exact
    assert (result["calls"] > 0) == bf16
    assert result["max_abs"] < (0.5 if bf16 else 1e-5), result


# ----- inkling2_prefill_mask_with_sink (item 1b) -----


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_mask_with_sink(monkeypatch, shape, fused):
    """Expected bitwise: the mask values are the reference's copies and the
    SDPA call shapes are the same."""
    from mlx2.runtime.models.inkling2 import language

    geometry, bf16, pads = shape
    model = _model(geometry, bf16)
    tokens = mx.random.randint(0, 500, (2 if pads else 1, 70), key=mx.random.key(2))
    monkeypatch.setattr(language, "_MASK_BUDGET_BYTES", 1 << 14)
    base, got, receipt, off_exact = _ab(model, tokens, PLAN, pads,
                                        {"prefill_mask_with_sink": True},
                                        {"fuse_attention": fused})
    result = {**_gap(got, base), "calls": receipt["prefill_mask_with_sink_calls"]}
    RESULTS[f"mask_with_sink/{'fused' if fused else 'plain'}/{_shape_id(shape)}"] = result
    assert off_exact and result["calls"] > 0
    assert result["bitwise"], result


# ----- inkling2_head_kernel_wide (item 2) -----


@pytest.mark.parametrize("rows,n,k", [(1, 1000, 128), (3, 4097, 2304), (16, 513, 640),
                                      (20, 2048, 4096 + 128)])
def test_head_gemv_wide_is_bitwise_narrow(rows, n, k):
    from mlx2.runtime.models.inkling2.language import head_gemv

    mx.random.seed(rows + n)
    x = mx.random.normal((rows, k)).astype(mx.bfloat16)
    w = (mx.random.normal((n + 7, k)) * 0.05).astype(mx.bfloat16)
    narrow = head_gemv(x, w[:n])
    wide = head_gemv(x, w[:n], wide=True)
    ref = x.astype(mx.float32) @ w[:n].astype(mx.float32).T
    result = {**_gap(wide, narrow), "max_abs_vs_fp32": float(mx.abs(wide - ref).max().item())}
    RESULTS[f"head_wide/gemv/{rows}x{n}x{k}"] = result
    assert result["bitwise"], result


@pytest.mark.parametrize("geometry", ["tiny", "wide"])
def test_head_kernel_wide_model_level(geometry):
    model = _model(geometry, True)
    tokens = mx.random.randint(0, 500, (2, 70), key=mx.random.key(3))
    base, got, receipt, off_exact = _ab(model, tokens, PLAN, None,
                                        {"head_kernel_wide": True}, {"head_kernel": True})
    result = {**_gap(got, base), "calls": receipt["head_kernel_wide_calls"]}
    RESULTS[f"head_wide/model/{geometry}"] = result
    assert off_exact and result["calls"] > 0
    assert result["bitwise"], result


# ----- inkling2_prefill_tiled_attention (item 1c) -----


def _attention_case(seed, *, b, hq, hkv, length, size, dim, extent, window, pads, dtype,
                    capacity=None):
    mx.random.seed(seed)

    def unit(shape):
        x = mx.random.normal(shape)
        return x / mx.linalg.norm(x, axis=-1, keepdims=True) * 4.0

    # q strided as the model's (projected [B, L, H, D], transposed).
    q = unit((b, length, hq, dim)).astype(dtype).transpose(0, 2, 1, 3)
    # K/V as a cache buffer view (capacity past the fill), as KVCache hands them.
    cap = capacity or size
    k = unit((b, hkv, cap, dim)).astype(dtype)[:, :, :size]
    v = mx.random.normal((b, hkv, cap, dim)).astype(dtype)[:, :, :size]
    rel = (mx.random.normal((b, length, hq, extent)) * 0.5).astype(dtype)
    sinks = mx.random.normal((b, length, hq)).astype(dtype)
    pad = None if pads is None else mx.array(list(pads))
    kw = dict(offset=size - length, left_padding=pad, window=window, scale=dim ** -0.5)
    return (q, k, v, rel, sinks), kw


TILED_CASES = [
    dict(b=1, hq=24, hkv=8, length=512, size=512, dim=128, extent=1024, window=0, pads=None),
    dict(b=1, hq=24, hkv=8, length=512, size=4096, dim=128, extent=1024, window=0, pads=None,
         capacity=4352),
    dict(b=2, hq=24, hkv=8, length=300, size=2348, dim=128, extent=1024, window=0,
         pads=(0, 517)),
    dict(b=1, hq=24, hkv=8, length=37, size=1500, dim=128, extent=64, window=0, pads=None),
    dict(b=1, hq=24, hkv=8, length=512, size=1535, dim=128, extent=1024, window=1024,
         pads=None),
    dict(b=2, hq=6, hkv=2, length=45, size=90, dim=32, extent=16, window=8, pads=(0, 11)),
    dict(b=1, hq=8, hkv=1, length=100, size=200, dim=64, extent=16, window=0, pads=None),
    dict(b=1, hq=4, hkv=4, length=2, size=9, dim=32, extent=8, window=0, pads=None),
]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("case", range(len(TILED_CASES)))
def test_tiled_kernel_matches_composed_reference(case, dtype):
    from mlx2.runtime.models.inkling2 import splitk
    from mlx2.runtime.models.inkling2.prefill_attention import (
        decline_reason,
        tiled_prefill_attention,
        tiled_reference,
    )

    spec = TILED_CASES[case]
    args, kw = _attention_case(case, dtype=dtype, **spec)
    q, k, v = args[:3]
    assert decline_reason(q, k, v, n_heads=spec["hq"], n_kv=spec["hkv"],
                          head_dim=spec["dim"]) is None
    got = tiled_prefill_attention(*args, **kw)
    ref = splitk.reference_attention(*args, **kw)
    law = tiled_reference(*args, **kw)
    mx.eval(got, ref, law)
    result = {"vs_composed": _gap(got, ref)["max_abs"], "vs_law": _gap(got, law)["max_abs"],
              "law_vs_composed": _gap(law, ref)["max_abs"],
              "nan": bool(mx.isnan(got).any().item())}
    RESULTS[f"tiled/kernel/{case}/{dtype}"] = {**spec, "pads": spec["pads"], **result}
    assert not result["nan"]
    bound = 1e-4 if dtype == mx.float32 else 3e-2
    assert result["vs_composed"] < bound and result["vs_law"] < bound, result


def test_tiled_kernel_scalar_path_matches_vector_path():
    """A K/V view whose rows are not 16-byte aligned takes the in-kernel
    scalar loads: same values as the aligned vector loads."""
    from mlx2.runtime.models.inkling2.prefill_attention import tiled_prefill_attention

    args, kw = _attention_case(9, b=1, hq=6, hkv=2, length=64, size=300, dim=128,
                               extent=64, window=0, pads=None, dtype=mx.bfloat16)
    q, k, v, rel, sinks = args
    k_odd = mx.concatenate([mx.zeros_like(k[:, :, :1]), k], axis=2)[:, :, 1:]
    v_odd = mx.concatenate([mx.zeros_like(v[:, :, :1]), v], axis=2)[:, :, 1:]
    a = tiled_prefill_attention(q, k, v, rel, sinks, **kw)
    b = tiled_prefill_attention(q, k_odd, v_odd, rel, sinks, **kw)
    RESULTS["tiled/scalar_vs_vector"] = _gap(a, b)
    assert mx.array_equal(a, b)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_tiled_model_level(shape):
    """Prefill chunks of 300 and 512 rows plus decode; split-K off so short
    blocks stay on the composed path. A chaos control (the composed law in
    FP32, rounded once) bounds the BF16 gap."""
    geometry, bf16, pads = shape
    model = _model(geometry, bf16)
    tokens = mx.random.randint(0, 500, (2 if pads else 1, 816), key=mx.random.key(4))
    base, got, receipt, off_exact = _ab(model, tokens, LONG_PLAN, pads,
                                        {"prefill_tiled_attention": True})
    result = {**_gap(got, base), "calls": receipt["prefill_tiled_calls"],
              "declines": receipt["prefill_tiled_declines"]}
    RESULTS[f"tiled/model/{_shape_id(shape)}"] = result
    assert off_exact and result["calls"] == 8 * 2 and result["declines"] == {}
    if not bf16:
        assert result["max_abs"] < 1e-3, result
    else:
        assert result["argmax_equal"] > 0.5 and result["max_abs"] < 5.0, result


def _composed_sliced(q, k, v, rel, sinks, offset, scale):
    """The composed path for one global visit: 64 MiB budget counted at 4 B per
    element, BF16 mask slices, zero sink K/V column, one SDPA per slice."""
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    zeros = mx.zeros((b, hkv, 1, dim), k.dtype)
    kk, vv = mx.concatenate([k, zeros], 2), mx.concatenate([v, zeros], 2)
    rows = max(1, (64 << 20) // (b * hq * (size + 1) * 4))
    extent = rel.shape[-1]
    outs = []
    for a in range(0, length, rows):
        e = min(length, a + rows)
        d = (mx.arange(a, e)[:, None] + offset) - mx.arange(size)[None]
        m = mx.take_along_axis(rel[:, a:e].transpose(0, 2, 1, 3),
                               mx.broadcast_to(mx.clip(d, 0, extent - 1)[None, None],
                                               (b, hq, e - a, size)), -1)
        m = mx.where((d >= extent)[None, None], 0.0, m)
        m = mx.where((d < 0)[None, None], -1e30, m).astype(q.dtype)
        m = mx.concatenate([m, sinks[:, a:e].transpose(0, 2, 1)[..., None].astype(q.dtype)], -1)
        outs.append(mx.fast.scaled_dot_product_attention(q[:, :, a:e], kk, vv, scale=scale,
                                                         mask=m))
    return mx.concatenate(outs, 2)


def _ms(fn, repeats=5):
    import time

    mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        mx.eval(fn())
    mx.synchronize()
    return round((time.perf_counter() - t0) / repeats * 1e3, 3)


@pytest.mark.parametrize("size", [4096, 16384, 65536])
def test_tiled_prefill_timing_record(size):
    """Recorded, not asserted: one global visit of a 512-row chunk on the
    Mini's attention geometry (24/8 heads, D 128, extent 1024), the kernel
    against the composed sliced path."""
    from functools import partial

    from mlx2.runtime.models.inkling2.prefill_attention import tiled_prefill_attention

    args, kw = _attention_case(7, b=1, hq=24, hkv=8, length=512, size=size, dim=128,
                               extent=1024, window=0, pads=None, dtype=mx.bfloat16)
    composed = partial(_composed_sliced, *args, kw["offset"], kw["scale"])
    tiled = partial(tiled_prefill_attention, *args, **kw)
    RESULTS[f"tiled/timing/512x{size}"] = {
        "composed_ms": _ms(composed), "tiled_ms": _ms(tiled),
        "max_abs": _gap(tiled(), composed())["max_abs"]}


@pytest.mark.parametrize("nsg", [4, 8])
def test_tiled_threadgroup_width_record(monkeypatch, nsg):
    """Recorded: 4 vs 8 simdgroups per threadgroup at 16K keys (8 needs the
    pipeline to admit 256 threads); both must match the composed law."""
    from functools import partial

    from mlx2.runtime.models.inkling2 import prefill_attention as pa
    from mlx2.runtime.models.inkling2 import splitk

    monkeypatch.setattr(pa, "TILED_NSG", nsg)
    args, kw = _attention_case(8, b=1, hq=24, hkv=8, length=512, size=16384, dim=128,
                               extent=1024, window=0, pads=None, dtype=mx.bfloat16)
    tiled = partial(pa.tiled_prefill_attention, *args, **kw)
    gap = _gap(tiled(), splitk.reference_attention(*args, **kw))["max_abs"]
    RESULTS[f"tiled/nsg{nsg}/512x16384"] = {"tiled_ms": _ms(tiled), "max_abs": gap}
    assert gap < 3e-2
