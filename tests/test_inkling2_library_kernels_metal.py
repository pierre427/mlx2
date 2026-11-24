"""``inkling2/library_kernels`` on Metal (GPU; run under the GPU lease only).

Spec acceptance properties L3 (bitwise mat-vec over the shape grid, three
dtypes, row windows into a wider stacked matrix, optional real-model
weights), L5 (an injected tile mismatch is refused on the first call, which
still returns the reference; a correct key proves once and then serves) and
L6 (a fused MLX-functor chain equals its MLX-op sequence on special values).

Checkpoint: synthetic, unless ``INKLING2_LIBK_MODEL`` names an Inkling 2
checkpoint directory (``*.safetensors``); then L3 also runs on its 2-D BF16
projection weights and the receipt records ``<dir name>@<sha256 of the
first shard's header>``. ``INKLING2_LIBK_DRAWS`` sets the operand draws per
shape (default 3; the spec's full property is 20). ``INKLING2_LIBK_RECEIPT``
writes a JSON summary.
"""

import hashlib
import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

OUT_LENS = [3, 4, 511, 512, 2047, 2048, 4095, 4096, 8192]
KS = [64, 65, 1024, 2304, 8192]
DTYPES = [mx.float32, mx.float16, mx.bfloat16]
DRAWS = int(os.environ.get("INKLING2_LIBK_DRAWS", "3"))
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
    path = os.environ.get("INKLING2_LIBK_RECEIPT")
    if path:
        from mlx2.runtime.models.inkling2 import library_kernels as lib

        RESULTS["library"] = lib.library_receipt()
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True, default=str)


def _lib():
    from mlx2.runtime.models.inkling2 import library_kernels as lib

    return lib


def _bitwise(a, b):
    return _lib().same_bits([(a, b)]) is None


def _draw(shape, dtype, seed, scale=1.0):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", KS)
def test_one_row_grid_is_bitwise(dtype, k):
    lib = _lib()
    failures = []
    for n in OUT_LENS:
        for draw in range(DRAWS):
            x = _draw((1, k), dtype, 1000 * n + draw)
            w = _draw((n, k), dtype, 7 + 1000 * n + draw, 0.05)
            planned = lib.plan_matvec(x, w)
            assert not isinstance(planned, str), (n, k, planned)
            ok = _bitwise(planned.fused(), planned.reference())
            RESULTS[f"gemv/{dtype}/n{n}/k{k}/d{draw}"] = ok
            if not ok:
                failures.append((n, draw, planned.key))
    assert not failures, failures[:5]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", list(range(2, 16)))
def test_wide_rows_are_bitwise_or_declined(dtype, rows):
    lib = _lib()
    failures = []
    for n in (24, 64, 65, 1024, 3072, 8704):
        for draw in range(DRAWS):
            x = _draw((rows, 2304), dtype, 31 * n + draw)
            w = _draw((n, 2304), dtype, 5 + 31 * n + draw, 0.05)
            planned = lib.plan_matvec(x, w)
            if lib.gpu_generation() < lib.WIDE_MIN_GENERATION:
                assert planned == "gpu_generation"
                return
            assert not isinstance(planned, str), planned
            ok = _bitwise(planned.fused(), planned.reference())
            RESULTS[f"wide/{dtype}/r{rows}/n{n}/d{draw}"] = ok
            if not ok:
                failures.append((n, draw, planned.key))
    assert not failures, failures[:5]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_row_window_of_a_stacked_matrix(dtype, rows):
    lib = _lib()
    stacked = _draw((8704, 2304), dtype, 3, 0.05)
    x = _draw((rows, 2304), dtype, 4)
    for start, stop in [(0, 3072), (3072, 3456), (3456, 6528), (8680, 8704)]:
        w = stacked[start:stop]
        planned = lib.plan_matvec(x, w)
        if isinstance(planned, str):
            assert rows > 1 and planned in ("rows_dtype", "gpu_generation"), planned
            continue
        ok = _bitwise(planned.fused(), planned.reference())
        RESULTS[f"window/{dtype}/r{rows}/{start}:{stop}"] = ok
        assert ok, (start, stop)


def _checkpoint_weights(limit=12):
    root = os.environ.get("INKLING2_LIBK_MODEL")
    if not root:
        return None, []
    shards = sorted(Path(root).glob("*.safetensors"))
    if not shards:
        return None, []
    with open(shards[0], "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = handle.read(size)
    ident = f"{Path(root).name}@sha256:{hashlib.sha256(header).hexdigest()}"
    picked = []
    for shard in shards:
        for name, value in mx.load(str(shard)).items():
            if (value.ndim == 2 and value.dtype == mx.bfloat16 and name.endswith("proj.weight")
                    and value.shape[1] % 4 == 0):
                picked.append((name, value))
                if len(picked) >= limit:
                    return ident, picked
    return ident, picked


def test_real_model_weights_are_bitwise():
    lib = _lib()
    ident, weights = _checkpoint_weights()
    if not weights:
        pytest.skip("INKLING2_LIBK_MODEL not set (synthetic only)")
    RESULTS["checkpoint"] = ident
    for name, w in weights:
        for rows in (1, 2, 5, 8):
            x = _draw((rows, w.shape[1]), w.dtype, rows)
            planned = lib.plan_matvec(x, w)
            if isinstance(planned, str):
                RESULTS[f"model/{name}/r{rows}"] = "declined:" + planned
                continue
            ok = _bitwise(planned.fused(), planned.reference())
            RESULTS[f"model/{name}/r{rows}"] = ok
            assert ok, (name, rows)


def _injected(dtype, seed):
    """A launch with the wrong tiles (SN 4 where MLX uses SN 32: each output
    summed in another order), wrapped to count its launches."""
    lib = _lib()
    x = _draw((1, 2304), dtype, seed)
    w = _draw((512, 2304), dtype, seed + 1, 0.05)
    assert lib.matvec_plan(1, 512, 2304, dtype) == ("gemv", (4, 1, 1, 32, 4, 4))
    wrong = lib.plan_matvec(x, w, tiles=(1, 1, 8, 4, 4, 4))
    assert not isinstance(wrong, str)
    launches = []

    def counted():
        launches.append(1)
        return wrong.fused()

    return x, w, lib.Planned(wrong.key, counted, wrong.reference, wrong.pairs), launches


def test_injected_tile_mismatch_is_refused_and_never_relaunched():
    # FP32 output: the accumulation order shows in the stored bits. (In BF16
    # the FP32 sums of both orders usually round to the same BF16 value, so
    # a wrong tiling can pass a proof on one draw; the 2026-11-23 smoke saw
    # exactly that, see test_bf16_rounding_can_hide_a_wrong_order.)
    lib = _lib()
    x, w, planned, launches = _injected(mx.float32, 11)
    proofs, owner = lib.Proofs("t_injected"), type("Owner", (), {})()
    first, engaged = lib.run_planned(proofs, planned, owner, "c")
    assert not engaged and _bitwise(first, x @ w.T)
    verdict = proofs.receipt()["refused"]
    RESULTS["injected"] = verdict
    assert list(verdict.values()) == ["bits:differs0"], verdict
    for _ in range(3):
        out, engaged = lib.run_planned(proofs, planned, owner, "c")
        assert not engaged and _bitwise(out, x @ w.T)
    assert len(launches) == 1 and owner.c == {"refused_first": 1, "refused": 3}


def test_bf16_rounding_can_hide_a_wrong_order():
    """Records (does not require) whether one BF16 draw exposes the wrong
    tiling, and checks the bookkeeping either way: a passing proof is
    ``proved``, a failing one ``refused_first``; the returned value is the
    reference in both cases."""
    lib = _lib()
    x, w, planned, _launches = _injected(mx.bfloat16, 11)
    proofs, owner = lib.Proofs("t_injected_bf16"), type("Owner", (), {})()
    first, engaged = lib.run_planned(proofs, planned, owner, "c")
    assert not engaged and _bitwise(first, x @ w.T)
    refused = proofs.receipt()["refused"]
    RESULTS["injected_bf16"] = refused or "proved (bf16 rounding hid the order)"
    assert owner.c == ({"refused_first": 1} if refused else {"proved": 1})


def test_correct_key_proves_once_then_serves():
    lib = _lib()
    lib.MATVEC_PROOFS.reset()
    owner = type("Owner", (), {})()
    x = _draw((1, 2304), mx.bfloat16, 21)
    w = _draw((3072, 2304), mx.bfloat16, 22, 0.05)
    outs = [lib.matvec(x, w, owner) for _ in range(4)]
    assert all(_bitwise(o, x @ w.T) for o in outs)
    assert owner.matvec_counts == {"proved": 1, "calls": 3}, owner.matvec_counts


_CHAIN = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= n[0]) return;
    // MLX ops: s = sigmoid(g.astype(float32)) [f32 array]; p = a * s [f32
    // array, a promoted]; out = p.astype(T). Rounded where MLX materialises.
    const float s = Sigmoid()(float(g[i]));
    const float p = Multiply()(float(a[i]), s);
    out[i] = T(p);
"""


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_functor_chain_matches_mlx_ops_on_special_values(dtype):
    lib = _lib()
    header, why = lib.library("elementwise")
    assert why is None
    kernel = mx.fast.metal_kernel(name="lk_chain", input_names=["a", "g", "n"],
                                  output_names=["out"], header=header, source=_CHAIN)
    finfo = mx.finfo(dtype)
    specials = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 88.0, -88.0,
                float(finfo.max), float(finfo.min), 1e-40, -1e-40, 6e-8, 1e-30, -1e-30]  # subnormals
    rnd = mx.random.normal((4096,), key=mx.random.key(1)) * 30
    vals = mx.concatenate([mx.array(specials), rnd])
    a = vals.astype(dtype)
    g = mx.roll(vals, 3).astype(dtype)
    n = a.size
    fused = kernel(inputs=[a, g, mx.array([n], mx.uint32)], template=[("T", dtype)],
                   grid=(n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(n,)],
                   output_dtypes=[dtype])[0]
    reference = (a * mx.sigmoid(g.astype(mx.float32))).astype(dtype)
    ok = _bitwise(fused, reference)
    RESULTS[f"chain/{dtype}"] = ok
    assert ok
