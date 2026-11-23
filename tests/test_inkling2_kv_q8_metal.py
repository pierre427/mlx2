"""``inkling2_kv_q8``: int8 K/V attention kernels vs the CPU-fallback oracle (Metal).

Run under the GPU lock. The oracle is ``kv_q8.reference_attention``: the
packed K/V dequantized, then the composed path's dense mask + zero sink
column + SDPA (what ``Attention.__call__`` runs when the kernel declines).
Both arms read the same int8 codes, so the gap is the kernels' reduction
order and FP16 tile staging only, the same tolerance as the BF16 split-K
kernel (tests/test_inkling2_splitk_metal.py). The model-level tests bound
the Q8 model against the BF16 one and check the kernel route engages.
Set ``INKLING2_KV_Q8_METAL_RECEIPT`` to a path for a JSON summary.
"""

import json
import os

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

H, HK, D, EXTENT = 24, 8, 128, 1024  # Inkling-2-Mini global-visit geometry
RESULTS = {"kernel": [], "model": []}


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
    path = os.environ.get("INKLING2_KV_Q8_METAL_RECEIPT")
    if path:
        RESULTS["device"] = mx.metal.device_info().get("device_name")
        RESULTS["mlx"] = mx.__version__
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


def _inputs(b, length, size, dtype, seed, pad=None, layout="cache_view", dim=D, heads=H,
            kv_heads=HK):
    from mlx2.runtime.models.inkling2 import kv_q8

    mx.random.seed(seed)
    q = (mx.random.normal((b, length, heads, dim)) * 0.3).astype(dtype).transpose(0, 2, 1, 3)
    spare = 200 if layout == "cache_view" else 0
    k = kv_q8.pack((mx.random.normal((b, kv_heads, size + spare, dim)) * 0.3).astype(dtype))
    v = kv_q8.pack(mx.random.normal((b, kv_heads, size + spare, dim)).astype(dtype))
    if layout == "cache_view":
        # A KVCache fetch: a leading-key view of a buffer with spare capacity.
        k, v = k[:, :, :size], v[:, :, :size]
    elif layout == "odd_stride":
        # Rows one byte off the 16-byte grid: the in-kernel byte-load path.
        pad = mx.zeros((*k.shape[:-1], 1), mx.int8)
        k = mx.concatenate([pad, k], axis=-1)[..., 1:]
        v = mx.concatenate([pad, v], axis=-1)[..., 1:]
    elif layout == "batch_slice":
        # A filtered BatchKVCache: a batch-row view (nonzero base offset).
        k = mx.concatenate([k, k], axis=0)[b:]
        v = mx.concatenate([v, v], axis=0)[b:]
    rel = (mx.random.normal((b, length, heads, EXTENT)) * 2).astype(dtype)
    sinks = (mx.random.normal((b, length, heads)) * 2).astype(dtype)
    left = None if pad is None else mx.array(pad, dtype=mx.int32)
    return q, k, v, rel, sinks, left


def _check(q, k, v, rel, sinks, left, dtype, label):
    from mlx2.runtime.models.inkling2 import kv_q8

    size, length, dim = k.shape[2], q.shape[2], q.shape[3]
    assert kv_q8.decline_reason(q, k, v, n_heads=q.shape[1], n_kv=k.shape[1],
                                head_dim=dim) is None
    kw = dict(offset=size - length, left_padding=left, window=0, scale=dim ** -0.5)
    with mx.stream(mx.gpu):
        got = kv_q8.q8_attention(q, k, v, rel, sinks, **kw)
        ref = kv_q8.reference_attention(q, k, v, rel, sinks, **kw)
        mx.eval(got, ref)
    assert got.shape == ref.shape and got.dtype == q.dtype
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    scale = max(1.0, mx.abs(ref.astype(mx.float32)).max().item())
    tol = 2e-5 * scale if dtype == mx.float32 else 2 ** -7 * scale  # BF16 output rounding
    RESULTS["kernel"].append({"case": label, "max_abs_diff": diff, "tol": tol})
    assert diff <= tol, (diff, tol)
    return diff


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("size", [37, 520, 4096, 40960])
@pytest.mark.parametrize("length", [1, 2, 3, 5, 8])
def test_q8_kernels_match_the_oracle(length, size, b, dtype):
    """Decode (L = 1: the register kernel past 2048 keys, the packed split
    kernel below) and verify blocks of 2-8 rows; one split (37), split tails
    (520), multi-split (4K, 40K); B2 with left padding on row 1."""
    if size < length:
        pytest.skip("fewer keys than queries")
    pad = None if b == 1 else [0, min(37, size // 3)]
    args = _inputs(b, length, size, dtype, seed=size + length + b, pad=pad)
    _check(*args, dtype, f"L{length}-S{size}-B{b}-{dtype}")


@pytest.mark.parametrize("length", [1, 8])
@pytest.mark.parametrize("size", [131072])
def test_q8_kernels_long_keys(length, size):
    """128K keys (the A/B's longest cell): many splits, long bias-free runs."""
    args = _inputs(1, length, size, mx.bfloat16, seed=17)
    _check(*args, mx.bfloat16, f"long-L{length}-S{size}")


@pytest.mark.parametrize("layout", ["contiguous", "cache_view", "batch_slice", "odd_stride"])
@pytest.mark.parametrize("length", [1, 8])
def test_q8_kernels_kv_layouts(layout, length):
    """16-byte vector loads (contiguous, cache view, batch-row view) and the
    in-kernel byte path (odd row stride, not 16-byte aligned) give the same
    law: an unaligned plane is never misread."""
    args = _inputs(1, length, 5000, mx.bfloat16, seed=7, layout=layout)
    _check(*args, mx.bfloat16, f"layout-{layout}-L{length}")


@pytest.mark.parametrize("length,size", [(1, 300), (1, 3000), (8, 300), (8, 3000)])
def test_q8_kernels_negative_and_full_left_padding(length, size):
    """Negative padding means none; a row padded over every key attends only
    to the sink (output 0)."""
    args = _inputs(2, length, size, mx.bfloat16, seed=size, pad=[-2, size])
    _check(*args, mx.bfloat16, f"pad-L{length}-S{size}")


@pytest.mark.parametrize("dim,heads,kv_heads", [(32, 4, 2), (64, 8, 2)])
@pytest.mark.parametrize("length", [1, 4])
def test_q8_kernels_small_heads(dim, heads, kv_heads, length):
    """The tiny test model's geometry (D 32, one 32-wide group) and D 64."""
    args = _inputs(1, length, 3000, mx.float32, seed=dim, dim=dim, heads=heads,
                   kv_heads=kv_heads)
    _check(*args, mx.float32, f"small-D{dim}-L{length}")


@pytest.mark.parametrize("length", [1, 8])
def test_q8_kernels_peaked_softmax_at_clip_norm(length):
    """Q and K at the L2 clip norm (22.6) times a tau of 1.3: large, peaked
    logits, the worst case for FP16 staging of the dequantized K and of P."""
    from mlx2.runtime.models.inkling2 import kv_q8

    q, _, v, rel, sinks, _ = _inputs(1, length, 8192, mx.bfloat16, seed=11)
    mx.random.seed(12)
    unit = lambda a: a / mx.sqrt(mx.sum(mx.square(a.astype(mx.float32)), -1, keepdims=True))
    q = (unit(q) * 22.63 * 1.3).astype(mx.bfloat16)
    k = kv_q8.pack((unit(mx.random.normal((1, HK, 8192, D))) * 22.63).astype(mx.bfloat16))
    _check(q, k, v, rel * 1.3, sinks, None, mx.bfloat16, f"peaked-L{length}")


def _gap(got, base):
    def logp(x):
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    gaps = mx.abs(got - base).max(-1).reshape(-1)
    kl = (mx.exp(logp(base)) * (logp(base) - logp(got))).sum(-1).reshape(-1)
    return {"max_abs": gaps.max().item(), "median_abs": float(mx.sort(gaps)[gaps.size // 2].item()),
            "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
            "argmax_agree": (mx.argmax(got, -1) == mx.argmax(base, -1)).mean().item()}


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_q8_model_kernel_route_vs_fallback_and_bf16(bf16, rows):
    """Tiny model (D 32): verify blocks (8 rows) and one-token decode through
    the kernels. Against the same model with the kernel route declined (the
    blockwise fallback, same codes): near-exact. Against the BF16 model:
    the Q8 approximation, bounded on the median position and KL (a routed
    expert flip can move one position; see the CPU test)."""
    import test_inkling2_cpu as t
    from mlx.utils import tree_flatten

    from mlx2.runtime.models.inkling2 import kv_q8

    plan = [(0, 20), (20, 28)] + [(i, i + 1) for i in range(28, 44)]
    with mx.stream(mx.gpu):
        model = t._tiny_model()
        if bf16:
            model.load_weights([(k, v.astype(mx.bfloat16))
                                for k, v in tree_flatten(model.parameters())])
        tokens = mx.random.randint(0, 500, (rows, 44))
        base = t._run(model, tokens, plan)
        model.configure_kernels(kv_q8=True)
        got = t._run(model, tokens, plan)
        receipt = model.kernel_receipt()["kv_q8"]
        original = kv_q8.decline_reason
        kv_q8.decline_reason = lambda *a, **kw: "forced_fallback"
        try:
            fallback = t._run(model, tokens, plan)
        finally:
            kv_q8.decline_reason = original
        model.configure_kernels()
    calls = receipt["calls"]
    # The 20-row prefill chunk engages too (1 <= L < 32).
    assert calls.get("kernel_decode", 0) == 8 * 16 and calls.get("kernel_rows", 0) == 8 * 2
    vs_fallback = _gap(got, fallback)
    vs_bf16 = _gap(got, base)
    RESULTS["model"].append({"bf16": bf16, "rows": rows, "calls": calls,
                             "vs_fallback": vs_fallback, "vs_bf16": vs_bf16})
    assert vs_fallback["max_abs"] < (3e-2 if bf16 else 1e-3), vs_fallback
    assert vs_bf16["median_abs"] < (0.05 if bf16 else 0.01), vs_bf16
    assert vs_bf16["kl_mean"] < 1e-2 and vs_bf16["argmax_agree"] >= 0.9, vs_bf16


def test_q8_model_wide_geometry_decode():
    """D 128 (the Mini's head size: register kernel and 64-wide groups) on a
    wider tiny model, past 2048 keys so the L == 1 register kernel runs."""
    import test_inkling2_fuse_metal as f

    with mx.stream(mx.gpu):
        model = f._model("wide", bf16=True)
        tokens = mx.random.randint(0, 500, (1, 2100))
        plan = [(0, 1024), (1024, 2048), (2048, 2056)] + [(i, i + 1) for i in range(2056, 2100)]
        base = f._run(model, tokens, plan)
        model.configure_kernels(kv_q8=True)
        got = f._run(model, tokens, plan)
        calls = model.kernel_receipt()["kv_q8"]["calls"]
        model.configure_kernels()
    gap = _gap(got[:, 2048:], base[:, 2048:])
    RESULTS["model"].append({"case": "wide-2100", "calls": calls, "vs_bf16": gap})
    assert calls.get("kernel_decode", 0) == 8 * 44 and calls.get("kernel_rows", 0) == 8
    assert calls.get("kernel_prefill_tiled", 0) == 8 * 2  # the 1024-row prefill chunks
    assert model.kernel_receipt()["kv_q8"]["kv_q8_full_prefix_dequant"] == 0
    assert gap["kl_mean"] < 1e-2 and gap["argmax_agree"] >= 0.9, gap


@pytest.mark.parametrize("length", [32, 100, 512])
@pytest.mark.parametrize("size", [8192, 40960])
@pytest.mark.parametrize("b", [1, 2])
def test_q8_prefill_tiles_match_the_oracle(length, size, b):
    """Prefill chunks through 31-row kernel tiles (``tiled_attention``) vs
    the oracle over the whole chunk, B2 with left padding."""
    from mlx2.runtime.models.inkling2 import kv_q8

    pad = None if b == 1 else [0, 37]
    q, k, v, rel, sinks, left = _inputs(b, length, size, mx.bfloat16, seed=size + length, pad=pad)
    kw = dict(offset=size - length, left_padding=left, window=0, scale=D ** -0.5)
    counts = {}
    with mx.stream(mx.gpu):
        got = kv_q8.tiled_attention(q, k, v, rel, sinks, counts=counts, **kw)
        ref = kv_q8.reference_attention(q, k, v, rel, sinks, **kw)
        mx.eval(got, ref)
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    tol = 2 ** -7 * max(1.0, mx.abs(ref.astype(mx.float32)).max().item())
    RESULTS["kernel"].append({"case": f"prefill-L{length}-S{size}-B{b}", "max_abs_diff": diff,
                              "tol": tol, "tiles": counts.get("prefill_tiles")})
    assert counts["prefill_tiles"] == -(-length // kv_q8.PREFILL_TILE)
    assert diff <= tol, (diff, tol)


@pytest.mark.parametrize("length", [100])
def test_q8_blockwise_fallback_on_gpu_matches_the_oracle(length):
    """The fallback (declined geometry) on the GPU: key blocks of 4096."""
    from mlx2.runtime.models.inkling2 import kv_q8

    size = 20000
    q, k, v, rel, sinks, _ = _inputs(1, length, size, mx.bfloat16, seed=3)
    kw = dict(offset=size - length, left_padding=None, window=0, scale=D ** -0.5)
    with mx.stream(mx.gpu):
        got = kv_q8.blockwise_attention(q, k, v, rel, sinks, **kw)
        ref = kv_q8.reference_attention(q, k, v, rel, sinks, **kw)
        mx.eval(got, ref)
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert diff <= 2 ** -7 * max(1.0, mx.abs(ref.astype(mx.float32)).max().item())


def test_q8_long_chunked_prefill_peak_memory():
    """A 16K-token chunked prefill (512-row chunks) of the wide tiny model,
    Q8 vs BF16: peak-memory delta over the prefill, recorded. Q8 must not
    exceed BF16 (no whole-prefix BF16 copies), and no whole-prefix
    dequantize may run."""
    import test_inkling2_fuse_metal as f

    from mlx2.runtime.models.inkling2 import kv_q8

    n, chunk = 16384, 512
    peaks = {}
    with mx.stream(mx.gpu):
        model = f._model("wide", bf16=True)
        tokens = mx.random.randint(0, 500, (1, n))
        before = kv_q8.full_prefix_dequant_calls()
        for arm in ("bf16", "q8"):
            model.configure_kernels(kv_q8=arm == "q8")
            cache = model.make_cache()
            mx.eval(tokens)
            mx.clear_cache()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            for a in range(0, n, chunk):
                mx.eval(model(tokens[:, a:a + chunk], cache=cache))
                mx.clear_cache()
            peaks[arm] = {"peak_delta_mib": round((mx.get_peak_memory() - base) / 2**20, 1),
                          "kv": kv_q8.kv_bytes(cache[0::2])}
            del cache
        receipt = model.kernel_receipt()["kv_q8"]
        model.configure_kernels()
    RESULTS["model"].append({"case": "peak-memory-16k", "peaks": peaks,
                             "prefill_tiles": receipt["kv_q8_prefill_tiles"]})
    assert kv_q8.full_prefix_dequant_calls() == before
    assert receipt["kv_q8_prefill_tiles"] > 0
    assert peaks["q8"]["peak_delta_mib"] <= peaks["bf16"]["peak_delta_mib"] * 1.05 + 8


@pytest.mark.parametrize("size", [16384, 65536, 131072])
def test_q8_decode_kernel_bandwidth(size):
    """Micro-benchmark (recorded, not asserted beyond sanity; no thermal
    hold, not publishable): the one-token decode kernel alone at the Mini's
    global geometry, Q8 (16-byte loads) vs the BF16 split-K kernel, median
    of 30 timed launches after 5 warm-ups; effective GB/s counts the K/V
    bytes each arm must read."""
    import statistics
    import time

    from mlx2.runtime.models.inkling2 import kv_q8, splitk

    mx.random.seed(size)
    q = (mx.random.normal((1, 1, H, D)) * 0.3).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    kf = (mx.random.normal((1, HK, size, D)) * 0.3).astype(mx.bfloat16)
    vf = mx.random.normal((1, HK, size, D)).astype(mx.bfloat16)
    kq, vq = kv_q8.pack(kf), kv_q8.pack(vf)
    rel = (mx.random.normal((1, 1, H, EXTENT)) * 2).astype(mx.bfloat16)
    sinks = (mx.random.normal((1, 1, H)) * 2).astype(mx.bfloat16)
    mx.eval(q, kf, vf, kq, vq, rel, sinks)
    kw = dict(offset=size - 1, left_padding=None, window=0, scale=D ** -0.5)
    arms = {
        "q8": (lambda: kv_q8.q8_attention(q, kq, vq, rel, sinks, **kw), kq.nbytes + vq.nbytes),
        "bf16_splitk": (lambda: splitk.gqa_splitk_attention(q, kf, vf, rel, sinks, **kw),
                        kf.nbytes + vf.nbytes),
    }
    row = {"keys": size}
    with mx.stream(mx.gpu):
        for name, (fn, nbytes) in arms.items():
            for _ in range(5):
                mx.eval(fn())
            times = []
            for _ in range(30):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fn())
                mx.synchronize()
                times.append(time.perf_counter() - t0)
            sec = statistics.median(times)
            row[name] = {"median_us": round(sec * 1e6, 1), "kv_bytes": nbytes,
                         "effective_gb_s": round(nbytes / sec / 1e9, 1)}
    row["q8_speedup"] = round(row["bf16_splitk"]["median_us"] / row["q8"]["median_us"], 3)
    RESULTS.setdefault("bench", []).append(row)
    assert row["q8"]["median_us"] > 0 and row["bf16_splitk"]["median_us"] > 0
