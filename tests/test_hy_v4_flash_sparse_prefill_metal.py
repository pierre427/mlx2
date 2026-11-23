"""Metal check of the Hy4-flash sparse prefill attention kernel.

``hy_v4_flash_sparse_prefill.sparse_prefill_attend`` is compared with the two
existing indexed-attention laws it replaces or extends, on the same random
operands: the dense masked reference (``dense_indexed_attend``, the
``L > GATHER_MAX_QUERY`` default) and the per-row gather
(``gathered_indexed_attend``, the ``L <= 64`` path), plus a float64 NumPy
oracle on the same 16-bit-rounded operands.

Tolerances.  Every law reads identical 16-bit operands; the kernel keeps
scores, softmax and the value sum in fp32 and rounds once, at the output, so
its gate is against the float64 oracle: per row, at most 1 ulp of the row's
largest output (bf16 2^-7, fp16 2^-10, times the row max; the output's own
rounding is half of that).  Measured: 0.48-0.63 ulp in every case.  MLX's
SDPA (dense and gather paths) rounds intermediates to the input dtype, so it
sits 2-9x further from the oracle (bf16 1.8e-2..1.4e-1 abs on |out| <= 4.8);
the kernel-vs-dense gate is therefore the dense path's own per-row oracle
error plus 1 ulp.  Shapes are small (a few MB); heads, latent rank and rope
width are the real config's (32 heads, r 512, p 64).
"""

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="sparse prefill kernel is Metal-only")

from mlx2.runtime.models import hy_v4_flash as M  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_sparse_prefill as S  # noqa: E402

R, P = 512, 64
D = R + P
SCALE = (192 + 64) ** -0.5  # qk_nope + qk_rope of Hy4-flash
# Model-level gate: sparse-vs-fp32 error at most this many times dense-vs-fp32.
RATIO_GATE = 1.10


@pytest.fixture(autouse=True)
def _gpu():
    # tests/conftest.py makes the CPU the default device.
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _case(seed, *, B=1, H=32, L=80, T=300, K=40, start=220, dtype=mx.bfloat16, empty=0.25,
          lone_rows=0, q_scale=1.0):
    """Random operands; ``recalled`` holds distinct causal ids, a fraction
    ``empty`` of -1 slots, a few future ids (the kernel must mask them, as
    the dense path's causal term does) and ``lone_rows`` rows with a single
    valid id (only the query's own position)."""
    rng = np.random.default_rng(seed)
    q = mx.array((rng.standard_normal((B, H, L, D)) * q_scale).astype(np.float32)).astype(dtype)
    kv = mx.array(rng.standard_normal((B, T, D)).astype(np.float32)).astype(dtype)
    rec = np.full((B, L, K), -1, np.int32)
    for b in range(B):
        for l in range(L):
            pos = start + l
            if l < lone_rows:
                rec[b, l, rng.integers(K)] = pos
                continue
            n = min(int(K * (1 - empty)), pos + 1)
            ids = rng.choice(pos + 1, size=n, replace=False)
            rec[b, l, :n] = ids
            if pos + 1 < T and n < K:
                rec[b, l, n] = rng.integers(pos + 1, T)  # a future id: masked
            rng.shuffle(rec[b, l])
    sinks = mx.array(rng.standard_normal(H).astype(np.float32) * 2.0)
    return q, kv, mx.array(rec), sinks


def _oracle(q, kv, rec, sinks, start, first=0):
    """float64 reference on the rounded operands (sinks rounded to q.dtype)."""
    qn = np.array(q.astype(mx.float32), dtype=np.float64)
    kvn = np.array(kv.astype(mx.float32), dtype=np.float64)
    sn = np.array(sinks.astype(q.dtype).astype(mx.float32), dtype=np.float64)
    recn = np.array(rec)
    B, H, L, _ = qn.shape
    T = kvn.shape[1]
    out = np.zeros((B, H, L, R))
    for b in range(B):
        for l in range(L):
            ids = recn[b, l]
            rows = ids - first
            ok = (ids >= 0) & (ids <= start + l) & (rows >= 0) & (rows < T)
            keys = kvn[b, rows[ok]]
            s = SCALE * qn[b, :, l] @ keys.T  # [H, n]
            m = np.maximum(s.max(axis=1, initial=-np.inf), sn)
            e = np.exp(s - m[:, None])
            den = e.sum(axis=1) + np.exp(sn - m)
            out[b, :, l] = (e @ keys[:, :R]) / den[:, None]
    return out


def _f(a):
    return np.array(a.astype(mx.float32), dtype=np.float64)


def _eps(dtype):
    return float(np.finfo(np.float16).eps) if dtype == mx.float16 else 2.0 ** -7


def _rows(x):
    return np.abs(x).max(axis=-1, keepdims=True)


def _gate(got, oracle, dtype, *, ref=None):
    """``got`` within 1 ulp (of the row max) of the oracle; with ``ref`` (an
    MLX SDPA law), ``got`` also within ``ref``'s own oracle error + 1 ulp of it."""
    ulp = _eps(dtype) * np.maximum(_rows(oracle), 1e-3)
    err = np.abs(got - oracle)
    assert np.all(err <= ulp), f"oracle err {err.max():.3e}, worst {(err / ulp).max():.2f} ulp"
    if ref is not None:
        bound = _rows(ref - oracle) + ulp
        diff = np.abs(got - ref)
        assert np.all(diff <= bound), f"ref diff {diff.max():.3e}, worst ratio {(diff / bound).max():.2f}"
    return err.max()


def _run(q, kv, rec, sinks, start):
    T = kv.shape[1]
    L = q.shape[2]
    qpos, kpos = mx.arange(start, start + L), mx.arange(0, T)
    dense = M.dense_indexed_attend(q, kv, rec, sinks, SCALE, R, qpos, kpos)
    sparse = S.sparse_prefill_attend(q, kv, rec, sinks, SCALE, R, start=start)
    mx.eval(dense, sparse)
    return dense, sparse


CASES = [
    dict(seed=1),                                                   # bf16, offset chunk
    dict(seed=2, dtype=mx.float16),                                 # fp16
    dict(seed=3, start=0, T=96, L=96, K=48, lone_rows=3),           # causal edge: chunk at position 0
    dict(seed=4, H=16, K=37),                                       # 16 heads, K not a tile multiple
    dict(seed=5, B=2, L=72, K=64, empty=0.6),                       # two lanes, mostly empty slots
    dict(seed=6, q_scale=6.0),                                      # peaked softmax (sinks negligible)
    dict(seed=7, L=65, T=2100, start=2000, K=128),                  # 65 rows: first L above the gather bound
]


@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_kernel_matches_dense_reference_and_oracle(case):
    start = case.get("start", 220)
    dtype = case.get("dtype", mx.bfloat16)
    q, kv, rec, sinks = _case(**case)
    dense, sparse = _run(q, kv, rec, sinks, start)
    assert sparse.shape == dense.shape and sparse.dtype == q.dtype
    s = _f(sparse)
    assert np.isfinite(s).all()
    _gate(s, _oracle(q, kv, rec, sinks, start), dtype, ref=_f(dense))


def test_kernel_matches_gather_path():
    """At the gather path's shapes (<= 64 rows) the kernel agrees with it too
    (the gather path needs causal ids, as ``_attend`` masks them first)."""
    start, L = 500, 64
    q, kv, rec, sinks = _case(11, L=L, T=600, start=start, K=96)
    rec = mx.where(rec <= mx.arange(start, start + L)[None, :, None], rec, -1)
    gather = M.gathered_indexed_attend(q, kv, rec, sinks, SCALE, R)
    sparse = S.sparse_prefill_attend(q, kv, rec, sinks, SCALE, R, start=start)
    mx.eval(gather, sparse)
    _gate(_f(sparse), _oracle(q, kv, rec, sinks, start), mx.bfloat16, ref=_f(gather))


def test_all_empty_row_is_zero_and_sink_only():
    """A row with no valid slot attends only the sink: output exactly 0, not NaN."""
    q, kv, rec, sinks = _case(12, L=70)
    rec = mx.where(mx.arange(70)[None, :, None] == 5, -1, rec)
    _, sparse = _run(q, kv, rec, sinks, 220)
    s = _f(sparse)
    assert np.isfinite(s).all() and np.all(s[:, :, 5] == 0)


def test_first_offset_reads_rows_relative_to_first():
    """``first`` > 0: id ``i`` reads ``kv[i - first]``; ids before ``first`` are empty."""
    start, first = 400, 100
    q, kv, rec, sinks = _case(13, L=70, T=start + 70 - first, start=start, K=48)
    sparse = S.sparse_prefill_attend(q, kv, rec, sinks, SCALE, R, start=start, first=first)
    _gate(_f(sparse), _oracle(q, kv, rec, sinks, start, first), mx.bfloat16)


def test_geometry_gate():
    assert S.unsupported_reason(32, D, R, mx.bfloat16) is None
    assert S.unsupported_reason(16, D, R, mx.float16) is None
    assert S.unsupported_reason(32, D, R, mx.float32) == "dtype_float32"
    assert S.unsupported_reason(64, D, R, mx.bfloat16) == "heads_64"
    # The real geometry's static threadgroup arrays: 29,248 of 32,768 bytes.
    assert S.threadgroup_bytes(32, D) == 29248 <= S.THREADGROUP_MEMORY_LIMIT
    # A wider latent row would overflow the 32 KiB threadgroup: counted fallback.
    assert S.unsupported_reason(32, 1024, R, mx.bfloat16) == "threadgroup_memory_43584"
    assert S.unsupported_reason(32, D, R, mx.bfloat16, rows=S.MAX_GRID_ROWS + 1) == "grid_rows"
    assert S.unsupported_reason(32, D, R, mx.bfloat16, first=1) == "kv_first_nonzero"


def test_k2048_matches_dense_and_oracle():
    """The real recall width, K = 2048 slots (512 groups x 4), on a small chunk."""
    start = 2500
    q, kv, rec, sinks = _case(14, L=66, T=2600, start=start, K=2048, empty=0.1)
    dense, sparse = _run(q, kv, rec, sinks, start)
    _gate(_f(sparse), _oracle(q, kv, rec, sinks, start), mx.bfloat16, ref=_f(dense))


def test_dispatch_probe_failure_is_a_counted_reason(monkeypatch):
    """A pipeline build / dispatch failure of an instantiation is caught once by
    the probe and reported as a reason (the model then counts a fallback)."""
    def broken(*a, **k):
        raise RuntimeError("[metal::Device] Unable to build metal library")

    monkeypatch.setattr(S, "_PROBES", {})
    monkeypatch.setattr(S, "_launch", broken)
    assert S.unsupported_reason(32, D, R, mx.bfloat16) == "dispatch_probe_RuntimeError"
    monkeypatch.setattr(S, "_launch", lambda *a, **k: pytest.fail("probe result not cached"))
    assert S.unsupported_reason(32, D, R, mx.bfloat16) == "dispatch_probe_RuntimeError"


def _tiny_model(dtype, latent_kv="native"):
    """The CPU tests' tiny Hy4 fixture with a kernel-compatible indexed geometry
    (16 heads, latent 128, rope 64: D 192; the fixture shares the rope width
    with the sliding layers), cast to ``dtype``; ``latent_kv`` its row format."""
    from test_hy_v4_flash_cpu import build, checkpoint, tiny_config

    mx.set_default_device(mx.gpu)  # that module selects the CPU at import
    config = tiny_config(num_attention_heads=16, kv_lora_rank=128, qk_rope_head_dim=64,
                         compressed_indexer_head_dim=72, swa_qk_rope_head_dim=64)
    model, _ = build(config, checkpoint(config))
    if dtype != mx.float32:
        model.set_dtype(dtype)
    model.args.latent_kv = latent_kv
    return model


def _prefill(model, tokens, chunks):
    cache, outs = model.make_cache(), []
    for a, b in chunks:
        outs.append(model(mx.array([tokens[a:b]]), cache))
    outs += [model(mx.array([[t]]), cache) for t in tokens[chunks[-1][1]:]]
    return np.array(mx.concatenate(outs, axis=1).astype(mx.float32))


def _counts(model):
    attn = [l.self_attn for l in model.model.layers if l.self_attn.kind == M.INDEXED]
    declines = {}
    for a in attn:
        for k, v in a.sparse_prefill_declines.items():
            declines[k] = declines.get(k, 0) + v
    return sum(a.sparse_prefill_forwards for a in attn), declines


def test_model_switch_engages_counts_and_matches_dense():
    """Through the model: off is the dense path; on routes every > 64-row
    indexed prefill chunk (a first chunk and an offset second chunk) to the
    kernel, counts it and leaves decode on the gather path.

    The random tiny model is chaotic in 16-bit (its dense bf16 logits are
    already ~2.8 mean abs from its fp32 logits), so a direct sparse-vs-dense
    logit bound would measure that, not the kernel: both 16-bit routes are
    scored against the fp32 dense model instead, and the sparse route may
    not be further from it than the dense one.  Measured (deterministic:
    fixed seeds, no atomics): bf16 2.92 vs 2.78 mean abs (ratio 1.05), fp16
    1.03 vs 1.44 (0.71); the gate is RATIO_GATE = 1.10, the bf16 measurement
    plus 5% margin.  The fp8 test below uses the same gate."""
    tokens = [int(t) for t in np.random.default_rng(21).integers(4, 90, size=230)]
    chunks = [(0, 120), (120, 220)]
    ref = _prefill(_tiny_model(mx.float32), tokens, chunks)
    for dtype in (mx.bfloat16, mx.float16):
        model = _tiny_model(dtype)
        dense = _prefill(model, tokens, chunks)
        assert _counts(model) == (0, {})
        model.set_sparse_prefill_attention(True)
        assert model.sparse_prefill_attention
        sparse = _prefill(model, tokens, chunks)
        indexed = sum(1 for l in model.model.layers if l.self_attn.kind == M.INDEXED)
        assert _counts(model) == (2 * indexed, {})  # two prefill chunks, decode untouched
        assert np.isfinite(sparse).all()
        assert np.abs(sparse - ref).mean() <= RATIO_GATE * np.abs(dense - ref).mean()
    with pytest.raises(ValueError):
        model.set_sparse_prefill_attention(1)


def test_model_switch_declines_fp32_with_a_reason():
    tokens = [int(t) for t in np.random.default_rng(22).integers(4, 90, size=100)]
    model = _tiny_model(mx.float32)
    model.set_sparse_prefill_attention(True)
    _prefill(model, tokens, [(0, 100)])
    forwards, declines = _counts(model)
    assert forwards == 0 and declines == {"dtype_float32": sum(
        1 for l in model.model.layers if l.self_attn.kind == M.INDEXED)}


def test_model_first_nonzero_falls_back_and_counts():
    """An indexed cache whose first row is not position 0 (never produced
    today; a restored state could) takes the dense path with a counted reason."""
    tokens = [int(t) for t in np.random.default_rng(23).integers(4, 90, size=200)]
    model = _tiny_model(mx.bfloat16)
    model.set_sparse_prefill_attention(True)
    cache = model.make_cache()
    model(mx.array([tokens[:40]]), cache)
    for c in cache:
        if c.kind == M.INDEXED:
            c.start = 1
    model(mx.array([tokens[40:140]]), cache)
    forwards, declines = _counts(model)
    indexed = sum(1 for l in model.model.layers if l.self_attn.kind == M.INDEXED)
    assert forwards == 0 and declines == {"kv_first_nonzero": indexed}


def test_batch_cache_two_ragged_lanes():
    """Through HY4FlashBatchLayerCache: one joint chunk where lane 0 has 100
    rows (> 64: the kernel) and lane 1 has 40 (the gather path), each after
    its own history.  Each lane's logits match the same lane run alone with
    the switch on (the batch only joins projections and MoE)."""
    rng = np.random.default_rng(24)
    hist = [[int(t) for t in rng.integers(4, 90, size=n)] for n in (20, 30)]
    chunk = [[int(t) for t in rng.integers(4, 90, size=n)] for n in (100, 40)]
    model = _tiny_model(mx.float16)
    model.set_sparse_prefill_attention(True)
    lanes = []
    for h in hist:
        c = model.make_cache()
        model(mx.array([h]), c)
        lanes.append(c)
    alone = []
    for h, x in zip(hist, chunk):
        c = model.make_cache()
        model(mx.array([h]), c)
        alone.append(np.array(model(mx.array([x]), c).astype(mx.float32))[0])
    before = _counts(model)[0]
    merged = [M.HY4FlashBatchLayerCache.merge([a, b]) for a, b in zip(*lanes)]
    width = max(len(x) for x in chunk)
    for layer in merged:
        layer.prepare(lengths=[len(x) for x in chunk])
    joint = mx.array([x + [0] * (width - len(x)) for x in chunk])
    out = np.array(model(joint, merged).astype(mx.float32))
    indexed = sum(1 for l in model.model.layers if l.self_attn.kind == M.INDEXED)
    # The joint forward ran the kernel once per indexed layer (lane 0 only).
    assert _counts(model) == (before + indexed, {})
    assert [r.offset for r in merged[0].rows] == [120, 70]
    for i, x in enumerate(chunk):
        got, want = out[i, :len(x)], alone[i]
        # Measured 7.6e-4 (lane 0, fp16 batched projections) and 0 (lane 1).
        assert np.abs(got - want).max() <= 5e-3 * np.abs(want).max(), i


def test_fp8_latent_cache_engages_and_matches_dense():
    """fp8 latent rows: the chunk's rows are dequantized whole (as the dense
    path does) and then read by the kernel; same gate as the native test."""
    tokens = [int(t) for t in np.random.default_rng(25).integers(4, 90, size=210)]
    chunks = [(0, 100), (100, 200)]
    ref = _prefill(_tiny_model(mx.float32, "fp8"), tokens, chunks)
    model = _tiny_model(mx.float16, "fp8")
    assert isinstance(model.make_cache()[0], M.HY4FlashFP8LayerCache)
    dense = _prefill(model, tokens, chunks)
    model.set_sparse_prefill_attention(True)
    sparse = _prefill(model, tokens, chunks)
    indexed = sum(1 for l in model.model.layers if l.self_attn.kind == M.INDEXED)
    assert _counts(model) == (2 * indexed, {})
    assert np.abs(sparse - ref).mean() <= RATIO_GATE * np.abs(dense - ref).mean()
