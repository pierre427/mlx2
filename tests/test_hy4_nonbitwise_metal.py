"""Real-Metal gate for the Hy4-flash opt-in performance repairs (audit H1-H6).

Run with ``MLX2_RUN_METAL_TESTS=1``; set ``MLX2_HY4_NONBITWISE_METAL_RECEIPT``
to a path to get a JSON summary of every measured gap (device, MLX version,
per-test numbers).  Synthetic tensors at the real Hy4 widths and the tiny
random checkpoint: no model checkpoint, no speed or qualification claim.
The CPU side of each item is tests/test_hy4_nonbitwise_cpu.py.
"""

import json
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import hy_v4_flash as M
from mlx2.runtime.models import hy_v4_flash_fp8kv as F8
from mlx2.runtime.models import hy_v4_flash_sparse_prefill as S

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 for the real-Metal gate",
)

R, P = 512, 64
D = R + P
SCALE = (192 + 64) ** -0.5
RESULTS: dict = {}


def _record(name, **values):
    RESULTS[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in values.items()}


def teardown_module(_module):
    path = os.environ.get("MLX2_HY4_NONBITWISE_METAL_RECEIPT")
    if path:
        try:
            device = mx.device_info().get("device_name")
        except Exception:  # noqa: BLE001
            device = None
        with open(path, "w") as handle:
            json.dump({"device": device, "mlx": mx.__version__, "results": RESULTS}, handle,
                      indent=1, sort_keys=True)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def _recalled(rng, B, L, K, T, start, empty=0.25):
    rec = np.full((B, L, K), -1, np.int32)
    for b in range(B):
        for l in range(L):
            pos = start + l
            n = min(int(K * (1 - empty)), pos + 1)
            rec[b, l, :n] = rng.choice(pos + 1, size=n, replace=False)
            if pos + 1 < T and n < K:
                rec[b, l, n] = rng.integers(pos + 1, T)  # a future id: masked
            rng.shuffle(rec[b, l])
    return mx.array(rec)


# ------------------------------------------------ H3: fp8 in-kernel dequant --

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("B,L,T,K,start", [(1, 80, 300, 40, 220), (2, 72, 600, 64, 500), (1, 65, 2100, 2048, 2000)])
def test_h3_fp8_kernel_is_bitwise_unpack_then_kernel(dtype, B, L, T, K, start):
    rng = np.random.default_rng(B * 1000 + L)
    q = mx.array(rng.standard_normal((B, 32, L, D)).astype(np.float32)).astype(dtype)
    rows = rng.standard_normal((B, T, D)).astype(np.float32)
    rows[..., :R] *= np.exp(rng.uniform(-6, 6, size=(B, T, 1))).astype(np.float32)
    packed = F8.pack_rows(mx.array(rows), R)
    rec = _recalled(rng, B, L, K, T, start)
    sinks = mx.array(rng.standard_normal(32).astype(np.float32))
    assert S.fp8_unsupported_reason(32, D, R, dtype, row_bytes=int(packed.shape[-1]), rows=L, lanes=B) is None
    got = S.sparse_prefill_attend_fp8(q, packed, rec, sinks, SCALE, R, start=start)
    want = S.sparse_prefill_attend(q, F8.unpack_rows(packed, R, dtype), rec, sinks, SCALE, R, start=start)
    mx.eval(got, want)
    same = bool(mx.array_equal(got, want).item())
    _record(f"h3_fp8_kernel_bitwise[{dtype},{B},{L},{T},{K}]", bitwise=same)
    assert same


def test_h3_fp8_model_engages_in_kernel_and_matches_unpack_route(monkeypatch):
    """Model level: the kernel geometry of the sparse prefill Metal test
    (16 heads, latent 128) is not 16-byte aligned for fp8 rows, so it declines
    with a counted reason; a latent-512 fixture engages."""
    from test_hy_v4_flash_cpu import build, checkpoint, tiny_config

    monkeypatch.setattr(M, "SPARSE_PREFILL_MIN_KEYS", 0)
    config = tiny_config(num_attention_heads=16, kv_lora_rank=512, qk_rope_head_dim=64,
                         compressed_indexer_head_dim=72, swa_qk_rope_head_dim=64)
    tokens = [int(t) for t in np.random.default_rng(25).integers(4, 90, size=210)]
    outs = {}
    for arm in ("unpack", "in_kernel"):
        model, _ = build(config, checkpoint(config))
        model.set_dtype(mx.bfloat16)
        model.args.latent_kv = "fp8"
        model.set_sparse_prefill_attention(True)
        if arm == "unpack":
            monkeypatch.setattr(M, "_sparse_fp8_reason", lambda *a, **k: "forced_off")
        else:
            monkeypatch.undo()
            monkeypatch.setattr(M, "SPARSE_PREFILL_MIN_KEYS", 0)
        cache = model.make_cache()
        outs[arm] = [model(mx.array([tokens[a:b]]), cache) for a, b in ((0, 100), (100, 200))]
        indexed = [l.self_attn for l in model.model.layers if l.self_attn.kind == M.INDEXED]
        outs[arm + "_counts"] = sum(a.sparse_prefill_fp8_forwards for a in indexed)
    assert outs["in_kernel_counts"] > 0 and outs["unpack_counts"] == 0
    same = all(mx.array_equal(a, b).item() for a, b in zip(outs["unpack"], outs["in_kernel"]))
    _record("h3_fp8_model_bitwise", bitwise=same, fp8_forwards=outs["in_kernel_counts"])
    assert same


# -------------------------------------------- H1 / H6: indexer fused scores --

# (B, rows, G): decode, verify, prefill blocks, B > 1, rows not a multiple of
# the 8-row block, G not a multiple of the 16-key tile or the 256-key chunk.
@pytest.mark.parametrize("B,rows,G", [(1, 1, 2048), (1, 6, 16384), (1, 64, 8192), (2, 5, 3001),
                                      (3, 7, 777), (2, 1, 300), (1, 2048, 4099)])
@pytest.mark.parametrize("variant", ["v1", "simd"])
@pytest.mark.parametrize("key_dtype", [mx.float32, mx.bfloat16])
def test_h1_h6_indexer_kernels_select_reference_sets(B, rows, G, variant, key_dtype):
    from mlx2.runtime.models import hy_v4_flash_fusions as F

    rng = np.random.default_rng(3)
    Hh, d = 16, 128
    k = min(511, G - 1)
    start = 4 * G - rows - 37
    # q and gk are bf16-valued in the model (cast to fp32 after the bf16 RoPE).
    q = mx.array(rng.standard_normal((B, Hh, rows, d)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    gk = mx.array(rng.standard_normal((B, G, d)).astype(np.float32) * 0.3).astype(mx.bfloat16)
    gk = gk.astype(key_dtype)
    w = mx.array(rng.standard_normal((B, rows, Hh)).astype(np.float32) * 0.02)
    own = (mx.arange(start, start + rows) // 4).astype(mx.int32)
    ref = F.indexer_logits_reference(q, gk.astype(mx.float32), w, own)
    kernel = F.indexer_logits_simd if variant == "simd" else F.indexer_logits_metal
    neg = kernel(q, gk, w, own, negate=True)
    assert neg.shape == ref.shape
    assert mx.array_equal(mx.isfinite(ref), mx.isfinite(neg)).item()
    fin = mx.isfinite(ref)
    scale = float(mx.max(mx.where(fin, mx.abs(ref), 0.0)).item())
    gap = float(mx.max(mx.where(fin, mx.abs(ref + neg), 0.0)).item())
    want = np.sort(np.array(mx.argpartition(-ref, kth=k - 1, axis=-1)[..., :k]), axis=-1)
    got = np.sort(np.array(mx.argpartition(neg, kth=k - 1, axis=-1)[..., :k]), axis=-1)
    agree = float(np.mean([len(np.intersect1d(a, b)) / k for a, b in
                           zip(got.reshape(-1, k), want.reshape(-1, k))]))
    _record(f"h6_indexer[{variant},{key_dtype},{B},{rows},{G}]", max_abs_logit_gap=gap / max(scale, 1e-30),
            topk_set_agreement=agree)
    assert gap <= 2e-6 * scale
    np.testing.assert_array_equal(got, want)


def test_h6_simd_model_engages_and_matches_v1_selections(monkeypatch):
    """Tiny checkpoint on the GPU: the simd variant engages (counters) and its
    greedy tokens match the v1 kernel's."""
    from test_hy_v4_flash_cpu import LONG, _greedy, build, checkpoint, tiny_config

    from mlx2.runtime.models import hy_v4_flash_fusions as F

    config = tiny_config(compressed_indexer_num_attention_heads=16, compressed_indexer_head_dim=128)
    W = checkpoint(config)
    outs, stats = {}, {}
    for variant in (True, "simd"):
        model, _ = build(config, W)
        F.install(model, indexer_scores=variant)
        outs[variant] = _greedy(model, LONG[:80], 8)
        stats[variant] = F.status(model)["indexer_fused_scores"]
    assert stats["simd"]["simd_calls"] > 0 and stats["simd"]["simd_declines"] == {}
    assert stats[True]["simd_calls"] == 0 and stats[True]["fused_score_calls"] > 0
    _record("h6_model_greedy_equal", equal=outs[True] == outs["simd"])
    assert outs[True] == outs["simd"]


# ------------------------------------------------------ H4: union prefill --

@pytest.mark.parametrize("rows", [16, 64])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_h4_union_matches_dense_within_its_own_rounding(rows, dtype):
    """Union vs dense on the GPU at the real widths (32 heads, r 512, p 64):
    both are MLX SDPA over the same keys with other reduction lengths; the
    gap is gated at the dense path's own distance from an fp32 run plus one
    16-bit ulp of the row maximum."""
    from mlx2.runtime.models import hy_v4_flash_union_prefill as U

    rng = np.random.default_rng(rows)
    L, T, K, start = 192, 4096, 512, 3900
    q32 = mx.array(rng.standard_normal((1, 32, L, D)).astype(np.float32))
    kv32 = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
    rec = _recalled(rng, 1, L, K, T, start)
    sinks = mx.array(rng.standard_normal(32).astype(np.float32))
    qpos, kpos = mx.arange(start, start + L), mx.arange(T)
    q, kv = q32.astype(dtype), kv32.astype(dtype)
    exact = M.dense_indexed_attend(q.astype(mx.float32), kv.astype(mx.float32), rec, sinks, SCALE, R, qpos, kpos)
    dense = M.dense_indexed_attend(q, kv, rec, sinks, SCALE, R, qpos, kpos)
    union = U.union_indexed_attend(q, kv, rec, sinks, SCALE, R, qpos, rows, attend=M.mla_attend)
    mx.eval(exact, dense, union)
    f = lambda a: np.array(a.astype(mx.float32), dtype=np.float64)  # noqa: E731
    ex, de, un = f(exact), f(dense), f(union)
    ulp = (2.0 ** -7 if dtype == mx.bfloat16 else 2.0 ** -10) * np.abs(ex).max(axis=-1, keepdims=True)
    bound = np.abs(de - ex).max(axis=-1, keepdims=True) + ulp
    gap = np.abs(un - de)
    _record(f"h4_union_vs_dense[{rows},{dtype}]", max_abs=gap.max(), worst_ratio=(gap / bound).max(),
            union_err=np.abs(un - ex).mean(), dense_err=np.abs(de - ex).mean())
    assert np.all(gap <= bound)
    assert np.abs(un - ex).mean() <= 1.10 * np.abs(de - ex).mean()


def test_h4_model_union_engages_and_tracks_dense(monkeypatch):
    from test_hy_v4_flash_cpu import build, checkpoint, tiny_config

    monkeypatch.setattr(M, "SPARSE_PREFILL_MIN_KEYS", 0)
    config = tiny_config(num_attention_heads=16, kv_lora_rank=128, qk_rope_head_dim=64,
                         compressed_indexer_head_dim=72, swa_qk_rope_head_dim=64)
    W = checkpoint(config)
    tokens = [int(t) for t in np.random.default_rng(25).integers(4, 90, size=210)]
    outs = {}
    for rows in (0, 64):
        model, _ = build(config, W)
        model.set_dtype(mx.bfloat16)
        model.set_sparse_prefill_union_rows(rows)
        cache = model.make_cache()
        outs[rows] = np.concatenate([np.array(model(mx.array([tokens[a:b]]), cache).astype(mx.float32))
                                     for a, b in ((0, 100), (100, 200))], axis=1)
        indexed = [l.self_attn for l in model.model.layers if l.self_attn.kind == M.INDEXED]
        outs[f"{rows}_forwards"] = sum(a.sparse_prefill_union_forwards for a in indexed)
    assert outs["64_forwards"] > 0 and outs["0_forwards"] == 0
    gap = np.abs(outs[64] - outs[0]).max() / np.abs(outs[0]).max()
    top1 = float(np.mean(outs[64].argmax(-1) == outs[0].argmax(-1)))
    _record("h4_model_union_vs_dense", rel_max_logit_gap=gap, top1_agreement=top1)
    assert gap <= 5e-2 and top1 >= 0.97


# ------------------------------------------- H5: BF16-resident fp32 planes --

@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("N,K", [(512, 3072), (120832, 3072), (37, 64)])
def test_h5_gemv_matches_fp32_and_router_topk(rows, N, K):
    from mlx2.runtime.models import hy_v4_flash_resident as R

    rng = np.random.default_rng(rows + N)
    x = mx.array(rng.standard_normal((rows, K)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ w.astype(mx.float32).T
    got = R.f32_matmul(x, w, kernel=True)
    mx.eval(ref, got)
    rel = float(mx.max(mx.abs(got - ref)).item()) / max(float(mx.max(mx.abs(ref)).item()), 1e-30)
    record = {"rel_max_gap": rel}
    if N == 512:
        k = 16
        a = np.sort(np.array(mx.argpartition(-ref, kth=k - 1, axis=-1)[..., :k]), axis=-1)
        b = np.sort(np.array(mx.argpartition(-got, kth=k - 1, axis=-1)[..., :k]), axis=-1)
        record["router_top16_set_agreement"] = float(np.mean(
            [len(np.intersect1d(p, q)) / k for p, q in zip(a, b)]))
    _record(f"h5_gemv[{rows},{N},{K}]", **record)
    assert rel <= 1e-5


def test_h5_model_decode_takes_gemv_and_tracks_reference():
    from test_hy_v4_flash_cpu import LONG, build, checkpoint, tiny_config

    from mlx2.runtime.models import hy_v4_flash_resident as R

    config = tiny_config()
    W = {k: (np.array(mx.array(v).astype(mx.bfloat16).astype(mx.float32)) if v.dtype == np.float32 else v)
         for k, v in checkpoint(config).items()}
    ref, _ = build(config, W)
    model, _ = build(config, W)
    R.install(model, R.PARTS)
    before = R.STATS["gemv_calls"]
    outs = []
    for m in (ref, model):
        cache = m.make_cache()
        rows = [m(mx.array([LONG[:40]]), cache)[:, -1]]
        for t in (5, 9, 11, 13):
            rows.append(m(mx.array([[t]]), cache)[:, -1])
        outs.append(np.array(mx.concatenate(rows).astype(mx.float32)))
    assert R.STATS["gemv_calls"] > before
    gap = np.abs(outs[1] - outs[0]).max() / np.abs(outs[0]).max()
    _record("h5_model_decode", rel_max_logit_gap=gap,
            top1_agreement=float(np.mean(outs[1].argmax(-1) == outs[0].argmax(-1))))
    assert gap <= 1e-4
