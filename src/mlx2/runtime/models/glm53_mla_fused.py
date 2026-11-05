# SPDX-License-Identifier: MIT
"""Opt-in GLM-5.3 sparse-MLA decode glue in fewer launches (``glm53_mla_fused=decode_v1``).

At B1 single-token decode, ``Glm53GGUFModel._mla`` under the arm4 sparse laws
(incremental_v1 pools, topk_v1_kernel selection, gathered_v1 attention) runs
about 25 small dependent launches per MLA layer between the input projections
and ``attn_output``, and on every fourth token (the pool-completion step) the
~35-op fixed-order pooling of one new pool on top. ``decode_v1`` keeps every
projection, every cache write, the select kernel and the attention core, and
replaces the glue between them by four original kernels:

``prep``    one launch, three threadgroups: rms(q_a) (1536), rms(kv_a) (512)
            and the indexer key LayerNorm + kpool-gate concatenation (the raw
            indexer row). Replaces 2 RMSNorm + ~11 elementwise/reduce/concat.
``pool``    one launch: the fixed-order pooling law of one completed pool
            (``pool_fixed_order`` over 4 raw rows). Replaces ~35 launches on
            the pool-completion step.
``score``   one launch: the indexer pool scores after the q x pooled matmul
            (scale, ReLU, head weights, head sum). Replaces 5 launches.
``gather``  one launch: the selected latent rows and the additive 0/-inf mask
            the gathered SDPA consumes. Replaces clip, take, compare, where.

The q x pooled matmul and the gathered attention (MLX's SDPA, which for 64
query heads over one key head runs its unfused op chain: scale, matmul, mask
add, precise softmax, matmul) stay MLX calls with the same operands, so their
reduction orders are MLX's own; this module does not re-implement them.

Exactness: each kernel reproduces the MLX v0.32.4 kernels it replaces one
IEEE operation per reference operation, in the reference order:

* rms: ``rms_single_row`` (``N_READS`` 4, one simdgroup per 128 elements):
  ``acc += x * x`` contracted into ``fma`` (the AOT build's contraction,
  pinned by glm53_kda_fused's Metal tests), ``simd_sum``, a second
  ``simd_sum`` over the per-simdgroup slots (zeros beyond), then
  ``precise::rsqrt(acc / N + eps)`` and ``w * (x * inv)``.
* mean over 128: ``row_reduce_looped`` with one simdgroup (``in.size() / 128
  < 32`` rules out ``row_reduce_simple``): lane l folds ``total = x + total``
  over elements 4l..4l+3 from 0, then ``0 + total``, then ``simd_sum``;
  ``mean`` multiplies by the exact normalizer 1/128.
* ``x ** 2`` is MLX ``Power`` (``metal::pow``, precise under -fno-fast-math);
  ``rsqrt`` is ``precise::rsqrt``; ``exp`` is ``precise::exp``; maximum is
  MLX's ``Maximum`` (NaN in the first operand wins, then ``x > y ? x : y``).
* head sum over axis 2 of (1, 1, 32, P): ``col_reduce_looped`` (BM = BN = 32):
  ``c + 0`` per lane, lane h = head h, ``simd_sum``.

Every kernel is first-use gated bitwise against the live op chain on
synthetic inputs (``ready``); a failed gate declines and is reported.
Contraction is pinned with ``fp contract(off)`` and explicit ``fma``.

The pool-completion step also stops paying for earlier tokens: the reference
never reads a token's raw indexer row until its pool completes, so the raw
row's projections, LayerNorm chain and append of three tokens were deferred
into every fourth step. ``decode_v1`` ties each step's raw append into that
step's graph (``mx.depends``; no values change).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

POLICIES = ("reference", "decode_v1")
COUNTERS = (
    "mla_fused",            # a selecting decode step ran the fused glue
    "mla_fused_dense",      # a decode step at T <= 2048 ran prep (no ids exist)
    "mla_fused_pool",       # pools computed by the one-pool kernel
    "mla_fused_declined",   # decode_v1 selected but the reference chain ran (gate, law or device)
    "mla_fused_not_candidate",  # a prompt chunk or multi-row verify call: never the kernel's shape
)
Q_RANK, LATENT, RAW, IDX_DIM, IDX_HEADS, KPOOL = 1536, 512, 256, 128, 32, 4
PREP_THREADS = 384
SCORE_THREADS = 256
GATHER_THREADS = 128
_GATE = {"error": None}
_SCHEDULED = {"prep": 0, "pool": 0, "score": 0, "gather": 0}


def _hexf(value):
    """A float32 literal naming exactly ``np.float32(value)``."""
    return float(np.float32(value)).hex() + "f"


# ---- prep: rms(q_a), rms(kv_a), indexer key LayerNorm ++ kpool gate --------
_PREP_SOURCE = r"""
const uint tg = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
threadgroup float local_sums[32];
threadgroup float local_inv[1];
if (tg < 2) {
    #pragma clang fp contract(off)
    // rms_single_row: axis 1536 (384 threads) or 512 (128 threads).
    const uint axis = tg == 0 ? QA : KV;
    const uint nsg = axis / 128;
    const device float* x = tg == 0 ? qa : kv;
    const device float* w = tg == 0 ? wq : wkv;
    device float* out = tg == 0 ? qr : latent;
    const bool active = sg < nsg;
    float tx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc = 0.0f;
    if (active) {
        for (int i = 0; i < 4; i++) {
            tx[i] = x[lid * 4 + i];
            acc = metal::fma(tx[i], tx[i], acc);
        }
    }
    acc = simd_sum(acc);
    if (sg == 0) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && lane == 0) {
        local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) {
            local_inv[0] = metal::precise::rsqrt(acc / float(axis) + RMS_EPS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        const float inv = local_inv[0];
        for (int i = 0; i < 4; i++) {
            const float normed = tx[i] * inv;
            out[lid * 4 + i] = w[lid * 4 + i] * normed;
        }
    }
} else if (sg == 0) {
    #pragma clang fp contract(off)
    // Indexer key LayerNorm(128, eps 1e-6) as the reference op chain runs it.
    float v[4];
    for (int i = 0; i < 4; i++) {
        v[i] = key[4 * lane + i];
    }
    float t = 0.0f;
    for (int i = 0; i < 4; i++) {
        t = v[i] + t;
    }
    float total = 0.0f + t;
    const float mean = simd_sum(total) * INV128;
    float d[4];
    float q = 0.0f;
    for (int i = 0; i < 4; i++) {
        d[i] = v[i] - mean;
        const float sq = metal::precise::pow(d[i], 2.0f);
        q = sq + q;
    }
    total = 0.0f + q;
    const float var = simd_sum(total) * INV128;
    const float shifted = var + LN_EPS;
    const float r = metal::precise::rsqrt(shifted);
    for (int i = 0; i < 4; i++) {
        const uint c = 4 * lane + i;
        const float n = d[i] * r;
        const float scaled = n * kw[c];
        raw[c] = scaled + kb[c];
        raw[DIM + c] = gate[c];
    }
}
"""


# ---- pool: fixed-order pooling of one complete pool (4 raw rows) -----------
_POOL_SOURCE = r"""
#pragma clang fp contract(off)
const uint c = thread_position_in_grid.x;
float k[4];
float lg[4];
for (int j = 0; j < 4; j++) {
    k[j] = rows[j * 2 * DIM + c];
    lg[j] = rows[j * 2 * DIM + DIM + c] + ape[j * DIM + c];
}
const float m01 = glm53_max(lg[0], lg[1]);
const float m23 = glm53_max(lg[2], lg[3]);
const float m = glm53_max(m01, m23);
float e[4];
for (int j = 0; j < 4; j++) {
    const float shifted = lg[j] - m;
    e[j] = metal::precise::exp(shifted);
}
const float s01 = e[0] + e[1];
const float s012 = s01 + e[2];
const float s = s012 + e[3];
float pk[4];
for (int j = 0; j < 4; j++) {
    const float p = e[j] / s;
    pk[j] = p * k[j];
}
const float a01 = pk[0] + pk[1];
const float a012 = a01 + pk[2];
pooled[c] = a012 + pk[3];
"""


# ---- score: pool scores from the q x pooled matmul -------------------------
_SCORE_SOURCE = r"""
#pragma clang fp contract(off)
const uint p = threadgroup_position_in_grid.x * (SCORE_TG / 32) + simdgroup_index_in_threadgroup;
const uint h = thread_index_in_simdgroup;
const uint P = uint(meta[0]);
if (p >= P) {
    return;
}
const float a = mm[size_t(h) * P + p] * C_DIM;
const float relu = glm53_max(a, 0.0f);
const float w = proj[h] * C_HEADS;
const float c = relu * w;
const float v = c + 0.0f;
const float s = simd_sum(v);
if (h == 0) {
    scores[p] = s;
}
"""


# ---- gather: selected latent rows and the additive mask ---------------------
_GATHER_SOURCE = r"""
const uint c = thread_position_in_grid.x;
const uint k = thread_position_in_grid.y;
const int total = meta[0];
const int id = ids[k];
const int low = id > 0 ? id : 0;
const int src = low < total - 1 ? low : total - 1;
rows[size_t(k) * LATENT + c] = latent[size_t(src) * LATENT + c];
if (c == 0) {
    mask[k] = id >= 0 ? 0.0f : -INFINITY;
}
"""

_HEADER = r"""
// MLX's Maximum functor for floating types (binary_ops.h).
inline float glm53_max(float x, float y) {
    if (metal::isnan(x)) {
        return x;
    }
    return x > y ? x : y;
}
"""


def kernel_sources(rms_eps):
    """The four Metal bodies for one rms epsilon (inspectable without Metal)."""
    return {
        "prep": _PREP_SOURCE.replace("RMS_EPS", _hexf(rms_eps))
        .replace("INV128", _hexf(1.0 / 128))
        .replace("LN_EPS", _hexf(1e-6))
        .replace("QA", str(Q_RANK)).replace("KV", str(LATENT)).replace("DIM", str(IDX_DIM)),
        "pool": _POOL_SOURCE.replace("DIM", str(IDX_DIM)),
        "score": _SCORE_SOURCE.replace("C_DIM", _hexf(IDX_DIM**-0.5))
        .replace("C_HEADS", _hexf(IDX_HEADS**-0.5)).replace("SCORE_TG", str(SCORE_THREADS)),
        "gather": _GATHER_SOURCE.replace("LATENT", str(LATENT)),
    }


_SPECS = {
    "prep": (["qa", "kv", "key", "gate", "wq", "wkv", "kw", "kb"], ["qr", "latent", "raw"]),
    "pool": (["rows", "ape"], ["pooled"]),
    "score": (["mm", "proj", "meta"], ["scores"]),
    "gather": (["latent", "ids", "meta"], ["rows", "mask"]),
}


@lru_cache(maxsize=16)
def _kernel(kind, rms_eps):
    inputs, outputs = _SPECS[kind]
    tag = f"{np.float32(rms_eps).view(np.uint32):08x}" if kind == "prep" else "v1"
    return mx.fast.metal_kernel(
        name=f"mlx2_glm53_mla_{kind}_v1_{tag}", input_names=inputs, output_names=outputs,
        header=_HEADER, source=kernel_sources(rms_eps)[kind], ensure_row_contiguous=True,
    )


def _on_gpu():
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def _f32(*arrays):
    return all(a is not None and a.dtype == mx.float32 for a in arrays)


# ---- launches ---------------------------------------------------------------
def prep_admits(qa, kv, key, gate):
    return (
        _f32(qa, kv, key, gate)
        and qa.shape == (1, 1, Q_RANK) and kv.shape == (1, 1, LATENT)
        and key.shape == (1, 1, IDX_DIM) and gate.shape == (1, 1, IDX_DIM)
    )


def prep(qa, kv, key, gate, wq, wkv, kw, kb, *, rms_eps):
    """(qr (1,1,1536), latent row (1,1,1,512), raw indexer row (1,1,1,256))."""
    qr, latent, raw = _kernel("prep", float(rms_eps))(
        inputs=[qa, kv, key, gate, wq, wkv, kw, kb],
        grid=(3 * PREP_THREADS, 1, 1), threadgroup=(PREP_THREADS, 1, 1),
        output_shapes=[(1, 1, Q_RANK), (1, 1, 1, LATENT), (1, 1, 1, RAW)],
        output_dtypes=[mx.float32] * 3,
    )
    _SCHEDULED["prep"] += 1
    return qr, latent, raw


def pool_one(raw, ape):
    """raw (4, 256) committed rows of one pool -> pooled (1, 128); ``pool_fixed_order``'s bits."""
    (pooled,) = _kernel("pool", 0.0)(
        inputs=[raw, ape], grid=(IDX_DIM, 1, 1), threadgroup=(IDX_DIM, 1, 1),
        output_shapes=[(1, IDX_DIM)], output_dtypes=[mx.float32],
    )
    _SCHEDULED["pool"] += 1
    return pooled


def pool_scores(mm, proj):
    """mm (1, 1, 32, P) = q @ pooled^T, proj (1, 1, 32) unscaled -> scores (1, P)."""
    pools = mm.shape[-1]
    groups = -(-pools // (SCORE_THREADS // 32))
    (scores,) = _kernel("score", 0.0)(
        inputs=[mm, proj, mx.array([pools], dtype=mx.int32)],
        grid=(groups * SCORE_THREADS, 1, 1), threadgroup=(SCORE_THREADS, 1, 1),
        output_shapes=[(1, pools)], output_dtypes=[mx.float32],
    )
    _SCHEDULED["score"] += 1
    return scores


def gather(latent, ids):
    """latent (1, 1, T, 512), ids (1, W) -> rows (1, W, 512), additive mask (1, 1, 1, W)."""
    width, total = ids.shape[-1], latent.shape[2]
    rows, mask = _kernel("gather", 0.0)(
        inputs=[latent, ids, mx.array([total], dtype=mx.int32)],
        grid=(LATENT, width, 1), threadgroup=(GATHER_THREADS, 1, 1),
        output_shapes=[(1, width, LATENT), (1, 1, 1, width)], output_dtypes=[mx.float32] * 2,
    )
    _SCHEDULED["gather"] += 1
    return rows, mask


# ---- the reference op chains the gate compares against ----------------------
class _GateWeights:
    """The minimal weights surface pack_indexer_rows reads."""

    def __init__(self, arrays, gate):
        self.arrays, self.gate = arrays, gate

    def array(self, name):
        return self.arrays[name]

    def linear(self, name):
        if not name.endswith(".indexer.kpool_gate"):
            raise AssertionError(name)
        return lambda x: self.gate


def reference_prep(qa, kv, key, gate, wq, wkv, kw, kb, *, rms_eps):
    """The reference chain of _mla_prepare + pack_indexer_rows for the same inputs."""
    from . import glm53_gguf_indexer as indexer
    from .glm53_gguf import rms

    qr = rms(qa, wq, rms_eps)
    latent = rms(kv, wkv, rms_eps)[:, None]
    weights = _GateWeights({"g.indexer.k_norm.weight": kw, "g.indexer.k_norm.bias": kb}, gate)
    raw = indexer.pack_indexer_rows(kv, weights, "g", projected={"key": key})[:, None]
    return qr, latent, raw


def reference_scores(q, proj, pooled):
    from . import glm53_gguf_indexer as indexer

    return indexer.pool_scores(q, proj * IDX_HEADS**-0.5, pooled)[0]


def reference_gather(latent, ids):
    from . import glm53_sparse_attention as sa

    rows, valid = sa.gather_latent(latent, ids[None])
    return rows, mx.where(valid, mx.array(0.0, mx.float32), mx.array(-mx.inf, mx.float32))


def _bits(a):
    return np.array(a).view(np.uint32)


def _equal(got, want):
    mx.eval(got, want)
    return got.shape == want.shape and np.array_equal(_bits(got), _bits(want))


def gate_cases(seed=5303):
    """Synthetic inputs: real-scale, large, tiny, signed zeros and nonfinite values."""
    rng = np.random.default_rng(seed)

    def f(*shape, scale=1.0):
        return (rng.normal(size=shape) * scale).astype(np.float32)

    cases = []
    for scale in (1.0, 37.0, 1e-3):
        cases.append({
            "qa": f(1, 1, Q_RANK, scale=scale), "kv": f(1, 1, LATENT, scale=scale),
            "key": f(1, 1, IDX_DIM, scale=scale), "gate": f(1, 1, IDX_DIM, scale=2.0),
            "wq": 1 + f(Q_RANK, scale=0.2), "wkv": 1 + f(LATENT, scale=0.2),
            "kw": 1 + f(IDX_DIM, scale=0.2), "kb": f(IDX_DIM, scale=0.1),
        })
    edge = {k: v.copy() for k, v in cases[0].items()}
    edge["key"][0, 0, :7] = np.array([0.0, -0.0, 3e38, -3e38, 1e-38, -1e-38, 0.0], np.float32)
    edge["gate"][0, 0, :4] = np.array([np.inf, -np.inf, np.nan, 0.0], np.float32)
    cases.append(edge)
    return cases


def _check_prep(rms_eps):
    for case in gate_cases():
        arrays = {k: mx.array(v) for k, v in case.items()}
        got = prep(**arrays, rms_eps=rms_eps)
        want = reference_prep(**arrays, rms_eps=rms_eps)
        for name, g, w in zip(("qr", "latent", "raw"), got, want):
            if not _equal(g, w):
                raise ValueError(f"mlx2_glm53_mla_prep_v1 bit gate failed ({name})")


def _check_pool():
    from . import glm53_gguf_indexer as indexer

    rng = np.random.default_rng(5304)
    for scale in (1.0, 30.0):
        raw = (rng.normal(size=(KPOOL, RAW)) * scale).astype(np.float32)
        raw[1, IDX_DIM : IDX_DIM + 3] = [np.inf, -np.inf, np.nan]
        raw[2, :2] = [-0.0, 0.0]
        ape = rng.normal(0, 0.1, (KPOOL, IDX_DIM)).astype(np.float32)
        r, a = mx.array(raw), mx.array(ape)
        if not _equal(pool_one(r, a), indexer.pool_fixed_order(r, a)):
            raise ValueError("mlx2_glm53_mla_pool_v1 bit gate failed")


def _check_score():
    rng = np.random.default_rng(5305)
    for pools in (511, 2048, 4103):
        q = mx.array(rng.normal(size=(1, 1, IDX_HEADS, IDX_DIM)).astype(np.float32))
        pooled_np = rng.normal(size=(1, pools, IDX_DIM)).astype(np.float32)
        pooled_np[0, :3] = 0.0
        pooled_np[0, 3, 0] = np.nan
        pooled = mx.array(pooled_np)
        proj = mx.array(rng.normal(size=(1, 1, IDX_HEADS)).astype(np.float32))
        mm = q @ pooled[:, None].swapaxes(-1, -2)
        if not _equal(pool_scores(mm, proj), reference_scores(q, proj, pooled)):
            raise ValueError("mlx2_glm53_mla_score_v1 bit gate failed")


def _check_gather():
    rng = np.random.default_rng(5306)
    total, width = 4099, 2051
    latent = mx.array(rng.normal(size=(1, 1, total, LATENT)).astype(np.float32))
    ids = np.sort(rng.choice(total, size=width, replace=False)).astype(np.int32)
    ids[5:9] = -1
    ids[-2:] = -1
    ids = mx.array(ids[None])
    for got, want in zip(gather(latent, ids), reference_gather(latent, ids)):
        if not _equal(got, want):
            raise ValueError("mlx2_glm53_mla_gather_v1 bit gate failed")


@lru_cache(maxsize=4)
def self_check(rms_eps):
    """First-use bitwise GPU gate of all four kernels against their op chains."""
    if not _on_gpu():
        raise RuntimeError("glm53_mla_fused=decode_v1 requires the Metal GPU")
    _check_prep(float(rms_eps))
    _check_pool()
    _check_score()
    _check_gather()
    return True


def ready(rms_eps):
    """True when the kernels may serve: Metal GPU and the first-use gate passed."""
    if not _on_gpu() or _GATE["error"] is not None:
        return False
    try:
        return self_check(float(rms_eps))
    except (RuntimeError, ValueError) as error:  # declined, counted, reported
        _GATE["error"] = f"{type(error).__name__}: {error}"
        return False


def route_status():
    return {
        "policies": {"reference": {"selected_by_default": True},
                     "decode_v1": {"implemented": True, "qualified": False, "selected_by_default": False}},
        "gate_error": _GATE["error"],
        "scheduled_calls": dict(_SCHEDULED),
    }
