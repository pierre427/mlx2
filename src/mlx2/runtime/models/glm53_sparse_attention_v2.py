# SPDX-License-Identifier: MIT
"""GLM-5.3 fused gathered sparse attention v2 (design glm53-speed-round2.md 1.1, 2.1).

Absorbed MLA with one 512-wide latent head: the gathered rows of a query are
both its keys and its values and are shared by all 64 query heads, so each
gathered tile is staged once in threadgroup memory and serves every head
through ``simdgroup_float8x8`` MMAs (heads as ``M``).

Kernel ``mlx2_glm53_gathered_attention_v2``: one threadgroup per
``(key split s, query row z)``; ``SPLIT`` gathered ids per split.

* pass 1 (scores): tiles of ``8 * DSPLIT`` keys x ``512 / DSPLIT`` dims are
  staged ``DSPLIT`` times per key tile (16 KiB); simdgroup ``(g, kb)`` keeps
  the fragment ``S[8 heads of group g, 8 keys of block kb]`` in registers
  across the dim stages, so every score is one MMA chain over dims
  ``0..511`` in ascending 8-blocks whatever ``DSPLIT`` is. Scores land in the
  threadgroup ``S[64, SPLIT]`` scratch (16 KiB at ``SPLIT = 64``).
* statistics: 8 lanes per head over the split's keys, invalid keys (``-1``
  ids or slots past ``W``) at ``-inf``; ``m`` = max of ``SCALE * S``,
  ``P = exp(SCALE * S - m)`` written in place, ``l`` by a fixed xor tree.
  A split without a valid key writes ``m = -inf, l = 0`` and ``P = 0``; a
  NaN score makes ``m`` NaN (it is never dropped by ``max``).
* pass 2 (values): tiles of 8 keys x 512 dims are re-staged (rows from L2);
  simdgroup ``(g, slice)`` accumulates ``O[8 heads, 512 / DSPLIT dims] +=
  P[8, 8] . V[8, dims]``.

Every tile slot is written on every fill: rows of ``-1`` ids and slots past
the split's key count are zero, so no MMA reads stale threadgroup memory and
``P = 0`` meets ``V = 0`` (never NaN). Ids above the row's last committed
latent row are clipped to it, as ``gather_latent`` does.

Combine ``mlx2_glm53_gathered_attention_v2_combine``: per ``(head, row)``
the weights ``a_s = exp(m_s - m)`` (0 for empty splits) are computed once
per split, then ``out = sum_s a_s o_s / sum_s a_s l_s`` in ascending split
order; an empty selection gives ``0 / 0 = NaN`` (round-1 failure contract).

Rows live on grid z and share nothing, so any grouping of rows is bitwise
identical to one launch per row. The law is new (fp32 MMA order), so it is
accepted cross-policy under the near-tie contract, never as bitwise to
``gathered_v1``.

Sources ABI: ``base (T0, 512)`` committed rows shared by every row of a
launch, one concatenated ``tails (sum d_r, 512)`` buffer and ``row_meta
(L, 4) int32 = (T0_r, d_r, tail_offset_r, pos_r)``; a row's id ``< T0_r``
reads ``base``, otherwise ``tails[tail_offset_r + id - T0_r]``.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

HEADS, DIM, TOPK = 64, 512, 2048
SCALE = 256**-0.5
SPLITS_SUPPORTED = (32, 64)
DIM_SPLITS_SUPPORTED = (2, 4)
# The configuration the policy runs; the bench matrix decides it (R1).
DEFAULT_SPLIT, DEFAULT_DIM_SPLIT = 64, 4  # wp1-matrix-b0a7759-a1: fastest point in every cell
_GATE: dict = {}
GATE_REPORT: dict = {}
_SCHEDULED = {"launches": 0, "rows": 0}

_HEADER = "constant constexpr float SCALE = 0.0625f;  // 256 ** -0.5, exact\n"

# Shared tile fill: KEYS keys x COLS dims starting at key k0 and dim c0 into
# tile (row stride COLS), as float4. Zero for -1 ids and slots past W.
_PARTIAL_SOURCE = r"""
const uint split = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.z;
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr uint NT = 256 * DSPLIT;
constexpr uint KT1 = 8 * DSPLIT;          // pass-1 keys per tile
constexpr uint DW1 = 512 / DSPLIT;        // pass-1 dims per stage
constexpr uint DO = 512 / DSPLIT;         // pass-2 output dims per simdgroup
constexpr uint NF = DO / 8;
threadgroup float tile[4096];
threadgroup float S[64 * SPLIT];
const int T0 = row_meta[row * 4 + 0];
const int DR = row_meta[row * 4 + 1];
const int TOFF = row_meta[row * 4 + 2];
const int LAST = T0 + DR - 1;
const int first = int(split) * SPLIT;
const device int* rid = ids + size_t(row) * WIDTH;
const device float* qrow = q + size_t(row) * 64 * 512;
const uint g = sg / DSPLIT;
const uint sub = sg % DSPLIT;

// pass 1: scores
for (uint t = 0; t < SPLIT / KT1; t++) {
    const int k0 = first + int(t * KT1);
    simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint c = 0; c < DSPLIT; c++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 1024; e += NT) {
            const uint kk = e / (DW1 / 4);
            const uint c4 = e % (DW1 / 4);
            const int k = k0 + int(kk);
            float4 v = float4(0.0f);
            if (k < WIDTH) {
                int id = rid[k];
                if (id >= 0) {
                    id = min(id, LAST);
                    const device float4* src = (const device float4*)(id < T0
                        ? base + size_t(id) * 512 : tails + size_t(TOFF + id - T0) * 512);
                    v = src[c * (DW1 / 4) + c4];
                }
            }
            ((threadgroup float4*)tile)[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device float* qp = qrow + size_t(8 * g) * 512 + c * DW1;
        const threadgroup float* kp = tile + (8 * sub) * DW1;
        for (uint j = 0; j < DW1 / 8; j++) {
            simdgroup_float8x8 qf;
            simdgroup_float8x8 kf;
            simdgroup_load(qf, qp + 8 * j, 512);
            simdgroup_load(kf, kp + 8 * j, DW1, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s, qf, kf, s);
        }
    }
    simdgroup_store(s, S + (8 * g) * SPLIT + t * KT1 + 8 * sub, SPLIT);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// statistics: 8 lanes per head, fixed xor tree
if (tid < 512) {
    const uint h = tid >> 3;
    const uint lane8 = tid & 7;
    threadgroup float* srow = S + h * SPLIT;
    float m = -INFINITY;
    uint bad = 0;
    for (uint i = 0; i < SPLIT / 8; i++) {
        const uint kk = lane8 + 8 * i;
        const int k = first + int(kk);
        const bool valid = k < WIDTH && rid[k] >= 0;
        const float x = valid ? srow[kk] * SCALE : -INFINITY;
        bad |= isnan(x) ? 1u : 0u;
        m = max(m, x);
    }
    for (ushort o = 1; o < 8; o <<= 1) {
        m = max(m, simd_shuffle_xor(m, o));
        bad |= simd_shuffle_xor(bad, o);
    }
    if (bad) m = NAN;
    const bool empty = m == -INFINITY;
    float l = 0.0f;
    for (uint i = 0; i < SPLIT / 8; i++) {
        const uint kk = lane8 + 8 * i;
        const int k = first + int(kk);
        const bool valid = k < WIDTH && rid[k] >= 0;
        const float p = (valid && !empty) ? metal::precise::exp(srow[kk] * SCALE - m) : 0.0f;
        srow[kk] = p;
        l += p;
    }
    for (ushort o = 1; o < 8; o <<= 1) l += simd_shuffle_xor(l, o);
    if (lane8 == 0) {
        const size_t slot = (size_t(row) * 64 + h) * SPLITS + split;
        partial_m[slot] = empty ? -INFINITY : m;
        partial_l[slot] = empty ? 0.0f : l;
    }
}

// pass 2: values
simdgroup_float8x8 o[NF];
for (uint j = 0; j < NF; j++) o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
for (uint t = 0; t < SPLIT / 8; t++) {
    const int k0 = first + int(t * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += NT) {
        const uint kk = e >> 7;
        const uint c4 = e & 127;
        const int k = k0 + int(kk);
        float4 v = float4(0.0f);
        if (k < WIDTH) {
            int id = rid[k];
            if (id >= 0) {
                id = min(id, LAST);
                const device float4* src = (const device float4*)(id < T0
                    ? base + size_t(id) * 512 : tails + size_t(TOFF + id - T0) * 512);
                v = src[c4];
            }
        }
        ((threadgroup float4*)tile)[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 pf;
    simdgroup_load(pf, S + (8 * g) * SPLIT + t * 8, SPLIT);
    const threadgroup float* vp = tile + sub * DO;
    for (uint j = 0; j < NF; j++) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, vp + 8 * j, 512);
        simdgroup_multiply_accumulate(o[j], pf, vf, o[j]);
    }
}
device float* op = partial_o + ((size_t(row) * 64 + 8 * g) * SPLITS + split) * 512 + sub * DO;
for (uint j = 0; j < NF; j++) simdgroup_store(o[j], op + 8 * j, SPLITS * 512);
"""

_COMBINE_SOURCE = r"""
const uint c = thread_position_in_grid.x;
const uint head = thread_position_in_grid.y;
const uint row = thread_position_in_grid.z;
const uint tid = thread_index_in_threadgroup;
threadgroup float a[SPLITS];
const size_t slots = (size_t(row) * 64 + head) * SPLITS;
float m = -INFINITY;
for (uint s = 0; s < SPLITS; s++)
    if (partial_l[slots + s] != 0.0f) m = max(m, partial_m[slots + s]);
if (tid < SPLITS) {
    const float ls = partial_l[slots + tid];
    a[tid] = ls != 0.0f ? metal::precise::exp(partial_m[slots + tid] - m) : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float l = 0.0f;
float acc = 0.0f;
for (uint s = 0; s < SPLITS; s++) {
    l += a[s] * partial_l[slots + s];
    acc += a[s] * partial_o[(slots + s) * 512 + c];
}
out[(size_t(row) * 64 + head) * 512 + c] = acc / l;
"""


def kernel_spec():
    return {
        "partial": {"name": "mlx2_glm53_gathered_attention_v2",
                    "input_names": ["q", "base", "tails", "ids", "row_meta"],
                    "output_names": ["partial_m", "partial_l", "partial_o"],
                    "header": _HEADER, "source": _PARTIAL_SOURCE},
        "combine": {"name": "mlx2_glm53_gathered_attention_v2_combine",
                    "input_names": ["partial_m", "partial_l", "partial_o"], "output_names": ["out"],
                    "header": _HEADER, "source": _COMBINE_SOURCE},
    }


def _check_config(split, dim_split):
    if split not in SPLITS_SUPPORTED or dim_split not in DIM_SPLITS_SUPPORTED:
        raise ValueError(f"GLM-5.3 fused_v2 supports SPLIT {SPLITS_SUPPORTED} and DIM_SPLIT {DIM_SPLITS_SUPPORTED}")
    if split % (8 * dim_split):
        raise ValueError("GLM-5.3 fused_v2 SPLIT must be a multiple of the pass-1 tile (8 * DIM_SPLIT keys)")


def num_splits(width, split=DEFAULT_SPLIT):
    return -(-width // split)


def launch_plan(rows, width, *, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT):
    """Grids of one v2 launch over ``rows`` query rows and ``width`` ids each."""
    _check_config(split, dim_split)
    if type(rows) is not int or rows < 1 or type(width) is not int or width < 1:
        raise ValueError("GLM-5.3 fused_v2 launch needs at least one row and a positive width")
    splits = num_splits(width, split)
    threads = 256 * dim_split
    return {
        "splits": splits,
        "threads": threads,
        "threadgroup_bytes": 4 * (4096 + HEADS * split),
        "partial": {"grid": (threads * splits, 1, rows), "threadgroup": (threads, 1, 1),
                    "output_shapes": [(rows, HEADS, splits)] * 2 + [(rows, HEADS, splits, DIM)]},
        "combine": {"grid": (DIM, HEADS, rows), "threadgroup": (DIM, 1, 1),
                    "output_shapes": [(rows, HEADS, DIM)]},
        "template": [("SPLIT", split), ("DSPLIT", dim_split), ("SPLITS", splits), ("WIDTH", width)],
    }


def workspace_bytes(rows, *, width=TOPK + 3, split=DEFAULT_SPLIT):
    """Partials of one launch: rows * splits * 64 * (512 + 2) fp32 words."""
    return rows * num_splits(width, split) * HEADS * (DIM + 2) * 4


@lru_cache(maxsize=2)
def _kernel(kind):
    return mx.fast.metal_kernel(**kernel_spec()[kind], ensure_row_contiguous=True)


def require_metal():
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("GLM-5.3 fused_v2 requires Metal on the default GPU device; no fallback law")


def _launch(q_rows, base, tails, ids, row_meta, split, dim_split):
    """The two kernels; q_rows (L, 64, 512) -> (L, 64, 512). No gate, no checks."""
    rows, width = ids.shape
    plan = launch_plan(rows, width, split=split, dim_split=dim_split)
    m, l, o = _kernel("partial")(
        inputs=[q_rows, base, tails, ids, row_meta], template=plan["template"],
        grid=plan["partial"]["grid"], threadgroup=plan["partial"]["threadgroup"],
        output_shapes=plan["partial"]["output_shapes"], output_dtypes=[mx.float32] * 3)
    (out,) = _kernel("combine")(
        inputs=[m, l, o], template=plan["template"],
        grid=plan["combine"]["grid"], threadgroup=plan["combine"]["threadgroup"],
        output_shapes=plan["combine"]["output_shapes"], output_dtypes=[mx.float32])
    _SCHEDULED["launches"] += 1
    _SCHEDULED["rows"] += rows
    return out


def launch(q_rows, base, tails, ids, row_meta, *, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT):
    """Validated ABI entry: q_rows (L, 64, 512), base (T0, 512), tails (>= 1, 512),
    ids (L, W) int32, row_meta (L, 4) int32 -> (L, 64, 512) fp32."""
    _check_config(split, dim_split)
    if q_rows.ndim != 3 or q_rows.shape[1:] != (HEADS, DIM):
        raise ValueError("GLM-5.3 fused_v2 is built for 64 heads of 512")
    rows = q_rows.shape[0]
    if ids.ndim != 2 or ids.shape[0] != rows or row_meta.shape != (rows, 4):
        raise ValueError("GLM-5.3 fused_v2 needs ids (L, W) and row_meta (L, 4)")
    if base.ndim != 2 or base.shape[1] != DIM or tails.ndim != 2 or tails.shape[1] != DIM or tails.shape[0] < 1:
        raise ValueError("GLM-5.3 fused_v2 needs base (T0, 512) and a non-empty tails (n, 512) buffer")
    return _launch(q_rows.astype(mx.float32), base.astype(mx.float32), tails.astype(mx.float32),
                   ids.astype(mx.int32), row_meta.astype(mx.int32), split, dim_split)


def checked_launch(q_rows, base, tails, ids, row_meta, *, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT):
    """``launch`` behind the Metal requirement and the first-use gate (fails closed)."""
    require_metal()
    self_check(split=split, dim_split=dim_split)
    return launch(q_rows, base, tails, ids, row_meta, split=split, dim_split=dim_split)


# ---------------------------------------------------------------------------
# float64 oracle and the first-use gate (design 1.1 test (a)).

def oracle(q_rows, base, tails, ids, row_meta):
    """float64 reference of the ABI: (L, 64, 512) numpy, NaN rows for empty selections."""
    import numpy as np

    q = np.asarray(q_rows, np.float64)
    base, tails = np.asarray(base, np.float64), np.asarray(tails, np.float64)
    ids, meta = np.asarray(ids), np.asarray(row_meta)
    out = np.full(q.shape, np.nan)
    for r in range(q.shape[0]):
        t0, d, off, _ = (int(v) for v in meta[r])
        row = ids[r][ids[r] >= 0]
        if not len(row):
            continue
        keys = np.concatenate([base[:t0], tails[off : off + d]])[np.minimum(row, t0 + d - 1)]
        s = q[r] @ keys.T * SCALE
        p = np.exp(s - s.max(-1, keepdims=True))
        out[r] = (p / p.sum(-1, keepdims=True)) @ keys
    return out


def oracle_errors(got, want):
    """(max abs, rel RMS) over the finite oracle entries; None when NaN placement differs."""
    import numpy as np

    got = np.asarray(got, np.float64)
    nan = np.isnan(want)
    if not np.array_equal(np.isnan(got), nan):
        return None
    if nan.all():
        return 0.0, 0.0
    diff = got[~nan] - want[~nan]
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(want[~nan] ** 2)))


def gate_cases(seed=5303):
    """Synthetic unit-scale cases: a decode row at W = 2051, half-filled and empty
    rows, a tree base + distinct tails, and dense causal rows at pos 0, 3, 2047."""
    import numpy as np

    rng = np.random.default_rng(seed)
    width = TOPK + 3
    cases = []
    total = 4096
    latent = rng.normal(size=(total, DIM)).astype(np.float32)
    ids = np.full((3, width), -1, np.int32)
    ids[0] = np.sort(rng.choice(total, size=width, replace=False))
    ids[1, : width // 2] = np.sort(rng.choice(total, size=width // 2, replace=False))
    meta = np.array([[total, 0, 0, total - 1]] * 3, np.int32)
    q = rng.normal(size=(3, HEADS, DIM)).astype(np.float32)
    cases.append(("plain", q, latent, np.zeros((1, DIM), np.float32), ids, meta))
    t0 = total - 7
    tails = rng.normal(size=(2 + 5, DIM)).astype(np.float32)
    ids = np.full((2, width), -1, np.int32)
    ids[0, :width] = np.sort(np.concatenate([rng.choice(t0, size=width - 2, replace=False), [t0, t0 + 1]]))
    ids[1, :width] = np.sort(np.concatenate([rng.choice(t0, size=width - 5, replace=False), t0 + np.arange(5)]))
    meta = np.array([[t0, 2, 0, t0 + 1], [t0, 5, 2, t0 + 4]], np.int32)
    q = rng.normal(size=(2, HEADS, DIM)).astype(np.float32)
    cases.append(("tails", q, latent[:t0], tails, ids, meta))
    positions = np.array([0, 3, TOPK - 1], np.int32)
    ids = np.asarray(dense_row_ids_np(positions, width))
    meta = np.array([[TOPK, 0, 0, int(p)] for p in positions], np.int32)
    q = rng.normal(size=(3, HEADS, DIM)).astype(np.float32)
    cases.append(("dense", q, latent[:TOPK], np.zeros((1, DIM), np.float32), ids, meta))
    return cases


def dense_row_ids_np(positions, width):
    import numpy as np

    arange = np.arange(width, dtype=np.int32)
    positions = np.asarray(positions, np.int32)
    return np.where(arange[None] <= positions[:, None], arange[None], -1).astype(np.int32)


def ops_reference(q_rows, base, tails, ids, row_meta):
    """The gathered_v1 ops law (gather + SDPA, one row per call) on the same ABI operands."""
    import numpy as np

    from . import glm53_sparse_attention as sa

    meta = np.asarray(row_meta)
    rows = []
    for r in range(meta.shape[0]):
        t0, d, off, _ = (int(x) for x in meta[r])
        source = (sa.TreeLatentView(mx.array(base[:t0]), mx.array(tails[off : off + d]), t0) if d
                  else mx.array(base[:t0])[None, None])
        q = mx.array(q_rows[r : r + 1]).transpose(1, 0, 2)[None]
        rows.append(sa.gathered_attention(q, *sa.gather_latent(source, mx.array(ids[r : r + 1])[None])))
    return np.array(mx.concatenate(rows, axis=2))[0].transpose(1, 0, 2)


def self_check(*, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT, abs_bound=1e-5, rel_rms_bound=1e-6):
    """First-use GPU gate against float64; sticky verdict per configuration.

    Bound per case: max abs <= 1e-5, and rel RMS <= max(1e-6, the gathered_v1
    ops law's rel RMS on the same case). The 1e-6 rel-RMS line sits at the fp32
    floor of a 2051-row weighted average (the ops law itself exceeds it on
    unit-scale cases, e.g. 5385578a2's attention receipt), so the kernel is held
    to "no worse than the reference law against float64" where that is above
    1e-6. Both arms' errors are kept in GATE_REPORT.
    """
    key = (split, dim_split)
    verdict = _GATE.get(key)
    if verdict is True:
        return True
    if verdict is False:
        raise RuntimeError("GLM-5.3 fused_v2 kernel failed its first-use gate")
    require_metal()
    ok = True
    report = {}
    for name, q, base, tails, ids, meta in gate_cases():
        got = launch(mx.array(q), mx.array(base), mx.array(tails), mx.array(ids), mx.array(meta),
                     split=split, dim_split=dim_split)
        mx.eval(got)
        want = oracle(q, base, tails, ids, meta)
        errors = oracle_errors(got, want)
        ops = oracle_errors(ops_reference(q, base, tails, ids, meta), want)
        rel = max(rel_rms_bound, ops[1]) if ops is not None else rel_rms_bound
        passed = errors is not None and errors[0] <= abs_bound and errors[1] <= rel
        report[name] = {"kernel": errors, "ops_reference": ops, "rel_rms_bound": rel, "passed": passed}
        ok &= passed
    GATE_REPORT[f"split{split}_dsplit{dim_split}"] = report
    _GATE[key] = ok
    if not ok:
        raise RuntimeError("GLM-5.3 fused_v2 kernel failed its first-use gate")
    return True


def route_status():
    return {
        "first_use_gate": {f"split{s}_dsplit{d}": ("passed" if v else "failed") for (s, d), v in _GATE.items()}
        or "not_run",
        "config": {"split": DEFAULT_SPLIT, "dim_split": DEFAULT_DIM_SPLIT},
        "gate_report": dict(GATE_REPORT),
        "scheduled": dict(_SCHEDULED),
    }
