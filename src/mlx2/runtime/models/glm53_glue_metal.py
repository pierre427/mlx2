"""Original single-launch GLM-5.3 decode glue: MoE routing and the KDA gate epilogue.

Both kernels reproduce the reference op chains of ``Glm53GGUFModel._mlp`` and
``Glm53GGUFModel._kda`` bit for bit, one IEEE operation for one MLX primitive,
in the same order and without fusing a multiply into an add:

* ``mx.sigmoid`` is MLX's ``Sigmoid`` functor (``1 / (1 + precise::exp(|x|))``,
  reflected for x >= 0), the same expression ``precise_ops.sigmoid`` uses.
* ``mx.exp`` is ``metal::precise::exp``.
* ``mx.argsort`` is MLX's stable merge sort under ``LessThan`` (NaN sorts
  last). Ranking each expert by "strictly less, or equal and lower index"
  yields exactly the stable sorted positions, so the first TOPK ranks equal
  ``argsort(-choice)[..., :TOPK]`` including exact ties and NaNs.
* ``x.sum(-1)`` over a contiguous 8-wide row is MLX's ``row_reduce_small``
  ``thread_reduce``: ``total = v + total`` from 0, left to right.

Callers keep the reference ops off Metal or for unsupported dtypes/shapes;
on Metal the GPU tests in tests/test_glm53_decode_levers_cpu.py compare each
kernel with the reference chain bitwise.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import mlx.core as mx


def _sigmoid(x, out):
    """MLX's Sigmoid functor, spelled as in precise_ops (mx.sigmoid's bytes)."""
    return (
        f"{{ const float y_ = 1 / (1 + metal::precise::exp(metal::abs({x})));"
        f" {out} = ({x} < 0) ? y_ : 1 - y_; }}"
    )


_ROUTER_SOURCE = r"""
const uint i = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y;
threadgroup float v[E];
threadgroup float s[E];
threadgroup float sel[TOPK];
threadgroup uint sel_id[TOPK];
if (i < E) {
    const float x = logits[size_t(row) * E + i];
    float p;
    SIGMOID_P
    s[i] = p;
    // argsort(-(scores + bias)): one rounding for the add, an exact negation.
    const float choice = p + bias[i];
    v[i] = -choice;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (i < E) {
    const float vi = v[i];
    const bool ni = metal::isnan(vi);
    uint rank = 0;
    for (uint j = 0; j < E; j++) {
        const float vj = v[j];
        const bool nj = metal::isnan(vj);
        // MLX LessThan: NaN is greater than every number.
        const bool less_ji = (ni | nj) ? ((!nj) & ni) : (vj < vi);
        const bool less_ij = (ni | nj) ? ((!ni) & nj) : (vi < vj);
        rank += (less_ji || (!less_ij && j < i)) ? 1u : 0u;
    }
    if (rank < TOPK) {
        sel[rank] = s[i];
        sel_id[rank] = i;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (i < TOPK) {
    float total = 0.0f;
    for (uint k = 0; k < TOPK; k++) {
        total = sel[k] + total;
    }
    const float normalized = sel[i] / total;
    ids[size_t(row) * TOPK + i] = sel_id[i];
    weights[size_t(row) * TOPK + i] = normalized * SCALE;
}
"""


@lru_cache(maxsize=4)
def _router_kernel(scale):
    source = _ROUTER_SOURCE.replace("SIGMOID_P", _sigmoid("x", "p")).replace(
        "SCALE", repr(float(scale)) + "f"
    )
    return mx.fast.metal_kernel(
        name="mlx2_glm53_router_v1_" + repr(float(scale)).replace(".", "_"),
        input_names=["logits", "bias"],
        output_names=["ids", "weights"],
        source=source,
        ensure_row_contiguous=True,
    )


def router_admits(logits, bias, topk):
    """True when the fused router reproduces the reference chain for these inputs."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    experts = logits.shape[-1]
    return (
        logits.dtype == mx.float32
        and bias.dtype == mx.float32
        and bias.shape == (experts,)
        and 0 < topk <= experts <= 1024
        and logits.size > 0
    )


def fused_router(logits, bias, *, topk=8, scale=2.5):
    """(ids uint32, weights float32), each [..., topk], in one launch.

    Equal bit for bit to::

        scores = mx.sigmoid(logits)
        ids = mx.argsort(-(scores + bias), axis=-1)[..., :topk].astype(mx.uint32)
        w = mx.take_along_axis(scores, ids, axis=-1)
        w = w / w.sum(-1, keepdims=True) * scale
    """
    experts = logits.shape[-1]
    lead = logits.shape[:-1]
    rows = logits.size // experts
    threads = (experts + 31) // 32 * 32
    ids, weights = _router_kernel(scale)(
        inputs=[logits.reshape(rows, experts), bias],
        template=[("E", experts), ("TOPK", topk)],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, topk), (rows, topk)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return ids.reshape(*lead, topk), weights.reshape(*lead, topk)


_KDA_GATE_SOURCE = r"""
const uint t = thread_position_in_grid.x;
if (t >= TOTAL) return;
const uint c = t % INNER;
// decay = exp(-5 * sigmoid((a + dt_bias) * ssm_a[head])), one rounding per op.
const float shifted = a[t] + dt_bias[c];
const float scaled = shifted * ssm_a[c / DIM];
float sa;
SIGMOID_SA
const float negated = -5.0f * sa;
decay[t] = metal::precise::exp(negated);
float sg;
SIGMOID_SG
gate_out[t] = sg;
if (t < BETA_TOTAL) {
    float sb;
    SIGMOID_SB
    beta_out[t] = sb;
}
"""


@lru_cache(maxsize=1)
def _kda_gate_kernel():
    source = (
        _KDA_GATE_SOURCE.replace("SIGMOID_SA", _sigmoid("scaled", "sa"))
        .replace("SIGMOID_SG", _sigmoid("gate[t]", "sg"))
        .replace("SIGMOID_SB", _sigmoid("beta[t]", "sb"))
    )
    return mx.fast.metal_kernel(
        name="mlx2_glm53_kda_gate_v1",
        input_names=["a", "dt_bias", "ssm_a", "beta", "gate"],
        output_names=["decay", "beta_out", "gate_out"],
        source=source,
        ensure_row_contiguous=True,
    )


def kda_gate_admits(a, dt_bias, ssm_a, beta, gate, *, heads, dim):
    """True when the fused KDA gate epilogue reproduces the reference chain."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    inner = heads * dim
    if any(x.dtype != mx.float32 for x in (a, dt_bias, ssm_a, beta, gate)):
        return False
    rows = a.size // inner if inner else 0
    return (
        rows > 0
        and a.size == gate.size == rows * inner
        and a.shape[-1] == gate.shape[-1] == inner
        and beta.shape[-1] == heads
        and beta.size == rows * heads
        and dt_bias.size == inner
        and ssm_a.size == heads
    )


def fused_kda_gate(a, dt_bias, ssm_a, beta, gate, *, heads, dim):
    """(decay, beta, sigmoid(gate)) of one KDA layer in one launch.

    ``a`` and ``gate`` are [..., heads*dim], ``beta`` is [..., heads]; outputs
    keep those shapes. Equal bit for bit to::

        decay = mx.exp(-5 * mx.sigmoid((a + dt_bias) * ssm_a[head]))
        beta = mx.sigmoid(beta)
        gate = mx.sigmoid(gate)
    """
    inner = heads * dim
    rows = a.size // inner
    total = rows * inner
    decay, beta_out, gate_out = _kda_gate_kernel()(
        inputs=[
            a.reshape(rows, inner),
            dt_bias.reshape(inner),
            ssm_a.reshape(heads),
            beta.reshape(rows, heads),
            gate.reshape(rows, inner),
        ],
        template=[
            ("TOTAL", total),
            ("INNER", inner),
            ("DIM", dim),
            ("BETA_TOTAL", rows * heads),
        ],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(rows, inner), (rows, heads), (rows, inner)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return decay.reshape(a.shape), beta_out.reshape(beta.shape), gate_out.reshape(gate.shape)


# ---- glm53_router=fused_v2: the ffn_gate_inp F32 gemv and the router, one launch ----
#
# The router logits are ``x @ ffn_gate_inp.T`` with an F32 [E, K] row-major
# matrix: MLX 0.32.4's GEMVKernel<float, BM=1, BN=8, SM=1, SN=32, TM=4, TN=4>
# (gemv.h; the geometry Matmul::eval_gpu selects for N=144 < 4096 and
# K=4096 >= 16N). Every logit keeps that kernel's order (per-lane fused
# multiply-add chain over k = (32 s + l) 4 + 1024 i + tn, simd_shuffle_down
# 16..1, simdgroup 0 adds simdgroups 1..7 in order); GEMV_RPT logits per
# threadgroup (MLX: 4) only spreads those chains over more threadgroups. Each
# threadgroup publishes its logits with device atomics, and the last
# threadgroup to arrive (MSL 3.2+ device-scope fences, a per-launch counter it
# resets) runs _ROUTER_SOURCE's selection law verbatim on the gathered row.

_GEMV_TILE = r"""
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const uint tile = threadgroup_position_in_grid.x;
threadgroup float gemv_part[RPT * BN];
threadgroup uint last_shared;
threadgroup float lg[E];
// Simdgroup 8r + s is MLX gemv simdgroup s of row tile*RPT + r; lane v = 32 s + l
// accumulates k = 4v + 1024 it + tn in order (MLX's TM=4 lanes run this same
// chain for each of their rows).
{
    const uint gr = sg / BN, gs = sg % BN;
    // F32 or exact-BF16 weights (glm53_f32_weights): same float operand values.
    const device auto *mat = w + size_t(tile * RPT + gr) * K;
    float result = 0.0f;
    int bn = int(gs * 32 + lane) * TN;
    for (int it = 0; it < K / BLOCKN; ++it) {
        for (int tn = 0; tn < TN; tn++) result = glm_gemv_acc(result, mat[bn + tn], xin[bn + tn]);
        bn += BLOCKN;
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);
    if (lane == 0) gemv_part[gr * BN + gs] = result;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
device atomic_uint *published = (device atomic_uint *)(logits);
if (lid < RPT) {
    float r = gemv_part[lid * BN];
    for (int sgn = 1; sgn < BN; sgn++) r += gemv_part[lid * BN + sgn];
    atomic_store_explicit(published + tile * RPT + lid, as_type<uint>(r), memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
if (lid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    device atomic_uint *count = (device atomic_uint *)(counter);
    const uint arrived = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    uint last = arrived == uint(E / RPT - 1) ? 1u : 0u;
    if (last) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_store_explicit(count, 0u, memory_order_relaxed);
    }
    last_shared = last;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (last_shared == 0u) return;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
for (uint e = lid; e < E; e += RPT * BN * 32) {
    lg[e] = as_type<float>(atomic_load_explicit(published + e, memory_order_relaxed));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_GEMV_ACC_HEADER = r"""
#pragma clang fp contract(on)
inline float glm_gemv_acc(float acc, float a, float b) { acc += a * b; return acc; }
#pragma clang fp contract(off)
"""

_ROUTER_PROLOGUE = (
    "const uint i = thread_position_in_threadgroup.x;\n"
    "const uint row = threadgroup_position_in_grid.y;\n"
)
_ROUTER_READ = "const float x = logits[size_t(row) * E + i];"
GEMV_BN, GEMV_TM, GEMV_TN = 8, 4, 4
# Router logits per threadgroup (1 or 4; MLX itself uses 4 per threadgroup).
# Values do not depend on it: each logit's chains and reductions are fixed.
GEMV_RPT = 1


def _router_v2_source():
    """The gemv tile, the tail handshake, then _ROUTER_SOURCE with its row read
    from the gathered logits (row 0; one B1 row per launch)."""
    if not _ROUTER_SOURCE.lstrip("\n").startswith(_ROUTER_PROLOGUE) or _ROUTER_READ not in _ROUTER_SOURCE:
        raise RuntimeError("glm53 router v1 source changed; fused_v2 transcription refused")
    body = _ROUTER_SOURCE.lstrip("\n")[len(_ROUTER_PROLOGUE):]
    body = body.replace(_ROUTER_READ, "const float x = lg[i];")
    return _GEMV_TILE + "const uint i = lid;\nconst uint row = 0;\n" + body


@lru_cache(maxsize=8)
def _router_v2_kernel(scale, bf16_weight=False):
    source = _router_v2_source().replace("SIGMOID_P", _sigmoid("x", "p")).replace(
        "SCALE", repr(float(scale)) + "f"
    )
    return mx.fast.metal_kernel(
        name="mlx2_glm53_router_gemv_v2_" + repr(float(scale)).replace(".", "_")
        + ("_bf16w" if bf16_weight else ""),
        input_names=["xin", "w", "bias", "counter"],
        output_names=["ids", "weights", "logits"],
        header=_GEMV_ACC_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


COUNTER_WORDS = 16
# Inside a traced compile region the counters come from a block the region
# receives as an input, fresh on every call (glm53_compile_regions): a
# counter made while tracing would be one captured buffer shared by every
# replay, and two replays no data dependency orders could run concurrently.
_COUNTER_SUPPLY = ContextVar("glm53_arrival_counters", default=None)


def arrival_counter():
    """A zeroed uint32 [16] arrival counter for one last-threadgroup launch
    (count, a distinct id, padding: a device buffer, never a mergeable
    scalar). Never shared by two launches that MLX could run concurrently."""
    supply = _COUNTER_SUPPLY.get()
    if supply is not None:
        block, used = supply
        index = used[0]
        if (index + 1) * COUNTER_WORDS > block.size:
            raise RuntimeError("compile region arrival-counter block exhausted")
        used[0] += 1
        return block[index * COUNTER_WORDS:(index + 1) * COUNTER_WORDS]
    _COUNTER_IDS[0] += 1
    return mx.array([0, _COUNTER_IDS[0]] + [0] * (COUNTER_WORDS - 2), dtype=mx.uint32)


@contextmanager
def counter_supply(block):
    """Serve arrival_counter() from ``block`` (uint32 [k*16], zeroed) while tracing."""
    token = _COUNTER_SUPPLY.set((block, [0]))
    try:
        yield
    finally:
        _COUNTER_SUPPLY.reset(token)


def router_counter():
    return arrival_counter()


_COUNTER_IDS = [0]


def router_gemv_admits(x, weight, bias, topk):
    """True when fused_router_gemv reproduces ``x @ weight.T`` then fused_router."""
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if weight is None or weight.ndim != 2:
        return False
    experts, k = weight.shape
    return (
        x.dtype == mx.float32
        and weight.dtype in (mx.float32, mx.bfloat16)
        and bias.dtype == mx.float32
        and bias.shape == (experts,)
        and x.size == k
        and x.shape[-1] == k
        and experts % GEMV_TM == 0
        and experts % GEMV_RPT == 0
        and experts < 4096
        and k % (GEMV_BN * 32 * GEMV_TN) == 0
        and k >= 16 * experts
        and 0 < topk <= experts <= GEMV_RPT * GEMV_BN * 32
    )


def fused_router_gemv(x, weight, bias, counter=None, *, topk=8, scale=2.5, with_logits=False):
    """(ids uint32, weights float32), each [..., topk], for one row in one launch.

    Equal bit for bit to ``fused_router(x @ weight.T, bias)`` for x [..., K]
    holding one row, weight F32 [E, K] row-major (router_gemv_admits). A
    BF16 ``weight`` must hold the F32 matrix's values exactly (glm53_f32_weights
    makes it so): each element widens to the same float operand, so the
    outputs equal the F32 launch bit for bit with half the weight bytes.
    ``with_logits`` also returns the gathered logits [E] (tests). Each launch
    gets a fresh arrival counter by default: a counter shared by two launches
    that no data dependency orders could mix their arrivals.
    """
    if counter is None:
        counter = router_counter()
    experts, k = weight.shape
    lead = x.shape[:-1]
    ids, weights, logits = _router_v2_kernel(scale, weight.dtype == mx.bfloat16)(
        inputs=[x.reshape(k), weight, bias, counter],
        template=[("E", experts), ("TOPK", topk), ("K", k), ("BN", GEMV_BN), ("RPT", GEMV_RPT),
                  ("TN", GEMV_TN), ("BLOCKN", GEMV_BN * 32 * GEMV_TN)],
        grid=((experts // GEMV_RPT) * (GEMV_RPT * GEMV_BN * 32), 1, 1),
        threadgroup=(GEMV_RPT * GEMV_BN * 32, 1, 1),
        output_shapes=[(1, topk), (1, topk), (experts,)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32],
    )
    if with_logits:
        return ids.reshape(*lead, topk), weights.reshape(*lead, topk), logits
    return ids.reshape(*lead, topk), weights.reshape(*lead, topk)
