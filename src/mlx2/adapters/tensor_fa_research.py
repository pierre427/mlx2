"""Approximate tensor-API dense attention: isolated research experiment.

APPROXIMATE, default-off, UNQUALIFIED, UNSELECTED, never observed-used.
Nothing in the scheduler, cache, serving or model code imports this module,
and it is not a production route. Kernel body adapted from ggml
(llama.cpp PR 29570 @ 7f0d4d5e, MIT; see provenance/tensor-fa-research.json
and provenance/tensor-fa-research.NOTICE).

Arithmetic (deliberately NOT bit-identical to exact attention): queries are
scaled then rounded to half; Q*K^T accumulates in fp32 through tensor_ops;
probabilities are half(exp(s - M)); the row sum S is fp32 over those rounded
values; the running max M moves only when it grows by more than 8, with a
per-row scale; O is rescaled in a key block when any row of the threadgroup
rescaled (a race-free per-simdgroup flag); output is fp32 O / S.

Contract (refused, never approximated):
  * B1; q [1, Hq, L, D] float16/float32; k, v [1, Hkv, N, D] float16 logical
    arrays; D in {128, 256}; Hq % Hkv == 0; N % 64 == 0; 1 <= L <= 4096;
    N bounded so every int32 uniform and the causal round-up stay in range;
  * explicit Python float ``scale`` that stays finite, positive and normal
    after narrowing to fp32; explicit bool ``causal``; explicit int
    ``q_start`` with 0 <= q_start and q_start + L <= N (row i sits at
    absolute position q_start + i and, when causal, sees keys 0..q_start + i,
    so no admitted row is fully masked);
  * no mask, bias/ALiBi, softcap, sinks, quantized/packed tuples, and no
    stride/offset/index/selection/ragged metadata keywords.
Returned dtype: float32 [1, Hq, L, D].

What admission CANNOT verify (declared in every plan, never claimed):
  * memory layout: MLX exposes no public strides here, so a sliced or
    transposed view is accepted as its LOGICAL dense array; the native call
    passes ensure_row_contiguous=True and MLX may copy/materialize it. That
    conversion cost is outside kernel-only timing and must be measured
    separately;
  * key-set identity: a shape cannot prove that k/v hold the FULL logical key
    set. The caller must pass the complete dense keys for these query rows.
    A pre-gathered sparse/indexed subset (e.g. Flash-Next QSA) is a different
    computation and must never be passed here;
  * values: no finite-tensor validation (large or non-finite q/k/v, or
    half-scaled query overflow, need the native finite gate).

Host functions never call GPU APIs or evaluate tensors. The native entry
needs ``allow_research_metal=True``, the default GPU device, Metal, and an
M5-class (``applegpu_g17*``) device; the kernel is compiled lazily (once,
under a lock) on the first admitted call. Every threadgroup adds 1 to an
atomic engagement counter; ``confirm_engagement`` counts a launch only when
the counter read back on the GPU equals the planned threadgroup count.
"""

from __future__ import annotations

import itertools
import math
import threading
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import mlx.core as mx
import numpy as np

SCHEMA = "mlx2.tensor-fa-research.v1"
# Immutable: nothing may flip these at runtime.
STATE = MappingProxyType({"default": "off", "approximate": True, "exact": False, "qualified": False,
                          "selected": False, "observed_used": False, "route": None})
Q_ROWS, C_KEYS, N_SIMDGROUPS, SIMD_WIDTH = 32, 64, 8, 32       # upstream NQPSG, NCPSG, NSG
THREADS = SIMD_WIDTH * N_SIMDGROUPS
HEAD_DIMS = (128, 256)
MAX_QUERY = 4096
MAX_HEADS = 128
MAX_THREADGROUP_MEMORY = 32768
INT32_MAX = 2**31 - 1
# Largest N such that q_start + rows_end + C - 1 (the causal round-up) and
# N itself fit int32: q_start + rows_end <= N, so N + C - 1 <= INT32_MAX.
# MLX shapes are themselves int32, so for real arrays this and the round-up
# check below are defence in depth (tests exercise them with a lowered bound).
MAX_KV = (INT32_MAX - C_KEYS + 1) // C_KEYS * C_KEYS
RESCALE_GROWTH = 8.0
MASKED_SCORE = -np.finfo(np.float32).max                       # finite (fast math assumes no inf)
_QUERY_DTYPES = {mx.float16: "float16", mx.float32: "float32"}
LAYOUT_CONTRACT = MappingProxyType({
    "row_contiguity_host_verified": False,
    "materialization": ("native call uses ensure_row_contiguous=True; MLX may copy a non-contiguous view; "
                        "that cost is outside kernel-only timing and must be measured separately"),
    "key_set": ("caller contract: k/v are the complete dense logical keys/values for these query rows; "
                "shape and dtype cannot prove it, and a pre-gathered sparse/indexed subset is refused by contract"),
    "values": "not validated on the host; large or non-finite q/k/v and half-scaled overflow need the native finite gate",
})


class TensorFARefused(ValueError):
    """The request is outside the experiment's exact contract."""


class TensorFAUnavailable(RuntimeError):
    """The native experiment is not admitted here."""


# ---------------------------------------------------------------- admission

def threadgroup_memory_bytes(dim: int) -> int:
    """sq half[Q*D] + ss float[Q*C] + sp half[Q*C] + sr float[Q] + sgf int[NSG], padded to 16."""
    raw = Q_ROWS * dim * 2 + Q_ROWS * C_KEYS * (4 + 2) + Q_ROWS * 4 + N_SIMDGROUPS * 4
    return (raw + 15) // 16 * 16


@dataclass(frozen=True)
class TensorFAPlan:
    q_heads: int
    kv_heads: int
    query_len: int
    kv_len: int
    dim: int
    q_dtype: str
    scale: float                     # as supplied
    scale_fp32: float                # what the kernel and host mirror use
    causal: bool
    q_start: int
    query_blocks: int
    threadgroups: int
    grid: Tuple[int, int, int]
    threadgroup: Tuple[int, int, int]
    threadgroup_memory: int
    kv_end: Tuple[int, ...]          # per query block: keys the kernel loop reads (multiple of C)
    # default_factory: a mappingproxy default is rejected as mutable by some Python versions;
    # compare=False keeps the frozen plan hashable (the contract is a module constant).
    layout_contract: Mapping[str, Any] = field(default_factory=lambda: LAYOUT_CONTRACT, compare=False)

    def visible_keys(self, row: int) -> range:
        """Logical keys query ``row`` attends to."""
        if not 0 <= row < self.query_len:
            raise TensorFARefused(f"row {row} is outside 0..{self.query_len - 1}")
        return range(0, self.q_start + row + 1) if self.causal else range(self.kv_len)


_NAMED_EXTRAS = ("mask", "bias", "softcap", "sinks", "alibi")


def _reject_extras(extras: Mapping[str, Any]) -> None:
    for name in _NAMED_EXTRAS:
        value = extras.get(name)
        if value is not None and value is not False:
            raise TensorFARefused(f"{name} is not implemented by this experiment")
    unknown = sorted(set(extras) - set(_NAMED_EXTRAS))
    if unknown:
        raise TensorFARefused(f"unsupported metadata {unknown}: stride/offset/index/selection/ragged "
                              "descriptions are not accepted")


def _fp32_scale(scale) -> float:
    if type(scale) is not float or not math.isfinite(scale) or scale <= 0:
        raise TensorFARefused("scale must be an explicit finite positive Python float")
    with np.errstate(over="ignore", under="ignore"):
        narrowed = np.float32(scale)
    if not np.isfinite(narrowed) or narrowed < np.finfo(np.float32).tiny:
        raise TensorFARefused("scale must stay finite, positive and normal after fp32 narrowing")
    return float(narrowed)


def plan_tensor_fa(q, k, v, *, scale, causal, q_start, **extras) -> TensorFAPlan:
    """Pure host admission (shapes/dtypes/metadata only; no GPU API, no eval)."""
    _reject_extras(extras)
    for name, x in (("k", k), ("v", v)):
        if isinstance(x, (tuple, list)):
            raise TensorFARefused(f"{name} is a quantized/packed tuple; only materialized float16 KV is accepted")
    if not all(isinstance(x, mx.array) for x in (q, k, v)):
        raise TensorFARefused("q, k and v must be mx.array (sparse/indexed selection objects are refused)")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise TensorFARefused("q, k, v must be rank 4 [B, H, L|N, D]")
    batch, hq, length, dim = map(int, q.shape)
    if batch != 1 or int(k.shape[0]) != 1 or int(v.shape[0]) != 1:
        raise TensorFARefused("only B1 is supported (no batch or ragged metadata)")
    hkv, kv_len, k_dim = map(int, k.shape[1:])
    if tuple(v.shape) != tuple(k.shape):
        raise TensorFARefused("v must have k's shape (DK == DV)")
    if dim not in HEAD_DIMS or k_dim != dim:
        raise TensorFARefused(f"head dim must be one of {HEAD_DIMS} for q, k and v")
    if not 1 <= hkv <= hq <= MAX_HEADS or hq % hkv:
        raise TensorFARefused("heads must satisfy 1 <= Hkv <= Hq <= 128 and Hq % Hkv == 0")
    if q.dtype not in _QUERY_DTYPES:
        raise TensorFARefused("q must be float16 or float32")
    if k.dtype != mx.float16 or v.dtype != mx.float16:
        raise TensorFARefused("k and v must be materialized float16 (no automatic dequantization)")
    if not 1 <= length <= MAX_QUERY:
        raise TensorFARefused(f"query length must be 1..{MAX_QUERY}")
    if kv_len < C_KEYS or kv_len % C_KEYS:
        raise TensorFARefused(f"KV length must be a positive multiple of {C_KEYS}")
    if kv_len > MAX_KV:
        raise TensorFARefused(f"KV length must be <= {MAX_KV} so int32 uniforms and the causal round-up fit")
    scale_fp32 = _fp32_scale(scale)
    if type(causal) is not bool:
        raise TensorFARefused("causal must be an explicit bool")
    if type(q_start) is not int or q_start < 0 or q_start + length > kv_len:
        raise TensorFARefused("q_start must be an int with 0 <= q_start and q_start + L <= N")
    memory = threadgroup_memory_bytes(dim)
    if memory > MAX_THREADGROUP_MEMORY:
        raise TensorFARefused("threadgroup memory bound exceeded")
    blocks = -(-length // Q_ROWS)
    kv_end = []
    for block in range(blocks):
        last = min((block + 1) * Q_ROWS, length)            # exclusive row bound
        end = min(kv_len, -(-(q_start + last) // C_KEYS) * C_KEYS) if causal else kv_len
        if q_start + last + C_KEYS - 1 > INT32_MAX:          # the kernel's int round-up
            raise TensorFARefused("causal round-up exceeds int32")
        kv_end.append(end)
    return TensorFAPlan(hq, hkv, length, kv_len, dim, _QUERY_DTYPES[q.dtype], scale, scale_fp32, causal, q_start,
                        blocks, blocks * hq, (blocks * THREADS, hq, 1), (THREADS, 1, 1), memory, tuple(kv_end))


# ---------------------------------------------------------------- host diagnostic mirror

def host_mirror(q, k, v, plan: TensorFAPlan, *, trace=None) -> np.ndarray:
    """DIAGNOSTIC host model of the kernel's rounding/rescale law. NOT a native oracle.

    The tensor_ops reduction order is not modelled (numpy fp32 matmul), so its
    bits are not the kernel's. ``trace`` (a list) receives one record per
    rescale event: (query block, key block, row).
    """
    qh = (np.asarray(q, dtype=np.float32)[0] * np.float32(plan.scale_fp32)).astype(np.float16).astype(np.float32)
    kh = np.asarray(k, dtype=np.float16)[0].astype(np.float32)
    vh = np.asarray(v, dtype=np.float16)[0].astype(np.float32)
    group = plan.q_heads // plan.kv_heads
    out = np.zeros((plan.q_heads, plan.query_len, plan.dim), dtype=np.float32)
    for head in range(plan.q_heads):
        kv = head // group
        for block in range(plan.query_blocks):
            rows = range(block * Q_ROWS, min((block + 1) * Q_ROWS, plan.query_len))
            M = np.full(len(rows), -np.finfo(np.float32).max / 2, dtype=np.float32)
            S = np.zeros(len(rows), dtype=np.float32)
            O = np.zeros((len(rows), plan.dim), dtype=np.float32)
            qblock = qh[head, rows.start:rows.stop]
            for kb, ic in enumerate(range(0, plan.kv_end[block], C_KEYS)):
                s = qblock @ kh[kv, ic:ic + C_KEYS].T                    # fp32 scores
                if plan.causal:
                    qpos = plan.q_start + np.arange(rows.start, rows.stop)[:, None]
                    kpos = np.arange(ic, ic + C_KEYS)[None, :]
                    s = np.where(kpos > qpos, MASKED_SCORE, s).astype(np.float32)
                m = np.maximum(M, s.max(axis=1))
                grow = m > M + np.float32(RESCALE_GROWTH)
                ms = np.where(grow, np.exp(M - m), np.float32(1)).astype(np.float32)
                M = np.where(grow, m, M).astype(np.float32)
                if trace is not None:
                    trace.extend((block, kb, rows.start + int(r)) for r in np.flatnonzero(grow))
                p = np.exp(s - M[:, None]).astype(np.float16).astype(np.float32)
                S = (S * ms + p.sum(axis=1, dtype=np.float32)).astype(np.float32)
                O = (O * ms[:, None] + p @ vh[kv, ic:ic + C_KEYS]).astype(np.float32)
            inv = np.where(S == 0, np.float32(0), np.float32(1) / np.where(S == 0, 1, S)).astype(np.float32)
            out[head, rows.start:rows.stop] = O * inv[:, None]
    return out[None]


def reference_attention(q, k, v, plan: TensorFAPlan):
    """DIAGNOSTIC reference: unmodified mx.fast.scaled_dot_product_attention on the
    same logical key set (plan.visible_keys), fp32 inputs, the same fp32 scale.

    A separate research helper, not a production backend and not a native
    acceptance threshold; GQA is expanded explicitly and the causal mask
    encodes the absolute query start ``q_start``. Because it expands KV to Hq
    heads and converts to fp32, it is a FIDELITY diagnostic only and never a
    fair performance baseline: a future paired benchmark must keep the
    ordinary served dtype/GQA route and account conversion/materialization
    separately.
    """
    group = plan.q_heads // plan.kv_heads
    k_full = mx.repeat(k.astype(mx.float32), group, axis=1)
    v_full = mx.repeat(v.astype(mx.float32), group, axis=1)
    mask = None
    if plan.causal:
        qpos = plan.q_start + mx.arange(plan.query_len)[:, None]
        mask = mx.arange(plan.kv_len)[None, :] <= qpos
    return mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k_full, v_full, scale=plan.scale_fp32,
                                                 mask=mask)


# ---------------------------------------------------------------- kernel (adapted from ggml, MIT)

TENSOR_FA_HEADER = (
    "#include <metal_stdlib>\n#include <metal_tensor>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n"
)

# Adapted from ggml kernel_flash_attn_ext_tensor (llama.cpp PR 29570 @ 7f0d4d5e,
# MIT, Copyright (c) 2023-2026 The ggml authors; see provenance NOTICE).
TENSOR_FA_SOURCE = r"""
    constexpr int Q = 32, C = 64, NSG = 8, NW = 32;
    constexpr int NT = NW * NSG, NQ = Q / NSG, NC = C / NW;
    const int L = dims[0], N = dims[1], q_start = dims[2], causal = dims[3];
    const int iq1 = int(threadgroup_position_in_grid.x) * Q;
    const int hq = int(threadgroup_position_in_grid.y);
    const int hkv = hq / (HQ / HKV);
    const ushort sgitg = simdgroup_index_in_threadgroup;
    const ushort tiisg = thread_index_in_simdgroup;
    const short tiitg = sgitg * NW + tiisg;

    threadgroup half  sq[Q * D];     // queries, scale folded in, rounded to half
    threadgroup float ss[Q * C];     // scores
    threadgroup half  sp[Q * C];     // probabilities
    threadgroup float sr[Q];         // per-row scale of O
    threadgroup int   sgf[NSG];      // per-simdgroup "rescaled this key block" flag; slot s has ONE writer

    const float qscale = scale[0];
    const device T* qrow = q + ((size_t)hq * L + iq1) * D;
    for (int i = tiitg; i < Q * D; i += NT) {
        const int j = i / D;
        float x = 0.0f;
        if (iq1 + j < L) x = float(qrow[(size_t)j * D + (i % D)]);
        sq[i] = half(x * qscale);
    }
    device half* kh = (device half*)(k + (size_t)hkv * N * D);
    device half* vh = (device half*)(v + (size_t)hkv * N * D);

    float M[NQ];
    float S[NQ];
    for (short jj = 0; jj < NQ; ++jj) { M[jj] = -FLT_MAX / 2; S[jj] = 0.0f; }

    auto tq = tensor(sq, dextents<int32_t, 2>(D, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C, Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C, Q));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, D, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, D, C, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;
    auto tv0 = tensor(vh, dextents<int32_t, 2>(D, C), array<int, 2>({1, D}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) co[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int rows_end = min(iq1 + Q, L);
    const int kv_end = causal ? min(N, ((q_start + rows_end + C - 1) / C) * C) : N;
    for (int ic = 0; ic < kv_end; ic += C) {
        {
            auto tk = tensor(kh + (size_t)ic * D, dextents<int32_t, 2>(D, C), array<int, 2>({1, D}));
            mm_qk.run(tq, tk, ts);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Race-free replacement of ggml's single shared rescale flag (written by
        // lane 0 of every simdgroup): each simdgroup ORs its own rows' decisions in
        // registers (m is simd_max-reduced, so uniform across the simdgroup)
        // and lane 0 writes only its own slot sgf[sgitg], once per key block.
        bool sg_rescaled = false;
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = q_start + iq1 + j;
            float s[NC];
            for (short ii = 0; ii < NC; ++ii) {
                const int kpos = ic + ii * NW + tiisg;
                s[ii] = (causal && kpos > qpos) ? -FLT_MAX : ss[j * C + ii * NW + tiisg];
            }
            float m = M[jj];
            for (short ii = 0; ii < NC; ++ii) m = max(m, s[ii]);
            m = simd_max(m);
            float ms = 1.0f;
            if (m > M[jj] + 8.0f) {
                ms = exp(M[jj] - m);
                M[jj] = m;
                sg_rescaled = true;
            }
            float sum = 0.0f;
            for (short ii = 0; ii < NC; ++ii) {
                const half p = half(exp(s[ii] - M[jj]));
                sp[j * C + ii * NW + tiisg] = p;
                sum += float(p);
            }
            S[jj] = S[jj] * ms + simd_sum(sum);
            if (tiisg == 0) sr[j] = ms;
        }
        if (tiisg == 0) sgf[sgitg] = sg_rescaled ? 1 : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Every thread reads all NSG slots after the barrier: a uniform decision.
        // Rows that did not rescale carry sr == 1, so the per-row law is unchanged.
        bool any_rescaled = false;
        for (short g = 0; g < NSG; ++g) any_rescaled = any_rescaled || (sgf[g] != 0);
        if (any_rescaled) {
            for (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
            }
        }
        {
            auto tv = tensor(vh + (size_t)ic * D, dextents<int32_t, 2>(D, C), array<int, 2>({1, D}));
            mm_pv.run(tp, tv, co);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (tiisg == 0) sr[j] = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
    }
    // rows past L are clipped by the destination extents
    auto td = tensor(out + ((size_t)hq * L + iq1) * D, dextents<int32_t, 2>(D, L - iq1), array<int, 2>({1, D}));
    co.store(td);
    if (tiitg == 0) {
        atomic_fetch_add_explicit((device atomic_uint*)engaged, 1u, memory_order_relaxed);
    }
"""


# ---------------------------------------------------------------- native entry (gated)

_LOCK = threading.Lock()
_COUNTERS = {"host_plans": 0, "native_launches_issued": 0, "native_engagement_confirmed": 0}
_KERNEL_LOCK = threading.Lock()
_KERNEL = {}


class _LaunchRegistry:
    """Issued launches, keyed by a private token and held only WEAKLY.

    A record (weak ref, expected threadgroups, flag id) dies with its flag:
    the weakref callback removes the record and its id hint together, so no
    expected-count metadata outlives the tensor. The id index is a hint
    checked by identity (``ref() is flag``), so id reuse can never alias a
    dead record. ``claim`` consumes a record exactly once under the lock; GPU
    readback happens after the lock is released. An RLock keeps a callback
    fired by garbage collection inside a locked region from deadlocking.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._tokens = itertools.count(1)
        self._records = {}           # token -> (weakref, expected, flag id)
        self._by_id = {}             # flag id -> token (hint)

    def register(self, flag, expected: int) -> int:
        owner = weakref.ref(self)    # the callback must not keep the registry alive

        with self._lock:
            token = next(self._tokens)

            def forget(_ref, token=token):
                registry = owner()
                if registry is not None:
                    registry._forget(token)

            self._records[token] = (weakref.ref(flag, forget), int(expected), id(flag))
            self._by_id[id(flag)] = token
        return token

    def _forget(self, token: int) -> None:
        with self._lock:
            record = self._records.pop(token, None)
            if record is not None and self._by_id.get(record[2]) == token:
                del self._by_id[record[2]]

    def claim(self, flag):
        """Expected count for a live record of exactly ``flag``, consumed once; else None."""
        with self._lock:
            token = self._by_id.get(id(flag))
            record = self._records.get(token) if token is not None else None
            if record is None or record[0]() is not flag:
                return None
            del self._records[token]
            del self._by_id[record[2]]
            return record[1]

    def sizes(self) -> Tuple[int, int]:
        with self._lock:
            return len(self._records), len(self._by_id)


_REGISTRY = _LaunchRegistry()


def _bump(name: str) -> None:
    with _LOCK:
        _COUNTERS[name] += 1


def status(*, reset: bool = False) -> dict:
    with _LOCK:
        out = dict(_COUNTERS)
        if reset:
            for key in _COUNTERS:
                _COUNTERS[key] = 0
    records, hints = _REGISTRY.sizes()
    out.update(native_engaged=out["native_engagement_confirmed"] > 0, pending_records=records,
               pending_id_hints=hints)
    return {**out, **STATE}


def _admit_native(allow_research_metal) -> None:
    """Fail closed; the capability queries run only after the explicit opt-in."""
    if allow_research_metal is not True:
        raise TensorFAUnavailable("research Metal attention needs allow_research_metal=True (caller owns the GPU)")
    if mx.default_device() != mx.gpu:
        raise TensorFAUnavailable("default device is not the GPU")
    if not mx.metal.is_available():
        raise TensorFAUnavailable("Metal is unavailable")
    architecture = str(mx.device_info().get("architecture", ""))
    if not architecture.startswith("applegpu_g17"):
        raise TensorFAUnavailable(f"tensor-API experiment is limited to M5-class devices (got {architecture!r})")


def _kernel():
    with _KERNEL_LOCK:               # one construction, lazily, on an admitted native call
        if "fa" not in _KERNEL:
            _KERNEL["fa"] = mx.fast.metal_kernel(
                name="mlx2_tensor_fa_research_v1", input_names=["q", "k", "v", "scale", "dims"],
                output_names=["out", "engaged"], header=TENSOR_FA_HEADER, source=TENSOR_FA_SOURCE,
                ensure_row_contiguous=True)
        return _KERNEL["fa"]


def tensor_fa_attention(q, k, v, *, scale, causal, q_start, allow_research_metal=False, **extras):
    """UNQUALIFIED approximate attention: returns ``(out_fp32, engaged, plan)``.

    Lazy; nothing is evaluated here and no state or cache is touched. Inputs
    may be copied by MLX (ensure_row_contiguous; see ``plan.layout_contract``).
    Call ``confirm_engagement(engaged)`` after evaluation to count the launch.
    """
    plan = plan_tensor_fa(q, k, v, scale=scale, causal=causal, q_start=q_start, **extras)
    _bump("host_plans")
    _admit_native(allow_research_metal)
    kernel = _kernel()
    out, engaged = kernel(
        inputs=[q, k, v, mx.array([plan.scale_fp32], dtype=mx.float32),
                mx.array([plan.query_len, plan.kv_len, plan.q_start, int(plan.causal)], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", plan.dim), ("HQ", plan.q_heads), ("HKV", plan.kv_heads)],
        grid=plan.grid, threadgroup=plan.threadgroup,
        output_shapes=[(1, plan.q_heads, plan.query_len, plan.dim), (1,)],
        output_dtypes=[mx.float32, mx.uint32], init_value=0)
    with _KERNEL_LOCK:
        built = _KERNEL.get("fa") is kernel
    if built:                        # only the module-built kernel can ever confirm
        _REGISTRY.register(engaged, plan.threadgroups)
    _bump("native_launches_issued")
    return out, engaged, plan


def confirm_engagement(engaged) -> bool:
    """True only for a live flag this module issued whose GPU readback equals the planned threadgroups.

    Refusals that do not depend on the record (not an mx.array, default
    device not the GPU) return before the registry is touched, so they never
    consume a valid record. The record is consumed once under the registry
    lock; the readback (a GPU synchronization) runs after the lock is released.
    """
    if not isinstance(engaged, mx.array) or mx.default_device() != mx.gpu:
        return False
    expected = _REGISTRY.claim(engaged)
    if expected is None:
        return False
    if engaged.dtype != mx.uint32 or engaged.shape != (1,) or int(engaged.item()) != expected:
        return False
    _bump("native_engagement_confirmed")
    return True
