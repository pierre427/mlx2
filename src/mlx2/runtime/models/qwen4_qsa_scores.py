# SPDX-License-Identifier: Apache-2.0 AND MIT
# Idea from TensorFold 0.6.1 (Apache-2.0) at 17c73e18,
# ``kernels/qwen/flash_next/v1/attention.py`` ``_IDX_SCORES``: one threadgroup
# reads each pooled block once for every query row.  The arithmetic is MLX's
# (39400a0d4, ``steel/gemm/mma.h``, ``binary_ops.h``, ``reduce``; MIT), not
# TensorFold's.  See docs/PROVENANCE.md and
# provenance/tensorfold-0.6.1-flashnext-longctx.json.
"""Fused Qwen4 (Flash-Next) QSA block scores for decode and verify rows.

Past the indexer budget a decode or verify step scores every pooled key
block with a chain of MLX ops::

    s = mx.einsum("blhd,bnd->blnh", q.astype(f32), pooled.astype(f32))
    s = mx.sum(mx.maximum(s, 0), axis=-1) / math.sqrt(head_dim)
    s = mx.where(valid_blocks, s, -mx.inf)

that is a full fp32 copy of the pooled keys (the context / 4 blocks, every
step, in every QSA layer), a steel GEMM, and four elementwise / reduction
launches over [L, N, H].  ``block_scores`` produces the same bytes in ONE
launch that reads the bf16 pooled keys once for all of the step's rows
(TensorFold's keys-stationary scheduling):

* the products: MLX routes this fp32 GEMM (TF32 off, K = 128, N > K) to
  the non-NAX steel GEMM, whose every output element is one fp32
  ``simdgroup_multiply_accumulate`` chain over K in steps of 8 from a zero
  accumulator (``BlockMMA::mma`` / ``tile_matmad``), independent of the tile
  shape.  The kernel runs that chain with the same fragment layout
  (``BaseMMAFrag::get_coord``), the bf16 keys converted exactly to fp32 in
  registers;
* the epilogue: ``Maximum`` (NaN-propagating ``x > y ? x : y``), the
  head sum in head order from MLX's reduction identity, the division by the
  fp32 ``sqrt(head_dim)`` scalar, and the validity select.

``argpartition`` stays MLX, so the selected block ids are the stock ids
whenever the scores are the stock bytes.  Default off
(``MLX_QWEN4_QSA_FUSED_SCORES``, the Flash-Next policy field
``qsa_fused_scores``); admission is strict and every refusal is counted by
reason so the caller keeps the MLX ops.
"""

from __future__ import annotations

import math
import os
import threading
from collections import Counter
from typing import Optional

import mlx.core as mx

from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)

ENV_NAME = "MLX_QWEN4_QSA_FUSED_SCORES"
# Query rows (L * heads) a launch carries: four 8-row fragments.
MAX_MATRIX_ROWS = 32
MAX_BATCH = 64
_HEAD_DIM = 128
_SIMDGROUPS = 4
_FRAGS_PER_SIMD = 2  # 16 blocks per simdgroup, 64 per threadgroup
_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


def _parse_flag(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{ENV_NAME} must be 0/off or 1/on; got {raw!r}")


_ENABLED = _parse_flag(os.environ.get(ENV_NAME))
_LOCK = threading.Lock()
_COUNTS: Counter = Counter()
_KERNELS: dict = {}


def enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool) -> None:
    """Process switch: the policy sets it through the environment; in-process
    A/B harnesses flip it directly.  ``False`` is the kill switch."""
    global _ENABLED
    _ENABLED = bool(value)


def bump(name: str, amount: int = 1) -> None:
    with _LOCK:
        _COUNTS[name] += amount


def status(*, reset: bool = False) -> dict:
    with _LOCK:
        counts = dict(_COUNTS)
        if reset:
            _COUNTS.clear()
    return {"enabled": bool(_ENABLED), "counts": counts}


def _tf32_off() -> bool:
    # MLX latches MLX_ENABLE_TF32 at its first fp32 matmul; with TF32 on the
    # stock scores run the NAX GEMM, whose bits this kernel does not follow.
    return os.environ.get("MLX_ENABLE_TF32", "0").strip() in ("0", "")


def supported(q: mx.array, pooled: mx.array) -> Optional[str]:
    """``None`` when ``block_scores`` reproduces the stock chain for these
    operands, else the refusal reason.  Never evaluates an array."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if not hasattr(mx.fast, "metal_kernel"):
        return "device"
    if not _tf32_off():
        return "tf32"
    if q.ndim != 4 or pooled.ndim != 3:
        return "rank"
    batch, length, heads, dims = q.shape
    if not 1 <= batch <= MAX_BATCH or pooled.shape[0] != batch:
        return "batch"
    if dims != _HEAD_DIM or pooled.shape[2] != dims:
        return "head_dim"
    if length < 1 or length * heads > MAX_MATRIX_ROWS:
        return "rows"
    blocks = int(pooled.shape[1])
    # The stock GEMM routes N <= K or tiny tiles elsewhere (split-K / gemv).
    if blocks <= dims:
        return "blocks"
    if q.dtype not in _DTYPES or pooled.dtype not in _DTYPES:
        return "dtype"
    return None


_HEADER = """
#define UNROLL _Pragma("clang loop unroll(full)")
"""

_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int bat = int(threadgroup_position_in_grid.z);
    const int N = int(pooled_shape[1]);
    const int n_base = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * (8 * TN);
    constexpr int M = L * H;
    constexpr int MF = (M + 7) / 8;
    const device auto* qb = q + size_t(bat) * M * D;
    const device auto* pb = pooled + size_t(bat) * N * D;
    // BaseMMAFrag<float, 8, 8>::get_coord: lane holds (fm, fn), (fm, fn + 1)
    const short qid = short(lane) / 4;
    const short fm = (qid & 4) + ((short(lane) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;

    threadgroup float tile[SG][MF * 8][8 * TN];

    simdgroup_matrix<float, 8, 8> C[MF][TN];
    UNROLL for (int mf = 0; mf < MF; mf++)
      UNROLL for (int t = 0; t < TN; t++)
        C[mf][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    if (n_base < N) {
      UNROLL for (int k0 = 0; k0 < D; k0 += 8) {
        simdgroup_matrix<float, 8, 8> A[MF];
        UNROLL for (int mf = 0; mf < MF; mf++) {
          const int row = mf * 8 + fm;
          thread auto& a = A[mf].thread_elements();
          a[0] = row < M ? float(qb[row * D + k0 + fn]) : 0.0f;
          a[1] = row < M ? float(qb[row * D + k0 + fn + 1]) : 0.0f;
        }
        UNROLL for (int t = 0; t < TN; t++) {
          // B (K x N fragment): element (k, n) = pooled[n_base + 8 t + n][k0 + k]
          simdgroup_matrix<float, 8, 8> B;
          thread auto& b = B.thread_elements();
          const int n0 = n_base + 8 * t + fn;
          b[0] = n0 < N ? float(pb[size_t(n0) * D + k0 + fm]) : 0.0f;
          b[1] = n0 + 1 < N ? float(pb[size_t(n0 + 1) * D + k0 + fm]) : 0.0f;
          UNROLL for (int mf = 0; mf < MF; mf++)
            simdgroup_multiply_accumulate(C[mf][t], A[mf], B, C[mf][t]);
        }
      }
    }
    UNROLL for (int mf = 0; mf < MF; mf++)
      UNROLL for (int t = 0; t < TN; t++)
        simdgroup_store(C[mf][t], &tile[sg][mf * 8][8 * t], 8 * TN);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int offset = META[bat];
    const float scale = SCALE[0];
    for (int idx = int(lane); idx < L * 8 * TN; idx += 32) {
      const int l = idx / (8 * TN);
      const int c = idx - l * 8 * TN;
      const int n = n_base + c;
      if (n >= N) continue;
      float total = 0.0f;               // Sum's identity, then each head in order
      UNROLL for (int h = 0; h < H; h++) {
        const float x = tile[sg][l * H + h][c];
        const float r = metal::isnan(x) ? x : (x > 0.0f ? x : 0.0f);
        total = r + total;
      }
      const float s = total / scale;
      const bool valid = (n * RATIO + RATIO - 1) <= (offset + l);
      out[(size_t(bat) * L + l) * N + n] = valid ? s : -INFINITY;
    }
"""


def _kernel():
    found = _KERNELS.get("scores")
    if found is None:
        found = _KERNELS["scores"] = mx.fast.metal_kernel(
            name="mlx2_qwen4_qsa_block_scores",
            input_names=["q", "pooled", "META", "SCALE"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return found


def offset_supported(offset, batch: int) -> Optional[str]:
    """``None`` when ``offset`` (an int, or ``batch`` integer offsets) is a
    form ``block_scores`` reads, else the refusal reason."""
    if isinstance(offset, mx.array):
        if offset.size != batch or offset.dtype not in (
            mx.int32, mx.int64, mx.uint32
        ):
            return "offset"
        return None
    if isinstance(offset, int) and batch == 1:
        return None
    return "offset"


def _offsets(offset, batch: int) -> mx.array:
    if isinstance(offset, mx.array):
        return offset.reshape(batch).astype(mx.int32)
    return mx.array([int(offset)], dtype=mx.int32)


def block_scores(q: mx.array, pooled: mx.array, offset, ratio: int) -> mx.array:
    """Scores ``[B, L, N]`` (fp32) of the stock chain for ``q`` ``[B, L, H, D]``
    (normed, rotated query rows; row ``b``'s ``l`` at position
    ``offset[b] + l``) against ``pooled`` ``[B, N, D]``, with the blocks not
    closed at a row set to ``-inf``.  ``offset`` is an int or an integer
    array of ``B`` logical offsets (a batched cache's ``offset``).

    The caller checks ``supported`` (and ``offset_supported``) first.
    """
    batch, length, heads, dims = map(int, q.shape)
    blocks = int(pooled.shape[1])
    per_group = 8 * _FRAGS_PER_SIMD * _SIMDGROUPS
    groups = -(-blocks // per_group)
    # The stock chain divides by the fp32 rounding of the Python scalar.
    scale = mx.array([math.sqrt(dims)], dtype=mx.float32)
    (out,) = _kernel()(
        inputs=[
            mx.contiguous(q.reshape(batch, length * heads, dims)),
            mx.contiguous(pooled),
            _offsets(offset, batch),
            scale,
        ],
        template=[
            ("L", length),
            ("H", heads),
            ("D", dims),
            ("RATIO", int(ratio)),
            ("TN", _FRAGS_PER_SIMD),
            ("SG", _SIMDGROUPS),
        ],
        grid=(groups * 32 * _SIMDGROUPS, 1, batch),
        threadgroup=(32 * _SIMDGROUPS, 1, 1),
        output_shapes=[(batch, length, blocks)],
        output_dtypes=[mx.float32],
    )
    return out


def stock_scores(
    q: mx.array, pooled: mx.array, valid_blocks: mx.array, head_dim: int
) -> mx.array:
    """The MLX chain ``block_scores`` reproduces (reference and fallback)."""
    scores = mx.einsum(
        "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
    )
    scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(head_dim)
    return mx.where(valid_blocks, scores, -mx.inf)
