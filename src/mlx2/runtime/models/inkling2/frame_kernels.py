# SPDX-License-Identifier: Apache-2.0
# Mechanism after youssofal/MTPLX@3890be556160af48887d5bfbf5304c4ad5b9417f
# mtplx/models/inkling/fast/embedding.py (row gather and RMS norm in one
# launch), Apache-2.0; reimplemented for mlx2. The RMS norm's reduction order
# is MLX 0.32's ``rms_single_row`` (ml-explore/mlx, MIT,
# mlx/backend/metal/kernels/rms_norm.metal), restated here because MLX does
# not install that kernel as a header; see provenance/inkling2-mtplx-launch.json.
"""``inkling2_embed_kernel``: the token embedding and its norm in one launch.

Opt-in, default off, GPU only, unqualified. ``TextModel.embed`` is
``embed_norm(embed_tokens(ids))``: a gather launch and MLX's RMS-norm launch.
Here one threadgroup per token reads its table row and normalises it with
the reduction MLX's ``mx.fast.rms_norm`` runs for rows of at most 4096
values (``rms_single_row``): a threadgroup of ``32 * ceil(ceil(D / 4) / 32)``
threads, four consecutive values per thread squared and summed in order,
``simd_sum``, the per-simdgroup sums summed by one more ``simd_sum`` over a
zero-initialised 32-slot array, ``precise::rsqrt(sum / D + eps)``, and each
output ``w[i] * T(x[i] * inv)``. Proven bitwise against the two composed
launches on the first call of each shape (``library_kernels.run_proved``);
rows longer than 4096 (MLX's looped kernel) decline. 2 launches -> 1 per
forward (negligible for speed; it is here because it was on the list).

Counters: ``TextModel.embed_kernel_counts``, in
``kernel_receipt()["embed_kernel"]``.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from . import library_kernels as lib

PROOFS = lib.Proofs("embed_kernel")
_LOOPED_LIMIT = 4096
_N_READS = 4
_kernel = None

_EMBED_SOURCE = r"""
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const constant uint& axis_size = dims[0];
    threadgroup float inv_mean[1];
    threadgroup float sums[32];
    const device T* row = table + size_t(ids[gid]) * axis_size + lid * NR;
    const device T* wr = w + lid * NR;
    float acc = 0;
    float v[NR];
    if (lid * NR + NR <= axis_size) {
        for (int i = 0; i < NR; i++) {
            v[i] = row[i];
            acc += v[i] * v[i];
        }
    } else {
        for (int i = 0; i < NR; i++) {
            v[i] = (lid * NR + i < axis_size) ? (float)row[i] : 0;
            acc += v[i] * v[i];
        }
    }
    acc = simd_sum(acc);
    if (sg == 0) {
        sums[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(sums[lane]);
        if (lane == 0) {
            inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device T* dst = out + gid * size_t(axis_size) + lid * NR;
    if (lid * NR + NR <= axis_size) {
        for (int i = 0; i < NR; i++) {
            dst[i] = wr[i] * static_cast<T>(v[i] * inv_mean[0]);
        }
    } else {
        for (int i = 0; i < NR; i++) {
            if ((lid * NR + i) < axis_size) {
                dst[i] = wr[i] * static_cast<T>(v[i] * inv_mean[0]);
            }
        }
    }
"""


def embed(model, inputs):
    """``embed_norm(embed_tokens(inputs))`` in one launch, or None."""
    planned = plan(model, inputs)
    if isinstance(planned, str):
        lib.count(model, "embed_kernel_counts", "declined:" + planned)
        return None
    out, _engaged = lib.run_planned(PROOFS, planned, model, "embed_kernel_counts")
    return out


def plan(model, inputs):
    """A ``library_kernels.Planned`` for this call, or the decline reason."""
    global _kernel
    table_mod, norm = model.embed_tokens, model.embed_norm
    if mx.default_device() != mx.gpu:
        return "not_gpu"
    if type(table_mod) is not nn.Embedding or type(norm) is not nn.RMSNorm:
        return "module_type"
    table, weight = table_mod.weight, norm.weight
    width = table.shape[-1]
    if table.dtype != weight.dtype or table.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return "dtype"
    if width > _LOOPED_LIMIT or weight.shape != (width,):
        return "width"
    if inputs.dtype not in (mx.int32, mx.uint32) or inputs.size == 0:
        return "ids"
    per_thread = -(-width // _N_READS)
    threads = 32 * -(-per_thread // 32)
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="inkling2_embed_norm", input_names=["ids", "table", "w", "dims", "eps"],
            output_names=["out"], source=_EMBED_SOURCE)
    n = inputs.size

    def fused():
        return _kernel(
            inputs=[inputs.reshape(n), table, weight, mx.array([width], mx.uint32),
                    mx.array([norm.eps], mx.float32)],
            template=[("T", table.dtype), ("NR", _N_READS)],
            grid=(n * threads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(n, width)], output_dtypes=[table.dtype])[0].reshape(
                *inputs.shape, width)

    key = (tuple(inputs.shape), str(table.dtype), width)
    return lib.Planned(key, fused, lambda: norm(table_mod(inputs)), lambda a, b: [(a, b)])
