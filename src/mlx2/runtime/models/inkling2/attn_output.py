# SPDX-License-Identifier: Apache-2.0
# Mechanism after youssofal/MTPLX@3890be556160af48887d5bfbf5304c4ad5b9417f
# mtplx/models/inkling/fast/attend.py (the output gate applied in the
# split-K merge) and fast/attn_output.py (o_proj, the attention short conv
# and the residual as one launch, o_proj in MLX's one-row product order),
# Apache-2.0; reimplemented for mlx2 (no code copied), see
# provenance/inkling2-mtplx-launch.json.
"""Attention stage C: fewer launches after the split-K partials.

Two opt-in switches (default off, GPU only, unqualified):

``inkling2_gate_merge`` -- the output gate in the split-K merge. The
composed path merges the partials (one launch, BF16 out), transposes and
reshapes to [B, L, H * D] (a copy when L > 1), and applies the gate
(``fuse.gate_finish``, one compiled launch). ``splitk.merge_gated`` does the
merge and the gate in one launch with the same rounding points: the merge's
value rounded to the activation dtype, times MLX's own ``Sigmoid`` of the
FP32 gate (the functor compiled from the installed headers), rounded once.
One split (``ns == 1``: the partials kernel writes the output itself) keeps
the composed gate, counted ``declined:single_split``. Proven bitwise on the
first call of each shape (``library_kernels.run_proved``).

``inkling2_attn_output_kernel`` -- o_proj, the attention short convolution
and the residual add in one launch. The composed path runs o_proj (one MLX
product) and then ``inkling2_fuse_sconv_kernel``'s short-conv launch with its
``attn_residual`` / ``attn_boundary`` tail. Here each threadgroup computes a
tile of o_proj with MLX's own product code for the composed shape
(``GEMVKernel`` for one row, ``GemvWide`` for 2..8 rows; ``library_kernels``)
and the lane that wrote an output channel then runs that channel's
convolution and tail with the sconv kernel's statements (FP32 taps over
``[state; o_proj rows]``, ``(conv + x) / sqrt(5)`` rounded to the activation
dtype, plus the layer input for the residual tail) and writes the next conv
state. It requires the composed path to be the sconv kernel
(``inkling2_fuse_sconv`` + ``inkling2_fuse_sconv_kernel``, no padding mask,
<= 8 rows), and proves residual, o_proj output and next state bitwise on
first use per shape. Launches per visit: 2 -> 1 (o_proj + conv).

Counters: ``Attention.gate_merge_counts`` and
``DecoderLayer.attn_output_counts`` (``calls``, ``proved``, ``refused``,
``refused_first``, ``unproven``, ``declined:<why>``), summed in
``kernel_receipt()``.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from . import library_kernels as lib
from . import splitk

GATE_PROOFS = lib.Proofs("gate_merge")
OUTPUT_PROOFS = lib.Proofs("attn_output")
MAX_ROWS = 8
_kernels = {}


def splitk_finish(attn, q, k, v, rel, sink, gate, b, length, dtype, **kw):
    """``Attention``'s split-K attention plus ``_finish``, the gate in the merge."""
    parts = splitk.partials(q, k, v, rel, sink, **kw)
    if parts[0] == "out":
        lib.count(attn, "gate_merge_counts", "declined:single_split")
        return attn._finish(parts[1], gate, b, length, dtype)
    _kind, po, pm, pl, ns, sinkf = parts
    if gate.dtype != q.dtype or q.dtype != dtype or lib.library("elementwise")[0] is None:
        lib.count(attn, "gate_merge_counts", "declined:dtype_or_library")
        return attn._finish(splitk.merge(q, po, pm, pl, ns, sinkf), gate, b, length, dtype)
    key = (tuple(q.shape), str(q.dtype), ns, tuple(gate.shape))
    gated, _engaged = lib.run_proved(
        GATE_PROOFS, key, attn, "gate_merge_counts",
        lambda: splitk.merge_gated(q, po, pm, pl, ns, sinkf, gate),
        lambda: attn._gate(splitk.merge(q, po, pm, pl, ns, sinkf), gate, b, length, dtype),
        lambda a, r: [(a, r)])
    return attn._output(gated)


# ----- o_proj + attention short conv + residual ----------------------------

_EPILOGUE = r"""
    // The sconv kernel's statements (fuse_kernels._SCONV_SOURCE) for channel c.
    {
        const float inv = inv_arr[0];
        float w[4];
        for (int j = 0; j < 4; ++j) w[j] = wt[c * 4 + j];
        for (uint bb = 0; bb < NB; ++bb) {
            float ctx[3 + L];
            for (int j = 0; j < 3; ++j) ctx[j] = HAS_STATE ? state[(bb * 3 + j) * C + c] : 0.0f;
            for (int i = 0; i < L; ++i) ctx[3 + i] = float(y[(bb * L + i) * C + c]);
            for (int i = 0; i < L; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < 4; ++j) acc += w[j] * ctx[i + j];
                float xf = ctx[3 + i];
                float yv = float(T((acc + xf) / inv));
                size_t o = (bb * L + i) * C + c;
                if (TAIL == 2) {
                    res[o] = yv + float(xres[o]);
                } else {
                    res[o] = yv;
                }
            }
            for (int j = 0; j < 3; ++j) nstate[(bb * 3 + j) * C + c] = ctx[L + j];
        }
    }
"""


def _source(kind, tiles, dtype, rows, n):
    t = lib.metal_type(dtype)
    logical, per_group, threads = lib.logical_groups(kind, tiles, n)
    sgs = threads // 32
    call = lib.matvec_call(kind, tiles, dtype=t, rows=rows, mat="wo", vec="g", out="y",
                           n_ref="sizes[1]", logical="logical", lsg="lsg")
    if kind == "gemv":
        bm, _bn, sm, sn, tm, _tn = tiles
        block_m = bm * sm * tm
        # GEMVKernel's writer lanes and rows (BN == 1): lane % SN == 0 of each
        # simdgroup writes rows out_row .. out_row + TM - 1.
        owner = f"""
    if (lane % {sn} != 0) return;
    int out_row = int(logical) * {block_m} + ({sm} * int(lsg) + int(lane) / {sn}) * {tm};
    if (out_row >= C) return;
    out_row = out_row + {tm} <= C ? out_row : C - {tm};
    for (int tm = 0; tm < {tm}; ++tm) {{
        const int c = out_row + tm;
{_EPILOGUE}
    }}"""
    else:
        k_lanes = tiles[0]
        owner = f"""
    if (lane % {k_lanes} != 0) return;
    const int c = int(logical) * 4 + {32 // k_lanes} * int(lsg) + int(lane) / {k_lanes};
    if (c >= C) return;
{_EPILOGUE}"""
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint logical = threadgroup_position_in_grid.x * {per_group} + sg / {sgs};
    const uint lsg = sg % {sgs};
    threadgroup float tgp[1];
    if (logical >= {logical}) return;
    {call}
    {owner}
""", -(-logical // per_group)


def _kernel(kind, tiles, dtype, rows, n):
    key = (kind, tiles, dtype, rows, n)
    held = _kernels.get(key)
    if held is None:
        header, why = lib.library("matvec")
        if header is None:
            return None, why
        source, groups = _source(kind, tiles, dtype, rows, n)
        kernel = mx.fast.metal_kernel(
            name=f"inkling2_attn_output_{len(_kernels)}",
            input_names=["g", "wo", "sizes", "unit", "state", "wt", "inv_arr", "xres"],
            output_names=["y", "res", "nstate"], header=header, source=source)
        held = _kernels[key] = ((kernel, groups), None)
    return held


def residual(layer, gated, conv, mask, x):
    """``(residual, None)`` from one launch (or the composed path while a
    shape is being proven or after a refusal), or ``(None, o_proj(gated))``
    when declined (counted)."""
    from .language import _speculative_record

    planned = plan_residual(layer, gated, conv, mask, x)
    if isinstance(planned, str):
        lib.count(layer, "attn_output_counts", "declined:" + planned)
        return None, layer.self_attn.o_proj(gated)
    out, engaged = lib.run_planned(OUTPUT_PROOFS, planned, layer, "attn_output_counts")
    if not engaged:
        return out, None
    out_y, res, nstate = out
    sconv = layer.attn_sconv
    if getattr(conv, "speculating", False):
        _speculative_record(conv, sconv.conv_idx, planned.state, out_y.astype(mx.float32), None)
    conv[sconv.conv_idx] = nstate
    return res, None


def plan_residual(layer, gated, conv, mask, x):
    """A ``library_kernels.Planned`` (fused: ``(o_proj out, residual, next
    state)``, nothing written; reference: the composed o_proj + sconv, which
    writes the conv state), or the decline reason."""
    from . import fuse as _fuse
    from .language import _fp32

    attn, sconv = layer.self_attn, layer.attn_sconv
    if mx.default_device() != mx.gpu:
        return "not_gpu"
    if not (_fuse.on(sconv, "fuse_sconv") and _fuse.on(sconv, "fuse_sconv_kernel")):
        return "sconv_kernel_off"
    if mask is not None or conv is None or sconv.kernel_size != 4:
        return "layout"
    if type(attn.o_proj) is not nn.Linear or "bias" in attn.o_proj:
        return "projection_type"
    weight = attn.o_proj.weight
    dtype = gated.dtype
    if weight.dtype != dtype or x.dtype != dtype or dtype not in (mx.bfloat16, mx.float16,
                                                                   mx.float32):
        return "dtype"
    b, length, k = gated.shape
    channels = weight.shape[0]
    rows = b * length
    if rows > MAX_ROWS or x.shape != (b, length, channels):
        return "rows"
    state = conv[sconv.conv_idx]
    if state is not None and (state.shape != (b, 3, channels) or state.dtype != mx.float32):
        return "state"
    kind, tiles = lib.matvec_plan(rows, channels, k, dtype)
    if kind is None:
        return tiles
    if kind == "gemv" and tiles[1] != 1:
        return "tiles"
    (kernel, groups), why = _kernel(kind, tiles, dtype, rows, channels)
    if kernel is None:
        return "library"
    boundary = layer.is_block_boundary
    tail = "attn_boundary" if boundary else "attn_residual"

    def fused():
        out_y, res, nstate = kernel(
            inputs=[mx.contiguous(gated.reshape(rows, k)), weight,
                    mx.array([k, channels], mx.int32), mx.array([1.0], mx.float32),
                    state if state is not None else mx.zeros((1,), mx.float32),
                    _fp32(sconv.conv, "weight").reshape(channels, 4),
                    mx.array([math.sqrt(5)], mx.float32),
                    x if not boundary else mx.zeros((1,), dtype)],
            template=[("T", dtype), ("C", channels), ("L", length), ("NB", b),
                      ("TAIL", 1 if boundary else 2), ("HAS_STATE", state is not None)],
            grid=(groups * lib.THREADS, 1, 1), threadgroup=(lib.THREADS, 1, 1),
            output_shapes=[(rows, channels), (b, length, channels), (b, 3, channels)],
            output_dtypes=[dtype, mx.float32, mx.float32])
        return out_y.reshape(b, length, channels), res, nstate

    held = {}

    def reference():
        attn_out = attn.o_proj(gated)
        extra = () if boundary else (x,)
        out = sconv(attn_out, conv, mask, tail=tail, extra=extra)
        held["attn_out"] = attn_out
        return out

    key = (kind, tiles, str(dtype), b, length, k, channels, boundary, state is None)
    planned = lib.Planned(
        key, fused, reference,
        lambda f, r: [(f[1], r), (f[2], conv[sconv.conv_idx]), (f[0], held["attn_out"])])
    planned.state = state
    planned.held = held
    return planned
