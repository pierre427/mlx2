# SPDX-License-Identifier: Apache-2.0
# Mechanism after youssofal/MTPLX@3890be556160af48887d5bfbf5304c4ad5b9417f
# mtplx/models/inkling/fast/attn_input.py (``project``: every input
# projection of a visit's panel in one launch, in MLX's own product order),
# Apache-2.0; reimplemented for mlx2 (no code copied), see
# provenance/inkling2-mtplx-launch.json.
"""``inkling2_stage_a_kernel``: a visit's six input projections in one launch.

Opt-in, default off, GPU only, unqualified. ``Attention._project_inputs``
runs q, r, gate and sink on the visit's x and k, v on the key and value
inputs: six MLX products, six launches per visit (four with
``inkling2_fused_projections`` and distinct k/v inputs, one with equal
inputs). This runs the same products in one launch: every threadgroup takes
one tile of one product and calls MLX's own product code for it, compiled
from the installed headers (``library_kernels``, part ``"matvec"``), with the
tiles MLX's dispatch picks for that product's shape:

- 1 row (B1 decode): ``GEMVKernel`` (q, r, gate, k, v: ``bm4/8 bn1``;
  sink, ``K >= 16 N``: ``bm1 bn8`` with its threadgroup reduction);
- 2..8 rows (B2 decode, DSpark verify, B*L <= 8) of BF16/FP16 on M3 and
  later: ``GemvWide`` with MLX's ``k_lanes``;
- FP32 rows >= 2 (MLX's steel GEMM) and other shapes decline, counted.

So each output element is reduced in MLX's order for the composed path's
shape: bitwise to the composed path by construction, and proven so on the
first call of each shape (``library_kernels.run_proved``). The proving call
returns the composed result; a refused shape keeps the composed path.

What it does not do (MTPLX's stage A also fuses these): the clips, the
short convolutions, log-tau and the value norm stay as they are (or in
``inkling2_attn_decode_kernel``, which calls ``_project_inputs`` and so
picks this up too). Launches per visit: 6 -> 1 for the products.

Counters: ``Attention.stage_a_counts`` (``calls``, ``proved``,
``refused``, ``refused_first``, ``unproven``, ``declined:<why>``), summed in
``kernel_receipt()["stage_a"]``.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from . import library_kernels as lib

MAX_ROWS = 8
PROOFS = lib.Proofs("stage_a")
_kernels = {}
_NAMES = ("q_proj", "r_proj", "gate_proj", "sink_proj", "k_proj", "v_proj")


def _decline(attn, why):
    lib.count(attn, "stage_a_counts", "declined:" + why)
    return None


def _jobs(attn, x, key_input, value_input):
    """``[(input index, weight, slices)]``: the products the composed path
    runs, each with the output slices it feeds (name -> (start, stop))."""
    bounds = getattr(attn, "_fused_bounds", None)
    if bounds is None:
        return [(0 if name in _NAMES[:4] else (1 if name == "k_proj" else 2),
                 attn[name].weight, {name: None}) for name in _NAMES]
    stacked = attn._fused_weight
    if key_input is x and value_input is x:
        return [(0, stacked, {name: bounds[name] for name in _NAMES})]
    shared_end = bounds["sink_proj"][1]
    return [(0, stacked[:shared_end], {name: bounds[name] for name in _NAMES[:4]}),
            (1, attn.k_proj.weight, {"k_proj": None}),
            (2, attn.v_proj.weight, {"v_proj": None})]


def _source(plans, dtype, rows):
    """One launch: product j owns threadgroups [start_j, start_j + count_j)."""
    t = lib.metal_type(dtype)
    tgp = max(lib.tgp_floats(kind, tiles) for kind, tiles, _n, _inp in plans)
    lines = [
        "const uint lane = thread_index_in_simdgroup;",
        "const uint sg = simdgroup_index_in_threadgroup;",
        "const uint g = threadgroup_position_in_grid.x;",
        f"threadgroup float tgp[{tgp}];",
    ]
    start = 0
    for j, (kind, tiles, n, inp) in enumerate(plans):
        logical, per_group, threads = lib.logical_groups(kind, tiles, n)
        groups = -(-logical // per_group)
        sgs = threads // 32
        call = lib.matvec_call(kind, tiles, dtype=t, rows=rows, mat=f"w{j}", vec=f"x{inp}",
                               out=f"y{j}", n_ref=f"sizes[{1 + j}]", logical="logical",
                               lsg=f"sg % {sgs}")
        lines.append(
            f"if (g >= {start} && g < {start + groups}) {{\n"
            f"  const uint logical = (g - {start}) * {per_group} + sg / {sgs};\n"
            f"  if (logical < {logical}) {{ {call} }}\n"
            f"  return;\n}}")
        start += groups
    return "\n".join(lines), start


def _kernel(plans, dtype, rows, n_inputs):
    key = (tuple(plans), dtype, rows, n_inputs)
    held = _kernels.get(key)
    if held is not None:
        return held
    header, why = lib.library("matvec")
    if header is None:
        return None, why
    source, groups = _source(plans, dtype, rows)
    kernel = mx.fast.metal_kernel(
        name=f"inkling2_stage_a_{len(_kernels)}",
        input_names=[f"x{i}" for i in range(n_inputs)] + [f"w{j}" for j in range(len(plans))]
        + ["sizes", "unit"],
        output_names=[f"y{j}" for j in range(len(plans))],
        header=header, source=source)
    held = _kernels[key] = ((kernel, groups), None)
    return held


def _weights_ok(attn, dtype):
    for name in _NAMES:
        module = attn[name]
        if type(module) is not nn.Linear or "bias" in module:
            return "projection_type"
        if module.weight.dtype != dtype:
            return "weight_dtype"
    return None


def project(attn, x, key_input, value_input):
    """``(q, r, gate, sink, k, v)`` from one launch, or None (declined)."""
    planned = plan(attn, x, key_input, value_input)
    if isinstance(planned, str):
        return _decline(attn, planned)
    out, engaged = lib.run_planned(PROOFS, planned, attn, "stage_a_counts")
    if engaged and getattr(attn, "_fused_bounds", None) is not None:
        attn.fused_calls += 1  # the stacked products ran (in this launch)
    return out


def plan(attn, x, key_input, value_input):
    """A ``library_kernels.Planned`` for this call, or the decline reason."""
    if mx.default_device() != mx.gpu:
        return "not_gpu"
    dtype = x.dtype
    if dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return "dtype"
    inputs = [x, key_input, value_input]
    if any(a.dtype != dtype or a.shape != x.shape for a in inputs):
        return "input_shape"
    why = _weights_ok(attn, dtype)
    if why:
        return why
    lead, k = x.shape[:-1], x.shape[-1]
    rows = 1
    for d in lead:
        rows *= d
    if not 1 <= rows <= MAX_ROWS:
        return "rows"
    jobs = _jobs(attn, x, key_input, value_input)
    plans = []
    for inp, weight, _slices in jobs:
        kind, tiles = lib.matvec_plan(rows, weight.shape[0], k, dtype)
        if kind is None:
            return tiles
        if kind == "gemv" and lib.logical_groups(kind, tiles, weight.shape[0])[1] < 1:
            return "tiles"
        plans.append((kind, tiles, weight.shape[0], inp))
    distinct = [x]
    index = []
    for a in inputs:
        for i, held in enumerate(distinct):
            if a is held:
                index.append(i)
                break
        else:
            index.append(len(distinct))
            distinct.append(a)
    plans = [(kind, tiles, n, index[inp]) for kind, tiles, n, inp in plans]
    (kernel, groups), why = _kernel(plans, dtype, rows, len(distinct))
    if kernel is None:
        return "library"

    def fused():
        sizes = mx.array([k] + [n for _kind, _tiles, n, _inp in plans], mx.int32)
        outs = kernel(
            inputs=[*(a.reshape(rows, k) for a in distinct),
                    *(w for _inp, w, _s in jobs), sizes, mx.array([1.0], mx.float32)],
            grid=(groups * lib.THREADS, 1, 1), threadgroup=(lib.THREADS, 1, 1),
            output_shapes=[(rows, n) for _kind, _tiles, n, _inp in plans],
            output_dtypes=[dtype] * len(plans))
        named = {}
        for (inp, _w, slices), y in zip(jobs, outs):
            y = y.reshape(*lead, y.shape[-1])
            for name, span in slices.items():
                named[name] = y if span is None else y[..., span[0]:span[1]]
        return tuple(named[name] for name in _NAMES)

    # Shape, tiles and input sharing. MLX's wide product also needs 8-element
    # aligned offsets and 4-element aligned strides; every Inkling 2 slice and
    # stacked-weight offset is a multiple of the hidden size (2304), so the
    # key does not carry them (python cannot read an array's offset).
    key = (tuple(plans), str(dtype), rows, k, tuple(index))
    return lib.Planned(key, fused,
                       lambda: attn._project_inputs_composed(x, key_input, value_input),
                       lambda a, b: list(zip(a, b)))
