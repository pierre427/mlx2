# SPDX-License-Identifier: Apache-2.0 AND MIT
# Launch layout adapted from jundot/omlx PR #4052 (Apache-2.0) at d05afc7a
# (one-pass chains spread over simdgroups, combine launch) and PRs #4041 /
# #4105 (Apache-2.0; heads 7ef4d997 / 93e708a8: every verify-window row runs
# the vector SDPA plan of its own serial decode step).  ``select_sdpa_blocks``
# and the one-/two-pass plan are MLX 39400a0d4
# ``scaled_dot_product_attention.cpp`` (MIT).  See docs/PROVENANCE.md and the
# provenance records named in ``vector_sources``.
"""Launch geometry of the vector-SDPA kernel families (``vector_sources``).

MLX's plan for a vector SDPA call -- one pass below 1,024 keys, two passes
with ``select_sdpa_blocks`` partitions above -- and the grids mlx2's
transcriptions use for it:

* :func:`run_rows`: R query rows of ONE call over the same keys (Flash-Next
  ``sdpa_gate``): two-pass rows share a threadgroup, as in MLX.
* :func:`run_window`: R rows that each replay their own one-row plan over
  ``n_first + r`` keys (row-exact verify windows), one launch pair per run
  of rows that share a plan.

Kernels are looked up through the caller's ``chains(passes, masked)`` and
``combine(passes)`` callables, so each model module keeps its own compiled
kernel cache and names.  The grids, threadgroups, template arguments, input
order and output shapes are the ones the three model modules launched
before (``tests/test_technologies_window_attention_sources.py`` compares
the recorded launches).
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["select_sdpa_blocks", "run_rows", "run_window", "plan_runs"]


def select_sdpa_blocks(devc: str, n_keys: int, n_simds: int):
    """MLX 39400a0d4 ``select_sdpa_blocks`` for the 's' and 'd' GPU classes
    (``None`` elsewhere)."""
    if devc == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            if n_keys <= 8192:
                blocks = 128
            elif n_keys <= 32768:
                blocks = 256
            elif n_keys <= 65536:
                blocks = 512
            else:
                blocks = 1024
        return blocks
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
        return blocks
    return None


def plan_runs(plans):
    """``[(start, stop, plan)]``: maximal runs of consecutive rows sharing a
    plan."""
    runs = []
    start = 0
    for j in range(1, len(plans) + 1):
        if j == len(plans) or plans[j] != plans[start]:
            runs.append((start, j, plans[start]))
            start = j
    return runs


def _one_pass(chains, combine, q, keys, values, head_inputs, extra, combine_extra, *,
              dtype, head_dim, gqa, kv_heads, heads, count, hpt, sg):
    n_chains = 32 * (gqa // hpt)
    maxs, sums, outs = chains(1)(
        inputs=[q, keys, values] + head_inputs + extra,
        template=[("T", dtype), ("D", head_dim), ("GQA", gqa), ("HPT", hpt), ("SG", sg)],
        grid=(32 * n_chains, kv_heads, count),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[
            (count * heads * 32,),
            (count * heads * 32,),
            (count * heads * head_dim * 32,),
        ],
        output_dtypes=[mx.float32] * 3,
    )
    return combine(1)(
        inputs=[maxs, sums, outs] + combine_extra,
        template=[("T", dtype), ("D", head_dim)],
        grid=(heads * 1024, count, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(count, heads * head_dim)],
        output_dtypes=[dtype],
    )[0]


def _two_pass(chains, combine, q, keys, values, head_inputs, extra, combine_extra, *,
              dtype, head_dim, gqa, kv_heads, heads, count, blocks, rows_per_threadgroup):
    partials, sums, maxs = chains(2)(
        inputs=[q, keys, values] + head_inputs + extra,
        template=[("T", dtype), ("D", head_dim), ("GQA", gqa), ("BLOCKS", blocks)],
        grid=(32 * kv_heads, gqa, blocks * count),
        threadgroup=(32, gqa, rows_per_threadgroup),
        output_shapes=[
            (heads * count * blocks * head_dim,),
            (heads * count * blocks,),
            (heads * count * blocks,),
        ],
        output_dtypes=[dtype, mx.float32, mx.float32],
    )
    return combine(2)(
        inputs=[partials, sums, maxs] + combine_extra,
        template=[("T", dtype), ("D", head_dim), ("BLOCKS", blocks)],
        grid=(heads * 1024, count, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(count, heads * head_dim)],
        output_dtypes=[dtype],
    )[0]


def run_rows(chains, combine, queries, keys, values, scale_array, *, plan, blocks,
             hpt, sg, mask=None, gate=None):
    """``[1, R, H * D]``: MLX's vector SDPA of R query rows over one K/V view
    (``queries [1, H, R, D]``), times ``gate`` (``[.., R, H * D]``) when given.
    ``chains(passes)`` / ``combine(passes)`` return the compiled kernels for
    this call's mask and gate."""
    _, heads, rows, head_dim = queries.shape
    kv_heads = keys.shape[1]
    gqa = heads // kv_heads
    dtype = queries.dtype
    mask_inputs = []
    if mask is not None:
        # MLX broadcasts the mask to [B, H, R, N] and reads the row and key
        # strides of the axes that are not singletons.
        mask_inputs = [mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))]
    gate_inputs = [] if gate is None else [gate.reshape(rows, heads * head_dim)]
    common = dict(dtype=dtype, head_dim=head_dim, gqa=gqa, kv_heads=kv_heads, heads=heads,
                  count=rows)
    if plan == 1:
        out = _one_pass(chains, combine, queries, keys, values, [scale_array], mask_inputs,
                        gate_inputs, hpt=hpt, sg=sg, **common)
    else:
        out = _two_pass(chains, combine, queries, keys, values, [scale_array], mask_inputs,
                        gate_inputs, blocks=blocks, rows_per_threadgroup=rows, **common)
    return out.reshape(1, rows, heads * head_dim)


def run_window(chains, combine, queries, keys, values, scale_array, *, runs, n_first,
               int_array, hpt, sg, masks=None, gate=None):
    """``[1, R, H * D]``: row ``j`` is the one-row vector SDPA of
    ``queries[:, :, j]`` over keys ``0 .. n_first + j - 1`` (and its own
    concatenated mask row when ``masks`` is given), times ``gate`` row ``j``
    when given.  ``runs`` are ``plan_runs`` of the rows' one-row plans;
    ``int_array(v)`` returns the module's cached int32 scalar."""
    _, heads, rows, head_dim = queries.shape
    kv_heads = keys.shape[1]
    gqa = heads // kv_heads
    dtype = queries.dtype
    if gate is not None:
        gate = gate.reshape(rows, heads * head_dim)
    outputs = []
    for start, stop, plan in runs:
        count = stop - start
        q = queries[:, :, start:stop]
        g = [] if gate is None else [gate[start:stop]]
        first = n_first + start
        extra = []
        if masks is not None:
            base = start * n_first + (start * (start - 1)) // 2
            extra = [masks, int_array(base)]
        passes, blocks = plan
        common = dict(dtype=dtype, head_dim=head_dim, gqa=gqa, kv_heads=kv_heads, heads=heads,
                      count=count)
        head_inputs = [scale_array, int_array(first)]
        if passes == 1:
            out = _one_pass(chains, combine, q, keys, values, head_inputs, extra, g,
                            hpt=hpt, sg=sg, **common)
        else:
            out = _two_pass(chains, combine, q, keys, values, head_inputs, extra, g,
                            blocks=blocks, rows_per_threadgroup=1, **common)
        outputs.append(out)
    out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
    return out.reshape(1, rows, heads * head_dim)
