# SPDX-License-Identifier: MIT
"""Bounded expert-major prefill worklists without coefficient repacking.

This candidate batches existing Source gate/up/middle arithmetic. Original
experts in one batch have an identical local row count; Q1 experts may differ.
It never changes a serving policy or owns expert backing lifetimes.
See provenance/deepseek-v41-source-gate-up-prefill.json.
"""
from functools import lru_cache

from . import source_gate_up_fusion as gu

MAX_EXPERTS = 15
MAX_INPUT_BINDINGS = 20


def descriptor(kind, row_counts):
    """CPU-only worklist geometry and prefix offsets, never tensor values."""
    row_counts = tuple(row_counts)
    if not 1 <= len(row_counts) <= MAX_EXPERTS:
        raise ValueError("Source prefill GU requires1..15 expert objects")
    plans = [gu.descriptor(kind, rows) for rows in row_counts]
    if kind == "original" and len(set(row_counts)) != 1:
        raise ValueError("Source prefill GU original local row counts must match")
    offsets = [0]
    for rows in row_counts:
        offsets.append(offsets[-1]+rows)
    return {"kind": kind, "row_counts": row_counts, "offsets": tuple(offsets),
            "grid": (72*256, max(p['grid'][1] for p in plans), len(row_counts)),
            "threadgroup": (256, 1, 1), "output_shape": (offsets[-1], 2304),
            "logical_input_bindings": len(row_counts)+5,
            "expert_backing_bindings": len(row_counts),
            "threadgroup_middle_bytes": max(p['threadgroup_middle_bytes'] for p in plans),
            "raw_gate_up_global_bytes": 0, "coefficient_copy_bytes": 0}


def _geometry(x, experts, rows, scores, row_counts):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert, ExpertLayout
    experts = tuple(experts)
    if not experts or not isinstance(experts[0], PackedExpert):
        raise ValueError("Source prefill GU requires PackedExpert objects")
    kind = experts[0].layout.kind
    plan = descriptor(kind, row_counts)
    if len(experts) != len(plan['row_counts']):
        raise ValueError("Source prefill GU expert/count length mismatch")
    if (x.ndim != 2 or x.shape[1] != 5120 or not 1 <= x.shape[0] <= 1024
            or x.dtype != mx.float32 or rows.shape != (plan['output_shape'][0],)
            or rows.dtype != mx.uint32 or scores.shape != rows.shape or scores.dtype != mx.float32):
        raise ValueError("Source prefill GU input geometry/dtype mismatch")
    layouts = []
    for expert in experts:
        if (not isinstance(expert, PackedExpert) or not isinstance(expert.layout, ExpertLayout)
                or expert.layout.kind != kind or expert.backing.shape != (expert.layout.nbytes,)
                or expert.backing.dtype != mx.uint8):
            raise ValueError("Source prefill GU packed backing or format mismatch")
        layouts.append(gu._batch_layout(expert.layout))
    # Q1 row counts do not affect its dot law or compiled kernel identity.
    selected_rows = plan['row_counts'][0] if kind == 'original' else 1
    return (tuple(layouts), selected_rows), plan


def supports(x, experts, rows, scores, row_counts):
    """Structural gate; provider owns device and trusted row-map validity."""
    try:
        _geometry(x, experts, rows, scores, row_counts)
    except (ValueError, TypeError, AttributeError, KeyError, StopIteration):
        return False
    return True


def kernel_spec(key, *, exp_spelling="metal::exp", return_projections=False):
    """Generate source from primitive descriptors without holding tensors."""
    layouts, selected_rows = key
    layouts = tuple(layouts)
    if not layouts:
        raise ValueError("Source prefill GU empty layout key")
    kind = layouts[0][0]
    descriptor(kind, (selected_rows,)*len(layouts))
    if any(item[0] != kind for item in layouts):
        raise ValueError("Source prefill GU mixed-format batch declined")
    # Reuse its strict primitive offset validation and served-exp closed set.
    gu.batch_kernel_spec(layouts[:6], exp_spelling=exp_spelling,
                         return_projections=return_projections)
    for item in layouts[6:]:
        gu.batch_kernel_spec((item,), exp_spelling=exp_spelling,
                             return_projections=return_projections)
    ordinary = gu.kernel_spec(kind, exp_spelling=exp_spelling,
        selected_rows=selected_rows, return_projections=return_projections)
    pointers = (("uint32_t", "gate_q"), ("half", "gate_s"), ("half", "gate_b"),
                ("uint32_t", "up_q"), ("half", "up_s"), ("half", "up_b")) if kind == 'q1' else (
                ("uint32_t", "gate_q"), ("uint8_t", "gate_s"),
                ("uint32_t", "up_q"), ("uint8_t", "up_s"))
    source = ("const uint expert_slot=threadgroup_position_in_grid.z;\n"
              "const uint row_base=offsets[expert_slot];\n"
              "const uint row_count=offsets[expert_slot+1]-row_base;\n")
    if kind == 'q1':
        source += "if(threadgroup_position_in_grid.y>=row_count) return;\n"
    for dtype, name in pointers:
        source += f"const device {dtype}* {name};\n"
    source += "switch(expert_slot) {\n"
    for index, item in enumerate(layouts):
        source += f"case {index}: {{\n"
        for (dtype, name), offset in zip(pointers, item[1:]):
            source += f"{name}=(const device {dtype}*)(expert{index}+{offset});\n"
        source += "break;\n}\n"
    source += "default: return;\n}\n"
    body = ordinary['source']
    if kind == 'original' and selected_rows > 1:
        body = body.replace("rows[min(vec0+v,GU_ROWS-1)]", "rows[row_base+min(vec0+v,GU_ROWS-1)]")
        body = body.replace("scores[vec0+v]", "scores[row_base+vec0+v]")
        body = body.replace("(vec0+sg)*2304", "(row_base+vec0+sg)*2304")
        body = body.replace("(vec0+v)*2304", "(row_base+vec0+v)*2304")
    else:
        body = body.replace("rows[selected_row]", "rows[row_base+selected_row]")
        body = body.replace("scores[selected_row]", "scores[row_base+selected_row]")
        body = body.replace("selected_row*2304", "(row_base+selected_row)*2304")
    return {"header": ordinary['header'], "source": source+body,
            "input_names": ["x", "rows", "scores", "offsets", "inverse"]+
                           [f"expert{i}" for i in range(len(layouts))],
            "output_names": ordinary['output_names']}


@lru_cache(maxsize=256)
def _kernel(key, exp_spelling, return_projections):
    import hashlib
    import mlx.core as mx
    spec = kernel_spec(key, exp_spelling=exp_spelling, return_projections=return_projections)
    identity = hashlib.sha256(repr((key, exp_spelling, return_projections)).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name=f"mlx2_source_gu_prefill_{identity}_v1",
                               ensure_row_contiguous=True, **spec)


def pp_middle(x, experts, rows, scores, row_counts, *, exp_spelling="metal::exp",
              return_projections=False, row_packet=None, batch_index=None):
    """Return flat expert-major middle rows, with no raw GU global buffer.

    ``row_counts`` is CPU routing metadata. ``rows`` and ``scores`` contain
    concatenated matching occurrences in each expert's ordinary local order.
    Caller retains every backing through its completion epoch. Kernel caches
    retain primitive immutable descriptors only, never an expert or backing.
    """
    import mlx.core as mx
    experts = tuple(experts)
    key, plan = _geometry(x, experts, rows, scores, row_counts)
    # Validate flags before creating metadata or a Metal kernel.
    if exp_spelling not in gu.EXP_SPELLINGS or type(return_projections) is not bool:
        raise ValueError("Source prefill GU invalid epilogue/diagnostic option")
    if row_packet is None:
        if batch_index is not None:
            raise ValueError('Source prefill GU batch index requires its row packet')
        offsets=mx.array(plan['offsets'],mx.uint32)
    else:
        from ...adapters.deepseek_v41_source_row_packet import DeviceRowPacket
        if (not isinstance(row_packet,DeviceRowPacket) or type(batch_index) is not int
                or not 0 <= batch_index < len(row_packet.packet.plan.batches)
                or row_packet.packet.plan.batches[batch_index].row_counts != tuple(row_counts)
                or rows is not row_packet.view(f'batch{batch_index}.rows')):
            raise ValueError('Source prefill GU row packet batch mismatch')
        offsets=row_packet.view(f'batch{batch_index}.offsets')
    inputs = [x, rows, scores, offsets,
              mx.array([1/448], mx.float32)]+[e.backing for e in experts]
    count = 3 if return_projections else 1
    result = _kernel(key, exp_spelling, return_projections)(inputs=inputs,
        grid=plan['grid'], threadgroup=plan['threadgroup'],
        output_shapes=[plan['output_shape']]*count, output_dtypes=[mx.float32]*count)
    return tuple(result) if return_projections else result[0]
