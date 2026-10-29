# SPDX-License-Identifier: MIT
"""Explicit Source dispatch fusions; preserve its ordered F32 expert sum.

Mechanism provenance: provenance/deepseek-v41-source-dispatch-fusion.json.
Only immutable activation outputs are bound here; expert leases remain owned
by the Source provider's existing completion window.
"""
from functools import lru_cache

import numpy as np

POLICIES = ("reference", "hoisted_fused_fp32", "ordered_fused_fp32", "paired_q1_fused_fp32", "paired_fused_fp32", "epilogue_fused_fp32", "routed_fused_fp32")


def ordered_offsets(values, order):
    """Map original route slots to expert-sorted output rows, in sum order."""
    if (values.dtype != np.uint32 or values.ndim != 2 or not all(values.shape)
            or order.ndim != 1 or order.size != values.size):
        raise ValueError("Source ordered combine route geometry mismatch")
    inverse = np.empty(order.size, dtype=np.uint32)
    inverse[order] = np.arange(order.size, dtype=np.uint32)
    ranking = np.argsort(values, axis=1, kind="stable")
    return np.take_along_axis(inverse.reshape(values.shape), ranking, axis=1)


@lru_cache(maxsize=9)
def _combine_kernel(direct_count):
    import mlx.core as mx
    if direct_count:
        names = [f"part{i}" for i in range(direct_count)]
        body = "\n".join(f"total = total + part{i}[column];" for i in range(direct_count))
    else:
        names = ["sorted_output", "offsets"]
        body = """
        #pragma clang loop unroll(disable)
        for (uint slot=0; slot<TOPK; ++slot) {
            total = total + sorted_output[offsets[row*TOPK+slot]*WIDTH+column];
        }
        """
    return mx.fast.metal_kernel(
        name=f"mlx2_source_ordered_combine_{direct_count}_v1",
        input_names=names, output_names=["output"],
        header="#pragma clang fp contract(off)\n",
        source="""
        const uint i=thread_position_in_grid.x;
        if(i>=ROWS*WIDTH) return;
        const uint row=i/WIDTH, column=i%WIDTH;
        float total=0.0f;
        """ + body + "\noutput[i]=total;\n",
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"},
    )


def ordered_combine(parts, values, order, *, offsets=None, row_packet=None):
    """Ordered Source sum, optionally borrowing a validated frame-plan map.

    ``offsets`` is authoritative metadata from the same frame's route planner;
    the caller retains its immutable owner. No route contents are cached here.
    """
    import mlx.core as mx
    device_offsets=None
    if row_packet is not None:
        from .deepseek_v41_source_row_packet import DeviceRowPacket
        if (not isinstance(row_packet,DeviceRowPacket)
                or row_packet.packet.plan.values is not values
                or row_packet.packet.plan.order is not order
                or row_packet.packet.plan.combine_offsets is not offsets):
            raise ValueError('Source combine row packet belongs to another route frame')
        device_offsets=row_packet.view('combine_offsets')
    if offsets is None:
        offsets = ordered_offsets(values, order)
    elif (not isinstance(values,np.ndarray) or values.dtype!=np.uint32 or values.ndim!=2
            or not all(values.shape) or not isinstance(order,np.ndarray) or order.ndim!=1
            or order.size!=values.size or order.dtype.kind not in 'iu' or not isinstance(offsets,np.ndarray)
            or offsets.dtype!=np.uint32 or offsets.shape!=values.shape
            or not offsets.flags.c_contiguous or offsets.flags.writeable
            or np.any(offsets>=order.size)):
        raise ValueError('Source ordered combine precomputed frame-map geometry mismatch')
    if (not parts or any(p.ndim != 2 or p.dtype != mx.float32 for p in parts)
            or len({p.shape[1] for p in parts}) != 1
            or sum(p.shape[0] for p in parts) != values.size):
        raise ValueError("Source ordered combine output geometry mismatch")
    rows, topk = values.shape
    width = parts[0].shape[1]
    if width <= 0:
        raise ValueError("Source ordered combine empty output width")
    # The provider forms parts in ascending unique expert-ID order. For one
    # row with distinct routes this is already the required accumulation order.
    direct = rows == 1 and len(parts) == topk and all(p.shape[0] == 1 for p in parts)
    if direct:
        inputs = parts
    else:
        packed = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        inputs = [packed, mx.array(offsets, mx.uint32) if device_offsets is None else device_offsets]
    return _combine_kernel(topk if direct else 0)(
        inputs=inputs, template=[("ROWS", rows), ("WIDTH", width), ("TOPK", topk)],
        grid=(rows*width, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)], output_dtypes=[mx.float32],
    )[0]
