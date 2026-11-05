# SPDX-License-Identifier: MIT
"""Expert-union routed MoE plus the folded shared expert, for multi-row verify.

``kblock_moe_shared`` (glm53_shared_fold=kblock) runs the shared expert as
slot TOPK of every token in the routed launches and combines it after the
routed slots. For several verify rows that path decodes every routed expert
once per slot. This composition keeps that law for every row and reads each
routed expert once per run of sorted slots instead:

* routed slots: ``kblock_moe_union``'s gate_up_union / down_union kernels
  (each slot's mid/out row equals GATE_UP/DOWN's for that slot alone, and
  GATE_UP_SHARED / DOWN_SHARED are GATE_UP / DOWN with only the slot index
  and the weight pointer substituted, so a routed slot of the fold has the
  same value);
* the shared slot: the plain GATE_UP / DOWN kernels with a one-expert table
  (EXPERTS = 1, every id 0) over the shared tensors -- the fold's shared slot
  is exactly that code with the pointer swapped;
* the fold's own COMBINE_SHARED kernel over the interleaved
  ``[token, TOPK + 1]`` slot rows (routed slots in order, shared last).

Every output row therefore equals ``kblock_moe_shared``'s bit for bit
(chunk law, EXACT = 0, VEC = 1, AHEAD = 1: the kernels' header carries the
load-ahead branch, so every launch names its AHEAD). Original mlx2 code; it reuses
kblock_metal's kernels and does not modify them.
"""

from __future__ import annotations

from .kblock_metal import _grid, _kernel, kblock_admits
from .reader import GGUFError


def kblock_moe_union_shared(x, ids, scores, gate, up, down, gate_sh, up_sh, down_sh, *, rows=4, group=2):
    import mlx.core as mx

    if (len(gate.shape) != 3 or gate.shape != up.shape or gate.ggml_type != up.ggml_type
            or gate.ggml_type not in (11, 12, 13, 14) or down.ggml_type not in (11, 12, 13, 14)):
        raise GGUFError("expert union needs matching K-quant gate/up and a K-quant down")
    experts, m, k = gate.shape
    if (down.shape != (experts, k, m) or x.ndim != 2 or x.shape[-1] != k
            or not kblock_admits(gate.ggml_type, k) or not kblock_admits(down.ggml_type, m)):
        raise GGUFError("K-block expert/input axes or formats mismatch")
    if (tuple(gate_sh.shape) != (m, k) or tuple(up_sh.shape) != (m, k) or tuple(down_sh.shape) != (k, m)
            or {gate_sh.ggml_type, up_sh.ggml_type} != {gate.ggml_type} or down_sh.ggml_type != down.ggml_type
            or gate_sh.row_bytes != gate.row_bytes or down_sh.row_bytes != down.row_bytes):
        raise GGUFError("shared expert must match the routed expert formats and shapes")
    if ids.ndim != 2 or ids.dtype != mx.uint32 or scores.shape != ids.shape or ids.shape[0] != x.shape[0]:
        raise GGUFError("matching [tokens,topk] uint32 IDs and scores required")
    b, topk = ids.shape
    x = x.astype(mx.float32)
    slots = b * topk
    flat = ids.reshape(-1)
    order = mx.argsort(flat).astype(mx.uint32)
    common = [("K", k), ("M", m), ("N", k), ("TOPK", topk), ("EXPERTS", experts), ("ROWS", rows),
              ("VEC", 1), ("EXACT", 0), ("G", group), ("SLOTS", slots), ("AHEAD", 1)]
    mid = _kernel("gate_up_union")(
        inputs=[x, flat, order, gate.data, up.data],
        template=common + [("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type)],
        grid=(_grid(m, rows), 1, slots), threadgroup=(128, 1, 1),
        output_shapes=[(slots, m)], output_dtypes=[mx.float32],
    )[0]
    routed = _kernel("down_union")(
        inputs=[mid, flat, order, down.data],
        template=common + [("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type)],
        grid=(_grid(k, rows), 1, slots), threadgroup=(128, 1, 1),
        output_shapes=[(slots, k)], output_dtypes=[mx.float32],
    )[0]
    # The shared slot: one-expert routed kernels over the shared tensors.
    zero = mx.zeros((b,), dtype=mx.uint32)
    single = [("K", k), ("M", m), ("N", k), ("TOPK", 1), ("EXPERTS", 1), ("ROWS", rows),
              ("VEC", 1), ("EXACT", 0), ("AHEAD", 1)]
    mid_sh = _kernel("gate_up")(
        inputs=[x, zero, gate_sh.data, up_sh.data],
        template=single + [("ROW_BYTES", gate_sh.row_bytes), ("FORMAT", gate_sh.ggml_type)],
        grid=(_grid(m, rows), 1, b), threadgroup=(128, 1, 1),
        output_shapes=[(b, m)], output_dtypes=[mx.float32],
    )[0]
    shared = _kernel("down")(
        inputs=[mid_sh, zero, down_sh.data],
        template=single + [("ROW_BYTES", down_sh.row_bytes), ("FORMAT", down_sh.ggml_type)],
        grid=(_grid(k, rows), 1, b), threadgroup=(128, 1, 1),
        output_shapes=[(b, k)], output_dtypes=[mx.float32],
    )[0]
    slots_out = mx.concatenate(
        [routed.reshape(b, topk, k), shared.reshape(b, 1, k)], axis=1).reshape(b * (topk + 1), k)
    fold = [("K", k), ("M", m), ("N", k), ("TOPK", topk), ("EXPERTS", experts), ("ROWS", rows),
            ("VEC", 1), ("EXACT", 0), ("AHEAD", 1)]
    return _kernel("combine_shared")(
        inputs=[slots_out, scores.astype(mx.float32)],
        template=fold,
        grid=(k, b, 1), threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(b, k)], output_dtypes=[mx.float32],
    )[0]
