# SPDX-License-Identifier: Apache-2.0
"""Lossless IQ1_S T64 banks and device-routed V4.1 expert kernels.

The byte permutation follows Splash's T64 planes; decode and selected NAX
retain the existing local V4.1 arithmetic. Missing routes are provisional:
callers must inspect the GPU mask and discard/retry before publishing output.
See provenance/deepseek-v41-iq1-tiled-kernels.json. No provider is selected here.
"""

from functools import lru_cache

import numpy as np

from . import v41_prefill as prefill_base
from . import v41_row_exact as exact_base
from .reader import GGUFError
from .v41_iq1_splash import HEADER as IQ1_HEADER

REVISION = "mlx2.v41-iq1-t64-slot-bank-exact-nax-runtime-capacity.v2"
EXPERTS = 384
MISSING_SLOT = 0xffffffff


def plane_sizes(shape):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[0] % 64 or shape[1] % 256):
        raise GGUFError("IQ1 tiled shape requires positive N%64=0 and K%256=0")
    n, k = shape
    return n*(k//32)*4, n*(k//32)*2, n*(k//256)*2


def pack(raw, shape):
    """CPU-only, exact byte permutation to flat contiguous qs/qh/d planes."""
    sizes = plane_sizes(shape)
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint8
            or not raw.flags.c_contiguous or raw.size != sum(sizes)):
        raise GGUFError("IQ1 tiled pack requires exact contiguous host U8 bytes")
    n, k = shape
    blocks = raw.reshape(n, k//256, 50)
    qs = blocks[:, :, 2:34].reshape(n//64, 64, k//32, 4).transpose(0, 2, 1, 3)
    qh = blocks[:, :, 34:50].reshape(n//64, 64, k//32, 2).transpose(0, 2, 1, 3)
    d = blocks[:, :, :2].reshape(n//64, 64, k//256, 2).transpose(0, 2, 1, 3)
    return tuple(np.ascontiguousarray(v).reshape(-1) for v in (qs, qh, d))


def unpack(planes, shape):
    """CPU-only inverse permutation; proves storage bytes without dequantizing."""
    sizes = plane_sizes(shape)
    if (not isinstance(planes, (tuple, list)) or len(planes) != 3
            or any(not isinstance(v, np.ndarray) or v.dtype != np.uint8 or v.ndim != 1
                   or not v.flags.c_contiguous or v.size != size for v, size in zip(planes, sizes))):
        raise GGUFError("IQ1 tiled unpack requires exact flat contiguous host planes")
    n, k = shape
    raw = np.empty((n, k//256, 50), np.uint8)
    raw[:, :, 2:34] = planes[0].reshape(n//64, k//32, 64, 4).transpose(0, 2, 1, 3).reshape(n, k//256, 32)
    raw[:, :, 34:50] = planes[1].reshape(n//64, k//32, 64, 2).transpose(0, 2, 1, 3).reshape(n, k//256, 16)
    raw[:, :, :2] = planes[2].reshape(n//64, k//256, 64, 2).transpose(0, 2, 1, 3).reshape(n, k//256, 2)
    return raw.reshape(-1)


def bank_plane_sizes(capacity, hidden_size, intermediate_size):
    if type(capacity) is not int or not 0 <= capacity <= EXPERTS:
        raise GGUFError("IQ1 bank capacity must be 0..384")
    gate = plane_sizes((intermediate_size, hidden_size))
    down = plane_sizes((hidden_size, intermediate_size))
    return tuple(capacity*size for size in (*gate, *gate, *down))


HEADER = IQ1_HEADER+r"""
template<typename QS,typename QH,typename D>
inline float v41_iq1_tiled(QS qs, QH qh, D d, uint slot, uint row, uint k,
                           uint N, uint K) {
    const ulong group=ulong(slot)*N*(K/32)+(ulong(row/64)*(K/32)+k/32)*64+row%64;
    const ulong block=ulong(slot)*N*(K/256)+(ulong(row/64)*(K/256)+k/256)*64+row%64;
    const uint high=uint(qh[group*2])|(uint(qh[group*2+1])<<8);
    const ushort bits=ushort(d[block*2])|(ushort(d[block*2+1])<<8);
    return v41_iq1_scaled(float(as_type<half>(bits)),high,uint(qs[group*4+(k%32)/8]),k%32);
}
"""
NAX_HEADER = HEADER+"\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"

MISSING_SOURCE = r"""
const uint capacity=capacity_value[0];
const uint i=thread_position_in_grid.x;
if(i>=ROUTES) return;
const uint expert=ids[i];
if(expert>=EXPERTS) { missing[i]=true; return; }
missing[i]=slot_map[expert]>=capacity;
"""

EXACT_PREFIX = r"""
const uint lane=thread_index_in_simdgroup;
const uint row=threadgroup_position_in_grid.y*SIMDS+simdgroup_index_in_threadgroup;
const uint first=threadgroup_position_in_grid.z, count=1;
const uint expert=ids[first];
if(expert>=EXPERTS) { if(lane==0) y[ulong(first)*N+row]=0.0f; return; }
const uint slot=slot_map[expert];
if(slot>=capacity) { if(lane==0) y[ulong(first)*N+row]=0.0f; return; }
"""

EXACT_COMBINE = r"""
const uint row=thread_position_in_grid.x, token=thread_position_in_grid.y;
if(row>=N) return;
float total=0.0f;
for(uint slot=0;slot<TOPK;slot++) total+=rows[(ulong(token)*TOPK+slot)*N+row];
y[ulong(token)*N+row]=total;
"""


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise GGUFError("tiled IQ1 source anchor changed")
    return source.replace(before, after, 1)


def exact_spec(kind):
    if kind not in ("gate", "down"):
        raise GGUFError("unknown tiled exact projection")
    _, source = exact_base.kernel_spec(kind, 1, False)
    start = "float sg[ROWS]" if kind == "gate" else "float sums[ROWS]"
    plane = "gqs" if kind == "gate" else "wqs"
    # Shape metadata is a runtime buffer. Bank growth must not create a new
    # Metal specialization, and its physical extent remains the access bound.
    bound = f"const uint capacity=uint({plane}_shape[0]/(ulong(N)*K/8));\n"
    source = bound+EXACT_PREFIX+source[source.index(start):]
    names = ["x", "ids", "slot_map"]
    if kind == "gate":
        names += ["scores", "gqs", "gqh", "gd", "uqs", "uqh", "ud"]
        source = source.replace("order[first+r]", "(first+r)")
        for old, prefix in (("g", "g"), ("u", "u")):
            source = _replace_once(source, f"gguf_value({old},k,19)",
                f"v41_iq1_tiled({prefix}qs,{prefix}qh,{prefix}d,slot,row,k,N,K)")
    else:
        names += ["wqs", "wqh", "wd"]
        source = _replace_once(source, "gguf_value(w,k,19)", "v41_iq1_tiled(wqs,wqh,wd,slot,row,k,N,K)")
    return names, ["y"], source


SCAN_SOURCE = "const uint capacity=capacity_value[0];\n"+_replace_once(prefill_base.SCAN_SOURCE,
    "if(original_ids[row]>=EXPERTS) atomic_store_explicit(&invalid,1u,memory_order_relaxed);",
    "const uint expert=original_ids[row];\n"
    "    if(expert>=EXPERTS) atomic_store_explicit(&invalid,1u,memory_order_relaxed);\n"
    "    else if(slot_map[expert]>=capacity) atomic_store_explicit(&invalid,1u,memory_order_relaxed);")


def prefill_spec(kind):
    if kind == "scan":
        return ["sorted_ids", "original_ids", "slot_map", "capacity_value"], ["tiles", "tile_count", "status"], SCAN_SOURCE
    if kind == "combine":
        return ["rows", "inverse", "scores", "status"], ["out"], prefill_base.COMBINE_SOURCE
    if kind not in ("gate", "down"):
        raise GGUFError("unknown tiled prefill projection")
    source = prefill_base.GATE_SOURCE if kind == "gate" else prefill_base.DOWN_SOURCE
    # Slot checks are uniform over a threadgroup and precede every barrier and
    # bank access. The scan normally excludes such tiles; this also guards an
    # independently supplied descriptor.
    marker = "const uint rowbase=desc.x,expert=desc.y,valid_rows=desc.z;"
    guard = marker+r"""
const uint capacity=uint(wqs_shape[0]/(ulong(N)*K/8));
uint slot=capacity;
if(expert<EXPERTS) slot=slot_map[expert];
if(slot>=capacity) {
    for(uint i=tid;i<valid_rows*128;i+=128) {
        const uint col=threadgroup_position_in_grid.x*128+i%128;
        if(col<N) y[ulong(rowbase+i/128)*N+col]=0.0f;
    }
    return;
}
"""
    source = _replace_once(source, marker, guard)
    source = _replace_once(source, "const uint4 desc=*((device const uint4*)tiles+tile);",
        "const uint4 desc=uint4(tiles[4*tile],tiles[4*tile+1],tiles[4*tile+2],tiles[4*tile+3]);")
    names = ["x"]+( ["order"] if kind == "gate" else [])+["tiles", "tile_count", "slot_map", "wqs", "wqh", "wd"]
    source = _replace_once(source, "gguf_value(w+(ulong(expert)*N+row)*ROW_BYTES,k,FORMAT)",
        "v41_iq1_tiled(wqs,wqh,wd,slot,row,k,N,K)")
    if kind == "gate":
        names += ["uqs", "uqh", "ud", "scores"]
        source = _replace_once(source, "gguf_value(up+(ulong(expert)*N+row)*ROW_BYTES,k,FORMAT)",
            "v41_iq1_tiled(uqs,uqh,ud,slot,row,k,N,K)")
    return names, ["y"], source


@lru_cache(maxsize=8)
def kernel(kind):
    import mlx.core as mx
    if kind == "missing":
        names, outputs, source = ["ids", "slot_map", "capacity_value"], ["missing"], MISSING_SOURCE
    elif kind == "exact_combine":
        names, outputs, source = ["rows"], ["y"], EXACT_COMBINE
    elif kind.startswith("exact_"):
        names, outputs, source = exact_spec(kind[6:])
    else:
        names, outputs, source = prefill_spec(kind[8:])
    return mx.fast.metal_kernel(name="mlx2_v41_iq1_tiled_"+kind+"_v2", input_names=names,
        output_names=outputs, source=source, header=NAX_HEADER if kind.startswith("prefill_") else HEADER,
        ensure_row_contiguous=True)


def _routes(ids, slot_map, capacity):
    import mlx.core as mx
    if (type(capacity) is not int or not 0 <= capacity <= EXPERTS
            or not isinstance(ids, mx.array) or ids.dtype != mx.uint32 or ids.ndim != 2
            or not 1 <= ids.shape[0] <= 1024 or not 1 <= ids.shape[1] <= 6
            or not isinstance(slot_map, mx.array) or slot_map.dtype != mx.uint32 or slot_map.ndim != 1
            or not 1 <= slot_map.size <= EXPERTS or capacity > slot_map.size):
        raise GGUFError("tiled IQ1 routes require uint32[B,top6] and a map of 1..384 experts")


def missing_routes(ids, slot_map, capacity):
    """GPU bool per route; no expert IDs or hit decisions are read by Python."""
    import mlx.core as mx
    _routes(ids, slot_map, capacity)
    if capacity == 0:
        return mx.ones(ids.shape, mx.bool_)
    return kernel("missing")(inputs=[ids, slot_map, mx.array([capacity], mx.uint32)],
        template=[("ROUTES", ids.size), ("EXPERTS", slot_map.size)],
        grid=(((ids.size+255)//256)*256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[ids.shape], output_dtypes=[mx.bool_])[0]


def _validate(x, ids, scores, planes, slot_map, capacity, hidden_size, intermediate_size, prefill, max_scratch_bytes):
    import mlx.core as mx
    _routes(ids, slot_map, capacity)
    sizes = bank_plane_sizes(capacity, hidden_size, intermediate_size)
    if (type(prefill) is not bool or not isinstance(x, mx.array) or x.ndim != 2
            or x.shape != (ids.shape[0], hidden_size) or not isinstance(scores, mx.array)
            or scores.shape != ids.shape or x.dtype not in (mx.float32, mx.bfloat16, mx.float16)
            or scores.dtype not in (mx.float32, mx.bfloat16, mx.float16)
            or not isinstance(planes, (tuple, list)) or len(planes) != 9
            or any(not isinstance(p, mx.array) or p.dtype != mx.uint8 or p.ndim != 1
                   or p.size != size for p, size in zip(planes, sizes))):
        raise GGUFError("tiled IQ1 bank/input geometry differs from admission")
    # Mask and scan each carry one runtime U32 capacity scalar, never a
    # persistent GPU cache of specializations or control values.
    scratch = prefill_base.scratch_bytes(ids.shape[0], ids.shape[1], intermediate_size, hidden_size, slot_map.size)+8
    if type(max_scratch_bytes) is not int or max_scratch_bytes < scratch:
        raise GGUFError("tiled IQ1 activation scratch exceeds explicit bound")


def moe(x, ids, scores, planes, slot_map, *, capacity, hidden_size, intermediate_size,
        prefill=False, max_scratch_bytes=512 << 20, counters=None):
    """GPU-routed exact or explicitly selected NAX pass over immutable banks.

    The caller owns the bank through evaluation and must discard/retry when
    missing_routes is true. Empty banks return zeros. Missing NAX routes set
    scan status so the combine returns NaNs without reading undefined outputs.
    """
    import mlx.core as mx
    _validate(x, ids, scores, planes, slot_map, capacity, hidden_size, intermediate_size, prefill, max_scratch_bytes)
    if counters is not None:
        key = "tiled_nax_calls" if prefill else "tiled_exact_calls"
        counters[key] = counters.get(key, 0)+1
    if capacity == 0:
        return mx.zeros(x.shape, mx.float32)
    if prefill:
        if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
            raise GGUFError("tiled NAX requires an owned Apple M5 GPU window")
        return _prefill(x, ids, scores, planes, slot_map, capacity, hidden_size, intermediate_size)
    routes = ids.size
    mid = kernel("exact_gate")(inputs=[x.astype(mx.float32), ids, slot_map, scores.reshape(-1).astype(mx.float32), *planes[:6]],
        template=[("EXPERTS", slot_map.size), ("TOPK", ids.shape[1]), ("K", hidden_size), ("N", intermediate_size), ("ROWS", 1), ("SIMDS", 4)],
        grid=(128, intermediate_size//4, routes), threadgroup=(128, 1, 1),
        output_shapes=[(routes, intermediate_size)], output_dtypes=[mx.float32])[0]
    projected = kernel("exact_down")(inputs=[mid, ids, slot_map, *planes[6:]],
        template=[("EXPERTS", slot_map.size), ("K", intermediate_size), ("N", hidden_size), ("ROWS", 1), ("SIMDS", 4)],
        grid=(128, hidden_size//4, routes), threadgroup=(128, 1, 1),
        output_shapes=[(routes, hidden_size)], output_dtypes=[mx.float32])[0]
    return kernel("exact_combine")(inputs=[projected], template=[("N", hidden_size), ("TOPK", ids.shape[1])],
        grid=(((hidden_size+255)//256)*256, ids.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


def _prefill(x, ids, scores, planes, slot_map, capacity, hidden, intermediate):
    import mlx.core as mx
    rows = ids.size
    tiles_bound = prefill_base.plan_capacity(rows, slot_map.size)
    flat = ids.reshape(-1)
    order = mx.argsort(flat*mx.array(rows, mx.uint32)+mx.arange(rows, dtype=mx.uint32)).astype(mx.uint32)
    inverse = mx.argsort(order).astype(mx.uint32)
    tiles, count, status = kernel("prefill_scan")(inputs=[flat[order], flat, slot_map, mx.array([capacity], mx.uint32)],
        template=[("ROWS", rows), ("EXPERTS", slot_map.size), ("MAX_TILES", tiles_bound), ("TILE", prefill_base.TILE_ROWS)],
        grid=(512, 1, 1), threadgroup=(512, 1, 1),
        output_shapes=[(tiles_bound, 4), (1,), (1,)], output_dtypes=[mx.uint32]*3)
    mid = kernel("prefill_gate")(inputs=[x.astype(mx.float32), order, tiles, count, slot_map, *planes[:6], scores.reshape(-1).astype(mx.float32)],
        template=[("TOPK", ids.shape[1]), ("K", hidden), ("N", intermediate), ("EXPERTS", slot_map.size), ("MT", prefill_base.MT)],
        grid=(((intermediate+127)//128)*128, tiles_bound, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, intermediate)], output_dtypes=[mx.float32])[0]
    projected = kernel("prefill_down")(inputs=[mid, tiles, count, slot_map, *planes[6:]],
        template=[("K", intermediate), ("N", hidden), ("EXPERTS", slot_map.size), ("MT", prefill_base.MT)],
        grid=(((hidden+127)//128)*128, tiles_bound, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, hidden)], output_dtypes=[mx.float32])[0]
    return kernel("prefill_combine")(inputs=[projected, inverse, scores.reshape(-1).astype(mx.float32), status],
        template=[("TOKENS", ids.shape[0]), ("TOPK", ids.shape[1]), ("N", hidden)],
        grid=(((hidden+255)//256)*256, ids.shape[0], 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


def metal_compile_source(kind, *, constant_inputs=False):
    """Standalone offline compiler input; never initializes a GPU device."""
    if kind == "missing":
        names, outputs, source = ["ids", "slot_map", "capacity_value"], ["missing"], MISSING_SOURCE
    elif kind == "exact_combine":
        names, outputs, source = ["rows"], ["y"], EXACT_COMBINE
    elif kind.startswith("exact_"):
        names, outputs, source = exact_spec(kind[6:])
    else:
        names, outputs, source = prefill_spec(kind[8:])
    uints = {"ids", "slot_map", "order", "inverse", "tiles", "tile_count", "status", "sorted_ids", "original_ids", "capacity_value"}
    bindings = []
    for i, name in enumerate(names):
        dtype = "uint" if name in uints else "float" if name in ("x", "scores", "rows") else "uchar"
        space = "constant" if constant_inputs else "device"
        bindings.append(f"{space} const {dtype}* {name} [[buffer({i})]]")
    for name in names:
        if name+"_shape" in source:
            bindings.append(f"constant int* {name}_shape [[buffer({len(bindings)})]]")
    for i, name in enumerate(outputs, len(bindings)):
        dtype = "uint" if kind == "prefill_scan" else "bool" if kind == "missing" else "float"
        bindings.append(f"device {dtype}* {name} [[buffer({i})]]")
    attrs = ["uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]", "uint3 thread_position_in_grid [[thread_position_in_grid]]",
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]", "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]"]
    constants = "constant uint K=5120,N=2304,TOPK=6,ROWS=1,SIMDS=4,EXPERTS=384,ROUTES=6,TOKENS=1,MAX_TILES=6,TILE=16,MT=1;\n"
    return "#include <metal_stdlib>\nusing namespace metal;\n"+(NAX_HEADER if kind.startswith("prefill_") else HEADER)+constants+"kernel void probe("+",".join(bindings+attrs)+") {\n"+source+"\n}\n"
