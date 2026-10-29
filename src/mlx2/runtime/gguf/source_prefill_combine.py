# SPDX-License-Identifier: MIT
"""Bounded immutable direct-part Source combine; no expert-output concatenate.

Chunks visit parts in globally ascending expert order. Within every chunk,
each token's present route slots retain the ordinary sequential F32 sum.
Absent references perform no addition. The prior chunk's F32 output carries
the accumulator across dispatches without changing its value or precision.

Provenance: provenance/deepseek-v41-source-prefill-combine.json. This module
does not select any serving policy or change expert-weight ownership.
"""
from functools import lru_cache
from dataclasses import dataclass
import operator
import weakref

import numpy as np

MAX_PARTS = 28
MAX_LOCAL_ROWS = 1 << 16
_PACKET_SEAL = object()


def _validate_routes(values,order,chunk_parts):
    if type(chunk_parts) is not int or not 1 <= chunk_parts <= MAX_PARTS:
        raise ValueError('Source direct combine chunk_parts must be1..28')
    if (not isinstance(values,np.ndarray) or values.dtype != np.uint32
            or values.ndim != 2 or not 1 <= values.shape[0] <= 1024
            or not 1 <= values.shape[1] <= 6 or np.any(values >= 384)):
        raise ValueError('Source direct combine route geometry/identity mismatch')
    if (not isinstance(order,np.ndarray) or order.ndim != 1
            or order.size != values.size or order.dtype.kind not in 'iu'
            or np.any(order < 0) or np.any(order >= values.size)):
        raise ValueError('Source direct combine order geometry mismatch')
    order = order.astype(np.intp,copy=False)
    if np.any(np.bincount(order,minlength=values.size) != 1):
        raise ValueError('Source direct combine order must be a permutation')
    sorted_ids=values.ravel()[order]
    if (np.any(sorted_ids[1:] < sorted_ids[:-1])
            or np.any((sorted_ids[1:] == sorted_ids[:-1]) & (order[1:] < order[:-1]))):
        raise ValueError('Source direct combine order must be stable expert-ID sort')
    return order,sorted_ids


def plan(values, order, part_rows, *, chunk_parts=MAX_PARTS):
    """CPU plan of ordered uint32 references; no MLX evaluation or allocation."""
    order,_ = _validate_routes(values,order,chunk_parts)
    counts=[]
    for value in part_rows:
        try:
            if isinstance(value,bool):raise TypeError
            count=operator.index(value)
        except TypeError as error:
            raise ValueError('Source direct combine part row counts must be integers') from error
        if not 1 <= count <= MAX_LOCAL_ROWS:
            raise ValueError('Source direct combine local row bound exceeded')
        counts.append(count)
    if not counts or sum(counts) != values.size:
        raise ValueError('Source direct combine part row extent mismatch')
    from ...adapters.deepseek_v41_source_math import ordered_offsets
    offsets=ordered_offsets(values,order)
    stops=np.cumsum(counts,dtype=np.int64)
    starts=np.concatenate((np.zeros(1,np.int64),stops[:-1]))
    owner=np.searchsorted(stops,offsets,side='right')
    local=offsets-starts[owner]
    chunks=[]
    for begin in range(0,len(counts),chunk_parts):
        end=min(begin+chunk_parts,len(counts))
        refs=np.zeros(values.shape,np.uint32)
        selected=(owner >= begin) & (owner < end)
        refs[selected]=((owner[selected]-begin+1).astype(np.uint32)<<16)|local[selected].astype(np.uint32)
        first=begin==0
        chunks.append(dict(begin=begin,end=end,first=first,refs=refs,
                           input_bindings=end-begin+1+(0 if first else 1),
                           total_buffer_bindings=end-begin+2+(0 if first else 1)))
    return chunks


def kernel_spec(part_count, first):
    """Pure source metadata; source cache never retains arrays or expert parts."""
    if type(part_count) is not int or not 1 <= part_count <= MAX_PARTS or type(first) is not bool:
        raise ValueError('Source direct combine invalid kernel descriptor')
    names=[f'part{i}' for i in range(part_count)]
    names += ([] if first else ['previous'])+['refs']
    cases='\n'.join(f'case {i+1}: total=total+part{i}[row*WIDTH+column]; break;'
                    for i in range(part_count))
    source=r'''
const uint i=thread_position_in_grid.x;
if(i>=ROWS*WIDTH) return;
const uint token=i/WIDTH,column=i%WIDTH;
''' + ('float total=0.0f;\n' if first else 'float total=previous[i];\n') + r'''
#pragma clang loop unroll(disable)
for(uint slot=0;slot<TOPK;++slot) {
    const uint ref=refs[token*TOPK+slot];
    if(ref==0) continue;
    const uint part=ref>>16;
    const uint row=ref&0xffffu;
    switch(part) {
''' + cases + r'''
    }
}
output[i]=total;
'''
    return dict(input_names=names,output_names=['output'],
                header='#pragma clang fp contract(off)\n',source=source,
                ensure_row_contiguous=True,compile_options={'math_mode':'safe'})


@lru_cache(maxsize=MAX_PARTS*2)
def _kernel(part_count, first):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f'mlx2_source_direct_parts_{part_count}_first{int(first)}_v1',
        **kernel_spec(part_count,first))


def combine(parts, values, order, *, chunk_parts=MAX_PARTS):
    """Ordered sum from direct, contiguous F32 expert outputs in bounded groups.

    Only immutable output arrays are produced. The caller retains expert
    lifetimes through its existing completion window. Strided stand-alone
    callers may require MLX's normal contiguous-input materialization; the
    provider's ordinary per-expert projection outputs are already contiguous.
    """
    import mlx.core as mx
    parts=tuple(parts)
    if (not parts or any(p.ndim != 2 or p.dtype != mx.float32 for p in parts)
            or len({p.shape[1] for p in parts}) != 1 or parts[0].shape[1] <= 0):
        raise ValueError('Source direct combine part geometry/dtype mismatch')
    chunks=plan(values,order,[p.shape[0] for p in parts],chunk_parts=chunk_parts)
    rows,topk=values.shape;width=parts[0].shape[1]
    result=None
    for chunk in chunks:
        inputs=list(parts[chunk['begin']:chunk['end']])
        if not chunk['first']:inputs.append(result)
        inputs.append(mx.array(chunk['refs'],mx.uint32))
        result=_kernel(chunk['end']-chunk['begin'],chunk['first'])(
            inputs=inputs,template=[('ROWS',rows),('WIDTH',width),('TOPK',topk)],
            grid=(rows*width,1,1),threadgroup=(256,1,1),
            output_shapes=[(rows,width)],output_dtypes=[mx.float32])[0]
    return result


def _mapped_offsets(values,order,normalized_order,route_plan):
    if route_plan is None:
        from ...adapters.deepseek_v41_source_math import ordered_offsets
        return ordered_offsets(values,normalized_order)
    from ...adapters.deepseek_v41_source_prefill import Plan
    offsets=getattr(route_plan,'combine_offsets',None)
    if (not isinstance(route_plan,Plan) or route_plan.values is not values or route_plan.order is not order
            or values.flags.writeable or order.flags.writeable or not values.flags.c_contiguous
            or not order.flags.c_contiguous or not isinstance(offsets,np.ndarray)
            or offsets.dtype!=np.uint32 or offsets.shape!=values.shape
            or offsets.flags.writeable or not offsets.flags.c_contiguous
            or np.any(offsets>=values.size)):
        raise ValueError('Source mapped combine frame-plan ownership/offset geometry mismatch')
    # Prove borrowed metadata encodes this exact stable per-token order without
    # constructing a second inverse/argsort. This runs once at packet creation.
    if np.any(np.bincount(offsets.ravel(),minlength=values.size)!=1):
        raise ValueError('Source mapped combine frame offsets are not a permutation')
    locations=normalized_order[offsets]
    ids=values.ravel()[locations]
    if (np.any(locations//values.shape[1]!=np.arange(len(values))[:,None])
            or np.any(ids[:,1:]<ids[:,:-1])
            or np.any((ids[:,1:]==ids[:,:-1])&(locations[:,1:]<locations[:,:-1]))):
        raise ValueError('Source mapped combine frame offsets do not match stable routes')
    return offsets


def mapped_plan(values, order, part_experts, *, chunk_parts=MAX_PARTS, route_plan=None):
    """Map bucketed expert segments back to the global stable route order.

    Each physical part lists (expert_id, row_count) segments in its own row
    order. A part may contain nonconsecutive IDs. Chunk boundaries follow
    ascending expert IDs, never physical part order, so carrying F32 totals
    across chunks preserves the ordinary addition sequence exactly.
    """
    normalized_order,sorted_routes=_validate_routes(values,order,chunk_parts)
    descriptors = tuple(tuple(group) for group in part_experts)
    owners, starts, counts, part_rows = {}, {}, {}, []
    for part, group in enumerate(descriptors):
        if not group:
            raise ValueError('Source mapped combine requires nonempty part descriptors')
        offset = 0
        for pair in group:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('Source mapped combine requires (expert,rowcount) segments')
            expert, count = pair
            try:
                if isinstance(expert, bool) or isinstance(count, bool): raise TypeError
                expert, count = operator.index(expert), operator.index(count)
            except TypeError as error:
                raise ValueError('Source mapped combine IDs/counts must be integers') from error
            if not 0 <= expert < 384 or not 1 <= count <= MAX_LOCAL_ROWS or expert in owners:
                raise ValueError('Source mapped combine duplicate ID or invalid segment')
            owners[expert], starts[expert], counts[expert] = part, offset, count
            offset += count
        if offset > MAX_LOCAL_ROWS:
            raise ValueError('Source mapped combine physical part row bound exceeded')
        part_rows.append(offset)
    if not part_rows or sum(part_rows)!=values.size:
        raise ValueError('Source mapped combine part row extent mismatch')
    observed = np.bincount(values.ravel(), minlength=384)
    if set(np.flatnonzero(observed)) != set(owners) or any(observed[e] != n for e,n in counts.items()):
        raise ValueError('Source mapped combine expert row counts differ from routes')
    offsets = _mapped_offsets(values,order,normalized_order,route_plan)
    ids=np.flatnonzero(observed)
    expert_owner=np.full(384,-1,np.intp);expert_starts=np.zeros(384,np.intp)
    for expert in ids:
        expert_owner[expert]=owners[expert];expert_starts[expert]=starts[expert]
    global_starts=np.cumsum(observed,dtype=np.intp)-observed
    global_owner=expert_owner[sorted_routes]
    global_local=np.arange(values.size,dtype=np.intp)-global_starts[sorted_routes]+expert_starts[sorted_routes]
    owner, local = global_owner[offsets], global_local[offsets]
    sorted_ids=sorted_routes[offsets]
    chunks, begin = [], 0
    while begin < len(ids):
        stop, bound = begin, set()
        while stop < len(ids):
            part=owners[ids[stop]]
            if part not in bound and len(bound)==chunk_parts:break
            bound.add(part)
            stop += 1
        selected_parts = tuple(sorted(bound))
        selected = (sorted_ids >= ids[begin]) & (sorted_ids <= ids[stop-1])
        slots=np.zeros(len(descriptors),np.uint32)
        slots[list(selected_parts)]=np.arange(1,len(selected_parts)+1,dtype=np.uint32)
        refs=np.where(selected,(slots[owner]<<np.uint32(16))|local.astype(np.uint32),np.uint32(0))
        refs.flags.writeable=False
        first = begin == 0
        chunks.append(dict(parts=selected_parts, first=first, refs=refs,
            first_expert=int(ids[begin]), last_expert=int(ids[stop-1]),
            input_bindings=len(selected_parts)+1+(0 if first else 1),
            total_buffer_bindings=len(selected_parts)+2+(0 if first else 1)))
        begin = stop
    return chunks


def combine_mapped(parts, values, order, part_experts, *, chunk_parts=MAX_PARTS):
    """Ordered F32 combine of immutable, potentially row-bucketed parts."""
    import mlx.core as mx
    parts, descriptors = tuple(parts), tuple(tuple(g) for g in part_experts)
    if (not parts or len(parts) != len(descriptors)
            or any(p.ndim != 2 or p.dtype != mx.float32 for p in parts)
            or len({p.shape[1] for p in parts}) != 1 or parts[0].shape[1] <= 0):
        raise ValueError('Source mapped combine part geometry/dtype mismatch')
    chunks = mapped_plan(values, order, descriptors, chunk_parts=chunk_parts)
    if any(p.shape[0] != sum(n for _,n in group) for p,group in zip(parts,descriptors)):
        raise ValueError('Source mapped combine physical part extent mismatch')
    rows, topk = values.shape
    width, result = parts[0].shape[1], None
    for chunk in chunks:
        inputs = [parts[i] for i in chunk['parts']]
        if not chunk['first']: inputs.append(result)
        inputs.append(mx.array(chunk['refs'], mx.uint32))
        result = _kernel(len(chunk['parts']), chunk['first'])(
            inputs=inputs, template=[('ROWS',rows),('WIDTH',width),('TOPK',topk)],
            grid=(rows*width,1,1), threadgroup=(256,1,1),
            output_shapes=[(rows,width)], output_dtypes=[mx.float32])[0]
    return result


@dataclass(frozen=True,eq=False)
class PreparedMappedChunk:
    parts: tuple
    first: bool
    refs: np.ndarray
    device_refs: object


@dataclass(frozen=True,eq=False)
class PreparedMappedCombine:
    """Frame-only packet; never globally cached or an expert-backing owner.

    Route snapshots/ref maps are readonly CPU metadata. Device metadata is
    uploaded once at construction. Weak identities bind exact immutable output
    parts without extending their lifetime; each consumer builds a fresh carry.
    """
    values: np.ndarray
    order: np.ndarray
    chunks: tuple
    part_refs: tuple
    part_shapes: tuple
    rows: int
    topk: int
    width: int
    metadata_bytes: int
    borrowed_route_plan: bool
    _seal: object


def prepare_mapped(parts,values,order,part_experts,*,chunk_parts=MAX_PARTS,route_plan=None):
    """Build one mapped packet inside the owning frame's preparation stage.

    An existing authoritative plan_jobs(reuse=True) may supply its exact owned
    readonly values/order/combine_offsets. Otherwise mutable route arrays are
    snapshotted once. No setup cost should be excluded from a whole-stage clock.
    """
    import mlx.core as mx
    parts,descriptors=tuple(parts),tuple(tuple(g)for g in part_experts)
    if (not parts or len(parts)!=len(descriptors)
            or any(p.ndim!=2 or p.dtype!=mx.float32 for p in parts)
            or len({p.shape[1]for p in parts})!=1 or parts[0].shape[1]<=0):
        raise ValueError('Source prepared mapped combine part geometry/dtype mismatch')
    if (not isinstance(values,np.ndarray)or values.ndim!=2 or values.dtype!=np.uint32
            or not 1<=values.shape[0]<=1024 or not 1<=values.shape[1]<=6
            or not isinstance(order,np.ndarray)or order.ndim!=1 or order.size!=values.size
            or order.dtype.kind not in 'iu'):
        raise ValueError('Source prepared mapped combine route geometry mismatch')
    if route_plan is None:
        if values.flags.writeable or not values.flags.c_contiguous:
            values=np.array(values,copy=True,order='C');values.flags.writeable=False
        if order.flags.writeable or not order.flags.c_contiguous:
            order=np.array(order,copy=True,order='C');order.flags.writeable=False
    chunks=mapped_plan(values,order,descriptors,chunk_parts=chunk_parts,route_plan=route_plan)
    if any(p.shape[0]!=sum(n for _,n in group)for p,group in zip(parts,descriptors)):
        raise ValueError('Source prepared mapped combine physical part extent mismatch')
    try:identities=tuple(weakref.ref(p)for p in parts)
    except TypeError as error:
        raise ValueError('Source prepared mapped combine requires weak-referenceable immutable parts')from error
    prepared=tuple(PreparedMappedChunk(c['parts'],c['first'],c['refs'],mx.array(c['refs'],mx.uint32))for c in chunks)
    return PreparedMappedCombine(values,order,prepared,identities,tuple(p.shape for p in parts),
        values.shape[0],values.shape[1],parts[0].shape[1],sum(c['refs'].nbytes for c in chunks),
        route_plan is not None,_PACKET_SEAL)


def combine_mapped_prepared(parts,packet):
    """Consume exact current part objects with prevalidated immutable refs.

    No route planning, metadata upload, concatenate, readback or synchronization
    occurs here. Parts and leases remain owned by the caller through completion.
    """
    import mlx.core as mx
    parts=tuple(parts)
    if (not isinstance(packet,PreparedMappedCombine)or packet._seal is not _PACKET_SEAL
            or len(parts)!=len(packet.part_refs)or packet.values.flags.writeable or packet.order.flags.writeable
            or any(ref()is not p or p.shape!=shape or p.ndim!=2 or p.dtype!=mx.float32
                   for p,ref,shape in zip(parts,packet.part_refs,packet.part_shapes))):
        raise ValueError('Source prepared mapped combine frame/part identity mismatch')
    result=None
    for chunk in packet.chunks:
        inputs=[parts[i]for i in chunk.parts]
        if not chunk.first:inputs.append(result)
        inputs.append(chunk.device_refs)
        result=_kernel(len(chunk.parts),chunk.first)(inputs=inputs,
            template=[('ROWS',packet.rows),('WIDTH',packet.width),('TOPK',packet.topk)],
            grid=(packet.rows*packet.width,1,1),threadgroup=(256,1,1),
            output_shapes=[(packet.rows,packet.width)],output_dtypes=[mx.float32])[0]
    return result
