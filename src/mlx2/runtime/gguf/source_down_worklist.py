# SPDX-License-Identifier: MIT
"""Bounded direct-owned-output Source PP DOWN candidate, never a default.

Dot/reduction programs come from the pinned old PP helpers; only the final
device store changes. Independent expert objects remain leased by the caller.
Provenance: provenance/deepseek-v41-source-down-worklist.json.
"""
from functools import lru_cache
import operator
import numpy as np

HIDDEN, INTERMEDIATE, MAX_EXPERTS = 5120, 2304, 15


@lru_cache(maxsize=2)
def _canonical_layout(kind):
    from ...adapters.deepseek_v41_source_buffers import canonical_layout
    return canonical_layout(kind)


@lru_cache(maxsize=1)
def _native_q1_layout():
    # Native bank metadata pinned in down-worklist-review/layout-metadata.json;
    # offsets address the payload, excluding its904-byte file header.
    from ...adapters.deepseek_v41_source_buffers import ExpertLayout,TensorSlice
    names=('down.qweight','gate.qweight','up.qweight','down.biases','down.scales',
           'gate.biases','gate.scales','up.biases','up.scales')
    canonical=_canonical_layout('q1');cursor=0;tensors=[]
    for name in names:
        t=canonical.tensor(name);tensors.append(TensorSlice(name,cursor,t.dtype,t.shape));cursor+=t.nbytes
    return ExpertLayout('q1',tuple(tensors))


def native_q1_layout():
    """Immutable CPU descriptor of the pinned native bank, never a payload copy."""
    return _native_q1_layout()


def _physical_layout(kind,variant):
    if variant=='canonical':return _canonical_layout(kind)
    if kind=='q1'and variant=='native_q1_v1':return _native_q1_layout()
    raise ValueError('Source DOWN worklist physical layout variant unsupported')


@lru_cache(maxsize=32)
def _layout_signature(layout):
    # Parser inventory order is metadata only; native borrow_q1_packed visits
    # schema order, whereas the physical payload is down-qweight first.
    return tuple(sorted((t.name,t.offset,t.dtype,t.shape)for t in layout.tensors))


def _layout_variant(layout):
    signature=_layout_signature(layout)
    if signature==_layout_signature(_canonical_layout(layout.kind)):return 'canonical'
    if layout.kind=='q1'and signature==_layout_signature(_native_q1_layout()):return 'native_q1_v1'
    raise ValueError('Source DOWN worklist physical payload layout unsupported')


def plan(kind, row_counts, *, traversal="flat"):
    if kind not in ("original", "q1"):
        raise ValueError("Source DOWN worklist unknown expert kind")
    if type(traversal)is not str or traversal not in ("flat","stock") or (traversal=="stock"and kind!="original"):
        raise ValueError("Source DOWN worklist traversal unsupported")
    counts = []
    for value in row_counts:
        try:
            if isinstance(value, bool):raise TypeError
            count = operator.index(value)
        except TypeError as error:
            raise ValueError("Source DOWN worklist row counts must be integers") from error
        if not 1 <= count <= (12 if kind == "original" else 3):
            raise ValueError("Source DOWN worklist local row band unsupported")
        counts.append(count)
    if not 1 <= len(counts) <= MAX_EXPERTS:
        raise ValueError("Source DOWN worklist requires 1..15 experts")
    if kind == "original" and len(set(counts)) != 1:
        raise ValueError("Source original DOWN requires homogeneous local rows")
    local = counts[0] if kind == "original" else 1
    vecs = 1 if local == 1 else (local + (local + 4)//5 - 1)//((local + 4)//5)
    groups = (local + vecs - 1)//vecs
    tile = 4 if kind == "original" and local > 1 else 8
    offsets = tuple(np.cumsum((0, *counts[:-1]), dtype=np.int64).tolist())
    owner_slots = (np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
                   if kind == "q1" else None)
    if owner_slots is not None:owner_slots.flags.writeable = False
    grid=((groups*64,HIDDEN//tile,len(counts)) if traversal=="stock" else
          (HIDDEN//tile*64,sum(counts)if kind=="q1"else len(counts)*groups,1))
    return dict(kind=kind,traversal=traversal,row_counts=tuple(counts),row_offsets=offsets,rows=sum(counts),
        experts=len(counts),local_rows=local,vecs=vecs,vector_groups=groups,tile=tile,
        owner_slots=owner_slots,grid=grid,
        threadgroup=(64,1,1),scratch_bytes=0,output_bytes=sum(counts)*HIDDEN*4,
        metadata_bytes=0 if owner_slots is None else owner_slots.nbytes,
        input_bindings=1+len(counts)+int(kind == "q1"))


def _replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError("Pinned Source DOWN store anchor changed: " + old)
    return text.replace(old, new)


def _owned_header(kind, local_rows, return_raw):
    """Keep all dot/reduction statements; change only address and final stores."""
    if kind == "q1":
        from .source_down_fusion import _q1_header
        text = _q1_header()
        text = _replace(text,"mlx2_source_q1_down_shared(","mlx2_source_q1_down_owned(")
        text = _replace(text,"threadgroup float* out,", "device float* out," +
                        (" device float* raw," if return_raw else ""))
        old = "out[row] =\n                static_cast<float>(accumulators[row]);"
        new = (("raw[output_row] = static_cast<float>(accumulators[row]);\n            " if return_raw else "") +
               "out[output_row] = float(bfloat(static_cast<float>(accumulators[row])));")
        return "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"+_replace(text,old,new)
    if local_rows == 1:
        from .source_down_fusion import _original_header
        text = _original_header()
        text = _replace(text,"fp_qmv_to_shared_impl(","fp_qmv_owned_impl(")
        text = _replace(text,"    threadgroup T* y,", "    device T* y," +
                        ("\n    device T* raw," if return_raw else ""))
        if text.count("    y += 0;") != 2:
            raise RuntimeError("Pinned original DOWN pointer anchors changed")
        for row in ("out_row", "used_out_row"):
            address = "tid.x * out_vec_size + " + row
            text = text.replace("    y += 0;", "    y += " + address + ";" +
                                ("\n    raw += " + address + ";" if return_raw else ""),1)
        for value in ("result[row] * inv_scale_enc", "result[row]"):
            old = "y[row] = static_cast<T>(" + value + ");"
            new = (("raw[row] = static_cast<T>(" + value + ");\n          " if return_raw else "") +
                   "y[row] = static_cast<T>(bfloat(static_cast<T>(" + value + ")));")
            text = _replace(text,old,new,2)
        return text
    from .source_down_prefill import _wide_header
    text = _wide_header(local_rows)
    text = _replace(text,"fp_qmv_wide_to_shared_impl(","fp_qmv_wide_owned_impl(")
    text = _replace(text,"    threadgroup T* y,", "    device T* y," +
                    ("\n    device T* raw," if return_raw else ""))
    old = "y[v * 4 + out_row - tid.y * 4] = static_cast<T>(result[v]);"
    address = "(vec0 + v) * out_vec_size + out_row"
    new = (("raw[" + address + "] = static_cast<T>(result[v]);\n        " if return_raw else "") +
           "y[" + address + "] = static_cast<T>(bfloat(static_cast<T>(result[v])));")
    return _replace(text,old,new)


def kernel_spec(kind, local_rows, expert_count, *, return_raw=False, layout_variant='canonical', traversal='flat'):
    if type(return_raw) is not bool:
        raise ValueError("Source DOWN worklist raw flag must be boolean")
    if type(local_rows)is not int or type(expert_count)is not int or not 1<=expert_count<=MAX_EXPERTS:
        raise ValueError("Source DOWN worklist compiled geometry must be bounded integers")
    if kind == "q1" and local_rows != 1:
        raise ValueError("Source Q1 DOWN has one row-independent shader")
    d = plan(kind,[local_rows]*expert_count,traversal=traversal)
    layout = _physical_layout(kind,layout_variant)
    qoff, soff = (layout.tensor("down."+field).offset for field in ("qweight","scales"))
    names = ["middles"] + (["owner_slots"] if kind == "q1" else []) + [f"expert{i}" for i in range(expert_count)]
    source = ("const uint tile=threadgroup_position_in_grid.y, slot=threadgroup_position_in_grid.z, vector_group=threadgroup_position_in_grid.x;\n"
              +f"const uint row_base=slot*{local_rows};\n"
              +"const uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;\n" if traversal=='stock' else """
const uint tile=threadgroup_position_in_grid.x, job=threadgroup_position_in_grid.y;
const uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
""")
    if kind == "q1":
        source += "const uint slot=owner_slots[job], row_base=job, vector_group=0;\n"
    elif traversal=="flat":
        source += f"const uint slot=job/{d['vector_groups']}, vector_group=job%{d['vector_groups']}, row_base=slot*{local_rows};\n"
    source += "const device uchar* backing=nullptr;\nswitch(slot) {\n" + "\n".join(
        f"case {i}:backing=expert{i};break;" for i in range(expert_count)) + "\n}\n"
    source += f"const device uint32_t* q=(const device uint32_t*)(backing+{qoff});\n"
    raw = ",raw+row_base*5120" if return_raw else ""
    if kind == "q1":
        boff = layout.tensor("down.biases").offset
        source += f"const device half* scales=(const device half*)(backing+{soff});\nconst device half* biases=(const device half*)(backing+{boff});\n"
        source += f"mlx2_source_q1_down_owned(middles+row_base*2304,q,scales,biases,output+row_base*5120{raw},tile,sg,lane);\n"
    else:
        source += f"const device uint8_t* scales=backing+{soff};\n"
        if local_rows == 1:
            source += f"mlx2_source_down_stock::fp_qmv_owned_impl<float,32,4,false>(q,scales,nullptr,middles+row_base*2304,output+row_base*5120{raw},mlx2_source_down_stock::K,mlx2_source_down_stock::N,uint3(0,tile,0),sg,lane);\n"
        else:
            source += f"mlx2_source_stock_qmv::fp_qmv_wide_owned_impl<float,32,4,{d['vecs']},16>(q,scales,middles+row_base*2304,output+row_base*5120{raw},mlx2_source_stock_qmv::K,mlx2_source_stock_qmv::N,mlx2_source_stock_qmv::M,uint3(vector_group,tile,0),sg,lane);\n"
    return dict(d,header=_owned_header(kind,local_rows,return_raw),source=source,
        input_names=names,output_names=["output"]+(["raw"]if return_raw else[]),
        total_buffer_bindings=len(names)+1+int(return_raw),layout_variant=layout_variant)


@lru_cache(maxsize=1024)
def _kernel_for_traversal(kind,local_rows,expert_count,return_raw,layout_variant,traversal):
    # At most780 distinct lazy programs:360original pertraversal+60Q1flat.
    # Keys contain primitive immutable descriptors only, never tensor references.
    import mlx.core as mx
    s=kernel_spec(kind,local_rows,expert_count,return_raw=return_raw,layout_variant=layout_variant,traversal=traversal)
    suffix='_stock'if traversal=='stock'else''
    return mx.fast.metal_kernel(name=f"mlx2_source_down_worklist_{kind}_{layout_variant}_m{local_rows}_e{expert_count}_raw{int(return_raw)}{suffix}",
        input_names=s['input_names'],output_names=s['output_names'],header=s['header'],source=s['source'],
        ensure_row_contiguous=False)


def _kernel(kind,local_rows,expert_count,return_raw,layout_variant,traversal='flat'):
    # Keep the existing default factory call contract; actual cache always keys
    # the explicit traversal value, including its default.
    return _kernel_for_traversal(kind,local_rows,expert_count,return_raw,layout_variant,traversal)


_kernel.cache_info=_kernel_for_traversal.cache_info
_kernel.cache_clear=_kernel_for_traversal.cache_clear


def compilation_source(kind,local_rows,expert_count,*,return_raw=False,layout_variant='canonical',traversal='flat'):
    """CPU-only standalone Metal source for address-space/geometry compilation."""
    s=kernel_spec(kind,local_rows,expert_count,return_raw=return_raw,layout_variant=layout_variant,traversal=traversal)
    parameters=[]
    for i,name in enumerate(s['input_names']):
        dtype='float'if name=='middles'else 'uint'if name=='owner_slots'else 'uchar'
        parameters.append(f'const device {dtype}* {name} [[buffer({i})]]')
    for i,name in enumerate(s['output_names'],len(parameters)):
        parameters.append(f'device float* {name} [[buffer({i})]]')
    parameters.extend(('uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]',
        'uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]',
        'uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]'))
    return s['header']+'\nusing namespace metal;\nkernel void source_down_worklist_offline(\n'+',\n'.join(parameters)+') {\n'+s['source']+'\n}\n'


def prepare_inputs(middles,experts,row_counts,*,traversal="flat"):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,ExpertLayout
    experts=tuple(experts)
    if not experts or not all(isinstance(e,PackedExpert) and isinstance(e.layout,ExpertLayout) for e in experts):
        raise ValueError("Source DOWN worklist requires immutable PackedExpert objects")
    kind=experts[0].layout.kind
    if any(e.layout.kind!=kind for e in experts):
        raise ValueError("Source DOWN worklist batch kinds must be homogeneous")
    d=plan(kind,row_counts,traversal=traversal)
    if len(experts)!=d['experts'] or middles.shape!=(d['rows'],INTERMEDIATE) or middles.dtype!=mx.float32:
        raise ValueError("Source DOWN worklist middle/expert segment geometry mismatch")
    variant=_layout_variant(experts[0].layout)
    canonical=_physical_layout(kind,variant)
    for expert in experts:
        if expert.has_compact_backing is not True:
            raise ValueError("Source DOWN worklist requires loader-issued compact backing capability")
        if _layout_variant(expert.layout)!=variant:
            raise ValueError("Source DOWN worklist requires homogeneous admitted physical payload offsets")
        backing=expert.backing
        if backing.ndim!=1 or backing.dtype!=mx.uint8 or backing.shape!=(canonical.nbytes,):
            raise ValueError("Source DOWN worklist backing geometry mismatch")
        for name in ('down.qweight','down.scales')+(('down.biases',)if kind=='q1'else()):
            t=canonical.tensor(name);expected={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}[t.dtype]
            if expert[name].shape!=t.shape or expert[name].dtype!=expected:
                raise ValueError("Source DOWN worklist packed typed-view mismatch")
    d['layout_variant']=variant
    return d,experts


def pp_down(middles,experts,row_counts,*,return_raw=False,traversal="flat"):
    """Packed input rows -> contiguous BF16-owned F32 contribution rows.

    Q1 metadata uploads once percall. Only bounded middle activations may be
    made contiguous; independent expert backings are never normalized/copied.
    Caller owns GPU admission and backing leases through output completion.
    """
    import mlx.core as mx
    if type(return_raw)is not bool:raise ValueError("Source DOWN worklist raw flag must be boolean")
    d,experts=prepare_inputs(middles,experts,row_counts,traversal=traversal)
    from .source_mxfp4_pair import _device_supported
    if not _device_supported():
        raise ValueError("Source DOWN worklist requires a caller-owned M5 Max GPU window")
    inputs=[mx.contiguous(middles)]
    if d['kind']=='q1':inputs.append(mx.array(d['owner_slots'],mx.uint32))
    inputs.extend(e.backing for e in experts)
    args=(d['kind'],d['local_rows'],d['experts'],return_raw,d['layout_variant'])
    kernel=(_kernel(*args)if traversal=='flat'else _kernel(*args,traversal))
    result=kernel(inputs=inputs,
        grid=d['grid'],threadgroup=d['threadgroup'],output_shapes=[(d['rows'],HIDDEN)]*(2 if return_raw else 1),
        output_dtypes=[mx.float32]*(2 if return_raw else 1))
    return tuple(result)if return_raw else result[0]
