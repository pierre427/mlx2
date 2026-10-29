# SPDX-License-Identifier: MIT
"""Explicit mapped Source gate/up and FP8 middle candidate.

The original M1 dot uses the pinned installed MXFP4 helpers. Q1 preserves its
existing lane/K/addition law. Every BF16 boundary is explicit; only the final
F32 down-input is stored globally. This module never selects a serving policy.
See provenance/deepseek-v41-source-gate-up-fusion.json.
"""
from functools import lru_cache

WIDTH = 5120
INTERMEDIATE = 2304
CHANNELS_PER_TILE = 32
THREADS = 256
EXP_SPELLINGS = ("metal::exp", "metal::precise::exp")


def descriptor(kind, rows):
    """CPU geometry; original M1 fast and M2–12 wide keep separate dot laws."""
    if kind not in ("original", "q1"):
        raise ValueError("Source gate/up fusion unknown format")
    if type(rows) is not int or not 1 <= rows <= 1024:
        raise ValueError("Source gate/up fusion row count outside1..1024")
    if kind == "original" and rows > 12:
        raise ValueError("Source gate/up fusion original row band exceeds12")
    vecs=1 if kind != "original" or rows==1 else (rows+(rows+4)//5-1)//((rows+4)//5)
    groups=(rows+vecs-1)//vecs
    return {"kind":kind,"rows":rows,"grid":(INTERMEDIATE//CHANNELS_PER_TILE*THREADS,groups,1),
            "threadgroup":(THREADS,1,1),"output_shape":(rows,INTERMEDIATE),
            "channels_per_tile":CHANNELS_PER_TILE,"simdgroups":8,
            "vecs":vecs,"vector_groups":groups,"output_phases":2 if kind=="original" and rows>1 else 1,
            "threadgroup_middle_bytes":vecs*32*4,"raw_gate_up_global_bytes":0}


def _kind(expert):
    return "q1" if "gate.biases" in expert else "original"


def _geometry(x, rows, scores, expert):
    import mlx.core as mx
    if (x.ndim != 2 or x.shape[1] != WIDTH or not 1 <= x.shape[0] <= 1024 or x.dtype != mx.float32
            or rows.ndim != 1 or not rows.shape[0] or rows.dtype != mx.uint32
            or scores.shape != rows.shape or scores.dtype != mx.float32):
        raise ValueError("Source gate/up fusion input geometry/dtype mismatch")
    kind=_kind(expert);descriptor(kind,rows.shape[0])
    for projection in ("gate","up"):
        q,s=expert.get(projection+".qweight"),expert.get(projection+".scales")
        if kind == "q1":
            b=expert.get(projection+".biases")
            if (q is None or s is None or b is None or q.shape != (INTERMEDIATE,WIDTH//32)
                    or q.dtype != mx.uint32 or s.shape != (INTERMEDIATE,WIDTH//64)
                    or s.dtype != mx.float16 or b.shape != s.shape or b.dtype != mx.float16):
                raise ValueError("Source gate/up fusion Q1 tensor geometry/dtype mismatch")
        elif (q is None or s is None or q.shape != (INTERMEDIATE,WIDTH//8)
                or q.dtype != mx.uint32 or s.shape != (INTERMEDIATE,WIDTH//32)
                or s.dtype != mx.uint8 or projection+".biases" in expert):
            raise ValueError("Source gate/up fusion original tensor geometry/dtype mismatch")
    return kind


def supports(x, rows, scores, expert):
    """Structural gate only; row indices are trusted provider-generated maps."""
    try:
        _geometry(x,rows,scores,expert)
    except (ValueError,KeyError,AttributeError,TypeError):
        return False
    return True


_EPILOGUE = r"""
inline float source_gu_bf16(float v) {
    const uint bits=as_type<uint>(v);
    return as_type<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}
inline float source_gu_weighted(float g, float u, float score) {
    #pragma clang fp contract(off)
    g=min(source_gu_bf16(g),10.0f);
    u=clamp(source_gu_bf16(u),-10.0f,10.0f);
    const float t=1.0f/(1.0f+__EXP__(abs(g)));
    const float sigmoid=g<0.0f ? t : 1.0f-t;
    const float silu=g*sigmoid;
    const float active=silu*u;
    return source_gu_bf16(active*score);
}
inline float source_gu_e4m3(float value) {
    const float a=min(abs(value),448.0f);
    const int exponent=max(int((as_type<uint>(a)>>23)&255u)-127-3,-9);
    const float step=as_type<float>(uint(exponent+127)<<23);
    return copysign(min(rint(a/step)*step,448.0f),value);
}
inline float source_gu_qat(float value, float maximum, float inverse) {
    #pragma clang fp contract(off)
    const uint bits=as_type<uint>(max(maximum,1e-4f)*inverse);
    const uint exponent=((bits>>23)&255u)+uint((bits&0x7fffffu)!=0u);
    const float scale=as_type<float>(exponent<<23);
    return source_gu_e4m3(clamp(value/scale,-448.0f,448.0f))*scale;
}
"""

_COMMON = r"""
const uint lane=thread_index_in_simdgroup;
const uint sg=simdgroup_index_in_threadgroup;
const uint tile=threadgroup_position_in_grid.x;
const uint selected_row=threadgroup_position_in_grid.y;
const uint input_row=rows[selected_row];
const uint out_row=tile*32+sg*4;
threadgroup float weighted[32];
float gate_result[4]={0.0f},up_result[4]={0.0f};
"""

_ORIGINAL = r"""
using namespace mlx2_source_stock_qmv;
const device uint8_t* gw=(const device uint8_t*)gate_q+out_row*2560+lane*8;
const device uint8_t* uw=(const device uint8_t*)up_q+out_row*2560+lane*8;
const device uint8_t* gs=gate_s+out_row*160+lane/2;
const device uint8_t* us=up_s+out_row*160+lane/2;
const device float* xp=x+input_row*5120+lane*16;
float x_thread[16];
for(int k=0;k<5120;k+=512) {
    load_vector<float,float,16>(xp,x_thread);
    for(int r=0;r<4;r++) {
        const float scale_g=dequantize_scale<float,32>(gs[r*160]);
        gate_result[r]+=qdot<float,16,4>(gw+r*2560,x_thread,scale_g);
        const float scale_u=dequantize_scale<float,32>(us[r*160]);
        up_result[r]+=qdot<float,16,4>(uw+r*2560,x_thread,scale_u);
    }
    gw+=256;uw+=256;gs+=16;us+=16;xp+=512;
}
"""

_Q1 = r"""
for(uint block_start=lane*16;block_start<5120;block_start+=512) {
    float x_thread[16];
    float total_sum=0.0f;
    #pragma clang loop unroll(full)
    for(uint i=0;i<16;i++) {
        const float value=x[input_row*5120+block_start+i];
        x_thread[i]=value;
        total_sum+=value;
    }
    const uint group=block_start/64;
    const uint packed_column=block_start>>5;
    const uint packed_shift=block_start&31;
    for(uint r=0;r<4;r++) {
        const uint output_row=out_row+r;
        const ushort gp=ushort(gate_q[output_row*160+packed_column]>>packed_shift);
        const ushort up=ushort(up_q[output_row*160+packed_column]>>packed_shift);
        float selected_g=0.0f,selected_u=0.0f;
        #pragma clang loop unroll(full)
        for(uint i=0;i<16;i++) {
            selected_g+=(gp&(ushort(1)<<i)) ? x_thread[i] : 0.0f;
            selected_u+=(up&(ushort(1)<<i)) ? x_thread[i] : 0.0f;
        }
        const uint pi=output_row*80+group;
        gate_result[r]+=selected_g*float(gate_s[pi]);
        gate_result[r]+=total_sum*float(gate_b[pi]);
        up_result[r]+=selected_u*float(up_s[pi]);
        up_result[r]+=total_sum*float(up_b[pi]);
    }
}
"""

_STORE = r"""
for(uint r=0;r<4;r++) {
    gate_result[r]=simd_sum(gate_result[r]);
    up_result[r]=simd_sum(up_result[r]);
    if(lane==0) weighted[sg*4+r]=source_gu_weighted(gate_result[r],up_result[r],scores[selected_row]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {
    const float value=weighted[lane];
    float maximum=abs(value);
    for(uint step=16;step>0;step/=2) maximum=max(maximum,simd_shuffle_xor(maximum,step));
    out[selected_row*2304+tile*32+lane]=source_gu_qat(value,maximum,inverse[0]);
}
"""


_WIDE_ORIGINAL = r"""
using namespace mlx2_source_stock_qmv;
constexpr int GU_ROWS=__ROWS__,GU_VECS=__VECS__;
const uint lane=thread_index_in_simdgroup;
const uint sg=simdgroup_index_in_threadgroup;
const uint tile=threadgroup_position_in_grid.x;
const uint k_lane=lane%16,sg_row=lane/16;
const int vec0=threadgroup_position_in_grid.y*GU_VECS;
threadgroup float weighted[GU_VECS*32];
const device float* xv[GU_VECS];
for(int v=0;v<GU_VECS;v++) xv[v]=x+rows[min(vec0+v,GU_ROWS-1)]*5120;
for(int phase=0;phase<2;phase++) {
    const uint out_row=tile*32+phase*16+sg*2+sg_row;
    const device uint8_t* gwrow=(const device uint8_t*)gate_q+out_row*2560;
    const device uint8_t* uwrow=(const device uint8_t*)up_q+out_row*2560;
    const device uint8_t* gsrow=gate_s+out_row*160;
    const device uint8_t* usrow=up_s+out_row*160;
    float gate_result[GU_VECS]={0.0f},up_result[GU_VECS]={0.0f};
    for(int g=k_lane;g<160;g+=16) {
        const int k0=g*32;
        const float scale_g=dequantize_scale<float,32>(gsrow[g]);
        const float scale_u=dequantize_scale<float,32>(usrow[g]);
        const device uint16_t* gw=(const device uint16_t*)(gwrow+k0/2);
        const device uint16_t* uw=(const device uint16_t*)(uwrow+k0/2);
        float4 gate_w4[8],up_w4[8];
        #pragma unroll
        for(int j=0;j<8;j++) {
            gate_w4[j]=float4(Dequantize<4>{}(gw[j]),Dequantize<4>{}(gw[j]>>4),
                Dequantize<4>{}(gw[j]>>8),Dequantize<4>{}(gw[j]>>12));
            up_w4[j]=float4(Dequantize<4>{}(uw[j]),Dequantize<4>{}(uw[j]>>4),
                Dequantize<4>{}(uw[j]>>8),Dequantize<4>{}(uw[j]>>12));
        }
        #pragma unroll
        for(int v=0;v<GU_VECS;v++) {
            const device float4* xv4=(const device float4*)(xv[v]+k0);
            float gate_acc=0.0f,up_acc=0.0f;
            #pragma unroll
            for(int j=0;j<8;j++) {
                const float4 activation=float4(xv4[j]);
                gate_acc+=dot(gate_w4[j],activation);
                up_acc+=dot(up_w4[j],activation);
            }
            gate_result[v]+=scale_g*gate_acc;
            up_result[v]+=scale_u*up_acc;
        }
    }
    for(int v=0;v<GU_VECS;v++) {
        gate_result[v]+=simd_shuffle_down(gate_result[v],8);
        gate_result[v]+=simd_shuffle_down(gate_result[v],4);
        gate_result[v]+=simd_shuffle_down(gate_result[v],2);
        gate_result[v]+=simd_shuffle_down(gate_result[v],1);
        up_result[v]+=simd_shuffle_down(up_result[v],8);
        up_result[v]+=simd_shuffle_down(up_result[v],4);
        up_result[v]+=simd_shuffle_down(up_result[v],2);
        up_result[v]+=simd_shuffle_down(up_result[v],1);
    }
    if(k_lane==0) {
        for(int v=0;v<GU_VECS;v++) {
            if(vec0+v<GU_ROWS) {
                // SOURCE_GU_WIDE_RAW_STORE
                weighted[v*32+phase*16+sg*2+sg_row]=source_gu_weighted(
                    gate_result[v],up_result[v],scores[vec0+v]);
            }
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg<GU_VECS && vec0+int(sg)<GU_ROWS) {
    const float value=weighted[sg*32+lane];
    float maximum=abs(value);
    for(uint step=16;step>0;step/=2) maximum=max(maximum,simd_shuffle_xor(maximum,step));
    out[(vec0+sg)*2304+tile*32+lane]=source_gu_qat(value,maximum,inverse[0]);
}
"""


def _store_source(return_projections):
    if not return_projections:
        return _STORE
    return _STORE.replace("    if(lane==0) weighted",r"""
    if(lane==0) {
        raw_gate[selected_row*2304+out_row+r]=gate_result[r];
        raw_up[selected_row*2304+out_row+r]=up_result[r];
    }
    if(lane==0) weighted""")


def kernel_spec(kind, *, exp_spelling="metal::exp",selected_rows=1,return_projections=False):
    """CPU source preparation only; compilation occurs on an owned GPU later."""
    plan=descriptor(kind,selected_rows)
    if type(return_projections) is not bool:
        raise ValueError("Source gate/up fusion diagnostic flag must be boolean")
    if exp_spelling not in EXP_SPELLINGS:
        raise ValueError("Source gate/up fusion unknown sigmoid spelling")
    header=""
    if kind == "original":
        from .source_mxfp4_pair import _source_header
        header=_source_header(1)
    # Dot loops retain the stock compiler contraction law. Only the
    # pointwise/QAT functions locally disable contraction at their boundaries.
    header+=_EPILOGUE.replace("__EXP__",exp_spelling)
    names=["x","rows","scores","gate_q","gate_s","up_q","up_s","inverse"]
    if kind == "q1":names=["x","rows","scores","gate_q","gate_s","gate_b","up_q","up_s","up_b","inverse"]
    if kind=="original" and selected_rows>1:
        source=_WIDE_ORIGINAL.replace("__ROWS__",str(selected_rows)).replace("__VECS__",str(plan['vecs']))
        raw=""
        if return_projections:
            raw="raw_gate[(vec0+v)*2304+out_row]=gate_result[v];\nraw_up[(vec0+v)*2304+out_row]=up_result[v];"
        source=source.replace("// SOURCE_GU_WIDE_RAW_STORE",raw)
    else:
        source=_COMMON+(_Q1 if kind=="q1" else _ORIGINAL)+_store_source(return_projections)
    return {"header":header,"source":source,
            "input_names":names,"output_names":["out"]+(["raw_gate","raw_up"] if return_projections else [])}


@lru_cache(maxsize=32)
def _kernel(kind,exp_spelling,selected_rows=1,return_projections=False):
    import mlx.core as mx
    spec=kernel_spec(kind,exp_spelling=exp_spelling,selected_rows=selected_rows,return_projections=return_projections)
    suffix=(f"_wide_m{selected_rows}" if kind=="original" and selected_rows>1 else "")+("_raw" if return_projections else "")
    return mx.fast.metal_kernel(name=f"mlx2_source_gu_middle_{kind}_{'precise' if 'precise' in exp_spelling else 'fast'}{suffix}_v1",
        ensure_row_contiguous=True,**spec)


def middle(x,rows,scores,expert,*,exp_spelling="metal::exp",return_projections=False):
    """Mapped hoisted-QAT input → only the rounded/QAT F32 down-input.

    ``rows`` is the trusted provider-created token-row map, and ``scores``
    contains its matching route occurrences. No data readback is introduced.
    The caller supplies the served sigmoid spelling selected by its gate.
    """
    import mlx.core as mx
    kind=_geometry(x,rows,scores,expert);plan=descriptor(kind,rows.shape[0])
    if exp_spelling not in EXP_SPELLINGS:
        raise ValueError("Source gate/up fusion unknown sigmoid spelling")
    args=[x,rows,scores]
    fields=(".qweight",".scales",".biases") if kind=="q1" else (".qweight",".scales")
    args.extend(expert[p+field] for p in ("gate","up") for field in fields)
    args.append(mx.array([1/448],mx.float32))
    if type(return_projections) is not bool:
        raise ValueError("Source gate/up fusion diagnostic flag must be boolean")
    count=3 if return_projections else 1
    selected_rows=rows.shape[0] if kind=="original" else 1
    result=_kernel(kind,exp_spelling,selected_rows,return_projections)(inputs=args,grid=plan['grid'],threadgroup=plan['threadgroup'],
        output_shapes=[plan['output_shape']]*count,output_dtypes=[mx.float32]*count)
    return tuple(result) if return_projections else result[0]


def batch_descriptor(count):
    """One token with independent packed expert planes; no weight copies."""
    if type(count) is not int or not 1 <= count <= 6:
        raise ValueError("Source batched gate/up requires1..6 experts")
    return {"grid":(INTERMEDIATE//CHANNELS_PER_TILE*THREADS,1,count),
            "threadgroup":(THREADS,1,1),"output_shape":(count,INTERMEDIATE),
            "logical_input_bindings":count+4,"expert_backing_bindings":count,
            "threadgroup_middle_bytes":128,"raw_gate_up_global_bytes":0}


@lru_cache(maxsize=32)
def _batch_layout(layout):
    """Validate immutable CPU descriptors once; retain no tensor or backing."""
    from ...adapters.deepseek_v41_source_buffers import ExpertLayout
    ExpertLayout(layout.kind,layout.tensors)
    fields=(".qweight",".scales",".biases") if layout.kind=="q1" else (".qweight",".scales")
    return (layout.kind,)+tuple(layout.tensor(p+field).offset
        for p in ("gate","up") for field in fields)


def _batch_geometry(x,experts,scores,slots):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,ExpertLayout
    experts=tuple(experts);batch_descriptor(len(experts))
    if (x.shape != (1,WIDTH) or x.dtype != mx.float32
            or scores.ndim != 1 or not 1 <= scores.shape[0] <= 6 or scores.dtype != mx.float32
            or slots.shape != (len(experts),) or slots.dtype != mx.uint32):
        raise ValueError("Source batched gate/up input geometry/dtype mismatch")
    key=[]
    for expert in experts:
        if not isinstance(expert,PackedExpert) or not isinstance(expert.layout,ExpertLayout):
            raise ValueError("Source batched gate/up requires validated PackedExpert objects")
        layout=expert.layout
        # Descriptor validation is cached by immutable value. Recheck the live
        # backing on every call: it is not retained in the descriptor cache.
        if (expert.backing.shape != (layout.nbytes,) or expert.backing.dtype != mx.uint8):
            raise ValueError("Source batched gate/up backing geometry/dtype mismatch")
        key.append(_batch_layout(layout))
    return tuple(key)


def batch_supports(x,experts,scores,slots):
    """Structural width-one batch gate; caller owns device and slot validity."""
    try:
        _batch_geometry(x,experts,scores,slots)
    except (ValueError,TypeError,AttributeError,KeyError,StopIteration):
        return False
    return True


def batch_kernel_spec(key,*,exp_spelling="metal::exp",return_projections=False):
    """Source generation from primitive format/offset metadata, no GPU work."""
    key=tuple(key);batch_descriptor(len(key))
    if exp_spelling not in EXP_SPELLINGS:
        raise ValueError("Source batched gate/up unknown sigmoid spelling")
    if type(return_projections) is not bool:
        raise ValueError("Source batched gate/up diagnostic flag must be boolean")
    # The production key is produced only by the full layout validator.
    # Refuse malformed manually supplied metadata rather than emitting code.
    for item in key:
        if (not isinstance(item,tuple) or not item or item[0] not in ("original","q1")
                or len(item)!=(7 if item[0]=="q1" else 5)
                or any(type(offset) is not int or offset<0 for offset in item[1:])):
            raise ValueError("Source batched gate/up layout metadata mismatch")
    header=""
    if any(item[0]=="original" for item in key):
        from .source_mxfp4_pair import _source_header
        header=_source_header(1)
    header+=_EPILOGUE.replace("__EXP__",exp_spelling)
    common=_COMMON.replace("const uint selected_row=threadgroup_position_in_grid.y;","const uint selected_row=0;")
    common=common.replace("const uint input_row=rows[selected_row];","const uint input_row=0;")
    source="const uint expert_slot=threadgroup_position_in_grid.z;\n"+common+"\nswitch(expert_slot) {\n"
    store=_store_source(return_projections).replace("scores[selected_row]","scores[slots[expert_slot]]")
    store=store.replace("selected_row*2304","expert_slot*2304")
    for index,item in enumerate(key):
        kind,*offsets=item
        if kind=="q1":
            pointers=(("uint32_t","gate_q"),("half","gate_s"),("half","gate_b"),
                      ("uint32_t","up_q"),("half","up_s"),("half","up_b"))
        else:
            pointers=(("uint32_t","gate_q"),("uint8_t","gate_s"),
                      ("uint32_t","up_q"),("uint8_t","up_s"))
        source+=f"case {index}: {{\n"
        for (dtype,name),offset in zip(pointers,offsets):
            source+=f"const device {dtype}* {name}=(const device {dtype}*)(expert{index}+{offset});\n"
        source+=(_Q1 if kind=="q1" else _ORIGINAL)+store+"\nbreak;\n}\n"
    source+="}\n"
    return {"header":header,"source":source,
            "input_names":["x","scores","slots","inverse"]+[f"expert{i}" for i in range(len(key))],
            "output_names":["out"]+(["raw_gate","raw_up"] if return_projections else [])}


@lru_cache(maxsize=64)
def _batch_kernel(key,exp_spelling,return_projections=False):
    import hashlib
    import mlx.core as mx
    spec=batch_kernel_spec(key,exp_spelling=exp_spelling,return_projections=return_projections)
    identity=hashlib.sha256(repr((key,exp_spelling)).encode()).hexdigest()[:16]+("_raw" if return_projections else "")
    return mx.fast.metal_kernel(name=f"mlx2_source_gu_batch_{identity}_v1",
        ensure_row_contiguous=True,**spec)


def batch_middle(x,experts,scores,slots,*,exp_spelling="metal::exp",return_projections=False):
    """One launch over1..6 packed original/Q1 experts for a single token.

    The provider supplies experts in its desired stable output order and a
    trusted uint32 slot map into scores. Kernels cache only format/offset
    tuples. Expert backings remain owned by the provider's completion lease.
    """
    import mlx.core as mx
    experts=tuple(experts);key=_batch_geometry(x,experts,scores,slots)
    if exp_spelling not in EXP_SPELLINGS:
        raise ValueError("Source batched gate/up unknown sigmoid spelling")
    if type(return_projections) is not bool:
        raise ValueError("Source batched gate/up diagnostic flag must be boolean")
    plan=batch_descriptor(len(experts))
    values=[x,scores,slots,mx.array([1/448],mx.float32)]+[e.backing for e in experts]
    count=3 if return_projections else 1
    kernel=_batch_kernel(key,exp_spelling,True) if return_projections else _batch_kernel(key,exp_spelling)
    result=kernel(inputs=values,grid=plan['grid'],threadgroup=plan['threadgroup'],
        output_shapes=[plan['output_shape']]*count,output_dtypes=[mx.float32]*count)
    return tuple(result) if return_projections else result[0]
