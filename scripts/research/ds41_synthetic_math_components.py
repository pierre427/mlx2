"""Synthetic Source decode math costs, no model/checkpoint startup.

run(mx,check_live=None)->JSON. The caller owns the GPU and memory admission.
No MLX/NumPy import or device work occurs on module import.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import statistics
import time

SEED=410126
REPEATS=7
DENSE=(('qa',5120,1280),('qb',1280,32768),('kv',5120,512),('wob',8192,5120))
MAX_ARRAY_BYTES=2*1024**3


def plan():
    from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config
    c=DeepSeekV41Config()
    assert (c.hidden_size,c.q_lora_rank,c.num_attention_heads*c.head_dim,c.o_groups*c.o_lora_rank)==(5120,1280,32768,8192)
    dense_bytes=sum(k*n*6 for _,k,n in DENSE) # resident F32 plus BF16 variants
    quantized_bytes=sum(k*n+n*(k//32) for _,k,n in DENSE)
    expert_bytes=6*(18_800_640+6_635_520)
    grouped_bytes=8*4096*1024*2
    grouped_cast_bytes=8*4096*1024*4
    bound=dense_bytes+quantized_bytes+expert_bytes+grouped_bytes+grouped_cast_bytes+64*1024**2
    if bound>=MAX_ARRAY_BYTES:raise ValueError('Synthetic operands exceed2GiB bound')
    return dict(schema='mlx2.ds41-synthetic-math-plan.v1',seed=SEED,repeats=REPEATS,
        input_rows=1,hidden=c.hidden_size,intermediate=c.moe_intermediate_size,experts=6,
        dense=[dict(name=name,k=k,n=n,dtypes=['float32','bfloat16'])for name,k,n in DENSE],
        quantized_dense=[dict(name=name,k=k,n=n,mode='mxfp8',qweight_shape=[n,k//4],
            qweight_dtype='uint32',scales_shape=[n,k//32],scales_dtype='uint8',
            activation_dtypes=['float32','bfloat16'],bf16_optional=True)for name,k,n in DENSE],
        added_MXFP8_operand_bytes=quantized_bytes,
        grouped_woa=dict(groups=8,k=4096,n=1024,coefficient_dtype='bfloat16'),
        resident_operand_upper_bound_bytes=bound,array_budget_bytes=MAX_ARRAY_BYTES,
        head_excluded='Full129280x5120 F32 head alone exceeds2GiB; this bounded function does not substitute a cropped head',
        scope='Synthetic operator costs only; no checkpoint/model/correctness/quality or serving-rate claim')


def _arrays(value):
    if isinstance(value,(tuple,list)):return tuple(x for v in value for x in _arrays(v))
    return(value,)


def _measure(mx,fn):
    start=time.perf_counter_ns();value=fn();submitted=time.perf_counter_ns()
    mx.eval(*_arrays(value));done=time.perf_counter_ns()
    return value,dict(host_graph_submit_seconds=(submitted-start)/1e9,
        completion_wait_seconds=(done-submitted)/1e9,total_seconds=(done-start)/1e9)


def _live(callback):
    if callback is not None and callback()is False:raise RuntimeError('Synthetic component live guard refused')


def _finite(mx,value):
    return all(bool(mx.all(mx.isfinite(x)).item())for x in _arrays(value))


def _sample(mx,rng,shape,scale=.125):
    import numpy as np
    return mx.array(rng.standard_normal(shape,dtype=np.float32)*np.float32(scale))


def _expert(mx,rng,kind):
    import numpy as np
    from mlx2.adapters.deepseek_v41_source_buffers import HostExpert,canonical_layout,materialize
    layout=canonical_layout(kind);slab=np.empty(layout.nbytes,np.uint8);host=HostExpert(slab,layout)
    # Fill through the owned slab; HostExpert's read-only views preserve source ownership.
    for t in layout.tensors:
        view=slab[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape)
        if t.name.endswith('qweight'):view[:]=rng.integers(0,2**32,size=t.shape,dtype=np.uint32)
        elif kind=='original':view[:]=119 # E8M0 2^-8, every FP4 code finite
        elif t.name.endswith('scales'):view[:]=np.float16(1/64)
        else:view[:]=np.float16(-1/128)
    value=materialize(host)
    mx.eval(value.backing,*value.values())
    return value


def _fp8_payload(rng,k,n):
    """Finite E4M3 bytes and exact Source output-shared32x32 scale geometry."""
    import numpy as np
    if type(k)is not int or type(n)is not int or k<=0 or n<=0 or k%32:
        raise ValueError('Synthetic MXFP8 requires positive N and Kgroup32')
    magnitude=rng.integers(0,65,size=(n,k),dtype=np.uint8)
    sign=rng.integers(0,2,size=(n,k),dtype=np.uint8)*np.uint8(128)
    codes=np.bitwise_or(magnitude,sign) # abs code0x00..0x40, no E4M3 nonfinite
    tile_scales=rng.integers(119,123,size=((n+31)//32,k//32),dtype=np.uint8)
    scales=np.repeat(tile_scales,32,axis=0)[:n]
    return codes.view('<u4'),scales


def _qmm(mx,x,q,scales):
    return mx.quantized_matmul(x,q,scales=scales,mode='mxfp8')


def run(mx,check_live=None):
    if check_live is not None and not callable(check_live):raise ValueError('check_live must be callable orNone')
    p=plan();_live(check_live)
    if mx.default_device()!=mx.gpu:raise ValueError('Synthetic Source math requires caller-owned GPU device')
    import numpy as np
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    from mlx2.runtime.models.deepseek_v41_fused import rounded,expand as fused_expand
    from mlx2.runtime.models.deepseek_v4 import hc_expand
    from mlx2.runtime.models.deepseek_v41_source_hc import split
    from mlx2.runtime.gguf.source_gate_up_fusion import batch_middle
    from mlx2.runtime.gguf.source_down_fusion import down_combine
    from mlx2.runtime.gguf.source_middle_fusion import exp_spelling
    rng=np.random.default_rng(SEED);cases=[]
    def add(name,fn,**metadata):cases.append(dict(name=name,fn=fn,metadata=metadata,samples=[]))
    x=_sample(mx,rng,(1,5120));norm_weight=mx.ones((5120,),mx.float32)
    residual=_sample(mx,rng,(1,1,4,5120));branch=_sample(mx,rng,(1,1,5120))
    mixes=_sample(mx,rng,(1,1,24),.05);scale=mx.array([.1,.1,.1],mx.float32);base=mx.zeros((24,),mx.float32)
    pre,post,comb=split(mixes,scale,base,iterations=20,eps=1e-6)
    mx.eval(x,norm_weight,residual,branch,pre,post,comb)
    add('source_fp8_QAT_H5120',lambda:rounded(fused_quantize(x,'fp8')),shape=[1,5120],function='fused_quantize+rounded')
    add('source_RMSnorm_BF16owner_H5120',lambda:rounded(mx.fast.rms_norm(x,norm_weight,1e-20)),shape=[1,5120],function='Model.norm owning expression')
    add('source_BF16owner_H5120',lambda:rounded(x),shape=[1,5120],function='deepseek_v41_fused.rounded')
    add('source_HC_split_sinkhorn20',lambda:split(mixes,scale,base,iterations=20,eps=1e-6),shape=[1,1,24],function='deepseek_v41_source_hc.split')
    add('source_HC_expand_reference_owner',lambda:rounded(hc_expand(branch,residual,post,comb,policy='reference')),shape=[1,1,4,5120],function='Model.expand reference owning expression')
    add('source_HC_expand_fused_owner',lambda:fused_expand(branch,residual,post,comb),shape=[1,1,4,5120],function='deepseek_v41_fused.expand')
    add('source_HC_collapse_owner',lambda:rounded(mx.sum(residual*pre[...,None],-2)),shape=[1,1,4,5120],function='Model.collapse owning expression')
    for name,k,n in DENSE:
        _live(check_live);xf=_sample(mx,rng,(1,k));wf=_sample(mx,rng,(n,k),1/128)
        xb=xf.astype(mx.bfloat16);wb=wf.astype(mx.bfloat16);mx.eval(xf,wf,xb,wb)
        add('dense_GEMV_'+name+'_F32',lambda a=xf,b=wf:a@b.T,k=k,n=n,dtype='float32',scope='GEMV intrinsic, not a Source MXFP8 projection wrapper')
        add('dense_GEMV_'+name+'_BF16',lambda a=xb,b=wb:a@b.T,k=k,n=n,dtype='bfloat16',scope='BF16 intrinsic profile, not selected Source F32 reduction law')
        qhost,shost=_fp8_payload(rng,k,n);q=mx.array(qhost);s=mx.array(shost)
        prepared=fused_quantize(xf.astype(mx.float32),'fp8').astype(mx.float32)
        prepared_bf16=prepared.astype(mx.bfloat16);mx.eval(q,s,prepared,prepared_bf16)
        add('source_MXFP8_'+name+'_preparedF32',lambda a=prepared,b=q,c=s:_qmm(mx,a,b,c),
            k=k,n=n,dtype='float32',function="Source.linear mx.quantized_matmul(prepared,q,scales,mode='mxfp8')",
            QAT_input_prepared_outside_clock=True,shared_weight_key=name)
        add('source_MXFP8_'+name+'_BF16_activation_diagnostic',lambda a=prepared_bf16,b=q,c=s:_qmm(mx,a,b,c),
            k=k,n=n,dtype='bfloat16',optional_dtype=True,shared_weight_key=name,
            scope='Same packed q/scales; activation dtype cost only, not selected Source F32 arithmetic')
    woa_x=_sample(mx,rng,(1,1,8,4096));woa_w=_sample(mx,rng,(8,1024,4096),1/128).astype(mx.bfloat16);mx.eval(woa_x,woa_w)
    add('source_WoA_grouped_F32_including_coefficient_cast',lambda:mx.einsum('btgi,goi->btgo',woa_x,woa_w.astype(mx.float32)),groups=8,k=4096,n=1024,function='SourceWeights.grouped owning expression including per-call BF16→F32 coefficient cast')
    original=[];q1=[]
    for _ in range(6):
        _live(check_live);original.append(_expert(mx,rng,'original'));q1.append(_expert(mx,rng,'q1'))
    banks={'original6':tuple(original),'q1_6':tuple(q1),'mixed3plus3':tuple(v for pair in zip(original[:3],q1[:3])for v in pair)}
    gu_input=rounded(fused_quantize(x,'fp8'));down_input=rounded(fused_quantize(_sample(mx,rng,(6,2304)),'fp8'))
    scores=mx.array([1/6]*6,mx.float32);slots=mx.array(list(range(6)),mx.uint32);ids=(3,17,29,52,80,121)
    spelling=exp_spelling();mx.eval(gu_input,down_input,scores,slots)
    def gu(bank):return batch_middle(gu_input,bank,scores,slots,exp_spelling=spelling)
    def down(bank,middle=down_input):return down_combine(tuple(middle[i:i+1]for i in range(6)),bank,ids)
    def chain(bank):return down(bank,gu(bank))
    for name,bank in banks.items():
        kinds=[e.layout.kind for e in bank]
        add('source_M1_batch_middle_'+name,lambda b=bank:gu(b),function='source_gate_up_fusion.batch_middle',kinds=kinds,input_shape=[1,5120],output_shape=[6,2304])
        add('source_M1_down_combine_'+name,lambda b=bank:down(b),function='source_down_fusion.down_combine',kinds=kinds,input_shape=[6,2304],output_shape=[1,5120])
        add('source_M1_GU_middle_DOWN_combine_'+name,lambda b=bank:chain(b),function='same actualGU/middle/down/ordered owner combination',kinds=kinds)
    # Compile/first-use gates occur here, outside seven steady timing rounds.
    unsupported=[]
    for case in cases:
        _live(check_live)
        try:value,cold=_measure(mx,case['fn'])
        except ValueError as error:
            if not case['metadata'].get('optional_dtype',False):raise
            unsupported.append(dict(name=case['name'],metadata=case['metadata'],error=str(error),
                scope='Optional BF16 activation execution unavailable in this run; not a quality verdict'))
            continue
        case['warm_compile']=cold
        if not _finite(mx,value):raise FloatingPointError('Nonfinite synthetic warmup:'+case['name'])
    cases=[case for case in cases if 'warm_compile'in case]
    order=[]
    for rep in range(REPEATS):
        offset=rep%len(cases);indices=list(range(offset,len(cases)))+list(range(offset));order.append([cases[i]['name']for i in indices])
        for index in indices:
            _live(check_live);case=cases[index];value,timing=_measure(mx,case['fn'])
            if not _finite(mx,value):raise FloatingPointError('Nonfinite synthetic execution:'+case['name'])
            case['samples'].append(timing)
    rows=[]
    for case in cases:
        rows.append(dict(name=case['name'],metadata=case['metadata'],warm_compile=case['warm_compile'],samples=case['samples'],
            medians={k:statistics.median(s[k]for s in case['samples'])for k in case['samples'][0]},finite=True))
    files=['runtime/models/deepseek_v41_config.py','runtime/models/deepseek_v41_quant.py','runtime/models/deepseek_v41_fused.py','runtime/models/deepseek_v41_source_hc.py','runtime/gguf/source_gate_up_fusion.py','runtime/gguf/source_down_fusion.py','adapters/deepseek_v41_source_buffers.py','adapters/deepseek_v41_source_weights.py']
    root=Path(__file__).resolve().parents[2]/'src/mlx2'
    return dict(schema='mlx2.ds41-synthetic-math-components.v1',plan=p,complete=True,finite_execution=True,
        case_count=len(rows),cases=rows,unsupported_cases=unsupported,rotated_order=order,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in files},
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy_seed=SEED,
        timing_scope='Fresh graph submission then mx.eval output completion; inputs resident. Warm compile/first-use separate. Finite reductions/readbacks/check_live outside timed intervals. No isolated GPU timestamp or model rate.',
        weight_scope='Seeded synthetic independent canonical packed experts and synthetic dense matrices only; no artifact/checkpoint/model constructor or real-weight read',
        math_parity_or_quality_oracle=False,performance_qualified=False,model_gain_claim=False)
