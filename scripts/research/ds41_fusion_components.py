"""Owned-probe components for Source fusion; called only after lease validation."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src


def run(mx,np,bank,artifact,out,report,compare,atomic):
    from mlx2.runtime.gguf import source_q1,source_mxfp4_pair,source_middle_fusion as epi
    from mlx2.runtime.gguf import source_gate_up_fusion as gu,source_down_fusion as down
    from mlx2.runtime.models.deepseek_v4 import limited_swiglu
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    rounded=lambda x:x.astype(mx.bfloat16).astype(mx.float32)
    rng=np.random.default_rng(4102530);report['fusion']=[]
    def record(name,functions,**meta):
        variants=compare(mx,np,functions,repeats=3)
        report['fusion'].append(dict(name=name,**meta,variants=variants));atomic(out/'results.json',report)
    for count in (1,7,99):
        g=rng.normal(0,8,(count,2304)).astype(np.float32)
        u=rng.normal(0,8,g.shape).astype(np.float32)
        g[:,:16]=np.array([0,-0.,.25,-.25,10,-10,10.03125,-10.03125,1e-20,-1e-20,100,-100,1.00390625,-1.00390625,2**-40,-2**-40],np.float32)
        gate,up=mx.array(g),mx.array(u)
        scores=mx.array(rng.uniform(.01,.99,count*3).astype(np.float32));slots=mx.array(np.arange(count,dtype=np.uint32)*3+1)
        mx.eval(gate,up,scores,slots)
        record('middle',{
            'reference':lambda:fused_quantize(rounded(limited_swiglu(rounded(gate),rounded(up))*scores[slots,None]),'fp8'),
            'fused':lambda:epi.middle(gate,up,scores,slots)},rows=count)
    report['served_silu_exp']=epi.exp_spelling()
    for kind in ('original','q1'):
        e=next(e for e,v in bank.items() if ('gate.biases' in v)==(kind=='q1'))
        data=bank[e];fields=('.qweight','.scales','.biases') if kind=='q1' else ('.qweight','.scales')
        args=tuple(data[p+f] for p in ('gate','up') for f in fields)
        x=fused_quantize(rounded(mx.array(rng.normal(0,.25,(101,5120)).astype(np.float32))),'fp8');mx.eval(x)
        def projection(v,p):
            if kind=='q1':return source_q1.matmul(v,*(data[p+f] for f in fields))
            return mx.quantized_matmul(v,data[p+'.qweight'],scales=data[p+'.scales'],mode='mxfp4')
        for count in ((1,2,3,7,12) if kind=='original' else (1,3,7,32,99)):
            rows=mx.array((np.arange(count,dtype=np.uint32)*7+5)%101);mx.eval(rows)
            record('mapped_pair',{
                'reference':lambda:mx.stack([projection(x[rows],'gate'),projection(x[rows],'up')]),
                'mapped':lambda:mx.stack((source_q1.paired_matmul_mapped if kind=='q1' else source_mxfp4_pair.paired_matmul_mapped)(x,rows,*args))
            },kind=kind,rows=count,expert=e)
        for count in ((1,2,3,5,6,7,10,12) if kind=='original' else (1,7,99)):
            rows=mx.array((np.arange(count,dtype=np.uint32)*7+5)%101)
            scores=mx.array(rng.uniform(.01,.99,count).astype(np.float32));mx.eval(rows,scores)
            record('gate_up_middle',{
                'reference':lambda:fused_quantize(rounded(limited_swiglu(rounded(projection(x[rows],'gate')),rounded(projection(x[rows],'up')))*scores[:,None]),'fp8'),
                'fused':lambda:gu.middle(x,rows,scores,data,exp_spelling=report['served_silu_exp'])
            },kind=kind,rows=count,expert=e)
            def reference_raw():
                g,u=projection(x[rows],'gate'),projection(x[rows],'up')
                m=fused_quantize(rounded(limited_swiglu(rounded(g),rounded(u))*scores[:,None]),'fp8')
                return mx.concatenate([m,g,u],axis=0)
            def fused_raw():
                return mx.concatenate(gu.middle(x,rows,scores,data,exp_spelling=report['served_silu_exp'],return_projections=True),axis=0)
            record('gate_up_raw_and_middle',{'reference':reference_raw,'fused':fused_raw},kind=kind,rows=count,expert=e)
    original=[e for e,v in bank.items() if 'down.biases' not in v]
    q1=[e for e,v in bank.items() if 'down.biases' in v]
    for kinds,ids in (('original',original[:6]),('q1',q1[:6]),('mixed',original[:3]+q1[:3])):
        ids=list(reversed(ids));experts=[bank[e] for e in ids]
        mids=[fused_quantize(rounded(mx.array(rng.normal(0,.25,(1,2304)).astype(np.float32))),'fp8') for e in ids];mx.eval(mids)
        def reference():
            raw=[];total=mx.zeros((1,5120),mx.float32)
            for i in sorted(range(len(ids)),key=lambda i:ids[i]):
                d=experts[i]
                value=(source_q1.matmul(mids[i],*(d['down'+f] for f in ('.qweight','.scales','.biases')))
                    if 'down.biases' in d else mx.quantized_matmul(mids[i],d['down.qweight'],scales=d['down.scales'],mode='mxfp4'))
                raw.append(value);total=total+rounded(value)
            return mx.concatenate([total,*raw],axis=0)
        for tile in (4,8):
            def fused():
                value,raw=down.down_combine(mids,experts,ids,tile_rows=tile,return_contributions=True)
                return mx.concatenate([value,raw],axis=0)
            record('down_combine',{'reference':reference,'fused':fused},kinds=kinds,ids=ids,tile_rows=tile)
        x=fused_quantize(rounded(mx.array(rng.normal(0,.25,(1,5120)).astype(np.float32))),'fp8')
        scores=mx.array(rng.uniform(.01,.99,6).astype(np.float32))
        slots=mx.array([5,0,2,4,1,3],mx.uint32);mx.eval(x,scores,slots)
        def ordinary_middles():
            values=[]
            for i,data in enumerate(experts):
                def proj(p):
                    return (source_q1.matmul(x,*(data[p+f] for f in ('.qweight','.scales','.biases')))
                        if 'gate.biases' in data else mx.quantized_matmul(x,data[p+'.qweight'],scales=data[p+'.scales'],mode='mxfp4'))
                values.append(fused_quantize(rounded(limited_swiglu(rounded(proj('gate')),rounded(proj('up')))*scores[slots[i]]),'fp8'))
            return mx.concatenate(values,axis=0)
        def batch():return gu.batch_middle(x,experts,scores,slots,exp_spelling=report['served_silu_exp'])
        record('batch_gate_up_middle',{'reference':ordinary_middles,'fused':batch},kinds=kinds,ids=ids)
        def batch_reference_raw():
            gs=[];us=[]
            for data in experts:
                def proj(p):
                    return (source_q1.matmul(x,*(data[p+f] for f in ('.qweight','.scales','.biases')))
                        if 'gate.biases' in data else mx.quantized_matmul(x,data[p+'.qweight'],scales=data[p+'.scales'],mode='mxfp4'))
                gs.append(proj('gate'));us.append(proj('up'))
            return mx.concatenate([ordinary_middles(),*gs,*us],axis=0)
        def batch_raw():return mx.concatenate(gu.batch_middle(x,experts,scores,slots,exp_spelling=report['served_silu_exp'],return_projections=True),axis=0)
        record('batch_gate_up_raw_and_middle',{'reference':batch_reference_raw,'fused':batch_raw},kinds=kinds,ids=ids)
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    from mlx2.runtime.gguf.source_prefill_combine import combine
    routes=__import__('json').loads((out/'routes.json').read_text())
    for name in ('prefill7','prefill99'):
        ids=np.array(routes[name],np.uint32);order=np.argsort(ids.ravel(),kind='stable')
        keys=ids.ravel()[order];cuts=(np.flatnonzero(np.diff(keys))+1).tolist()
        raw=rng.normal(0,1,(ids.size,5120)).astype(np.float32)
        raw[:,0]=np.resize(np.array([2**25,1,-2**25,-0.,2**-120,-2**-120],np.float32),ids.size)
        packed=rounded(mx.array(raw[order]));parts=list(mx.split(packed,cuts,axis=0));mx.eval(parts)
        record('prefill_direct_combine',{'reference':lambda:ordered_combine(parts,ids,order),
            'direct':lambda:combine(parts,ids,order)},rows=len(ids),parts=len(parts),width=5120)
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest,q1_read
    qroot,records,_=load_q1_manifest(report['q1_root'],report['original_fingerprint'])
    report['packed_objects']=[]
    for kind,e in (('original',original[0]),('q1',q1[0])):
        host=(buffers.read_original_packed(artifact,0,e) if kind=='original'
              else buffers.borrow_q1_packed(q1_read(qroot,records[(0,e)])))
        before=mx.get_active_memory();packed=buffers.materialize(host)
        mx.eval(packed.backing,*packed.values());after=mx.get_active_memory()
        exact=all(np.array(packed[k]).tobytes()==np.array(bank[e][k]).tobytes() for k in packed)
        row=dict(kind=kind,expert=e,exact=exact,allocation_nbytes=packed.allocation_nbytes,
                 active_memory_delta=after-before,one_backing_bound=(0<=after-before<=packed.allocation_nbytes+(1<<20)))
        report['packed_objects'].append(row);atomic(out/'results.json',report)
        del packed,host
        mx.synchronize()
