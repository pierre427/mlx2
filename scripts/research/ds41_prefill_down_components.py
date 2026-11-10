"""Owned-GPU Source prefill down and mapped-assembly component probes."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path


def run_down(mx,np,bank,artifact,out,report,compare,atomic):
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert
    from mlx2.runtime.gguf import source_down_prefill as pp,source_q1
    from mlx2.runtime.gguf.source_prefill_combine import combine_mapped
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    if mx.default_device()!=mx.gpu:
        raise ValueError('Source prefill components require caller-owned GPU; no CPU fallback')
    if 'prefill_down' in report or 'prefill_combine' in report:
        raise ValueError('Refusing to overwrite prefill down/combine evidence')
    report['prefill_down'],report['prefill_combine']=[],[]
    meta=report['prefill_down_metadata']={
        'scope':'Actual packed original/Q1 coefficients; synthetic finite FP8-QAT middles; raw/rounded down parity; mapped assembly uses independent synthetic contributions',
        'qualified':False,'maximum_backing_bindings':15,'maximum_total_buffer_bindings_with_raw':20,
        'allocator_ceiling_bytes':6<<30,'original_multirow_dispatch':'candidate until each raw stock oracle passes',
        'timing_scope':'3 interleaved repetitions after raw gates; diagnostic concatenation is outside serving candidate design'}
    out=Path(out);rng=np.random.default_rng(4102531)
    rounded=lambda x:x.astype(mx.bfloat16).astype(mx.float32)
    def observe():
        peak=int(mx.get_peak_memory())
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        meta['peak_mlx_bytes']=max(meta.get('peak_mlx_bytes',0),peak)
        if peak>meta['allocator_ceiling_bytes']:
            atomic(out/'results.json',report)
            raise MemoryError('Source PP down/combine allocator exceeded6GiB')
    for kind in ('original','q1'):
        actual=[(e,v)for e,v in sorted(bank.items())if ('down.biases'in v)==(kind=='q1')]
        if not actual or not all(isinstance(v,PackedExpert)for _,v in actual):
            raise ValueError('Parent must preload immutable packed actual '+kind+' bank')
        cases=[(r,)*n for r in (1,2,3,7,12)for n in (1,3,15)] if kind=='original'else[(1,),(7,),(99,),(1,3,7),(1,3,7,15,99)*3]
        for counts in cases:
            experts=[actual[i%len(actual)][1]for i in range(len(counts))]
            ids=[actual[i%len(actual)][0]for i in range(len(counts))]
            middle=fused_quantize(rounded(mx.array(rng.normal(0,.25,(sum(counts),2304)).astype(np.float32))),'fp8')
            mx.eval(middle)
            starts=np.cumsum([0]+list(counts))
            def ordinary():
                raw=[]
                for i,e in enumerate(experts):
                    x=middle[int(starts[i]):int(starts[i+1])]
                    raw.append(source_q1.matmul(x,e['down.qweight'],e['down.scales'],e['down.biases']) if kind=='q1'else
                        mx.quantized_matmul(x,e['down.qweight'],scales=e['down.scales'],mode='mxfp4'))
                raw=raw[0]if len(raw)==1 else mx.concatenate(raw,0)
                return mx.concatenate((rounded(raw),raw),0)
            def candidate():
                value,raw=pp.pp_down(middle,experts,counts,return_raw=True)
                return mx.concatenate((value,raw),0)
            variants=compare(mx,np,{'reference':ordinary,'batch_down':candidate},repeats=3)
            observe()
            report['prefill_down'].append({'name':'prefill_down_raw_and_rounded','kind':kind,
                'row_counts':list(counts),'expert_ids':ids,'unique_actual_experts':len(set(ids)),
                'reused_actual_objects':len(set(ids))<len(ids),'stock_down_launches':len(counts),
                'batch_down_launches':1,'variants':variants})
            atomic(out/'results.json',report)
        experts=None;middle=None
        mx.synchronize();mx.clear_cache();observe()
    # These assembly buffers intentionally scramble nonconsecutive expert IDs.
    # Per-expert additions, not group subtotals, are the exact reference.
    for rows in (1,7,99):
        values=rng.integers(0,61,(rows,6),dtype=np.uint32)
        values[::3,2]=values[::3,0]
        order=np.argsort(values.ravel(),kind='stable')
        raw=rng.normal(0,.25,(rows,6,5120)).astype(np.float32)
        raw[:,:,0]=np.resize(np.array([2**25,1,-2**25,1,-0.,0.],np.float32),(rows,6))
        raw[:,:,1]=-0.
        unique,counts=np.unique(values,return_counts=True)
        sorted_raw=raw.reshape(-1,5120)[order]
        ordinary_cpu=list(np.split(sorted_raw,np.cumsum(counts)[:-1]))
        ordinary_parts=[mx.array(p)for p in ordinary_cpu]
        for n in (1,7,31):
            groups=min(n,len(unique));parts=[];descriptors=[]
            for i in reversed(range(groups)):
                take=list(range(i,len(unique),groups))
                parts.append(mx.array(np.concatenate([ordinary_cpu[j]for j in take])))
                descriptors.append(tuple((int(unique[j]),int(counts[j]))for j in take))
            mx.eval(parts,ordinary_parts)
            variants=compare(mx,np,{
                'reference':lambda:ordered_combine(ordinary_parts,values,order),
                'mapped_parts':lambda:combine_mapped(parts,values,order,descriptors)},repeats=3)
            observe()
            report['prefill_combine'].append({'name':'prefill_bucketed_ordered_combine','rows':rows,
                'topk':6,'physical_parts':len(parts),'logical_experts':len(unique),
                'concat_payload_avoided_bytes':values.size*5120*4,
                'input_profile':'synthetic F32 including cancellation and signed-zero; no early expert/group subtotal',
                'variants':variants})
            atomic(out/'results.json',report)
        parts=None;ordinary_parts=None
        mx.synchronize();mx.clear_cache();observe()
    meta['required_raw_exact']=all(v['exact']for key in ('prefill_down','prefill_combine')for r in report[key]for v in r['variants'].values())
    meta['down_cases']=len(report['prefill_down']);meta['combine_cases']=len(report['prefill_combine'])
    atomic(out/'results.json',report)


def run_down_missing(mx,np,bank,artifact,out,report,compare,atomic):
    """Only missing original local-row laws, including all V5/tail cases.

    Existing run_down is intentionally unchanged so its earlier frozen
   20down+9assembly proof remains independently reproducible.
    """
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert
    from mlx2.runtime.gguf import source_down_prefill as pp
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    if mx.default_device()!=mx.gpu:
        raise ValueError('Source missing-down components require caller-owned GPU; no CPU fallback')
    if 'prefill_down' in report:
        raise ValueError('Refusing to overwrite Source missing-down evidence')
    actual=[(e,v)for e,v in sorted(bank.items())if 'down.biases' not in v]
    if not actual or not all(isinstance(v,PackedExpert)for _,v in actual):
        raise ValueError('Parent must preload immutable packed actual original bank')
    report['prefill_down']=[]
    meta=report['prefill_down_metadata']={
        'scope':'Targeted actual original payload / synthetic finite FP8-QAT middle raw+rounded down proof; no GU, assembly or BF16-wide repetition',
        'targeted_missing_rows':[4,5,6,8,9,10,11],
        'extra_V5_E15_rows':[5,9,10],'qualified':False,
        'allocator_ceiling_bytes':6<<30,'maximum_total_buffer_bindings_with_raw':20}
    out=Path(out);rng=np.random.default_rng(4102532)
    rounded=lambda x:x.astype(mx.bfloat16).astype(mx.float32)
    cases=[(r,3)for r in (4,5,6,8,9,10,11)]+[(r,15)for r in (5,9,10)]
    for rows,count in cases:
        counts=(rows,)*count
        experts=[actual[i%len(actual)][1]for i in range(count)]
        ids=[actual[i%len(actual)][0]for i in range(count)]
        middle=fused_quantize(rounded(mx.array(rng.normal(0,.25,(rows*count,2304)).astype(np.float32))),'fp8')
        mx.eval(middle)
        def ordinary():
            raw=[mx.quantized_matmul(middle[i*rows:(i+1)*rows],e['down.qweight'],
                scales=e['down.scales'],mode='mxfp4')for i,e in enumerate(experts)]
            raw=mx.concatenate(raw,0)
            return mx.concatenate((rounded(raw),raw),0)
        def candidate():
            value,raw=pp.pp_down(middle,experts,counts,return_raw=True)
            return mx.concatenate((value,raw),0)
        variants=compare(mx,np,{'reference':ordinary,'batch_down':candidate},repeats=3)
        peak=int(mx.get_peak_memory())
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        meta['peak_mlx_bytes']=max(meta.get('peak_mlx_bytes',0),peak)
        report['prefill_down'].append({'name':'prefill_down_missing_raw_and_rounded','kind':'original',
            'row_counts':list(counts),'expert_ids':ids,'unique_actual_experts':len(set(ids)),
            'reused_actual_objects':len(set(ids))<len(ids),'vecs_per_tg':pp.plan('original',counts)['vecs'],
            'stock_down_launches':count,'batch_down_launches':1,'variants':variants})
        atomic(out/'results.json',report)
        if peak>meta['allocator_ceiling_bytes']:
            raise MemoryError('Source missing-down components exceeded6GiB allocator ceiling')
    meta['cases']=len(report['prefill_down'])
    meta['required_raw_exact']=all(v['exact']for r in report['prefill_down']for v in r['variants'].values())
    atomic(out/'results.json',report)
