"""Bounded Source prefill component probes; caller owns GPU and paired leases.

Importing this file does no GPU work. This helper never acquires ownership,
changes serving policies, or claims model-path qualification.
"""
from pathlib import Path
import time


def _packed_bank(mx, bank, artifact, report, kind, maximum=15):
    """Return at most15 actual layer0 objects; no synthetic coefficients."""
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    selected=[e for e,v in sorted(bank.items()) if ('gate.biases' in v)==(kind=='q1')][:maximum]
    if not selected:
        raise ValueError('Source PP components require actual '+kind+' experts')
    experts=[];hosts=[];load_bytes=0;started=time.perf_counter()
    root=records=None
    for e in selected:
        value=bank[e]
        if not isinstance(value,buffers.PackedExpert):
            if kind=='original':host=buffers.read_original_packed(artifact,0,e)
            else:
                from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest,q1_read
                if root is None:
                    root,records,fingerprint=load_q1_manifest(report['q1_root'],report['original_fingerprint'])
                    if report.get('q1_fingerprint') and fingerprint != report['q1_fingerprint']:
                        raise ValueError('Source PP Q1 bank fingerprint changed')
                host=buffers.borrow_q1_packed(q1_read(root,records[(0,e)]))
            hosts.append(host);load_bytes+=host.layout.nbytes;value=buffers.materialize(host)
        experts.append(value)
    # Uploads complete before host slabs leave this helper; caller owns returned
    # packed backings until all raw/timed consumers finish.
    mx.eval(*[e.backing for e in experts])
    return selected,experts,{'kind':kind,'unique_actual_objects':len(experts),
        'materialized_payload_bytes':load_bytes,'maximum_staged_host_payload_bytes':load_bytes,
        'load_prepare_seconds':time.perf_counter()-started}


def run_gu(mx,np,bank,artifact,out,report,compare,atomic):
    """Compare raw gate/up plus middle against existing GU and stock oracles."""
    from mlx2.runtime.gguf import source_gate_up_fusion as gu
    from mlx2.runtime.gguf import source_gate_up_prefill as pp
    from mlx2.runtime.gguf import source_q1
    from mlx2.runtime.gguf.source_middle_fusion import exp_spelling
    from mlx2.runtime.models.deepseek_v4 import limited_swiglu
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    if 'prefill_gu' in report:
        raise ValueError('Refusing to overwrite Source prefill GU evidence')
    out=Path(out);limit=6<<30;report['prefill_gu']=[]
    meta=report['prefill_gu_metadata']={
        'checkpoint':'DeepSeek-V4.1-Flash-1017-Source@'+artifact['identity']['fingerprint'],
        'scope':'Actual original/Q1 layer0 expert payloads; synthetic BF16-rounded FP8-QAT inputs; raw F32 dot and middle parity, not model quality',
        'qualified':False,'maximum_input_bindings':20,'maximum_backings':15,
        'allocator_ceiling_bytes':limit,'maximum_observed_peak_mlx_bytes':int(mx.get_peak_memory()),
        'timing_scope':'compare helper admits exact variants only;3 interleaved repetitions; setup excluded',
        'loads':[],'served_silu_exp':exp_spelling()}
    rounded=lambda value:value.astype(mx.bfloat16).astype(mx.float32)
    rng=np.random.default_rng(4102527)
    x=fused_quantize(rounded(mx.array(rng.normal(0,.25,(101,5120)).astype(np.float32))),'fp8')
    mx.eval(x)

    def observe():
        peak=int(mx.get_peak_memory());meta['maximum_observed_peak_mlx_bytes']=max(meta['maximum_observed_peak_mlx_bytes'],peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>limit:
            atomic(out/'results.json',report)
            raise MemoryError('Source prefill GU exceeded6GiB MLX allocator ceiling')

    for kind in ('original','q1'):
        actual_ids,actual,load=_packed_bank(mx,bank,artifact,report,kind)
        meta['loads'].append(load);observe()
        cases=[(rows,)*count for rows in (1,2,3,7,12) for count in (1,3,15)] if kind=='original' else [
            (1,),(7,),(99,),(1,3,7),(1,3,7,15,99)*3]
        for counts in cases:
            count=len(counts);experts=[actual[i%len(actual)]for i in range(count)]
            ids=[actual_ids[i%len(actual_ids)]for i in range(count)]
            plan=pp.descriptor(kind,counts);offsets=plan['offsets']
            # Per-plane row/skew variation catches wrong pointer/offset selection
            # even when15planes reuse a smaller actual loaded expert bank.
            cpu_rows=np.concatenate([(np.arange(n,dtype=np.uint32)*7+i*11+5)%101 for i,n in enumerate(counts)])
            rows=mx.array(cpu_rows);scores=mx.array(rng.uniform(.01,.99,sum(counts)).astype(np.float32));mx.eval(rows,scores)
            def existing():
                outputs=[gu.middle(x,rows[a:b],scores[a:b],e,
                    exp_spelling=meta['served_silu_exp'],return_projections=True)
                    for e,a,b in zip(experts,offsets[:-1],offsets[1:])]
                return mx.concatenate([value[j]for j in range(3)for value in outputs],axis=0)
            def candidate():
                return mx.concatenate(pp.pp_middle(x,experts,rows,scores,counts,
                    exp_spelling=meta['served_silu_exp'],return_projections=True),axis=0)
            functions={'reference':existing,'worklist':candidate}
            # Bridge to ordinary stock QMM/Q1 math on every one-expert law and
            # one variable-count Q1 batch, independent of the existing GU path.
            stock_oracle=count==1 or (kind=='q1' and counts==(1,3,7))
            if stock_oracle:
                def stock():
                    outputs=[]
                    for e,a,b in zip(experts,offsets[:-1],offsets[1:]):
                        activation=x[rows[a:b]]
                        def project(p):
                            if kind=='q1':return source_q1.matmul(activation,
                                *(e[p+f]for f in ('.qweight','.scales','.biases')))
                            return mx.quantized_matmul(activation,e[p+'.qweight'],scales=e[p+'.scales'],mode='mxfp4')
                        g,u=project('gate'),project('up')
                        m=fused_quantize(rounded(limited_swiglu(rounded(g),rounded(u))*scores[a:b,None]),'fp8')
                        outputs.append((m,g,u))
                    return mx.concatenate([value[j]for j in range(3)for value in outputs],axis=0)
                functions['stock']=stock
            variants=compare(mx,np,functions,repeats=3);observe()
            row={'name':'prefill_gate_up_raw_and_middle','kind':kind,'row_counts':list(counts),
                'expert_ids':ids,'unique_actual_expert_ids':len(set(ids)),
                'reused_actual_objects':count>len(set(ids)),'input_bindings':plan['logical_input_bindings'],
                'output_shape':list(plan['output_shape']),'existing_gu_launches':count,
                'worklist_gu_launches':1,'stock_oracle':stock_oracle,'variants':variants}
            report['prefill_gu'].append(row);atomic(out/'results.json',report)
        experts=None;del actual
        mx.synchronize();mx.clear_cache();observe()
    meta['required_raw_exact']=all(v['exact']for row in report['prefill_gu']for v in row['variants'].values())
    meta['cases']=len(report['prefill_gu']);atomic(out/'results.json',report)
