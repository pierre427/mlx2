#!/usr/bin/env python3
"""Bounded visible-attention component oracle; GPU requires verified ownership.

No full model is loaded. Source/ownership guards are shared with the packed
fixture unit. Synthetic checks are not qualification or model throughput.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic, compare, measure, source_receipt, verify_ownership
from mlx2.runtime.models.deepseek_v4_visible_attention import reference_visible_attention


PLANS=(
    dict(name='cold-local-single',ratio=0,offset=0,length=1),
    dict(name='local-verifier8',ratio=0,offset=120,length=8),
    dict(name='ratio4-pool-boundary',ratio=4,offset=125,length=5),
    dict(name='ratio128-pool-boundary',ratio=128,offset=125,length=5),
    dict(name='ratio4-indexer-top512',ratio=4,offset=4093,length=8,topk=512,indexer=True),
    dict(name='ratio128-long-position',ratio=128,offset=32765,length=8),
    dict(name='ratio4-long-position-top512',ratio=4,offset=262141,length=8,topk=512),
    dict(name='ratio4-verifier-row-invariance',ratio=4,offset=125,length=8,invariance=True),
    dict(name='ratio128-verifier-row-invariance',ratio=128,offset=253,length=8,invariance=True),
    dict(name='sink-dominates-stable-zero',ratio=4,offset=13,length=3,sink=1000.),
)


def fixture(plan, *, seed=20261024):
    rng=np.random.default_rng(seed+plan['offset']+plan['ratio'])
    length=plan['length']; offset=plan['offset']; ratio=plan['ratio']; window=128
    local_count=min(offset,window-1)+length
    pools=0 if not ratio else (offset+length)//ratio
    data={
        'q':rng.normal(0,.2,(1,64,length,512)).astype(np.float32),
        'local':rng.normal(0,.3,(1,1,local_count,512)).astype(np.float32),
        'pooled':None if not pools else rng.normal(0,.3,(1,pools,512)).astype(np.float32),
        'sinks':rng.normal(0,.3,(64,)).astype(np.float32),
        'indices':None,'window':window,'offset':offset,'ratio':ratio,
    }
    if 'sink' in plan:
        data['sinks'][:]=plan['sink']
    if plan.get('topk'):
        width=plan['topk']
        data['indices']=np.stack([rng.choice((offset+t+1)//ratio,width,replace=False)
                                  for t in range(length)],axis=0)[None].astype(np.uint32)
    if plan.get('indexer'):
        data['index_query']=rng.normal(0,.2,(1,64,length,128)).astype(np.float32)
        data['index_pooled']=rng.normal(0,.3,(1,pools,128)).astype(np.float32)
        data['index_weights']=rng.normal(0,.2,(1,length,64)).astype(np.float32)
        data['indices']=independent_indexer(data,plan['topk'])
    return data


def independent_indexer(data, width):
    q,p,w=(data[k] for k in ('index_query','index_pooled','index_weights'))
    result=[]
    for t in range(q.shape[2]):
        # Independent float64 score law; masked future pools cannot enter top-k.
        score=np.maximum(q[0,:,t].astype(np.float64)@p[0].astype(np.float64).T,0)
        score=(score*w[0,t,:,None].astype(np.float64)/np.sqrt(64*128)).sum(axis=0)
        score[np.arange(p.shape[1]) >= (data['offset']+t+1)//4]=-np.inf
        result.append(np.argpartition(-score,width-1)[:width])
    return np.array(result,dtype=np.uint32)[None]


def row_inputs(data, row):
    absolute=data['offset']+row
    base=data['offset']+data['q'].shape[2]-data['local'].shape[2]
    end=absolute-base+1; start=max(0,end-data['window'])
    pools=0 if not data['ratio'] else (absolute+1)//data['ratio']
    return dict(q=data['q'][:,:,row:row+1],local=data['local'][:,:,start:end],
                pooled=None if data['pooled'] is None else data['pooled'][:,:pools],
                indices=None if data['indices'] is None else data['indices'][:,row:row+1],
                sinks=data['sinks'],offset=absolute,window=data['window'],ratio=data['ratio'])


def sampled_reference(data, heads, rows):
    pieces=[]
    for row in rows:
        values=row_inputs(data,row)
        values['q']=values['q'][:,heads]
        values['sinks']=values['sinks'][heads]
        pieces.append(reference_visible_attention(**values))
    return np.concatenate(pieces,axis=2)


def fixture_bytes(data):
    return sum(value.nbytes for value in data.values() if isinstance(value,np.ndarray))


def estimated_fixture_bytes(plan):
    length=plan['length']; offset=plan['offset']; ratio=plan['ratio']
    pools=0 if not ratio else (offset+length)//ratio
    total=(64*length*512 + (min(offset,127)+length)*512 + pools*512 + 64)*4
    total+=length*plan.get('topk',0)*4
    if plan.get('indexer'):
        total+=(64*length*128+pools*128+length*64)*4
    return total


def run_case(plan, *, gpu, repetitions, guard, max_working_bytes):
    bound=256*(1<<20)+2*estimated_fixture_bytes(plan)
    if bound>max_working_bytes:
        raise RuntimeError(f"fixture{plan['name']} bound {bound} exceeds memory cap")
    before=time.perf_counter(); data=fixture(plan)
    preparation=time.perf_counter()-before
    if fixture_bytes(data)>estimated_fixture_bytes(plan):
        raise RuntimeError("fixture allocation exceeded precomputed bound")
    heads=[0,17,63]; rows=sorted({0,data['q'].shape[2]//2,data['q'].shape[2]-1})
    expected=sampled_reference(data,heads,rows)
    result={**plan,'physical_width':1,'query_heads':64,'head_dim':512,
            'pool_rows':0 if data['pooled'] is None else data['pooled'].shape[1],
            'local_band_rows':data['local'].shape[2],
            'fixture_preparation_seconds':preparation,'fixture_host_bytes':fixture_bytes(data),
            'conservative_working_bound_bytes':bound,
            'oracle':{'arithmetic':'independent NumPy float64 dense softmax over exact visible set',
                      'heads_checked':heads,'query_rows_checked':rows,'all_output_finiteness_checked':gpu},
            'route':'visible_rows_fp32','qualified':False,'artifact_loaded':False}
    if not gpu:
        result['cpu_oracle_finite']=bool(np.isfinite(expected).all())
        return result
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4_visible_attention import visible_attention,route_status
    arrays={name:None if data[name] is None else mx.array(data[name])
            for name in ('q','local','pooled','indices','sinks')}
    if plan.get('indexer'):
        from mlx2.runtime.models.deepseek_v4 import index_select
        iq,ip,iw=(mx.array(data[name]) for name in ('index_query','index_pooled','index_weights'))
        ix,timing=measure(lambda:index_select(iq,ip,iw,None,plan['topk'],offset=data['offset']),repetitions)
        match=np.array_equal(np.sort(ix,axis=-1),np.sort(data['indices'],axis=-1))
        result['indexer']={'selected_set_matches_cpu_float64':bool(match),'timing':timing,
                           'top_k':plan['topk'],'query_heads':64,'head_dim':128}
        data['indices']=ix.astype(np.uint32); arrays['indices']=mx.array(data['indices'])
        # Attention compares the exact selected set actually produced by the indexer.
        expected=sampled_reference(data,heads,rows)
    mx.eval([value for value in arrays.values() if value is not None])
    guard(); status_before=route_status()
    def invoke():
        return visible_attention(**arrays,offset=data['offset'],window=data['window'],ratio=data['ratio'])
    actual,timing=measure(invoke,repetitions)
    timing['first_evaluation_scope']='first target-fixture dispatch after separate synthetic self-check; excludes input upload; compiler cache not proven cold'
    sampled=actual[:,heads][:,:,rows]
    result['numerical']=compare(sampled,expected,atol=3e-6,rtol=3e-5)
    result['all_output_finite']=bool(np.isfinite(actual).all())
    result['timing']=timing
    if plan.get('invariance'):
        serial=[]
        for row in range(data['q'].shape[2]):
            one=row_inputs(data,row)
            device={key:None if one[key] is None else mx.array(one[key])
                    for key in ('q','local','pooled','indices','sinks')}
            out=visible_attention(**device,offset=one['offset'],window=one['window'],ratio=one['ratio'])
            mx.eval(out); serial.append(np.array(out))
        serial=np.concatenate(serial,axis=2)
        result['verifier_row_invariance']={'bitwise_equal':bool(np.array_equal(actual,serial)),
                                           'numerical':compare(actual,serial,atol=0,rtol=0),
                                           'scope':'all64heads/all8queries compared against separate B1 single-query dispatches'}
    status_after=route_status()
    result['route_status_before']=status_before; result['route_status_after']=status_after
    result['evaluated_candidate_calls']=status_after['scheduled_target_calls']-status_before['scheduled_target_calls']
    guard()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--ownership-window',type=Path)
    parser.add_argument('--owner-path',type=Path,action='append')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--max-working-bytes',type=int,default=1<<30)
    parser.add_argument('--case',action='append',choices=[x['name'] for x in PLANS])
    parser.add_argument('--target-checkpoint',default=None)
    args=parser.parse_args(argv)
    if args.output.exists(): parser.error('refusing existing evidence directory')
    if not 1<=args.repetitions<=20 or not 0<args.max_working_bytes<=2<<30:
        parser.error('bounded repetitions1..20 and positive memory cap<=2GiB required')
    owner_paths=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owner_paths)!=2):
        parser.error('GPU requires ownership-window and both matching owner receipts')
    def guard():
        return verify_ownership(args.ownership_window,owner_paths) if args.run_gpu else None
    ownership=guard()
    source=source_receipt() # Verifies imported frozen tree before any GPU work.
    from mlx2.runtime.models import deepseek_v4_visible_attention as candidate
    source['attention_module_sha256']=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
    source['attention_probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source['attention_oracle']={'package':'numpy','version':np.__version__,'dtype':'float64'}
    chosen=[p for p in PLANS if not args.case or p['name'] in args.case]
    plan_hash=hashlib.sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True)
    receipt={'schema':'mlx2.deepseek-v4-visible-attention-fixtures.v1','status':'running',
             'qualification':False,'performance_claim':False,'artifact_loaded':False,
             'checkpoint':'synthetic-deepseek-v4-attention-fixtures@sha256:'+plan_hash,
             'target_checkpoint_not_loaded':args.target_checkpoint,
             'source':source,'ownership':ownership,'run_gpu':args.run_gpu,'cases':[],
             'timing_scope':'observational component fixture timings; no model throughput or thermal replication claim'}
    atomic(args.output/'results.json',receipt)
    try:
        if args.run_gpu:
            import mlx.core as mx
            mx.set_default_device(mx.gpu); mx.set_memory_limit(args.max_working_bytes); mx.set_cache_limit(64<<20)
            receipt['device_info']=mx.metal.device_info()
            start=time.perf_counter(); candidate.self_check()
            receipt['synthetic_first_use_check_seconds']=time.perf_counter()-start
            receipt['synthetic_first_use_status']=candidate.route_status()
        for plan in chosen:
            guard()
            case=run_case(plan,gpu=args.run_gpu,repetitions=args.repetitions,guard=guard,max_working_bytes=args.max_working_bytes)
            receipt['cases'].append(case); atomic(args.output/'results.json',receipt)
            print(json.dumps({'case':case['name'],'numerical':case.get('numerical'),'timing':case.get('timing')}),flush=True)
            gc.collect()
            if args.run_gpu: mx.clear_cache()
        receipt['all_numerical_checks_passed']=all(
            c.get('cpu_oracle_finite',True) and c.get('all_output_finite',True)
            and c.get('numerical',{}).get('passed',True)
            and c.get('indexer',{}).get('selected_set_matches_cpu_float64',True)
            and c.get('verifier_row_invariance',{}).get('bitwise_equal',True) for c in receipt['cases'])
        receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        atomic(args.output/'results.json',receipt)
    return 0 if receipt['all_numerical_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
