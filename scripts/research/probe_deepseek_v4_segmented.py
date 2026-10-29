#!/usr/bin/env python3
"""Owned native-GGUF segmented-prefill component fixtures; no model qualification."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import (
    atomic,compare,independent_dequant,measure,source_receipt,synthetic,upload,verify_ownership,
)
from mlx2.runtime.gguf.segmented_prefill import reference_plan,scratch_bytes


def fixture(tokens,k,hidden,experts):
    if not 6<=experts<=256: raise ValueError('fixtures require6..256experts')
    rng=np.random.default_rng(260424+tokens)
    ids=np.array([rng.choice(experts,6,replace=False) for _ in range(tokens)],np.uint32)
    scores=rng.random((tokens,6),dtype=np.float32)
    scores=1.5*scores/scores.sum(-1,keepdims=True)
    x=rng.normal(0,.3,(tokens,k)).astype(np.float32)
    gate=synthetic((experts,hidden,k),16,seed=4)
    up=synthetic((experts,hidden,k),16,seed=5)
    down=synthetic((experts,k,hidden),10,seed=6)
    return x,ids,scores,gate,up,down


def working_bound(tokens,k,hidden,experts):
    packed=experts*k*hidden*(66*2+84)//256
    # Host+GPU packed, FP32/half-rounded reference oneexpertatime, assignedoutputs,
    # candidate scratch, reference temporaries and allocator/compiler allowance.
    return 256*(1<<20)+2*packed+6*k*hidden*4+scratch_bytes(tokens,6,hidden,k,experts)+tokens*6*k*4


def independent_half_oracle(x,ids,scores,gate,up,down):
    tokens,k=x.shape; experts,hidden=gate.shape[:2]
    # Decode one expert at a time. Never constructE256 floating expert tables.
    rows=np.zeros((tokens,6,k),np.float32)
    xhalf=x.astype(np.float16).astype(np.float32)
    clipped_gate=clipped_up=0
    for expert in range(experts):
        token,slot=np.nonzero(ids==expert)
        if not len(token): continue
        g=independent_dequant(gate[expert],16,(hidden,k)).astype(np.float16).astype(np.float32)
        u=independent_dequant(up[expert],16,(hidden,k)).astype(np.float16).astype(np.float32)
        d=independent_dequant(down[expert],10,(k,hidden)).astype(np.float16).astype(np.float32)
        gv=xhalf[token]@g.T; uv=xhalf[token]@u.T
        clipped_gate+=int(np.count_nonzero(gv>10)); clipped_up+=int(np.count_nonzero(np.abs(uv)>10))
        gv=np.minimum(gv,10); uv=np.clip(uv,-10,10)
        middle=(gv/(1+np.exp(-gv)))*uv
        rows[token,slot]=middle.astype(np.float16).astype(np.float32)@d.T
    expected=np.zeros((tokens,k),np.float32)
    for slot in range(6): expected+=scores[:,slot,None]*rows[:,slot]
    return expected,{'gate_values_clipped':clipped_gate,'up_values_clipped':clipped_up,
                     'law':'FP16roundedx/decodedweights,F32matmul+clippedSwiGLU,FP16roundedmid/downweights,F32matmul,originalslotF32scorecombine'}


def run_case(tokens,k,hidden,experts,*,gpu,repetitions,cap,guard,ordinary):
    bound=working_bound(tokens,k,hidden,experts)
    if bound>cap: raise RuntimeError(f'fixture bound{bound} exceeds workingcap{cap}')
    x,ids,scores,gate,up,down=fixture(tokens,k,hidden,experts)
    cpu_plan=reference_plan(ids,experts)
    expected,law=independent_half_oracle(x,ids,scores,gate,up,down)
    result={'tokens':tokens,'topk':6,'experts':experts,'input':k,'hidden':hidden,
            'cpu_plan_tiles':len(cpu_plan['tiles']),'average_rows_per_expert':tokens*6/experts,
            'tile_capacity':cpu_plan['max_tiles'],'scratch_bytes':scratch_bytes(tokens,6,hidden,k,experts),
            'conservative_working_bound_bytes':bound,'oracle':law,'qualified':False,
            'route':'gguf-segmented-nax-fp16-v1','artifact_loaded':False}
    if not gpu:
        result['cpu_reference_finite']=bool(np.isfinite(expected).all()); return result
    import mlx.core as mx
    from mlx2.runtime.gguf.segmented_prefill import segmented_moe_prefill,build_plan
    gm=upload(gate,16,(experts,hidden,k),'gate'); um=upload(up,16,(experts,hidden,k),'up')
    dm=upload(down,10,(experts,k,hidden),'down')
    xm,im,sm=mx.array(x),mx.array(ids),mx.array(scores)
    plan=build_plan(im,experts)
    mx.eval(plan.order,plan.inverse,plan.sorted_ids,plan.tiles,plan.tile_count,plan.status)
    count=int(plan.tile_count.item()); status=int(plan.status.item())
    result['gpu_plan']={'status':status,'tiles':count,
        'stable_order_equal':bool(np.array_equal(np.array(plan.order),cpu_plan['order'])),
        'inverse_equal':bool(np.array_equal(np.array(plan.inverse),cpu_plan['inverse'])),
        'sorted_ids_equal':bool(np.array_equal(np.array(plan.sorted_ids),cpu_plan['sorted_ids'])),
        'tile_descriptors_equal':bool(count==len(cpu_plan['tiles']) and np.array_equal(np.array(plan.tiles)[:count],cpu_plan['tiles']))}
    guard()
    actual,timing=measure(lambda:segmented_moe_prefill(xm,im,sm,gm,um,dm,max_scratch_bytes=cap),repetitions)
    result['numerical_vs_fp16_operand_oracle']=compare(actual,expected,atol=.02,rtol=.005)
    result['timing']=timing
    if ordinary:
        from mlx2.runtime.gguf.packed_metal import packed_moe
        baseline,bt=measure(lambda:packed_moe(xm,im,sm,gm,um,dm),repetitions)
        result['ordinary_fp32_reference']={'timing':bt,'different_arithmetic':True,
            'candidate_vs_ordinary':compare(actual,baseline,atol=.02,rtol=.02),
            'observational_kernel_speed_ratio':bt['warm_median_seconds']/timing['warm_median_seconds']}
    result['allocator_peak_bytes']=mx.get_peak_memory()
    guard(); return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--ownership-window',type=Path)
    parser.add_argument('--owner-path',type=Path,action='append')
    parser.add_argument('--batches',type=int,nargs='+',default=[32,128,512])
    parser.add_argument('--experts',type=int,default=32)
    parser.add_argument('--full-shapes',action='store_true')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--max-working-bytes',type=int,default=1<<30)
    parser.add_argument('--skip-ordinary',action='store_true')
    args=parser.parse_args(argv)
    if args.output.exists(): parser.error('refusing existing evidence directory')
    if any(not 1<=x<=2048 for x in args.batches) or not 6<=args.experts<=256 or not 1<=args.repetitions<=20 or not 0<args.max_working_bytes<=2<<30:
        parser.error('boundedbatches1..2048,experts6..256,repetitions1..20,workingcap<=2GiBrequired')
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owners)!=2): parser.error('GPU requires matching external ownership')
    def guard(): return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard(); source=source_receipt()
    from mlx2.runtime.gguf import segmented_prefill as candidate
    source['segmented_module_sha256']=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
    source['segmented_probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    k,hidden=(4096,2048) if args.full_shapes else (256,256)
    for batch in args.batches:
        if working_bound(batch,k,hidden,args.experts)>args.max_working_bytes:
            parser.error('fixture allocation bound exceeds requested memory cap before allocation')
    args.output.mkdir(parents=True)
    fixture_hash=hashlib.sha256(json.dumps([args.batches,args.experts,k,hidden,260424]).encode()).hexdigest()
    receipt={'schema':'mlx2.deepseek-v4-segmented-prefill-fixtures.v1','status':'running',
             'checkpoint':'synthetic-segmented-prefill@sha256:'+fixture_hash,
             'source':source,'ownership':ownership,'run_gpu':args.run_gpu,
             'qualification':False,'performance_claim':False,'artifact_loaded':False,'cases':[],
             'timing_scope':'observational kernel fixtures, no model throughput or thermally controlled comparison'}
    atomic(args.output/'results.json',receipt)
    try:
        if args.run_gpu:
            import mlx.core as mx
            mx.set_default_device(mx.gpu);mx.set_memory_limit(args.max_working_bytes);mx.set_cache_limit(64<<20)
            receipt['device_info']=mx.metal.device_info()
        for batch in args.batches:
            guard();case=run_case(batch,k,hidden,args.experts,gpu=args.run_gpu,repetitions=args.repetitions,
                                 cap=args.max_working_bytes,guard=guard,ordinary=not args.skip_ordinary)
            receipt['cases'].append(case);atomic(args.output/'results.json',receipt)
            print(json.dumps({'tokens':batch,'plan':case.get('gpu_plan'),'numerical':case.get('numerical_vs_fp16_operand_oracle'),'timing':case.get('timing')}),flush=True)
            gc.collect()
            if args.run_gpu:mx.clear_cache()
        receipt['all_numerical_checks_passed']=all(c.get('cpu_reference_finite',True)
            and c.get('numerical_vs_fp16_operand_oracle',{}).get('passed',True)
            and all(value is True for key,value in c.get('gpu_plan',{}).items() if key.endswith('_equal'))
            and c.get('gpu_plan',{}).get('status',0)==0 for c in receipt['cases'])
        receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:atomic(args.output/'results.json',receipt)
    return 0 if receipt['all_numerical_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
