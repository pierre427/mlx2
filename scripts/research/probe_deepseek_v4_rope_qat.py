#!/usr/bin/env python3
"""Owned synthetic fused RoPE/QAT comparison; no model qualification."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic,compare,measure,source_receipt,verify_ownership
from mlx2.runtime.models import deepseek_v4_rope_qat as fused


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--ownership-window',type=Path)
    parser.add_argument('--owner-path',type=Path,action='append')
    parser.add_argument('--repetitions',type=int,default=3)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('refusing existing evidence directory')
    if not 1<=args.repetitions<=20:parser.error('bounded repetitions1..20 required')
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owners)!=2):parser.error('GPU requires ownership-window and paired receipts')
    def guard():return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard();source=source_receipt()
    source['rope_qat_module_sha256']=hashlib.sha256(Path(fused.__file__).read_bytes()).hexdigest()
    source['rope_qat_probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plans=[(1,0,1),(9,128,1),(128,262144,4),(513,0,128)]
    identity=hashlib.sha256(json.dumps(plans).encode()).hexdigest()
    result=dict(schema='mlx2.deepseek-v4-rope-qat-fixtures.v1',status='running',source=source,ownership=ownership,
                checkpoint='synthetic-rope-qat-fixtures@sha256:'+identity,run_gpu=args.run_gpu,
                qualification=False,performance_claim=False,artifact_loaded=False,cases=[])
    args.output.mkdir(parents=True);atomic(args.output/'results.json',result)
    try:
        import mlx.core as mx
        from mlx2.runtime.models.deepseek_v4 import Rotary,simulate_fp8_kv
        mx.set_default_device(mx.gpu if args.run_gpu else mx.cpu)
        if args.run_gpu:mx.set_memory_limit(512<<20);mx.set_cache_limit(32<<20)
        rope=Rotary(64,10000,{'type':'yarn','original_max_position_embeddings':65536,'factor':16})
        for length,start,stride in plans:
            guard();rng=np.random.default_rng(20261024+length)
            value=rng.normal(size=(1,2,length,512)).astype(np.float32)
            # Multiple scales and underflow, including signed-zero inputs.
            value[...,:448]*=np.exp2(rng.integers(-15,8,size=(*value.shape[:-1],7,1))).repeat(64,-1).reshape(*value.shape[:-1],448)
            value[...,0]=np.float32(-0.)
            positions=(start+np.arange(length)*stride).astype(np.float32)
            a=mx.array(value);p=mx.array(positions)
            baseline=lambda:simulate_fp8_kv(rope(a,positions=p))
            expected,timing=measure(baseline,args.repetitions)
            oracle=fused.reference_rope_fp8(value,np.array(rope.inv),positions)
            check=compare(expected,oracle,atol=3e-6,rtol=3e-6)
            case=dict(length=length,start=start,stride=stride,baseline_oracle=check,baseline_timing=timing,passed=check['passed'])
            if args.run_gpu:
                actual,timing=measure(lambda:fused.rope_fp8(a,rope.inv,p),args.repetitions)
                numerical=compare(actual,expected,atol=3e-6,rtol=3e-6)
                exact_prefix=bool(np.array_equal(actual[...,:448].view(np.uint32),expected[...,:448].view(np.uint32)))
                case.update(candidate_timing=timing,candidate_baseline=numerical,bitwise_qat_prefix=exact_prefix)
                case['passed'] &= numerical['passed'] and exact_prefix
            case['passed']=bool(case['passed']);guard();result['cases'].append(case);atomic(args.output/'results.json',result)
        result.update(status='completed',all_checks_passed=all(c['passed'] for c in result['cases']),route_status=fused.route_status())
    except BaseException as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:atomic(args.output/'results.json',result)
    return 0 if result['all_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
