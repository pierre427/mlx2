#!/usr/bin/env python3
"""Lease-guarded synthetic HC primitive oracle, not model qualification."""
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
from mlx2.runtime.models import deepseek_v4_hc as hc


def fixture(rows,magnitude):
    rng=np.random.default_rng(20261024+rows)
    return {'mixes':rng.normal(size=(rows,24)).astype(np.float32)*magnitude,
            'scale':np.array([.7,.3,.9],np.float32),'base':rng.normal(size=24).astype(np.float32),
            'value':rng.normal(size=(rows,4,4096)).astype(np.float32),
            'branch':rng.normal(size=(rows,4096)).astype(np.float32),
            'head_mix':rng.normal(size=(rows,4)).astype(np.float32)*magnitude,
            'head_scale':np.array([.7],np.float32),'head_base':rng.normal(size=4).astype(np.float32)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--ownership-window',type=Path);p.add_argument('--owner-path',type=Path,action='append')
    p.add_argument('--repetitions',type=int,default=3)
    args=p.parse_args(argv)
    if args.output.exists():p.error('refusing existing evidence directory')
    if not 1<=args.repetitions<=20:p.error('bounded repetitions1..20 required')
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owners)!=2):p.error('GPU requires ownership-window and paired receipts')
    def guard():return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard();source=source_receipt()
    source['hc_module_sha256']=hashlib.sha256(Path(hc.__file__).read_bytes()).hexdigest()
    source['hc_probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plans=[(1,.1),(7,10.),(128,.1),(8,1000.)]
    identity=hashlib.sha256(json.dumps(plans).encode()).hexdigest()
    result=dict(schema='mlx2.deepseek-v4-hc-fixtures.v1',status='running',source=source,ownership=ownership,
                checkpoint='synthetic-hc-fixtures@sha256:'+identity,run_gpu=args.run_gpu,
                qualification=False,performance_claim=False,artifact_loaded=False,cases=[])
    args.output.mkdir(parents=True);atomic(args.output/'results.json',result)
    try:
        if args.run_gpu:
            import mlx.core as mx
            mx.set_default_device(mx.gpu);mx.set_memory_limit(512<<20);mx.set_cache_limit(32<<20)
            result['device_info']=mx.metal.device_info()
        for rows,magnitude in plans:
            guard();data=fixture(rows,magnitude)
            pre,post,comb=hc.reference_split(data['mixes'],data['scale'],data['base'])
            expected={'split':np.concatenate((pre,post,comb.reshape(rows,16)),axis=-1),
                      'sum':np.sum(data['value'].astype(np.float64)*pre[...,None],axis=1),
                      'expand':data['branch'][:,None].astype(np.float64)*post[...,None]+np.einsum('rsh,rsd->rhd',comb,data['value'].astype(np.float64))}
            head_z=data['head_mix'].astype(np.float64)*data['head_scale']+data['head_base']
            head_w=np.exp(-np.logaddexp(0,-head_z))+1e-6
            expected['head']=np.sum(data['value'].astype(np.float64)*head_w[...,None],axis=1)
            case=dict(rows=rows,magnitude=magnitude,passed=all(np.isfinite(a).all() for a in expected.values()),checks={})
            if args.run_gpu:
                a={k:mx.array(v) for k,v in data.items()};mx.eval(list(a.values()))
                coeff=hc.split_sinkhorn(a['mixes'],a['scale'],a['base']);mx.eval(coeff)
                invocations={
                    'sum':lambda:hc.weighted_sum(a['value'],coeff[0]),
                    'expand':lambda:hc.expand(a['branch'],a['value'],coeff[1],coeff[2]),
                    'head':lambda:hc.head_reduce(a['value'],a['head_mix'],a['head_scale'],a['head_base'])}
                # Evaluate one split call per timing sample; return its complete coefficient row.
                def split_call():
                    p,q,c=hc.split_sinkhorn(a['mixes'],a['scale'],a['base'])
                    return mx.concatenate((p,q,c.reshape(rows,16)),axis=-1)
                invocations['split']=split_call
                actual={}
                for name,fn in invocations.items():
                    value,timing=measure(fn,args.repetitions);actual[name]=value
                    check=compare(value,expected[name],atol=5e-6,rtol=5e-5)
                    case['checks'][name]={'numerical':check,'timing':timing};case['passed'] &= check['passed']
                last=rows-1
                one=hc.split_sinkhorn(a['mixes'][last:last+1],a['scale'],a['base'])
                serial=mx.concatenate((one[0],one[1],one[2].reshape(1,16)),axis=-1);mx.eval(serial)
                case['coefficient_row_invariance']=bool(np.array_equal(actual['split'][last:last+1],np.array(serial)))
                case['passed'] &= case['coefficient_row_invariance']
            case['passed']=bool(case['passed']);guard()
            result['cases'].append(case);atomic(args.output/'results.json',result)
        result['status']='completed';result['all_checks_passed']=all(c['passed'] for c in result['cases'])
    except BaseException as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:atomic(args.output/'results.json',result)
    return 0 if result['all_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
