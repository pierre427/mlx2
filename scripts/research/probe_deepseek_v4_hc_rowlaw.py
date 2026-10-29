#!/usr/bin/env python3
"""Bounded HC FP32 projection row-law diagnostic; no new kernel or AST claim.

Synthetic actual-size 16384->24/4 projections isolate identical normalized
inputs from RMS normalization. Default explicitly uses CPU. GPU execution
requires the caller's live source-bound CPG and matching paired ownership.
"""
from __future__ import annotations
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic,compare,source_receipt,verify_ownership


def fixture(outputs,kind):
    rng=np.random.default_rng(20261024+outputs)
    value=rng.normal(size=(6,16384)).astype(np.float32)
    weight=(rng.normal(size=(outputs,16384))/128).astype(np.float16)
    if kind=='cancellation':
        value[:,1::2]=value[:,::2]
        weight[:,1::2]=-weight[:,::2]
        # Retain a tiny nonzero residual in a bounded subset of pairs.
        weight[:,1:65:2]=np.nextafter(weight[:,1:65:2],np.float16(np.inf))
    return value,weight


def row_difference(actual,reference):
    a,b=np.asarray(actual,np.float32),np.asarray(reference,np.float32)
    if a.shape!=b.shape:raise ValueError('row-law comparison requires identical shapes')
    bits=a.view(np.uint32)!=b.view(np.uint32);coordinates=np.argwhere(bits)
    result={'bitwise_equal':not bool(coordinates.size),'differing_values':int(bits.sum()),
            'numerical':compare(a,b,atol=2e-5,rtol=2e-5)}
    if coordinates.size:
        row,column=map(int,coordinates[0])
        result['first_divergence']={'row':row,'column':column,
            'batched_value':float(a[row,column]),'single_value':float(b[row,column]),
            'batched_bits':int(a.view(np.uint32)[row,column]),'single_bits':int(b.view(np.uint32)[row,column]),
            'single_mix_row':b[row].tolist(),'batched_mix_row':a[row].tolist(),
            'scope':'first differing HC projection row only, not full-model or output-token divergence'}
    return result


def run_case(outputs,kind,*,gpu,guard=lambda:None,repetitions=3):
    import mlx.core as mx
    raw,half=fixture(outputs,kind)
    value=mx.array(raw);weight=mx.array(half).astype(mx.float32);mx.eval(value,weight)
    # Materialize each single-row RMS result first: every matrix width below
    # sees the same physical FP32 input values, not a different RMS kernel.
    guard();single_norm=mx.concatenate([mx.fast.rms_norm(value[i:i+1],None,1e-20) for i in range(6)],axis=0)
    mx.eval(single_norm)
    oracle=np.asarray(single_norm).astype(np.float64)@half.astype(np.float64).T
    guard();single=mx.concatenate([single_norm[i:i+1]@weight.T for i in range(6)],axis=0);mx.eval(single)
    expected=np.array(single);fixed=np.array(single_norm)
    cases=[]
    for width in range(1,7):
        guard();start=time.perf_counter();out=single_norm[:width]@weight.T;mx.eval(out)
        first=time.perf_counter()-start;actual=np.array(out);warm=[]
        for _ in range(repetitions):
            guard();start=time.perf_counter();out=single_norm[:width]@weight.T;mx.eval(out);warm.append(time.perf_counter()-start)
        # Separate RMS row-law observation cannot contaminate fixed-input test.
        norm=mx.fast.rms_norm(value[:width],None,1e-20);mx.eval(norm)
        projection=row_difference(actual,expected[:width])
        normalization=row_difference(np.array(norm),fixed[:width])
        # Large RMS rows are never serialized. Only first differing coordinate
        # and bit patterns are retained for normalization attribution.
        if 'first_divergence' in normalization:
            for key in ('single_mix_row','batched_mix_row'):normalization['first_divergence'].pop(key)
        cases.append({'outputs':outputs,'fixture':kind,'rows':width,'input_width':16384,
            'physical_batch_width':1,'meaning':'causal target rows, not concurrent client lanes',
            'fixed_normalized_input_sha256':hashlib.sha256(fixed[:width].tobytes()).hexdigest(),
            'weight_f16_sha256':hashlib.sha256(half.tobytes()).hexdigest(),
            'projection':projection,'normalization_separate':normalization,
            'independent_fp64_oracle':compare(actual,oracle[:width],atol=2e-4,rtol=2e-4),
            'first_eval_seconds':first,'warm_eval_seconds':warm,
            'timing_scope':'synthetic observational diagnostic; no controlled throughput or model claim'})
    return cases


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--ownership-window',type=Path)
    p.add_argument('--owner-path',type=Path,action='append');p.add_argument('--repetitions',type=int,default=3)
    args=p.parse_args(argv)
    if args.output.exists():p.error('refusing existing evidence directory')
    if not 1<=args.repetitions<=5:p.error('bounded repetitions1..5 required')
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owners)!=2):p.error('GPU requires existing window and two paired receipts')
    def guard():return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard();source=source_receipt()
    fixture_identity=hashlib.sha256(b''.join(a.tobytes()+w.tobytes() for o in (24,4) for k in ('random','cancellation') for a,w in [fixture(o,k)])).hexdigest()
    receipt={'schema':'mlx2.deepseek-v4-hc-projection-rowlaw.v1','status':'running','source':source,'ownership':ownership,
        'checkpoint':'synthetic-HC16384-projection@'+fixture_identity,'actual_artifact_loaded':False,
        'run_gpu':args.run_gpu,'qualification':False,'ast_claim':False,'performance_claim':False,'cases':[]}
    args.output.mkdir(parents=True);atomic(args.output/'results.json',receipt)
    import mlx.core as mx
    previous=mx.default_device()
    try:
        mx.set_default_device(mx.gpu if args.run_gpu else mx.cpu)
        if args.run_gpu:mx.set_memory_limit(256<<20);mx.set_cache_limit(32<<20);receipt['device_info']=mx.metal.device_info()
        for outputs in (24,4):
            for kind in ('random','cancellation'):
                receipt['cases']+=run_case(outputs,kind,gpu=args.run_gpu,guard=guard,repetitions=args.repetitions)
                atomic(args.output/'results.json',receipt)
        receipt['projection_row_invariance_bitwise_established']=all(c['projection']['bitwise_equal'] for c in receipt['cases'])
        receipt['normalization_row_invariance_bitwise_established']=all(c['normalization_separate']['bitwise_equal'] for c in receipt['cases'])
        receipt['independent_oracle_all_pass']=all(c['independent_fp64_oracle']['passed'] for c in receipt['cases'])
        receipt['diagnostic_completed_finite']=all(c['projection']['numerical']['finite'] and c['normalization_separate']['numerical']['finite'] for c in receipt['cases'])
        receipt['interpretation']='Projection attribution only. A row-law difference is retained, not hidden by a relaxed tolerance; no first-block/full-model causation or AST equivalence established.'
        receipt['status']='completed';guard()
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        atomic(args.output/'results.json',receipt);mx.set_default_device(previous)
    return 0 if receipt['diagnostic_completed_finite'] and receipt['independent_oracle_all_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
