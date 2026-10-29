#!/usr/bin/env python3
"""Q1 component suite callable only inside a caller-owned GPU window.

Import/inventory/refusal checks are CPU-only. run(mx, check_live) creates no
ownership window itself and must be invoked only after user GPU approval.
Provenance: deepseek-v41-plan-q1-components.json. This is not a model benchmark.
"""
import hashlib
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0,str(ROOT/'src'))
SCHEMA='mlx2.ds41-q1-few-row-components.v1'
SCOPES=('fresh_copy_cast_consumer','prepared_kernel_lower_bound')


def inventory():
    """36 finite exact-geometry cases; no device/module math import."""
    shapes={'gate':(5120,2304),'up':(5120,2304),'down':(2304,5120)}
    return tuple(dict(case_id=f'{projection}-m{rows}-{layout}',projection=projection,
                      rows=rows,input_dims=shapes[projection][0],output_dims=shapes[projection][1],
                      layout=layout,source_rows=rows+3 if layout=='mapped' else rows,
                      scopes=SCOPES,coefficient_format='synthetic_affine_q1_group64_fp16_parameters',
                      coefficient_repack_bytes=0)
                 for projection in shapes for rows in range(2,8) for layout in ('compact','mapped'))


def _guard(mx, check_live):
    if not callable(check_live):
        raise ValueError('Q1 components require a caller ownership/live callback')
    if not hasattr(mx,'gpu') or not callable(getattr(mx,'default_device',None)) or mx.default_device()!=mx.gpu:
        raise ValueError('Q1 components require a caller-selected GPU backend')
    # Existing ownership checkers conventionally raise on failure and return
    # None on success; explicit True is accepted too. No other return is proof.
    verdict=check_live()
    if verdict is not None and verdict is not True:
        raise RuntimeError('Q1 component caller ownership/live check refused')


def _classification(reference, candidate):
    ref=np.asarray(reference,dtype=np.float32)
    actual=np.asarray(candidate,dtype=np.float32)
    if ref.shape!=actual.shape:raise ValueError('Q1 comparison output shape changed')
    finite=bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(actual)))
    exact=bool(np.array_equal(ref.view(np.uint32),actual.view(np.uint32)))
    return dict(classification=('raw_exact_sample' if exact and finite else
                                'distinct_numerical_profile' if finite else 'nonfinite_failure'),
                finite=finite,raw_bits_exact=exact,
                differing_words=int(np.count_nonzero(ref.view(np.uint32)!=actual.view(np.uint32))),
                maximum_absolute_delta=float(np.max(np.abs(ref.astype(np.float64)-actual.astype(np.float64)))) if finite else None,
                reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                scope='This finite sample only; neither broad quality nor qualification')


def _f64_errors(x, q, scales, biases, reference, candidate):
    """Four vocabulary/projection columns with independently unpacked Q1 math."""
    k=x.shape[1];n=q.shape[0]
    columns=tuple(sorted(set((0,1,n//2,n-1))))
    indices=np.arange(k,dtype=np.uint32)
    outputs=[]
    for column in columns:
        bits=((q[column,indices//32]>>(indices%32))&np.uint32(1)).astype(np.float64)
        decoded=bits*scales[column,indices//64].astype(np.float64)+biases[column,indices//64].astype(np.float64)
        outputs.append(x.astype(np.float64)@decoded)
    oracle=np.stack(outputs,axis=1)
    result={'columns':columns,'oracle':'independent F64 affine unpack and dot; diagnostic, not an automatic tolerance gate'}
    for label,value in (('reference',reference),('candidate',candidate)):
        selected=np.asarray(value,dtype=np.float32)[:,columns].astype(np.float64)
        if not np.all(np.isfinite(selected)):
            result[label]={'finite':False,'maximum_absolute_error':None,'maximum_relative_error':None}
            continue
        delta=np.abs(selected-oracle)
        result[label]={'finite':True,'maximum_absolute_error':float(delta.max()),
                       'maximum_relative_error':float((delta/np.maximum(np.abs(oracle),1e-12)).max())}
    return result


def _source_hashes():
    paths=('scripts/research/ds41_plan_q1_components.py',
           'src/mlx2/runtime/gguf/source_q1.py','src/mlx2/runtime/gguf/source_q1_few_rows.py')
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in paths}


def _run_case(mx, check_live, spec, *, repetitions, seed):
    from mlx2.runtime.gguf import source_q1 as stock, source_q1_few_rows as few
    _guard(mx,check_live)
    rng=np.random.default_rng(seed)
    m,k,n=spec['rows'],spec['input_dims'],spec['output_dims']
    mapped=spec['layout']=='mapped';source_rows=spec['source_rows']
    # No checkpoint is loaded. These are deterministic synthetic packed codes,
    # FP16 affine parameters and synthetic activation values.
    q_cpu=rng.integers(0,2**32,size=(n,k//32),dtype=np.uint32)
    s_cpu=rng.uniform(.01,.5,size=(n,k//64)).astype(np.float16)
    b_cpu=rng.uniform(-.25,.25,size=(n,k//64)).astype(np.float16)
    source_cpu=rng.normal(0,.25,size=(source_rows,k)).astype(np.float32)
    row_map=(rng.permutation(source_rows)[:m].astype(np.uint32) if mapped
             else np.arange(m,dtype=np.uint32))
    if mapped and np.array_equal(row_map,np.arange(m,dtype=np.uint32)):
        row_map=row_map[::-1].copy()
    # Distinct owners prevent a candidate accidentally borrowing a reference
    # activation/weight result. Coefficient construction is outside timing.
    raw_ref=mx.array(source_cpu,mx.bfloat16);raw_candidate=mx.array(source_cpu.copy(),mx.bfloat16)
    q_ref=mx.array(q_cpu,mx.uint32);q_candidate=mx.array(q_cpu.copy(),mx.uint32)
    s_ref=mx.array(s_cpu,mx.float16);s_candidate=mx.array(s_cpu.copy(),mx.float16)
    b_ref=mx.array(b_cpu,mx.float16);b_candidate=mx.array(b_cpu.copy(),mx.float16)
    prepared_ref=raw_ref.astype(mx.float32);prepared_candidate=raw_candidate.astype(mx.float32)
    device_map=mx.array(row_map,mx.uint32) if mapped else None
    selected_ref=prepared_ref[device_map] if mapped else prepared_ref
    p=few.plan(spec['projection'],m,mapped=mapped,source_rows=source_rows)
    mx.eval(raw_ref,raw_candidate,q_ref,q_candidate,s_ref,s_candidate,b_ref,b_candidate,
            prepared_ref,prepared_candidate,selected_ref,*([device_map]if mapped else []))
    _guard(mx,check_live)

    def fresh_reference():
        value=mx.asarray(raw_ref,copy=True).astype(mx.float32)
        if mapped:value=value[mx.array(row_map,mx.uint32)]
        return stock.matmul(value,q_ref,s_ref,b_ref)

    def fresh_candidate():
        value=mx.asarray(raw_candidate,copy=True).astype(mx.float32)
        return few.project(value,q_candidate,s_candidate,b_candidate,
                           projection=spec['projection'],rows=row_map if mapped else None)

    def prepared_reference():
        return stock.kernel()(inputs=[selected_ref,q_ref,s_ref,b_ref],template=[('T',mx.float32)],
            grid=(((n+7)//8)*64,m,1),threadgroup=(64,1,1),
            output_shapes=[(m*n,)],output_dtypes=[mx.float32])[0].reshape(m,n)

    def prepared_candidate_kernel():
        inputs=[prepared_candidate]+([device_map]if mapped else [])+[q_candidate,s_candidate,b_candidate]
        return few._kernel(m,mapped)(inputs=inputs,grid=p['grid'],threadgroup=p['threadgroup'],
            output_shapes=[p['output_shape']],output_dtypes=[mx.float32])[0]

    builders={'fresh_copy_cast_consumer':{'reference':fresh_reference,'candidate':fresh_candidate},
              'prepared_kernel_lower_bound':{'reference':prepared_reference,'candidate':prepared_candidate_kernel}}
    warmup=[]
    for scope in SCOPES:
        for arm in ('reference','candidate'):
            _guard(mx,check_live);start=time.perf_counter_ns()
            value=builders[scope][arm]();mx.eval(value)
            warmup.append(dict(scope=scope,arm=arm,wall_seconds=(time.perf_counter_ns()-start)*1e-9,
                               interpretation='first graph/build/compile/evaluate envelope; excluded from steady samples'))
    samples=[];randomizer=random.Random(seed)
    for scope in SCOPES:
        initial_order=randomizer.randrange(2)
        for repetition in range(repetitions):
            arms=['reference','candidate']
            if (initial_order+repetition)%2:arms.reverse()
            for order,arm in enumerate(arms):
                _guard(mx,check_live);start=time.perf_counter_ns()
                value=builders[scope][arm]();mx.eval(value)
                elapsed=(time.perf_counter_ns()-start)*1e-9
                if elapsed<=0:raise RuntimeError('Q1 component clock did not advance')
                samples.append(dict(scope=scope,repetition=repetition,order=order,arm=arm,wall_seconds=elapsed))
    comparisons={};timings={}
    for scope in SCOPES:
        _guard(mx,check_live)
        reference=builders[scope]['reference']();candidate=builders[scope]['candidate']()
        mx.eval(reference,candidate)
        ref_cpu=np.asarray(reference);candidate_cpu=np.asarray(candidate)
        comparisons[scope]=_classification(ref_cpu,candidate_cpu)
        if scope=='fresh_copy_cast_consumer':
            source_values=np.asarray(prepared_ref)[row_map]
            comparisons[scope]['sampled_f64']=_f64_errors(source_values,q_cpu,s_cpu,b_cpu,ref_cpu,candidate_cpu)
        by_arm={arm:[r['wall_seconds']for r in samples if r['scope']==scope and r['arm']==arm]
                for arm in ('reference','candidate')}
        pairs=[]
        for repetition in range(repetitions):
            row={r['arm']:r['wall_seconds']for r in samples
                 if r['scope']==scope and r['repetition']==repetition}
            pairs.append(row['reference']/row['candidate'])
        timings[scope]={'reference_median_seconds':statistics.median(by_arm['reference']),
                        'candidate_median_seconds':statistics.median(by_arm['candidate']),
                        'median_paired_ratio':statistics.median(pairs),
                        'ratio_of_independent_medians':statistics.median(by_arm['reference'])/statistics.median(by_arm['candidate']),
                        'interpretation':'complete scope wall includes lazy host evaluation/encoding/GPU/completion; not pure GPU time'}
    _guard(mx,check_live)
    fixture_hashes={name:hashlib.sha256(array.tobytes()).hexdigest() for name,array in
                    (('qweight',q_cpu),('scales',s_cpu),('biases',b_cpu),
                     ('source_input_f32',source_cpu),('row_map',row_map))}
    return {'case':spec,'seed':seed,'profile':few.PROFILE,'fixture_hashes':fixture_hashes,
            'warmup':warmup,'raw_samples':samples,
            'timings':timings,'comparisons':comparisons,
            'independent_input_and_coefficient_owners':True,'row_map':row_map.tolist(),
            'scope_contract':{'fresh_copy_cast_consumer':'fresh BF16 input owner copy, F32 cast, mapped row upload plus stock gather or candidate mapped consumer, and public projection wrapper; QAT/FFN/leases excluded',
                              'prepared_kernel_lower_bound':'admitted prepared F32 inputs/coefficients/map and stock pre-gather outside timing; direct kernels only, not a serving cost'},
            'synthetic_inputs':True,'checkpoint_loaded':False,'model_speed_claim':False}


def run(mx, check_live, *, repetitions=7, seed=410126):
    """Execute only after outer controller grants GPU and owns paired leases.

    check_live must raise on any ownership/reserve/source/thermal violation.
    The controller owns memory caps, atomic receipts, process cleanup and yield.
    """
    _guard(mx,check_live)
    if type(repetitions)is not int or not 1<=repetitions<=21 or type(seed)is not int:
        raise ValueError('Q1 components require bounded repetitions and an explicit integer seed')
    before=_source_hashes()
    results=[]
    for index,spec in enumerate(inventory()):
        results.append(_run_case(mx,check_live,spec,repetitions=repetitions,seed=seed+index))
    if _source_hashes()!=before:raise RuntimeError('Q1 component sources changed during execution')
    fixture_binding=tuple((r['case']['case_id'],tuple(sorted(r['fixture_hashes'].items())))for r in results)
    return {'schema':SCHEMA,'artifact':'synthetic-ds41-q1-components@'+hashlib.sha256(repr(fixture_binding).encode()).hexdigest(),
            'source_hashes':before,'source_revision':'caller must bind exact commit and complete frozen source snapshot',
            'cases':results,'case_count':len(results),'repetitions':repetitions,'seed':seed,
            'GPU_component_executed':True,'checkpoint_loaded':False,'qualified':False,
            'model_speed_claim':False,'ownership':'caller-owned, check_live between cells and samples',
            'numerical_scope':'raw sample equality or explicit distinct-profile diagnostics; no model quality conclusion'}


if __name__=='__main__':
    raise SystemExit('This suite must be imported by an approved, paired-ownership GPU controller; it has no standalone runner.')
