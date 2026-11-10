#!/usr/bin/env python3
"""Actual full-head coefficients, synthetic activations, caller-owned GPU only.

Import and plan() perform no artifact reads or MLX import. This module never
starts a model, server or GPU ownership window. No CLI executes it.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import importlib.util
from pathlib import Path
import random
import statistics
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=Path(str(Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Source'))
FINGERPRINT='c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb'
CHECKPOINT='DeepSeek-V4.1-Flash-1017-Source@'+FINGERPRINT
SHAPE=(129280,5120)
ARRAY_BUDGET=6<<30
CELLS=('normal','alternating_cancellation','signed_zero_tiny','sparse_spikes')
_QUARANTINE=[]


class HeadSuiteError(RuntimeError):
    def __init__(self,message,partial_result):
        super().__init__(message);self.partial_result=partial_result


def plan():
    words=SHAPE[0]*SHAPE[1]
    retained=words*6
    if retained+16*SHAPE[0]*4+(4*SHAPE[1]*4)>ARRAY_BUDGET:
        raise ValueError('Full-head diagnostic exceeds its component array bound')
    return dict(schema='mlx2.ds41-plan-head-components-plan.v1',checkpoint=CHECKPOINT,
        model_path=str(MODEL_PATH),weight_shape=SHAPE,tensor='head.weight',cells=CELLS,
        case_count=len(CELLS)*2,rows=1,physical_width=1,output_shape=(1,1,SHAPE[0]),
        direct_bf16_bytes=words*2,reference_f32_bytes=words*4,
        diagnostic_retained_coefficient_bytes=retained,array_budget_bytes=ARRAY_BUDGET,
        diagnostic_companion_bytes=words*2,production_companion_bytes=0,
        maximum_host_tensor_read_bytes=words*2,
        synthetic_activations=True,model_loaded=False,qualified=False,model_speed_claim=False,
        clock='fresh prepared-consumer graph plus mx.eval; one-time head F32 conversion and warm compile excluded and reported')


def _guard(mx,callback):
    if not callable(callback):raise ValueError('Head components require caller ownership/live callback')
    if not hasattr(mx,'gpu') or not callable(getattr(mx,'default_device',None)) or mx.default_device()!=mx.gpu:
        raise ValueError('Head components require caller-selected GPU backend')
    if callback() is not True:raise RuntimeError('Head ownership/live callback must return True')


def validate_artifact(artifact,model_path=MODEL_PATH):
    """CPU metadata admission; no tensor payload read."""
    if artifact.get('identity',{}).get('fingerprint')!=FINGERPRINT:
        raise ValueError('Actual head artifact fingerprint mismatch')
    if Path(artifact.get('root','')).resolve()!=Path(model_path).resolve():
        raise ValueError('Actual head artifact path mismatch')
    name=artifact.get('aliases',{}).get('output.weight')
    row=artifact.get('tensors',{}).get(name,{})
    if (name!='head.weight' or row.get('dtype')!='BF16' or tuple(row.get('shape',()))!=SHAPE
            or row.get('size')!=SHAPE[0]*SHAPE[1]*2 or row.get('file') not in artifact.get('seals',{})):
        raise ValueError('Actual head must be the complete original BF16 tensor')
    return name,row


def _activation(cell,seed):
    rng=np.random.default_rng(seed);k=SHAPE[1]
    if cell=='normal':x=rng.normal(0,.2,k).astype(np.float32)
    elif cell=='alternating_cancellation':
        x=np.where(np.arange(k)%2,-.125,.125).astype(np.float32);x[::17]+=np.float32(1e-6)
    elif cell=='signed_zero_tiny':
        x=np.zeros(k,np.float32);x[1::2]=-0.;x[::31]=np.float32(2**-126);x[1::31]=-np.float32(2**-126)
    elif cell=='sparse_spikes':
        x=np.zeros(k,np.float32);x[[0,63,64,k//2,k-1]]=[1.,-.5,.25,-1.,.125]
    else:raise ValueError('Unknown synthetic head activation')
    return x.reshape(1,1,k)


def _source_hashes():
    paths=('scripts/research/ds41_plan_head_components.py',
           'src/mlx2/runtime/models/deepseek_v41_head_experiment.py',
           'src/mlx2/runtime/gguf/source_dense.py',
           'src/mlx2/adapters/deepseek_v41_source_artifact.py',
           'scripts/research/ds41_plan_attention_components.py')
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in paths}


def _compare(reference,candidate,activation,sample_weights,columns):
    a=np.asarray(reference,np.float32).reshape(-1);b=np.asarray(candidate,np.float32).reshape(-1)
    finite=bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    exact=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))
    oracle=sample_weights.astype(np.float64)@activation.reshape(-1).astype(np.float64)
    errors={}
    for label,value in (('reference',a),('candidate',b)):
        selected=value[list(columns)].astype(np.float64)
        delta=np.abs(selected-oracle)
        errors[label]={'finite':bool(np.all(np.isfinite(selected))),
            'maximum_absolute_error':float(delta.max()) if np.all(np.isfinite(delta)) else None,
            'maximum_relative_error':float((delta/np.maximum(np.abs(oracle),1e-12)).max()) if np.all(np.isfinite(delta)) else None}
    return dict(classification='raw_exact_sample' if finite and exact else 'distinct_numerical_profile' if finite else 'nonfinite_failure',
        finite=finite,raw_bits_exact=exact,differing_words=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
        maximum_absolute_delta=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if finite else None,
        reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        sampled_f64={'columns':columns,'oracle':'actual BF16 coefficients widened independently to F64; diagnostic only','errors':errors})


def run(mx,check_live,*,model_path=MODEL_PATH,repetitions=7,seed=410126):
    _guard(mx,check_live)
    if type(repetitions)is not int or not 1<=repetitions<=21 or type(seed)is not int:
        raise ValueError('Head components require bounded repetitions and integer seed')
    from mlx2.adapters import deepseek_v41_source_artifact as source
    from mlx2.runtime.models import deepseek_v41_head_experiment as head
    # Resolve the reviewed metadata helper from the same frozen root rather
    # than whichever checkout supplies a namespace-package import on sys.path.
    helper_spec=importlib.util.spec_from_file_location('_head_source_binding',
        ROOT/'scripts/research/ds41_plan_attention_components.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    spec=plan();hashes=_source_hashes();revision,snapshot=helper.source_binding(ROOT)
    result={'schema':'mlx2.ds41-plan-head-components.v1','plan':spec,'checkpoint':CHECKPOINT,
        'source_revision':revision,'source_snapshot':snapshot,'source_hashes':hashes,
        'cells':[],'seed':seed,'repetitions':repetitions,'model_loaded':False,
        'synthetic_activations':True,'qualified':False,'model_speed_claim':False}
    owners=[];pending=[];error=None;cleanup=None
    try:
        _guard(mx,check_live);start=time.perf_counter_ns()
        artifact=source.inspect_artifact(model_path);name,row=validate_artifact(artifact,model_path)
        result['artifact_head']={'name':name,'extent':dict(row),'seal':dict(artifact['seals'][row['file']]),
            'artifact_fingerprint':artifact['identity']['fingerprint'],'payload_read_names':[name]}
        _guard(mx,check_live)
        raw=source.read_tensors(artifact,(name,))[name];owners.append(raw)
        if raw.dtype!=np.dtype('<u2') or raw.shape!=SHAPE:raise ValueError('Actual head read returned unexpected BF16 geometry')
        result['artifact_head']['payload_sha256']=hashlib.sha256(raw).hexdigest()
        columns=(0,1,SHAPE[0]//2,SHAPE[0]-1)
        samples=(raw[list(columns)].astype(np.uint32)<<16).view(np.float32)
        result['coefficient_read_and_hash_seconds']=(time.perf_counter_ns()-start)*1e-9
        _guard(mx,check_live);start=time.perf_counter_ns()
        weight=mx.array(raw,mx.uint16).view(mx.bfloat16);owners.append(weight)
        direct=head.prepare(weight,profile=head.DIRECT,checkpoint=CHECKPOINT,source_revision=revision)
        reference=head.prepare(weight,profile=head.REFERENCE,checkpoint=CHECKPOINT,source_revision=revision)
        owners.extend((direct,reference));mx.eval(direct.coefficients,reference.coefficients)
        result['coefficient_upload_and_once_F32_preparation_seconds']=(time.perf_counter_ns()-start)*1e-9
        owners=[o for o in owners if o is not raw];del raw
        for index,cell in enumerate(CELLS):
            _guard(mx,check_live);cpu=_activation(cell,seed+index)
            x_ref=mx.array(cpu,mx.float32);x_direct=mx.array(cpu.copy(),mx.float32)
            mx.eval(x_ref,x_direct)
            builders={'reference':lambda:head.project(x_ref,reference),'direct_bf16_f32':lambda:head.project(x_direct,direct)}
            record={'cell':cell,'activation_sha256':hashlib.sha256(cpu.tobytes()).hexdigest(),'warmup':[],'raw_samples':[]}
            result['cells'].append(record)
            for arm in builders:
                _guard(mx,check_live);start=time.perf_counter_ns()
                value=builders[arm]();pending.append(value);mx.eval(value);pending.clear()
                record['warmup'].append({'arm':arm,'wall_seconds':(time.perf_counter_ns()-start)*1e-9,
                    'scope':'first graph/compile/evaluate envelope, excluded from steady samples'})
            initial=random.Random(seed+index).randrange(2)
            for repetition in range(repetitions):
                arms=list(builders)
                if (initial+repetition)%2:arms.reverse()
                for order,arm in enumerate(arms):
                    _guard(mx,check_live);start=time.perf_counter_ns()
                    value=builders[arm]();pending.append(value);mx.eval(value);pending.clear()
                    record['raw_samples'].append(dict(repetition=repetition,order=order,arm=arm,
                        wall_seconds=(time.perf_counter_ns()-start)*1e-9))
            _guard(mx,check_live)
            a=builders['reference']();b=builders['direct_bf16_f32']();pending.extend((a,b));mx.eval(a,b)
            record['comparison']=_compare(a,b,cpu,samples,columns);pending.clear()
            by_arm={arm:[v['wall_seconds']for v in record['raw_samples']if v['arm']==arm]for arm in builders}
            ratios=[]
            for repetition in range(repetitions):
                pair={v['arm']:v['wall_seconds']for v in record['raw_samples']if v['repetition']==repetition}
                ratios.append(pair['reference']/pair['direct_bf16_f32'])
            record['timing']={'reference_median_seconds':statistics.median(by_arm['reference']),
                'direct_median_seconds':statistics.median(by_arm['direct_bf16_f32']),
                'median_paired_ratio':statistics.median(ratios),
                'ratio_of_independent_medians':statistics.median(by_arm['reference'])/statistics.median(by_arm['direct_bf16_f32']),
                'scope':'prepared full-head consumer host+encode+GPU+completion wall; no per-token weight cast; no model extrapolation'}
        _guard(mx,check_live)
        for filename,seal in artifact['seals'].items():source.checked_file(artifact['root'],filename,seal)
        if _source_hashes()!=hashes:raise RuntimeError('Head probe implementation changed during execution')
        result['source_and_artifact_unchanged']=True
    except BaseException as exc:
        error=exc;result['error']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        try:
            mx.synchronize();pending.clear();owners.clear();cleanup='synchronized_and_released'
        except BaseException as exc:
            _QUARANTINE.append((owners,pending));cleanup='failed_synchronize_owners_quarantined'
            if error is None:error=exc
        result['cleanup']=cleanup
    if error is not None:raise HeadSuiteError('Actual-head component aborted: '+str(error),result)from error
    return result


if __name__=='__main__':
    raise SystemExit('Import run(mx,check_live) from an approved paired-ownership controller; no standalone execution.')
