#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare DS4.1 component units on CPU; execute only with explicit --run-gpu.

Prepare performs no hardware, telemetry, ownership or MLX calls. Execution
requires the existing CPG/radio and paired-lock wrapper plus mandatory HID
thermal gating. Provenance: deepseek-v41-plan-components-controller.json.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research'),str(ROOT/'scripts')]
from mlx2.experimental.ds41_performance import runtime_source_paths,source_snapshot,write_new_json

SCHEMA='mlx2.ds41-plan-owned-components.v1'
MEMORY_LIMIT=6<<30
HOST_INITIAL=15<<30
HOST_RESERVE=9<<30
PROFILE='m5_hid_nominal_v1'
POLICY=dict(required_thermal_state=0,max_die_temperature_c=60.,
            max_hid_battery_temperature_c=40.,max_temperature_delta_c=.5,
            required_samples=3,max_sample_age_seconds=10.,live_cache_seconds=1.)
SUITES={'attention':'ds41_plan_attention_components','q1':'ds41_plan_q1_components',
        'head':'ds41_plan_head_components','resident':'ds41_native_resource_components',
        'bank':'ds41_resident_bank_components','expert':'ds41_resident_kernel_components',
        'prefill_bank':'ds41_resident_prefill_components',
        'prefill_wide':'ds41_resident_prefill_wide_components',
        'prefill_persistent':'ds41_resident_prefill_persistent_components'}
PURPOSES=('functional','performance')
PERFORMANCE_ONLY_GATES=frozenset({'hid_three_sample_stability','swap_stable'})


def blocking_gates(gates,purpose):
    if purpose not in PURPOSES:raise ValueError('Unknown component purpose')
    return [k for k,v in gates.items()if v is not True
            and not(purpose=='functional'and k in PERFORMANCE_ONLY_GATES)]


def canonical(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def inventory(suite):
    if suite not in SUITES:raise ValueError('Unknown component suite')
    if suite=='resident':
        return dict(resources=15,maximum_staging_bytes=65536,
                    good_batches=3,status_cases=['missing','invalid','stale_generation','stale_serial'],
                    overlapping_generations=2,
                    modes=['tracked_staging_v1','direct_array_v1_if_tracked'],serving_backend=False)
    module=importlib.import_module(SUITES[suite])
    return canonical(module.plan() if suite in ('attention','head','bank','expert','prefill_bank','prefill_wide','prefill_persistent') else list(module.inventory()))


def component_checkpoint(suite,plan,plan_sha):
    return plan['checkpoint'] if suite in ('head','bank','expert','prefill_bank','prefill_wide','prefill_persistent') else 'synthetic-ds41-plan-'+suite+'@'+plan_sha


def read_json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>8<<20:
        raise ValueError('Unsafe component metadata file')
    return json.loads(path.read_text())


def prepare(out,suite,*,purpose='performance'):
    """Freeze explicit source inputs, preserving any earlier attempt."""
    if purpose not in PURPOSES:raise ValueError('Unknown component purpose')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve existing component evidence directory')
    plan=inventory(suite)
    paths=runtime_source_paths(ROOT,additional=(
        ['native/ds41_resident_pilot/'+name for name in
         ('pilot.cpp','lookup.metal','kernels.h.in','CMakeLists.txt')]
        if suite=='resident' else ()))
    snapshot=source_snapshot(ROOT,paths)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if re.fullmatch('[0-9a-f]{40}',revision) is None:raise ValueError('Invalid source commit')
    out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
    write_new_json(out/'source-inputs.json',snapshot)
    for row in snapshot['files']:
        target=source/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/row['path'],target)
    if source_snapshot(source,paths)!=snapshot or source_snapshot(ROOT,paths)!=snapshot:
        raise ValueError('Source changed while freezing component unit')
    metadata=dict(source_revision=revision,source_sha256=snapshot['sha256'])
    write_new_json(source/'desktop-source.json',metadata)
    plan_sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    spec=dict(schema=SCHEMA,suite=suite,purpose=purpose,suite_inventory=plan,out=str(out),job_id=out.name,
        checkpoint=component_checkpoint(suite,plan,plan_sha),
        source_revision=revision,source_sha256=snapshot['sha256'],
        memory_limit_bytes=MEMORY_LIMIT,host_initial_floor_bytes=HOST_INITIAL,
        host_live_reserve_bytes=HOST_RESERVE,numerical_environment={'MLX_ENABLE_TF32':'0'},
        thermal_profile=PROFILE,thermal_policy=POLICY,mcp_url='http://127.0.0.1:8766/mcp',
        executed=False,gpu_authorization='pending_user_approval',rate_eligible=False,
        qualification=False,performance_qualified=False,model_loaded=False)
    if suite=='resident':
        binary=ROOT/'.local-ds41-implementation/native-resource-build/_ds41_resident_pilot.cpython-312-darwin.so'
        if binary.is_symlink()or not binary.is_file():raise ValueError('Native pilot must be built first')
        native=out/'native';native.mkdir();target=native/binary.name
        shutil.copyfile(binary,target)
        digest=hashlib.sha256(binary.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Native pilot binary changed during freeze')
        spec['native_binary']={'name':binary.name,'sha256':digest}
    write_new_json(out/'spec.json',spec)
    return spec


def validate_spec(out):
    out=Path(out).resolve();spec=read_json(out/'spec.json')
    fixed=dict(schema=SCHEMA,out=str(out),job_id=out.name,memory_limit_bytes=MEMORY_LIMIT,
        host_initial_floor_bytes=HOST_INITIAL,host_live_reserve_bytes=HOST_RESERVE,
        numerical_environment={'MLX_ENABLE_TF32':'0'},thermal_profile=PROFILE,thermal_policy=POLICY,
        mcp_url='http://127.0.0.1:8766/mcp',executed=False,gpu_authorization='pending_user_approval',
        rate_eligible=False,qualification=False,performance_qualified=False,model_loaded=False)
    if any(spec.get(k)!=v or type(spec.get(k))is not type(v)for k,v in fixed.items()):
        raise ValueError('Invalid sealed component bounds/policy')
    if spec.get('purpose','performance')not in PURPOSES:
        raise ValueError('Invalid sealed component purpose')
    if spec.get('suite')not in SUITES or spec.get('suite_inventory')!=inventory(spec['suite']):
        raise ValueError('Component suite inventory changed')
    plan_sha=hashlib.sha256(json.dumps(spec['suite_inventory'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if spec.get('checkpoint')!=component_checkpoint(spec['suite'],spec['suite_inventory'],plan_sha):
        raise ValueError('Component fixture-plan artifact binding changed')
    if re.fullmatch('[0-9a-f]{40}',spec.get('source_revision',''))is None:
        raise ValueError('Invalid component source revision')
    verify_source(out,spec)
    if spec['suite']=='resident':
        native=spec.get('native_binary',{});name=native.get('name','')
        if name!='_ds41_resident_pilot.cpython-312-darwin.so':raise ValueError('Unexpected native module')
        binary=out/'native'/name
        if binary.is_symlink()or hashlib.sha256(binary.read_bytes()).hexdigest()!=native.get('sha256'):
            raise ValueError('Native pilot binary changed')
    return spec


def verify_source(out,spec):
    out=Path(out);source=out/'source';snapshot=read_json(out/'source-inputs.json')
    names=[row['path']for row in snapshot['files']]
    if (snapshot.get('sha256')!=spec.get('source_sha256')
            or source_snapshot(source,names)!=snapshot):
        raise ValueError('Frozen component source changed')
    meta=read_json(source/'desktop-source.json')
    if meta!={'source_revision':spec['source_revision'],'source_sha256':spec['source_sha256']}:
        raise ValueError('Frozen component metadata changed')
    expected=set(names)|{'desktop-source.json'}
    actual={str(p.relative_to(source))for p in source.rglob('*')
            if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
    if actual!=expected:raise ValueError('Frozen component source inventory changed')
    return True


def dependencies():
    """Execution-only imports; never called by CPU prepare."""
    from probe_ds41_native_mx import host_admission,prelease_admission,monitor_child
    from probe_deepseek_v4_decode_compare import sample_environment,thermal_failures,foreign_waiters,gate_samples
    from mlx2.desktop_campaign_worker import atomic,check_ownership
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    def mlx():
        import mlx.core as mx
        return mx
    def suite_run(name,mx,callback):
        return importlib.import_module(SUITES[name]).run(mx,callback)
    return SimpleNamespace(host=host_admission,prelease=prelease_admission,monitor=monitor_child,
        sample=sample_environment,failures=thermal_failures,waiters=foreign_waiters,gates=gate_samples,
        atomic=atomic,ownership=check_ownership,lease=DesktopCPGLease,spawn=spawn,
        mlx=mlx,suite_run=suite_run,sleep=time.sleep,clock=time.monotonic,wall=time.time)


def checked_sample(d):
    value=d.sample(PROFILE);thermal=value.get('thermal',{})
    if (value.get('thermal_profile')!=PROFILE or value.get('thermal_ok')is not True
            or type(thermal.get('thermal_state'))is not int or thermal['thermal_state']!=0
            or any(type(thermal.get(k))not in (int,float)or not math.isfinite(thermal[k])
                   for k in ('die_max_c','hid_battery_temperature_c'))
            or any(d.failures(thermal,PROFILE).values())
            or value.get('foreign_model_processes')
            or any(type(value.get(k))is not int or value[k]<0
                   for k in ('Swapins','Swapouts','swap_used_bytes'))):
        raise RuntimeError('Component thermal/foreign/swap telemetry refused')
    return value


def three_samples(d,purpose='performance'):
    rows=[]
    for i in range(3):
        if i:d.sleep(.25)
        rows.append(checked_sample(d))
    gates=d.gates(rows,rows,now=d.wall())
    failures=blocking_gates(gates,purpose)
    if failures:
        failure=RuntimeError('Component admission gates failed: '+','.join(failures)+'; yield without cooling wait')
        failure.admission_samples=rows
        failure.admission_gates=gates
        raise failure
    return rows


class LiveGuard:
    def __init__(self,d,spec,out,before):
        self.d,self.spec,self.out,self.before=d,spec,Path(out),before
        self.rows=list(before);self.last=None;self.owner=None
        self.purpose=spec.get('purpose','performance');self.performance_failures=[]

    def __call__(self,force=False):
        d=self.d;now=d.clock()
        if not force and self.last is not None and 0<=now-self.last<1.:return True
        self.owner=d.ownership(self.spec)
        host=d.host(HOST_RESERVE)
        sample=checked_sample(d)
        if d.waiters(self.owner['desktop_lease']['cpg_session']):
            raise RuntimeError('Queued foreign GPU waiter; component yields')
        self.rows.append(sample)
        # Stable three-sample brackets belong before and after work. Mixing an
        # old bracket with a fresh sample after compilation gives a false stale
        # bracket. Live samples retain nominal ceilings, recency, environment
        # and swap gates without inventing a three-sample stability verdict.
        baseline=self.before[0]
        swapped=any(sample[k]!=baseline[k]for k in ('Swapins','Swapouts','swap_used_bytes'))
        if swapped:self.performance_failures.append(dict(gate='swap_stable',time_unix=sample['time_unix'],
            deltas={k:sample[k]-baseline[k]for k in ('Swapins','Swapouts','swap_used_bytes')}))
        if (not 0<=d.wall()-sample.get('temperature_sampled_at_unix',sample['time_unix'])<=10.
                or sample['numerics_environment']!=baseline['numerics_environment']
                or(swapped and self.purpose=='performance')):
            raise RuntimeError('Component live recency/environment/swap gate failed')
        d.atomic(self.out/'live-checks.json',dict(samples=self.rows,host_available_bytes=host.available_bytes,
            cache_max_seconds=1.,ownership_checked=True,purpose=self.purpose,
            performance_ineligible_events=self.performance_failures,
            rate_eligible=False,timing_scope='diagnostic_only'if self.purpose=='functional'else'component_timing'))
        self.last=d.clock()
        return True


def suite_verdict(suite,result,spec):
    if suite in ('prefill_bank','prefill_wide','prefill_persistent'):
        cases=result.get('cases',[]);controls=result.get('refusals',[])
        expected=spec['suite_inventory']
        complete=(result.get('passed')is True and result.get('case_count')==len(cases)+len(controls)
            and [x.get('label')for x in cases]==expected['valid_cases']
            and [x.get('label')for x in controls]==expected['refusal_cases']
            and result.get('checkpoint')==spec['checkpoint'])
        return complete,complete and all(c.get('passed')is True for c in cases+controls)
    if suite in ('bank','expert'):
        cases=result.get('cases',[]);controls=result.get('status_cases',[])
        expected=spec['suite_inventory']['mixtures']
        complete=(result.get('passed')is True and result.get('case_count')==len(expected)+4
            and len(cases)==len(expected) and len(controls)==4
            and [x.get('mixture')for x in cases]==spec['suite_inventory']['mixtures']
            and {x.get('label')for x in controls}=={'duplicate','invalid','generation','missing'}
            and result.get('checkpoint')==spec['checkpoint'])
        if suite=='expert':
            variants=spec['suite_inventory']['kernel_variants']
            complete=complete and all([v.get('variant')for v in c.get('kernel_variants',[])]==variants
                and all(v.get('passed')is True and len(v.get('timing',{}).get('raw_pairs',[]))==spec['suite_inventory']['timing_pairs']
                    for v in c['kernel_variants'])for c in cases)
            complete=complete and all([v.get('variant')for v in c.get('kernel_variants',[])]==variants
                and all(v.get('passed')is True for v in c['kernel_variants'])for c in controls)
        return complete,complete and all(c.get('passed')is True for c in cases+controls)
    if suite=='resident':
        cases=result.get('cases',[])
        complete=(result.get('passed')is True and result.get('case_count')==len(cases)
                  and len(cases)in(9,16)and result.get('serving_backend')is False
                  and sum(c.get('label')=='overlapping_generation'for c in cases)==2)
        return complete,complete and all(c.get('passed')is True for c in cases)
    if suite=='head':
        cells=result.get('cells',[])
        complete=(result.get('source_and_artifact_unchanged')is True
            and result.get('cleanup')=='synchronized_and_released'
            and [x.get('cell')for x in cells]==spec['suite_inventory']['cells']
            and all(len(x.get('raw_samples',[]))==14 for x in cells))
        return complete,complete and all(x.get('comparison',{}).get('finite')is True for x in cells)
    if suite=='attention':
        complete=(result.get('complete')is True and
                  len(result.get('cases',[]))==spec['suite_inventory']['case_count'])
        return complete,result.get('numerical_validity_passed')is True
    cases=result.get('cases',[]);expected=spec['suite_inventory']
    complete=(result.get('case_count')==len(expected)and len(cases)==len(expected)
        and canonical([x.get('case')for x in cases])==expected)
    finite=complete and all(set(x.get('comparisons',{}))=={'fresh_copy_cast_consumer','prepared_kernel_lower_bound'}
        and all(v.get('finite')is True for v in x['comparisons'].values())for x in cases)
    # Finite raw mismatch remains a distinct-profile observation, not quality failure.
    return complete,finite


def unit(out,*,run_gpu=False,d=None):
    if run_gpu is not True:raise PermissionError('Explicit user-approved --run-gpu is required')
    out=Path(out).resolve();spec=validate_spec(out)
    purpose=spec.get('purpose','performance')
    attempt=read_json(out/'run-attempt.json')
    if attempt!={'source_sha256':spec['source_sha256'],'run_gpu':True,'purpose':purpose}:
        raise PermissionError('Component unit lacks parent execution authorization receipt')
    if os.environ.get('MLX_ENABLE_TF32')!='0':raise ValueError('Component requires TF32 disabled before MLX import')
    d=dependencies()if d is None else d
    report=dict(schema=SCHEMA,checkpoint=spec['checkpoint'],suite=spec['suite'],
        purpose=purpose,timing_scope='diagnostic_only'if purpose=='functional'else'thermally_gated_components',
        source_revision=spec['source_revision'],source_sha256=spec['source_sha256'],passed=False,
        methodological_passed=False,numerical_passed=False,qualification=False,
        performance_qualified=False,model_loaded=False,rate_eligible=False)
    mx=None
    try:
        report['ownership_before']=d.ownership(spec)
        verify_source(out,spec);d.host(HOST_INITIAL)
        before=three_samples(d,purpose);report['before']=before
        report['initial_performance_gates']=d.gates(before,before,now=d.wall())
        guard=LiveGuard(d,spec,out,before);guard(force=True)
        mx=d.mlx();mx.set_default_device(mx.gpu);mx.set_memory_limit(MEMORY_LIMIT);mx.set_cache_limit(128<<20)
        result=d.suite_run(spec['suite'],mx,guard)
        result=dict(result,controller_purpose=purpose,rate_eligible=False,performance_qualified=False,
                    controller_timing_scope=report['timing_scope'])
        write_new_json(out/'component-result.json',result)
        mx.synchronize();guard(force=True)
        report['peak_mlx_bytes']=int(mx.get_peak_memory())
        if report['peak_mlx_bytes']>MEMORY_LIMIT:raise MemoryError('Component exceeded sealed6GiB allocation ceiling')
        after=three_samples(d,purpose);report['after']=after
        gates=d.gates(before,after,now=d.wall());report['performance_gates']=gates
        report['live_performance_ineligible_events']=guard.performance_failures
        report['performance_ineligible_reasons']=(['purpose_functional']if purpose=='functional'else[])+[k for k,v in gates.items()if v is not True]
        report['component_timing_eligible']=purpose=='performance'and all(gates.values())and not guard.performance_failures
        if blocking_gates(gates,purpose):raise RuntimeError('Post-component thermal/contamination gate failed')
        report['ownership_after']=d.ownership(spec);d.host(HOST_RESERVE);verify_source(out,spec)
        method,numerics=suite_verdict(spec['suite'],result,spec)
        report.update(methodological_passed=method,numerical_passed=numerics,
                      passed=method and numerics,source_unchanged=True)
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        if hasattr(error,'admission_samples'):
            report['admission_samples']=error.admission_samples
            report['admission_gates']=error.admission_gates
        if hasattr(error,'partial_result'):write_new_json(out/'partial-component-result.json',error.partial_result)
        write_new_json(out/'error.json',report['error'])
    finally:
        if mx is not None:
            try:
                mx.synchronize();report['peak_mlx_bytes']=int(mx.get_peak_memory())
                if report['peak_mlx_bytes']>MEMORY_LIMIT:report.update(passed=False,memory_ceiling_exceeded=True)
                gc.collect();mx.clear_cache()
            except BaseException as error:
                report.update(passed=False,cleanup_error={'type':type(error).__name__,'message':str(error)})
        try:report['source_unchanged']=verify_source(out,spec)
        except BaseException as error:report.update(passed=False,source_unchanged=False,source_error=str(error))
        write_new_json(out/'results.json',report)
    return 0 if report['passed']else 1


def run_prepared(out,*,run_gpu=False,d=None):
    if run_gpu is not True:raise PermissionError('Explicit user-approved --run-gpu is required')
    out=Path(out).resolve();spec=validate_spec(out)
    purpose=spec.get('purpose','performance')
    write_new_json(out/'run-attempt.json',{'source_sha256':spec['source_sha256'],'run_gpu':True,'purpose':purpose})
    d=dependencies()if d is None else d
    child=lease=None;result=1
    try:
        d.host(HOST_INITIAL);before=three_samples(d,purpose);waiters=d.waiters('')
        d.prelease(before[-1],waiters,queue_ready=False)
        d.atomic(out/'prelease-environment.json',dict(samples=before,foreign_waiters=waiters,purpose=purpose,
            performance_gates=d.gates(before,before,now=d.wall()),rate_eligible=False))
        with d.lease(out,out.name,checkpoint=spec['checkpoint'],source_revision=spec['source_revision'])as lease:
            source=out/'source'
            command=[sys.executable,str(source/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],
                '--label',out.name,'--desktop-cpg-lease',str(lease.path),'--mcp-url',spec['mcp_url'],
                '--receipt',str(out/'paired-locks.json'),'--',sys.executable,
                str(source/'scripts/research/probe_ds41_plan_components.py'),'unit','--output',str(out),'--run-gpu']
            try:
                with(out/'process.log').open('x')as log:
                    child=d.spawn(command,env=dict(os.environ,PYTHONPATH=str(source/'src')+
                        (os.pathsep+str(out/'native')if spec['suite']=='resident'else''),MLX_ENABLE_TF32='0'),
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    d.atomic(out/'process.json',dict(pid=child.pid,command=command))
                    result=d.monitor(child,lease,out,queue_ready=False)
                    if result==0:
                        child_report=read_json(out/'results.json')
                        if (any(child_report.get(k)is not True for k in ('passed','methodological_passed','numerical_passed','source_unchanged'))
                                or any(child_report.get(k)!=spec[k]for k in ('checkpoint','suite','source_revision','source_sha256'))
                                or child_report.get('purpose')!=purpose):
                            result=1
                    lease.success=result==0
            finally:
                if child is not None:child.stop(timeout=20)
    except BaseException as error:
        detail={'type':type(error).__name__,'message':str(error)}
        if hasattr(error,'admission_samples'):
            detail.update(admission_samples=error.admission_samples,admission_gates=error.admission_gates)
        write_new_json(out/'controller-error.json',detail)
        result=1
    finally:
        try:paired=read_json(out/'paired-locks.json')if(out/'paired-locks.json').is_file()else{}
        except BaseException as error:paired={'release_error':str(error)}
        stopped=child is None or not child.live()
        released=bool(lease and paired.get('finished_at')and not paired.get('release_error')
            and paired.get('status')in ('command_completed','command_failed','error','interrupted')
            and paired.get('owner',{}).get('session')==lease.owner['cpg_session']
            and paired.get('owner',{}).get('label')==out.name)
        lease_errors=bool(lease and any('error'in lease.owner.get(k,{})for k in ('release_task','complete_worker')))
        source_error=None
        try:source_ok=verify_source(out,spec)
        except BaseException as error:source_ok=False;source_error=str(error)
        write_new_json(out/'release-verified.json',dict(owned_processes_absent=stopped,
            paired_release_verified=released,cpg_release_errors=lease_errors,source_unchanged=source_ok,
            source_error=source_error))
        if not stopped or not released or lease_errors or not source_ok:result=1
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run','unit'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',choices=tuple(SUITES));parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--purpose',choices=PURPOSES)
    args=parser.parse_args(argv)
    if args.mode=='prepare':
        if args.run_gpu or args.suite is None:parser.error('prepare needs --suite and forbids --run-gpu')
        prepare(args.output,args.suite,purpose=args.purpose or'performance');return 0
    if args.suite is not None or args.purpose is not None:parser.error('execution reads its sealed suite/purpose; overrides forbidden')
    return (run_prepared if args.mode=='run'else unit)(args.output,run_gpu=args.run_gpu)


if __name__=='__main__':raise SystemExit(main())
