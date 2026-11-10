#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded dynamic-M original-MX runtime smoke; parent owns GPU execution.

Reuses unchanged source freeze, capture-copy and fair paired ownership helpers.
Provenance: provenance/deepseek-v41-native-mx-runtime-smoke.json.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
import probe_ds41_native_mx as base
import probe_ds41_native_ffn as ffn
from mlx2.desktop_campaign_worker import atomic,check_ownership
from mlx2.desktop_runner import source_manifest,digest

SCHEMA='mlx2.ds41-native-mx-runtime-owned.v1'
OPTIONS={'repeats':2,'profile':'source_original_pp_half_dynamic_m_v1','case_count':18,'padding_rows':17}


def validate_spec(spec,out):
    base.validate_queue_spec(spec)
    if (spec.get('schema')!=SCHEMA or Path(spec.get('out','')).resolve()!=out.resolve()
            or spec.get('job_id')!=out.name or spec.get('native_runtime_options')!=OPTIONS
            or type(spec.get('native_runtime_options'))is not dict
            or any(type(spec['native_runtime_options'].get(k))is not type(v)for k,v in OPTIONS.items())
            or spec.get('memory_limit_bytes')!=base.MEMORY_LIMIT
            or spec.get('host_live_reserve_bytes')!=base.HOST_RESERVE
            or spec.get('host_initial_floor_bytes')!=base.HOST_INITIAL
            or spec.get('rate_eligible')is not False or spec.get('numerical_environment')!={'MLX_ENABLE_TF32':'0'}
            or os.environ.get('MLX_ENABLE_TF32')!='0'):
        raise ValueError('Invalid sealed dynamic runtime smoke spec')


def component(out):
    spec=base.checked_json(out/'spec.json');report={**spec,'passed':False,'qualification':False,'performance_qualified':False}
    mx=None
    try:
        validate_spec(spec,out);report['ownership']=check_ownership(spec)
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('Frozen runtime smoke source changed')
        report['host_available_before_component']=base.host_admission(base.HOST_INITIAL).available_bytes
        from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE
        report['before']=sample_environment(HID_PROFILE)
        if [p for p in report['before']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('Foreign model; runtime smoke yields')
        import mlx.core as mx
        import numpy as np
        mx.set_default_device(mx.gpu);mx.set_memory_limit(base.MEMORY_LIMIT);mx.set_cache_limit(128<<20)
        from ds41_native_mx_runtime_components import run_components,VERDICTS
        def guarded_write(path,value):
            host=base.host_admission(base.HOST_RESERVE)
            report['minimum_live_host_available_bytes']=min(report.get('minimum_live_host_available_bytes',host.available_bytes),host.available_bytes)
            atomic(path,value)
        result=run_components(mx,np,spec['native_mx_capture_path'],out,report,None,guarded_write)
        report['peak_mlx_bytes']=max(int(mx.get_peak_memory()),report.get('component_peak_mlx_bytes',0))
        report['after']=sample_environment(HID_PROFILE)
        if [p for p in report['after']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('Foreign model entered runtime smoke')
        report['ownership_after']=check_ownership(spec);base.host_admission(base.HOST_RESERVE)
        report['engineering_verdicts']=result['verdicts']
        report['engineering_passed']=set(result['verdicts'])==VERDICTS and all(v is True for v in result['verdicts'].values())
        report['methodological_passed']=bool(result['complete']and result['harness_completed']and result['required_reference_controls_exact'])
        report['passed']=report['engineering_passed']and report['methodological_passed']
        report['passed_scope']='Runtime geometry/actual operands/tail/stability/finite/FP64 controls; no model-quality or speed claim'
        if report['peak_mlx_bytes']>base.MEMORY_LIMIT:raise MemoryError('Runtime smoke6GiB allocator limit')
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('Frozen runtime smoke source changed')
    except BaseException as error:
        report.update(passed=False,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    finally:
        if mx is not None:
            try:mx.synchronize();gc.collect();mx.clear_cache()
            except BaseException as error:report.update(passed=False,cleanup_error=dict(type=type(error).__name__,message=str(error)))
        atomic(out/'results.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'))),flush=True)
    return 0 if report['passed']else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('component','unit'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--capture-unit',type=Path)
    p.add_argument('--ffn-capture-unit',type=Path);p.add_argument('--queue-ready',action='store_true')
    a=p.parse_args(argv);out=a.output.resolve()
    if a.mode=='unit':
        if a.capture_unit is not None or a.ffn_capture_unit is not None or a.queue_ready:p.error('unit accepts sealed spec only')
        return component(out)
    if a.capture_unit is None or a.ffn_capture_unit is None:p.error('component requires both sealed captures')
    if out.exists():raise FileExistsError('Refusing existing runtime smoke evidence directory')
    host=base.host_admission(base.HOST_INITIAL);out.mkdir(parents=True,exist_ok=False)
    (out/'down-capture').mkdir();down=base.copy_capture(a.capture_unit,out/'down-capture')
    (out/'ffn-capture').mkdir();full=ffn.copy_capture(a.ffn_capture_unit,out/'ffn-capture')
    ds,fs=base.checked_json(down/'spec.json'),base.checked_json(full/'spec.json')
    for key in ('checkpoint','model','context_cap','prompt_tokens'):
        if ds.get(key)!=fs.get(key):raise ValueError('Runtime smoke captures have different model/request identity')
    source,sha,revision=base.freeze_source(out)
    spec=dict(schema=SCHEMA,**{k:ds[k]for k in ('checkpoint','model','context_cap','prompt_tokens')},
        out=str(out),job_id=out.name,source_revision=revision,source_snapshot_sha256=sha,
        capture_source_snapshot_sha256=ds['source_snapshot_sha256'],ffn_capture_source_snapshot_sha256=fs['source_snapshot_sha256'],
        native_mx_capture_path=str(down),native_ffn_capture_path=str(full),native_runtime_options=dict(OPTIONS),
        memory_limit_bytes=base.MEMORY_LIMIT,host_live_reserve_bytes=base.HOST_RESERVE,host_initial_floor_bytes=base.HOST_INITIAL,
        host_available_before=host.available_bytes,numerical_environment={'MLX_ENABLE_TF32':'0'},mcp_url='http://127.0.0.1:8766/mcp',
        thermal_gating='User waived; live admission/ownership retained',rate_eligible=False,
        scope='Dynamic runtime actual/derived/counterfactual operand and padded-tail engineering smoke; no model or speed claim')
    if a.queue_ready:spec.update(queue_ready=True,queue_policy=dict(base.QUEUE_POLICY))
    atomic(out/'spec.json',spec)
    from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE,foreign_waiters
    before=sample_environment(HID_PROFILE);waiters=foreign_waiters('')
    atomic(out/'prelease-environment.json',dict(environment=before,foreign_waiters=waiters,queue_ready=a.queue_ready))
    try:base.prelease_admission(before,waiters,queue_ready=a.queue_ready)
    except RuntimeError:
        atomic(out/'admission-refused.json',dict(reason='foreign model or queued waiter',gpu_lease_acquired=False));raise
    base.host_admission(base.HOST_INITIAL)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    result=1;child=None
    try:
        with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=revision)as lease:
            cmd=[sys.executable,str(source/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],'--label',out.name,
                '--desktop-cpg-lease',str(lease.path),'--mcp-url',spec['mcp_url'],'--receipt',str(out/'paired-locks.json'),'--',
                sys.executable,str(source/'scripts/research/probe_ds41_native_mx_runtime.py'),'unit','--output',str(out)]
            try:
                with(out/'process.log').open('x')as log:
                    env=dict(os.environ,PYTHONPATH=str(source/'src'),MLX_ENABLE_TF32='0')
                    child=spawn(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic(out/'process.json',dict(pid=child.pid,command=cmd))
                    result=base.monitor_child(child,lease,out,queue_ready=a.queue_ready);lease.success=result==0
            finally:
                if child is not None:child.stop(timeout=20)
    finally:
        atomic(out/'release-verified.json',dict(finished_at=time.time(),owners_absent=all(not Path(p).exists()for p in
            ('/Users/Shared/mlxuag/gpu.lock/owner.json','/tmp/gpu.lock/owner.json')),source_unchanged=digest(source_manifest(source))==sha))
    return result


if __name__=='__main__':raise SystemExit(main())
