#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Owned small synthetic function timings; no checkpoint/model/server loading."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse,gc,importlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
import probe_ds41_native_mx as base
from mlx2.desktop_campaign_worker import atomic,check_ownership
from mlx2.desktop_runner import source_manifest,digest
SCHEMA='mlx2.ds41-synthetic-decode-functions.v1'
MODULES={'kv':'ds41_synthetic_kv_components','attention':'ds41_synthetic_attention_components','math':'ds41_synthetic_math_components','head':'ds41_synthetic_head_components','kv_token':'ds41_synthetic_kv_token_components','sdpa':'ds41_synthetic_sdpa_components','prefill_reuse':'ds41_synthetic_prefill_reuse_components','down_stage':'ds41_synthetic_down_stage_components','q1_provider':'ds41_synthetic_q1_provider','original_down_concat':'ds41_synthetic_original_down_concat'}

def validate(spec,out):
 base.validate_queue_spec(spec)
 if (spec.get('schema')!=SCHEMA or Path(spec.get('out','')).resolve()!=out.resolve() or spec.get('job_id')!=out.name
     or type(spec.get('suites'))is not list or not spec['suites'] or len(set(spec['suites']))!=len(spec['suites']) or not set(spec['suites'])<=set(MODULES) or spec.get('checkpoint_loaded')is not False
     or spec.get('memory_limit_bytes')!=base.MEMORY_LIMIT or spec.get('host_live_reserve_bytes')!=base.HOST_RESERVE
     or spec.get('host_initial_floor_bytes')!=base.HOST_INITIAL or spec.get('ownership_poll_seconds')!=1 or spec.get('rate_eligible')is not False
     or spec.get('numerical_environment')!={'MLX_ENABLE_TF32':'0'} or os.environ.get('MLX_ENABLE_TF32')!='0'):
  raise ValueError('Synthetic component sealed spec mismatch')

def unit(out):
 spec=base.checked_json(out/'spec.json');r={**spec,'passed':False,'qualification':False,'performance_qualified':False};mx=None
 try:
  validate(spec,out);r['ownership']=check_ownership(spec)
  if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('Frozen source changed')
  r['host_available_before']=base.host_admission(base.HOST_INITIAL).available_bytes
  from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE
  r['before']=sample_environment(HID_PROFILE)
  if [p for p in r['before']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('Foreign model present')
  import mlx.core as mx
  mx.set_default_device(mx.gpu);mx.set_memory_limit(base.MEMORY_LIMIT);mx.set_cache_limit(128<<20)
  r['device']=mx.device_info();r['suites_results']={}
  last_ownership=[0.]
  def check_live():
   h=base.host_admission(base.HOST_RESERVE);r['minimum_live_host_available_bytes']=min(r.get('minimum_live_host_available_bytes',h.available_bytes),h.available_bytes)
   now=time.monotonic()
   if now-last_ownership[0]>=1:
    check_ownership(spec);last_ownership[0]=now
  for name in spec['suites']:
   module=MODULES[name]
   check_live();r['phase']=name;atomic(out/'results.json',r)
   value=importlib.import_module(module).run(mx,check_live=check_live)
   r['suites_results'][name]=value;atomic(out/(name+'-results.json'),value)
   mx.synchronize();gc.collect();mx.clear_cache();atomic(out/'results.json',r)
  r['peak_mlx_bytes']=int(mx.get_peak_memory());r['after']=sample_environment(HID_PROFILE)
  if r['peak_mlx_bytes']>base.MEMORY_LIMIT:raise MemoryError('Synthetic peak exceeded6GiB')
  if [p for p in r['after']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('Foreign model entered')
  check_live();r['ownership_after']=check_ownership(spec)
  if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('Frozen source changed')
  r.update(passed=True,phase='completed',scope='Synthetic actual-function costs, not served-model throughput; repeated operands/cache effects explicit; no numerical-quality claim')
 except BaseException as e:r.update(passed=False,error=dict(type=type(e).__name__,message=str(e)),traceback=traceback.format_exc())
 finally:
  if mx is not None:
   try:mx.synchronize();gc.collect();mx.clear_cache()
   except BaseException as e:r.update(passed=False,cleanup_error=str(e))
  atomic(out/'results.json',r)
 print(json.dumps(dict(passed=r['passed'],phase=r.get('phase'),error=r.get('error'))),flush=True)
 return 0 if r['passed']else 1

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('component','unit'));p.add_argument('--output',type=Path,required=True);p.add_argument('--queue-ready',action='store_true');p.add_argument('--suites',nargs='+',choices=tuple(MODULES),default=list(MODULES));a=p.parse_args();out=a.output.resolve()
 if a.mode=='unit':
  if a.queue_ready:p.error('unit reads sealed queue policy only')
  return unit(out)
 if out.exists():raise FileExistsError('Refusing existing synthetic evidence directory')
 host=base.host_admission(base.HOST_INITIAL);out.mkdir(parents=True)
 source,sha,revision=base.freeze_source(out)
 ident=digest(dict(kind='synthetic-function-tests',source_snapshot_sha256=sha,hidden=5120,head_dim=512,index_dim=128,window=128,lengths=[100,128,1024,4096,16384]))
 spec=dict(schema=SCHEMA,checkpoint='synthetic-ds41-functions@'+ident,checkpoint_loaded=False,model='synthetic arrays only; no model instance',out=str(out),job_id=out.name,source_revision=revision,source_snapshot_sha256=sha,suites=a.suites,memory_limit_bytes=base.MEMORY_LIMIT,host_live_reserve_bytes=base.HOST_RESERVE,host_initial_floor_bytes=base.HOST_INITIAL,host_available_before=host.available_bytes,rate_eligible=False,ownership_poll_seconds=1,numerical_environment={'MLX_ENABLE_TF32':'0'},mcp_url='http://127.0.0.1:8766/mcp',thermal_gating='User waived; ownership/live admission retained')
 if a.queue_ready:spec.update(queue_ready=True,queue_policy=dict(base.QUEUE_POLICY))
 atomic(out/'spec.json',spec)
 from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE,foreign_waiters
 before=sample_environment(HID_PROFILE);waiters=foreign_waiters('');atomic(out/'prelease-environment.json',dict(environment=before,foreign_waiters=waiters,queue_ready=a.queue_ready))
 base.prelease_admission(before,waiters,queue_ready=a.queue_ready)
 from mlx2.desktop_campaign_lease import DesktopCPGLease
 from mlx2.desktop_campaign_process import spawn
 child=None;code=1
 try:
  with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=revision)as lease:
   cmd=[sys.executable,str(source/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],'--label',out.name,'--desktop-cpg-lease',str(lease.path),'--mcp-url',spec['mcp_url'],'--receipt',str(out/'paired-locks.json'),'--',sys.executable,str(source/'scripts/research/probe_ds41_synthetic_decode.py'),'unit','--output',str(out)]
   try:
    with(out/'process.log').open('x')as f:
     child=spawn(cmd,env=dict(os.environ,PYTHONPATH=str(source/'src'),MLX_ENABLE_TF32='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     atomic(out/'process.json',dict(pid=child.pid,command=cmd));code=base.monitor_child(child,lease,out,queue_ready=a.queue_ready);lease.success=code==0
   finally:
    if child is not None:child.stop(timeout=20)
 finally:atomic(out/'release-verified.json',dict(finished_at=time.time(),owners_absent=all(not Path(p).exists()for p in ('/Users/Shared/mlxuag/gpu.lock/owner.json','/tmp/gpu.lock/owner.json')),source_unchanged=digest(source_manifest(source))==sha))
 return code
if __name__=='__main__':raise SystemExit(main())
