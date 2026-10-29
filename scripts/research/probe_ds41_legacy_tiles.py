#!/usr/bin/env python3
"""Owned, bounded Source reference-prefix capture and exact legacy-tile probes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
from mlx2.desktop_campaign_worker import atomic,check_ownership
from mlx2.desktop_runner import source_manifest,digest


def copy_capture(path,out):
    path=path.resolve(strict=True);parent=path.parent
    release=json.loads((parent/'release-verified.json').read_text())
    if release.get('owners_absent')is not True or release.get('source_unchanged')is not True:
        raise ValueError('capture not released with unchanged source')
    manifest=json.loads(path.read_text());spec=json.loads((parent/'spec.json').read_text())
    parser=parent/'harness/legacy_gu_capture.py'
    if parser.is_symlink() or hashlib.sha256(parser.read_bytes()).hexdigest()!=spec['harness_sha256']['legacy_gu_capture.py']:
        raise ValueError('capture parser seal mismatch')
    dest=out/'reference-capture';(dest/'harness').mkdir(parents=True)
    shutil.copyfile(parent/'spec.json',dest/'spec.json');shutil.copyfile(parser,dest/'harness/legacy_gu_capture.py')
    for context in ('policy.json','results.json','release-verified.json','request.json'):
        file=parent/context
        if file.exists():shutil.copyfile(file,dest/context)
    total=0
    for case in manifest['cases']:
        name=case['path']
        if Path(name).name!=name or not name.endswith('.npz'):raise ValueError('unsafe capture filename')
        source=parent/name;total+=source.stat().st_size
        if source.is_symlink() or total>40<<20 or hashlib.sha256(source.read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('capture payload seal/size mismatch')
        shutil.copyfile(source,dest/name)
    shutil.copyfile(path,dest/path.name)
    return dest/path.name


def component(out):
    spec=json.loads((out/'spec.json').read_text())
    if spec['experiment']=='legacy_prefix':
        from ds41_legacy_prefix_capture import run
        return run(out,ROOT)
    report={**spec,'passed':False,'qualification':False,'performance_qualified':False}
    mx=None
    try:
        report['ownership']=check_ownership(spec)
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('frozen source changed')
        from mlx2.runtime.os_memory import host_memory_snapshot
        h=host_memory_snapshot()
        if h is None or h.available_bytes<(15<<30):raise MemoryError('component needs6GiB allocation plus9GiB reserve')
        from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE
        report['before']=sample_environment(HID_PROFILE)
        if [p for p in report['before']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('foreign model; yield')
        import mlx.core as mx
        import numpy as np
        mx.set_default_device(mx.gpu);mx.set_memory_limit(6<<30);mx.set_cache_limit(128<<20)
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
        from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed,materialize
        from ds41_prefill_legacy_gu_components import run_legacy_gu
        from probe_deepseek_v41_source_math import compare
        artifact=inspect_artifact(spec['model'])
        if artifact['identity']['fingerprint']!=spec['original_fingerprint']:raise ValueError('checkpoint changed')
        bank={}
        def load_expert(layer,expert):
            key=layer,expert
            if key not in bank:
                host=host_memory_snapshot()
                if host is None or host.available_bytes<(9<<30):raise MemoryError('live host reserve refused')
                raw=read_original_packed(artifact,layer,expert);bank[key]=materialize(raw)
                mx.eval(bank[key].backing,*bank[key].values());del raw
                if mx.get_peak_memory()>6<<30:raise MemoryError('component allocation ceiling')
            return bank[key]
        if spec['experiment']=='legacy_tuning':
            from ds41_legacy_tiles_tuning import run_tuning
            result=run_tuning(mx,np,spec['legacy_capture_path'],load_expert,out,report,compare,atomic)
        elif spec['experiment']=='legacy_batch':
            from ds41_prefill_legacy_batch_components import run_legacy_batch
            result=run_legacy_batch(mx,np,spec['legacy_capture_path'],load_expert,out,report,compare,atomic)
        elif spec['experiment']=='nax_profile':
            from ds41_prefill_nax_profile_components import run_nax_profile
            result=run_nax_profile(mx,np,spec['legacy_capture_path'],load_expert,out,report,compare,atomic)
        elif spec['experiment']=='nax_batch':
            from ds41_prefill_nax_batch_components import run_nax_batch
            result=run_nax_batch(mx,np,spec['legacy_capture_path'],load_expert,out,report,compare,atomic)
        elif spec['experiment']=='small_m':
            from ds41_small_m_nax_components import run_small_m
            result=run_small_m(mx,np,spec['legacy_capture_path'],load_expert,out,report,compare,atomic)
        else:
            result=run_legacy_gu(mx,np,spec['legacy_capture_path'],load_expert,out,report,compare,atomic)
        report['peak_mlx_bytes']=mx.get_peak_memory();report['after']=sample_environment(HID_PROFILE)
        if spec['experiment'] in ('legacy_tuning','nax_profile','nax_batch','small_m'):
            report['passed']=bool(result['complete'] and result['harness_completed'] and result['required_reference_controls_exact'])
            if spec['experiment']=='legacy_tuning':report['numerical_verdicts']=result['verdicts']
            elif spec['experiment']=='nax_profile':report['numerical_verdicts']={k:result[k]for k in('candidate_finite','ordered_reduction_exact','candidate_raw_and_boundary_exact')}
            elif spec['experiment']=='small_m':
                report['numerical_verdicts']=result['verdicts']
                report['passed'] &= result['verdicts']['all_engineering_exact'] and result['verdicts']['all_finite']
            else:
                report['numerical_verdicts']={k:result[k]for k in('all_batch_addressing_exact','all_finite')}
                report['passed'] &= result['all_batch_addressing_exact'] and result['all_finite']
                if spec.get('nax_batch_options',{}).get('fused_middle',False):
                    report['numerical_verdicts']['all_fused_middle_exact']=result.get('all_fused_middle_exact')
                    report['passed'] &= result.get('all_fused_middle_exact') is True
            report['passed_scope']=('Captured stock controls and same-profile batch addressing; stock-vs-native numeric differences are separate, no model-quality verdict'
                if spec['experiment'] in ('nax_batch','small_m')else'Harness/reference-control pass; candidate numerical verdicts remain separate')
        else:
            verdict='all_raw_boundary_partial_exact' if spec['experiment']=='legacy_batch' else 'all_raw_and_boundaries_exact'
            report['numerical_verdict_field']=verdict
            report['passed']=bool(result['complete'] and result[verdict])
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('frozen source changed')
    except BaseException as error:
        report.update(passed=False,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    finally:
        if mx is not None:
            try:mx.synchronize();gc.collect();mx.clear_cache()
            except BaseException as error:
                report.update(passed=False,cleanup_error=dict(type=type(error).__name__,message=str(error)))
        atomic(out/'results.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'))),flush=True)
    return 0 if report['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prefix','component','batch','tune','nax-profile','nax-batch','small-m','unit'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--capture',type=Path)
    p.add_argument('--fused-middle',action='store_true',help='Exact-gated paired ordered reductions and Source middle consumer')
    p.add_argument('--all-small-m',action='store_true',help='Include every single-expert M8 through M12 in the bounded small-M probe')
    args=p.parse_args();out=args.output.resolve()
    if args.mode=='unit':return component(out)
    if args.fused_middle and args.mode!='nax-batch':p.error('--fused-middle requires nax-batch')
    if args.all_small_m and args.mode!='small-m':p.error('--all-small-m requires small-m')
    from mlx2.runtime.os_memory import host_memory_snapshot
    host=host_memory_snapshot();floor=(41 if args.mode=='prefix' else 15)<<30
    if host is None or host.available_bytes<floor:raise MemoryError('CPU-only admission refused before lease')
    if args.mode in ('component','batch','tune','nax-profile','nax-batch','small-m') and args.capture is None:p.error('component needs --capture')
    out.mkdir(parents=True,exist_ok=False)
    native_mode=args.mode in ('nax-profile','nax-batch','small-m')
    prior=args.capture.resolve().parent if native_mode else ROOT/'qualification/runs/ds41-source-prefill-gu-seal-dedup-20261025-r1'
    prior_spec=json.loads((prior/'spec.json').read_text());policy=json.loads((prior/'policy.json').read_text())
    if not native_mode:
        policy['deepseek_source_experts']['preheat']=False
        policy['deepseek_source_experts']['atlas_path']=str(out/'atlas.json')
        shutil.copyfile(ROOT/'.local-source-lfu/atlas75-seed.json',out/'atlas.json')
    else:
        policy={'numerical_profile':('small_m_nax_p1_p5_ordered_f32_v1'if args.mode=='small-m'else'mxfp4_nax_k16_ordered_f32_v1'),'serving_selected':False,'scope':'Bounded original-weight derived-input operators only'}
    shutil.copyfile(prior/'request.json',out/'request.json');atomic(out/'policy.json',policy)
    harness=out/'harness';harness.mkdir()
    for f in (ROOT/'.local-source-lfu').glob('*.py'):shutil.copyfile(f,harness/f.name)
    cap=copy_capture(args.capture,out) if args.capture else None
    manifest=source_manifest(ROOT);sha=digest(manifest);source=out/'source';source.mkdir()
    atomic(out/'source-inputs.json',dict(source_sha256=sha,files=manifest))
    for rel in manifest:
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    frozen=source_manifest(source)
    if digest(frozen)!=sha:
        atomic(out/'source-freeze-failure.json',dict(expected_sha256=sha,actual_sha256=digest(frozen),
            changed_files=[k for k in sorted(set(manifest)|set(frozen))if manifest.get(k)!=frozen.get(k)],
            gpu_lease_acquired=False))
        raise RuntimeError('source freeze inconsistent')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic(source/'desktop-source.json',dict(source_revision=revision,source_sha256=sha,files=manifest))
    spec={**{k:prior_spec[k]for k in ('checkpoint','model','context_cap','prompt_tokens')},
        'out':str(out),'job_id':out.name,'source_revision':revision,'source_snapshot_sha256':sha,
        'original_fingerprint':'c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb',
        'experiment':{'prefix':'legacy_prefix','batch':'legacy_batch','tune':'legacy_tuning','component':'legacy_component','nax-profile':'nax_profile','nax-batch':'nax_batch','small-m':'small_m'}[args.mode],
        'nax_batch_options':({'fused_middle':True}if args.fused_middle else {}),
        'small_m_options':({'all_single_m':True}if args.all_small_m else {}),
        'memory_limit_bytes':(32 if args.mode=='prefix' else 6)<<30,'host_available_before':host.available_bytes,
        'legacy_capture_path':str(cap)if cap else None,'mcp_url':'http://127.0.0.1:8766/mcp',
        'harness_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in harness.glob('*.py')},
        'numerical_environment':{'MLX_ENABLE_TF32':'0'},'thermal_gating':'user waived; live admission and paired ownership retained',
        'scope':'Bounded reference prefix / actual-weight operator diagnostic; no full-bank residency, throughput or qualification claim'}
    atomic(out/'spec.json',spec)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    result=1;child=None
    try:
        with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=revision)as lease:
            command=[sys.executable,str(source/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],
                '--label',out.name,'--desktop-cpg-lease',str(lease.path),'--mcp-url',spec['mcp_url'],
                '--receipt',str(out/'paired-locks.json'),'--',sys.executable,
                str(source/'scripts/research/probe_ds41_legacy_tiles.py'),'unit','--output',str(out)]
            try:
                with(out/'process.log').open('x')as log:
                    child=spawn(command,env=dict(os.environ,PYTHONPATH=str(source/'src'),MLX_ENABLE_TF32='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic(out/'process.json',dict(pid=child.pid,command=command));deadline=time.monotonic()+600
                    while child.poll()is None:
                        lease.check()
                        if time.monotonic()>deadline:raise TimeoutError('bounded tile unit deadline')
                        time.sleep(1)
                    result=child.returncode;lease.success=result==0
            finally:
                if child is not None:child.stop(timeout=20)
    finally:
        atomic(out/'release-verified.json',dict(finished_at=time.time(),owners_absent=all(not Path(x).exists()for x in ['/Users/Shared/mlxuag/gpu.lock/owner.json','/tmp/gpu.lock/owner.json']),source_unchanged=digest(source_manifest(source))==sha))
    return result

if __name__=='__main__':raise SystemExit(main())
