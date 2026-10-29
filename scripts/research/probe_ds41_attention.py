#!/usr/bin/env python3
"""Bounded actual-operand attention replay under the ordinary paired owner.

No serving route is selected here. The captured ordinary result is the
component reference; numerical acceptance is separate from harness completion.
See provenance/deepseek-v41-attention-owned-probes.json.
"""
import argparse
import gc
import hashlib
import importlib.util
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


def copy_capture(unit,out,*,projection=False,hc=False,dest_name='reference-capture'):
    """Copy bounded evidence and its sealed parser, never arbitrary payloads."""
    unit=unit.resolve(strict=True)
    spec=json.loads((unit/'spec.json').read_text())
    result=json.loads((unit/'results.json').read_text())
    release=json.loads((unit/'release-verified.json').read_text())
    if (result.get('status')!='completed' or result.get('passed')is not True
            or release.get('owners_absent')is not True or release.get('source_unchanged')is not True):
        raise ValueError('Capture model unit incomplete or not released')
    if projection and hc:raise ValueError('Conflicting capture schemas')
    if dest_name not in ('reference-capture','projection-reference-capture'):raise ValueError('Unsafe capture destination')
    parser_name='hc_boundary_probe.py'if hc else'attention_projection_probe.py'if projection else'attention_probe.py'
    parser=unit/'harness'/parser_name
    if parser.is_symlink() or hashlib.sha256(parser.read_bytes()).hexdigest()!=spec['harness_sha256'][parser_name]:
        raise ValueError('Attention capture parser seal mismatch')
    dest=out/dest_name;(dest/'harness').mkdir(parents=True)
    shutil.copyfile(parser,dest/'harness'/parser_name)
    if hashlib.sha256((dest/'harness'/parser_name).read_bytes()).hexdigest()!=spec['harness_sha256'][parser_name]:
        raise ValueError('Copied attention parser changed')
    for name in ('spec.json','request.json','policy.json','results.json','release-verified.json'):
        shutil.copyfile(unit/name,dest/name)
    total=0
    prefix='hc-boundary'if hc else'attention-projection'if projection else'attention-'
    files=sorted(set(unit.glob(prefix+'*.json'))|set(unit.glob(prefix+'*.npz')))
    if not files:raise ValueError('No attention capture evidence')
    for source in files:
        size=source.stat().st_size;total+=size
        if source.is_symlink() or total>160<<20:raise ValueError('Attention capture copy budget exceeded')
        shutil.copyfile(source,dest/source.name)
    loaded=importlib.util.spec_from_file_location('_sealed_attention_capture_parser',dest/'harness'/parser_name)
    parser_module=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(parser_module)
    capture_report,cases=parser_module.load_capture(dest,checkpoint=spec['checkpoint'],source_snapshot=spec['source_snapshot_sha256'])
    if len(cases)!=(12 if projection else 6):raise ValueError('Capture PP/decode case inventory incomplete')
    copy_receipt='capture-copy.json'if dest_name=='reference-capture'else'projection-capture-copy.json'
    atomic(out/copy_receipt,dict(source_unit=str(unit),source_spec_sha256=hashlib.sha256((unit/'spec.json').read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256((dest/p.name).read_bytes()).hexdigest()for p in files},total_bytes=total,
        validated_cases=len(cases),capture_checkpoint=capture_report['checkpoint']))
    return dest


def component(out):
    spec=json.loads((out/'spec.json').read_text())
    report={**spec,'passed':False,'qualification':False,'performance_qualified':False}
    mx=None
    try:
        report['ownership']=check_ownership(spec)
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('Frozen source changed')
        from mlx2.runtime.os_memory import host_memory_snapshot
        h=host_memory_snapshot()
        if h is None or h.available_bytes<15<<30:raise MemoryError('Component needs6GiB ceiling plus9GiB live reserve')
        from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE
        report['before']=sample_environment(HID_PROFILE)
        if [p for p in report['before']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('Foreign model; yield')
        import mlx.core as mx
        import numpy as np
        mx.set_default_device(mx.gpu);mx.set_memory_limit(6<<30);mx.set_cache_limit(128<<20)
        if spec.get('experiment')=='dwarfstar':
            from ds41_dwarfstar_attention_components import run_components
        elif spec.get('experiment')=='omlx_nax':
            from ds41_omlx_nax_attention_components import run_components
        elif spec.get('experiment')=='head_group':
            from ds41_head_group_components import run_components
        elif spec.get('experiment')=='hc_boundary':
            from ds41_hc_boundary_components import run_components
        elif spec.get('experiment')=='output_chain':
            from ds41_output_chain_components import run_components
        elif spec.get('experiment')=='woa_pp':
            from ds41_woa_pp_components import run_components
        elif spec.get('experiment')=='qakv':
            from ds41_qakv_components import run_components
        elif spec.get('experiment')=='fp8_tiles':
            from ds41_fp8_projection_components import run_components
        elif spec.get('experiment')=='projection':
            from ds41_attention_projection_components import run_components
        elif spec.get('experiment')=='allocation':
            from ds41_attention_allocations import run_allocations as run_components
        else:
            from ds41_attention_components import run_components
        from probe_deepseek_v41_source_math import compare
        if spec.get('metal_trace'):
            from ds41_metal_trace import trace_component
            with trace_component(out,report,atomic):
                result=run_components(mx,np,spec['attention_capture_path'],out,report,compare,atomic)
        else:
            result=run_components(mx,np,spec['attention_capture_path'],out,report,compare,atomic)
        if spec.get('with_allocation'):
            from ds41_attention_allocations import run_allocations
            allocation=run_allocations(mx,np,spec['attention_capture_path'],out,report,compare,atomic)
            if not(allocation['complete']and allocation['required_reference_controls_exact']):
                raise RuntimeError('Evaluated allocation reference gates failed')
        report['peak_mlx_bytes']=max(int(mx.get_peak_memory()),report.get('component_peak_mlx_bytes',0));report['after']=sample_environment(HID_PROFILE)
        # Candidate verdicts do not determine whether the reference harness ran.
        report['passed']=bool(result['complete'] and result['harness_completed'] and result['required_reference_controls_exact'])
        report['numerical_verdicts']=result['verdicts']
        report['passed_scope']='Harness and saved-reference replay; candidate numerical verdicts separate'
        if report['peak_mlx_bytes']>6<<30:raise MemoryError('Component allocation ceiling exceeded')
        if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise RuntimeError('Frozen source changed')
    except BaseException as error:
        report.update(passed=False,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    finally:
        if mx is not None:
            try:
                mx.synchronize();gc.collect();mx.clear_cache()
            except BaseException as error:
                report.update(passed=False,cleanup_error=dict(type=type(error).__name__,message=str(error)))
        atomic(out/'results.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'))),flush=True)
    return 0 if report['passed']else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('component','unit'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capture-unit',type=Path)
    p.add_argument('--projection-capture-unit',type=Path)
    p.add_argument('--experiment',choices=('components','allocation','projection','fp8_tiles','qakv','output_chain','woa_pp','hc_boundary','head_group','dwarfstar','omlx_nax'),default='components')
    p.add_argument('--with-allocation',action='store_true')
    p.add_argument('--metal-trace',action='store_true')
    p.add_argument('--fp8-followup',action='store_true',help='Seven interleaved repeats of the bounded FP8 shortlist')
    p.add_argument('--output-chain-followup',action='store_true',help='Seven repeats of the inverse-RoPE/QAT/layout candidate')
    p.add_argument('--hc-followup',action='store_true',help='Seven repeats of the exact HC frontier and sequential variants')
    p.add_argument('--head-group-followup',action='store_true',help='Seven repeats of four-head variants and the stock reference')
    p.add_argument('--peer-followup',action='store_true',help='Seven repeats of a peer kernel with captured and current broadcast controls')
    p.add_argument('--dwarfstar-fused-stage',action='store_true',help='Byte-gated fused decode staging with unchanged peer attention math')
    p.add_argument('--projection-native-profile',action='store_true',help='Seven-repeat native BF16 PP projection numerical profile, with independent error evidence')
    p.add_argument('--woa-native-profile',action='store_true',help='Seven-repeat native BF16 grouped WoA PP profile including production coefficient-cast costs')
    args=p.parse_args();out=args.output.resolve()
    if args.mode=='unit':return component(out)
    if args.capture_unit is None:p.error('component requires --capture-unit')
    if args.fp8_followup and args.experiment!='fp8_tiles':p.error('--fp8-followup requires fp8_tiles')
    if args.output_chain_followup and args.experiment!='output_chain':p.error('--output-chain-followup requires output_chain')
    if args.hc_followup and args.experiment!='hc_boundary':p.error('--hc-followup requires hc_boundary')
    if args.head_group_followup and args.experiment!='head_group':p.error('--head-group-followup requires head_group')
    if args.peer_followup and args.experiment not in ('dwarfstar','omlx_nax'):p.error('--peer-followup requires dwarfstar or omlx_nax')
    if args.dwarfstar_fused_stage and args.experiment!='dwarfstar':p.error('--dwarfstar-fused-stage requires dwarfstar')
    if args.projection_native_profile and args.experiment!='projection':p.error('--projection-native-profile requires projection')
    if args.woa_native_profile and args.experiment!='woa_pp':p.error('--woa-native-profile requires woa_pp')
    if args.experiment=='woa_pp' and args.projection_capture_unit is None:p.error('woa_pp requires --projection-capture-unit')
    from mlx2.runtime.os_memory import host_memory_snapshot
    host=host_memory_snapshot()
    if host is None or host.available_bytes<15<<30:raise MemoryError('CPU-only admission refused before lease')
    out.mkdir(parents=True,exist_ok=False)
    if args.with_allocation and args.experiment!='components':p.error('--with-allocation requires components')
    capture=copy_capture(args.capture_unit,out,projection=args.experiment in ('projection','fp8_tiles','qakv'),hc=args.experiment=='hc_boundary')
    projection_capture=None
    if args.projection_capture_unit:
        if args.experiment not in ('output_chain','woa_pp'):p.error('--projection-capture-unit requires output_chain or woa_pp')
        projection_capture=copy_capture(args.projection_capture_unit,out,projection=True,dest_name='projection-reference-capture')
    capture_spec=json.loads((capture/'spec.json').read_text())
    manifest=source_manifest(ROOT);sha=digest(manifest);source=out/'source';source.mkdir()
    atomic(out/'source-inputs.json',dict(source_sha256=sha,files=manifest))
    for rel in manifest:
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    frozen=source_manifest(source)
    if digest(frozen)!=sha:
        atomic(out/'source-freeze-failure.json',dict(expected_sha256=sha,actual_sha256=digest(frozen),
            changed_files=[k for k in sorted(set(manifest)|set(frozen))if manifest.get(k)!=frozen.get(k)],gpu_lease_acquired=False))
        raise RuntimeError('Source freeze inconsistent')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic(source/'desktop-source.json',dict(source_revision=revision,source_sha256=sha,files=manifest))
    spec={**{k:capture_spec[k]for k in ('checkpoint','model','context_cap','prompt_tokens')},
        'out':str(out),'job_id':out.name,'source_revision':revision,'source_snapshot_sha256':sha,
        'capture_source_snapshot_sha256':capture_spec['source_snapshot_sha256'],
        'experiment':args.experiment,
        'with_allocation':args.with_allocation,
        'metal_trace':args.metal_trace,
        'fp8_projection_options':({'repeats':7,'pp_variants':['m32_n32','m64_n32'],
            'decode_variants':['s2_r4','s2_r8']}if args.fp8_followup else {}),
        'output_chain_options':({'repeats':7,'variants':['packed_inverse_QAT']}if args.output_chain_followup else {}),
        'hc_boundary_options':({'repeats':7,'policies':['stock_frontier','inline_sequential']}if args.hc_followup else {}),
        'head_group_options':({'repeats':7,'timing_variants':['h4_s1','h4_s4']}if args.head_group_followup else {}),
        'peer_options':({'repeats':7}if args.peer_followup else {}),
        'dwarfstar_options':({'prepare_mode':'fused_decode'}if args.dwarfstar_fused_stage else {}),
        'projection_options':({'numerical_profile':'native_bf16_pp_v1','repeats':7}if args.projection_native_profile else {}),
        'woa_options':({'numerical_profile':'native_bf16_grouped_pp_v1','repeats':7}if args.woa_native_profile else {}),
        'attention_capture_path':str(capture),'memory_limit_bytes':6<<30,'host_available_before':host.available_bytes,
        'output_chain_projection_capture_path':str(projection_capture)if projection_capture else None,
        'mcp_url':'http://127.0.0.1:8766/mcp','numerical_environment':{'MLX_ENABLE_TF32':'0'},
        'cpu_oracle_environment':({'VECLIB_MAXIMUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
            if args.experiment in ('dwarfstar','omlx_nax','woa_pp')or args.projection_native_profile else {}),
        'thermal_gating':'user waived; live admission and paired ownership retained',
        'scope':'Actual attention operand replay and unselected dataflow candidates; no throughput or qualification claim'}
    atomic(out/'spec.json',spec)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    result=1;child=None
    try:
        with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=revision)as lease:
            command=[sys.executable,str(source/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],
                '--label',out.name,'--desktop-cpg-lease',str(lease.path),'--mcp-url',spec['mcp_url'],
                '--receipt',str(out/'paired-locks.json'),'--',sys.executable,
                str(source/'scripts/research/probe_ds41_attention.py'),'unit','--output',str(out)]
            try:
                with(out/'process.log').open('x')as log:
                    child_env=dict(os.environ,PYTHONPATH=str(source/'src'),MLX_ENABLE_TF32='0')
                    child_env.update(spec['cpu_oracle_environment'])
                    child=spawn(command,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic(out/'process.json',dict(pid=child.pid,command=command));deadline=time.monotonic()+600
                    while child.poll()is None:
                        lease.check()
                        if time.monotonic()>deadline:raise TimeoutError('Bounded attention component deadline')
                        time.sleep(1)
                    result=child.returncode;lease.success=result==0
            finally:
                if child is not None:child.stop(timeout=20)
    finally:
        atomic(out/'release-verified.json',dict(finished_at=time.time(),owners_absent=all(not Path(x).exists()for x in ['/Users/Shared/mlxuag/gpu.lock/owner.json','/tmp/gpu.lock/owner.json']),source_unchanged=digest(source_manifest(source))==sha))
    return result


if __name__=='__main__':raise SystemExit(main())
