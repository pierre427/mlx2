#!/usr/bin/env python3
"""Bounded owned synthetic SC-MXFP4 Metal oracle; no model qualification.

Temporarily pauses the known local Kimi service, restoring it in finally.
The project qualification recipe requires one resident model and paired locks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from mlx2.desktop_campaign_worker import atomic,check_ownership

SERVICE='com.example.mlx2-kimi-linear-48b-q8'
PLIST=Path.home()/'Library/LaunchAgents'/f'{SERVICE}.plist'


def environment():
    import psutil
    foreign=[]; preparation=[]
    for p in psutil.process_iter(['pid','cmdline']):
        command=' '.join(p.info['cmdline'] or [])
        if len(p.info['cmdline'] or [])>1 and Path(p.info['cmdline'][1]).name=='convert_sc_mxfp4.py':
            preparation.append({'pid':p.pid,'command':command})
        if any(x in command for x in ('mlx2.server','mlx2 serve','mlx_lm.server','omlx serve','llama-server')):
            executable=Path((p.info['cmdline'] or [''])[0]).name
            if executable not in {'zsh','bash','sh','rg'} and 'run_with_gpu_locks.py' not in command:
                foreign.append({'pid':p.pid,'command':command})
    return {'time':time.time(),'foreign_model_processes':foreign,
        'concurrent_cpu_model_preparation':preparation,'host_load_average':os.getloadavg(),
        'waiters':sorted(p.name for p in Path('/Users/Shared/mlxuag/gpu.lock.waiters').iterdir()),
        'swap':psutil.swap_memory()._asdict(),
        'swapusage':subprocess.check_output(['sysctl','vm.swapusage'],text=True),
        'thermal_pressure':subprocess.check_output(['pmset','-g','therm'],text=True)}


def unit(spec):
    out=Path(spec['out']); result={**spec,'passed':False,'execution_kind':'synthetic_reference',
        'not_applicable':['autoregressive serving','KV','APCv2','streaming'], 'qualified':False}
    try:
        result['ownership']=check_ownership(spec)
        for name,digest in spec['source_files'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('source changed: '+name)
        result['before']=environment()
        if result['before']['foreign_model_processes']: raise RuntimeError('foreign model process; yield')
        assert sum(i*i for i in range(128))==690880
        result['ordinary_reference_smoke']={'device':'CPU','sum_squares':690880,'passed':True}
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),MLX2_RUN_GPU_TESTS='1')
        command=[sys.executable,'-m','pytest','-q','tests/test_microscaling_metal.py','--junitxml='+str(out/'pytest.xml')]
        with (out/'pytest.log').open('w') as log:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
        result['returncode']=code; result['passed']=code==0
    except BaseException as e: result['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    finally:
        result['after']=environment(); atomic(out/'results.json',result)
    return 0 if result['passed'] else 1


def owned(out):
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    out=out.resolve(); out.mkdir(exist_ok=False)
    files=['src/mlx2/runtime/microscaling/format.py','src/mlx2/runtime/microscaling/kernels.py',
           'tests/test_microscaling_metal.py','scripts/research/probe_microscaling_components.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    fixture={'seed_recipe':'test_microscaling_metal shape-dependent deterministic NumPy RNG','sources':hashes}
    fingerprint=hashlib.sha256(json.dumps(fixture,sort_keys=True).encode()).hexdigest()
    spec={'out':str(out),'job_id':out.name,'checkpoint':'synthetic-sc-mxfp4-components@recipe-sha256:'+fingerprint,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_files':hashes,'mcp_url':'http://127.0.0.1:8766/mcp'}
    atomic(out/'spec.json',spec); before=environment(); atomic(out/'prelease-environment.json',before)
    if before['waiters'] or any(p.is_dir() for p in (Path('/tmp/gpu.lock'),Path('/Users/Shared/mlxuag/gpu.lock'))):
        raise RuntimeError('yield to existing waiter/owner before pausing a service')
    loaded=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/{SERVICE}'],capture_output=True).returncode==0
    service={'label':SERVICE,'loaded_before':loaded,'paused':False,'restored':False}
    child=None
    try:
        if loaded:
            subprocess.run(['launchctl','bootout',f'gui/{os.getuid()}/{SERVICE}'],check=True)
            service['paused']=True; atomic(out/'service.json',service)
        with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=spec['source_revision']) as lease:
            try:
                cmd=[sys.executable,str(ROOT/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],
                    '--label',out.name,'--desktop-cpg-lease',str(lease.path),'--receipt',str(out/'paired-locks.json'),
                    '--',sys.executable,'-u',str(Path(__file__).resolve()),'unit','--out',str(out)]
                child=spawn(cmd,start_new_session=True); deadline=time.monotonic()+240
                while child.poll() is None:
                    lease.check()
                    if time.monotonic()>deadline: raise TimeoutError('four-minute bounded component unit')
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
                lease.success=child.returncode==0
                return child.returncode
            finally:
                if child is not None: child.stop(timeout=15)
    finally:
        if service['paused']:
            restore=subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(PLIST)],capture_output=True,text=True)
            service['restore_returncode']=restore.returncode; service['restore_stderr']=restore.stderr
            service['restored']=restore.returncode==0
        atomic(out/'service.json',service)
        if service['paused'] and not service['restored']: raise RuntimeError('Kimi service restoration failed')


if __name__=='__main__':
    def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(signum,interrupted)
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['owned','unit']); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); raise SystemExit(unit(json.loads((a.out/'spec.json').read_text())) if a.mode=='unit' else owned(a.out))
