"""Owned operator-only DS4.1 dense/KV diagnostic; no serving qualification."""
from pathlib import Path
import gc,json,os,time,traceback


def run(out,root):
    from mlx2.desktop_campaign_worker import atomic,check_ownership
    from mlx2.desktop_runner import source_manifest,digest
    from mlx2.runtime.os_memory import host_memory_snapshot
    from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE
    spec=json.loads((out/'spec.json').read_text())
    report={**spec,'passed':False,'qualification':False,'performance_qualified':False}
    mx=None
    try:
        report['ownership']=check_ownership(spec)
        if digest(source_manifest(root))!=spec['source_snapshot_sha256']:
            raise RuntimeError('frozen operator source changed')
        host=host_memory_snapshot()
        if host is None or host.available_bytes<(17<<30):
            raise MemoryError('round2 operator needs8GiBprocess+9GiB live reserve')
        report['host_available_before']=host.available_bytes
        report['before']=sample_environment(HID_PROFILE)
        foreign=[r for r in report['before']['foreign_model_processes'] if r['pid']!=os.getpid()]
        if foreign:raise RuntimeError('foreign model present; operator unit yields')
        import mlx.core as mx
        import numpy as np
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
        from probe_deepseek_v41_source_math import compare
        mx.set_default_device(mx.gpu);mx.set_memory_limit(8<<30);mx.set_cache_limit(128<<20)
        artifact=inspect_artifact(spec['model'])
        if artifact['identity']['fingerprint']!=spec['original_fingerprint']:
            raise ValueError('operator artifact fingerprint changed')
        def write():atomic(out/'results.json',report)
        write()
        from ds41_round2_dense_components import run as dense
        dense(mx,np,artifact,out,report,compare,atomic)
        mx.synchronize();gc.collect();mx.clear_cache()
        from ds41_round2_kv_components import run as kv
        kv(mx,np,report,write)
        report['peak_mlx_bytes']=max(mx.get_peak_memory(),report.get('component_peak_mlx_bytes',0))
        report['after']=sample_environment(HID_PROFILE)
        report['passed']=bool(report['kv_components']['required_raw_exact'] and report['dense_components']['required_raw_exact'])
        if digest(source_manifest(root))!=spec['source_snapshot_sha256']:
            raise RuntimeError('frozen operator source changed during probe')
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        if mx is not None:
            mx.synchronize();gc.collect();mx.clear_cache()
        atomic(out/'results.json',report)
    print(json.dumps({'passed':report['passed'],'error':report.get('error'),'peak_mlx_bytes':report.get('peak_mlx_bytes')}),flush=True)
    return 0 if report['passed'] else 1
