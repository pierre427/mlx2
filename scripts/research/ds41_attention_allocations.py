"""Evaluated MLX allocation peaks, not DRAM-traffic or timing measurements."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import gc
import hashlib


def run_allocations(mx,np,capture_path,out,report,compare,atomic):
    from ds41_attention_components import capture_module,restored_inputs,raw_details
    from mlx2.runtime.models import deepseek_v41_attention_probe as probe
    if mx.default_device()!=mx.gpu:raise RuntimeError('Allocation probe requires owned GPU')
    if not hasattr(mx,'reset_peak_memory'):raise RuntimeError('MLX allocation reset unavailable')
    parser,path=capture_module(capture_path);manifest,cases=parser.load_capture(path)
    if report['checkpoint']!=manifest['checkpoint']:raise ValueError('Allocation capture identity mismatch')
    rows=report['attention_allocations']=[];maximum=0;controls=True
    for case,arrays in cases:
        if case['phase']!='pp99':continue
        args,gathered,layout=restored_inputs(mx,np,case,arrays)
        if gathered is not None:raise ValueError('PP allocation probe cannot reuse width-one gather')
        def call(variant):return probe.attention(*args,variant=variant,all_retained=True)
        for variant in ('reference','broadcast'):
            value=call(variant);mx.eval(value)
            check=raw_details(np,arrays['reference_raw'],np.array(value))
            controls=controls and check['exact'];del value
            if not check['exact']:raise ValueError('Allocation probe raw reference gate failed')
        for repetition in range(2):
            for variant in (('reference','broadcast')if repetition==0 else('broadcast','reference')):
                mx.synchronize();gc.collect();mx.clear_cache();mx.reset_peak_memory()
                before=int(mx.get_active_memory())
                value=call(variant);mx.eval(value)
                peak=int(mx.get_peak_memory());active=int(mx.get_active_memory());maximum=max(maximum,peak)
                if peak>6<<30:raise MemoryError('Allocation probe exceeds6GiB ceiling')
                row=dict(layer=case['layer'],phase=case['phase'],variant=variant,repetition=repetition,
                    active_before_bytes=before,active_with_output_bytes=active,peak_bytes=peak,
                    incremental_peak_bytes=max(0,peak-before),output_bytes=int(value.nbytes),
                    output_sha256=hashlib.sha256(np.array(value).tobytes()).hexdigest())
                del value;mx.synchronize();gc.collect();mx.clear_cache()
                row['active_after_output_release_bytes']=int(mx.get_active_memory());rows.append(row)
                atomic(out/'results.json',report)
        args=None;del gathered;gc.collect();mx.clear_cache()
    report['allocation_scope']='MLX allocator incremental peak for evaluated PP operators, two reversed-order repetitions; not physical DRAM bytes or isolated GPU time'
    report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),maximum)
    result=dict(complete=len(rows)==12,harness_completed=True,required_reference_controls_exact=controls,
        verdicts={'allocation_samples_complete':len(rows)==12})
    report['allocation_result']=result
    return result
