"""Two-block reference-prefix capture; parent owns paired GPU unit and cleanup.

This is a bounded diagnostic operator, not a complete model forward/serving run.
No MLX import or GPU execution on import.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import traceback

LIMIT=32<<30
HOST_FLOOR=9<<30
MAX_LAYERS=2
ORIGINAL_BYTES=18_800_640

class PrefixIncomplete(RuntimeError):pass

def prefix_bound(plan):
    fields=('dense_payload_bytes','dense_prepared_bound_bytes','dense_staging_bound_bytes')
    if any(type(plan.get(k))is not int or plan[k]<0 for k in fields):raise ValueError('Missing dense bounds')
    parts={k:plan[k]for k in fields}
    parts.update(prefix_expert_worst_bytes=MAX_LAYERS*384*ORIGINAL_BYTES,
                 original_staging_bytes=ORIGINAL_BYTES,metadata_bytes=256<<20,
                 engram_bytes=16<<20,scratch_bytes=2<<30)
    total=sum(parts.values())
    if total>LIMIT:raise MemoryError('Conservative two-layer prefix bound exceeds32GiB')
    return dict(parts=parts,total_bytes=total,allocator_limit_bytes=LIMIT,maximum_expert_layers=MAX_LAYERS)

def memory_guard(mx,host_snapshot,samples,stage,layer=None):
    host=host_snapshot()
    active=int(mx.get_active_memory());peak=int(mx.get_peak_memory())
    row=dict(stage=stage,layer=layer,active_mlx_bytes=active,peak_mlx_bytes=peak,
             host_available_bytes=None if host is None else host.available_bytes)
    samples.append(row)
    if active>LIMIT or peak>LIMIT:raise MemoryError('Prefix actual MLX memory exceeded32GiB')
    if host is None or host.available_bytes<HOST_FLOOR:raise MemoryError('Prefix lost9GiB live host reserve')
    return row

@contextmanager
def block_guard(cls,guard,complete,abort):
    original=cls.__call__
    def call(block,*args,**kwargs):
        if complete():raise abort('Captured all four results before next block')
        if type(block.layer)is not int or not 0<=block.layer<MAX_LAYERS:
            raise PrefixIncomplete('Required four cases missing before layer2; bounded prefix stopped')
        guard('before_block',block.layer)
        result=original(block,*args,**kwargs)
        guard('after_block',block.layer)
        return result
    cls.__call__=call
    try:yield
    finally:cls.__call__=original

def sealed_capture(unit,spec):
    path=unit/'harness/legacy_gu_capture.py'
    expected=spec.get('harness_sha256',{}).get('legacy_gu_capture.py')
    if path.is_symlink() or not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Private prefix capture hook seal mismatch')
    descriptor=importlib.util.spec_from_file_location('_ds41_prefix_capture_hook',path)
    module=importlib.util.module_from_spec(descriptor);descriptor.loader.exec_module(module)
    return module

def run(out,root):
    from mlx2.desktop_campaign_worker import atomic,check_ownership
    from mlx2.desktop_runner import source_manifest,digest
    from mlx2.runtime.os_memory import host_memory_snapshot
    from probe_deepseek_v4_decode_compare import sample_environment,HID_PROFILE
    out,root=Path(out),Path(root);spec=json.loads((out/'spec.json').read_text())
    report={**spec,'schema':'mlx2.source-reference-prefix-capture.v1','status':'starting','passed':False,
            'qualification':False,'performance_qualified':False,'model_forward_completed':False,
            'output_quality_evaluated':False,'performance_measurement':False,'maximum_layers':MAX_LAYERS,
            'allocator_limit_bytes':LIMIT,'host_floor_bytes':HOST_FLOOR,'memory_samples':[]}
    mx=adapter=cache=None;capture=None
    def write():atomic(out/'results.json',report)
    try:
        report['ownership']=check_ownership(spec)
        if digest(source_manifest(root))!=spec['source_snapshot_sha256']:raise RuntimeError('Prefix source changed before run')
        host=host_memory_snapshot()
        if host is None or host.available_bytes<LIMIT+HOST_FLOOR:raise MemoryError('Prefix requires41GiB fresh host availability')
        report['host_available_before']=host.available_bytes
        report['before']=sample_environment(HID_PROFILE)
        if [p for p in report['before']['foreign_model_processes']if p['pid']!=os.getpid()]:raise RuntimeError('Foreign model process present; prefix yields')
        capture=sealed_capture(out,spec)
        import mlx.core as mx
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
        from mlx2.adapters.deepseek_v41_source_weights import weight_plan
        from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
        from mlx2.runtime.models.deepseek_v41 import Block
        from mlx2.runtime.streamed_load import WeightStreamRequest
        from mlx2.serving import preload_admission_limit_bytes
        mx.set_default_device(mx.gpu);mx.set_memory_limit(LIMIT);mx.set_cache_limit(128<<20)
        artifact=inspect_artifact(spec['model'])
        if artifact['identity']['fingerprint']!=spec['original_fingerprint']:raise ValueError('Original Source fingerprint changed')
        policy=json.loads((out/'policy.json').read_text())
        policy={k:v for k,v in policy.items()if k not in ('moe_expert_streaming','host_memory_signals')}
        report['effective_execution_policy']=policy
        report['policy_file_sha256']=hashlib.sha256((out/'policy.json').read_bytes()).hexdigest()
        ep=policy['deepseek_source_experts']
        if ep.get('mode')!='hot_original_q1' or ep.get('budget_bytes')!=75_000_000_000 or ep.get('hot_bytes')!=60_000_000_000 or ep.get('preheat')is not False:
            raise ValueError('Prefix requires unchanged75GB/hot60 fixed map and preheatFalse')
        if policy.get('deepseek_source_prefill_wide_dtype','reference')!='reference':raise ValueError('Rejected wide dtype cannot capture reference')
        request=json.loads((out/'request.json').read_text());tokens=request['token_ids']
        if len(tokens)!=100 or any(type(t)is not int or t<0 for t in tokens):raise ValueError('Prefix requires original100-token request')
        report['request_sha256']=hashlib.sha256((out/'request.json').read_bytes()).hexdigest()
        report['input_tokens']=tokens[:99];report['prefix_bound']=prefix_bound(weight_plan(artifact,ep['budget_bytes']))
        # Structural adapter admission still sees the unchanged75GB cache contract.
        # Actual process allocation is independently capped32GiB and block count2.
        admission_limit=preload_admission_limit_bytes()
        if not admission_limit:raise RuntimeError('Unmeasured structural adapter admission')
        report['structural_admission_limit_bytes']=admission_limit;write()
        adapter=DeepSeekV41SourceAdapter(spec['model'],execution_policy=policy,
            weight_streaming=WeightStreamRequest(mode='moe_experts',budget_bytes=ep['budget_bytes'],
                admission_limit_bytes=admission_limit,read_workers=2,bypass_page_cache=True))
        stream=adapter.weight_stream
        if stream.preheat_receipt.get('performed') or stream._cache_bytes:raise RuntimeError('Prefix unexpectedly preheated experts')
        report['expert_initial']=stream.receipt()
        guard=lambda stage,layer=None:memory_guard(mx,host_memory_snapshot,report['memory_samples'],stage,layer)
        guard('loaded');cache=adapter.model.make_cache();report['status']='capturing';write()
        original_acquire=stream.acquire
        def acquire(prefix,expert):
            if int(prefix.split('.')[-1])>=MAX_LAYERS:raise PrefixIncomplete('Expert acquire beyond prefix limit')
            return original_acquire(prefix,expert)
        stream.acquire=acquire
        try:
            try:
                with capture.instrument(adapter.model,out,'reference_prefix',0,stop_on_complete=True)as captured:
                    with block_guard(Block,guard,lambda:len(captured['cases'])==4,capture.CaptureComplete):
                        adapter.model.prefill_forward(mx.array([tokens[:99]],mx.uint32),cache)
                raise RuntimeError('Prefix unexpectedly completed a full model forward')
            except capture.CaptureComplete:
                report['expected_capture_abort']=True
        finally:del stream.acquire
        mx.synchronize();guard('captured')
        manifest,cases=capture.load_capture(out/'legacy-gu-capture-0000.json',checkpoint=spec['checkpoint'],source_snapshot=spec['source_snapshot_sha256'])
        report['capture_file']='legacy-gu-capture-0000.json';report['captured_cases']=len(cases)
        report['expert_layers_loaded']=sorted({k[0]for k in stream._cache})
        if any(layer>=MAX_LAYERS for layer in report['expert_layers_loaded']):raise RuntimeError('Prefix loaded an excluded expert layer')
        report['expert_final']=stream.receipt();report['status']='captured';report['passed']=True
        if digest(source_manifest(root))!=spec['source_snapshot_sha256']:raise RuntimeError('Prefix source changed during run')
    except BaseException as exc:
        report.update(passed=False,status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        errors=[]
        if mx is not None:
            try:mx.synchronize()
            except BaseException as exc:errors.append('fence:'+type(exc).__name__)
        if cache is not None:
            for state in cache:
                try:state.close()
                except BaseException as exc:errors.append('cache:'+type(exc).__name__)
            cache=None
        if adapter is not None:
            try:adapter.close()
            except BaseException as exc:errors.append('adapter:'+type(exc).__name__)
            else:adapter=None
        if mx is not None:
            try:
                mx.synchronize();gc.collect();mx.clear_cache()
                report['peak_mlx_bytes']=int(mx.get_peak_memory())
                if report['peak_mlx_bytes']>LIMIT:errors.append('peak_limit')
            except BaseException as exc:errors.append('final_fence:'+type(exc).__name__)
        report['cleanup_errors']=errors;report['state_discarded']=cache is None and adapter is None
        if errors:report.update(passed=False,status='cleanup_failed')
        elif report['passed']:report['status']='completed'
        write()
    print(json.dumps({'passed':report['passed'],'status':report['status'],'error':report.get('error'),'model_forward_completed':False}),flush=True)
    return 0 if report["passed"] else 1
