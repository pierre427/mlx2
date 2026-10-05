"""Root-run explicit same-geometry cold packed hybrid prefill control, default-off.

Requires a combined source-bound native binary with matrix/multirow APIs.
Exact full logits, every conv/GDN state and logical FA KV versus serial ordinary.
100s supervisor/RSS48GiB; GPUQ fresh owner check; no model weights in parent.
"""
from __future__ import annotations
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
from varlen_hybrid_http_gate import MODEL,MANIFEST,WHEEL
from varlen_hybrid_serving_smoke import save
from varlen_hybrid_b1_numeric_gate import tensor_metrics,compare_slots,validate_physical
ROOT=Path(__file__).resolve().parents[2]
from varlen_packed_long_contract import COUNTS,exact_prompts,validate_q1
DEFAULT_NATIVE='/tmp/mlx2-packed-long-q1-handoff-build-1004/_paged_kv_native.cpython-312-darwin.so'
DEFAULT_NATIVE_SHA='599e57b2648acd0d2cf896eca78bb3f88087fe0fed16f13abf5896c3637afbdb'
MAX_SECONDS=100;MAX_RSS=48<<30
FAILURE_ROOTS=[]


def plan():
    return {'schema':'mlx2.hybrid-packed-prefill-long-geometry-gate.v1','status':'planned','gpu_executed':False,
        'qualified':False,'price_usable':False,'serving_selected':False,'default_off':True,
        'contexts':list(COUNTS),'numeric_parity':'not_tested','hard_seconds':MAX_SECONDS,
        'max_rss_bytes':MAX_RSS,'reference':'same pinned model packed mixed_forward ordinary segmented SDPA; serial exact arm retained separately',
        'scope':'same packed projection geometry native-component exact control + B2 Q1; serial parity separately, no qualification/performance'}


def make_profile(identity,*unused):
    sys.path.insert(0,str(ROOT/'src'))
    from mlx2.runtime.hybrid_packed_prefill_long import make_long_profile
    return make_long_profile(identity)


def compare_bootstrap(boots,reference_rows,initial,candidate,mx):
    from varlen_hybrid_fa_boundary_probe import _export_logical_native_kv
    result={'logits':[],'recurrent':[],'kv':[]}
    for lane,(boot,reference,ordinary) in enumerate(zip(boots,reference_rows,initial)):
        result['logits'].append({'lane':lane,**tensor_metrics(boot.logits[0],ordinary,mx)})
        for ordinal,layer in enumerate(candidate.layer_map.recurrent):
            for slot in (0,1):
                result['recurrent'].append({'lane':lane,'layer':layer,'slot':slot,
                    **tensor_metrics(boot.recurrent_caches[ordinal].cache[slot],reference[layer].cache[slot],mx)})
        for ordinal,layer in enumerate(candidate.layer_map.full_attention):
            owner=candidate._serving_resources.owners[lane]._public.layers[ordinal]
            keys,values=_export_logical_native_kv(owner,mx)
            rk,rv=reference[layer].keys_and_values()
            for plane,actual,expected in (('key',keys,rk[0]),('value',values,rv[0])):
                result['kv'].append({'lane':lane,'layer':layer,'plane':plane,
                    **tensor_metrics(actual,expected,mx)})
    result['passed']=all(r['finite'] and r['exact'] for group in ('logits','recurrent','kv') for r in result[group])
    return result


def stock_mixed_reference(model, ids, mx):
    """Independent cold caches, real-row mixed_forward, same final QMM shape."""
    rows=tuple(tuple(model.make_cache()) for _ in ids)
    segments=[(mx.array([list(tokens)]),list(caches)) for tokens,caches in zip(ids,rows)]
    hidden=model.mixed_forward(segments)
    if len(hidden)!=len(ids) or any(x.shape[:2]!=(1,len(tokens)) for x,tokens in zip(hidden,ids)):
        raise ValueError('stock mixed real-row output geometry differs')
    final=mx.concatenate([x[:,-1:,:] for x in hidden],axis=0)
    logits=model.logits(final)[:,0,:]
    mx.eval(logits,*[v for caches in rows for c in caches for v in c.state if v is not None])
    return rows,tuple(logits[lane] for lane in range(len(ids)))


def execute(args,result):
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner']=_gpuq_owner()
    if not args.native or not args.native_sha256 or len(args.native_sha256)!=64:
        raise ValueError('explicit combined native path/SHA required')
    if hashlib.sha256(Path(args.native).read_bytes()).hexdigest()!=args.native_sha256:
        raise ValueError('combined native hash differs')
    sys.path.insert(0,str(Path(args.native).parent));sys.path.insert(0,str(ROOT/'src'))
    from mlx2.runtime.paged_price_identity import cached_live_price_identity
    identity=cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
        adapter_artifact_root=Path(args.model).resolve())
    if not args.long_fused:raise ValueError('explicit long fused selector required')
    proposed=make_profile(identity);result['identity']=identity
    result['explicit_profile_arm']='nax_fused_stock_long'
    result['native_reader_scratch_bytes']=0
    result['native_reader_simultaneous_scratch_bytes']=result['native_reader_scratch_bytes']*args.prefill_eval_block_size
    if args.prepare_profile:
        save(args.profile,proposed);result['status']='profile_prepared';return
    os.environ.update(proposed['required_environment'])
    from mlx2.runtime.hybrid_packed_prefill import load_profile,create_cold_packed_hybrid
    profile=load_profile(args.profile,live_identity=identity,counts=COUNTS,environment=os.environ)
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter,configure_environment
    configure_environment()
    import mlx.core as mx
    from mlx2.runtime import qwen35_paged_graph_factory as resources_module
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from varlen_hybrid_ordinary_b2_reference import OrdinaryHybridB2Reference
    adapter=candidate=None;owners=();branches=[];prepared=[]
    initial_charge=resources_module._CHARGED
    result.update(status='running',gpu_executed=True)
    try:
        result['phase']='load_model'
        began=time.perf_counter();adapter=Qwen3827BAdapter(args.model,require_mtp=False)
        result['load_seconds']=time.perf_counter()-began
        if cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
                adapter_artifact_root=Path(adapter.identity['path']).resolve())!=identity:
            raise RuntimeError('loaded identity differs from frozen profile')
        model=getattr(adapter.model,'language_model',adapter.model)
        prompts=exact_prompts(adapter);ids=tuple(p for _,p in prompts)
        result['prompt_token_ids']=[list(p) for p in ids]
        # Ordinary reference is independent, unpadded and same input. Preserve
        # the factory's final-token vocabulary projection shape in each lane.
        reference_rows=tuple(tuple(model.make_cache()) for _ in ids);initial=[]
        began=time.perf_counter()
        for tokens,caches in zip(ids,reference_rows):
            hidden=model.model(mx.array([list(tokens)]),cache=list(caches))
            logits=model.logits(hidden[:,-1:,:])[:,-1,:]
            mx.eval(logits,*[v for c in caches for v in c.state if v is not None])
            initial.append(logits[0])
        result['serial_ordinary_prefill_seconds']=time.perf_counter()-began
        serial_rows=reference_rows;serial_initial=tuple(initial)
        began=time.perf_counter()
        reference_rows,initial=stock_mixed_reference(model,ids,mx)
        result['stock_mixed_prefill_seconds']=time.perf_counter()-began
        result['geometry_control']={'projection_rows':sum(COUNTS),'attention':'ordinary_per_segment_SDPA','gdn':'ordinary_mixed_segmented','native_dispatches':0,'final_projection_shape':[2,1,model.args.hidden_size]}
        requests=tuple((lane,adapter.identity['fingerprint'],tokens,cap)
            for lane,(tokens,cap) in enumerate(zip(ids,(2,4))))
        began=time.perf_counter()
        result['phase']='packed_prefill_factory'
        owners,candidate,boots=create_cold_packed_hybrid(adapter,requests,profile=profile,
            live_identity=identity,permit_candidate=True)
        result['packed_install_seconds']=time.perf_counter()-began
        result['packed_receipt']=candidate._packed_prefill_receipt
        result['serial_bootstrap']=compare_bootstrap(boots,serial_rows,serial_initial,candidate,mx)
        result['serial_exact_parity']='passed' if result['serial_bootstrap']['passed'] else 'failed'
        result['bootstrap']=compare_bootstrap(boots,reference_rows,initial,candidate,mx)
        reference=OrdinaryHybridB2Reference.from_row_caches(model,reference_rows,
            full_attention_layers=candidate.layer_map.full_attention,
            recurrent_layers=candidate.layer_map.recurrent,expected_offsets=COUNTS,mx=mx)
        from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation
        from mlx2.runtime.paged_native_graph_group import can_run_research_graph_b2,run_research_graph_b2
        from mlx2.runtime.generate import StopSequenceMatcher
        from types import SimpleNamespace
        sampled=[]
        def sampler(logprobs):
            value=mx.argmax(logprobs,axis=-1);sampled.append(value);return value
        continuations=tuple(NativeQwen3Continuation(uid=lane,revision=adapter.identity['fingerprint'],
            prompt_tokens=tokens,first_logits=boot.logits[0],owner=owner,candidate=candidate,
            maximum=cap,sampler=sampler,processors=[],matcher=StopSequenceMatcher([]))
            for lane,(tokens,boot,owner,cap) in enumerate(zip(ids,boots,owners,(2,4))))
        result['phase']='first_continuation_samples'
        first=tuple(lane.next() for lane in continuations)
        if any(r.mtp_receipt.get('prefill_mode')!='native_packed_prefill' or r.mtp_receipt.get('prefill_layout')!='real_rows' or r.mtp_receipt.get('native_prefill_observed_used') is not True for r in first):
            raise RuntimeError('actual first continuation packed attribution differs')
        inputs=tuple(lane._pending_token for lane in continuations)
        result['first_token_ids']=[response.token for response in first]
        result['cold_handoff_generations']=[owner._public.generation for owner in owners]
        if candidate.bootstrap_generation!=0 or result['cold_handoff_generations']!=[0,0]:
            raise RuntimeError('cold native generation-zero handoff differs')
        if not can_run_research_graph_b2(continuations):raise RuntimeError('actual cold B2 graph handoff refused')
        captured={};original=candidate.forward_staged
        def capture(lanes,branches,**kwargs):
            packed,physical=original(lanes,branches,**kwargs)
            captured.update(logits=packed,physical=physical,
                states=tuple(SimpleNamespace(recurrent_caches=b.recurrent_caches) for b in branches))
            return packed,physical
        candidate.forward_staged=capture
        began=time.perf_counter()
        result['phase']='actual_b2_q1_handoff'
        try:responses=run_research_graph_b2(continuations)
        finally:candidate.forward_staged=original
        if any(r.mtp_receipt.get('prefill_mode')!='native_packed_prefill' or r.mtp_receipt.get('prefill_layout')!='real_rows' or r.mtp_receipt.get('native_prefill_observed_used') is not True for r in responses):
            raise RuntimeError('actual B2 packed attribution differs')
        result['q1_native_seconds']=time.perf_counter()-began
        result['phase']='ordinary_b2_q1_reference'
        ordinary=reference.forward_one(inputs)
        physical=captured['physical'];validate_q1(physical,2,candidate.native_layer_count)
        result['q1_handoff']={'logits':tensor_metrics(captured['logits'],ordinary,mx),
            'recurrent':compare_slots(captured['states'],reference.merged_cache,candidate.layer_map.recurrent,mx),
            'physical':physical,'same_input_ids':inputs,'actual_sampler_calls':len(sampled),
            'generation_after_commit':[owner._public.generation for owner in owners],
            'actual_continuation_receipts':[r.mtp_receipt for r in responses],
            'actual_sampled_token_ids':[r.token for r in responses]}
        if len(sampled)!=4 or result['q1_handoff']['generation_after_commit']!=[1,1]:
            raise RuntimeError('actual paired sampler/publication handoff differs')
        result['numeric_parity']='passed' if (result['bootstrap']['passed'] and
            result['q1_handoff']['logits']['finite'] and result['q1_handoff']['logits']['exact'] and
            result['q1_handoff']['recurrent']['passed']) else 'failed'
        result['native_component_exact_parity']=result['numeric_parity']
        result['status']='geometry_exact_passed' if result['numeric_parity']=='passed' else 'geometry_numeric_difference'
    finally:
        primary_error = sys.exc_info()[1]
        if primary_error is not None:
            result['error_phase'] = result.get('phase', 'unknown')
            result['primary_error'] = repr(primary_error)
            result['primary_traceback'] = traceback.format_exc()
        try:
            for state in prepared:state.rollback()
            for branch in branches:branch.rollback()
            if candidate is not None:
                for owner in owners:owner.close()
                deadline=time.monotonic()+5
                while not candidate._serving_resources.closed and time.monotonic()<deadline:
                    candidate._serving_resources.reap();resources_module.reap_hybrid_admission_orphans();time.sleep(.005)
                writer=candidate.backend.writer
                result['cleanup']={'pending_epochs':len(writer.pending_epochs),'pending_ledger':writer.ledger.pending_count,
                    'allocated_pages':writer.pool.allocated_count,'owners_retired':all(o.fully_retired for o in owners),
                    'resources_closed':candidate._serving_resources.closed,'global_charge_restored':resources_module._CHARGED==initial_charge}
                if (writer.pending_epochs or writer.ledger.pending_count or writer.pool.allocated_count or
                    not result['cleanup']['owners_retired'] or not result['cleanup']['resources_closed'] or
                    not result['cleanup']['global_charge_restored']):
                    FAILURE_ROOTS.append((adapter,candidate,owners,branches,prepared));raise RuntimeError('packed oracle retains native resources')
            else:
                resources_module.reap_hybrid_admission_orphans()
                result['failure_cleanup']={'global_charge_restored':resources_module._CHARGED==initial_charge,
                    'orphan_resources':len(resources_module._ORPHANS)}
            if adapter is not None:adapter.close()
        except BaseException as cleanup_error:
            result['cleanup_error'] = repr(cleanup_error)
            result['cleanup_traceback'] = traceback.format_exc()
            root = (adapter, candidate, owners, branches, prepared)
            if not any(existing[1] is candidate for existing in FAILURE_ROOTS):
                FAILURE_ROOTS.append(root)
            if primary_error is None:
                raise


def supervise(args,result):
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],start_new_session=True)
    began=time.monotonic();peak=0
    try:
        while process.poll() is None:
            if time.monotonic()-began>MAX_SECONDS:raise TimeoutError('100 second packed prefill deadline')
            try:rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(process.pid)],text=True).strip() or '0')*1024
            except (subprocess.CalledProcessError,ValueError):rss=0
            peak=max(peak,rss)
            if rss>MAX_RSS:raise MemoryError('48 GiB packed prefill RSS ceiling')
            time.sleep(.2)
        if args.output.is_file():
            result=json.loads(args.output.read_text());result.update(supervisor_peak_rss_bytes=peak,supervisor_seconds=time.monotonic()-began);save(args.output,result)
        return process.returncode
    except BaseException as error:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
        result.update(status='failed',error=repr(error),worker_killed=True);save(args.output,result);return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL);parser.add_argument('--artifact-manifest',default=MANIFEST)
    parser.add_argument('--mlx-wheel',default=WHEEL);parser.add_argument('--native',default=DEFAULT_NATIVE)
    parser.add_argument('--native-sha256',default=DEFAULT_NATIVE_SHA);parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-profile',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--prefill-eval-block-size',type=int,choices=(1,),default=1);parser.add_argument('--long-fused',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();result=plan()
    result['prefill_long_nax_requested']=args.long_fused
    result['prefill_eval_block_size_requested']=args.prefill_eval_block_size
    if (args.prepare_profile or args.execute) and not args.worker:return supervise(args,result)
    code=0
    try:
        if args.prepare_profile or args.execute:execute(args,result)
        if result.get('numeric_parity')=='failed':code=1
    except BaseException as error:result.update(status='failed',error=repr(error),traceback=traceback.format_exc(),error_phase=result.get('error_phase',result.get('phase','unknown')),retained_failure_roots=len(FAILURE_ROOTS));code=1
    save(args.output,result);return code

if __name__=='__main__':raise SystemExit(main())
