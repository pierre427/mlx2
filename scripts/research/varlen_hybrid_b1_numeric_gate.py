"""Root-run bounded B2->B1 hybrid tensor comparator, dry by default.

Actual shared native factory and atomic owners; ordinary merged caches filter
through their runtime method. Exact tensor comparison is diagnostic, not a
serving qualification. No model dtype casts or scheduler/adapter changes.
Original Apache-2.0 research script from private80a57205; reuses the existing
hybrid shared factory, HTTP tokenizer fixture and ordinary B2 cache reference.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from varlen_hybrid_http_gate import MODEL,MANIFEST,WHEEL,NATIVE,NATIVE_SHA,exact_prompts
from varlen_hybrid_serving_smoke import make_profile,save
ROOT=Path(__file__).resolve().parents[2]
MAX_SECONDS=100
MAX_RSS=48<<30
FAILURE_ROOTS=[]


def plan():
    return {'schema':'mlx2.hybrid-b1-tensor-gate.v1','status':'planned','gpu_executed':False,
        'qualified':False,'price_usable':False,'serving_selected':False,
        'numeric_parity':'not_tested','contexts':[32,96],'widths':[2,1,1],
        'same_input_reference':'stock merged B2 then runtime cache.filter([1])',
        'acceptance':'exact logits and every recurrent slot, finite and same dtype',
        'hard_seconds':MAX_SECONDS,'max_rss_bytes':MAX_RSS}


def tensor_metrics(actual,expected,mx):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise RuntimeError('numeric comparator shape/dtype differs')
    a=actual.astype(mx.float32);b=expected.astype(mx.float32)
    finite=bool(mx.all(mx.isfinite(a)&mx.isfinite(b)).item())
    delta=mx.abs(a-b)
    return {'shape':list(actual.shape),'dtype':str(actual.dtype),'finite':finite,
        'max_abs':float(mx.max(delta).item()),
        'max_rel':float(mx.max(delta/mx.maximum(mx.abs(b),1e-6)).item()),
        'exact_fraction':float(mx.mean((actual==expected).astype(mx.float32)).item()),
        'exact':bool(mx.all(actual==expected).item())}


def compare_slots(branches,reference_caches,layer_map,mx):
    rows=[]
    for ordinal,layer in enumerate(layer_map):
        reference=reference_caches[layer]
        for lane,branch in enumerate(branches):
            private=branch.recurrent_caches[ordinal]
            if len(private.cache)!=2 or len(reference.cache)!=2:raise RuntimeError('incomplete recurrent slots')
            for slot in (0,1):
                metrics=tensor_metrics(private.cache[slot],reference.cache[slot][lane:lane+1],mx)
                rows.append({'lane':lane,'layer':layer,'slot':slot,**metrics})
    return {'passed':all(row['finite'] and row['exact'] for row in rows),
        'max_abs':max(row['max_abs'] for row in rows),'slots':rows}


def validate_physical(receipt,width,depth,stock_singleton=False):
    if (receipt.get('packed_lanes')!=width or receipt.get('native_tile_dispatches')!=depth or
            receipt.get('native_stock_reduction_dispatches')!=(depth if width==2 or stock_singleton else 0) or
            receipt.get('q1_simd_stripes')!=(32 if width==2 or stock_singleton else 16) or
            receipt.get('grouped_q1_writes')!=(depth if width==2 else 0) or
            (width==1 and receipt.get('scalar_native_writes')!=receipt.get('expected_scalar_write_spans')) or
            receipt.get('native_split_partial_dispatches')!=0 or receipt.get('native_split_reduce_dispatches')!=0):
        raise RuntimeError('B2/B1 physical mechanism proof differs')


def execute(args,result):
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner']=_gpuq_owner()
    if hashlib.sha256(Path(args.native).read_bytes()).hexdigest()!=args.native_sha256:
        raise RuntimeError('frozen native hash differs')
    sys.path.insert(0,str(Path(args.native).parent));sys.path.insert(0,str(ROOT/'src'))
    from mlx2.runtime.paged_price_identity import cached_live_price_identity
    identity=cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
        adapter_artifact_root=Path(args.model).resolve())
    result['identity']=identity
    proposed=make_profile(identity,True)
    proposed['stock_singleton']=args.stock_singleton
    proposed['required_environment']['MLX2_PAGED_Q1_STOCK_SINGLETON']='1' if args.stock_singleton else '0'
    if args.prepare_profile:
        save(args.profile,proposed);result['status']='profile_prepared';return
    from mlx2.runtime.paged_hybrid_research_profile import load_hybrid_research_profile
    os.environ.update(proposed['required_environment'])
    profile=load_hybrid_research_profile(args.profile,live_identity=identity,
        context_lengths=(32,96),environment=os.environ)
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter,configure_environment
    configure_environment()
    import mlx.core as mx
    from mlx2.runtime import qwen35_paged_graph_factory as factory
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from varlen_hybrid_ordinary_b2_reference import OrdinaryHybridB2Reference
    result.update(gpu_executed=True,status='running',steps=[],numeric_parity='not_tested')
    initial_charge=factory._CHARGED
    adapter=candidate=None;owners=();branches=[];prepared=[]
    try:
        start=time.perf_counter();adapter=Qwen3827BAdapter(args.model,require_mtp=False)
        result['load_seconds']=time.perf_counter()-start
        model=getattr(adapter.model,'language_model',adapter.model)
        prompts=exact_prompts(adapter);result['prompt_token_ids']=[list(ids) for _,ids in prompts]
        if cached_live_price_identity(Path(args.artifact_manifest),Path(args.mlx_wheel),Path(args.native),
                adapter_artifact_root=Path(adapter.identity['path']).resolve())!=identity:
            raise RuntimeError('loaded model identity differs')
        requests=tuple((lane,adapter.identity['fingerprint'],ids,cap)
            for lane,((_,ids),cap) in enumerate(zip(prompts,(2,4))))
        owners,candidate,bootstrap=factory.create_shared_hybrid_graph_pack(adapter,requests,
            profile=profile,permit_candidate=True,cancelled=lambda:False)
        reference_rows=tuple(tuple(model.make_cache()) for _ in prompts)
        initial=[]
        for (_,ids),caches in zip(prompts,reference_rows):
            hidden=model.model(mx.array([list(ids)]),cache=list(caches))
            logits=model.logits(hidden[:,-1:,:])[:,-1,:]
            mx.eval(logits,*[leaf for cache in caches for leaf in cache.state if leaf is not None])
            initial.append(logits[0])
        reference=OrdinaryHybridB2Reference.from_row_caches(model,reference_rows,
            full_attention_layers=candidate.layer_map.full_attention,
            recurrent_layers=candidate.layer_map.recurrent,expected_offsets=(32,96),mx=mx)
        result['bootstrap_comparison']=[tensor_metrics(boot.logits[0],ordinary,mx)
            for boot,ordinary in zip(bootstrap,initial)]
        if not all(item['finite'] and item['exact'] for item in result['bootstrap_comparison']):
            result['numeric_parity']='failed';raise RuntimeError('last-token-projection bootstrap baseline is not exact')
        next_ids=tuple(int(mx.argmax(logit).item()) for logit in initial)
        active_owners=owners;reference_caches=reference.merged_cache
        for step,width in enumerate((2,1,1)):
            # Inputs come from ordinary logits; candidate and ordinary consume identical IDs.
            input_ids=next_ids
            probe=None
            if args.fa_probe and width==1:
                from varlen_hybrid_b1_fa_boundary_probe import B1FABoundaryProbe
                from mlx2.runtime.models import qwen38_27b as attention_module
                probe=B1FABoundaryProbe();candidate._fa_boundary_probe=probe.candidate_callback
            branches=[owner.begin(CandidateRequest(lane,adapter.identity['fingerprint'],1,('kv','gdn')))
                for lane,owner in zip((0,1) if width==2 else (1,),active_owners)]
            started=time.perf_counter()
            packed,physical=candidate.forward_staged(tuple(candidate.packed_lane((token,),branch)
                for token,branch in zip(input_ids,branches)),tuple(branches),
                permit_candidate=True,reserve_scratch=candidate.reserve_serving_scratch)
            mx.eval(packed);native_seconds=time.perf_counter()-started
            validate_physical(physical,width,candidate.native_layer_count,args.stock_singleton and width==1)
            if physical.get('native_stock_singleton_dispatches',0)!=(candidate.native_layer_count if args.stock_singleton and width==1 else 0):
                raise RuntimeError('actual singleton stock32 counter proof differs')
            started=time.perf_counter()
            if width==2:ordinary=reference.forward_one(input_ids)
            else:
                with probe.capture_ordinary(attention_module) if probe else nullcontext():
                    hidden=model.model(mx.array([[input_ids[0]]]),cache=reference_caches)
                    ordinary=model.logits(hidden)[:,-1,:]
                    mx.eval(ordinary,*[leaf for cache in reference_caches for leaf in cache.state if leaf is not None])
            ordinary_seconds=time.perf_counter()-started
            if probe:
                result.setdefault('B1_FA_boundary_probes',[]).append({'step':step,**probe.compare(tuple(branches),mx=mx)})
                candidate._fa_boundary_probe=None
            logit=tensor_metrics(packed,ordinary,mx)
            slots=compare_slots(tuple(branches),reference_caches,candidate.layer_map.recurrent,mx)
            greedy=[int(mx.argmax(packed[lane]).item())==int(mx.argmax(ordinary[lane]).item()) for lane in range(width)]
            result['steps'].append({'step':step,'width':width,'same_input_ids':list(input_ids),
                'logits':logit,'recurrent':slots,'greedy_equal':greedy,'physical':physical,
                'native_seconds':native_seconds,'ordinary_seconds':ordinary_seconds,
                'counters':candidate.backend.profile_counters_snapshot()})
            # Keep collecting bounded same-input diagnostic differences; never publish a parity claim yet.
            prepared=[branch.prepare(1) for branch in branches]
            for state in prepared:state.publish()
            branches=[];prepared=[]
            for owner in active_owners:owner.reap_retired()
            next_ids=tuple(int(mx.argmax(ordinary[lane]).item()) for lane in range(width))
            if step==0:
                owners[0].close();candidate._serving_resources.reap()
                if not owners[0].fully_retired:raise RuntimeError('completed B2 lane did not retire before B1')
                # Actual ordinary survivor operation; keeps merged-cache geometry, no fresh B1 prefill.
                for cache in reference_caches:cache.filter([1])
                next_ids=(next_ids[1],);active_owners=(owners[1],)
            save(args.output,result)
        result['physical_before_close']=candidate.backend.profile_counters_snapshot()
        result['B2_tensor_parity']='passed' if all(
            step['logits']['finite'] and step['logits']['exact'] and step['recurrent']['passed']
            for step in result['steps'] if step['width']==2) else 'failed'
        result['B1_tensor_parity']='passed' if all(
            step['logits']['finite'] and step['logits']['exact'] and step['recurrent']['passed']
            for step in result['steps'] if step['width']==1) else 'failed'
        result['numeric_parity']='passed' if (
            all(item['finite'] and item['exact'] for item in result['bootstrap_comparison']) and
            all(step['logits']['finite'] and step['logits']['exact'] and step['recurrent']['passed'] and
                all(step['greedy_equal']) for step in result['steps'])) else 'failed'
        result['status']='passed' if result['numeric_parity']=='passed' else 'numeric_difference'
    finally:
        for state in prepared:state.rollback()
        for branch in branches:branch.rollback()
        if candidate is not None:
            for owner in owners:owner.close()
            deadline=time.monotonic()+5
            while not candidate._serving_resources.closed and time.monotonic()<deadline:
                candidate._serving_resources.reap();factory.reap_hybrid_admission_orphans();time.sleep(.005)
            writer=candidate.backend.writer
            result['cleanup']={'pending_epochs':len(writer.pending_epochs),
                'pending_ledger':writer.ledger.pending_count,'allocated_pages':writer.pool.allocated_count,
                'owners_retired':all(owner.fully_retired for owner in owners),
                'resources_closed':candidate._serving_resources.closed,'global_charge_restored':factory._CHARGED==initial_charge}
            clean=(not writer.pending_epochs and not writer.ledger.pending_count and not writer.pool.allocated_count and
                result['cleanup']['owners_retired'] and result['cleanup']['resources_closed'] and factory._CHARGED==initial_charge)
            if not clean:
                FAILURE_ROOTS.append((adapter,candidate,owners,branches,prepared));raise RuntimeError('B1 comparator resources retained')
        if adapter is not None:adapter.close()


def supervise(args,result):
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],start_new_session=True)
    began=time.monotonic();peak=0
    try:
        while process.poll() is None:
            if time.monotonic()-began>MAX_SECONDS:raise TimeoutError('100 second B1 comparator deadline')
            try:rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(process.pid)],text=True).strip() or '0')*1024
            except (subprocess.CalledProcessError,ValueError):rss=0
            peak=max(peak,rss)
            if rss>MAX_RSS:raise MemoryError('48 GiB B1 comparator RSS ceiling')
            time.sleep(.2)
        if args.output.is_file():
            result=json.loads(args.output.read_text());result.update(supervisor_peak_rss_bytes=peak,supervisor_seconds=time.monotonic()-began);save(args.output,result)
        return process.returncode
    except BaseException as error:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
        if args.output.is_file():result=json.loads(args.output.read_text())
        result.update(status='failed',error=repr(error),worker_killed=True);save(args.output,result);return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL);parser.add_argument('--artifact-manifest',default=MANIFEST)
    parser.add_argument('--mlx-wheel',default=WHEEL);parser.add_argument('--native',default=NATIVE)
    parser.add_argument('--native-sha256',default=NATIVE_SHA)
    parser.add_argument('--profile',type=Path,default=Path('/tmp/mlx2-hybrid-b1-numeric-profile-1004.json'))
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-profile',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--stock-singleton',action='store_true',help='explicit short B1 stock32 candidate; flag-off fallback unchanged')
    parser.add_argument('--fa-probe',action='store_true',help='actual B1 FA3/7/55 tensors and separately labelled stock replay')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();result=plan();result['stock_singleton']=args.stock_singleton
    if (args.prepare_profile or args.execute) and not args.worker:return supervise(args,result)
    code=0
    try:
        if args.prepare_profile or args.execute:execute(args,result)
        if result.get('numeric_parity')=='failed':code=1
    except BaseException as error:result.update(status='failed',error=repr(error),retained_failure_roots=len(FAILURE_ROOTS));code=1
    save(args.output,result);return code

if __name__=='__main__':raise SystemExit(main())
