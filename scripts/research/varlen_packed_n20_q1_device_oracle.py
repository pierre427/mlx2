"""Default-off N20 long-Q1 raw BF16, generation and retirement oracle.

The GPU arm requires a root-owned GPUQ lease. It does not qualify a route or
measure model speed. B1 survivor numerical error is reported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = ((1025,), (1024, 1025), (1025, 2049, 4096),
         tuple(1025 + 31*i for i in range(20)))
LARGE_WRITE_COUNTS = tuple(7000 + i % 3 for i in range(20))
MAX_SECONDS = 90
MAX_RSS = 4 << 30
FAILURE_ROOTS: list[object] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_plan(contexts: tuple[int, ...]) -> dict:
    if (type(contexts) is not tuple or not 1 <= len(contexts) <= 20 or
            any(type(n) is not int or not 1 <= n <= 8192 for n in contexts) or
            max(contexts) <= 1024):
        raise ValueError('N20 Q1 oracle requires 1..20 bounded long visible widths')
    pages = sum((n+63)//64 for n in contexts)
    scratch = len(contexts) * 24 * 128 * 258 * 4
    if scratch > 64*1024*1024: raise ValueError('N20 Q1 scratch exceeds64MiB')
    return dict(contexts=list(contexts), batch=len(contexts),
                dense_length=max(contexts), capacity_pages=pages+len(contexts),
                scratch_bytes=scratch, expected_grouped_rows=len(contexts),
                reference='same-batch stock dense SDPA with virtual left padding and visible mask')


def bounds(start: float) -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': rss *= 1024
    if rss > MAX_RSS: raise MemoryError('N20 Q1 oracle RSS cap exceeded')
    if time.monotonic()-start > MAX_SECONDS: raise TimeoutError('N20 Q1 oracle deadline')
    return rss


def preflight(source: str, binary: Path, binary_sha: str) -> dict:
    if (not source or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,
                                               text=True).strip()!=source or
            subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,
                                    text=True).strip()):
        raise RuntimeError('N20 Q1 oracle source must be exact and clean')
    if not binary.is_file() or sha(binary)!=binary_sha:
        raise RuntimeError('N20 Q1 native binary identity differs')
    return dict(schema='mlx2.packed-n20-q1-device-oracle.v1',status='preflight_passed',
                gpu_executed=False,qualified=False,default_off=True,
                source_commit=source,native_path=str(binary),native_sha256=binary_sha,
                hard_seconds=MAX_SECONDS,max_rss_bytes=MAX_RSS,
                cases=[case_plan(c) for c in CASES],
                large_write_only={'counts':list(LARGE_WRITE_COUNTS),
                                  'rows':sum(LARGE_WRITE_COUNTS),
                                  'page_tables':sum((n+63)//64 for n in LARGE_WRITE_COUNTS),
                                  'read_tested':False})


def run_large_write_only(mx,native,start:float,result:dict)->None:
    """Exercise near-real B20 row/page metadata without a huge Q/output graph."""
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    counts=LARGE_WRITE_COUNTS
    profile=TokenKVProfile(4,256,'bfloat16')
    capacity=sum((n+63)//64 for n in counts)+20
    stream=mx.default_stream(mx.gpu)
    pool=PagedKVPool(capacity)
    arena=NativeWriteBackend(capacity*profile.page_bytes,stream,
                             permit_candidate=True,storage_dtype='bfloat16')
    result['native_arena_kv_plane_bytes']=2*arena.plane_bytes
    writer=PagedKVWriteOwner(pool,arena,page_bytes=profile.page_bytes,permit_candidate=True)
    backend=NativeQwen3PagedBackend(writer,timeout_s=10,permit_candidate=True)
    owners=tuple(PagedKVTokenOwner(writer,profile,permit_candidate=True) for _ in counts)
    roots=[arena,writer,backend,owners]
    proved=False
    try:
        with mx.stream(stream):
            mx.random.seed(4204)
            keys=mx.random.uniform(-.25,.25,shape=(sum(counts),4,256)).astype(mx.bfloat16)
            values=mx.random.uniform(-.25,.25,shape=keys.shape).astype(mx.bfloat16)
            roots.extend((keys,values))
            mx.eval(keys,values)
            before=(native.grouped_n20_write_count(arena._arena),
                    native.grouped_n20_row_count(arena._arena))
            tickets=backend.append_packed_multirow_n20(
                owners,keys,values,counts,permit_candidate=True)
            roots.append(tickets)
            mx.eval(tickets[0].dependency)
            deadline=time.monotonic()+10
            while writer.pending_epochs:
                if time.monotonic()>=deadline:raise TimeoutError('large B20 writer terminal deadline')
                writer.poll_completions(wait_timeout_s=.01)
            for owner in owners:owner.poll_completions()
            delta=(native.grouped_n20_write_count(arena._arena)-before[0],
                   native.grouped_n20_row_count(arena._arena)-before[1])
            result.update(physical_write_delta=delta[0],physical_row_delta=delta[1],
                          accepted_offsets=[o.offset for o in owners],
                          pending_epochs=writer.pending_epochs,
                          pending_leases=writer.ledger.pending_count)
            if (delta!=(1,sum(counts)) or tuple(o.offset for o in owners)!=counts or
                    writer.poisoned or writer.pending_epochs or writer.ledger.pending_count):
                raise RuntimeError('large B20 write dispatch/state/terminal proof differs')
            proved=True
            checked=0
            for lane in (0,9,19):
                table=owners[lane].accepted_handles()
                begin=sum(counts[:lane])
                for token in (0,counts[lane]-1):
                    head=lane%4
                    offset=table[token//64].page_id*profile.page_bytes+\
                           (head*64+token%64)*512
                    raw=arena.diagnostic_read(tickets[0].dependency,offset,512,
                                              permit_diagnostic=True)
                    expected=(mx.contiguous(keys[begin+token,head]).view(mx.uint8).reshape(-1),
                              mx.contiguous(values[begin+token,head]).view(mx.uint8).reshape(-1))
                    mx.eval(*raw,*expected)
                    if any(not bool(mx.all(a==b).item()) for a,b in zip(raw,expected)):
                        raise RuntimeError('large B20 raw KV spot check differs')
                    checked+=2
            result.update(raw_bf16_checks=checked,raw_bf16_equal=True,
                          rss_bytes=bounds(start),status='passed')
    finally:
        active_error=sys.exc_info()[0] is not None
        try:
            mx.synchronize(stream)
            if proved and not writer.poisoned and not writer.pending_epochs and not writer.ledger.pending_count:
                for owner in owners:owner.close()
                arena.close_after_terminal()
                result.update(owners_retired=True,all_pages_retired=pool.free_count==pool.capacity,
                              free_pages=pool.free_count)
                if not result['all_pages_retired'] and not active_error:
                    raise RuntimeError('large B20 pages retained')
            else:
                FAILURE_ROOTS.append(tuple(roots));result['failure_roots_retained']=True
        except BaseException as error:
            FAILURE_ROOTS.append(tuple(roots));result['failure_roots_retained']=True
            result['cleanup_error']=f'{type(error).__name__}: {error}'
            if not active_error:raise


def run_case(mx, native, contexts: tuple[int, ...], start: float, result: dict) -> None:
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from mlx2.runtime.paged_native_retirement import reap_native_request_owner

    n=len(contexts);plan=case_plan(contexts)
    stream=mx.default_stream(mx.gpu)
    profile=TokenKVProfile(4,256,'bfloat16')
    pool=PagedKVPool(plan['capacity_pages'])
    arena=NativeWriteBackend(pool.capacity*profile.page_bytes,stream,
                             permit_candidate=True,storage_dtype='bfloat16')
    result['native_arena_kv_plane_bytes']=2*arena.plane_bytes
    writer=PagedKVWriteOwner(pool,arena,page_bytes=profile.page_bytes,
                             permit_candidate=True)
    backend=NativeQwen3PagedBackend(writer,timeout_s=10,permit_candidate=True,
                                    profile_host=True)
    layers=tuple(PagedKVTokenOwner(writer,profile,permit_candidate=True)
                 for _ in contexts)
    roots=[arena,writer,backend,layers]
    owners=[];branches=[];prepared=[];use=None;tickets=None
    proved=False;published=False
    try:
        with mx.stream(stream):
            mx.random.seed(2204+n)
            keys=tuple(mx.random.uniform(-.25,.25,shape=(length,4,256)).astype(mx.bfloat16)
                       for length in contexts)
            values=tuple(mx.random.uniform(-.25,.25,shape=(length,4,256)).astype(mx.bfloat16)
                         for length in contexts)
            query=mx.random.uniform(-.25,.25,shape=(n,24,256)).astype(mx.bfloat16)
            roots.extend((keys,values,query))
            mx.eval(query,*keys,*values)
            prefixes=tuple(length-1 for length in contexts)
            prefix_tickets=backend.append_packed_multirow_n20(
                layers,mx.concatenate([k[:-1] for k in keys]),
                mx.concatenate([v[:-1] for v in values]),prefixes,
                permit_candidate=True)
            roots.append(prefix_tickets)
            mx.eval(prefix_tickets[0].dependency)
            deadline=time.monotonic()+10
            while writer.pending_epochs:
                if time.monotonic()>=deadline:
                    raise TimeoutError('N20 prefix writer terminal deadline')
                writer.poll_completions(wait_timeout_s=.01)
            for layer in layers:layer.poll_completions()
            result['published_prefix_offsets']=[layer.offset for layer in layers]
            if tuple(layer.offset for layer in layers)!=prefixes:
                raise RuntimeError('N20 Q1 published prefix offsets differ')
            revision='n20-q1-oracle-1004'
            owners=[NativeAtomicRequestOwner(revision,(layer,),{},supported_planes=('kv',),
                                            enabled=True,reuse_private_tail=True)
                    for layer in layers]
            branches=[owner.begin(CandidateRequest(i,revision,1,('kv',)))
                      for i,owner in enumerate(owners)]
            private=tuple(branch.layers[0] for branch in branches)
            before=(native.grouped_n20_write_count(arena._arena),
                    native.grouped_n20_row_count(arena._arena),
                    native.q1_stock_long_n20_partial_dispatch_count(arena._arena),
                    native.q1_stock_long_n20_reduce_dispatch_count(arena._arena),
                    native.q1_scalar_dispatch_count(arena._arena),
                    native.q1_stock_long_n20_singleton_partial_dispatch_count(arena._arena),
                    native.q1_stock_long_n20_singleton_reduce_dispatch_count(arena._arena))
            q1_keys=mx.stack([k[-1] for k in keys])
            q1_values=mx.stack([v[-1] for v in values])
            roots.extend((q1_keys,q1_values))
            mx.eval(q1_keys,q1_values)
            result['q1_source_layout']={
                'keys':dict(native.diagnostic_n20_source_layout(q1_keys,True)),
                'values':dict(native.diagnostic_n20_source_layout(q1_values,True)),
            }
            tickets=backend.append_staged_grouped_q1_n20(
                private,q1_keys,q1_values,permit_candidate=True)
            use=backend.prepare_read_n20(private,(1,)*n,query_heads=24,
                                         permit_candidate=True)
            output=backend.read_staged(use,query,tickets,scale=256**-.5)
            roots.extend((owners,branches,private,tickets,use,output))
            mx.eval(output)
            proofs=backend.drain_staged(private,(use,))
            after=(native.grouped_n20_write_count(arena._arena),
                   native.grouped_n20_row_count(arena._arena),
                   native.q1_stock_long_n20_partial_dispatch_count(arena._arena),
                   native.q1_stock_long_n20_reduce_dispatch_count(arena._arena),
                   native.q1_scalar_dispatch_count(arena._arena),
                   native.q1_stock_long_n20_singleton_partial_dispatch_count(arena._arena),
                   native.q1_stock_long_n20_singleton_reduce_dispatch_count(arena._arena))
            delta=tuple(a-b for a,b in zip(after,before))
            result.update(physical_counter_delta=list(delta),read_proofs=len(proofs),
                          read_terminal_success=use.terminal_succeeded,
                          private_offsets=[layer.offset for layer in private],
                          pending_epochs=writer.pending_epochs,
                          pending_leases=writer.ledger.pending_count)
            expected=(1,n,1,1,0,0,0) if n>1 else (1,1,1,1,0,1,1)
            if (delta!=expected or len(proofs)!=1 or
                    proofs[0].event!=(use.lease.epoch,True) or
                    tuple(layer.offset for layer in private)!=contexts or
                    writer.poisoned or writer.pending_epochs or writer.ledger.pending_count):
                raise RuntimeError('N20 Q1 dispatch/terminal/private state proof differs')
            proved=True
            for lane,branch in enumerate(branches):
                branch.prove_staged_layer_read(0,lane,proofs[0])
            prepared=[branch.prepare(1) for branch in branches]
            for state in prepared:state.publish()
            published=True
            branches=[];prepared=[]
            snapshot_offsets=[];snapshot_generations=[];byte_checks=0
            for lane,owner in enumerate(owners):
                snapshot=owner.snapshot()
                try:
                    snapshot_offsets.append(snapshot.offset)
                    snapshot_generations.append(snapshot.generation)
                    table=snapshot.layer_owners[0].accepted_handles()
                    for token in (0,contexts[lane]-1):
                        head=lane%4
                        offset=table[token//64].page_id*profile.page_bytes+\
                               (head*64+token%64)*256*2
                        raw=arena.diagnostic_read(tickets[0].dependency,offset,512,
                                                  permit_diagnostic=True)
                        expected_raw=(mx.contiguous(keys[lane][token,head]).view(mx.uint8).reshape(-1),
                                      mx.contiguous(values[lane][token,head]).view(mx.uint8).reshape(-1))
                        mx.eval(*raw,*expected_raw)
                        if any(not bool(mx.all(a==b).item()) for a,b in zip(raw,expected_raw)):
                            raise RuntimeError('N20 Q1 accepted KV payload differs')
                        byte_checks+=2
                    mx.synchronize(stream)
                finally:snapshot.close()
                owner.reap_retired()
            result.update(public_offsets=snapshot_offsets,
                          public_generations=snapshot_generations,
                          raw_bf16_kv_checks=byte_checks,
                          raw_bf16_kv_equal=True)
            if tuple(snapshot_offsets)!=contexts or any(g!=1 for g in snapshot_generations):
                raise RuntimeError('N20 Q1 public generation differs')
            dense_length=max(contexts)
            dense_k=mx.stack([mx.pad(k,((dense_length-length,0),(0,0),(0,0)))
                              for length,k in zip(contexts,keys)]).transpose(0,2,1,3)
            dense_v=mx.stack([mx.pad(v,((dense_length-length,0),(0,0),(0,0)))
                              for length,v in zip(contexts,values)]).transpose(0,2,1,3)
            mask=mx.stack([mx.arange(dense_length)>=dense_length-length
                           for length in contexts]).reshape(n,1,1,dense_length)
            reference=mx.fast.scaled_dot_product_attention(
                query[:,:,None,:],dense_k,dense_v,scale=256**-.5,mask=mask)
            reference=reference.reshape(n,24,256)
            roots.extend((dense_k,dense_v,mask,reference))
            mx.eval(reference)
            exact=bool(mx.all(output.view(mx.uint16)==reference.view(mx.uint16)).item())
            diff=mx.abs(output.astype(mx.float32)-reference.astype(mx.float32))
            finite=bool(mx.all(mx.isfinite(output)).item()) and bool(mx.all(mx.isfinite(reference)).item())
            result.update(output_dtype=str(output.dtype),reference_dtype=str(reference.dtype),
                          finite=finite,raw_bit_equal=exact,
                          max_abs_error=float(mx.max(diff).item()),
                          exact_fraction=float(mx.mean((output==reference).astype(mx.float32)).item()),
                          raw_mismatch_count=int(mx.sum((output.view(mx.uint16)!=reference.view(mx.uint16)).astype(mx.int32)).item()),
                          numerical_scope='B1 survivor diagnostic' if n==1 else 'same-batch raw BF16 stock')
            if not finite or not exact:
                raise RuntimeError('N20 Q1 same-batch raw BF16 stock oracle differs')
            bounds(start)
            result['status']='passed'
    finally:
        active_error=sys.exc_info()[0] is not None
        try:
            mx.synchronize(stream)
            if proved and published and not writer.poisoned and not writer.pending_epochs and not writer.ledger.pending_count:
                for owner in owners:
                    owner.close();reap_native_request_owner(owner,writer,backend)
                arena.close_after_terminal()
                result.update(owners_retired=all(owner.fully_retired for owner in owners),
                              free_pages=pool.free_count,all_pages_retired=pool.free_count==pool.capacity,
                              pending_epochs=writer.pending_epochs,
                              pending_leases=writer.ledger.pending_count)
                if (not result['owners_retired'] or not result['all_pages_retired']) and not active_error:
                    raise RuntimeError('N20 Q1 terminal-proven retirement differs')
            else:
                FAILURE_ROOTS.append(tuple(roots+[owners,branches,prepared,use,tickets]))
                result['failure_roots_retained']=True
        except BaseException as error:
            FAILURE_ROOTS.append(tuple(roots+[owners,branches,prepared,use,tickets]))
            result['cleanup_error']=f'{type(error).__name__}: {error}'
            result['failure_roots_retained']=True
            if not active_error:raise


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--native-path',type=Path,required=True)
    parser.add_argument('--native-sha256',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--large-write-only',action='store_true')
    args=parser.parse_args()
    result={'schema':'mlx2.packed-n20-q1-device-oracle.v1','status':'failed',
            'gpu_executed':False,'cases':[]}
    try:
        result.update(preflight(args.source_commit,args.native_path,args.native_sha256))
        if not args.preflight_only:
            from varlen_pack_price_bench import _gpuq_owner
            result['gpuq_owner']=_gpuq_owner()
            for key,value in {'MLX2_PAGED_PACKED_N20':'1',
                              'MLX2_PAGED_PREFILL_MATRIX':'1',
                              'MLX2_PAGED_PREFILL_NAX_LONG_FUSED':'1',
                              'MLX2_PAGED_PREFILL_NAX_EXACT':'0',
                              'MLX2_PAGED_Q1_STOCK_LONG':'1',
                              'MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON':'1',
                              'MLX2_PAGED_Q1_SPLIT_KV':'0',
                              'MLX2_PAGED_Q1_STOCK_SDPA':'0'}.items():os.environ[key]=value
            os.environ.pop('MLX_SDPA_BLOCKS',None)
            sys.path.insert(0,str(args.native_path.parent))
            sys.path.insert(0,str(ROOT/'src'))
            import _paged_kv_native as native
            if Path(native.__file__).resolve()!=args.native_path.resolve():
                raise RuntimeError('N20 Q1 native import path differs')
            cap=native.packed_n20_capability()
            if (cap.get('version')!=1 or cap.get('max_spans')!=20 or
                    cap.get('q1_read_dispatches')!=2 or
                    cap.get('b1_stock_long_dispatches')!=2 or
                    cap.get('b1_stock_long_selector')!='MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON'):
                raise RuntimeError('N20 Q1 capability differs')
            import mlx.core as mx
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('N20 Q1 hard cap')))
            signal.alarm(MAX_SECONDS)
            start=time.monotonic()
            result['gpu_executed']=True
            if args.large_write_only:
                case={'scope':'20-lane near-real 140k-row grouped writer; no attention read',
                      'counts':list(LARGE_WRITE_COUNTS),'status':'running'}
                result['cases'].append(case)
                try:run_large_write_only(mx,native,start,case)
                except BaseException as error:
                    case['status']='failed';case['error']=f'{type(error).__name__}: {error}'
                    raise
            else:
                for contexts in CASES:
                    case=case_plan(contexts)
                    case['status']='running'
                    result['cases'].append(case)
                    try:run_case(mx,native,contexts,start,case)
                    except BaseException as error:
                        case['status']='failed'
                        case['error']=f'{type(error).__name__}: {error}'
                        raise
            result['status']='passed'
            result['elapsed_seconds']=time.monotonic()-start
            result['peak_rss_bytes']=bounds(start)
    except BaseException as error:
        result['status']='failed'
        result['error']=f'{type(error).__name__}: {error}'
        result['retained_failure_roots']=len(FAILURE_ROOTS)
        raise
    finally:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
