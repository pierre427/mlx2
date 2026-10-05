#!/usr/bin/env python3
"""Short native K>0 shrinking-cohort and selected-state publication gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PREFIXES=(1025,1057,1089)
DECLARED=(3,2,1)
EXECUTED=(3,1,1)
MAX_SECONDS=60
FAILURE_ROOTS=[]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


@dataclass
class Box:
    leaf:object


def clone_boxes(caches):return tuple(Box(cache.leaf) for cache in caches)


def wait_writer(writer,owners,deadline):
    while writer.pending_epochs:
        if time.monotonic()>=deadline:raise TimeoutError('native writer terminal deadline')
        writer.poll_completions(wait_timeout_s=.01)
    for owner in owners:owner.poll_completions()


def run(mx,native,result):
    from mlx2.runtime.paged_gdn_checkpoint import GDNBoundaryCheckpoint
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from mlx2.runtime.paged_native_atomic_owner import (
        NativeAtomicRequestOwner,publish_native_cohort)
    from mlx2.runtime.paged_native_retirement import reap_native_request_owner
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

    profile=TokenKVProfile(4,256,'bfloat16')
    capacity=sum((prefix+depth+63)//64+1 for prefix,depth in zip(PREFIXES,DECLARED))
    pool=PagedKVPool(capacity);stream=mx.default_stream(mx.gpu)
    arena=NativeWriteBackend(capacity*profile.page_bytes,stream,
        permit_candidate=True,storage_dtype='bfloat16')
    writer=PagedKVWriteOwner(pool,arena,page_bytes=profile.page_bytes,
        permit_candidate=True)
    backend=NativeQwen3PagedBackend(writer,timeout_s=10,permit_candidate=True,
        profile_host=True)
    roots=[arena,writer,backend];owners=[];branches=[];prepared=[];published=False
    try:
        mx.random.seed(4104)
        public_layers=tuple(PagedKVTokenOwner(writer,profile,permit_candidate=True)
                            for _ in PREFIXES)
        prefix_keys=tuple(mx.random.uniform(-.25,.25,shape=(n,4,256)).astype(mx.bfloat16)
                          for n in PREFIXES)
        prefix_values=tuple(mx.random.uniform(-.25,.25,shape=(n,4,256)).astype(mx.bfloat16)
                            for n in PREFIXES)
        roots.extend((public_layers,prefix_keys,prefix_values));mx.eval(*prefix_keys,*prefix_values)
        prefix_tickets=backend.append_packed_multirow_n20(public_layers,
            mx.concatenate(prefix_keys),mx.concatenate(prefix_values),PREFIXES,
            permit_candidate=True)
        roots.append(prefix_tickets);mx.eval(prefix_tickets[0].dependency)
        wait_writer(writer,public_layers,time.monotonic()+10)
        revision='native-n20-online-ragged-device-gate-v1'
        for uid,(layer,prefix) in enumerate(zip(public_layers,PREFIXES)):
            state=(Box(mx.array([uid,0],dtype=mx.float32)),)
            checkpoint=GDNBoundaryCheckpoint(revision,uid,prefix,0,state)
            owners.append(NativeAtomicRequestOwner(revision,(layer,),
                {'gdn':(checkpoint,)},supported_planes=('kv','gdn'),enabled=True,
                checkpoint_planes=('gdn',),accepted_prefix_checkpoints=True,
                recurrent_clone=clone_boxes,lane_id=uid,reuse_private_tail=True))
        branches=[owner.begin(CandidateRequest(uid,revision,rows,('kv','gdn')))
                  for uid,(owner,rows) in enumerate(zip(owners,DECLARED))]
        histories_k=[prefix_keys[i] for i in range(3)]
        histories_v=[prefix_values[i] for i in range(3)]
        active=[0,1,2];rounds=[];all_exact=True
        names=('grouped_n20_write_count','grouped_n20_row_count',
               'q1_stock_long_n20_partial_dispatch_count',
               'q1_stock_long_n20_reduce_dispatch_count','q1_scalar_dispatch_count',
               'q1_stock_long_n20_singleton_partial_dispatch_count',
               'q1_stock_long_n20_singleton_reduce_dispatch_count')
        mx.reset_peak_memory()
        for step in range(max(EXECUTED)):
            current=tuple(uid for uid in active if step<EXECUTED[uid]);width=len(current)
            q=mx.random.uniform(-.25,.25,shape=(width,24,256)).astype(mx.bfloat16)
            k=mx.random.uniform(-.25,.25,shape=(width,4,256)).astype(mx.bfloat16)
            v=mx.random.uniform(-.25,.25,shape=(width,4,256)).astype(mx.bfloat16)
            mx.eval(q,k,v);private=tuple(branches[uid].layers[0] for uid in current)
            before=tuple(int(getattr(native,name)(arena._arena)) for name in names)
            tickets=backend.append_staged_grouped_q1_n20(private,k,v,permit_candidate=True)
            use=backend.prepare_read_n20(private,(1,)*width,query_heads=24,
                                          permit_candidate=True)
            output=backend.read_staged(use,q,tickets,scale=256**-.5);mx.eval(output)
            proofs=backend.drain_staged(private,(use,));after=tuple(
                int(getattr(native,name)(arena._arena)) for name in names)
            delta=tuple(a-b for a,b in zip(after,before))
            expected=(1,width,1,1,0,1 if width==1 else 0,1 if width==1 else 0)
            if delta!=expected or len(proofs)!=1:
                raise RuntimeError('ragged physical dispatch proof differs')
            for row,uid in enumerate(current):
                branches[uid].prove_staged_layer_read(0,row,proofs[0])
                branches[uid].recurrent_caches[0].leaf=mx.array(
                    [uid,step+1],dtype=mx.float32)
                histories_k[uid]=mx.concatenate((histories_k[uid],k[row:row+1]))
                histories_v[uid]=mx.concatenate((histories_v[uid],v[row:row+1]))
            mx.eval(*(branches[uid].recurrent_caches[0].leaf for uid in current))
            lengths=tuple(int(histories_k[uid].shape[0]) for uid in current)
            dense=max(lengths)
            dense_k=mx.stack([mx.pad(histories_k[uid],((dense-n,0),(0,0),(0,0)))
                              for uid,n in zip(current,lengths)]).transpose(0,2,1,3)
            dense_v=mx.stack([mx.pad(histories_v[uid],((dense-n,0),(0,0),(0,0)))
                              for uid,n in zip(current,lengths)]).transpose(0,2,1,3)
            mask=mx.stack([mx.arange(dense)>=dense-n for n in lengths]).reshape(width,1,1,dense)
            reference=mx.fast.scaled_dot_product_attention(
                q[:,:,None,:],dense_k,dense_v,scale=256**-.5,mask=mask).reshape(width,24,256)
            mx.eval(reference)
            exact=bool(mx.all(output.view(mx.uint16)==reference.view(mx.uint16)).item())
            all_exact=all_exact and exact
            if not exact:raise RuntimeError('ragged Q1 stock oracle differs')
            following=[]
            for uid in current:
                if step+1<EXECUTED[uid]:following.append(uid)
                else:branches[uid].stage_recurrent_prefix(
                    branches[uid].recurrent_caches,accepted_rows=step+1,
                    offset=PREFIXES[uid]+step+1)
            rounds.append({'step':step,'lane_uids':list(current),'width':width,
                           'physical_counter_delta':list(delta),'raw_bf16_equal':exact})
            active=following
        for branch,rows in zip(branches,EXECUTED):branch.seal_executed_rows(rows)
        prepared=[branch.prepare(rows) for branch,rows in zip(branches,EXECUTED)]
        publish_native_cohort(tuple(prepared));published=True
        public=[]
        for uid,owner in enumerate(owners):
            with owner.snapshot() as view:
                checkpoint=dict(view.companions)['gdn'][0]
                public.append({'uid':uid,'offset':view.offset,'generation':view.generation,
                    'gdn_offset':checkpoint.offset,'gdn_generation':checkpoint.generation,
                    'gdn_leaf':checkpoint.caches[0].leaf.tolist()})
        expected_offsets=[a+b for a,b in zip(PREFIXES,EXECUTED)]
        if ([row['offset'] for row in public]!=expected_offsets or
                [row['gdn_offset'] for row in public]!=expected_offsets or
                any(row['generation']!=1 or row['gdn_generation']!=1 for row in public)):
            raise RuntimeError('selected KV/GDN publication differs')
        result.update(status='passed',gpu_executed=True,rounds=rounds,
            declared_query_lengths=list(DECLARED),executed_query_lengths=list(EXECUTED),
            public_state=public,raw_bf16_equal=all_exact,
            grouped_round_widths=[row['width'] for row in rounds],
            peak_mlx_bytes=int(mx.get_peak_memory()),active_mlx_bytes=int(mx.get_active_memory()),
            cache_mlx_bytes=int(mx.get_cache_memory()),peak_rss_bytes=rss_bytes(),
            pending_epochs=len(writer.pending_epochs),pending_leases=writer.ledger.pending_count)
    finally:
        active_error=sys.exc_info()[0] is not None
        try:
            mx.synchronize(stream)
            if published and not writer.poisoned and not writer.pending_epochs and not writer.ledger.pending_count:
                for owner in owners:
                    owner.reap_retired();owner.close();reap_native_request_owner(owner,writer,backend)
                arena.close_after_terminal()
                result.update(owners_retired=all(owner.fully_retired for owner in owners),
                    all_pages_retired=pool.free_count==pool.capacity,free_pages=pool.free_count)
                if (not result['owners_retired'] or not result['all_pages_retired']) and not active_error:
                    raise RuntimeError('ragged gate retirement differs')
            else:
                FAILURE_ROOTS.append((roots,owners,branches,prepared));result['failure_roots_retained']=True
        except BaseException as error:
            FAILURE_ROOTS.append((roots,owners,branches,prepared));result['failure_roots_retained']=True
            result['cleanup_error']=f'{type(error).__name__}: {error}'
            if not active_error:raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True)
    parser.add_argument('--native-path',type=Path,required=True)
    parser.add_argument('--native-sha256',required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();result={'schema':'mlx2.native-n20-online-ragged-device-gate.v1',
        'status':'failed','gpu_executed':False,'qualified':False,'performance_claim':False}
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        if head!=args.source_commit or dirty:raise RuntimeError('device gate source must be exact and clean')
        if not args.native_path.is_file() or sha(args.native_path)!=args.native_sha256:
            raise RuntimeError('device gate native identity differs')
        from varlen_pack_price_bench import _gpuq_owner
        result.update(source_commit=head,native_path=str(args.native_path),
            native_sha256=args.native_sha256,gpuq_owner=_gpuq_owner())
        for key,value in {'MLX2_PAGED_PACKED_N20':'1','MLX2_PAGED_PREFILL_MATRIX':'1',
            'MLX2_PAGED_PREFILL_NAX_LONG_FUSED':'1','MLX2_PAGED_PREFILL_NAX_EXACT':'0',
            'MLX2_PAGED_Q1_STOCK_LONG':'1',
            'MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON':'1','MLX2_PAGED_Q1_SPLIT_KV':'0',
            'MLX2_PAGED_Q1_STOCK_SDPA':'0'}.items():os.environ[key]=value
        sys.path[:0]=[str(args.native_path.parent),str(ROOT/'src')]
        import _paged_kv_native as native
        import mlx.core as mx
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('hard cap')))
        signal.alarm(MAX_SECONDS);started=time.monotonic();run(mx,native,result)
        result['elapsed_seconds']=time.monotonic()-started
    except BaseException as error:
        result['error']=f'{type(error).__name__}: {error}';result['retained_failure_roots']=len(FAILURE_ROOTS)
        raise
    finally:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
