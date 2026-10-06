"""Default-off fused paged BF16 long NAX reader oracle; root owns GPU.

No runtime import occurs before explicit execution, clean source and dual lease
validation. This is attention-layer numerical/ownership evidence only.
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
ROOT=Path(__file__).resolve().parents[2]
BINARY=Path('/tmp/mlx2-prefill-long-nax-build-1004/_paged_kv_native.cpython-312-darwin.so')
BINARY_SHA='713550511e98213b62793d8df2d9eb1c77062c5264503980099e2ef0daf8015a'
CASES=(((256,257),(0,0)),((320,321),(31,7)),((1024,1025),(0,0)))
MAX_SECONDS=90
MAX_RSS=4<<30
FAILURE_ROOTS=[]


def case_plan(counts,prefixes):
    if (len(counts)!=2 or len(prefixes)!=2 or any(type(n)is not int or not 256<=n<=8192 for n in counts)
            or any(type(n)is not int or n<0 for n in prefixes)
            or any(n+p>8192 for n,p in zip(counts,prefixes))):
        raise ValueError('two bounded causal spans required')
    capacity=sum((n+p+63)//64+2 for n,p in zip(counts,prefixes))
    return dict(counts=list(counts),prefixes=list(prefixes),total_rows=sum(counts),
                capacity_pages=capacity,plane_bytes=capacity*4*64*256*2,
                expected_long_nax_dispatches=1,global_scratch_bytes=0,threadgroup_bytes=16384)


def preflight():
    return dict(schema='mlx2.prefill-long-nax-device-oracle.v1',status='planned',gpu_executed=False,
                qualified=False,default_off=True,dtype='bfloat16',query_heads=24,kv_heads=4,head_dim=256,
                native_binary=str(BINARY),native_sha256=BINARY_SHA,hard_seconds=MAX_SECONDS,max_rss_bytes=MAX_RSS,
                reference='same-operand per-span forced-fused stock causal SDPA; padded B2 and default stock are diagnostics',
                numerical_gate={'atol':0,'rtol':0,'raw_bit_exact_required':True},
                cases=[case_plan(c,p) for c,p in CASES])


def bounds(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    if rss>MAX_RSS:raise MemoryError('oracle4GiB RSS ceiling')
    if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('oracle90second deadline')
    return rss


def retire_proven_case(synchronize,layers,writer,arena,pool,receipt):
    # Quiesce reference and native graph roots before releasing private pages.
    # Called only after exact successful read proof and drained write epochs.
    synchronize()
    if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count:
        raise RuntimeError('oracle quiescence has poison or pending native work')
    for layer in layers:layer.close()
    arena.close_after_terminal()
    receipt.update(pending_write_epochs=0,pending_ledger=0,free_pages=pool.free_count,
                   owners_unattached=True,global_scratch_charge_bytes=0,scratch_charge_released_bytes=receipt.get('charged_scratch_bytes',0),
                   retirement_after_proven_quiescence=True)
    if pool.free_count!=pool.capacity:raise RuntimeError('oracle physical pages retained')


def prepare_long_staged_read(owners, counts):
    """Oracle-only lease with a widened *legacy scalar* validation ceiling.

    The fused NAX read allocates zero global scratch. The generic plan's
    65,536-work-item proxy applies to its scalar kernel and rejects long
    prompts before the native selector. This helper preserves every span,
    page-generation and lease check, changing only the proxy ceiling.
    """
    from mlx2.runtime.paged_attention_pack import PackedTokenRead
    from mlx2.runtime.paged_attention_plan import PagedAttentionPlan,SequenceSpan
    from mlx2.runtime.paged_attention_metal import build_paged_read_metadata
    writer,profile=owners[0].writer,owners[0].profile
    spans=[];table=[];row_begin=0
    for owner,count in zip(owners,counts):
        handles=owner.staged_handles(count)
        sequence=owner.sequence
        end=owner.offset+count
        spans.append(SequenceSpan(row_begin,count,owner.offset,end,
            sequence.retained_start,sequence.first_block,len(table),len(handles),
            max(handle.generation for handle in handles),'causal',None))
        table.extend(handles);row_begin+=count
    unique=tuple(dict.fromkeys(table))
    lease=writer.ledger.prepare(unique)
    try:
        plan=PagedAttentionPlan(spans=tuple(spans),page_table=tuple(table),
            total_rows=row_begin,query_heads=24,kv_heads=profile.kv_heads,
            head_dim=profile.head_dim,dtype=profile.dtype,
            pool_capacity=writer.pool.capacity,
            live_generations={h.page_id:h.generation for h in unique},
            max_work_items=16_777_216,max_scratch_bytes=8<<30)
        metadata=build_paged_read_metadata(plan)
    except BaseException:
        writer.ledger.abort_before_submit(lease)
        raise
    return PackedTokenRead(writer,plan,metadata,lease)


def execute_case(mx,native,plan,start,receipt):
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    stream=mx.default_stream(mx.gpu);profile=TokenKVProfile(4,256,'bfloat16')
    pool=PagedKVPool(plan['capacity_pages'])
    arena=NativeWriteBackend(plan['plane_bytes'],stream,permit_candidate=True,storage_dtype='bfloat16')
    writer=PagedKVWriteOwner(pool,arena,page_bytes=profile.page_bytes,permit_candidate=True)
    backend=NativeQwen3PagedBackend(writer,timeout_s=5,permit_candidate=True,profile_host=True)
    layers=tuple(PagedKVTokenOwner(writer,profile,permit_candidate=True) for _ in range(2))
    keys=values=query=output=reference=use=tickets=None;terminal_proven=False;numeric_passed=False
    try:
        with mx.stream(stream):
            mx.random.seed(1004)
            ends=[n+p for n,p in zip(plan['counts'],plan['prefixes'])]
            keys=[mx.random.uniform(-.25,.25,shape=(n,4,256)).astype(mx.bfloat16) for n in ends]
            values=[mx.random.uniform(-.25,.25,shape=(n,4,256)).astype(mx.bfloat16) for n in ends]
            query=mx.random.uniform(-.25,.25,shape=(sum(plan['counts']),24,256)).astype(mx.bfloat16)
            mx.eval(query,*keys,*values)
            receipt['query_layout_requested']='row_major_unit_channel_stride'
            receipt['physical_strides_validated_in_native_eval']=True
            for lane,prefix in enumerate(plan['prefixes']):
                if prefix:backend.append_completed((layers[lane],),keys[lane][:prefix],values[lane][:prefix],(prefix,))
            before=native.prefill_long_nax_dispatch_count(arena._arena)
            before_matrix=native.prefill_matrix_dispatch_count(arena._arena)
            before_write=native.grouped_multirow_write_count(arena._arena)

            tick=time.monotonic()
            tickets=backend.append_packed_multirow(layers,
                 mx.concatenate([k[p:] for k,p in zip(keys,plan['prefixes'])]),
                 mx.concatenate([v[p:] for v,p in zip(values,plan['prefixes'])]),
                 tuple(plan['counts']),permit_candidate=True)
            use=prepare_long_staged_read(layers,tuple(plan['counts']))
            output=backend.read_staged(use,query,tickets,scale=256**-.5)
            mx.eval(output)
            proofs=backend.drain_staged(layers,(use,))
            receipt['append_read_terminal_seconds']=time.monotonic()-tick
            delta=native.prefill_long_nax_dispatch_count(arena._arena)-before
            receipt['long_nax_dispatch_delta']=delta
            receipt['matrix_dispatch_delta']=native.prefill_matrix_dispatch_count(arena._arena)-before_matrix
            receipt['packed_write_dispatch_delta']=native.grouped_multirow_write_count(arena._arena)-before_write
            if receipt['packed_write_dispatch_delta']!=1:
                raise RuntimeError('packed multirow physical write dispatch differs')
            if delta!=1 or receipt['matrix_dispatch_delta']!=1 or len(proofs)!=1 or proofs[0].event!=(use.lease.epoch,True):
                raise RuntimeError('long NAX physical dispatch/matched terminal proof differs')
            if tuple(layer.offset for layer in layers)!=tuple(ends):raise RuntimeError('cold staged offsets differ')
            if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count:
                raise RuntimeError('oracle native work remains after drain')
            terminal_proven=True
            # Reference is one stock B2 launch with virtual left padding, not
            # independent B1 launches whose dispatch policy may differ.
            q_length=max(plan['counts']);kv_length=max(ends);qs=[];ks=[];vs=[];masks=[];begin=0
            for n,p,k,v in zip(plan['counts'],plan['prefixes'],keys,values):
                qpad=q_length-n;kvpad=kv_length-(p+n)
                q=query[begin:begin+n].transpose(1,0,2)
                qs.append(mx.pad(q,((0,0),(qpad,0),(0,0))))
                ks.append(mx.pad(k.transpose(1,0,2),((0,0),(kvpad,0),(0,0))))
                vs.append(mx.pad(v.transpose(1,0,2),((0,0),(kvpad,0),(0,0))))
                rows=mx.arange(q_length)[:,None];cols=mx.arange(kv_length)[None,:]
                causal=cols<=kvpad+p+rows-qpad
                masks.append(((rows>=qpad)&(cols>=kvpad)&causal)|
                             ((rows<qpad)&(cols==kvpad)))
                begin+=n
            dense_q,dense_k,dense_v=mx.stack(qs),mx.stack(ks),mx.stack(vs)
            dense_mask=mx.stack(masks)[:,None]
            default_dense=mx.fast.scaled_dot_product_attention(dense_q,dense_k,dense_v,
                    scale=256**-.5,mask=dense_mask)
            dense=mx.fast.scaled_dot_product_attention(dense_q,dense_k,dense_v,
                    scale=256**-.5,mask=dense_mask,force_fused=True)
            padded_reference=mx.concatenate([dense[i,:,q_length-n:,:].transpose(1,0,2)
                                      for i,n in enumerate(plan['counts'])]);mx.eval(padded_reference)
            default_reference=mx.concatenate([default_dense[i,:,q_length-n:,:].transpose(1,0,2)
                                      for i,n in enumerate(plan['counts'])]);mx.eval(default_reference)
            lane_refs=[];begin=0
            for n,k,v in zip(plan['counts'],keys,values):
                q=query[begin:begin+n].transpose(1,0,2)[None]
                lane=mx.fast.scaled_dot_product_attention(
                    q,k.transpose(1,0,2)[None],v.transpose(1,0,2)[None],
                    scale=256**-.5,mask='causal',force_fused=True)
                lane_refs.append(lane[0].transpose(1,0,2));begin+=n
            reference=mx.concatenate(lane_refs);mx.eval(reference)
            difference=mx.abs(output.astype(mx.float32)-reference.astype(mx.float32))
            finite=bool(mx.all(mx.isfinite(output)).item()) and bool(mx.all(mx.isfinite(reference)).item())
            close=bool(mx.all(output.view(mx.uint16)==reference.view(mx.uint16)).item())
            receipt.update(output_dtype=str(output.dtype),finite=finite,allclose=close,
                           max_abs_error=float(mx.max(difference).item()),
                           exact_elements_fraction=float(mx.mean((output==reference).astype(mx.float32)).item()),
                           raw_bit_equal=bool(mx.all(output.view(mx.uint16)==reference.view(mx.uint16)).item()),
                           padded_fused_raw_equal=bool(mx.all(output.view(mx.uint16)==padded_reference.view(mx.uint16)).item()),
                           padded_fused_exact_fraction=float(mx.mean((output==padded_reference).astype(mx.float32)).item()),
                           default_padded_stock_raw_equal=bool(mx.all(output.view(mx.uint16)==default_reference.view(mx.uint16)).item()),
                           read_terminal_epoch=use.lease.epoch,read_terminal_success=use.terminal_succeeded)
            if output.dtype!=mx.bfloat16 or not finite or not close or not receipt['raw_bit_equal']:raise RuntimeError('long fused NAX raw BF16 numerical oracle differs')
            bounds(start);numeric_passed=True
    finally:
        active_error=sys.exc_info()[0] is not None
        if terminal_proven:
            try:
                retire_proven_case(lambda:mx.synchronize(stream),layers,writer,arena,pool,receipt)
                receipt['failure_retirement_after_proven_quiescence']=not numeric_passed
            except BaseException as exc:
                receipt['retirement_error']=repr(exc)
                FAILURE_ROOTS.append((arena,writer,backend,layers,keys,values,query,output,reference,use,tickets))
                if not active_error:raise
        if not terminal_proven:
            # Ambiguous or failed native terminals cannot release private pages.
            FAILURE_ROOTS.append((arena,writer,backend,layers,keys,values,query,output,reference,use,tickets))


def execute(result,expected_source):
    start=time.monotonic()
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if not expected_source or actual!=expected_source or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('exact clean oracle source required')
    from varlen_pack_price_bench import _gpuq_owner
    result['gpuq_owner']=_gpuq_owner()
    if hashlib.sha256(BINARY.read_bytes()).hexdigest()!=BINARY_SHA:raise RuntimeError('native binary hash differs')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('oracle hard deadline')))
    signal.alarm(MAX_SECONDS)
    os.environ['MLX2_PAGED_PREFILL_MATRIX']='1'
    os.environ['MLX2_PAGED_PREFILL_NAX_EXACT']='0'
    os.environ['MLX2_PAGED_PREFILL_NAX_LONG_FUSED']='1'
    os.environ['MLX2_PAGED_GROUPED_MULTIROW_WRITE']='1'
    sys.path.insert(0,str(BINARY.parent));sys.path.insert(0,str(ROOT/'src'))
    import _paged_kv_native as native
    if Path(native.__file__).resolve()!=BINARY.resolve():raise RuntimeError('native import path differs')
    capability=native.prefill_long_nax_capability()
    if (capability.get('version')!=1 or capability.get('max_query_count')!=8192 or
        capability.get('max_spans')!=2 or capability.get('max_causal_end')!=8192 or
        capability.get('physical_dispatches')!=1 or capability.get('scratch_bytes')!=0 or
        capability.get('selector')!='MLX2_PAGED_PREFILL_NAX_LONG_FUSED'):
        raise RuntimeError('fused long NAX capability differs')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():raise RuntimeError('native GPU required')
    result.update(gpu_executed=True,source_commit=actual,capability=capability,device=dict(mx.device_info()))
    try:
        for plan in result['cases']:execute_case(mx,native,plan,start,plan)
        result.update(status='passed',total_seconds=time.monotonic()-start,peak_rss_bytes=bounds(start))
    finally:signal.alarm(0)


def main():
    global BINARY,BINARY_SHA
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-long',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--expected-source')
    parser.add_argument('--native-binary',type=Path,default=BINARY)
    parser.add_argument('--native-sha256',default=BINARY_SHA)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    BINARY=args.native_binary;BINARY_SHA=args.native_sha256
    result=preflight();code=0
    if args.include_long:
        result['cases'].append(case_plan((6950,6929),(0,0)))
    try:
        if args.execute:execute(result,args.expected_source)
    except BaseException as exc:result.update(status='failed',error=repr(exc),retained_failure_roots=len(FAILURE_ROOTS));code=1
    args.output.write_text(json.dumps(result,indent=2)+'\n');return code
if __name__=='__main__':raise SystemExit(main())
