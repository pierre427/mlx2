"""Default-off segmented-causal BF16 matrix reader oracle; root owns GPU.

No runtime import occurs before explicit execution, clean source and dual lease
validation. This is attention-layer numerical/ownership evidence only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
BINARY=Path('/tmp/mlx2-prefill-matrix-short-fix-build-1004/_paged_kv_native.cpython-312-darwin.so')
BINARY_SHA='31f1f0500b4a40f6351e157befc22b789667ee80c9f7d4d90f57285ccbc861ed'
CASES=(((17,33),(0,0)),((65,97),(0,0)),((34,65),(31,7)))
MAX_SECONDS=60
MAX_RSS=4<<30
FAILURE_ROOTS=[]
NAX_BINARY=Path('/tmp/mlx2-prefill-nax-exact-build-1004/_paged_kv_native.cpython-312-darwin.so')
NAX_BINARY_SHA='b3e9d0878ae1048f0337bc6b45a5982058eae74580ad70e66186299cb4ed2ca0'
NAX_EXACT=False


def case_plan(counts,prefixes):
    if (len(counts)!=2 or len(prefixes)!=2 or any(type(n)is not int or not 9<=n<=1023 for n in counts)
            or any(type(n)is not int or n<0 for n in prefixes)
            or any(n+p>8192 for n,p in zip(counts,prefixes))):
        raise ValueError('two bounded causal spans required')
    capacity=sum((n+p+63)//64+2 for n,p in zip(counts,prefixes))
    return dict(counts=list(counts),prefixes=list(prefixes),total_rows=sum(counts),
                capacity_pages=capacity,plane_bytes=capacity*4*64*256*2,
                expected_matrix_dispatches=1,global_scratch_bytes=0,threadgroup_bytes=20736)


def preflight():
    return dict(schema='mlx2.prefill-matrix-device-oracle.v1',status='planned',gpu_executed=False,
                qualified=False,default_off=True,dtype='bfloat16',query_heads=24,kv_heads=4,head_dim=256,
                native_binary=str(BINARY),native_sha256=BINARY_SHA,hard_seconds=MAX_SECONDS,max_rss_bytes=MAX_RSS,
                reference='per-span stock causal masked multi-query SDPA; no lane padding or scalar native proof',
                numerical_gate={'atol':.00025,'rtol':.01,'raw_bit_exact_required':False},
                cases=[case_plan(c,p) for c,p in CASES])


def bounds(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    if rss>MAX_RSS:raise MemoryError('oracle4GiB RSS ceiling')
    if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('oracle60second deadline')
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


def execute_case(mx,native,plan,start,receipt):
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    from mlx2.runtime.paged_attention_pack import prepare_staged_token_read
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
            if any(plan['prefixes']):
                query=mx.random.uniform(-.25,.25,shape=(24,sum(plan['counts']),512)).astype(mx.bfloat16)[:,:,::2].transpose(1,0,2)
            else:
                query=mx.random.uniform(-.25,.25,shape=(sum(plan['counts']),24,256)).astype(mx.bfloat16)
            mx.eval(query,*keys,*values)
            receipt['query_layout_requested']='head_major_dim_stride2' if any(plan['prefixes']) else 'row_major'
            receipt['physical_strides_validated_in_native_eval']=True
            for lane,prefix in enumerate(plan['prefixes']):
                if prefix:backend.append_completed((layers[lane],),keys[lane][:prefix],values[lane][:prefix],(prefix,))
            before=native.prefill_matrix_dispatch_count(arena._arena)
            before_nax=[getattr(native,'prefill_nax_'+stage+'_dispatch_count')(arena._arena) for stage in ('score','softmax','value')] if NAX_EXACT else None
            tick=time.monotonic()
            tickets=backend.append_staged(layers,mx.concatenate([k[p:] for k,p in zip(keys,plan['prefixes'])]),
                                         mx.concatenate([v[p:] for v,p in zip(values,plan['prefixes'])]),tuple(plan['counts']))
            use=prepare_staged_token_read(layers,tuple(plan['counts']),query_heads=24,permit_candidate=True)
            output=backend.read_staged(use,query,tickets,scale=256**-.5)
            mx.eval(output)
            proofs=backend.drain_staged(layers,(use,))
            receipt['append_read_terminal_seconds']=time.monotonic()-tick
            delta=native.prefill_matrix_dispatch_count(arena._arena)-before
            receipt['matrix_dispatch_delta']=delta
            if NAX_EXACT:
                physical=[getattr(native,'prefill_nax_'+stage+'_dispatch_count')(arena._arena)-value for stage,value in zip(('score','softmax','value'),before_nax)]
                receipt['nax_physical_dispatch_deltas']=physical
                if physical!=[1,1,1]:raise RuntimeError('NAX three-stage physical proof differs')
            if delta!=1 or len(proofs)!=1 or proofs[0].event!=(use.lease.epoch,True):
                raise RuntimeError('matrix physical dispatch/matched terminal proof differs')
            if tuple(layer.offset for layer in layers)!=tuple(ends):raise RuntimeError('cold staged offsets differ')
            if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count:
                raise RuntimeError('oracle native work remains after drain')
            terminal_proven=True
            references=[];begin=0
            for n,p,k,v in zip(plan['counts'],plan['prefixes'],keys,values):
                q=query[begin:begin+n].transpose(1,0,2)[None]
                # Actual offset causal bounds, including a preexisting prefix.
                mask=(mx.arange(p+n)[None,:]<=mx.arange(p,p+n)[:,None])[None,None]
                refs=mx.fast.scaled_dot_product_attention(q,k.transpose(1,0,2)[None],v.transpose(1,0,2)[None],
                                                         scale=256**-.5,mask=mask)
                references.append(refs[0].transpose(1,0,2));begin+=n
            reference=mx.concatenate(references);mx.eval(reference)
            difference=mx.abs(output.astype(mx.float32)-reference.astype(mx.float32))
            finite=bool(mx.all(mx.isfinite(output)).item()) and bool(mx.all(mx.isfinite(reference)).item())
            close=bool(mx.all(difference<=.00025+.01*mx.abs(reference.astype(mx.float32))).item())
            receipt.update(output_dtype=str(output.dtype),finite=finite,allclose=close,
                           max_abs_error=float(mx.max(difference).item()),
                           exact_elements_fraction=float(mx.mean((output==reference).astype(mx.float32)).item()),
                           raw_bit_equal=bool(mx.all(output.view(mx.uint16)==reference.view(mx.uint16)).item()),
                           read_terminal_epoch=use.lease.epoch,read_terminal_success=use.terminal_succeeded)
            if output.dtype!=mx.bfloat16 or not finite or not close or (NAX_EXACT and not receipt['raw_bit_equal']):raise RuntimeError('matrix numerical oracle differs')
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
    os.environ['MLX2_PAGED_PREFILL_NAX_EXACT']='1' if NAX_EXACT else '0'
    sys.path.insert(0,str(BINARY.parent));sys.path.insert(0,str(ROOT/'src'))
    import _paged_kv_native as native
    if Path(native.__file__).resolve()!=BINARY.resolve():raise RuntimeError('native import path differs')
    capability=native.prefill_matrix_capability()
    if (capability.get('head_dim')!=256 or capability.get('storage_dtype')!='bfloat16'
            or capability.get('segmented_causal') is not True or capability.get('version')!=2
            or capability.get('max_query_count')!=1023 or capability.get('passes')!=2):raise RuntimeError('matrix capability differs')
    if NAX_EXACT:
        exact_cap=native.prefill_nax_capability()
        if exact_cap.get('scratch_bound_bytes')!=3195072 or exact_cap.get('physical_dispatches')!=3:
            raise RuntimeError('exact NAX charged capability differs')
        result['nax_capability']=exact_cap
        for plan in result['cases']:
            if any(n>129 or n+p>129 for n,p in zip(plan['counts'],plan['prefixes'])):raise RuntimeError('exact NAX short bounds differ')
            plan['charged_scratch_bytes']=plan['total_rows']*24*129*2*2
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():raise RuntimeError('native GPU required')
    result.update(gpu_executed=True,source_commit=actual,capability=capability,device=dict(mx.device_info()))
    try:
        for plan in result['cases']:execute_case(mx,native,plan,start,plan)
        result.update(status='passed',total_seconds=time.monotonic()-start,peak_rss_bytes=bounds(start))
    finally:signal.alarm(0)


def main():
    global BINARY,BINARY_SHA,NAX_EXACT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nax-exact',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--expected-source')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();NAX_EXACT=args.nax_exact
    if NAX_EXACT:BINARY=NAX_BINARY;BINARY_SHA=NAX_BINARY_SHA
    result=preflight();code=0
    result['nax_exact_candidate']=NAX_EXACT
    if NAX_EXACT:
        result['numerical_gate']['raw_bit_exact_required']=True
        for plan in result['cases']:
            plan['global_scratch_bytes']=plan['charged_scratch_bytes']=plan['total_rows']*24*129*2*2
            plan['threadgroup_bytes']=256
    try:
        if args.execute:execute(result,args.expected_source)
    except BaseException as exc:result.update(status='failed',error=repr(exc),retained_failure_roots=len(FAILURE_ROOTS));code=1
    args.output.write_text(json.dumps(result,indent=2)+'\n');return code
if __name__=='__main__':raise SystemExit(main())
