"""Root-run B20 attention-only paired oracle; no runtime imports during preflight."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import zipfile
ROOT=Path(__file__).resolve().parents[2]
MAX_SECONDS=100
MAX_BYTES=12<<30
CACHE_BYTES=256<<20
NATIVE=Path('/tmp/mlx2-n20-q1-b1-stock-build-1004/_paged_kv_native.cpython-312-darwin.so')
NATIVE_SHA='2ddf5b91a88bbf8f451c1eaaae9919bdff1e6effd161262b7df61c4e5d2cf350'
WHEEL=Path.home()/'.cache/uv/sdists-v9/path/6b22317da775785a/d5Co9cAdGp8_XULv/mlx-0.32.2.dev20260919+39400a0d4-cp312-cp312-macosx_26_0_arm64.whl'
WHEEL_SHA='9cf6fc8312845b233a10092c35d6575870d2fabedaa9bc04f38f8c41394eee48'
VERSION='0.32.2.dev20260919+39400a0d4'
INPUTS=Path('/tmp/mlx2-spomin400-nativeN-inputs.json')
INPUTS_SHA='d5271584b7e215b71eae4c56950d35e0f3c8b197b1cb6a9f325dacc31ee96d0a'
ENV={'MLX2_PAGED_PACKED_N20':'1','MLX2_PAGED_PREFILL_MATRIX':'1',
     'MLX2_PAGED_PREFILL_NAX_LONG_FUSED':'1','MLX2_PAGED_PREFILL_NAX_EXACT':'0',
     'MLX2_PAGED_Q1_STOCK_LONG':'1','MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON':'1',
     'MLX2_PAGED_Q1_SPLIT_KV':'0','MLX2_PAGED_Q1_STOCK_REDUCTION':'0',
     'MLX2_PAGED_Q1_STOCK_SINGLETON':'0'}
FAILURE_ROOTS=[]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def source_check(expected,root=ROOT):
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual!=expected or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('attention oracle requires exact clean source')
    return actual


def geometry(document,domain):
    if (document.get('schema')!='mlx2.spomin-400case-native-inputs.v1' or
            document.get('case_count')!=400 or document.get('cases_per_domain')!=20 or
            document.get('client_concurrency')!=20 or len(document.get('rows',()))!=400 or
            len(document.get('domain_order',()))!=20 or domain not in document['domain_order']):
        raise ValueError('actual frozen400-case B20 input scope required')
    rows=[r for r in document['rows'] if r['domain']==domain]
    counts=tuple(r['prompt_tokens'] for r in rows)
    if (len(rows)!=20 or len({r['case_id'] for r in rows})!=20 or
            any(type(n)is not int or not 256<=n<=8192 or
                len(r['prompt_token_ids'])!=n for r,n in zip(rows,counts)) or
            sum(counts)>163840):raise ValueError('actual domain20 row geometry differs')
    pages=sum((n+63)//64 for n in counts)+20
    if pages>2560:raise ValueError('native N20 page capacity differs')
    return {'domain':domain,'counts':list(counts),'case_ids':[r['case_id'] for r in rows],
            'total_real_rows':sum(counts),'capacity_pages':pages,
            'native_arena_kv_plane_bytes':2*pages*4*64*256*2,
            'query_heads':24,'kv_heads':4,'head_dim':256,'dtype':'bfloat16',
            'score_scratch_bytes':0}


def wheel_members(path):
    with zipfile.ZipFile(path) as archive:
        members={n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()
                 if n.startswith('mlx/') and (n.endswith('.so') or n.endswith('.dylib'))}
    if not any(n.startswith('mlx/core.') and n.endswith('.so') for n in members):
        raise ValueError('pinned wheel core extension absent')
    return members


def preflight(source,native,native_sha,wheel,wheel_sha,inputs,inputs_sha,domain,root=ROOT):
    source_check(source,root)
    for path,expected,label in ((native,native_sha,'native'),(wheel,wheel_sha,'wheel'),(inputs,inputs_sha,'inputs')):
        if not Path(path).is_file() or sha(path)!=expected:raise RuntimeError(label+' hash differs')
    document=json.loads(Path(inputs).read_text())
    return {'schema':'mlx2.n20-attention-controlled-perf.v1','status':'preflight_passed',
            'source_commit':source,'native_path':str(Path(native).resolve()),'native_sha256':native_sha,
            'mlx_wheel_path':str(Path(wheel).resolve()),'mlx_wheel_sha256':wheel_sha,
            'required_mlx_version':VERSION,'mlx_wheel_binary_members':wheel_members(wheel),
            'prepared_inputs_path':str(Path(inputs).resolve()),'prepared_inputs_raw_sha256':inputs_sha,
            'prepared_inputs_declared_payload_sha256':document.get('inputs_sha256'),
            'geometry':geometry(document,domain),'required_environment':dict(ENV),
            'schedule':[['stock','native'],['native','stock'],['stock','native'],['native','stock']],
            'warmup_pairs':1,'measured_pairs':3,'hard_seconds':MAX_SECONDS,'max_rss_bytes':MAX_BYTES,
            'max_mlx_active_bytes':MAX_BYTES,'mlx_cache_limit_bytes':CACHE_BYTES,
            'scope':'attention-only; setup KV write and raw oracles excluded from arm timings',
            'evaluation_scope':'MLX evaluation includes submission/completion; not device-only time',
            'synthetic_operand_seed':20261004,'synthetic_operand_range':[-.25,.25],
            'operand_scope':'bounded synthetic BF16 QKV; prompt source determines actual ragged geometry',
            'q1_policy_selected':'explicit N20 B1 stock-long two-pass; not exercised',
            'raw_bf16_exact_required':True,'qualified':False,'price_usable':False,'gpu_executed':False}


def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


def memory(mx):
    return {'process_peak_rss_bytes':rss_bytes(),'mlx_active_bytes':int(mx.get_active_memory()),
            'mlx_peak_bytes':int(mx.get_peak_memory()),'mlx_cache_bytes':int(mx.get_cache_memory())}


def bounds(mx,start):
    state=memory(mx)
    if state['process_peak_rss_bytes']>MAX_BYTES or state['mlx_active_bytes']>MAX_BYTES:
        raise MemoryError('attention oracle12GiB RSS/MLX-active guard exceeded')
    if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('attention oracle100s guard')
    return state


def verify_loaded_wheel(mx,receipt):
    if importlib.metadata.version('mlx')!=VERSION:raise RuntimeError('installed MLX version differs')
    package=Path(mx.__file__).resolve().parent
    observed={}
    for member,expected in receipt['mlx_wheel_binary_members'].items():
        path=package/member.removeprefix('mlx/')
        if not path.is_file() or sha(path)!=expected:raise RuntimeError('loaded MLX wheel member differs: '+member)
        observed[member]=expected
    receipt.update(loaded_mlx_core_path=str(Path(mx.__file__).resolve()),loaded_mlx_binary_members=observed)


def terminal_proof(result,use,proofs,writer,owners,counts,counters):
    if (counters!=(1,sum(counts),1) or len(proofs)!=1 or
            proofs[0].event!=(use.lease.epoch,True) or use.terminal_succeeded is not True or
            tuple(o.offset for o in owners)!=counts or writer.poisoned or
            writer.pending_epochs or writer.ledger.pending_count):
        raise RuntimeError('native exact dispatch/row/terminal proof differs')
    result.update(physical_counters={'grouped_n20_write_count':counters[0],
                  'grouped_n20_row_count':counters[1],'prefill_long_n20_dispatch_count':counters[2]},
                  read_terminal_success=True,read_epoch=use.lease.epoch,owner_offsets=list(counts))


def retire(mx,stream,arena,writer,backend,owners,use,pool,roots,result):
    """Retire healthy proven roots; never replace the primary pair exception."""
    began=time.perf_counter()
    try:
        mx.synchronize(stream)
        if use is not None and use.state=='prepared':use.abort_before_submit()
        if use is not None and use.state=='submitted':backend.drain_staged(owners,(use,))
        deadline=time.monotonic()+2
        while writer.pending_epochs and not writer.poisoned:
            if time.monotonic()>deadline:raise TimeoutError('cleanup write terminal deadline')
            writer.poll_completions(wait_timeout_s=.01)
        for owner in owners:owner.poll_completions()
        if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count or backend._orphaned_reads:
            raise RuntimeError('cleanup lacks complete terminal proof')
        for owner in owners:owner.close()
        arena.close_after_terminal()
        if not arena._closed:raise RuntimeError('cleanup arena handle remains open')
        if pool.free_count!=pool.capacity:raise RuntimeError('cleanup native pages retained')
        result.update(all_pages_retired=True,free_pages=pool.free_count,pool_capacity=pool.capacity,
                      pending_epochs=0,pending_leases=0,orphaned_reads=0,arena_closed=True,failure_roots_retained=False)
    except BaseException as exc:
        FAILURE_ROOTS.append(tuple(roots));result.update(all_pages_retired=False,
            failure_roots_retained=True,cleanup_error=f'{type(exc).__name__}: {exc}')
        raise
    finally:result['full_owner_arena_retirement_seconds']=time.perf_counter()-began


def raw_oracle(mx,native_output,stock_outputs,counts):
    exact=True;finite=True;max_error=0.;equal_elements=0.;elements=0;start=0
    for count,reference in zip(counts,stock_outputs):
        for offset in range(0,count,1024):
            end=min(count,offset+1024);a=native_output[start+offset:start+end];b=reference[offset:end]
            bits=(a.view(mx.uint16)==b.view(mx.uint16))
            checks=mx.stack((mx.all(bits).astype(mx.float32),
                (mx.all(mx.isfinite(a))&mx.all(mx.isfinite(b))).astype(mx.float32),
                mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))),
                mx.sum(bits.astype(mx.float32))))
            mx.eval(checks);values=checks.tolist();size=(end-offset)*24*256
            exact=exact and values[0]==1;finite=finite and values[1]==1
            max_error=max(max_error,values[2]);equal_elements+=values[3];elements+=size
            del a,b,bits,checks
        start+=count
    return {'raw_bf16_bit_equal':exact,'finite':finite,'max_abs_error':max_error,
            'raw_exact_fraction':equal_elements/elements,'checked_elements':elements}


def run_pair(mx,native,inputs,counts,plan,order,index,start,result):
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner,TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend,PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
    query,keys,values,packed_k,packed_v=inputs
    stream=mx.default_stream(mx.gpu);pool=PagedKVPool(plan['capacity_pages'])
    profile=TokenKVProfile(4,256,'bfloat16')
    arena=NativeWriteBackend(pool.capacity*profile.page_bytes,stream,permit_candidate=True,storage_dtype='bfloat16')
    writer=PagedKVWriteOwner(pool,arena,page_bytes=profile.page_bytes,permit_candidate=True)
    backend=NativeQwen3PagedBackend(writer,timeout_s=5,permit_candidate=True,profile_host=True)
    owners=tuple(PagedKVTokenOwner(writer,profile,permit_candidate=True) for _ in counts)
    roots=[inputs,arena,writer,backend,owners];use=None;native_output=None;stock_outputs=None
    result.update(index=index,warmup=index==0,order=list(order),status='running',timings={})
    original=None
    try:
        with mx.stream(stream):
            t=time.perf_counter();tickets=backend.append_packed_multirow_n20(owners,packed_k,packed_v,counts,permit_candidate=True)
            roots.append(tickets);mx.eval(*(t.dependency for t in tickets))
            result['setup_completed_kv_write_seconds']=time.perf_counter()-t
            if any(o.offset!=0 for o in owners):raise RuntimeError('cold owner was prematurely published')
            bounds(mx,start)
            for arm in order:
                mx.clear_cache();mx.reset_peak_memory();result.setdefault('memory',{})[arm+'_before']=bounds(mx,start)
                if arm=='native':
                    t=time.perf_counter();use=backend.prepare_read_n20(owners,counts,query_heads=24,permit_candidate=True,profile_host=True)
                    result['timings']['native_host_plan_seconds']=time.perf_counter()-t;roots.append(use)
                    t=time.perf_counter();native_output=backend.read_staged(use,query,tickets,scale=256**-.5)
                    result['timings']['native_bind_seconds']=time.perf_counter()-t;roots.append(native_output)
                    t=time.perf_counter();mx.eval(native_output)
                    result['timings']['native_eval_seconds']=time.perf_counter()-t
                    t=time.perf_counter();proofs=backend.drain_staged(owners,(use,))
                    result['timings']['native_terminal_drain_seconds']=time.perf_counter()-t
                    counters=(native.grouped_n20_write_count(arena._arena),native.grouped_n20_row_count(arena._arena),native.prefill_long_n20_dispatch_count(arena._arena))
                    terminal_proof(result,use,proofs,writer,owners,counts,counters)
                else:
                    t=time.perf_counter();stock=[];row=0
                    for n,k,v in zip(counts,keys,values):
                        output=mx.fast.scaled_dot_product_attention(query[row:row+n].transpose(1,0,2)[None],
                            k.transpose(1,0,2)[None],v.transpose(1,0,2)[None],scale=256**-.5,mask='causal',force_fused=True)
                        stock.append(output[0].transpose(1,0,2));row+=n
                    stock_outputs=tuple(stock);roots.append(stock_outputs)
                    result['timings']['stock_graph_bind_seconds']=time.perf_counter()-t
                    t=time.perf_counter();mx.eval(*stock_outputs)
                    result['timings']['stock_eval_seconds']=time.perf_counter()-t
                result['memory'][arm+'_after']=bounds(mx,start)
            if native_output.dtype!=mx.bfloat16 or any(x.dtype!=mx.bfloat16 for x in stock_outputs):
                raise RuntimeError('output dtype differs from same-BF16 oracle')
            t=time.perf_counter();result['numeric']=raw_oracle(mx,native_output,stock_outputs,counts)
            result['oracle_seconds']=time.perf_counter()-t
            if not result['numeric']['raw_bf16_bit_equal'] or not result['numeric']['finite']:
                raise RuntimeError('raw BF16 per-span forced-fused stock differs')
            result['native_host_profile_ns']=dict(backend.host_profile_ns)
            result['status']='passed'
    except BaseException as exc:
        original=exc;result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        try:retire(mx,stream,arena,writer,backend,owners,use,pool,roots,result)
        except BaseException as cleanup:
            if original is None:original=cleanup;result.update(status='failed',error=f'{type(cleanup).__name__}: {cleanup}')
        result['memory_after_retirement']=memory(mx)
    if original is not None:raise original
    return result


def summarize(pairs):
    measured=[p for p in pairs if not p['warmup']]
    if len(measured)!=3 or any(p['status']!='passed' or not p['all_pages_retired'] for p in pairs):
        raise RuntimeError('complete four-pair parity/retirement evidence required')
    cells=[]
    for p in measured:
        t=p['timings'];native=sum(t[k] for k in ('native_host_plan_seconds','native_bind_seconds','native_eval_seconds','native_terminal_drain_seconds'))
        stock=t['stock_graph_bind_seconds']+t['stock_eval_seconds']
        cells.append({'index':p['index'],'native_attention_host_complete_seconds':native,'stock_attention_host_complete_seconds':stock,
            'stock_over_native_ratio':stock/native,'native_eval_seconds':t['native_eval_seconds'],'stock_eval_seconds':t['stock_eval_seconds']})
    return {'paired_cells':cells,'median_paired_stock_over_native_ratio':statistics.median(c['stock_over_native_ratio'] for c in cells),
            'mean_native_attention_host_complete_seconds':statistics.mean(c['native_attention_host_complete_seconds'] for c in cells),
            'mean_stock_attention_host_complete_seconds':statistics.mean(c['stock_attention_host_complete_seconds'] for c in cells),
            'timing_exclusions':['source generation','arena allocation/KV write setup','raw numerical checks','page/arena close after both arms'],
            'scope':'controlled attention primitive only; no model/token/s or serving performance claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--native-path',type=Path,default=NATIVE);p.add_argument('--native-sha256',default=NATIVE_SHA)
    p.add_argument('--mlx-wheel',type=Path,default=WHEEL);p.add_argument('--mlx-wheel-sha256',default=WHEEL_SHA)
    p.add_argument('--inputs-path',type=Path,default=INPUTS);p.add_argument('--inputs-sha256',default=INPUTS_SHA)
    p.add_argument('--domain',default='software_architecture');p.add_argument('--execute',action='store_true')
    a=p.parse_args();result={'schema':'mlx2.n20-attention-controlled-perf.v1','status':'failed','gpu_executed':False,'pairs':[]}
    began=time.monotonic();stop=threading.Event()
    def checkpoint():
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    try:
        result.update(preflight(a.source_commit,a.native_path,a.native_sha256,a.mlx_wheel,a.mlx_wheel_sha256,a.inputs_path,a.inputs_sha256,a.domain))
        if a.execute:
            from varlen_pack_price_bench import _gpuq_owner
            result['gpuq_owner']=_gpuq_owner();os.environ.update(ENV);result['phase']='runtime_binding'
            def interrupt(*_):
                if result.get('rss_guard_triggered_bytes',0)>MAX_BYTES:raise MemoryError('attention12GiB RSS watcher')
                raise TimeoutError('attention hard100s cap')
            signal.signal(signal.SIGALRM,interrupt)
            signal.alarm(max(1,int(MAX_SECONDS-(time.monotonic()-began))))
            sys.path.insert(0,str(a.native_path.parent));sys.path.insert(0,str(ROOT/'src'))
            import _paged_kv_native as native
            if Path(native.__file__).resolve()!=a.native_path.resolve():raise RuntimeError('loaded native path differs')
            cap=native.packed_n20_capability()
            required={'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,'max_spans':20,
                'max_total_rows':163840,'max_page_ids':2560,'prefill_score_scratch_bytes':0,'prefill_read_dispatches':1,'architecture':'s'}
            if any(cap.get(k)!=v for k,v in required.items()):raise RuntimeError('native N20 prefill capability differs')
            result['native_capability']=cap
            import mlx.core as mx
            verify_loaded_wheel(mx,result)
            if not str(mx.device_info().get('architecture','')).endswith('s'):raise RuntimeError('pinned architecture differs')
            mx.set_cache_limit(CACHE_BYTES)
            def monitor():
                while not stop.wait(.05):
                    observed=rss_bytes()
                    if observed>MAX_BYTES:
                        result['rss_guard_triggered_bytes']=observed
                        os.kill(os.getpid(),signal.SIGALRM);return
            threading.Thread(target=monitor,name='oracle-rss-guard',daemon=True).start()
            counts=tuple(result['geometry']['counts']);stream=mx.default_stream(mx.gpu)
            result['gpu_executed']=True;result['status']='running';result['phase']='source_operand_generation';checkpoint()
            with mx.stream(stream):
                mx.random.seed(20261004)
                keys=tuple(mx.random.uniform(-.25,.25,shape=(n,4,256)).astype(mx.bfloat16) for n in counts)
                values=tuple(mx.random.uniform(-.25,.25,shape=(n,4,256)).astype(mx.bfloat16) for n in counts)
                query=mx.random.uniform(-.25,.25,shape=(sum(counts),24,256)).astype(mx.bfloat16)
                packed_k=mx.concatenate(keys,axis=0);packed_v=mx.concatenate(values,axis=0)
                mx.eval(query,*keys,*values,packed_k,packed_v)
                frozen=(query,keys,values,packed_k,packed_v);result['input_memory']=bounds(mx,began)
                for i,order in enumerate(result['schedule']):
                    pair={};result['pairs'].append(pair);result['phase']='warmup_pair' if i==0 else 'measured_pair';result['pair_index']=i;checkpoint()
                    run_pair(mx,native,frozen,counts,result['geometry'],order,i,began,pair)
                    mx.clear_cache();bounds(mx,began);checkpoint()
            result['summary']=summarize(result['pairs']);result['status']='passed';result['phase']='source_retirement'
            result['final_memory']=bounds(mx,began)
            del frozen,query,keys,values,packed_k,packed_v
            mx.synchronize(stream);mx.clear_cache();result['memory_after_source_retirement']=memory(mx)
            result['elapsed_seconds']=time.monotonic()-began;result['phase']='complete'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        stop.set();signal.alarm(0);a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
