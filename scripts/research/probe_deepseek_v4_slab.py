#!/usr/bin/env python3
"""Owned causal-slab numerical/rollback fixtures; not AST qualification."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import (
    atomic,
    compare,
    source_receipt,
    synthetic,
    verify_ownership,
)


def fixture(gpu,packed):
    import mlx.core as mx
    before=mx.default_device()
    path=ROOT/'tests/test_deepseek_v4_forward_cpu.py'
    spec=importlib.util.spec_from_file_location('slab_cpu_fixture',path)
    module=importlib.util.module_from_spec(spec)
    try:spec.loader.exec_module(module)
    finally:mx.set_default_device(before)
    c=dataclasses.replace(module.config((4,128)),head_dim=512,qk_rope_head_dim=64,
                          index_head_dim=128,simulate_quantization=True)
    if packed:
        c=dataclasses.replace(c,hidden_size=256,q_lora_rank=256,o_lora_rank=256,
                              moe_intermediate_size=256)
    class Weights(module.SmallWeights):
        def __init__(self,c):
            super().__init__(c);self.packed={}
            if packed:
                for name,value in self.data.items():
                    if value.ndim>=2 and 'tid2eid' not in name:
                        self.data[name]=value.astype(mx.float16)
        def matrix(self,name,shape,kind):
            from mlx2.runtime.gguf.packed_metal import PackedMatrix
            from mlx2.runtime.gguf.reader import GGUFTensor
            if name not in self.packed:
                seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
                raw=synthetic(shape,kind,seed)
                self.packed[name]=PackedMatrix(mx.array(raw.reshape(-1)),GGUFTensor(name,shape[::-1],kind,0,raw.nbytes))
            return self.packed[name]
        def linear(self,name):
            native=any(x in name for x in ('hc_','compressor','indexer.','ffn_gate_inp'))
            return super().linear(name) if not packed or native else self.matrix(name,self.data[name].shape,8)
        def grouped(self,x,name,groups):
            if not packed:return super().grouped(x,name,groups)
            from mlx2.runtime.gguf.packed_metal import grouped_linear
            shape=(groups,self.data[name].shape[0]//groups,x.shape[-1])
            return grouped_linear(x,self.matrix(name,shape,8))
        def moe(self,x,ids,scores,prefix):
            if not packed:return super().moe(x,ids,scores,prefix)
            from mlx2.runtime.gguf.packed_metal import packed_moe
            return packed_moe(x,ids,scores,*[self.matrix(prefix+'.ffn_'+kind+'_exps.weight',self.data[prefix+'.ffn_'+kind+'_exps.weight'].shape,fmt) for kind,fmt in [('gate',16),('up',16),('down',10)]])
    return c,Weights(c),hashlib.sha256(path.read_bytes()).hexdigest()


def host(value):
    if isinstance(value,(tuple,list)):return tuple(host(x) for x in value)
    return None if value is None else np.array(value)


def equal(a,b):
    if isinstance(a,(tuple,list)):
        return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a is b if a is None or b is None else np.array_equal(a,b)


def divergence(a,b):
    a,b=np.asarray(a),np.asarray(b);aa=np.argmax(a,axis=-1);bb=np.argmax(b,axis=-1)
    differing=np.flatnonzero(aa.ravel()!=bb.ravel())
    result={'logits_bitwise_equal':bool(np.array_equal(a,b)),
            'selected_tokens_equal':bool(not differing.size),'numerical':compare(a,b,atol=2e-5,rtol=2e-5),
            'equivalence_verdict':'not_assessed: synthetic fixture has no functional continuation oracle'}
    if differing.size:
        row=int(differing[0]);ar=a.reshape(-1,a.shape[-1])[row];br=b.reshape(-1,b.shape[-1])[row]
        selected={int(aa.ravel()[row]),int(bb.ravel()[row])};window=sorted(selected|set(map(int,np.argsort(-ar)[:4]))|set(map(int,np.argsort(-br)[:4])))
        result['first_selected_token_divergence']={'row':row,'shared_selected_prefix':aa.ravel()[:row].tolist(),
            'slab_selected':int(aa.ravel()[row]),'ordinary_selected':int(bb.ravel()[row]),
            'bounded_window':[{'id':i,'slab_logit':float(ar[i]),'ordinary_logit':float(br[i])} for i in window],
            'slab_cross_margin':float(ar.max()-ar[int(bb.ravel()[row])]),
            'ordinary_cross_margin':float(br.max()-br[int(aa.ravel()[row])]),
            'scope':'already-selected first differing synthetic position only; no near-tie qualification claim'}
    return result


def run_fixtures(*,gpu,packed,guard=lambda:None,policies=None,on_case=None):
    import mlx.core as mx

    from mlx2.adapters.deepseek_v4_slab import DeepSeekV4CausalSlabVerifyRows
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model
    previous=mx.default_device();mx.set_default_device(mx.gpu if gpu else mx.cpu)
    try:
        c,w,fixture_hash=fixture(gpu,packed)
        model=DeepSeekV4Model(c,w,identity='synthetic-causal-slab-QAT',**(policies or {}))
        base=model.make_cache();prefix=mx.array(np.arange(126)[None]%c.vocab_size,dtype=mx.int32)
        guard();out=model(prefix,base);mx.eval(out,[x.state for x in base])
        snapshots=[x.snapshot() for x in base];mx.eval([s.arrays for s in snapshots])
        def branch():
            cache=model.make_cache()
            for x,s in zip(cache,snapshots):x.restore(s)
            return cache
        cases=[]
        for width in range(1,7):
            guard();inputs=mx.array([list(range(1,width+1))],dtype=mx.int32)
            ordinary=branch();lo=[];hi=[];start=time.perf_counter()
            for column in range(width):
                a,b=model.forward_with_taps(inputs[:,column:column+1],ordinary,(0,1));lo.append(a);hi.append(b)
            expected=mx.concatenate(lo,axis=1);expected_hidden=mx.concatenate(hi,axis=1)
            mx.eval(expected,expected_hidden,[x.state for x in ordinary]);ordinary_seconds=time.perf_counter()-start
            ordinary_logits=np.array(expected);ordinary_hidden=np.array(expected_hidden)
            for accepted in range(width+1):
                guard()
                cache=branch();owner=DeepSeekV4CausalSlabVerifyRows(model,[cache],(0,1))
                tx=owner.begin([width]);start=time.perf_counter();actual,hidden=tx.forward(inputs)
                mx.eval(actual,hidden,[x.state for x in cache]);slab_seconds=time.perf_counter()-start
                metrics=divergence(np.array(actual),ordinary_logits)
                features=compare(np.array(hidden),ordinary_hidden,atol=2e-5,rtol=2e-5)
                tx.commit([accepted]);mx.eval([x.state for x in cache])
                exact_expected=branch()
                for i in range(accepted):
                    a,b=model.forward_with_taps(inputs[:,i:i+1],exact_expected,(0,1));mx.eval(a,b,[x.state for x in exact_expected])
                state_equal=all(equal(host(a.state),host(b.state)) and a.meta_state==b.meta_state for a,b in zip(cache,exact_expected))
                cases.append({'width':width,'physical_batch_width':1,'causal_target_rows':width,
                    'width_semantics':'causal verification rows, never client lanes or batch width',
                    'accepted':accepted,'prefix':126,'slab':metrics,
                    'hc_features_numerical':features,'hc_features_bitwise_equal':bool(np.array_equal(np.array(hidden),ordinary_hidden)),
                    'committed_state_bitwise_vs_ordinary':state_equal,
                    'partial_replay_exact':state_equal if accepted<width else None,
                    'offsets_correct':all(x.offset==126+accepted for x in cache),
                    'owner_closed':tx.closed and owner._active is None,
                    'ordinary_sequence_seconds':ordinary_seconds,'slab_forward_seconds':slab_seconds,
                    'timing_scope':'observational synthetic forwards; first compile and repeated cache warmth not controlled',
                    'route':owner.receipt_settings})
                if on_case is not None:on_case(cases[-1])
                for x in cache+exact_expected:x.close()
            for x in ordinary:x.close()
        guard();left=branch();right=branch()
        la=DeepSeekV4CausalSlabVerifyRows(model,[left],(0,1)).begin([6])
        lb=DeepSeekV4CausalSlabVerifyRows(model,[right],(0,1)).begin([6])
        al,ah=la.forward(mx.array([[1,2,3,4,5,6]],dtype=mx.int32))
        bl,bh=lb.forward(mx.array([[1,2,3,8,9,10]],dtype=mx.int32))
        mx.eval(al,ah,bl,bh)
        causality={'earlier_logits_bitwise_unchanged':bool(np.array_equal(np.array(al[:,:3]),np.array(bl[:,:3]))),
                   'earlier_hc_features_bitwise_unchanged':bool(np.array_equal(np.array(ah[:,:3]),np.array(bh[:,:3]))),
                   'logits_numerical':compare(np.array(al[:,:3]),np.array(bl[:,:3]),atol=0,rtol=0),
                   'scope':'same-width perturbation of future rows4..6 cannot affect rows1..3 across ratio4/128 boundary'}
        la.abort();lb.abort()
        causality['abort_boundary_exact']=all(equal(host(x.state),host(s.arrays)) for row in (left,right) for x,s in zip(row,snapshots))
        for x in base+left+right:x.close()
        return cases,fixture_hash,causality
    finally:mx.set_default_device(previous)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--ownership-window',type=Path)
    p.add_argument('--owner-path',type=Path,action='append');p.add_argument('--packed-weights',action='store_true')
    p.add_argument('--optimized',action='store_true',help='explicit visible-attention/fused-HC synthetic profile')
    p.add_argument('--hc-projection',choices=('reference','fixed_fp32'),default='reference')
    args=p.parse_args(argv)
    if args.output.exists():p.error('refusing existing evidence directory')
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if (args.packed_weights or args.optimized or args.hc_projection != "reference") and not args.run_gpu:p.error('packed/optimized fixtures require explicit owned GPU')
    if args.run_gpu and (args.ownership_window is None or len(owners)!=2):p.error('GPU requires ownership window and paired receipts')
    def guard():return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard();source=source_receipt();args.output.mkdir(parents=True)
    receipt={'schema':'mlx2.deepseek-v4-causal-slab-fixtures.v1','status':'running','source':source,
        'checkpoint':'synthetic-QAT-causal-slab@seed27','ownership':ownership,'run_gpu':args.run_gpu,
        'packed_weights':args.packed_weights,'optimized':args.optimized,'hc_projection_policy':args.hc_projection,'qualification':False,'ast_claim':False,
        'physical_batch_width':1,'maximum_causal_target_rows':6,
        'actual_artifact_loaded':False,'performance_claim':False,'cases':[]}
    atomic(args.output/'results.json',receipt)
    try:
        import mlx.core as mx
        if args.run_gpu:mx.set_memory_limit(512<<20);mx.set_cache_limit(32<<20)
        policies={'attention_policy':'visible_rows_fp32','hc_policy':'fused_fp32'} if args.optimized else {}
        policies['hc_projection_policy']=args.hc_projection
        def record_case(case):
            receipt['cases'].append(case)
            atomic(args.output/'results.json',receipt)
        receipt['cases'],receipt['fixture_source_sha256'],receipt['causality']=run_fixtures(gpu=args.run_gpu,packed=args.packed_weights,guard=guard,policies=policies,on_case=record_case)
        receipt['transaction_checks_passed']=all(c['owner_closed'] and c['offsets_correct'] and c['partial_replay_exact'] is not False and c['slab']['numerical']['finite'] and c['hc_features_numerical']['finite'] for c in receipt['cases']) and all(receipt['causality'][k] for k in ('earlier_logits_bitwise_unchanged','earlier_hc_features_bitwise_unchanged','abort_boundary_exact'))
        receipt['row_invariance_bitwise_established']=all(c['slab']['logits_bitwise_equal'] and c['hc_features_bitwise_equal'] and c['committed_state_bitwise_vs_ordinary'] for c in receipt['cases'])
        receipt['status']='completed';guard()
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:atomic(args.output/'results.json',receipt)
    return 0 if receipt['transaction_checks_passed'] and receipt['row_invariance_bitwise_established'] else 1


if __name__=='__main__':raise SystemExit(main())
