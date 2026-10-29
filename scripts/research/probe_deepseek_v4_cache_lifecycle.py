#!/usr/bin/env python3
"""Guarded packed-cache lifecycle/APCv2 persistence fixture; no model loaded."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic,source_receipt,verify_ownership
from mlx2.runtime.models.deepseek_v4_cache import DeepSeekV4LayerCache
from mlx2.runtime.models.deepseek_v4_packed_cache import DeepSeekV4PackedLayerCache
from mlx2.runtime.models.deepseek_v4_packed_kv import pack_fp8_kv,unpack_fp8_kv


def advance(cache,n):
    xp=cache.xp;offset=cache.offset
    # Dyadic values/uniform gates make this a bit-exact lifecycle oracle on
    # both devices, isolating storage/state from reduction-rounding changes.
    def values(width):
        x=(np.arange(offset*width,(offset+n)*width).reshape(1,n,width)%17-8).astype(np.float32)/8
        return xp.array(x)
    local=values(512)[:,None]
    cache.update_local(local)
    for pool in (cache.compressor,cache.indexer):
        if pool is None:continue
        width=pool.head_dim*(2 if pool.ratio==4 else 1)
        raw,_=pool.compress(values(width),xp.zeros((1,n,width),dtype=xp.float32),
                            xp.zeros((pool.ratio,width),dtype=xp.float32),offset)
        if pool.head_dim==512:raw=unpack_fp8_kv(pack_fp8_kv(raw,xp=xp),xp=xp)
        pool.append_pooled(raw)


def equal(a,b):
    if a is None or b is None:return a is None and b is None
    if isinstance(a,(tuple,list)):
        return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return np.array_equal(np.array(a),np.array(b))


def run_lifecycle(output,*,gpu,guard=lambda:None):
    import mlx.core as mx
    from mlx2.runtime.models.cache import save_prompt_cache,load_prompt_cache,prompt_cache_payload
    from mlx2.runtime.cow_cache import freeze_prompt_cache,restore_prompt_cache
    old=mx.default_device();mx.set_default_device(mx.gpu if gpu else mx.cpu)
    frozen=branch=None;cases=[]
    try:
        for ratio in (0,4,128):
            guard()
            c=DeepSeekV4PackedLayerCache(ratio,xp=mx,identity='synthetic-packed-cache-v2')
            cpu=DeepSeekV4PackedLayerCache(ratio,xp=np,identity=c.identity)
            baseline=DeepSeekV4LayerCache(ratio,xp=np,identity='synthetic-fp32')
            for n in (3,126,9):
                for cache in (c,cpu,baseline):advance(cache,n)
            checkpoint=c.snapshot();mx.eval(c.state)
            checks={'cpu_raw_state_exact':equal(c.state,cpu.state),
                    'baseline_local_exact':equal(c.keys,baseline.keys),
                    'baseline_pooled_exact':ratio==0 or equal(c.compressor.pooled,baseline.compressor.pooled)}
            # Reject five speculative rows, restore the committed boundary,
            # then replay an accepted prefix and compare against fresh CPU state.
            advance(c,5);c.restore(checkpoint)
            checks['rejected_branch_restore_exact']=equal(c.state,checkpoint.arrays)
            advance(c,2);advance(cpu,2)
            checks['accepted_replay_exact']=equal(c.state,cpu.state)
            arrays,metadata=prompt_cache_payload([c])
            checks['uint8_serialized_planes']=any(a.dtype==mx.uint8 for a in arrays.values())
            checks['string_metadata']=all(isinstance(v,str) for v in metadata.values())
            path=output/f'ratio{ratio}.safetensors'
            save_prompt_cache(str(path),[c],{'fixture':'synthetic-packed-cache-v2'})
            restored,meta=load_prompt_cache(str(path),return_metadata=True)
            checks['disk_state_exact']=equal(restored[0].state,c.state) and restored[0].meta_state==c.meta_state
            checks['disk_metadata']=meta=={'fixture':'synthetic-packed-cache-v2'}
            frozen,_=freeze_prompt_cache([c],key='synthetic',tokens=tuple(range(c.offset)),cache_type='deepseek_v4')
            branch=restore_prompt_cache(frozen)
            checks['apc_state_exact']=equal(branch[0].state,c.state)
            before=c.snapshot();advance(branch[0],1)
            checks['apc_branch_isolation']=equal(c.state,before.arrays) and branch[0].offset==c.offset+1
            branch.close();branch=None;frozen.close();frozen=None
            for item in restored:item.close()
            cases.append(dict(ratio=ratio,offset=c.offset,physical_bytes=c.nbytes,
                              state_revision=c.meta_state[0],checks=checks,passed=all(checks.values())))
            c.close();guard()
        # All-empty arrays need class-based MLX restoration, not dtype inference.
        for cls in (DeepSeekV4LayerCache,DeepSeekV4PackedLayerCache):
            empty=cls(0,xp=mx,identity='synthetic-empty')
            path=output/(cls.__name__+'-empty.safetensors')
            save_prompt_cache(str(path),[empty]);loaded=load_prompt_cache(str(path))[0]
            cases.append(dict(empty_class=cls.__name__,passed=loaded.meta_state==empty.meta_state and loaded.empty()))
        return cases
    finally:
        if branch is not None:branch.close()
        if frozen is not None:frozen.close()
        mx.set_default_device(old)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--ownership-window',type=Path)
    p.add_argument('--owner-path',type=Path,action='append');args=p.parse_args(argv)
    if args.output.exists():p.error('refusing existing evidence directory')
    owners=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owners)!=2):p.error('GPU requires ownership-window and paired receipts')
    def guard():return verify_ownership(args.ownership_window,owners) if args.run_gpu else None
    ownership=guard();source=source_receipt()
    source['cache_sources']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/'src/mlx2/runtime/models/deepseek_v4_cache.py',ROOT/'src/mlx2/runtime/models/deepseek_v4_packed_cache.py',Path(__file__)]}
    args.output.mkdir(parents=True)
    result=dict(schema='mlx2.deepseek-v4-cache-lifecycle.v1',status='running',source=source,ownership=ownership,
                checkpoint='synthetic-dyadic-packed-cache-v2@sha256:'+source['cache_sources'][str(Path(__file__).relative_to(ROOT))],
                run_gpu=args.run_gpu,qualification=False,artifact_loaded=False,cases=[])
    atomic(args.output/'results.json',result)
    try:
        if args.run_gpu:
            import mlx.core as mx
            mx.set_memory_limit(512<<20);mx.set_cache_limit(32<<20)
        result['cases']=run_lifecycle(args.output,gpu=args.run_gpu,guard=guard)
        result['status']='completed';result['all_checks_passed']=all(c['passed'] for c in result['cases'])
    except BaseException as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:atomic(args.output/'results.json',result)
    return 0 if result['all_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
