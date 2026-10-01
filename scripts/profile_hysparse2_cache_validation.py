"""CPU metadata-only cache validation cost; shared fake KV, no inference claim."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segments',type=int,nargs='+',default=[256,512,1024,2048])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or any(not 1<=n<=4096 for n in args.segments):
        p.error('fresh output and segment counts1..4096 required')
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.experimental.hysparse2.config import Config
    from mlx2.experimental.hysparse2.model import Model
    from mlx2.experimental.hysparse2.train import file_hash
    c=replace(Config.smoke(),self_layers=25,self_full_layer=12,cross_blocks=4,
              sparse_per_block=5,local_window=128,prefill_chunk=256)
    model=Model(c)
    model.eval()
    k=mx.zeros((1,1,256,c.head_dim));v=mx.zeros_like(k)
    local=k[:,:,:128];local_v=v[:,:,:128]
    mx.eval(k,v,local,local_v)
    report={'schema':'mlx2.hysparse2-cache-validation-profile.v1','device':'cpu',
            'metadata_only':True,'shared_fake_kv':True,'layers':c.layers,'cells':[],
            'optimization_implemented':False,'runtime_speed_qualified':False}
    for count in args.segments:
        cache=model.new_cache()
        blocks=[(k,v,i*256) for i in range(count)]
        cache.boundary=mx.zeros((1,1,c.residual_streams,c.hidden_size))
        cache.ple_history=mx.zeros((1,c.semantic_ngram-1),dtype=mx.int32)
        def state(n):
            cache.length=n*256
            for i,layer in enumerate(model.self_decoder):
                cache.self_kv[i]=([(local,local_v,cache.length-128)] if layer.kind=='swa' else blocks[:n])
            for i,layer in enumerate(model.cross_decoder):
                if layer.kind=='cross':cache.cross_kv[i]=blocks[:n]
        # Exclude state construction from validation time. This proxy models
        # checks over growing internally produced histories, not actual kernels.
        elapsed=0.0
        for n in range(1,count+1):
            state(n)
            start=time.perf_counter();model.validate_cache_state(cache);elapsed+=time.perf_counter()-start
        start=time.perf_counter();model.validate_cache_state(cache);endpoint=time.perf_counter()-start
        report['cells'].append({'segments':count,'logical_tokens':count*256,
            'per_chunk_checks':count,'per_chunk_check_seconds':elapsed,
            'single_endpoint_check_seconds':endpoint,
            'per_chunk_segment_visits':5*count*(count+1)//2+24*count,
            'endpoint_segment_visits':5*count+24})
    report['completed']=True
    report['source_sha256']={str(path):file_hash(path) for path in
                            [Path(__file__),Path('src/mlx2/experimental/hysparse2/model.py')]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
