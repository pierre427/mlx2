"""Reject owner transitions before ordinary inference publication; no quality claim."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--cpu-smoke',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or (not args.cpu_smoke and args.checkpoint is None):
        p.error('fresh output and checkpoint for GPU required')
    with nullcontext() if args.cpu_smoke else gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model
        from mlx2.experimental.hysparse2.train import _load_model_state
        mx.set_default_device(mx.cpu if args.cpu_smoke else mx.gpu)
        if not args.cpu_smoke:
            mx.set_memory_limit(12<<30)
            mx.set_cache_limit(128<<20)
        mx.random.seed(42)
        c=Config.smoke() if args.cpu_smoke else Config(**json.loads((args.checkpoint/'state.json').read_text())['config'])
        model=Model(c)
        if not args.cpu_smoke: _load_model_state(args.checkpoint,model)
        model.eval()
        tokens=(mx.arange(4 if args.cpu_smoke else 144)[None]*17+1)%c.vocab_size
        report={'schema':'mlx2.hysparse2-inference-owner.v1','device':'cpu' if args.cpu_smoke else 'gpu',
                'serving_route_qualified':False,'cases':[], 'parameters':c.capacity()['parameters']}
        for operation in ('prefill','decode','prefill_cache_only'):
            for change in ('parameters','capsules'):
                _,cache=model.prefill(tokens)
                hook='_append' if operation=='prefill_cache_only' else '_cross'
                original=getattr(model,hook)
                def changed(*a,**kw):
                    value=original(*a,**kw)
                    if change=='parameters': model.update({'embedding':{'weight':model.embedding.weight}})
                    else: model.attach_semantic_capsules(None)
                    return value
                setattr(model,hook,changed)
                try:
                    try:
                        if operation=='decode': model.decode(mx.array([[5]]),cache)
                        else: model.prefill(tokens,return_logits=operation!='prefill_cache_only')
                    except ValueError as exc: assert 'revision' in str(exc)
                    else: raise AssertionError('stale ordinary result accepted')
                finally: setattr(model,hook,original)
                _,fresh=model.prefill(tokens)
                got=model.decode(mx.array([[5]]),fresh)
                _,reference=model.prefill(tokens)
                expected=model.decode(mx.array([[5]]),reference)
                error=float(mx.max(mx.abs(got-expected)).item())
                assert error==0 and bool(mx.all(mx.isfinite(got)).item())
                report['cases'].append({'operation':operation,'change':change,'rejected':True,'fresh_retry_error':error})
        report['completed']=True
        if not args.cpu_smoke: report['peak_gib']=mx.get_peak_memory()/(1<<30)
    report['source_sha256']={str(p):file_hash(p) for p in (Path(__file__),Path('src/mlx2/experimental/hysparse2/model.py'))}
    report['checkpoint_sha256']=None if args.cpu_smoke else file_hash(args.checkpoint/'model.safetensors')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__': main()
