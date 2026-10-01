"""CPU inference composition of an isolated materialization candidate; no training."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('fresh receipt required')
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.experimental.hysparse2 import model as model_module
    from mlx2.experimental.hysparse2.config import Config
    from mlx2.experimental.hysparse2.batching import ResearchBatcher
    from mlx2.experimental.hysparse2.apc import EndpointAPC
    from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory
    from mlx2.runtime.apc_v2 import APCv2
    from mlx2.runtime.semantic_capsules import CapsuleStore
    from mlx2.runtime.semantic_memory import SEMANTIC_SCHEMA
    spec = importlib.util.spec_from_file_location('isolated_materialization', args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    stats = candidate._MATERIALIZATION_STATS
    config = replace(Config.smoke(), self_layers=25, self_full_layer=12,
                     cross_blocks=4, sparse_per_block=5, local_window=128,
                     global_tokens=32, prefill_chunk=128, query_tile=32, key_tile=16,
                     candidate_block_size=16, candidate_blocks=4, diffusion_layers=2,
                     diffusion_conditioning='prefix', diffusion_position_encoding='sinusoidal')
    mx.random.seed(91)
    model = model_module.Model(config)
    model.eval()
    mx.eval(model.parameters())
    prompts = [[(j*17+i+1) % config.vocab_size for j in range(n)]
               for i,n in enumerate((512,513,512))]
    original = model_module.attention
    def error(a,b):
        return float(mx.max(mx.abs(a-b)).item())
    report = {'schema':'mlx2.hysparse2-materialization-composition-cpu.v1',
              'device':'cpu','config':config.as_dict(),'layers':config.layers,
              'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
              'production_selected':False,'gpu_composition_qualified':False,
              'trained':False,'arms':{}}
    outputs=[]
    with tempfile.TemporaryDirectory() as directory:
        store=CapsuleStore(Path(directory)/'capsules')
        bindings=dict(model_binding='random-cpu-fixture',tokenizer_binding='character-fixture',
                      runtime_binding='hysparse2-capsule-v1')
        text='tool arguments require evidence'
        capsule=store.put(kind='semantic_base', data={
            'schema':SEMANTIC_SCHEMA,'concepts':{'a':{'label':'tool'},'b':{'label':text}},
            'edges':[{'subject':'a','relation':'has_property','object':'b','authority':'committed',
                      'evidence_digest':hashlib.sha256(text.encode()).hexdigest()}],'proposals':[]},
            **bindings,provenance={'source':'original synthetic CPU composition fixture'})
        memory=CapsuleMemory.from_store(store,capsule.digest,**bindings,
                encode=lambda s:[ord(x)%config.vocab_size for x in s],vocab_size=config.vocab_size)
        model.attach_semantic_capsules(memory)
        try:
            for name,function in [('reference',original),('candidate',candidate.attention)]:
                model_module.attention=function
                stats['evaluations']=0
                batcher=ResearchBatcher(model,max_lanes=2)
                logits,caches,receipt=batcher.prefill(prompts)
                rows=[[31+i] for i in range(3)]
                decoded,updated,_=batcher.decode(rows,caches)
                mx.eval(logits,decoded)
                engine=APCv2(max_size=4,layout_name='hysparse2-endpoint-v1')
                hit=None
                try:
                    bridge=EndpointAPC(model,engine,checkpoint_revision='random-cpu-fixture',
                                       tokenizer_fingerprint='character-fixture')
                    bridge.publish(prompts[0],caches[0])
                    restored,hit=bridge.restore(prompts[0]+rows[0])
                    assert hit.hit and restored.length==512
                    before=restored.length
                    proposal,proposal_receipt=model.diffusion_propose(restored,count=8,steps=3)
                    mx.eval(proposal)
                    assert restored.length==before
                    got=model.decode(mx.array([rows[0]]),restored)
                    expected=model.decode(mx.array([rows[0]]),caches[0].fork())
                    mx.eval(got,expected)
                    apc_error=error(got,expected);assert apc_error==0
                    count=stats['evaluations'] if name=='candidate' else None
                    if name=='candidate': assert count>0
                    report['arms'][name]={'apc_restore_decode_error':apc_error,
                        'periodic_materializations':count,'batch_receipt':receipt,
                        'proposal_receipt':proposal_receipt,'diffusion_preserved_prefix':True}
                    outputs.append((logits,decoded,got,proposal))
                finally:
                    if hit is not None: hit.cache.close()
                    engine.close()
            a,b=outputs
            errors=[error(x,y) for group_a,group_b in zip(a[:2],b[:2],strict=True)
                    for x,y in zip(group_a,group_b,strict=True)]
            errors.append(error(a[2],b[2]));assert max(errors)==0,errors
            assert a[3].tolist()==b[3].tolist()
            report.update(completed=True,logits_max_error=max(errors),proposal_tokens_equal=True,
                          capsule_binding=model.capsule_binding)
        finally:
            model_module.attention=original
    report['source_hashes'] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(model_module.__file__),
                     Path('src/mlx2/experimental/hysparse2/attention.py'),
                     Path('src/mlx2/experimental/hysparse2/apc.py'),
                     Path('src/mlx2/experimental/hysparse2/batching.py'),
                     Path('src/mlx2/experimental/hysparse2/capsule_memory.py')]
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':True,'max_error':report['logits_max_error'],
                      'evaluations':report['arms']['candidate']['periodic_materializations']}))

if __name__=='__main__':
    main()
