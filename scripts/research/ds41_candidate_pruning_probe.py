# SPDX-License-Identifier: MIT
"""Checkpoint-seeded synthetic candidate-pruning component; caller owns GPU.

No MLX/model/GPU work on import or in plan(). run(adapter,unit,guard) uses an
already-loaded idle Source adapter, captures bounded operands from one short
ordinary prefill, and runs real Attention.select with a synthetic extended pool.
It never publishes model/APCv2 state or claims full long-context qualification.
Provenance: provenance/deepseek-v41-candidate-pruning-probe-20261027.json.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MAX_COMPONENT_BYTES=192<<20
SEED_ROWS=32


def plan(config,*,pool_rows=None):
    capacity=config.candidate_topk_blocks*config.candidate_block_size
    count=capacity+1 if pool_rows is None else pool_rows
    if (capacity!=16384 or config.index_topk!=512 or config.index_n_heads!=32
            or config.index_head_dim!=128 or config.num_attention_heads!=64 or config.head_dim!=512):
        raise ValueError('probe requires actual Source32x128 index/64x512 attention geometry and16384candidate capacity')
    if type(count)is not int or not capacity<count<=capacity+8:
        raise ValueError('synthetic pool must be one bounded block beyond actual candidate capacity')
    layer=next((i for i in config.index_source_layer_ids if i>config.candidate_source_layer_id),None)
    if layer is None or config.compress_ratios[layer]!=1:
        raise ValueError('probe requires a post-candidate ratio1 index-source layer')
    # Host/device/temporary upper envelope: 3KV +4index planes, both score arms,
    # selected gathers, captured short-frame operands and bounded weight scratch.
    bound=3*count*512*4+4*count*128*4+4*32*count*4+(16<<20)
    if bound>MAX_COMPONENT_BYTES:raise ValueError('candidate component exceeds independent192MiB envelope')
    return {'schema':'mlx2.ds41-candidate-pruning-plan.v1','layer':layer,'pool_rows':count,
        'candidate_capacity':capacity,'selected_topk':config.index_topk,'query_rows':1,'seed_rows_maximum':SEED_ROWS,
        'component_allocation_envelope_bytes':bound,'component_budget_bytes':MAX_COMPONENT_BYTES,
        'cases':['published_candidate_blocks','causal_tail','empty_candidates'],
        'extension':'repeat bounded actual captured index/KV rows with deterministic row scaling; synthetic pool, not a real long-context cache',
        'required_policy_action':'candidate_only only changes post-layer20 index scoring; no prefill-step change is needed for this isolated selector. Actual<=16K context/source128 chunks cannot establish full long-context pruning benefit. Component evidence alone does not select a serving default.',
        'APCv2':'not_applicable: private synthetic source/shared selection state; no checkpoint publication',
        'qualification':False,'performance_claim':False}


def exact_oracle(reference,candidate):
    """Strict host comparison of already-computed small selected/owner outputs."""
    import numpy as np
    gates={};errors={}
    for name in ('indices','valid','selected_scores','owner_output'):
        a,b=reference[name],candidate[name]
        if a.shape!=b.shape or a.dtype!=b.dtype:gates[name+'_exact']=False;continue
        gates[name+'_exact']=a.tobytes()==b.tobytes()
        if a.dtype.kind=='f':
            finite=np.isfinite(a)&np.isfinite(b)
            errors[name+'_max_abs_on_finite']=float(np.max(np.abs(a[finite]-b[finite])))if np.any(finite)else 0.0
    gates['selected_scores_finite_where_valid']=bool(np.isfinite(reference['selected_scores'][reference['valid']]).all()and np.isfinite(candidate['selected_scores'][candidate['valid']]).all())
    gates['owner_output_finite']=bool(np.isfinite(reference['owner_output']).all()and np.isfinite(candidate['owner_output']).all())
    return {'passed':all(gates.values()),'gates':gates,'errors':errors,
            'raw_score_reduction_equivalence':'exact_bits_required; no tolerance/near-tie promotion',
            'state_scope':'private selection dictionaries only; no authoritative model/KV cache is mutated'}


@contextmanager
def _temporary_attribute(owner,name,value):
    # Restoring a resolved bound class method into __dict__ changes later copy
    # semantics: a shallow copy would keep calling the original instance.
    absent=object();previous=vars(owner).get(name,absent)
    setattr(owner,name,value)
    try:yield
    finally:
        if previous is absent:delattr(owner,name)
        else:setattr(owner,name,previous)


def _dispatch_select(selector,*args,**kwargs):
    # Dispatch the owning class implementation on THIS isolated policy proxy,
    # never an instance attribute holding another attention's bound method.
    return type(selector).select(selector,*args,**kwargs)


class _ModelPolicy:
    def __init__(self,model,policy):
        self.original=model;self.candidate_score_policy=policy;self.scheduled_kv_calls={}
    def __getattr__(self,name):return getattr(self.original,name)


def _capture(adapter,mx,guard,layer):
    import mlx2.runtime.models.deepseek_v41 as owning
    model=adapter.model;attention=model.layers[layer].attn
    saved_select=attention.select;saved_attention=owning.selected_attention
    captured={};pending=False;cache=None
    def own(value):return mx.asarray(value,copy=True)
    def select(x,qr,source,start,shared):
        nonlocal pending
        result=saved_select(x,qr,source,start,shared)
        if not captured:
            keys=getattr(source,'index_keys',None)
            if keys is None or not keys.shape[1]:raise ValueError('actual capture requires nonempty ordinary index plane')
            captured.update(x=own(x[:,-1:]),qr=own(qr[:,-1:]),index_seed=own(keys[:,:SEED_ROWS]),
                            actual_start=int(start),actual_input_rows=int(x.shape[1]),actual_pool_rows=int(keys.shape[1]))
            pending=True
        return result
    def math(q,local,mask,pooled,indices,valid,sinks,*args,**kwargs):
        nonlocal pending
        result=saved_attention(q,local,mask,pooled,indices,valid,sinks,*args,**kwargs)
        if pending:
            if pooled is None or not pooled.shape[1]:raise ValueError('actual capture requires nonempty ordinary KV plane')
            captured.update(q=own(q[:,:,-1:]),local=own(local),mask=own(mask[:,:,-1:]),
                            kv_seed=own(pooled[:,:SEED_ROWS]),sinks=own(sinks))
            pending=False
        return result
    with _temporary_attribute(attention,'select',select), _temporary_attribute(owning,'selected_attention',math):
        try:
            guard()
            request={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
            tokens=adapter.prompt_tokens(request);cache=model.make_cache()
            logits=model.prefill_forward(mx.array([tokens],mx.uint32),cache)
            mx.eval(logits,[row.state for row in cache],*[v for v in captured.values()if isinstance(v,mx.array)])
            if not all(key in captured for key in ('x','qr','index_seed','q','local','mask','kv_seed','sinks')):
                raise ValueError('actual short capture did not reach the ordinary selected-attention branch')
            if captured['local'].shape[2]>256:raise ValueError('captured local band exceeds bounded component contract')
            captured['prompt_token_ids']=list(tokens)
            captured['capture_first_selected_token']=int(mx.argmax(logits[0,-1]).item())
            return attention,captured
        finally:
            if cache is not None:
                for row in cache:
                    close=getattr(row,'close',None)
                    if callable(close):close()
            cache=None


def run(adapter,unit,guard,*,output=None,pool_rows=None):
    """Run only inside caller-owned one-artifact window; never load a model.

    unit/spec.json and the usual frozen-source environment/desktop manifest
    must bind checkpoint, source_revision and source_snapshot_sha256. The
    callback checks the live parent ownership; paired receipts are rechecked.
    Caller owns stopping/releasing its process window in finally. No APC state
    may be published from the private synthetic pool created here.
    """
    from mlx2.desktop_campaign_worker import atomic
    from mlx2.desktop_runner import digest,source_manifest
    from probe_deepseek_v4_packed import verify_ownership,source_receipt
    unit=Path(unit);output=Path(output)if output is not None else unit/'candidate-pruning-component'
    if output.exists()and any(output.iterdir()):raise ValueError('refusing nonempty candidate-pruning evidence directory')
    output.mkdir(parents=True,exist_ok=True)
    def checked():
        guard();verify_ownership(unit,[Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')])
    checked();spec=json.loads((unit/'spec.json').read_text())
    if adapter.identity['fingerprint']!=spec['checkpoint'].rsplit('@',1)[1]:raise ValueError('loaded adapter checkpoint differs from owned exact artifact')
    source=source_receipt()
    if source['commit']!=spec['source_revision']or source['source_snapshot_sha256']!=spec['source_snapshot_sha256']:
        raise ValueError('component source differs from frozen owned source')
    if digest(source_manifest(ROOT))!=spec['source_snapshot_sha256']:raise ValueError('frozen component bytes changed')
    model=adapter.model
    if getattr(model,'_weight_executions',0)or getattr(adapter.weight_stream,'_active',False):raise ValueError('component requires an idle caller-owned adapter')
    p=plan(model.args,pool_rows=pool_rows)
    report={'schema':'mlx2.ds41-candidate-pruning-component.v1','checkpoint':spec['checkpoint'],
        'source_revision':spec['source_revision'],'source_snapshot_sha256':spec['source_snapshot_sha256'],'plan':p,
        'evidence_kind':'actual-checkpoint short-frame operands plus synthetic extended index/KV pools',
        'qualification':False,'long_context_model_qualification':False,'performance_claim':False,
        'APCv2':'not_applicable; synthetic source cache never published','cases':[],'passed':False}
    atomic(output/'results.json',report)
    import mlx.core as mx
    import numpy as np
    from types import SimpleNamespace
    import mlx2.runtime.models.deepseek_v41 as owning
    try:
        checked();attention,seed=_capture(adapter,mx,checked,p['layer'])
        def sha(array):
            value=np.asarray(array);return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
        report['capture']={key:(dict(shape=list(value.shape),dtype=str(value.dtype),sha256=sha(value))if isinstance(value,mx.array)else value)for key,value in seed.items()}
        count=p['pool_rows'];capacity=p['candidate_capacity']
        row=mx.arange(count,dtype=mx.float32);scale=(.75+.5*row/count)[None,:,None]
        index_seed=seed['index_seed'];kv_seed=seed['kv_seed']
        indices=mx.arange(count,dtype=mx.uint32)
        keys=index_seed[:,indices%index_seed.shape[1]]*scale
        pooled=kv_seed[:,indices%kv_seed.shape[1]]*scale
        source_cache=SimpleNamespace(index_keys=keys,pooled=pooled,_pool_count=count)
        source_owners=(id(source_cache.index_keys),id(source_cache.pooled),source_cache._pool_count)
        mx.eval(keys,pooled)
        for name in p['cases']:
            checked();start=count-2 if name=='causal_tail'else count-1
            lens=mx.array([[[start+1]]],mx.int32)
            prior=owning.candidate_blocks(row[None,None],lens,model.args.candidate_topk_blocks,model.args.candidate_block_size)
            if name=='empty_candidates':prior=mx.zeros_like(prior)
            mx.eval(prior)
            row_report={'name':name,'synthetic_start':start,'synthetic_pool_rows':count,'arms':{}}
            values={}
            for policy in ('reference','candidate_only'):
                checked();proxy=_ModelPolicy(model,policy);selector=copy.copy(attention);selector.model=proxy
                shared={'candidates':prior};score_input=[];saved_topk=owning.stable_pool_topk
                def topk(score,k):
                    result=saved_topk(score,k);score_input.append(score);return result
                owning.stable_pool_topk=topk
                try:_dispatch_select(selector,seed['x'],seed['qr'],source_cache,start,shared)
                finally:owning.stable_pool_topk=saved_topk
                if len(score_input)!=1:raise ValueError('bounded query must issue exactly one actual stable-pool topK')
                selected_scores=mx.take_along_axis(score_input[0],shared['indices'],-1)
                raw=owning.selected_attention(seed['q'],seed['local'],seed['mask'],pooled,shared['indices'],shared['selected_mask'],seed['sinks'])
                owner=model.round(raw)
                mx.eval(shared['indices'],shared['selected_mask'],selected_scores,owner)
                values[policy]={'indices':np.asarray(shared['indices']),'valid':np.asarray(shared['selected_mask']),
                    'selected_scores':np.asarray(selected_scores),'owner_output':np.asarray(owner)}
                row_report['arms'][policy]={'selected_policy':selector.model.candidate_score_policy,
                    'selector_dispatch':'owning class select on isolated policy proxy; instance-bound-method bypass',
                    'scheduled_kv_calls':dict(proxy.scheduled_kv_calls),
                    'output_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest()for key,value in values[policy].items()},
                    'valid_selected_rows':int(values[policy]['valid'].sum())}
            row_report['oracle']=exact_oracle(values['reference'],values['candidate_only'])
            counters=row_report['arms']['candidate_only']['scheduled_kv_calls']
            row_report['gates']={'candidate_unscored_positive':counters.get('candidate_only_unscored_rows',0)>0,
                'candidate_scored_capacity':counters.get('candidate_only_score_rows')==capacity,
                'source_owners_unchanged':source_owners==(id(source_cache.index_keys),id(source_cache.pooled),source_cache._pool_count)}
            row_report['passed']=row_report['oracle']['passed']and all(row_report['gates'].values())
            report['cases'].append(row_report);atomic(output/'results.json',report)
        checked();report['passed']=all(row['passed']for row in report['cases'])and len(report['cases'])==len(p['cases'])
        report['source_unchanged']=digest(source_manifest(ROOT))==spec['source_snapshot_sha256']
        report['passed']=report['passed']and report['source_unchanged']
        return report
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        atomic(output/'results.json',report)
