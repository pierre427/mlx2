# SPDX-License-Identifier: MIT
"""Parent-owned actual ordinary H32/K128 index-component capture and gate.

Loading/prefill/ordinary branch walks are setup. No performance qualification,
APCv2 publication or generic AST admission follows from these components.
"""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time

LAYERS=(2,20)
_QUARANTINE=[]


def _write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x')as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def groups(records):
    """CPU geometry only; chronological lengths are never padded together."""
    result=defaultdict(list)
    for index,record in enumerate(records):
        if(record.get('heads')!=32 or record.get('k')!=128 or type(record.get('count'))is not int
                or record['count']<=512 or record.get('layer')not in LAYERS
                or type(record.get('node'))is not int or not 0<=record['node']<16):
            raise ValueError('Actual index component geometry differs')
        result[(record['layer'],record['count'])].append(index)
    if not result:raise ValueError('No actual long-index operands captured')
    if any(len(values)>16 for values in result.values()):raise ValueError('Index component tile class exceeds16rows')
    return tuple((layer,count,tuple(indices))for(layer,count),indices in sorted(result.items()))


@contextmanager
def capture():
    """Observe real ordinary einsum, projection and weighted head reduction."""
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v41 import Attention
    original_select,original_linear=Attention.select,Attention.linear
    original_einsum,original_sum=mx.einsum,mx.sum
    state={'current':None,'node':None};records=[]
    def linear(self,x,name,**kwargs):
        value=original_linear(self,x,name,**kwargs)
        current=state['current']
        if current is not None and name=='indexer.proj':current['raw_weight']=value
        return value
    def einsum(equation,*operands,**kwargs):
        value=original_einsum(equation,*operands,**kwargs)
        current=state['current']
        if current is not None and equation=='bhsd,btd->bsht':
            if'query'in current:raise ValueError('Unexpected repeated ordinary index dot')
            query,keys=operands
            if query.shape!=(1,32,1,128)or keys.shape[0]!=1 or keys.shape[-1]!=128:
                raise ValueError('Actual ordinary index input geometry differs')
            # Preserve the actual owning output first; copied inputs must later
            # reproduce this exact operator or the archive gate refuses.
            current.update(query=mx.asarray(query,copy=True),keys=mx.asarray(keys,copy=True),raw=value)
        return value
    def summation(array,*args,**kwargs):
        value=original_sum(array,*args,**kwargs)
        current=state['current'];axis=kwargs.get('axis',args[0]if args else None)
        if(current is not None and'raw'in current and array.shape==current['raw'].shape and axis==2):
            if'score'in current:raise ValueError('Unexpected repeated ordinary weighted reduction')
            current.update(weighted=array,score=value)
        return value
    def select(self,x,qr,source,start,shared):
        eligible=self.layer in LAYERS and state['node']is not None and source.index_keys.shape[1]>512
        previous=state['current'];current={}if eligible else None;state['current']=current
        try:value=original_select(self,x,qr,source,start,shared)
        finally:state['current']=previous
        if eligible:
            if set(current)!={'query','keys','raw','raw_weight','weighted','score'}:
                raise ValueError('Actual ordinary index capture incomplete')
            current.update(layer=self.layer,node=state['node'],start=start,ratio=self.ratio,
                heads=32,k=128,count=source.index_keys.shape[1],indices=mx.asarray(shared['indices'],copy=True),
                valid=mx.asarray(shared['selected_mask'],copy=True),
                candidates=None if self.layer!=self.c.candidate_source_layer_id else mx.asarray(shared['candidates'],copy=True))
            records.append(current)
        return value
    Attention.select,Attention.linear,mx.einsum,mx.sum=select,linear,einsum,summation
    try:yield state,records
    finally:Attention.select,Attention.linear,mx.einsum,mx.sum=original_select,original_linear,original_einsum,original_sum


def run(adapter,unit,check_live,*,context_tokens=1024):
    if not callable(check_live):raise ValueError('Index component requires owned live callback')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Index component requires parent-owned GPU')
    if context_tokens!=1024:raise ValueError('This actual component capture binds prefix1024')
    import numpy as np
    from mlx2.adapters.deepseek_v41_dspark import load_source_dspark
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    from ds41_resident_prefill_model_gate import _checkpoint
    def live():
        if check_live()is False:raise RuntimeError('Index component ownership lost')
        return True
    unit=Path(unit);path=unit/'ast-index-components.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Index component evidence exists')
    spec=json.loads((unit/'spec.json').read_text());model=adapter.model;weights=adapter.weights
    result=dict(schema='mlx2.ds41-ast-index-components.v1',checkpoint=_checkpoint(adapter),
        source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='prefix',
        completed=False,passed=False,qualified=False,rate_eligible=False,APCv2_publication=False,
        model_capture_performed=True,context_tokens=context_tokens,actual_layers=list(LAYERS))
    prefix=local=None;head=None;records=[];drained=True;saved_persist=weights.expert_stream.persist_atlas
    try:
        live();_write(path,result);weights.expert_stream.persist_atlas=lambda:None
        request={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
        suffix=list(adapter.tokenizer.encode(adapter.render_prompt(request),add_special_tokens=False))
        filler=list(adapter.tokenizer.encode('A neutral memory test uses numbered lines and ordinary words. ',add_special_tokens=False))
        if not filler or len(suffix)>=context_tokens:raise ValueError('Index prefix construction invalid')
        body=(filler*((context_tokens-len(suffix)+len(filler)-1)//len(filler)))[:context_tokens-len(suffix)]
        tokens=body+suffix;prefix=model.make_cache();start=time.perf_counter();drained=False
        logits,features=model.forward_with_taps(mx.array([tokens],mx.uint32),prefix,(37,38,39))
        mx.eval(logits,features,[row.state for row in prefix]);drained=True;live()
        result.update(prefill_seconds=time.perf_counter()-start,prompt_ids=tokens,
            prompt_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),prefix_frontier=prefix[0].offset)
        if prefix[0].offset!=context_tokens:raise ValueError('Actual index prefix frontier differs')
        anchor=int(mx.argmax(logits[0,-1]).item());head=load_source_dspark(model,expert_cache_bytes=512<<20,check_live=live)
        history=head.make_history();head.record_features(features,0,history);paths,confidence=head.propose_paths(anchor,history);head.close();head=None
        result.update(anchor=anchor,paths=[list(v)for v in paths],status='ordinary_index_capture');_write(path,result)
        begin=time.perf_counter()
        with capture()as(state,records):
            for branch,path_tokens in enumerate(paths):
                local,_,_=snapshot_prompt_cache_descriptors(prefix,require_committed=True)
                for depth,token in enumerate((anchor,*path_tokens)):
                    node=0 if depth==0 else 1+5*branch+depth-1
                    state['node']=node if branch==0 or depth else None
                    live();drained=False
                    value,taps=model.forward_with_taps(mx.array([[token]],mx.uint32),local,(37,38,39))
                    mx.eval(value,taps,[row.state for row in local]);drained=True
                for row in local:row.close()
                local=None
            mx.eval([[row[name]for name in('query','keys','raw','raw_weight','weighted','score','indices','valid','candidates')]for row in records])
        live();plans=groups(records)
        if (len(records)!=31 or {r['node']for r in records if r['layer']==2}!=set(range(1,16))
                or {r['node']for r in records if r['layer']==20}!=set(range(16))):
            raise ValueError('Actual index capture does not cover expected31 chronological queries')
        result.update(ordinary_capture_seconds=time.perf_counter()-begin,
            record_count=len(records),classes=[dict(layer=l,count=n,nodes=[records[i]['node']for i in ids])for l,n,ids in plans])
        # Persist actual operands and owning outputs BEFORE any candidate shader.
        arrays={};metadata=[];byte_count=0
        for index,row in enumerate(records):
            metadata.append({k:row[k]for k in('layer','node','start','ratio','heads','k','count')})
            for name in('query','keys','raw','raw_weight','weighted','score','indices','valid','candidates'):
                if row[name]is None:continue
                array=np.asarray(row[name]).copy();byte_count+=array.nbytes
                if byte_count>192<<20:raise ValueError('Actual index fixture exceeds192MiB bound')
                arrays[f'case{index}_{name}']=array
        archive=unit/'ast-index-operands.npz'
        with archive.open('xb')as stream:np.savez(stream,**arrays)
        fixture=dict(schema='mlx2.ds41-ast-index-fixture.v1',checkpoint=result['checkpoint'],
            source_revision=result['source_revision'],source_snapshot_sha256=result['source_snapshot_sha256'],
            archive=str(archive.resolve()),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            array_bytes=byte_count,cases=metadata,actual_ordinary_outputs_complete=True,candidate_attempted=False)
        fixture_path=unit/'ast-index-fixture.json';_write(fixture_path,fixture)
        result.update(status='fixture_saved',fixture=dict(path=str(fixture_path),sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            archive_sha256=fixture['archive_sha256'],array_bytes=byte_count));_write(path,result)
        drained=False;value=_gate(adapter,records,result,path,live);drained=True;return value
    except BaseException as error:
        result.update(status='failed',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.eval([[r[n]for n in('query','keys','raw','raw_weight','weighted','score')]for r in records]);mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((prefix,local,head,records))
        if drained:
            if local is not None:
                for row in local:row.close()
            if prefix is not None:
                for row in prefix:row.close()
            if head is not None:head.close()
        weights.expert_stream.persist_atlas=saved_persist;_write(path,result)


def _gate(adapter,records,result,path,live):
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import stable_pool_topk
    from mlx2.runtime.models.deepseek_v41 import candidate_blocks
    from ds41_ast_attention_matmul_components import _compare
    from ds41_resident_prefill_model_gate import _counters
    model=adapter.model;plans=groups(records)
    result['classes']=[dict(layer=l,count=n,nodes=[records[i]['node']for i in ids])for l,n,ids in plans]
    scale=model.args.index_head_dim**-.5*model.args.index_n_heads**-.5
    fresh=[];weighted=[];indices=[];candidate_masks=[]
    for row in records:
        raw=mx.einsum('bhsd,btd->bsht',row['query'],row['keys'])
        score=mx.sum(mx.maximum(raw,0)*(row['raw_weight']*scale)[:,:, :,None],2)
        fresh.append(raw);weighted.append(score);indices.append(stable_pool_topk(score,512))
        candidate_masks.append(None if row['candidates']is None else candidate_blocks(score,
            mx.array([[[ (row['start']+1)//row['ratio'] ]]],mx.int32),model.args.candidate_topk_blocks,
            model.args.candidate_block_size,finite_all_retained=True))
    mx.eval(fresh,weighted,indices,candidate_masks);live()
    controls=[]
    for row,raw,score,idx,mask in zip(records,fresh,weighted,indices,candidate_masks):
        checks=dict(raw=_compare(raw,row['raw']),weighted=_compare(score,row['score']),indices=_compare(idx,row['indices']))
        if mask is not None:checks['candidates']=_compare(mask,row['candidates'])
        controls.append(dict(layer=row['layer'],node=row['node'],checks=checks,passed=all(v['equal']and v['finite']for v in checks.values())))
    result['owning_control_checks']=controls;_write(path,result)
    if not all(row['passed']for row in controls):raise ValueError('Copied actual index operands do not reproduce owning operation')
    # New kernel import/compile occurs only after source-bound archive and
    # true owning controls are durable. No standalone surrogate admission.
    from mlx2.runtime.gguf import source_ast_batched_index_matmul as kernel
    before=_counters(adapter);families={}
    for implementation in('native_batch_v1','pinned_regular_s64_v1'):
        comparisons=[]
        for layer,count,ids in plans:
            left=mx.concatenate([records[i]['query'].reshape(1,32,128)for i in ids],0)
            right=mx.swapaxes(mx.concatenate([records[i]['keys']for i in ids],0),-1,-2)
            output=kernel.private_matmul(left,right,check_live=live,implementation=implementation)
            reference=mx.concatenate([records[i]['raw'][:,0]for i in ids],0)
            weight=mx.concatenate([records[i]['raw_weight']for i in ids],0)
            scores=mx.sum(mx.maximum(output,0)*(weight*scale).transpose(0,2,1),axis=1)[:,None]
            expected_score=mx.concatenate([records[i]['score']for i in ids],0)
            top=stable_pool_topk(scores,512);expected_top=mx.concatenate([records[i]['indices']for i in ids],0)
            mx.eval(output,scores,top);live()
            checks=dict(raw=_compare(output,reference),weighted=_compare(scores,expected_score),indices=_compare(top,expected_top))
            comparisons.append(dict(layer=layer,count=count,nodes=[records[i]['node']for i in ids],
                plan=kernel.plan(len(ids),count,implementation=implementation),checks=checks,
                passed=all(v['equal']and v['finite']for v in checks.values())))
        families[implementation]=dict(cases=comparisons,passed=all(v['passed']for v in comparisons),
            exact_classes=sum(v['passed']for v in comparisons),class_count=len(comparisons))
        result.update(status='candidate_families',candidate_families=families);_write(path,result)
    delta={k:v-before.get(k,0)for k,v in _counters(adapter).items()if type(v)in(int,float)}
    result.update(status='passed'if any(v['passed']for v in families.values())else'raw_mismatch',completed=True,
        passed=any(v['passed']for v in families.values()),passing_families=[k for k,v in families.items()if v['passed']],
        implementation_identity=kernel.implementation_identity(),component_counter_delta=delta,
        no_component_expert_or_engram_io=all(delta.get(k)==0 for k in('expert_bytes_read','expert_uploads','host_route_materializations','engram_bytes_read')))
    return result


def run_replay(adapter,unit,check_live,fixture_path,fixture_sha):
    """Replay persisted actual index operands without PP/model/DSpark recapture."""
    if not callable(check_live):raise ValueError('Index replay requires owned live callback')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Index replay requires parent-owned GPU')
    import numpy as np
    from ds41_resident_prefill_model_gate import _checkpoint
    def live():
        if check_live()is False:raise RuntimeError('Index replay ownership lost')
        return True
    unit=Path(unit);path=unit/'ast-index-components.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Index replay evidence exists')
    spec=json.loads((unit/'spec.json').read_text());result=dict(schema='mlx2.ds41-ast-index-components.v1',
        checkpoint=_checkpoint(adapter),source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mode='actual_fixture_replay',
        completed=False,passed=False,qualified=False,rate_eligible=False,APCv2_publication=False,
        model_capture_performed=False,status='validating_fixture')
    records=[];drained=True
    try:
        live();_write(path,result);fixture_path=Path(fixture_path).resolve();raw=fixture_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=fixture_sha:raise ValueError('Actual index fixture SHA differs')
        fixture=json.loads(raw);producer=fixture_path.parent;archive=Path(fixture['archive']).resolve()
        if(fixture.get('schema')!='mlx2.ds41-ast-index-fixture.v1'or fixture.get('checkpoint')!=result['checkpoint']
                or fixture.get('actual_ordinary_outputs_complete')is not True or fixture.get('candidate_attempted')is not False
                or len(fixture.get('cases',[]))!=31 or archive.parent!=producer):
            raise ValueError('Actual index fixture identity/completion differs')
        source=json.loads((producer/'source/desktop-source.json').read_text())
        if(source['source_sha256']!=fixture['source_snapshot_sha256']or source['source_revision']!=fixture['source_revision']):
            raise ValueError('Actual index fixture source binding differs')
        if archive.stat().st_size>192<<20 or hashlib.sha256(archive.read_bytes()).hexdigest()!=fixture['archive_sha256']:
            raise ValueError('Actual index archive bound/SHA differs')
        host_bytes=0
        with np.load(archive,allow_pickle=False)as values:
            for index,metadata in enumerate(fixture['cases']):
                row=dict(metadata)
                for name in('query','keys','raw','raw_weight','weighted','score','indices','valid','candidates'):
                    key=f'case{index}_{name}'
                    if name=='candidates'and key not in values:row[name]=None;continue
                    array=values[key];host_bytes+=array.nbytes;row[name]=mx.array(array)
                if(row['query'].shape!=(1,32,1,128)or row['keys'].shape!=(1,row['count'],128)
                        or row['raw'].shape!=(1,1,32,row['count'])or row['raw_weight'].shape!=(1,1,32)
                        or row['score'].shape!=(1,1,row['count'])or row['indices'].shape!=(1,1,512)):
                    raise ValueError('Actual index archive operand geometry differs')
                records.append(row)
        if host_bytes!=fixture['array_bytes']:raise ValueError('Actual index archive byte inventory differs')
        result.update(status='fixture_verified',record_count=len(records),
            fixture=dict(path=str(fixture_path),sha256=fixture_sha,archive_sha256=fixture['archive_sha256'],array_bytes=host_bytes,
                producer_source_snapshot_sha256=fixture['source_snapshot_sha256']))
        _write(path,result);drained=False;value=_gate(adapter,records,result,path,live);drained=True;return value
    except BaseException as error:
        result.update(status='failed',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append(records)
        _write(path,result)
