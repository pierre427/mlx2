"""Untimed same-input node1 stage oracle for the parent-owned Source GPU.

One candidate tile is returned unchanged. Extra B1 component calls are lazy
diagnostics retained until one post-tile evaluation, never candidate work or
performance evidence. No imports compile shaders or load a checkpoint.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import re

LAYERS=tuple(range(9))
NODE=1
_QUARANTINE=[]


def _write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x')as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _comparison(left,right):
    import numpy as np
    left,right=np.asarray(left),np.asarray(right)
    if left.shape!=right.shape or left.dtype!=right.dtype:
        return dict(equal=False,geometry_mismatch=True,left_shape=list(left.shape),right_shape=list(right.shape),
                    left_dtype=str(left.dtype),right_dtype=str(right.dtype))
    exact=left.tobytes()==right.tobytes()
    difference=np.abs(left.astype(np.float64)-right.astype(np.float64))
    different=(left.view(np.uint8).reshape(-1,left.dtype.itemsize)!=
               right.view(np.uint8).reshape(-1,right.dtype.itemsize)).any(axis=1)
    indices=np.flatnonzero(different)
    return dict(equal=exact,shape=list(left.shape),dtype=str(left.dtype),
        finite=bool(np.isfinite(left).all()and np.isfinite(right).all()),
        different_elements=int(indices.size),first_different_element=None if not indices.size else int(indices[0]),
        maximum_absolute_error=0. if not difference.size else float(difference.max()),
        candidate_sha256=hashlib.sha256(left.tobytes()).hexdigest(),
        oracle_sha256=hashlib.sha256(right.tobytes()).hexdigest())


@contextmanager
def instrument(model,check_live):
    import mlx.core as mx
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as attention
    from mlx2.runtime.models.deepseek_v41 import selected_attention
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    if not callable(check_live):raise ValueError('Exact stage probe needs an owned live guard')
    saved={name:getattr(dense,name)for name in('linear_tile','paired_linear_tile','grouped_tile',
        'shared_expert_tile','mixes_tile','_float_batch_matmul')}
    old_core,old_half=attention.attention_core,attention.attention_half_tile
    old_private=attention._private_matmul
    old_pointwise={name:getattr(mx,name)for name in('logsumexp','logaddexp','exp')}
    records=[];active_layer=[None];hc=[None];nested=[False];subcontext=[None]
    def row(value):return mx.contiguous(value[:,NODE:NODE+1])
    def layer(name):
        match=re.match(r'blk\.(\d+)\.',name)
        return None if match is None else int(match[1])
    def retain(kind,number,candidate,oracle,*,name=None,input_value=None,bf16_boundary=False,scope='same_input_node1'):
        records.append(dict(sequence=len(records),kind=kind,layer=number,name=name,
            candidate=candidate,oracle=oracle,input=input_value,bf16_boundary=bf16_boundary,scope=scope))
    def linear(m,x,name,rounded=True):
        value=saved['linear_tile'](m,x,name,rounded=rounded)
        number=layer(name)
        if not nested[0]and number in LAYERS and x.shape[1]==16:
            one=row(x);reference=m.linear(one,name,rounded=rounded)
            retain('linear',number,row(value),reference,name=name,input_value=one,bf16_boundary=rounded)
        return value
    def paired(m,x,a,b,rounded=True):
        value=saved['paired_linear_tile'](m,x,a,b,rounded=rounded)
        number=layer(a)
        if not nested[0]and number in LAYERS and x.shape[1]==16:
            one=row(x)
            reference=(m.weights.paired_project(one,a,b,rounded=rounded)if m.can_pair(one,a,b)else
                       (m.linear(one,a,rounded=rounded),m.linear(one,b,rounded=rounded)))
            for name,candidate,oracle in zip((a,b),value,reference):
                retain('paired_linear',number,row(candidate),oracle,name=name,input_value=one,bf16_boundary=rounded)
        return value
    def grouped(m,x,name,groups=8):
        value=saved['grouped_tile'](m,x,name,groups=groups)
        number=layer(name)
        if not nested[0]and number in LAYERS and x.shape[1]==16:
            one=row(x);reference=m.weights.grouped(one,name,groups=groups)
            retain('grouped_raw',number,row(value),reference,name=name,input_value=one)
            retain('grouped_rounded',number,m.round(row(value)),m.round(reference),name=name,bf16_boundary=True)
        return value
    def shared(block,x):
        value=saved['shared_expert_tile'](block,x)
        if not nested[0]and block.layer in LAYERS and x.shape[1]==16:
            one=row(x);reference=block.shared_expert(one)
            retain('shared_expert',block.layer,row(value),reference,name=block.prefix,input_value=one,bf16_boundary=True)
        return value
    def float_projection(x,weight):
        value=saved['_float_batch_matmul'](x,weight)
        if hc[0]is not None and x.shape[1]==16:
            number,kind=hc[0];one=row(x);reference=one@weight.T
            retain('HC_raw_projection',number,row(value),reference,name=kind,input_value=one)
        return value
    def mixes(block,h,kind):
        prior=hc[0];hc[0]=(block.layer,kind)if block.layer in LAYERS and not nested[0]else None
        try:value=saved['mixes_tile'](block,h,kind)
        finally:hc[0]=prior
        if not nested[0]and block.layer in LAYERS:
            one=row(h);reference=block.mixes(one,kind)
            for suffix,candidate,oracle in zip(('pre','post','comb'),value,reference):
                retain('HC_'+suffix,block.layer,row(candidate),oracle,name=kind,input_value=one)
        return value
    def core(q,segments,sinks,*,owner=None,shared=None):
        # The existing core has a bounded, known operation order per count
        # class. Preserve the production graph and observe its private dot
        # products/reductions without evaluating any individual operator.
        steps=[]
        for local_count,selected_count,nodes in attention.count_classes(segments,owner,shared):
            phases=['QK_local']+(['QK_selected']if selected_count else[])+['PV_local']+(['PV_selected']if selected_count else[])
            steps.extend((phase,nodes)for phase in phases)
        previous=subcontext[0]
        subcontext[0]=dict(steps=steps,cursor=0,current=None)
        try:value=old_core(q,segments,sinks,owner=owner,shared=shared)
        finally:subcontext[0]=previous
        number=active_layer[0]
        if not nested[0]and number in LAYERS:
            query=mx.contiguous(q[:,:,NODE:NODE+1])
            local=mx.contiguous(mx.take(segments.local_table,mx.array(segments.local_rows[NODE],mx.int32),axis=2))
            mask=mx.ones((1,1,1,local.shape[2]),mx.bool_)
            pooled=indices=valid=None
            if owner is not None and owner.logical_rows[NODE]:
                pooled=mx.contiguous(mx.take(owner.pool_table,mx.array(owner.logical_rows[NODE],mx.int32),axis=1))
                indices=shared[NODE]['indices'];valid=shared[NODE]['selected_mask']
            reference=selected_attention(query,local,mask,pooled,indices,valid,sinks)
            retain('attention_core',number,mx.contiguous(value[:,:,NODE:NODE+1]),reference,
                name=f'local{local.shape[2]}_selected{0 if indices is None else indices.shape[-1]}',input_value=query)
        return value
    def private_matmul(left,right,**kwargs):
        value=old_private(left,right,**kwargs)
        context=subcontext[0]
        if context is not None:
            if context['cursor']>=len(context['steps']):raise ValueError('Unexpected exactattention matrix operation')
            phase,nodes=context['steps'][context['cursor']];context['cursor']+=1
            context['current']=None if NODE not in nodes else(nodes,nodes.index(NODE))
            if context['current']is not None:
                _,position=context['current']
                # Restore the actual ordinaryB1 four-dimensional head layout,
                # allowing its native stock dispatch to make its normal choice.
                one_left=mx.contiguous(left[position:position+1])[:,:,None]
                source=right[position:position+1]if right.shape[0]>1 else right
                one_right=(mx.swapaxes(mx.contiguous(mx.swapaxes(source,-1,-2)),-1,-2)
                           if phase.startswith('QK')else mx.contiguous(source))[:,None]
                reference=one_left@one_right
                retain('attention_'+phase,active_layer[0],mx.contiguous(value[position:position+1])[:,:,None],reference,
                    name=f'B{left.shape[0]}_M{left.shape[1]}_K{left.shape[2]}_N{right.shape[-1]}',
                    input_value=one_left)
        return value
    def pointwise(name):
        original=old_pointwise[name]
        def invoke(*args,**kwargs):
            value=original(*args,**kwargs)
            context=subcontext[0]
            current=None if context is None else context['current']
            if current is not None:
                nodes,position=current
                if isinstance(value,mx.array)and value.ndim>=1 and value.shape[0]==len(nodes):
                    inputs=tuple(mx.contiguous(arg[position:position+1])if isinstance(arg,mx.array)
                                 and arg.ndim and arg.shape[0]==len(nodes)else arg for arg in args)
                    reference=original(*inputs,**kwargs)
                    retain('attention_'+name,active_layer[0],mx.contiguous(value[position:position+1]),reference,
                        name=f'B{len(nodes)}_shape'+str(tuple(value.shape[1:])),
                        input_value=inputs[0]if inputs and isinstance(inputs[0],mx.array)else None)
            return value
        return invoke
    def half(block,h,pre,base,segments,shared):
        previous=active_layer[0];active_layer[0]=block.layer
        try:value=old_half(block,h,pre,base,segments,shared)
        finally:active_layer[0]=previous
        if block.layer in LAYERS:
            descriptor=value[-1]
            # Node1's parent is root0. The root's current-layer cache is a
            # genuine attention input. History is not consumed by this half;
            # layer0 tails are assigned by the target after the half returns.
            if descriptor.base.history:
                state=snapshot_prompt_cache_descriptors([descriptor.base])[0][0]
                state.keys=mx.take(descriptor.local_table,mx.array(descriptor.local_rows[0],mx.int32),axis=2)
                state.offset=descriptor.offset(0)
            else:state=descriptor.materialize(0)
            cache=list(base)
            for index,source in enumerate(segments):
                if source is not None and source.base.source:cache[index]=source.materialize(NODE)
            cache[block.layer]=state
            nested[0]=True
            try:reference=block.attention_half(row(h),row(pre),state,cache,dict(shared[NODE]))
            finally:nested[0]=False
            for name,candidate,oracle in zip(('h','ffn_input','ffn_pre','post','comb'),value[:5],reference):
                retain('complete_attention_half_'+name,block.layer,row(candidate),oracle,
                       input_value=row(h)if name=='h'else None,bf16_boundary=name in('h','ffn_input'))
            # This half owns no Engram tail write; compare only its five
            # attention planes and leave the target's history law untouched.
            for name in('keys','buf_kv','buf_gate','_pooled','_index_keys'):
                candidate=(mx.take(descriptor.local_table,mx.array(descriptor.local_rows[NODE],mx.int32),axis=2)
                           if name=='keys'else getattr(descriptor.base if descriptor.base.history else
                                                       descriptor.materialize(NODE),name))
                oracle=getattr(state,name)
                if candidate is not None or oracle is not None:
                    if(candidate is None)!=(oracle is None):raise ValueError('Same-input half plane inventory differs')
                    retain('complete_attention_cache_'+name,block.layer,candidate,oracle,bf16_boundary=True)
        return value
    dense.linear_tile,dense.paired_linear_tile,dense.grouped_tile=linear,paired,grouped
    dense.shared_expert_tile,dense.mixes_tile,dense._float_batch_matmul=shared,mixes,float_projection
    attention.attention_core,attention.attention_half_tile=core,half
    attention._private_matmul=private_matmul
    for name in old_pointwise:setattr(mx,name,pointwise(name))
    try:yield records
    finally:
        for name,value in saved.items():setattr(dense,name,value)
        attention.attention_core,attention.attention_half_tile=old_core,old_half
        attention._private_matmul=old_private
        for name,value in old_pointwise.items():setattr(mx,name,value)


def run(adapter,unit,check_live):
    if not callable(check_live):raise ValueError('Exact stage probe needs an owned live guard')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Exact stage probe requires parent-owned GPU')
    import numpy as np
    from mlx2.adapters.deepseek_v41_dspark import load_source_dspark
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
    from ds41_resident_prefill_model_gate import _checkpoint
    def live():
        if check_live()is False:raise RuntimeError('Exact stage probe ownership lost')
    live();unit=Path(unit);path=unit/'ast-exact-stage-probe.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Stage probe evidence exists')
    model=adapter.model;weights=adapter.weights
    if getattr(weights,'source_resident_bank_session',None)is not None:
        raise ValueError('Exact stage probe requires ordinary streaming ownership, no bank')
    result=dict(schema='mlx2.ds41-ast-exact-stage-probe.v1',checkpoint=_checkpoint(adapter),
        node=NODE,layers=list(LAYERS),status='prefix',completed=False,qualified=False,rate_eligible=False,
        APCv2_publication=False,extra_component_oracles=True,no_per_operator_evaluation=True,
        source_runtime='Frozen v40 candidate; new probe overlay only',helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    spec=json.loads((unit/'spec.json').read_text());result.update(source_revision=spec.get('source_revision'),
        source_snapshot_sha256=spec.get('source_snapshot_sha256'))
    _write(path,result);prefix=None;head=tile=None;records=[];drained=True
    saved_persist=getattr(weights.expert_stream,'persist_atlas',None)
    try:
        weights.expert_stream.persist_atlas=lambda:None
        request={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
        encoded=adapter.tokenizer.encode(adapter.render_prompt(request),add_special_tokens=False)
        ids=list(getattr(encoded,'ids',encoded));prefix=model.make_cache();drained=False
        logits,features=model.forward_with_taps(mx.array([ids],mx.uint32),prefix,(37,38,39))
        mx.eval(logits,features,[row.state for row in prefix]);drained=True;live()
        anchor=int(mx.argmax(logits[0,-1]).item())
        head=load_source_dspark(model,expert_cache_bytes=512<<20,check_live=check_live)
        history=head.make_history();head.record_features(features,0,history)
        paths,confidence=head.propose_paths(anchor,history);head.close();head=None
        result.update(status='same_input_stage_tile',prompt_ids=ids,anchor=anchor,paths=[list(v)for v in paths],
            prefix_frontier=prefix[0].offset);_write(path,result)
        with instrument(model,check_live)as records:
            drained=False
            tile=execute_batched_tree(model,prefix,anchor,paths,expert_law='source_ast_m1_rows_v1',
                                      forward_law='fully_batched_m1_v1')
            # Retain every lazy extra component result until this single
            # post-tile diagnostic evaluation. No operator is timed.
            mx.eval(tile.logits,tile.features,[(r['candidate'],r['oracle'],r['input'])for r in records])
            drained=True
        live();rows=[];raw={};host_bytes=0
        for record in records:
            candidate,oracle=np.asarray(record['candidate']),np.asarray(record['oracle'])
            row={k:v for k,v in record.items()if k not in('candidate','oracle','input')}
            row['comparison']=_comparison(candidate,oracle);rows.append(row)
            if not row['comparison']['equal']and len(raw)<48:
                arrays={'candidate':candidate,'oracle':oracle}
                if record['input']is not None:arrays['input']=np.asarray(record['input'])
                for label,array in arrays.items():
                    if host_bytes+array.nbytes>64<<20:break
                    raw[f"stage{row['sequence']}_{label}"]=array.copy();host_bytes+=array.nbytes
        archive=unit/'ast-exact-stage-raw.npz'
        with archive.open('xb')as stream:np.savez(stream,**raw)
        bad=[row for row in rows if not row['comparison']['equal']]
        owner_bad=[row for row in bad if row['bf16_boundary']]
        result.update(status='completed',completed=True,stage_count=len(rows),all_same_input_raw_exact=not bad,
            first_divergent_operator=None if not bad else bad[0],
            first_BF16_or_attention_state_divergence=None if not owner_bad else owner_bad[0],
            stages=rows,raw_arrays_path=str(archive),raw_arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            raw_host_bytes=host_bytes,tile_receipt=tile.receipt,
            interpretation='Raw diagnostic differences are localization outcomes, not qualification/performance or automatic root-cause proof')
        return result
    except BaseException as error:
        result.update(status='failed',completed=False,error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.eval([(r['candidate'],r['oracle'],r['input'])for r in records]);mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((prefix,head,tile,records))
        if drained:
            if tile is not None:tile.transaction.abort()
            if head is not None:head.close()
            if prefix is not None:
                for row in prefix:row.close()
        if saved_persist is not None:weights.expert_stream.persist_atlas=saved_persist
        _write(path,result)
