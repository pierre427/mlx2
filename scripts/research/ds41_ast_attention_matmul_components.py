"""Parent-owned actual-input F32 attention-matrix arithmetic gate.

One DSpark tile captures primitive operands; component controls are B1 stock
matrix calls, never serial whole-model execution. No timing/qualification claim.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os

NODE=1
LAYERS=tuple(range(9))
_QUARANTINE=[]


def _write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x')as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


@contextmanager
def capture(model):
    """Keep production G operands, role and grouping; return results unchanged."""
    import mlx.core as mx
    from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as attention
    original_half,original_core,original_matrix=attention.attention_half_tile,attention.attention_core,attention._private_matmul
    layer=[None];context=[None];cases=[]
    def half(block,*args,**kwargs):
        previous=layer[0];layer[0]=block.layer
        try:return original_half(block,*args,**kwargs)
        finally:layer[0]=previous
    def core(q,segments,sinks,*,owner=None,shared=None):
        steps=[]
        for local,selected,nodes in attention.count_classes(segments,owner,shared):
            roles=['QK_local']+(['QK_selected']if selected else[])+['PV_local']+(['PV_selected']if selected else[])
            steps.extend((role,nodes)for role in roles)
        old=context[0];context[0]=dict(steps=steps,cursor=0)
        try:return original_core(q,segments,sinks,owner=owner,shared=shared)
        finally:context[0]=old
    def matrix(left,right,**kwargs):
        value=original_matrix(left,right,**kwargs)
        state=context[0]
        if state is not None:
            if state['cursor']>=len(state['steps']):raise ValueError('Unexpected captured attention operator')
            role,nodes=state['steps'][state['cursor']];state['cursor']+=1
            if layer[0]in LAYERS and NODE in nodes:
                transpose=role.startswith('QK')
                if 'transpose_b'in kwargs and kwargs['transpose_b']!=transpose:
                    raise ValueError('Production attention transpose role differs')
                cases.append(dict(layer=layer[0],role=role,nodes=tuple(nodes),node_position=nodes.index(NODE),
                    transpose_b=transpose,left=mx.contiguous(left),right=mx.contiguous(right),
                    captured_output=value))
        return value
    attention.attention_half_tile,attention.attention_core,attention._private_matmul=half,core,matrix
    try:yield cases
    finally:
        attention.attention_half_tile,attention.attention_core,attention._private_matmul=original_half,original_core,original_matrix


def _compare(left,right):
    import numpy as np
    a,b=np.asarray(left),np.asarray(right)
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Attention matrix geometry/dtype differs')
    changed=np.any(a.view(np.uint8).reshape(-1,a.dtype.itemsize)!=b.view(np.uint8).reshape(-1,b.dtype.itemsize),axis=1)
    return dict(equal=a.tobytes()==b.tobytes(),finite=bool(np.isfinite(a).all()and np.isfinite(b).all()),
        shape=list(a.shape),dtype=str(a.dtype),different_elements=int(changed.sum()),
        maximum_absolute_error=0. if not a.size else float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
        candidate_sha256=hashlib.sha256(a.tobytes()).hexdigest(),reference_sha256=hashlib.sha256(b.tobytes()).hexdigest())


CANDIDATE_FAMILIES=('pinned_stock_v2','head_axis_padded_v1','head_axis_unpadded_v1','role_hybrid_v1')
CONTROL_LAWS=('literal_head_axis_surrogate_v1','actual_role_einsum_v1','actual_local_reshape_selected_einsum_v1')

def role_control(left,right,role):
    """Literal owning B1 selected einsum; no evaluation or candidate dispatch."""
    import mlx.core as mx
    if left.shape[0]!=1 or right.shape[0]!=1 or role not in('QK_selected','PV_selected'):
        raise ValueError('Selected role control requires one owning B1 row')
    query=mx.contiguous(left)[:,:,None]
    if role=='QK_selected':
        rows=mx.contiguous(mx.swapaxes(right,-1,-2))[:,None]
        return mx.einsum('bhld,blkd->bhlk',query,rows)[:,:,0]
    return mx.einsum('bhlk,blkd->bhld',query,mx.contiguous(right)[:,None])[:,:,0]



def local_control(left,right,*,transpose_b):
    """Literal B1 local operands with ordinary singleton strides on BOTH sides."""
    import mlx.core as mx
    if left.shape[0]!=1 or right.shape[0]!=1:
        raise ValueError('Local role control requires one owning B1 row')
    _,heads,k=left.shape;n=right.shape[-1]
    query=mx.contiguous(left).reshape(1,heads,1,k)
    if transpose_b:
        kv=mx.contiguous(mx.swapaxes(right,-1,-2)).reshape(1,1,n,k)
        coefficient=mx.swapaxes(kv,-1,-2)
    else:coefficient=mx.contiguous(right).reshape(1,1,k,n)
    return mx.matmul(query,coefficient)[:,:,0]


def run_captured(cases,check_live,*,persist_before_candidate=None,
                 candidate_families=('pinned_stock_v2',),persist_family=None,
                 control_law='literal_head_axis_surrogate_v1'):
    """Pure F32 components, using already-owned GPU operands and live guard."""
    if not callable(check_live):raise ValueError('Attention matrix components require owned live guard')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Attention matrix components require parent-owned GPU')
    from mlx2.runtime.gguf import source_ast_batched_splitk as kernels
    def live():
        if check_live()is False:raise RuntimeError('Attention matrix component ownership lost')
        return True
    candidate_families=tuple(candidate_families)
    if (not candidate_families or len(set(candidate_families))!=len(candidate_families)
            or any(name not in CANDIDATE_FAMILIES for name in candidate_families)):
        raise ValueError('Attention matrix candidate family inventory differs')
    if control_law not in CONTROL_LAWS:raise ValueError('Attention control law is unknown')
    controls=[];literal_controls=[];legacy=[];plans=[]
    live()
    for case in cases:
        left,right=case['left'],case['right'];batches,m,k=left.shape;n=right.shape[-1]
        plans.append(kernels.plan(batches,m,n,k,transpose_b=case['transpose_b']))
        rows=[];actual_rows=[]
        for index in range(batches):
            a=mx.contiguous(left[index:index+1])[:,:,None]
            source=right[index:index+1]
            # QK's ordinary right operand is a TRANSPOSE VIEW over compact
            # local/selected KV[N,K], not a newly contiguous[K,N] matrix.
            # Storage layout controls the pinned ordinary NT/NN dispatch.
            b=(mx.swapaxes(mx.contiguous(mx.swapaxes(source,-1,-2)),-1,-2)
               if case['transpose_b']else mx.contiguous(source))[:,None]
            rows.append((a@b)[:,:,0])
            if control_law in('actual_role_einsum_v1','actual_local_reshape_selected_einsum_v1')and case['role'].endswith('_selected'):
                actual_rows.append(role_control(left[index:index+1],source,case['role']))
            elif control_law=='actual_local_reshape_selected_einsum_v1':
                actual_rows.append(local_control(left[index:index+1],source,transpose_b=case['transpose_b']))
        literal=mx.concatenate(rows,axis=0)
        literal_controls.append(literal)
        controls.append(mx.concatenate(actual_rows,axis=0)if actual_rows else literal)
        # Original padded regular-batch path, retained as an arithmetic
        # negative control; it does not dispatch the new pinned helper.
        storage=mx.concatenate((left.reshape(batches,m*k),mx.zeros((batches,1),mx.float32)),axis=1)
        vectors=mx.as_strided(storage,(batches,m,k),(m*k+1,k,1))
        old_right=(mx.swapaxes(mx.contiguous(mx.swapaxes(right,-1,-2)),-1,-2)
                   if case['transpose_b']else right)
        legacy.append(mx.matmul(vectors,old_right))
    # The recoverable input/control fixture precedes any new candidate shader
    # creation. A compiler failure must not require another model capture.
    mx.eval([(case['left'],case['right'])for case in cases],literal_controls,controls,legacy);live()
    fixture=None
    if persist_before_candidate is not None:
        fixture=persist_before_candidate(cases,literal_controls,legacy,plans)
    families={};family_outputs={}
    legacy_comparisons=[_compare(old,reference)for old,reference in zip(legacy,controls)]
    for family in candidate_families:
        outputs=[];family_plans=[]
        if family!='pinned_stock_v2':
            from mlx2.runtime.gguf import source_ast_head_axis_matmul as head_axis
        for case,stock_plan in zip(cases,plans):
            left,right=case['left'],case['right']
            selected_impl=('pinned_stock_v2'if case['role'].endswith('_selected')else'head_axis_padded_v1')if family=='role_hybrid_v1'else family
            if selected_impl=='pinned_stock_v2':
                family_plans.append(stock_plan)
                outputs.append(kernels.private_matmul(left,right,transpose_b=case['transpose_b'],check_live=live))
            else:
                padded=selected_impl=='head_axis_padded_v1'
                family_plans.append(head_axis.plan(batches=left.shape[0],heads=left.shape[1],k=left.shape[-1],
                    n=right.shape[-1],transpose_b=case['transpose_b'],padded=padded))
                outputs.append(head_axis.private_matmul(left,right,transpose_b=case['transpose_b'],padded=padded,check_live=live))
        mx.eval(outputs);live();records=[]
        for index,(case,candidate,reference,old,plan)in enumerate(zip(cases,outputs,controls,legacy_comparisons,family_plans)):
            comparison=_compare(candidate,reference)
            records.append(dict(index=index,layer=case['layer'],role=case['role'],nodes=list(case['nodes']),
                transpose_b=case['transpose_b'],plan=plan,raw=comparison,
                old_regular_batch_control=old,literal_head_axis_surrogate_vs_actual_control=_compare(literal_controls[index],reference),
                reference_law=('actual_B1_selected_einsum'if control_law!='literal_head_axis_surrogate_v1'and case['role'].endswith('_selected')else
                    'actual_B1_local_reshape_matmul'if control_law=='actual_local_reshape_selected_einsum_v1'else'literal_B1_head_axis_matmul_surrogate'),
                selected_implementation=('pinned_stock_v2'if case['role'].endswith('_selected')else'head_axis_padded_v1')if family=='role_hybrid_v1'else family,
                passed=comparison['equal']and comparison['finite']))
        families[family]=dict(records=records,passed=bool(records)and all(row['passed']for row in records),
            exact_cases=sum(row['passed']for row in records),case_count=len(records),
            implementation_identity=(hashlib.sha256((kernels.implementation_identity()+head_axis.implementation_identity()).encode()).hexdigest()if family=='role_hybrid_v1'else
                kernels.implementation_identity()if family=='pinned_stock_v2'else head_axis.implementation_identity()),
            role='retained_surrogate_negative'if control_law=='literal_head_axis_surrogate_v1'and family=='pinned_stock_v2'and len(candidate_families)>1 else'experimental_candidate',
            by_role={role:dict(case_count=sum(r['role']==role for r in records),
                exact_cases=sum(r['role']==role and r['passed']for r in records),
                legacy_regular_batch_exact_cases=sum(r['role']==role and r['old_regular_batch_control']['equal']for r in records),
                literal_surrogate_exact_cases=sum(r['role']==role and r['literal_head_axis_surrogate_vs_actual_control']['equal']for r in records))
                for role in sorted({r['role']for r in records})})
        family_outputs[family]=outputs
        if persist_family is not None:persist_family(family,families[family])
    primary=('pinned_stock_v2'if control_law=='actual_local_reshape_selected_einsum_v1'and'pinned_stock_v2'in candidate_families else
             'role_hybrid_v1'if control_law=='actual_role_einsum_v1'and'role_hybrid_v1'in candidate_families else
             next((name for name in candidate_families if name!='pinned_stock_v2'),candidate_families[0]))
    candidates=(list(candidate_families)if control_law=='actual_local_reshape_selected_einsum_v1'else
                [name for name in candidate_families if name!='pinned_stock_v2']or list(candidate_families))
    passing=[name for name in candidates if families[name]['passed']]
    return dict(records=families[primary]['records'],outputs=family_outputs[primary],controls=controls,
                passed=bool(passing),passing_families=passing,families=families,selected_or_defaulted=False,
                primary_family=primary,control_law=control_law,
                implementation_identity=families[primary]['implementation_identity'],fixture=fixture)


def run(adapter,unit,check_live):
    if not callable(check_live):raise ValueError('Attention matrix capture needs owned live guard')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Attention matrix capture requires parent-owned GPU')
    import numpy as np
    from mlx2.adapters.deepseek_v41_dspark import load_source_dspark
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
    from ds41_resident_prefill_model_gate import _checkpoint,_counters
    def live():
        if check_live()is False:raise RuntimeError('Attention matrix capture ownership lost')
        return True
    live();unit=Path(unit);path=unit/'ast-attention-matmul-components.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Attention matrix evidence exists')
    model=adapter.model;weights=adapter.weights
    if getattr(weights,'source_resident_bank_session',None)is not None:raise ValueError('Capture requires no resident bank')
    result=dict(schema='mlx2.ds41-ast-attention-matmul-components.v1',checkpoint=_checkpoint(adapter),
        status='capture',completed=False,passed=False,qualified=False,rate_eligible=False,APCv2_publication=False,
        node=NODE,layers=list(LAYERS),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_scope='Actual DSpark candidate operands from one16row tile; no claim ordinary numerical admission')
    spec=json.loads((unit/'spec.json').read_text());result.update(source_revision=spec.get('source_revision'),source_snapshot_sha256=spec.get('source_snapshot_sha256'))
    _write(path,result);prefix=head=tile=None;cases=[];gate=None;drained=True
    old_persist=getattr(weights.expert_stream,'persist_atlas',None)
    try:
        weights.expert_stream.persist_atlas=lambda:None
        request={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
        encoded=adapter.tokenizer.encode(adapter.render_prompt(request),add_special_tokens=False)
        ids=list(getattr(encoded,'ids',encoded));prefix=model.make_cache();drained=False
        logits,features=model.forward_with_taps(mx.array([ids],mx.uint32),prefix,(37,38,39))
        mx.eval(logits,features,[state.state for state in prefix]);drained=True;live()
        anchor=int(mx.argmax(logits[0,-1]).item())
        head=load_source_dspark(model,expert_cache_bytes=512<<20,check_live=live)
        history=head.make_history();head.record_features(features,0,history);paths,_=head.propose_paths(anchor,history)
        head.close();head=None
        result.update(prompt_ids=ids,anchor=anchor,paths=[list(row)for row in paths],prefix_frontier=prefix[0].offset)
        with capture(model)as cases:
            drained=False
            tile=execute_batched_tree(model,prefix,anchor,paths,expert_law='source_ast_m1_rows_v1',forward_law='fully_batched_m1_v1')
            mx.eval(tile.logits,tile.features,[(case['left'],case['right'],case['captured_output'])for case in cases]);drained=True
        if len(cases)!=32:raise ValueError('Actual9-layer attention capture did not produce32 node1 components')
        live();before=_counters(adapter);result.update(status='components',case_count=len(cases));_write(path,result)
        def persist_inputs(captured,controls,legacy,plans):
            arrays={}
            for index,(case,control,old)in enumerate(zip(captured,controls,legacy)):
                for key,value in(('left',case['left']),('right',case['right']),
                                  ('ordinary_control',control),('old_regular_batch',old)):
                    arrays[f'case{index}_{key}']=np.asarray(value).copy()
            host_bytes=sum(value.nbytes for value in arrays.values())
            if host_bytes>128<<20:raise MemoryError('Actual attention operand archive exceeds128MiB bound')
            archive=unit/'ast-attention-matmul-operands.npz'
            with archive.open('xb')as stream:np.savez(stream,**arrays)
            digest=hashlib.sha256(archive.read_bytes()).hexdigest()
            fixture=dict(schema='mlx2.ds41-ast-attention-matmul-fixture.v1',
                checkpoint=result['checkpoint'],source_revision=result['source_revision'],
                source_snapshot_sha256=result['source_snapshot_sha256'],helper_sha256=result['helper_sha256'],
                producer_unit=str(unit.resolve()),archive=str(archive.resolve()),archive_sha256=digest,
                array_bytes=host_bytes,candidate_shader_attempted=False,
                ordinary_controls_completed=True,qualified=False,
                cases=[dict(index=index,layer=case['layer'],role=case['role'],nodes=list(case['nodes']),
                    transpose_b=case['transpose_b'],left_shape=list(case['left'].shape),right_shape=list(case['right'].shape),
                    dtype='float32',plan=plan)for index,(case,plan)in enumerate(zip(captured,plans))])
            manifest=unit/'ast-attention-matmul-fixture.json';_write(manifest,fixture)
            receipt=dict(manifest=str(manifest.resolve()),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                archive=str(archive.resolve()),archive_sha256=digest,array_bytes=host_bytes,
                persisted_before_candidate_compile=True)
            result.update(status='operand_fixture_saved',operand_fixture=receipt,
                operand_archive=str(archive),operand_archive_bytes=host_bytes,operand_archive_sha256=digest)
            _write(path,result)
            return receipt
        drained=False;gate=run_captured(cases,live,persist_before_candidate=persist_inputs);drained=True
        after=_counters(adapter);delta={key:after[key]-before.get(key,0)for key in after if type(after[key])in(int,float)}
        result.update(status='passed'if gate['passed']else'raw_mismatch',completed=True,passed=gate['passed'],
            cases=gate['records'],implementation_identity=gate['implementation_identity'],component_counter_delta=delta,
            component_no_expert_or_engram_io=all(delta.get(key,0)==0 for key in('expert_bytes_read','expert_uploads','engram_bytes_read','host_route_materializations','ordinary_replays')),
            first_mismatch=next((row for row in gate['records']if not row['passed']),None))
        return result
    except BaseException as error:
        result.update(status='failed',completed=False,error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((prefix,head,tile,cases,gate))
        if drained:
            if tile is not None:tile.transaction.abort()
            if head is not None:head.close()
            if prefix is not None:
                for state in prefix:state.close()
        if old_persist is not None:weights.expert_stream.persist_atlas=old_persist
        _write(path,result)


def _literal(path,name):
    """Read a pinned declaration without importing archived producer code."""
    import ast
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign)and any(isinstance(v,ast.Name)and v.id==name for v in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Producer arithmetic declaration missing:'+name)


def run_replay(adapter,unit,check_live,fixture_path,fixture_sha,*,candidate_families=('pinned_stock_v2',),
               control_law='literal_head_axis_surrogate_v1'):
    """Replay exact actual F32 fixtures; no DSpark, prefix or model forward."""
    if not callable(check_live):raise ValueError('Attention fixture replay needs owned live guard')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Attention fixture replay requires parent-owned GPU')
    import numpy as np
    from mlx2.runtime.gguf import source_ast_batched_splitk as kernels,source_dense
    from ds41_resident_prefill_model_gate import _checkpoint,_counters
    def live():
        if check_live()is False:raise RuntimeError('Attention fixture replay ownership lost')
        return True
    live();unit=Path(unit);path=unit/'ast-attention-matmul-components.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Replay evidence exists')
    result=dict(schema='mlx2.ds41-ast-attention-matmul-components.v1',mode='fixture_replay',control_law=control_law,
        checkpoint=_checkpoint(adapter),status='validating_fixture',completed=False,passed=False,
        qualified=False,rate_eligible=False,APCv2_publication=False,model_capture_performed=False,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    spec=json.loads((unit/'spec.json').read_text());result.update(source_revision=spec.get('source_revision'),source_snapshot_sha256=spec.get('source_snapshot_sha256'))
    _write(path,result);cases=[];gate=None;drained=True
    try:
        fixture_path=Path(fixture_path).resolve();raw=fixture_path.read_bytes()
        if type(fixture_sha)is not str or len(fixture_sha)!=64 or hashlib.sha256(raw).hexdigest()!=fixture_sha:
            raise ValueError('Actual attention fixture manifest digest differs')
        fixture=json.loads(raw)
        if(fixture.get('schema')!='mlx2.ds41-ast-attention-matmul-fixture.v1'
           or fixture.get('checkpoint')!=result['checkpoint']or fixture.get('ordinary_controls_completed')is not True
           or fixture.get('candidate_shader_attempted')is not False or len(fixture.get('cases',[]))!=32):
            raise ValueError('Actual attention fixture identity/completion contract differs')
        producer=Path(fixture['producer_unit']).resolve();archive=Path(fixture['archive']).resolve()
        if fixture_path.parent!=producer or archive.parent!=producer:
            raise ValueError('Actual attention fixture is outside its source-bound producer unit')
        source_manifest=json.loads((producer/'source/desktop-source.json').read_text())
        if(source_manifest.get('source_sha256')!=fixture['source_snapshot_sha256']
           or source_manifest.get('source_revision')!=fixture['source_revision']):
            raise ValueError('Actual attention producer source seal differs')
        roots=producer/'source'
        for relative in('src/mlx2/runtime/gguf/source_ast_batched_splitk.py','src/mlx2/runtime/gguf/source_dense.py'):
            if hashlib.sha256((roots/relative).read_bytes()).hexdigest()!=source_manifest['files'].get(relative):
                raise ValueError('Actual attention producer arithmetic source changed')
        old_kernel=roots/'src/mlx2/runtime/gguf/source_ast_batched_splitk.py'
        old_dense=roots/'src/mlx2/runtime/gguf/source_dense.py'
        if(_literal(old_kernel,'HASHES')!=kernels.HASHES or _literal(old_dense,'HASHES')!=source_dense.HASHES
           or _literal(old_dense,'LIBRARY_SHA256')!=source_dense.LIBRARY_SHA256):
            raise ValueError('Actual attention fixture installed-header/library arithmetic binding changed')
        # Verify the actual installed closure before accepting stored controls;
        # then recompute current ordinary controls and require raw equality.
        kernels._installed()
        if os.environ.get('MLX_ENABLE_TF32')!='0':raise ValueError('Replay requires explicit TF32 disabled')
        if archive.stat().st_size>130<<20 or hashlib.sha256(archive.read_bytes()).hexdigest()!=fixture['archive_sha256']:
            raise ValueError('Actual attention operand archive changed or exceeds bound')
        expected={f'case{i}_{name}'for i in range(32)for name in('left','right','ordinary_control','old_regular_batch')}
        stored=[];host_bytes=0
        with np.load(archive,allow_pickle=False)as arrays:
            if set(arrays.files)!=expected:raise ValueError('Actual attention operand array inventory differs')
            for index,metadata in enumerate(fixture['cases']):
                role=metadata['role'];left=arrays[f'case{index}_left'];right=arrays[f'case{index}_right'];control=arrays[f'case{index}_ordinary_control'];old=arrays[f'case{index}_old_regular_batch']
                values=(left,right,control,old);host_bytes+=sum(v.nbytes for v in values)
                if(any(v.dtype!=np.float32 or not np.isfinite(v).all()for v in values)
                   or index!=metadata['index']or metadata['layer']not in range(9)
                   or role not in('QK_local','QK_selected','PV_local','PV_selected')
                   or metadata['transpose_b']!=role.startswith('QK')or metadata['nodes']!=[1,6,11]
                   or list(left.shape)!=metadata['left_shape']or list(right.shape)!=metadata['right_shape']
                   or left.shape[:2]!=(3,64)or right.shape[0]!=3 or left.shape[2]!=right.shape[1]
                   or control.shape!=(3,64,right.shape[2])or old.shape!=control.shape or host_bytes>128<<20):
                    raise ValueError('Actual attention fixture shape/role/finite contract differs')
                cases.append(dict(layer=metadata['layer'],role=role,nodes=tuple(metadata['nodes']),
                    transpose_b=metadata['transpose_b'],left=mx.array(left),right=mx.array(right)))
                stored.append(control.copy())
        if host_bytes!=fixture['array_bytes']:raise ValueError('Actual attention fixture byte inventory differs')
        result.update(status='fixture_verified',fixture=dict(path=str(fixture_path),sha256=fixture_sha,
            producer_source_snapshot_sha256=fixture['source_snapshot_sha256'],archive=str(archive),
            archive_sha256=fixture['archive_sha256'],array_bytes=host_bytes,header_library_pins_match=True),case_count=32)
        _write(path,result);live();before=_counters(adapter)
        def verify_controls(captured,controls,legacy,plans):
            checks=[_compare(value,reference)for value,reference in zip(controls,stored)]
            result['stored_ordinary_control_checks']=checks
            if not all(row['equal']and row['finite']for row in checks):
                raise ValueError('Current ordinary controls differ from stored installed-baseline fixture')
            result.update(status='stored_literal_controls_exact',all32_current_ordinary_controls_match_stored=True,all32_current_literal_surrogate_controls_match_stored=True,
                stored_control_scope='literal_B1_head_axis_surrogate; selected-role stored values are not owning einsum controls')
            _write(path,result)
            return result['fixture']
        def persist_family(name,value):
            result.setdefault('candidate_families',{})[name]=value
            result['status']='candidate_family_completed';_write(path,result)
        drained=False;gate=run_captured(cases,live,persist_before_candidate=verify_controls,
            candidate_families=candidate_families,persist_family=persist_family,control_law=control_law);drained=True
        after=_counters(adapter);delta={key:after[key]-before.get(key,0)for key in after if type(after[key])in(int,float)}
        result.update(status='passed'if gate['passed']else'raw_mismatch',completed=True,passed=gate['passed'],
            cases=gate['records'],implementation_identity=gate['implementation_identity'],component_counter_delta=delta,
            candidate_families=gate['families'],passing_families=gate['passing_families'],selected_or_defaulted=False,
            primary_candidate_family=gate['primary_family'],top_level_cases_scope='primary_candidate_family',
            component_no_expert_or_engram_io=all(delta.get(key,0)==0 for key in('expert_bytes_read','expert_uploads','engram_bytes_read','host_route_materializations','ordinary_replays')),
            first_mismatch=next((row for row in gate['records']if not row['passed']),None))
        return result
    except BaseException as error:
        result.update(status='failed',completed=False,error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((cases,gate))
        _write(path,result)
