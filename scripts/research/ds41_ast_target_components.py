"""Owned actual-Source AST target functional probe, never standalone GPU owner.

The outer one-artifact CPG/radio/thermal/source controller loads the ordinary
Source target, runs its reference smoke, and supplies check_live. No server,
checkpoint loading, lease acquisition or APC publication occurs here.
"""
import numpy as np

_PROBE_QUARANTINE=[]


def _proof_snapshot(cache):
    """Preserve physical capacity/owner descriptors, not visible state only."""
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
    return snapshot_prompt_cache_descriptors(cache,require_committed=True)[0]


def _first_failed_proof_check(value,path=()):
    checks=('raw_bits_equal','finite','topology_equal','private_schema_expected','metadata_equal','capacity_equal','absence_equal')
    if isinstance(value,dict):
        for key in checks:
            if key in value and value[key]is not True:
                return dict(path=list(path+(key,)),value=value[key],comparison=value)
        for key,item in value.items():
            failed=_first_failed_proof_check(item,path+(key,))
            if failed is not None:return failed
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):
            failed=_first_failed_proof_check(item,path+(index,))
            if failed is not None:return failed
    return None


def _private_proof_diagnostic(path,model,result,controls):
    """Persist bounded first-failure evidence before admission or refusal."""
    if path is None:return
    from pathlib import Path
    import json,os
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    if path.exists()or temporary.exists():raise FileExistsError('Private proof diagnostic already exists')
    nodes=[]
    for value in controls:
        failed=_first_failed_proof_check(value)
        nodes.append(dict(node=value['node'],passed=failed is None,
            logits_raw_equal=value['logits']['raw_bits_equal'],features_raw_equal=value['features']['raw_bits_equal'],
            first_failed_check=failed))
    report=dict(schema='mlx2.ds41-prepared-private-gate.v1',target_model_identity=model.identity,
        source_revision_token=result.transaction.provider_revision_token,
        forward_law=result.receipt.get('forward_law'),numerical_profile=result.transaction.profile,
        completed=True,passed=all(row['passed']for row in nodes),nodes=nodes,
        first_failed_node=next((row for row in nodes if not row['passed']),None),
        proof_local_admission_granted=False,APCv2_insertions=0,qualified=False,performance_claim=False,
        snapshot_contract='Capacity-preserving independent descriptors; visible arrays and topology/capacity gates remain exact')
    with temporary.open('x')as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def compare_arrays(a, b, *, atol, rtol):
    """Diagnostics include valid zero-length source partial-group planes."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Source AST comparison shape mismatch: {a.shape} versus {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"Source AST comparison dtype mismatch: {a.dtype} versus {b.dtype}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    maximum = (float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
               if a.size else 0.0)
    return dict(shape=list(a.shape), finite=finite,
                raw_bits_equal=bool(a.tobytes() == b.tobytes()),
                tolerance_valid=bool(finite and np.allclose(a, b, atol=atol, rtol=rtol)),
                maximum_absolute_error=maximum)


def _run(model, cache, check_live, *, anchor=3,
        paths=((4, 5, 6, 7, 8), (9, 10, 11, 12, 13), (14, 15, 16, 17, 18)),
        atol=3e-3, rtol=3e-3, expert_law="ordinary_m16_v1"):
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_target import execute_tree, _clone
    if not callable(check_live):
        raise ValueError("Source AST probe requires caller-owned live/thermal callback")
    check_live()
    if mx.default_device() != mx.gpu:
        raise ValueError("actual Source AST component requires the owned GPU backend")
    if getattr(model.weights, "source_resident_gather_policy", "reference") != "reference":
        raise ValueError("AST ordinary comparison requires reference Source expert policy at load")
    baseline = [_clone(row) for row in cache]
    result = execute_tree(model, cache, anchor, paths, expert_law=expert_law)
    # Save a genuinely partial accepted branch before releasing the private
    # transaction. The next ordinary calls use independent private cache rows.
    committed = result.transaction.commit_paths([(0, 6, 7, 8)])[0]
    def compare(a, b):
        return compare_arrays(a, b, atol=atol, rtol=rtol)
    def states(left, right):
        rows = []
        for a, b in zip(left, right):
            row = dict(metadata_equal=a.meta_state == b.meta_state, planes=[])
            for x, y in zip(a.state, b.state):
                row["planes"].append(dict(absence_equal=(x is None) == (y is None))
                    if x is None or y is None else compare(x, y))
            rows.append(row)
        return rows
    controls, raw, branch8 = [], {}, None
    for rank, path in enumerate(paths):
        local = [_clone(row) for row in baseline]
        for depth, token in enumerate((anchor,) + tuple(path)):
            check_live()
            logits, features = model.forward_with_taps(mx.array([[token]], mx.int32), local, (37, 38, 39))
            mx.eval(logits, features, [row.state for row in local])
            node = 0 if depth == 0 else 1 + rank*5 + depth-1
            controls.append(dict(node=node, branch=rank, logits=compare(result.logits[:, node:node+1], logits),
                                 features=compare(result.features[:, node:node+1], features),
                                 state=states(result.frontiers[node], local)))
            raw[f"ordinary_logits_b{rank}_d{depth}"] = np.asarray(logits)
            if rank == 1 and depth == 3:
                branch8 = [_clone(row) for row in local]
    check_live()
    partial_state = states(committed, branch8)
    # Same supplied continuation token in both arms isolates committed state;
    # quality/accepted-length testing remains a separate text-level oracle.
    continuation_token = int(paths[1][3])
    left = model(mx.array([[continuation_token]], mx.int32), committed, logits_to_keep=0)
    right = model(mx.array([[continuation_token]], mx.int32), branch8, logits_to_keep=0)
    mx.eval(left, right, [row.state for row in committed], [row.state for row in branch8])
    raw.update(candidate_logits=np.asarray(result.logits), candidate_features=np.asarray(result.features),
               committed_continuation_logits=np.asarray(left), ordinary_continuation_logits=np.asarray(right))
    check_live()
    return dict(receipt=result.receipt, comparisons=controls,
                partial_commit_state=partial_state, continuation=compare(left, right),
                continuation_state=states(committed, branch8), live_cache_unchanged=states(cache, baseline),
                numerical_classification="profile_drift_diagnostics_pending_actual_oracle",
                quality_verdict="not_tested", qualification=False, performance_claim=False,
                tolerances=dict(atol=atol, rtol=rtol), raw_arrays=raw)


def _validate_batched_proof_entry(model,expert_law,forward_law,resident_prepare):
    resident=expert_law in ('source_ast_resident_rows_v1','source_ast_hybrid_rows_v1')
    if (forward_law!='fully_batched_m1_v1' or expert_law not in ('source_ast_m1_rows_v1','source_ast_resident_rows_v1','source_ast_hybrid_rows_v1')
            or getattr(model,'_mlx2_source_ast_target_laws',None)!=(expert_law,forward_law)):
        raise ValueError('Batched AST proof requires target laws bound by oracle preparation at load')
    weights=model.weights
    if (getattr(weights,'source_resident_gather_policy','reference')!='reference'
            or getattr(weights,'source_resident_bank_session',None)is not None):
        raise ValueError('Batched AST proof requires ordinary expert ownership before controls')
    if resident and not callable(resident_prepare):
        raise ValueError('Resident AST proof requires closed-cohort preparation after ordinary controls')
    if not resident and resident_prepare is not None:
        raise ValueError('Ordinary M1 AST proof cannot install a resident preparation callback')
    return resident


def _validate_resident_proof_owner(model):
    manager=getattr(model,'_source_ast_bank_manager',None)
    if manager is not None:
        manager._idle()
        if (manager.model is not model or manager.bank is None
                or getattr(model.weights,'_source_ast_bank_manager',None)is not manager):
            raise ValueError('Hybrid proof manager/model/store ownership differs')
        return manager.bank.receipt()
    weights=model.weights;session=getattr(weights,'source_resident_bank_session',None)
    if (getattr(weights,'source_resident_gather_policy','reference')!='resident_bank_v1' or session is None
            or any(getattr(session,key,False)for key in ('active','replaying','poisoned'))):
        raise ValueError('Resident AST proof preparation did not install an idle owned bank')
    bank=getattr(session,'bank',None)
    if bank is None or getattr(weights.expert_stream,'_resident_bank',None)is not bank:
        raise ValueError('Resident AST proof bank/store ownership differs')
    bank.assert_current()
    return bank.receipt()


def _run_batched_exact(model, cache, check_live, *, anchor, paths, atol, rtol, expert_law, forward_law,diagnostic_path=None,resident_prepare=None,expected_hybrid_route=None):
    """Prepare a numerical oracle with the target identity fixed at load.

    A proof-local admission follows every private node's raw payload/topology
    gate. It exists only to exercise ordinary prepared/commit/abort operations;
    no APC entry is inserted and the actual serving oracle is still external.
    """
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_target import execute_tree, _paths, PARENTS
    from mlx2.adapters.deepseek_v41_ast_batched_target import _EXACT_PUBLICATION_SEAL
    if not callable(check_live):raise ValueError('Batched AST proof requires caller-owned live callback')
    if check_live()is False or mx.default_device()!=mx.gpu:
        raise ValueError('Batched AST proof requires the owned GPU backend')
    resident=_validate_batched_proof_entry(model,expert_law,forward_law,resident_prepare)
    PATHS=_paths(PARENTS)
    def compare(a,b):return compare_arrays(a,b,atol=atol,rtol=rtol)
    def states(left,right,*,private=False):
        rows=[]
        for a,b in zip(left,right):
            row=dict(planes=[])
            if private:
                row['topology_equal']=tuple(a.meta_state[1:])==tuple(b.meta_state[1:])
                row['private_schema_expected']=getattr(a,'ast_private_profile',None)=='source_ast_m1_batch_axis_segments_candidate_v1'
            else:row['metadata_equal']=a.meta_state==b.meta_state
            row['capacity_equal']=a._capacity==b._capacity
            row['candidate_capacity']=a._capacity;row['ordinary_capacity']=b._capacity
            row['candidate_schema']=a.meta_state[0];row['ordinary_schema']=b.meta_state[0]
            for x,y in zip(a.state,b.state):
                row['planes'].append(dict(absence_equal=(x is None)==(y is None))if x is None or y is None else compare(x,y))
            rows.append(row)
        return rows
    def exact(value):
        if isinstance(value,dict):
            for key in('raw_bits_equal','finite','topology_equal','private_schema_expected','metadata_equal','capacity_equal','absence_equal'):
                if key in value and value[key]is not True:return False
            return all(exact(v)for v in value.values())
        return all(exact(v)for v in value)if isinstance(value,(tuple,list))else True
    owners=[];result=abort_result=None;raw={};controls=[];private_controls=[]
    resident_context=None;resident_entered=False;resident_preparation=None
    baseline=_proof_snapshot(cache);owners.append(baseline)
    ordinary={};ordinary_logits={};ordinary_features={}
    try:
        # Ordinary execution finishes before the private tree owns the model.
        for rank,path in enumerate(paths):
            local=_proof_snapshot(baseline);owners.append(local)
            for depth,token in enumerate((anchor,*path)):
                if check_live()is False:raise RuntimeError('Batched AST proof ownership lost')
                logits,features=model.forward_with_taps(mx.array([[token]],mx.int32),local,(37,38,39))
                mx.eval(logits,features,[row.state for row in local])
                node=0 if depth==0 else 1+rank*5+depth-1
                raw[f'ordinary_logits_b{rank}_d{depth}']=np.asarray(logits)
                if node not in ordinary:
                    ordinary[node]=_proof_snapshot(local);owners.append(ordinary[node])
                    ordinary_logits[node]=logits;ordinary_features[node]=features
        if resident:
            # The caller owns discovery/preheating and its closed-cohort bank.
            # Every ordinary frontier above predates this numerical consumer.
            resident_context=resident_prepare(model,cache,anchor,paths,check_live)
            if not hasattr(resident_context,'__enter__')or not hasattr(resident_context,'__exit__'):
                raise ValueError('Resident AST preparation must return an owned context manager')
            preparation=resident_context.__enter__();resident_entered=True
            resident_preparation=dict(bank=_validate_resident_proof_owner(model),preparation=preparation,
                ordinary_controls_completed_before_bank=True)
        result=execute_tree(model,cache,anchor,paths,expert_law=expert_law,forward_law=forward_law)
        if expert_law=='source_ast_hybrid_rows_v1':
            actual=result.receipt.get('hybrid_bank')or{}
            if expected_hybrid_route=='resident':
                accepted=(actual.get('actual_expert_route')=='resident'and actual.get('exact_M1_recovery')is False)
            elif expected_hybrid_route=='m1_recovery':
                accepted=(actual.get('actual_expert_route')=='m1'and actual.get('exact_M1_recovery')is True)
            else:accepted=False
            if not accepted:raise ValueError('Hybrid proof did not execute its requested genuine resident/miss arm')
        for node in range(16):
            private=result.transaction.diagnostic_cache(PATHS[node]);owners.append(private)
            mx.eval([row.state for row in private])
            private_controls.append(dict(node=node,logits=compare(result.logits[:,node:node+1],ordinary_logits[node]),
                features=compare(result.features[:,node:node+1],ordinary_features[node]),
                state=states(private,ordinary[node],private=True)))
        _private_proof_diagnostic(diagnostic_path,model,result,private_controls)
        if not exact(private_controls):raise ValueError('Batched AST private raw node/state gate failed before proof-local admission')
        # Source/revision-bound and deliberately after the complete raw gate.
        result.transaction._admit_exact_publication(_EXACT_PUBLICATION_SEAL,result.transaction.provider_revision_token)
        for rank,path in enumerate(paths):
            for depth in range(6):
                node=0 if depth==0 else 1+rank*5+depth-1
                prepared=result.transaction.prepared_cache(PATHS[node]);owners.append(prepared)
                mx.eval([row.state for row in prepared])
                controls.append(dict(node=node,branch=rank,logits=compare(result.logits[:,node:node+1],ordinary_logits[node]),
                    features=compare(result.features[:,node:node+1],ordinary_features[node]),state=states(prepared,ordinary[node])))
        if not exact(controls):raise ValueError('Batched AST admitted prepared frontier differs from ordinary')
        committed=result.transaction.commit_paths([(0,6,7,8)])[0];owners.append(committed)
        branch8=_proof_snapshot(ordinary[8]);owners.append(branch8)
        partial_state=states(committed,branch8)
        continuation_token=int(paths[1][3])
        left=model(mx.array([[continuation_token]],mx.int32),committed,logits_to_keep=0)
        if resident:
            # This exact path/token is ordinary node9, captured BEFORE bank
            # installation. Do not compare the new consumer against itself.
            right=ordinary_logits[9];reference_continuation=ordinary[9]
        else:
            right=model(mx.array([[continuation_token]],mx.int32),branch8,logits_to_keep=0)
            reference_continuation=branch8
        mx.eval(left,right,[row.state for row in committed],[row.state for row in reference_continuation])
        continuation=compare(left,right);continuation_state=states(committed,reference_continuation)
        raw.update(candidate_logits=np.asarray(result.logits),candidate_features=np.asarray(result.features),
            committed_continuation_logits=np.asarray(left),ordinary_continuation_logits=np.asarray(right))
        abort_result=execute_tree(model,cache,anchor,paths,expert_law=expert_law,forward_law=forward_law)
        abort_result.transaction.abort();abort_result.transaction.abort()
        try:abort_result.transaction.commit_paths([(0,)])
        except ValueError:closed_refused=True
        else:closed_refused=False
        isolated=states(cache,baseline)
        gates=dict(private_all16_raw_exact=exact(private_controls),prepared_all16_raw_exact=exact(controls),
            committed_state_exact=exact(partial_state),continuation_exact=exact(continuation)and exact(continuation_state),
            isolated=exact(isolated),abort_closed=abort_result.transaction.closed,commit_after_abort_refused=closed_refused)
        if not all(gates.values()):raise ValueError('Batched AST commit/abort/continuation/isolation gate failed')
        return dict(receipt=result.receipt,comparisons=controls,private_pre_admission=private_controls,
            partial_commit_state=partial_state,continuation=continuation,continuation_state=continuation_state,
            live_cache_unchanged=isolated,batched_exact_gates=gates,proof_local_admission=True,
            APCv2_insertions=0,resident_preparation=resident_preparation,
            continuation_reference_scope=('prebank_ordinary_branch1_depth4_node9'if resident else'ordinary_replay'),
            hybrid_expected_route=expected_hybrid_route,
            numerical_classification='raw_bits_exact',quality_verdict='not_tested',
            qualification=False,performance_claim=False,tolerances=dict(atol=atol,rtol=rtol),raw_arrays=raw)
    except BaseException:
        # Preserve owners if a failed queue cannot be drained. This helper
        # never owns/reacquires the GPU and cannot declare the queue safe.
        try:
            mx.eval([[row.state for row in owner]for owner in owners],
                    None if result is None else(result.logits,result.features),
                    None if abort_result is None else(abort_result.logits,abort_result.features))
            mx.synchronize()
        except BaseException:
            _PROBE_QUARANTINE.append((model,cache,owners,result,abort_result,resident_context))
            owners=[];result=abort_result=None;resident_context=None
        raise
    finally:
        import sys
        try:
            if result is not None:result.transaction.abort()
            if abort_result is not None:abort_result.transaction.abort()
            for owner in owners:
                for row in owner:row.close()
        finally:
            if resident_context is not None and resident_entered:
                resident_context.__exit__(*sys.exc_info())


def run(model, cache, check_live, *, anchor=3,
        paths=((4, 5, 6, 7, 8), (9, 10, 11, 12, 13), (14, 15, 16, 17, 18)),
        atol=3e-3, rtol=3e-3, diagnose_moe_layer=None, diagnose_moe_layers=None, expert_law=None,forward_law=None,diagnostic_path=None,resident_prepare=None,hybrid_miss_prepare=None):
    """Optional layer14/node12 oracle is untimed and source-result returning."""
    import os
    layers_env = os.environ.get("MLX2_DS41_AST_MOE_DIAGNOSTIC_LAYERS")
    if diagnose_moe_layers is None and layers_env:
        if layers_env != "0:15":
            raise ValueError("AST diagnostic layers environment supports only0:15")
        diagnose_moe_layers = tuple(range(15))
    if diagnose_moe_layer is None and os.environ.get("MLX2_DS41_AST_MOE_DIAGNOSTIC") == "1":
        diagnose_moe_layer = 14
    if expert_law is None:
        expert_law = os.environ.get("MLX2_DS41_AST_EXPERT_LAW", "ordinary_m16_v1")
    if forward_law is None:forward_law=os.environ.get('MLX2_DS41_AST_FORWARD_LAW','reference')
    if expert_law=='source_ast_hybrid_rows_v1':
        if forward_law!='fully_batched_m1_v1' or not callable(resident_prepare)or not callable(hybrid_miss_prepare):
            raise ValueError('Hybrid oracle proof requires fresh full-cohort and genuine missing-cohort owned contexts')
        if diagnose_moe_layer is not None or diagnose_moe_layers is not None:
            raise ValueError('Hybrid oracle proof cannot combine old-law localization')
        first=_run_batched_exact(model,cache,check_live,anchor=anchor,paths=paths,atol=atol,rtol=rtol,
            expert_law=expert_law,forward_law=forward_law,diagnostic_path=diagnostic_path,
            resident_prepare=resident_prepare,expected_hybrid_route='resident')
        miss_path=None
        if diagnostic_path is not None:
            from pathlib import Path
            base=Path(diagnostic_path);miss_path=base.with_name(base.stem+'-hybrid-miss'+base.suffix)
        second=_run_batched_exact(model,cache,check_live,anchor=anchor,paths=paths,atol=atol,rtol=rtol,
            expert_law=expert_law,forward_law=forward_law,diagnostic_path=miss_path,
            resident_prepare=hybrid_miss_prepare,expected_hybrid_route='m1_recovery')
        second_raw=second.pop('raw_arrays')
        first['raw_arrays'].update({'hybrid_miss_'+key:value for key,value in second_raw.items()})
        first['hybrid_miss_proof']=second
        first['hybrid_exact_gates']=dict(resident_full_operations=all(first['batched_exact_gates'].values()),
            genuine_miss_full_operations=all(second['batched_exact_gates'].values()),
            resident_executed=first['receipt']['hybrid_bank']['actual_expert_route']=='resident',
            genuine_miss_M1_recovery=second['receipt']['hybrid_bank']['exact_M1_recovery']is True,
            declared_revision_equal=first['receipt']['source_revision_token']==second['receipt']['source_revision_token'])
        if not all(first['hybrid_exact_gates'].values()):raise ValueError('Hybrid resident/miss proof gates differ')
        return first
    if hybrid_miss_prepare is not None:raise ValueError('Genuine missing-cohort callback requires explicit hybrid law')
    if forward_law!='reference':
        if diagnose_moe_layer is not None or diagnose_moe_layers is not None:
            raise ValueError('Batched AST proof cannot combine old-law localization')
        return _run_batched_exact(model,cache,check_live,anchor=anchor,paths=paths,atol=atol,rtol=rtol,
                                  expert_law=expert_law,forward_law=forward_law,diagnostic_path=diagnostic_path,resident_prepare=resident_prepare)
    if resident_prepare is not None:raise ValueError('Resident AST preparation requires explicit fully_batched_m1_v1')
    kwargs = dict(anchor=anchor, paths=paths, atol=atol, rtol=rtol, expert_law=expert_law)
    if diagnose_moe_layer is None and diagnose_moe_layers is None:
        return _run(model, cache, check_live, **kwargs)
    if not callable(check_live):
        raise ValueError("Source AST diagnostic requires caller-owned live callback")
    if expert_law != "ordinary_m16_v1":
        raise ValueError("Old-law localization diagnostic requires ordinary_m16_v1")
    import mlx.core as mx
    check_live()
    if mx.default_device() != mx.gpu:
        raise ValueError("Source AST diagnostic requires the owned GPU backend")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("ds41_ast_moe_diagnostic", Path(__file__).with_name("ds41_ast_moe_diagnostic.py"))
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    with diagnostic.instrument(model, check_live, layer=diagnose_moe_layer,
                               layers=diagnose_moe_layers) as report:
        result = _run(model, cache, check_live, **kwargs)
    result["raw_arrays"].update({"moe_diagnostic_" + name: value
                                 for name, value in report.pop("raw_arrays").items()})
    result["moe_diagnostic"] = report
    return result
