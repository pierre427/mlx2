"""Parent-owned actual DSpark AST scheduler proof; no GPU ownership on import.

The outer one-artifact controller must run ordinary reference/ownership checks,
load the same Source policy as the exact proof, and supply a live guard.
"""
from pathlib import Path
import hashlib
import json

LAW='source_ast_m1_rows_v1'
RESIDENT_LAW='source_ast_resident_rows_v1'
HYBRID_LAW='source_ast_hybrid_rows_v1'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def validate_prefreeze_policy(proof_unit,policy):
    """CPU-only exact load-policy/source gate before an owned GPU unit.

    Paths participate in the existing target identity. Equal file content
    does not authorize normalizing those paths or rewriting a proof binding.
    """
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.adapters.deepseek_v41_ast_target import implementation_identity
    proof_unit=Path(proof_unit)
    previous=json.loads((proof_unit/'ordinary-reference.json').read_text())['diagnostics']['execution_policy']
    # Match the owned ordinary-reference harness's adapter load: these fields
    # belong to the streaming/host controller, and atlas/preheat are restored
    # by that controller rather than included in target math identity.
    load_policy={key:value for key,value in policy.items()if key not in ('moe_expert_streaming','host_memory_signals')}
    if 'deepseek_source_experts'in load_policy:
        load_policy['deepseek_source_experts']={key:value for key,value in load_policy['deepseek_source_experts'].items()
                                               if key not in ('atlas_path','preheat')}
    normalized=source_policy(load_policy)
    forward=normalized.get('deepseek_source_dspark_target_forward','reference')
    previous_forward=previous.get('deepseek_source_dspark_target_forward','reference')
    if forward!='reference' or previous_forward!='reference':
        for key in('deepseek_source_dspark_target_forward','deepseek_source_dspark_target_expert_law'):
            if normalized.get(key)!=previous.get(key):
                raise ValueError('Source AST prefreeze load-bound target law differs: '+key)
    current={key:value for key,value in normalized.items()if not key.startswith('deepseek_source_dspark_')}
    previous={key:value for key,value in previous.items()if not key.startswith('deepseek_source_dspark_')}
    if current!=previous:
        differences={key:dict(proof=previous.get(key),requested=current.get(key))
                     for key in sorted(set(previous)|set(current))if previous.get(key)!=current.get(key)}
        raise ValueError('Source AST prefreeze target policy differs: '+json.dumps(differences,sort_keys=True))
    for name,digest in implementation_identity():
        if _sha(proof_unit/'source'/name)!=digest:
            raise ValueError('Source AST prefreeze target source differs: '+name)
    return dict(policy_sha256=hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest(),
                target_sources_exact=True,policy_exact=True)


def validate_proof(proof_unit,model,provider):
    """Read actual completed proof and reject drift before creating an oracle."""
    import numpy as np
    from mlx2.adapters.deepseek_v41_ast_target import implementation_identity,PARENTS
    proof_unit=Path(proof_unit).resolve()
    record=json.loads((proof_unit/'dspark-ast-functional.json').read_text())
    comparison=json.loads((proof_unit/'dspark-target-comparison.json').read_text())
    spec=json.loads((proof_unit/'spec.json').read_text())
    receipt=comparison['receipt']
    requested_law=getattr(provider,'expert_law',LAW)
    if (requested_law not in (LAW,RESIDENT_LAW,HYBRID_LAW) or
            (requested_law in (RESIDENT_LAW,HYBRID_LAW) and getattr(provider,'forward_law','reference')!='fully_batched_m1_v1')):
        raise ValueError('Actual Source AST proof requests an unsupported expert/forward law')
    if (record.get('schema')!='mlx2.ds41-dspark-ast-owned-functional.v1'
            or record.get('stage')!='completed'or record.get('passed')is not True
            or record.get('raw_bits_exact')is not True or record.get('numerical_classification')!='raw_bits_exact'
            or record.get('source_revision')!=spec.get('source_revision')
            or record.get('source_snapshot_sha256')!=spec.get('source_snapshot_sha256')
            or record.get('checkpoint')!=spec.get('checkpoint')
            or receipt!=record.get('target_receipt')or receipt.get('expert_law')!=requested_law
            or receipt.get('forward_law','reference')!=getattr(provider,'forward_law','reference')
            or receipt.get('source_revision_token')!=provider.source_revision_token
            or receipt.get('checkpoint')!=model.identity or receipt.get('parents')!=list(PARENTS)
            or receipt.get('tensor_tile_rows')!=16 or receipt.get('grouped_moe_calls')!=40
            or receipt.get('serial_whole_model_calls')!=0 or receipt.get('physical_target_invocations')!=1):
        raise ValueError('Actual Source AST proof identity/law/completion differs: '+json.dumps(dict(
            proof_model=receipt.get('checkpoint'),live_model=model.identity,
            proof_provider=receipt.get('source_revision_token'),live_provider=provider.source_revision_token,
            proof_law=receipt.get('expert_law'),stage=record.get('stage'),raw_bits_exact=record.get('raw_bits_exact')),sort_keys=True))
    if getattr(provider,'forward_law','reference')!='reference':
        expected_gates={'private_all16_raw_exact','prepared_all16_raw_exact','committed_state_exact',
            'continuation_exact','isolated','abort_closed','commit_after_abort_refused'}
        gates=comparison.get('batched_exact_gates',{})
        if (provider.forward_law!='fully_batched_m1_v1' or set(gates)!=expected_gates
                or not all(value is True for value in gates.values())
                or comparison.get('proof_local_admission')is not True or comparison.get('APCv2_insertions')!=0):
            raise ValueError('Actual batched Source AST proof lacks prepared/commit/abort/isolation gates')
        if requested_law in (RESIDENT_LAW,HYBRID_LAW):
            preparation=comparison.get('resident_preparation')or{}
            bank=preparation.get('bank')or{}
            if (preparation.get('ordinary_controls_completed_before_bank')is not True
                    or bank.get('published')is not True or bank.get('active_leases')!=0
                    or comparison.get('continuation_reference_scope')!='prebank_ordinary_branch1_depth4_node9'
                    or receipt.get('source_route_host_materializations')!=0
                    or receipt.get('resident_bank_tile')is not True
                    or receipt.get('zero_host_expert_ids_claimed')is not True):
                raise ValueError('Actual resident Source AST proof lacks independent prebank controls/owned GPU-route gates')
        if requested_law==HYBRID_LAW:
            expected_hybrid={'resident_full_operations','genuine_miss_full_operations','resident_executed',
                             'genuine_miss_M1_recovery','declared_revision_equal'}
            hybrid=comparison.get('hybrid_exact_gates')or{}
            miss=comparison.get('hybrid_miss_proof')or{};miss_receipt=miss.get('receipt')or{}
            route=miss_receipt.get('hybrid_bank')or{}
            if (set(hybrid)!=expected_hybrid or not all(value is True for value in hybrid.values())
                    or set(miss.get('batched_exact_gates',{}))!=expected_gates
                    or not all(value is True for value in miss['batched_exact_gates'].values())
                    or miss.get('proof_local_admission')is not True or miss.get('APCv2_insertions')!=0
                    or miss.get('continuation_reference_scope')!='prebank_ordinary_branch1_depth4_node9'
                    or miss.get('numerical_classification')!='raw_bits_exact'
                    or miss_receipt.get('expert_law')!=HYBRID_LAW
                    or miss_receipt.get('forward_law')!='fully_batched_m1_v1'
                    or miss_receipt.get('checkpoint')!=model.identity or miss_receipt.get('parents')!=list(PARENTS)
                    or miss_receipt.get('tensor_tile_rows')!=16 or miss_receipt.get('grouped_moe_calls')!=80
                    or miss_receipt.get('physical_target_invocations')!=2
                    or miss_receipt.get('failed_private_target_invocations')!=1
                    or miss_receipt.get('successful_target_invocations')!=1
                    or(miss.get('resident_preparation')or{}).get('ordinary_controls_completed_before_bank')is not True
                    or miss_receipt.get('source_revision_token')!=provider.source_revision_token
                    or route.get('actual_expert_route')!='m1'or route.get('exact_M1_recovery')is not True
                    or (receipt.get('hybrid_bank')or{}).get('actual_expert_route')!='resident'):
                raise ValueError('Actual hybrid Source AST proof lacks fresh resident and genuine-miss full-operation gates')
    arrays=[];metadata=[];absence=[]
    def walk(value):
        if isinstance(value,dict):
            if 'raw_bits_equal'in value:arrays.append(value)
            if 'metadata_equal'in value:metadata.append(value['metadata_equal'])
            if 'absence_equal'in value:absence.append(value['absence_equal'])
            for item in value.values():walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    walk(comparison)
    if (len(arrays)!=record.get('comparison_count')or len(arrays)<1234
            or len(metadata)!=record.get('metadata_checks')or len(absence)!=record.get('absence_checks')
            or not metadata or not absence or not all(v is True for v in metadata+absence)
            or not all(v.get('raw_bits_equal')is True and v.get('finite')is True for v in arrays)
            or set(row['node']for row in comparison['comparisons'])!=set(range(16))):
        raise ValueError('Actual Source AST proof did not pass every node/state gate')
    raw=proof_unit/'dspark-ast-raw.npz'
    if _sha(raw)!=record.get('raw_arrays_sha256'):raise ValueError('Actual Source AST raw array digest changed')
    with np.load(raw,allow_pickle=False)as data:
        if data['candidate_logits'].shape[:2]!=(1,16)or data['candidate_features'].shape!=(1,16,15360):
            raise ValueError('Actual Source AST raw target geometry differs')
        for branch in range(3):
            for depth in range(6):
                node=0 if depth==0 else 1+branch*5+depth-1
                left=data['candidate_logits'][:,node:node+1];right=data[f'ordinary_logits_b{branch}_d{depth}']
                if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():
                    raise ValueError('Actual Source AST raw target logits differ')
        if requested_law==HYBRID_LAW:
            if (data['hybrid_miss_candidate_logits'].shape[:2]!=(1,16)
                    or data['hybrid_miss_candidate_features'].shape!=(1,16,15360)):
                raise ValueError('Actual hybrid missing-cohort raw geometry differs')
            for branch in range(3):
                for depth in range(6):
                    node=0 if depth==0 else 1+branch*5+depth-1
                    left=data['hybrid_miss_candidate_logits'][:,node:node+1]
                    right=data[f'hybrid_miss_ordinary_logits_b{branch}_d{depth}']
                    if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():
                        raise ValueError('Actual hybrid missing-cohort raw logits differ')
            left=data['hybrid_miss_committed_continuation_logits'];right=data['hybrid_miss_ordinary_continuation_logits']
            if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():
                raise ValueError('Actual hybrid missing-cohort raw continuation differs')
        left=data['committed_continuation_logits'];right=data['ordinary_continuation_logits']
        if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():
            raise ValueError('Actual Source AST raw committed continuation differs')
    source=proof_unit/'source'
    identities=list(implementation_identity())
    root=Path(__file__).resolve().parents[2]
    names=['scripts/research/ds41_ast_target_components.py','src/mlx2/adapters/deepseek_v41_dspark.py',
           'src/mlx2/runtime/drafters/deepseek_v41_dspark.py','src/mlx2/adapters/deepseek_v41_dspark_ast.py',
           'src/mlx2/adapters/deepseek_v41_source.py']
    identities += [(name,_sha(root/name))for name in names]
    for name,digest in identities:
        if not(source/name).is_file()or _sha(source/name)!=digest:
            raise ValueError('Actual Source AST proof source changed: '+name)
    return dict(record=record,proof_unit=str(proof_unit),receipt_sha256=_sha(proof_unit/'dspark-target-comparison.json'),
                raw_sha256=_sha(raw),source_identity=dict(identities))


def run(adapter,unit,guard,*,proof_unit,prompt_ids=None,max_tokens=8,forward_law='reference',expert_law=LAW):
    """Run one real generic AST tile, then ordinary continuation to completion.

    Returns evidence, writes ast-serving-oracle.json for same-profile loopback
    serving, and leaves the adapter's ordinary route intact. No rate claims.
    """
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_dspark_ast import SourceDSparkASTBinding
    unit=Path(unit);output=unit/'dspark-ast-serving.json'
    if output.exists()or(unit/'ast-serving-oracle.json').exists():
        raise ValueError('Refusing to overwrite Source AST serving evidence')
    if not callable(guard)or guard()is not True or mx.default_device()!=mx.gpu:
        raise ValueError('Source AST serving probe requires caller-owned GPU guard')
    if type(max_tokens)is not int or not 8<=max_tokens<=16:raise ValueError('Bounded AST serving probe needs8..16 outputs')
    if (expert_law not in (LAW,RESIDENT_LAW,HYBRID_LAW) or
            (expert_law in (RESIDENT_LAW,HYBRID_LAW) and forward_law!='fully_batched_m1_v1')):
        raise ValueError('Source AST serving proof expert/forward law differs')
    if forward_law not in('reference','fully_batched_m1_v1'):
        raise ValueError('Source AST serving proof supports reference or explicit exact batched target')
    ids=list(prompt_ids)if prompt_ids is not None else list(adapter.tokenizer.encode('Reply with exactly the word Hello.',add_special_tokens=False))
    if not ids or len(ids)>128:raise ValueError('Bounded AST serving prompt must contain1..128 tokens')
    record=dict(schema='mlx2.ds41-dspark-ast-generic-serving.v1',stage='ordinary-reference',passed=False,
                qualified=False,performance_claim=False,prompt_tokens=len(ids),max_tokens=max_tokens,
                forward_law=forward_law,expert_law=expert_law)
    _atomic(output,record)
    binding=batch=None;cache=None;trace_context=None;provider_trace=None
    actual=[];receipts=[];handoff=False;counts={};rounds=[]
    try:
        # The same prompt and ordinary arithmetic is the output oracle. Cold
        # caches avoid fabricating APC publication during this direct probe.
        cache=adapter.model.make_cache();expected=[]
        for step in range(max_tokens):
            guard();tokens=ids if step==0 else [expected[-1]]
            logits=adapter.model(mx.array([tokens],mx.uint32),cache,logits_to_keep=0)
            expected.append(int(mx.argmax(logits[0,-1]).item()))
        for item in cache:item.close()
        cache=None
        # Attribution control with the generic external route's actual
        # prefill partition: all but the last prompt token are body-only;
        # the last token supplies the initial anchor. Preserve the original
        # whole-prompt reference and its strict output gate as separate data.
        cache=adapter.model.make_cache();serving_expected=[]
        if len(ids)>1:
            guard();features=adapter.model.prefill_body(mx.array([ids[:-1]],mx.uint32),cache,(37,38,39))
            mx.eval(features,[item.state for item in cache])
        for step in range(max_tokens):
            guard();token=ids[-1]if step==0 else serving_expected[-1]
            logits,_=adapter.model.forward_with_taps(mx.array([[token]],mx.uint32),cache,(37,38,39))
            serving_expected.append(int(mx.argmax(logits[0,-1]).item()))
        for item in cache:item.close()
        cache=None
        record.update(whole_prompt_reference_token_ids=list(expected),serving_prefill_reference_token_ids=serving_expected,
                      reference_prefill_partitions=dict(whole=[len(ids)],generic=([len(ids)-1,1]if len(ids)>1 else[1])))
        whole_expected=list(expected)
        expected=serving_expected
        record.update(ordinary_reference_profile='external_body_Nminus1_then_B1_anchor_and_decode',
                      whole_prompt_control_exact=(whole_expected==expected),
                      whole_prompt_control_first_divergence=next((dict(index=i,whole=a,matched=b)
                          for i,(a,b)in enumerate(zip(whole_expected,expected))if a!=b),None),
                      whole_prompt_control_scope='partition_drift_attribution_only; AST oracle uses matched serving prefill')
        _atomic(output,record)
        binding=SourceDSparkASTBinding(adapter,check_live=guard,expert_law=expert_law,forward_law=forward_law)
        proof=validate_proof(proof_unit,adapter.model,binding.provider)
        if proof['record']['head']['head_fingerprint']!=binding.backbone.record['fingerprint']:
            raise ValueError('Actual Source DSpark head fingerprint differs')
        oracle=binding.oracle_template()
        # Profile's target algorithms must also be byte-identical to proof.
        frozen=Path(proof_unit)/'source'
        for name,digest in binding.target_binding['source_hashes'].items():
            if not(frozen/name).is_file()or _sha(frozen/name)!=digest:
                raise ValueError('AST scheduler/source profile changed: '+name)
        oracle.update(checks=dict.fromkeys(oracle['checks'],True),receipt_sha256=proof['receipt_sha256'],
                      numerical_classification='raw_bits_exact')
        oracle_path=unit/'ast-serving-oracle.json';_atomic(oracle_path,oracle)
        binding.oracle_path=oracle_path
        binding.provider.with_ast_oracle(oracle_path,artifact=binding.target_artifact,capture_layers=(37,38,39),target_binding=binding.target_binding)
        import importlib.util
        trace_spec=importlib.util.spec_from_file_location('ds41_ast_owned_trace',Path(__file__).with_name('ds41_ast_provider_trace.py'))
        trace_module=importlib.util.module_from_spec(trace_spec);trace_spec.loader.exec_module(trace_module)
        trace_context=trace_module.instrument(binding.provider,guard)
        provider_trace=trace_context.__enter__()
        record['provider_trace']=provider_trace
        batch=binding.create_external_batch(prefill_step_size=128,stop_tokens=())
        batch.insert([ids],max_tokens=[max_tokens],sampling_configs=[{'sampling_temp':0}])
        record.update(stage='generic-ast-request',checkpoint=proof['record']['checkpoint'],source_revision=proof['record']['source_revision'],
            proof=proof,oracle_path=str(oracle_path.resolve()),ordinary_token_ids=expected,binding=binding.receipt())
        _atomic(output,record)
        for _ in range(64):
            before=[dict(uid=lane.uid,anchor=lane.anchor,generated=lane.generated,history=list(lane.history),
                         cache_offsets=[item.offset for item in lane.cache],tail_rows=int(lane.tail.shape[1]),
                         ordinary=lane.ordinary)for lane in batch.lanes.values()]
            guard();_,responses=batch.next()
            for response in responses:
                actual.append(int(response.token))
                if response.speculative_receipt is not None:receipts.append(response.speculative_receipt)
            rounds.append(dict(before=before,emitted=[int(response.token)for response in responses],
                after=[dict(uid=lane.uid,anchor=lane.anchor,generated=lane.generated,history=list(lane.history),
                            cache_offsets=[item.offset for item in lane.cache],tail_rows=int(lane.tail.shape[1]),
                            ordinary=lane.ordinary,accepted=lane.accepted,proposed=lane.proposed)
                       for lane in batch.lanes.values()],provider_launches=binding.provider.launches))
            record.update(ast_token_ids=list(actual),speculative_receipts=receipts,
                          rounds=rounds,provider=binding.provider.receipt(),acceptance=counts)
            _atomic(output,record)
            if binding.provider.launches and not handoff:
                if binding.provider.launches!=1:raise ValueError('Probe exceeded one16row target tile')
                for uid,lane in batch.lanes.items():
                    counts=dict(accepted=lane.accepted,proposed=lane.proposed,external_rounds=lane.external_rounds,
                                verify_accept_span_hist=dict(lane.verify_accept_span_hist))
                    batch.disable_speculation(uid)
                handoff=True
            if not batch.lanes:break
        else:raise RuntimeError('Source AST generic request stalled')
        tiles=[r.get('ast_tree_tiles')for r in receipts if isinstance(r,dict)and isinstance(r.get('ast_tree_tiles'),dict)]
        from itertools import zip_longest
        divergence=next((dict(index=i,ordinary=a,ast=b)for i,(a,b)in enumerate(zip_longest(expected,actual))if a!=b),None)
        record.update(ast_token_ids=actual,speculative_receipts=receipts,ast_tree_tiles=tiles,
                      provider=binding.provider.receipt(),acceptance=counts,rounds=rounds,
                      first_divergence=divergence,one_tile_then_explicit_ordinary_continuation=handoff,
                      serving_prefill_reference_exact=(actual==serving_expected))
        _atomic(output,record)
        if actual!=expected:raise ValueError('Source AST generic output differs from ordinary target')
        if binding.provider.launches!=1 or not any(t.get('observed_used')is True and t.get('target_rows')==16 for t in tiles):
            raise ValueError('Source AST request did not observe one16row target tile')
        record.update(stage='completed',passed=True,ordinary_token_ids=expected,ast_token_ids=actual,
            token_ids_exact=True,provider=binding.provider.receipt(),ast_tree_tiles=tiles,acceptance=counts,
            speculative_receipts=receipts,one_tile_then_explicit_ordinary_continuation=True,
            selected=True,observed_used=True,quality_verdict='one short ordinary-equivalent greedy request; broader draft acceptance quality untested')
        _atomic(output,record);return record
    except BaseException as error:
        record.update(ast_token_ids=list(actual),speculative_receipts=receipts,rounds=rounds,acceptance=counts)
        if binding is not None:record['provider']=binding.provider.receipt()
        record.update(stage='failed',passed=False,error=dict(type=type(error).__name__,message=str(error)))
        _atomic(output,record);raise
    finally:
        if batch is not None:batch.close()
        if trace_context is not None:
            trace_context.__exit__(None,None,None)
            record['provider_trace']=provider_trace
            _atomic(output,record)
        if binding is not None:binding.close()
        if cache is not None:
            for item in cache:item.close()
