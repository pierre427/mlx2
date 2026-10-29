# SPDX-License-Identifier: MIT
"""One already-owned direct DS4.1 advanced option unit; no ownership acquisition.

Usage: python ds41_advanced_worker.py SOURCE UNIT MODEL [ACTION]
The outer controller supplies frozen source/spec/policy, CPG and paired locks.
"""
import json
import os
from pathlib import Path
import sys
import time

ACTIONS=('canonical_resources','resident_gather','resident_prefill','dspark_prepare','dspark_ast',
         'dspark_native_batched','dspark_descendant','dspark_ordinary_m16','dspark_resident_exact','dspark_hybrid_exact','candidate_pruning')


def dspark_action_policy(action):
    """Complete minimal explicit load maps; never mutate a loaded policy."""
    if not action.startswith('dspark_')or action not in ACTIONS:raise ValueError('Unknown DSpark action')
    policy=dict(deepseek_source_expert_math='routed_fused_fp32',deepseek_source_bf16_loads='woa_f32',
        deepseek_kv_storage='reference',deepseek_candidate_score='reference',deepseek_ced='reference',
        deepseek_vision='reference',deepseek_physical_batch='b1',deepseek_source_ffn_input='reference',
        deepseek_source_resident_gather='reference',deepseek_source_prefill_partition='adaptive_v1')
    if action not in ('dspark_native_batched','dspark_ordinary_m16'):
        policy.update(deepseek_source_dspark_prepare='oracle_v1',deepseek_source_dspark_target_forward='fully_batched_m1_v1',
            deepseek_source_dspark_target_expert_law={'dspark_resident_exact':'source_ast_resident_rows_v1',
                'dspark_hybrid_exact':'source_ast_hybrid_rows_v1'}.get(action,'source_ast_m1_rows_v1'))
    if action in ('dspark_resident_exact','dspark_hybrid_exact'):
        policy['deepseek_source_prefill_partition']='canonical_m1_v1'
    return policy


AST_MIN_PREFIX_TOKENS=258
AST_MAX_PREFIX_TOKENS=384


def ast_fixture_plan(prefix_tokens):
    if type(prefix_tokens)is not int or not AST_MIN_PREFIX_TOKENS<=prefix_tokens<=AST_MAX_PREFIX_TOKENS:
        raise ValueError('Exact DSpark index fixture requires258..384 prefix tokens: ratio2 columns must exceed128')
    return dict(prefix_tokens=prefix_tokens,minimum_index_columns=prefix_tokens//2,
        index_kernel_minimum_columns=129,index_scoring_preserved=True,target_tile_rows=16,
        maximum_prompt_tokens=AST_MAX_PREFIX_TOKENS,classification='Admitted long-prefix fixture; no index bypass or kernel geometry change')


def ast_fixture_request(model,requested=AST_MIN_PREFIX_TOKENS):
    """Render a genuine bounded chat prefix and validate it before GPU load."""
    ast_fixture_plan(requested)
    from tokenizers import Tokenizer
    from mlx2.runtime.models.deepseek_v41_encoding import encode_messages
    tokenizer=Tokenizer.from_file(str(Path(model)/'tokenizer.json'))
    sentence='The reference archive contains stable records and verified revision identifiers. '
    for count in range(1,128):
        request={'messages':[{'role':'user','content':sentence*count+'Final instruction: Reply with exactly Hello.'}],'enable_thinking':False}
        rendered=encode_messages(request['messages'],thinking_mode='chat')
        tokens=tokenizer.encode(rendered,add_special_tokens=False).ids
        if len(tokens)>=requested:
            geometry=ast_fixture_plan(len(tokens))
            return dict(request=request,token_ids=tokens,rendered_sha256=__import__('hashlib').sha256(rendered.encode()).hexdigest(),geometry=geometry)
    raise ValueError('Could not construct bounded admitted DSpark prompt')


def resident_prefill_rows(policy,requested=None):
    if requested is None:
        return 12 if policy.get('deepseek_source_resident_prefill')=='expert_major_v1' else 99
    if type(requested)is not int or not 2<=requested<=128:raise ValueError('Resident PP diagnostic needs2..128 actual rows')
    return requested


def validate_policy(action, policy):
    if action not in ACTIONS:raise ValueError('Unknown advanced action')
    if action=='canonical_resources':
        if policy.get('deepseek_source_prefill_partition')!='canonical_m1_v1':
            raise ValueError('Canonical resource probe needs canonical_m1_v1')
        if policy.get('deepseek_source_expert_math')not in ('epilogue_fused_fp32','routed_fused_fp32'):
            raise ValueError('Actual immutable bank requires packed epilogue_fused_fp32 or routed_fused_fp32 experts')
    if action.startswith('resident_'):
        if policy.get('deepseek_source_resident_gather')!='resident_bank_v1' or policy.get('deepseek_source_expert_math')!='routed_fused_fp32':
            raise ValueError('Resident probes need resident_bank_v1 and routed_fused_fp32')
        if policy.get('deepseek_source_bf16_loads')not in ('woa_f32','direct_f32'):
            raise ValueError('Operand-policy coverage requires admitted deepseek_source_bf16_loads=woa_f32 or direct_f32 at load')
        if action=='resident_prefill' and (policy.get('deepseek_source_resident_prefill') not in ('expert_major_v1','expert_major_wide_v1') or policy.get('deepseek_source_prefill_partition','adaptive_v1')!='adaptive_v1'):
            raise ValueError('Multirow resident prefill needs adaptive_v1 and explicit expert_major policy')
    if action.startswith('dspark_'):
        if policy.get('deepseek_source_expert_math')!='routed_fused_fp32':
            raise ValueError('DSpark probes require routed_fused_fp32 expert boundaries')
        for key,reference in (('deepseek_kv_storage','reference'),('deepseek_source_ffn_input','reference'),
            ('deepseek_source_resident_gather','reference'),('deepseek_physical_batch','b1')):
            if policy.get(key,reference)!=reference:
                raise ValueError('DSpark action requires ordinary reference load prerequisite: '+key)
        if action not in ('dspark_native_batched','dspark_ordinary_m16')and policy.get('deepseek_source_bf16_loads')not in ('woa_f32','direct_f32'):
            raise ValueError('Exact M1 AST grouped WoA requires admitted deepseek_source_bf16_loads=woa_f32 or direct_f32 at load; dense QAT/FP8 PP fusions are optional')
        if action in ('dspark_native_batched','dspark_ordinary_m16'):
            if (policy.get('deepseek_source_dspark_prepare','off')!='off'
                    or policy.get('deepseek_source_dspark_ast','off')!='off'
                    or any(key in policy for key in ('deepseek_source_dspark_target_forward','deepseek_source_dspark_target_expert_law'))):
                raise ValueError('Native/reference diagnostic must load ordinary target; explicit off declarations are accepted but active/oracle target options refuse')
        else:
            if policy.get('deepseek_source_dspark_prepare')!='oracle_v1' or policy.get('deepseek_source_dspark_target_forward')!='fully_batched_m1_v1':
                raise ValueError('Fresh DSpark proof needs oracle_v1 preparation and fully_batched_m1_v1')
            expected={'dspark_resident_exact':'source_ast_resident_rows_v1',
                'dspark_hybrid_exact':'source_ast_hybrid_rows_v1'}.get(action,'source_ast_m1_rows_v1')
            if policy.get('deepseek_source_dspark_target_expert_law')!=expected:
                raise ValueError('Requested DSpark action and loaded exact target law differ')
            if policy.get('deepseek_source_dspark_ast','off')!='off':
                raise ValueError('Fresh proof loads AST off; only actual proof can enable serving')
            if action=='dspark_hybrid_exact' and policy.get('deepseek_source_prefill_partition')!='canonical_m1_v1':
                raise ValueError('Hybrid manager requires canonical_m1_v1 resource/frontier identity')


def persist_unhandled_failure(argv,error):
    """Retain preparation failures even before normal worker receipt creation."""
    if len(argv)<4:return
    source,unit,model=map(Path,argv[1:4])
    if not unit.is_dir():return
    path=unit/'results.json'
    if path.exists():
        record=json.loads(path.read_text())
        if record.get('status')in ('failed','cleanup_failed'):return
    else:
        spec=json.loads((unit/'spec.json').read_text())
        record=dict(schema='mlx2.ds41-advanced-worker.v1',
            action=argv[4]if len(argv)>4 else os.environ.get('MLX2_ADVANCED_ACTION'),
            checkpoint=spec.get('checkpoint'),source_revision=spec.get('source_revision'),
            source_snapshot_sha256=spec.get('source_snapshot_sha256'),
            qualification=False,performance_claim=False,model_loaded=False,numerical_test_executed=False,
            classification='method_failure_before_model_load',phase='policy_or_fixture_preparation')
    record.update(status='failed',passed=False,error={'type':type(error).__name__,'message':str(error)},finished_at=time.time())
    temporary=path.with_name(path.name+'.unhandled.tmp')
    with temporary.open('x')as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def main():
    if len(sys.argv) not in (4,5):raise SystemExit(__doc__)
    source,unit,model=map(Path,sys.argv[1:4])
    action=sys.argv[4] if len(sys.argv)==5 else os.environ.get('MLX2_ADVANCED_ACTION','canonical_resources')
    sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source/'scripts/research'))
    spec=json.loads((unit/'spec.json').read_text());policy=json.loads((unit/'policy.json').read_text())
    validate_policy(action,policy)
    fixture=None
    if action.startswith('dspark_')and action not in ('dspark_native_batched','dspark_ordinary_m16'):
        requested=int(os.environ.get('MLX2_AST_PREFIX_TOKENS',str(AST_MIN_PREFIX_TOKENS)))
        fixture=ast_fixture_request(model,requested)
    from ds41_advanced_options import owned_guard,canonical_prefix,pinned_inventory,resident_probe,prepare_dspark_oracle,expert_io_report,strict_counter_delta,EXPERT_COUNTER_KEYS,native_batched_probe,ordinary_m16_probe
    guard=owned_guard(unit);guard()
    from mlx2.desktop_campaign_worker import atomic
    from probe_deepseek_v4_packed import source_receipt
    record=dict(schema='mlx2.ds41-advanced-worker.v1',action=action,status='starting',passed=False,
        qualification=False,performance_claim=False,checkpoint=spec['checkpoint'],source=source_receipt(),
        source_revision=spec['source_revision'],source_snapshot_sha256=spec['source_snapshot_sha256'],started_at=time.time())
    atomic(unit/'results.json',record)
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.runtime.streamed_load import WeightStreamRequest
    from mlx2.serving import preload_admission_limit_bytes
    mx.set_default_device(mx.gpu)
    load_policy={k:v for k,v in policy.items()if k not in ('moe_expert_streaming','host_memory_signals')}
    # Match the existing owned ordinary-reference load identity exactly.
    experts=dict(load_policy['deepseek_source_experts'])
    experts.pop('atlas_path',None);experts.pop('preheat',None)
    load_policy['deepseek_source_experts']=experts
    adapter=smoke_cache=None
    def save(phase):
        record['phase']=phase;atomic(unit/'results.json',record)
    try:
        save('loading')
        adapter=DeepSeekV41SourceAdapter(model,execution_policy=load_policy,
            weight_streaming=WeightStreamRequest(mode='moe_experts',budget_bytes=experts['budget_bytes'],
                admission_limit_bytes=preload_admission_limit_bytes(),read_workers=2,bypass_page_cache=True))
        guard();save('ordinary-reference')
        request={'messages':[{'role':'user','content':'Reply with exactly Hello.'}],'enable_thinking':False}
        ids=list(adapter.tokenizer.encode(adapter.render_prompt(request),add_special_tokens=False))
        if not 2<=len(ids)<=32:raise ValueError('Advanced Hello prompt must remain bounded2..32 tokens')
        w=adapter.weights;saved=(w.source_resident_gather_policy,w.source_resident_prefill_policy)
        w.source_resident_gather_policy=w.source_resident_prefill_policy='reference'
        smoke_cache=adapter.model.make_cache();output=[]
        try:
            for step in range(16):
                guard();tokens=ids if step==0 else[output[-1]]
                logits=adapter.model(mx.array([tokens],mx.uint32),smoke_cache,logits_to_keep=1)
                mx.eval(logits,[s.state for s in smoke_cache])
                if not bool(mx.all(mx.isfinite(logits)).item()):raise RuntimeError('Nonfinite ordinary smoke logits')
                token=int(mx.argmax(logits[0,-1]).item());output.append(token)
                if token in adapter.tokenizer.eos_token_ids:break
        finally:
            w.source_resident_gather_policy,w.source_resident_prefill_policy=saved
        text=adapter.tokenizer.decode(output,skip_special_tokens=True)
        reference=dict(passed=text.strip()=='Hello.',text=text,output_token_ids=output,
            prompt_tokens=len(ids),checkpoint=spec['checkpoint'],source=record['source'],diagnostics=adapter.diagnostics(),
            numerical_control='Ordinary reference expert consumer under the same loaded model identity')
        atomic(unit/'ordinary-reference.json',reference);record['ordinary_reference']=reference
        for row in smoke_cache:row.close()
        smoke_cache=None
        if not reference['passed']:raise RuntimeError('Ordinary coherent-output smoke failed')
        save('advanced-'+action)
        if action=='canonical_resources':
            with canonical_prefix(adapter,unit,guard,ids) as (lookup,prefix_report):
                record['canonical_prefix']=prefix_report
                keys=tuple(adapter.weight_stream._cache)
                with pinned_inventory(adapter,keys,guard,consumer=False) as bank:
                    before=adapter.weight_stream.counters();strict_counter_delta(before,before,EXPERT_COUNTER_KEYS);bank.assert_current()
                    # Restore resources after exact numerical lookup; existing bank
                    # ownership is preserved, never replaced by advisory hints.
                    from mlx2.runtime.checkpoint_resources import restore_resources
                    restore=restore_resources(adapter,lookup,ids)
                    logits=adapter.model(mx.array([lookup.remaining_tokens],mx.uint32),lookup.cache,logits_to_keep=1)
                    mx.eval(logits,[s.state for s in lookup.cache]);guard();bank.assert_current()
                    after=adapter.weight_stream.counters()
                    evidence=expert_io_report(before,after)
                    finite_checks=[mx.all(mx.isfinite(logits))]+[mx.all(mx.isfinite(plane))for row in lookup.cache for plane in row.state if plane is not None]
                    mx.eval(finite_checks)
                    record['pinned_expert_cache']=dict(bank=bank.receipt(),resource_restore=restore,**evidence,
                        numerical_identity_unchanged=True,finite=all(bool(value.item())for value in finite_checks),
                        scope='Actual immutable bank remains provider-pinned through the followup forward; unseen experts use ordinary demand loading')
                row=record['pinned_expert_cache']
                record['passed']=prefix_report['passed']is True and all(row[key]is True for key in ('telemetry_valid','finite','zero_reads','zero_uploads','zero_misses','zero_evictions'))
        elif action in ('resident_gather','resident_prefill'):
            test_ids=ids
            if action=='resident_prefill':
                requested=os.environ.get('MLX2_ADVANCED_PREFILL_ROWS')
                rows=resident_prefill_rows(load_policy,None if requested is None else int(requested))
                test_ids=json.loads((unit/'request.json').read_text())['token_ids'][:rows]
                if len(test_ids)!=rows:raise ValueError('Resident PP fixture is shorter than requested actual rows')
            record['advanced']=resident_probe(adapter,unit,guard,test_ids,prefill=action=='resident_prefill')
            record['passed']=record['advanced']['passed']
        elif action=='candidate_pruning':
            from ds41_candidate_pruning_probe import run as run_candidate_pruning
            record['advanced']=run_candidate_pruning(adapter,unit,guard)
            record['passed']=record['advanced']['passed']
        else:
            if fixture is not None:
                ids=fixture['token_ids'];record['DSpark_fixture']=fixture
                atomic(unit/'dspark-fixture.json',fixture)
            if action=='dspark_native_batched':record['advanced']=native_batched_probe(adapter,unit,guard)
            elif action=='dspark_descendant':record['advanced']=native_batched_probe(adapter,unit,guard,descendant=True)
            elif action=='dspark_ordinary_m16':record['advanced']=ordinary_m16_probe(adapter,unit,guard,ids)
            else:
                law=load_policy['deepseek_source_dspark_target_expert_law']
                record['advanced']=prepare_dspark_oracle(adapter,unit,guard,ids,expert_law=law)
            record['passed']=record['advanced'].get('passed')is True
        record['status']='completed' if record['passed']else'failed';save(record['status'])
    except BaseException as error:
        record.update(status='failed',passed=False,error={'type':type(error).__name__,'message':str(error)})
        save('failed');raise
    finally:
        if smoke_cache is not None:
            for row in smoke_cache:row.close()
        prior_error=sys.exc_info()[1];cleanup_error=None
        if adapter is not None:
            try:
                adapter.close();record['adapter_closed']=True
            except BaseException as error:
                cleanup_error=error
                record.update(status='cleanup_failed',passed=False,adapter_closed=False,
                    cleanup_error={'type':type(error).__name__,'message':str(error)})
        record['finished_at']=time.time();atomic(unit/'results.json',record)
        if cleanup_error is not None:
            if prior_error is not None:prior_error.add_note('Advanced adapter cleanup failed: '+repr(cleanup_error))
            else:raise cleanup_error


if __name__=='__main__':
    try:main()
    except BaseException as error:
        try:persist_unhandled_failure(sys.argv,error)
        except BaseException as receipt_error:error.add_note('Failed to retain worker failure receipt: '+repr(receipt_error))
        raise
