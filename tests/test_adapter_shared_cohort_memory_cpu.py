"""Generic shared memory preflight, real serving helper, and old ABI counters."""
import ast,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from threading import Event
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mlx2.runtime.paged_cohort_memory import validate_shared_cohort_bound,token_digest
from mlx2.runtime.paged_pack_price import IDENTITY_FIELDS

def bound(requests):
    identity={k:'a'*64 for k in IDENTITY_FIELDS};identity.update(host='CPU',hardware='CPU',source_commit='b'*40,mlx_wheel_version='pinned')
    return dict(schema='mlx2.adapter-shared-cohort-memory.v1',route='adapter_native',source_identity=identity,
        source_input_ids=tuple(r[0] for r in requests),token_ids_sha256=tuple(token_digest(r[1]) for r in requests),output_caps=tuple(r[2] for r in requests),
        runtime_bytes=1000,components=dict(arena=500,stage=500),loaded_parameter_bytes=100,process_headroom_bytes=100,process_bound_bytes=1200,max_process_bytes=1300,profile_id='source_bound',qualified=False)

def helper(name):
    tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    env={'__package__':'mlx2'};exec(compile(ast.Module(body=[node],type_ignores=[]),'serving shared admission','exec'),env);return env[name]
class Tests(unittest.TestCase):
    def jobs(self,width=20):
        return tuple(NS(tenant_id='one',request=dict(batch_cohort=dict(id='closed',size=width),skip_writing_prefix_cache=True,native_research_input_id=str(i)),cancelled=Event(),preempted=False,native_b2_cached_tokens=0,admission_tokens=[i]*1025,native_b2_prompt=(),effective_max_tokens=192,native_cohort_memory_receipt=None) for i in range(width))
    def test_adapter_called_once_for_full_twenty_and_receipt_retained_by_every_member(self):
        jobs=self.jobs();calls=[];adapter=NS(native_cohort_memory_admission=lambda requests,**kw:(calls.append(requests) or bound(requests)))
        admission=helper('adapter_shared_cohort_admission');kwargs=dict(profile_path='p',manifest_path='m',mlx_wheel_path='w')
        receipt=admission(adapter,jobs,**kwargs);self.assertEqual(len(calls[0]),20);self.assertEqual(len(calls),1)
        jobs[0].native_b2_prompt=tuple(jobs[0].admission_tokens);jobs[0].admission_tokens=None
        self.assertIs(admission(adapter,jobs,**kwargs),receipt);self.assertEqual(len(calls),1)
        self.assertTrue(all(j.native_cohort_memory_receipt is receipt for j in jobs))
        jobs[19].effective_max_tokens=191
        with self.assertRaises(ValueError):admission(adapter,jobs,**kwargs)
        self.assertEqual(len(calls),1)
    def test_incomplete_cancelled_warm_or_missing_hook_refuse_before_adapter(self):
        calls=[];adapter=NS(native_cohort_memory_admission=lambda *a,**kw:calls.append(1));admission=helper('adapter_shared_cohort_admission');kwargs=dict(profile_path='p',manifest_path='m',mlx_wheel_path='w')
        for mutate in (lambda j:j[0].cancelled.set(),lambda j:setattr(j[0],'native_b2_cached_tokens',1),lambda j:j[0].request['batch_cohort'].update(size=19)):
            jobs=self.jobs();mutate(jobs)
            with self.assertRaises(ValueError):admission(adapter,jobs,**kwargs)
        with self.assertRaises(ValueError):admission(NS(),self.jobs(),**kwargs)
        self.assertEqual(calls,[])
    def test_wrong_component_sum_model_bytes_headroom_or_token_caps_fail_closed(self):
        requests=(('one',(1,)*1025,192),);valid=bound(requests)
        self.assertIs(validate_shared_cohort_bound(valid,requests),valid)
        for mutate in (lambda b:b.update(runtime_bytes=True),lambda b:b.update(components=dict(stage=900)),lambda b:b.update(process_bound_bytes=1100),lambda b:b.update(max_process_bytes=1199),lambda b:b.update(output_caps=(20,))):
            b=bound(requests);mutate(b)
            with self.assertRaises((ValueError,MemoryError)):validate_shared_cohort_bound(b,requests)
    def test_source_loop_requires_collective_headroom_before_skipping_ordinary_grants(self):
        tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'attaching_cohort is not None' in ast.unparse(n.test) and 'complete adapter shared cohort does not fit' in ast.unparse(n))
        jobs=self.jobs(2)
        for j in jobs:j.request['paged_native_packed_n20_research']=True
        receipt=bound(tuple((j.request['native_research_input_id'],tuple(j.admission_tokens),j.effective_max_tokens) for j in jobs));seen=[]
        env=dict(attaching_cohort=NS(jobs=jobs),prompt_lookup=False,self=NS(mtp=False),adapter=object(),adapter_shared_cohort_admission=lambda *a,**kw:receipt,os=NS(environ={}),controller=NS(hard_reserve_gib=2),admission_headroom=lambda:999,reclaim_allocator=lambda:None,evict_unused_checkpoint=lambda:False,apc=NS(),settle_footprint=None,Overloaded=RuntimeError,ensure_admission_headroom=lambda required,**kw:(seen.append(required) or False))
        with self.assertRaisesRegex(RuntimeError,'complete adapter shared'):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual collective headroom gate','exec'),env)
        self.assertEqual(seen,[1000+(2<<30)])
    def test_failed_collective_headroom_clears_cached_receipt_and_retry_rechecks(self):
        tree=ast.parse((ROOT/'src/mlx2/serving.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'attaching_cohort is not None' in ast.unparse(n.test) and 'complete adapter shared cohort does not fit' in ast.unparse(n))
        jobs=self.jobs(2)
        for j in jobs:j.request['paged_native_packed_n20_research']=True
        calls=[];costs=[];adapter=NS(native_cohort_memory_admission=lambda requests,**kw:(costs.append(1) or bound(requests)))
        admission=helper('adapter_shared_cohort_admission')
        env=dict(attaching_cohort=NS(jobs=jobs),prompt_lookup=False,self=NS(mtp=False),adapter=adapter,adapter_shared_cohort_admission=admission,os=NS(environ={'MLX2_NATIVE_PACKED_PREFILL_N20_PROFILE':'p','MLX2_NATIVE_PAGED_MANIFEST':'m','MLX2_NATIVE_PAGED_MLX_WHEEL':'w'}),controller=NS(hard_reserve_gib=2),admission_headroom=lambda:999,reclaim_allocator=lambda:None,evict_unused_checkpoint=lambda:False,apc=NS(),settle_footprint=None,Overloaded=RuntimeError)
        def gate(required,**kw):calls.append(required);return len(calls)==3
        env['ensure_admission_headroom']=gate
        for _ in range(2):
            with self.assertRaises(RuntimeError):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual retry gate','exec'),env)
            self.assertTrue(all(j.native_cohort_memory_receipt is None for j in jobs))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual accepted collective gate','exec'),env)
        self.assertEqual(len(calls),3);self.assertEqual(len(costs),3);self.assertTrue(all(j.native_cohort_memory_receipt is not None for j in jobs))
    def test_old_native_counter_abi_returns_zero_but_closed_arena_raises(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_kv_write.py').read_text());klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NativeWriteBackend')
        methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ('grouped_n20_write_count','grouped_n20_row_count','prefill_long_n20_dispatch_count','q1_stock_long_n20_partial_dispatch_count','q1_stock_long_n20_reduce_dispatch_count','q1_scalar_dispatch_count','q1_stock_long_n20_singleton_partial_dispatch_count','q1_stock_long_n20_singleton_reduce_dispatch_count')]
        arena=NS(_closed=False,_native=NS(),_arena=object())
        for method in methods:
            env={};exec(compile(ast.Module(body=[method],type_ignores=[]),'actual optional raw counter','exec'),env);fn=env[method.name]
            self.assertEqual(fn(arena),0);arena._closed=True
            with self.assertRaises(RuntimeError):fn(arena)
            arena._closed=False
if __name__=='__main__':unittest.main()
