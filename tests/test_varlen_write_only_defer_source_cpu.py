"""Direct CPU fault contracts. Extract actual methods; never import MLX/native."""
import ast
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

if __name__ != '__main__':
    raise unittest.SkipTest('run directly: no MLX runtime imports')
ROOT=Path(__file__).resolve().parents[1]

def load(path, names, ns, *, cls=None):
    tree=ast.parse((ROOT/path).read_text())
    body=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else tree.body
    selected=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    class RemoveRuntimeImports(ast.NodeTransformer):
        def visit_Import(self,node): return None
        def visit_ImportFrom(self,node): return None
    selected=[RemoveRuntimeImports().visit(n) for n in selected]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*( [ast.ClassDef(name=cls,bases=[],keywords=[],body=selected,decorator_list=[])] if cls else selected)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(ROOT/path),'exec'),ns)
    return ns[cls] if cls else ns

class Tensor:
    def __init__(self,shape,dtype):
        self.shape=shape; self.dtype=dtype; self.ndim=len(shape); self.size=math.prod(shape)
class Packed:
    def mark_submitted(self): self.state='submitted'; self.events.append('submitted')
    def abort_before_submit(self): self.state='closed'; self.events.append('aborted')

class WriteOnlyContracts(unittest.TestCase):
    def setUp(self):
        self.events=[]; self.async_error=False; self.eval_error=False; self.construct_error=False
        def async_eval(root):
            self.events.append(('async',root))
            if self.async_error: raise RuntimeError('ambiguous async')
        def eval_(*roots):
            self.events.append(('eval',roots))
            if self.eval_error: raise RuntimeError('ambiguous eval')
        self.mx=SimpleNamespace(array=Tensor,float16='float16',uint8='uint8',async_eval=async_eval,eval=eval_)
        ns={}
        Backend=load('src/mlx2/runtime/paged_kv_write.py',{'begin_deferred_q1','end_deferred_q1','deferred_q1_roots','grouped_q1_write'},ns,cls='NativeWriteBackend')
        self.arena=Backend(); a=self.arena
        a._closed=False; a.defer_staged_q1_eval=False; a.defer_staged_q1_writes=False
        a._deferred_q1_write_roots=[]; a._deferred_q1_read_roots=[]; a._arena=object(); a.stream=object(); a._mx=self.mx
        a.grouped_write_async_evals=0; a.staged_read_async_evals=0; a.deferred_q1_write_roots=0; a.deferred_q1_read_roots=0; a.deferred_q1_failure_flushes=0
        self.write_root=object(); self.read_root=object()
        a._native=SimpleNamespace(grouped_q1_write=lambda *args,**kwargs:self.write_root)
        def construct(*args):
            if self.construct_error: raise RuntimeError('construction failed')
            return self.read_root
        self.writer=SimpleNamespace(backend=a,poisoned=False,pending_epochs={},ledger=SimpleNamespace(pending_count=0))
        self.writer.poll_completions=lambda **kwargs:self.writer.pending_epochs.clear()
        self.ns={'NativeWriteBackend':Backend,'PackedTokenRead':Packed,'NativePagedReadSubmissionError':RuntimeError,'_validate_pinned_read_handles':lambda p:None,'native':SimpleNamespace(attention_read_fp16=construct),'mx':self.mx,'os':SimpleNamespace(environ={}), 'math':math}
        load('src/mlx2/runtime/paged_attention_native.py',{'native_paged_attention_read_fp16'},self.ns)
        span=SimpleNamespace(row_count=1,query_start=32,kv_end=33,retained_start=0,first_block=0,table_begin=0,table_count=1,window=None)
        self.use=Packed(); self.use.state='prepared'; self.use.events=[]; self.use.writer=self.writer; self.use.lease=SimpleNamespace(epoch=27)
        self.use.plan=SimpleNamespace(dtype='float16',head_dim=128,total_rows=2,query_heads=4,kv_heads=2,spans=(span,span),page_table=(SimpleNamespace(page_id=0),))
        backend_ns={'mx':self.mx,'time':time,'poll_native_paged_read_events':lambda a:((27,True),),'wait_native_paged_read_events':lambda *a:(), 'complete_staged_read_after_event':lambda u,e:setattr(u,'state','closed')}
        B=load('src/mlx2/runtime/qwen3_paged_native_backend.py',{'abort_deferred_q1','retain_deferred_q1_roots'},backend_ns,cls='NativeQwen3PagedBackend')
        self.owner=B(); self.owner.writer=self.writer; self.owner.timeout_s=.01; self.owner._failed=False; self.owner._orphaned_reads={}; self.owner.terminal_successes=0
    def write(self):
        return self.arena.grouped_q1_write(None,None,(0,1),(0,0),2,128,1)
    def read(self):
        return self.ns['native_paged_attention_read_fp16'](self.use,self.arena,Tensor((2,4,128),'float16'),Tensor((1,),'uint8'),permit_candidate=True)
    def test_write_only_keeps_read_eager_and_roots_until_terminal(self):
        self.arena.begin_deferred_q1(writes_only=True); self.write(); self.read()
        self.assertEqual(self.events,[('async',self.read_root)])
        self.assertEqual(self.arena.deferred_q1_roots(),(self.write_root,self.read_root))
        self.assertTrue(self.owner.retain_deferred_q1_roots((self.use,)))
        self.use.state='closed'; self.assertFalse(self.owner.retain_deferred_q1_roots((self.use,)))
        self.arena.end_deferred_q1(); self.assertEqual(self.arena.deferred_q1_roots(),())
        self.assertEqual((self.arena.grouped_write_async_evals,self.arena.staged_read_async_evals),(0,1))
    def test_read_async_ambiguity_retains_both_roots_after_writes_complete(self):
        self.arena.begin_deferred_q1(writes_only=True); self.write(); self.async_error=True
        with self.assertRaisesRegex(RuntimeError,'submission ambiguous'): self.read()
        self.assertEqual(self.use.state,'submitted'); self.eval_error=True
        with self.assertRaisesRegex(RuntimeError,'ambiguous eval'): self.owner.abort_deferred_q1((),(self.use,))
        self.assertEqual(self.owner._orphaned_reads,{27:self.use}); self.assertTrue(self.writer.poisoned)
        self.arena.end_deferred_q1(retain_roots=self.owner.retain_deferred_q1_roots((self.use,)))
        self.assertEqual(self.arena.deferred_q1_roots(),(self.write_root,self.read_root))
        with self.assertRaisesRegex(RuntimeError,'not idle'): self.arena.begin_deferred_q1(writes_only=True)
    def test_prepared_read_aborts_before_flush_and_never_becomes_submitted(self):
        self.arena.begin_deferred_q1(writes_only=True); self.write()
        def eval_(*roots):
            self.assertEqual(self.use.state,'closed'); self.events.append(('eval',roots))
        self.mx.eval=eval_; self.owner.abort_deferred_q1((),(self.use,))
        self.assertEqual(self.events,[('eval',(self.write_root,))]); self.assertEqual(self.use.events,['aborted'])
        self.assertEqual(self.arena.deferred_q1_failure_flushes,1)
    def test_construction_failure_aborts_prepared_lease_without_read_root(self):
        self.arena.begin_deferred_q1(writes_only=True); self.write(); self.construct_error=True
        with self.assertRaisesRegex(RuntimeError,'construction failed'): self.read()
        self.assertEqual(self.use.state,'closed'); self.assertEqual(self.arena.deferred_q1_roots(),(self.write_root,)); self.assertEqual(self.events,[])
    def test_failure_flush_drains_submitted_read_then_clears_roots(self):
        self.arena.begin_deferred_q1(writes_only=True); self.write(); self.read()
        self.owner.abort_deferred_q1((),(self.use,)); self.assertEqual(self.use.state,'closed'); self.assertEqual(self.owner.terminal_successes,1)
        self.assertFalse(self.owner.retain_deferred_q1_roots((self.use,))); self.arena.end_deferred_q1()
        self.assertEqual(self.arena.deferred_q1_roots(),()); self.assertTrue(self.writer.poisoned)
    def test_existing_all_defer_and_eager_modes_preserved(self):
        self.arena.begin_deferred_q1(); self.write(); self.read(); self.assertEqual(self.events,[])
        self.arena.end_deferred_q1(); self.write(); self.use.state='prepared'; self.read()
        self.assertEqual(self.events,[('async',self.write_root),('async',self.read_root)])

if __name__=='__main__': unittest.main()
