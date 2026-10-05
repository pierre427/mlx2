"""Direct source execution of read dependency and ambiguous-failure contracts.

Run directly. No MLX or native runtime imports; the explicit mlx module below
is a host-only array/graph double used by the extracted production method.
"""
import ast
from contextlib import nullcontext
import importlib.abc
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

if __name__ != '__main__':raise unittest.SkipTest('run directly without runtime imports')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':
            raise RuntimeError('runtime import prohibited')
sys.meta_path.insert(0,NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
class Array:
    dtype='uint8';ndim=1
    def __init__(self,value=1,size=1):self.value=value;self.size=size
class Ticket:
    def __init__(self,dependency,grouped):self.dependency=dependency;self.grouped_q1=grouped
class Read:
    def __init__(self,writer):self.writer=writer;self.plan=NS(dtype='float16',spans=(1,2));self.lease=NS(epoch=17);self.state='prepared'
    def abort_before_submit(self):self.state='closed'

class Contracts(unittest.TestCase):
    def setUp(self):
        self.graph=[];self.received=[];self.fail_after_submit=False
        self.mx=ModuleType('mlx.core');self.mx.array=Array;self.mx.uint8='uint8';self.mx.stream=lambda _:nullcontext()
        self.mx.ones=lambda *a,**kw:self.node('ones',Array())
        self.mx.depends=lambda value,dependencies:self.node('depends',value)
        self.mx.contiguous=lambda value,**kw:self.node('contiguous',value)
        mlx=ModuleType('mlx');mlx.__path__=[];mlx.core=self.mx
        self.modules=patch.dict(sys.modules,{'mlx':mlx,'mlx.core':self.mx});self.modules.start();self.addCleanup(self.modules.stop)
        tree=ast.parse((ROOT/'src/mlx2/runtime/qwen3_paged_native_backend.py').read_text())
        method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='read_staged')
        ns={'PackedTokenRead':Read,'WriteTicket':Ticket,'time':time,'native_paged_attention_read_fp16':self.native_read}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<read_staged>','exec'),ns);self.read=ns['read_staged']
        self.writer=NS(backend=NS(storage_dtype='float16',stream=object()),poisoned=False)
        self.backend=NS(writer=self.writer,_ready=lambda:None,direct_grouped_fence=True,profiling_enabled=False,
            read_submissions=0,staged_read_spans=[],_orphaned_reads={},_failed=False)
        self.use=Read(self.writer)
    def node(self,name,value):self.graph.append(name);return value
    def native_read(self,use,arena,queries,fence,**kw):
        self.received.append(fence);use.state='submitted'
        if self.fail_after_submit:raise RuntimeError('ambiguous read')
        return Array(fence.value)
    def call(self,tickets):return self.read(self.backend,self.use,object(),tickets,scale=.0625)
    def test_exact_success_and_failure_zero_bytes_are_passed_without_new_graph(self):
        for value in (1,0):
            with self.subTest(value=value):
                self.use=Read(self.writer);fence=Array(value)
                result=self.call((Ticket(fence,True),))
                self.assertIs(self.received[-1],fence);self.assertEqual(result.value,value)
                self.assertEqual(self.graph,[]);self.assertEqual(self.use.state,'submitted')
        self.assertEqual(self.backend.direct_grouped_fence_reads,2)
    def test_singleton_head_local_and_cow_dependencies_keep_joined_chain(self):
        for tickets in ((Ticket(Array(),False),),(Ticket(Array(),True),Ticket(Array(),False))):
            self.use=Read(self.writer);self.graph=[];self.call(tickets)
            self.assertEqual(self.graph,['ones','depends','contiguous'])
        self.assertEqual(getattr(self.backend,'direct_grouped_fence_reads',0),0)
    def test_malformed_exact_fence_aborts_only_prepared_read(self):
        with self.assertRaisesRegex(ValueError,'one native byte'):self.call((Ticket(Array(size=2),True),))
        self.assertEqual(self.use.state,'closed');self.assertTrue(self.writer.poisoned)
        self.assertEqual(self.received,[]);self.assertEqual(self.backend._orphaned_reads,{})
    def test_ambiguous_submitted_read_keeps_original_use_and_poison(self):
        self.fail_after_submit=True;fence=Array()
        with self.assertRaisesRegex(RuntimeError,'ambiguous read'):self.call((Ticket(fence,True),))
        self.assertEqual(self.use.state,'submitted');self.assertIs(self.backend._orphaned_reads[17],self.use)
        self.assertIs(self.received[0],fence);self.assertTrue(self.writer.poisoned);self.assertEqual(self.backend.read_submissions,1)

unittest.main()
