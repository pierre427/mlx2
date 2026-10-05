"""Run directly; source launch and counter contracts never import MLX."""
import ast
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

if __name__ != '__main__': raise unittest.SkipTest('run directly without runtime imports')
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'mlx' or name.startswith('mlx.') or name == '_paged_kv_native':
            raise RuntimeError('runtime import prohibited')
sys.meta_path.insert(0, NoRuntime())
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('long_gate', ROOT/'scripts/research/varlen_hybrid_q1_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class Contracts(unittest.TestCase):
    def test_deferred_diagnostic_requires_clean_exact_source_and_binary(self):
        args=NS(stock_long=False,q1_deferred=True,source_commit='a'*40,native_sha256='b'*64)
        result={'source_commit':'a'*40}
        gate.validate_stock_long_launch(args,result,{},source_clean=True)
        with self.assertRaisesRegex(ValueError,'clean pinned source'):
            gate.validate_stock_long_launch(args,result,{},source_clean=False)
        args.native_sha256=None
        with self.assertRaisesRegex(ValueError,'clean pinned source'):
            gate.validate_stock_long_launch(args,result,{},source_clean=True)
        args.q1_deferred=1
        with self.assertRaisesRegex(ValueError,'selector must be boolean'):
            gate.validate_stock_long_launch(args,result,{},source_clean=True)
    def test_write_only_deferred_launch_is_exclusive_and_pinned(self):
        args=NS(stock_long=False,q1_deferred=False,q1_deferred_writes_only=True,
                source_commit='a'*40,native_sha256='b'*64)
        result={'source_commit':'a'*40}
        gate.validate_stock_long_launch(args,result,{},source_clean=True)
        with self.assertRaisesRegex(ValueError,'clean pinned source'):
            gate.validate_stock_long_launch(args,result,{},source_clean=False)
        args.q1_deferred=True
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            gate.validate_stock_long_launch(args,result,{},source_clean=True)
    def test_short_direct_fence_diagnostic_requires_frozen_source_and_binary(self):
        args=NS(stock_long=False,q1_direct_grouped_fence=True,source_commit='a'*40,native_sha256='b'*64)
        result={'source_commit':'a'*40}
        gate.validate_stock_long_launch(args,result,{},source_clean=True)
        with self.assertRaises(ValueError):gate.validate_stock_long_launch(args,result,{},source_clean=False)
        args.native_sha256=None
        with self.assertRaises(ValueError):gate.validate_stock_long_launch(args,result,{},source_clean=True)
    def test_joined_diagnostic_requires_clean_pinned_source_and_binary(self):
        args=NS(stock_long=False,q1_joined_recurrent=True,source_commit='a'*40,native_sha256='b'*64)
        result={'source_commit':'a'*40}
        gate.validate_stock_long_launch(args,result,{},source_clean=True)
        with self.assertRaisesRegex(ValueError,'clean pinned source'):
            gate.validate_stock_long_launch(args,result,{},source_clean=False)
        args.source_commit='c'*40
        with self.assertRaisesRegex(ValueError,'clean pinned source'):
            gate.validate_stock_long_launch(args,result,{},source_clean=True)
    def test_frozen_launch_and_override_refusal(self):
        args=NS(stock_long=True,split_partition=0,source_commit='a'*40,native_sha256='b'*64)
        result={'source_commit':'a'*40};env={'MLX2_PAGED_Q1_STOCK_LONG':'1'}
        gate.validate_stock_long_launch(args,result,env,source_clean=True)
        for bad in ({'MLX2_PAGED_Q1_SPLIT_KV':'128'}, {'MLX_SDPA_BLOCKS':'64'}, {'MLX2_PAGED_Q1_STOCK_LONG':'0'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gate.validate_stock_long_launch(args,result,env|bad,source_clean=True)
        with self.assertRaises(ValueError): gate.validate_stock_long_launch(args,result,env,source_clean=False)
        with self.assertRaises(ValueError): gate.validate_stock_long_launch(args,{'source_commit':'c'*40},env,source_clean=True)
        args.native_sha256='unbound'
        with self.assertRaises(ValueError): gate.validate_stock_long_launch(args,result,env,source_clean=True)
    def test_inline_selector_requires_stock_long_clean_pin_and_exact_environment(self):
        args=NS(stock_long=True,stock_long_inline_metadata=True,split_partition=0,
                source_commit='a'*40,native_sha256='b'*64)
        result={'source_commit':'a'*40}
        env={'MLX2_PAGED_Q1_STOCK_LONG':'1','MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA':'1'}
        gate.validate_stock_long_launch(args,result,env,source_clean=True)
        for changed in ({'MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA':'0'},
                        {'MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA':'bad'}):
            with self.assertRaisesRegex(ValueError,'inline metadata environment'):
                gate.validate_stock_long_launch(args,result,env|changed,source_clean=True)
        args.stock_long=False
        with self.assertRaisesRegex(ValueError,'requires explicit stock-long'):
            gate.validate_stock_long_launch(args,result,env,source_clean=True)
        args.stock_long=True
        with self.assertRaisesRegex(ValueError,'clean pinned source'):
            gate.validate_stock_long_launch(args,result,env,source_clean=False)

    def test_physical_counter_wrappers_and_closed_arena(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_kv_write.py').read_text())
        names=('q1_stock_long_partial_dispatch_count','q1_stock_long_reduce_dispatch_count','q1_stock_long_metadata_dispatch_count')
        functions=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
        self.assertEqual(len(functions),3)
        ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),'<counter wrappers>','exec'),ns)
        calls=[]
        def counter(arena): calls.append(arena);return 16
        obj=NS(_closed=False,_arena=object(),_native=NS(**{name:counter for name in names}))
        for name in names:self.assertEqual(ns[name](obj),16)
        self.assertEqual(calls,[obj._arena]*3)
        obj._closed=True
        for name in names:
            with self.assertRaises(RuntimeError): ns[name](obj)
        self.assertEqual(len(calls),3)

    def test_old_native_snapshot_reports_absent_optional_counters_zero(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_kv_write.py').read_text())
        names=('q1_stock_reduction_dispatch_count','q1_stock_long_partial_dispatch_count','q1_stock_long_reduce_dispatch_count','q1_stock_long_metadata_dispatch_count')
        functions=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
        ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),'<wrappers>','exec'),ns)
        arena=NS(_closed=False,_arena=object(),_native=NS())
        for name in names:setattr(arena,name,lambda name=name:ns[name](arena))
        for name in ('q1_split_partial_dispatch_count','q1_split_reduce_dispatch_count','q1_tile_dispatch_count','grouped_q1_write_count','write_dispatch_count'):
            setattr(arena,name,lambda:7)
        tree=ast.parse((ROOT/'src/mlx2/runtime/qwen3_paged_native_backend.py').read_text())
        snapshot=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='profile_counters_snapshot')
        code={};exec(compile(ast.Module(body=[snapshot],type_ignores=[]),'<snapshot>','exec'),code)
        backend=NS(host_profile_ns={},read_work=[],profiling_enabled=True,writer=NS(backend=arena,ledger=NS(completed_epoch=9)))
        result=code['profile_counters_snapshot'](backend)
        for name in ('q1_stock_reduction_dispatches','q1_stock_long_partial_dispatches','q1_stock_long_reduce_dispatches','q1_stock_long_metadata_dispatches'):
            self.assertEqual(result[name],0)
        self.assertEqual(result['native_write_dispatches'],7)
        arena._native=NS(q1_stock_reduction_dispatch_count=lambda _:16)
        self.assertEqual(code['profile_counters_snapshot'](backend)['q1_stock_reduction_dispatches'],16)
        arena._closed=True
        for name in names:
            with self.assertRaises(RuntimeError):getattr(arena,name)()

    def test_requested_stock_capability_checks_use_raw_native_abi(self):
        tree=ast.parse((ROOT/'src/mlx2/runtime/paged_hybrid_research_profile.py').read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='require_hybrid_native_capabilities')
        ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'<admission>','exec'),ns)
        check=ns['require_hybrid_native_capabilities']
        check({'stock_reduction':False},NS())
        with self.assertRaises(ValueError):check({'stock_reduction':True},NS())
        check({'stock_reduction':True},NS(q1_stock_reduction_dispatch_count=lambda _:0))
        tree=ast.parse((ROOT/'scripts/research/varlen_hybrid_q1_gate.py').read_text())
        execute=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='execute')
        guard=next(node for node in execute.body if isinstance(node,ast.If) and 'native stock-long physical-counter ABI unavailable' in ast.unparse(node))
        for native in (NS(), NS(q1_stock_long_partial_dispatch_count=lambda _:0)):
            with self.assertRaises(ValueError):
                exec(compile(ast.Module(body=[guard],type_ignores=[]),'<raw long capability>','exec'),{'args':NS(stock_long=True),'_paged_kv_native':native})

unittest.main()
