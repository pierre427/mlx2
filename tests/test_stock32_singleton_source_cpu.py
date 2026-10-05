"""Native checked-admission/header and host ABI tests; no MLX/device imports."""
import ast
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
ROOT=Path(__file__).resolve().parents[1]
ARENA=(ROOT/'native/paged_kv/arena.cpp').read_text()

class SingletonCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(prefix='stock32-singleton-');cls.path=Path(cls.folder.name)
        start=ARENA.index('  const bool stock_requested = use_q1_stock_reduction(')
        end=ARENA.index('  // Preflight the survivor geometry',start)
        checked_source=ARENA[start:end]
        source='''#include "q1_stock_reduction.h"
#include "q1_geometry.h"
#include <array>
#include <vector>
#include <cstdlib>
#include <iostream>
int main(int argc,char** argv) {
 using namespace mlx2::paged_kv;
 if(argc!=8)return 3;
 setenv("MLX2_PAGED_Q1_STOCK_REDUCTION",argv[1],1);
 setenv("MLX2_PAGED_Q1_STOCK_SINGLETON",argv[2],1);
 setenv("MLX2_PAGED_Q1_SIMD_STRIPES","16",1);
 const auto rows=std::stoul(argv[3]);const auto visible=std::stoul(argv[4]);
 const bool short_q1=visible>=32&&visible<=128;
 const bool q1_simd_tile=short_q1;const bool q1_pair=rows==2;
 const std::array<uint32_t,2> visible_pair{uint32_t(visible),uint32_t(std::stoul(argv[7]))};
 std::vector<std::vector<uint32_t>> spans(rows,std::vector<uint32_t>(8,0));
 spans[0][3]=std::stoul(argv[5]);spans[0][7]=std::stoul(argv[6]);
 struct {unsigned partition_tokens=0;} split_plan;
 try {
'''+checked_source+'''
 (void)requested_stripes;
 std::cout<<stock_reduction;return 0;
 }catch(const std::invalid_argument&){return 2;}
}
'''
        (cls.path/'probe.cpp').write_text(source);cls.probe=cls.path/'probe'
        subprocess.run([shutil.which('c++') or 'c++','-std=c++20','-Wall','-Wextra','-Werror',
            '-I',str(ROOT/'native/paged_kv'),str(cls.path/'probe.cpp'),'-o',str(cls.probe)],check=True,capture_output=True)
    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()
    def invoke(self,stock,singleton,rows,visible,retained=0,window=0,other=96):
        return subprocess.run([str(self.probe),str(stock),str(singleton),str(rows),str(visible),str(retained),str(window),str(other)],capture_output=True,text=True)
    def test_actual_native_admission_B1_boundaries_and_default_off(self):
        for visible,expected in ((31,'0'),(32,'1'),(63,'1'),(64,'1'),(65,'1'),(127,'1'),(128,'1'),(129,'0'),(8192,'0')):
            result=self.invoke(1,1,1,visible);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout,expected)
        self.assertEqual(self.invoke(1,0,1,98).stdout,'0')
        self.assertEqual(self.invoke(0,0,1,98).stdout,'0')
        self.assertEqual(self.invoke(0,1,1,98).returncode,2)
        self.assertEqual(self.invoke(1,'yes',1,98).returncode,2)
    def test_actual_native_prefix_window_and_pair_alignment_stay_checked(self):
        self.assertEqual(self.invoke(1,1,1,98,retained=64).returncode,2)
        self.assertEqual(self.invoke(1,1,1,98,window=64).returncode,2)
        self.assertEqual(self.invoke(1,1,2,33,other=97).stdout,'1')
        self.assertEqual(self.invoke(1,1,2,33,other=98).returncode,2)
    def test_physical_singleton_counter_after_encoding_and_same_source(self):
        self.assertIn('if (outputs[0].shape(0) == 1) arena_->record_q1_stock_singleton_dispatch();',ARENA)
        first=ARENA.index('arena_->record_q1_stock_singleton_dispatch()')
        self.assertGreater(first,ARENA.rfind('encoder.dispatch_threads(',0,first))
        self.assertIn('stock_reduction ? stock_reduction_source(source)',ARENA)
        self.assertIn('q1_stock_singleton_dispatch_count',(ROOT/'native/paged_kv/binding.cpp').read_text())
    def test_profile_counter_ABI_refuses_old_binary_before_arena_allocation(self):
        path=ROOT/'src/mlx2/runtime/paged_hybrid_research_profile.py';tree=ast.parse(path.read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require_hybrid_native_capabilities','validate_hybrid_stock_pair')]
        ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
        profile={'stock_reduction':True,'stock_singleton':True,'q1_split_partition':0,'q1_simd_stripes':16,'max_tokens':4}
        self.assertTrue(ns['validate_hybrid_stock_pair'](profile,(32,96)))
        old=NS(q1_stock_reduction_dispatch_count=lambda _:0)
        with self.assertRaises(ValueError):ns['require_hybrid_native_capabilities'](profile,old)
        old.q1_stock_singleton_dispatch_count=lambda _:0
        ns['require_hybrid_native_capabilities'](profile,old)
        for value in ('true',1):
            with self.assertRaises(ValueError):ns['validate_hybrid_stock_pair']({**profile,'stock_singleton':value},(32,96))
        source=(ROOT/'src/mlx2/runtime/qwen35_paged_graph_factory.py').read_text()
        self.assertLess(source.index('require_hybrid_native_capabilities(profile, native_extension)'),source.index('native = NativeWriteBackend('))
    def test_closed_and_old_counter_wrapper_and_candidate_prewrite_guards(self):
        path=ROOT/'src/mlx2/runtime/paged_kv_write.py';tree=ast.parse(path.read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='q1_stock_singleton_dispatch_count')
        ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        counter=ns[fn.name];obj=NS(_closed=False,_native=NS(),_arena=object())
        self.assertEqual(counter(obj),0);obj._native.q1_stock_singleton_dispatch_count=lambda _:16;self.assertEqual(counter(obj),16)
        obj._closed=True
        with self.assertRaises(RuntimeError):counter(obj)
        source=(ROOT/'src/mlx2/adapters/qwen35_paged_candidate.py').read_text()
        self.assertLess(source.index('native singleton stock physical counter capability is unavailable'),source.index('for layer_index, layer in enumerate(self.trunk.layers)'))
        self.assertIn('(batch_size == 2 and abs(offsets[0] - offsets[1]) % 32)',source)
        self.assertIn('if batch_size == 1 and not short_q1:',source)
    def test_B1_runner_flagaware_physical_contract(self):
        sys.path.insert(0,str(ROOT/'scripts/research'))
        spec=importlib.util.spec_from_file_location('b1runner',ROOT/'scripts/research/varlen_hybrid_b1_numeric_gate.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        receipt={'packed_lanes':1,'native_tile_dispatches':16,'native_stock_reduction_dispatches':16,
            'q1_simd_stripes':32,'grouped_q1_writes':0,'scalar_native_writes':64,'expected_scalar_write_spans':64,
            'native_split_partial_dispatches':0,'native_split_reduce_dispatches':0}
        module.validate_physical(receipt,1,16,True)
        with self.assertRaises(RuntimeError):module.validate_physical(receipt,1,16,False)

if __name__=='__main__':unittest.main()
