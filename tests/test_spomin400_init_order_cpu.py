"""Profile pinning and worker failure wake proofs; never import MLX/native."""
import ast
import importlib.abc
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native' or (name.startswith('mlx2.runtime.models.') and name!='mlx2.runtime.models.import_env'):
            raise RuntimeError('tensor/model runtime import forbidden: '+name)
sys.meta_path.insert(0,Guard())
import spomin_400case_phased_http_bench as B
from mlx2.runtime import hybrid_packed_prefill_n as N
class Tests(unittest.TestCase):
    def test_actual_factory_environment_is_safe_before_model_imports(self):
        with patch.dict(os.environ,{},clear=True):
            env=N.environment();os.environ.update(env)
            self.assertEqual(os.environ['MLX_SDPA_BLOCKS'],'0')
            self.assertEqual(os.environ['MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON'],'1')
            # Exact source environment includes all settings model snapshots use.
            from mlx2.runtime.models import import_env
            name='cpu_init_order_probe';sys.modules[name]=object()
            try:
                import_env.snapshot(name)
                import_env.assert_profile_applied('N20 initialization',os.environ)
                os.environ.pop('MLX_SDPA_BLOCKS')
                with self.assertRaises(import_env.ImportOrderError):import_env.assert_profile_applied('N20 initialization',os.environ)
            finally:
                sys.modules.pop(name,None);import_env._SNAPSHOTS.pop(name,None)
    def test_real_init_pins_before_reachable_model_dependency_imports(self):
        source=(ROOT/'scripts/research/spomin_400case_phased_http_bench.py').read_text()
        tree=ast.parse(source);init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='init')
        pin=next(n.lineno for n in ast.walk(init) if isinstance(n,ast.Call) and ast.unparse(n.func)=='os.environ.update' and n.args and ast.unparse(n.args[0])=="profile['required_environment']")
        dependencies=('mlx2.adapters.qwen38_27b','mlx2.runtime.generate','mlx2.server','mlx2')
        imports=[n for n in ast.walk(init) if isinstance(n,ast.ImportFrom) and n.module in dependencies]
        self.assertEqual(len(imports),4)
        self.assertTrue(all(n.lineno>pin for n in imports))
        resources=next(n for n in ast.walk(init) if isinstance(n,ast.ImportFrom) and any(a.name=='qwen35_paged_graph_factory' for a in n.names))
        self.assertGreater(resources.lineno,pin)
        generate=(ROOT/'src/mlx2/runtime/generate.py').read_text()
        # Guards the actual dependency that makes moving generate unsafe.
        self.assertIn('from .models',generate)
    def test_worker_failure_returns_before_ready_timeout(self):
        engine=type('Engine',(),{'error':None,'ready':threading.Event()})()
        def fail():time.sleep(.01);engine.error='ImportOrderError: source numeric profile differs'
        thread=threading.Thread(target=fail);thread.start();begin=time.monotonic()
        with self.assertRaisesRegex(RuntimeError,'ImportOrderError'):B.wait_engine_ready(engine,2)
        thread.join();self.assertLess(time.monotonic()-begin,.3)
    def test_ready_and_timeout_are_distinct(self):
        engine=type('Engine',(),{'error':None,'ready':threading.Event()})();engine.ready.set();B.wait_engine_ready(engine,.01)
        engine.ready.clear()
        with self.assertRaisesRegex(RuntimeError,'timed out'):B.wait_engine_ready(engine,.01)
    def test_failed_init_has_owned_shutdown_and_does_not_start_requests(self):
        source=(ROOT/'scripts/research/spomin_400case_phased_http_bench.py').read_text()
        tree=ast.parse(source);serve=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='serve')
        handlers=[n for n in ast.walk(serve) if isinstance(n,ast.ExceptHandler)]
        initcleanup=next(n for h in handlers for n in ast.walk(h) if isinstance(n,ast.If) and ast.unparse(n.test)=="command.get('action') == 'init'")
        body=ast.unparse(initcleanup)
        self.assertLess(body.index('service.verify(owner)'),body.index('service.shutdown(owner)'))
        self.assertIn('initialization_cleanup_error',body)
        self.assertNotIn('service.start',body)
if __name__=='__main__':unittest.main()
