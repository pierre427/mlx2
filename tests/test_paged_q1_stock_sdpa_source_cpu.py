"""Run directly: source/fault contracts, no MLX runtime import or GPU device."""
import ast
from contextlib import nullcontext
import importlib.abc
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class NoMetal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("source-only contract prohibits MLX import")
        return None


if __name__ != "__main__":
    raise unittest.SkipTest("run directly: source contract blocks all MLX imports")
sys.meta_path.insert(0, NoMetal())
ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, values, dtype=None, *, shape=None):
        self.values = list(values)
        self.shape = shape or (len(self.values),)
        self.dtype = dtype or "int32"
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)
    def reshape(self, *shape):
        return Tensor(self.values, self.dtype, shape=shape)
    def __getitem__(self, index):
        if index == (None, slice(None)):
            return Tensor(self.values, self.dtype, shape=(1, len(self.values)))
        if index == (slice(None), None):
            return Tensor(self.values, self.dtype, shape=(len(self.values), 1))
        raise AssertionError(index)
    def __lt__(self, other):
        assert self.shape == (1, 128) and other.shape == (2, 1)
        return Tensor([value < length for length in other.values for value in self.values],
                      "bool", shape=(2, 128))


class Backend:
    pass


class Packed:
    def mark_submitted(self):
        assert self.state == "prepared"
        self.state = "submitted"
        self.events.append("submit")
    def abort_before_submit(self):
        assert self.state == "prepared"
        self.state = "closed"
        self.events.append("abort")


class StockSDPASourceContracts(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.backend = Backend()
        self.backend._arena = object()
        self.backend.stream = object()
        self.backend.stock_sdpa_graph_calls = 0
        self.backend.defer_staged_q1_eval = True
        self.backend._deferred_q1_read_roots = []
        self.backend.deferred_q1_read_roots = 0
        self.backend.staged_read_async_evals = 0
        def span(upper, retained=0, window=None):
            lower = retained if window is None else max(retained, upper-window)
            return SimpleNamespace(row_count=1, query_start=upper-1, kv_end=upper,
                retained_start=retained, first_block=retained//64, table_begin=0,
                table_count=1, window=window, visible_bounds=lambda _: (lower, upper))
        self.spans = (span(33), span(97))
        self.plan = SimpleNamespace(dtype="float16", total_rows=2, query_heads=16,
            kv_heads=8, head_dim=128, spans=self.spans,
            page_table=(SimpleNamespace(page_id=2), SimpleNamespace(page_id=3)))
        self.packed = Packed()
        self.packed.events = self.events
        self.packed.state = "prepared"
        self.packed.plan = self.plan
        self.packed.lease = SimpleNamespace(epoch=31)
        self.packed.writer = SimpleNamespace(backend=self.backend, poisoned=False)
        self.query = Tensor([], "float16", shape=(2,16,128))
        self.dependency = Tensor([1], "uint8")
        def gather(*args):
            self.events.append("gather_graph")
            self.assertEqual(args[7], 31)
            self.assertIs(args[2], self.dependency)
            return (Tensor([], "float16", shape=(2,8,128,128)),)*2
        self.native = SimpleNamespace(gather_q1_fp16=gather,
            attention_read_fp16=lambda *_args: self.events.append("ordinary_read") or self.query)
        def sdpa(q, k, v, *, scale, mask):
            self.events.append("sdpa_graph")
            self.assertEqual(q.shape, (2,16,1,128))
            self.assertEqual(k.shape, (2,8,128,128))
            self.assertEqual(mask.shape, (2,1,1,128))
            self.assertEqual([sum(mask.values[:128]), sum(mask.values[128:])], [33,97])
            return self.query.reshape(2,16,1,128)
        self.mx = SimpleNamespace(array=Tensor, float16="float16", uint8="uint8",
            arange=lambda n: Tensor(range(n)), stream=lambda _: nullcontext(),
            fast=SimpleNamespace(scaled_dot_product_attention=sdpa),
            async_eval=lambda value: self.events.append("async"))
        tree = ast.parse((ROOT / "src/mlx2/runtime/paged_attention_native.py").read_text())
        names = {"_stock_q1_sdpa", "native_paged_attention_read_fp16"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        class RemoveRuntimeImports(ast.NodeTransformer):
            def visit_Import(self, node):
                assert all(n.name in ("mlx.core", "_paged_kv_native") for n in node.names)
                return ast.Pass()
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],level=0),*nodes],type_ignores=[])
        module = ast.fix_missing_locations(RemoveRuntimeImports().visit(module))
        ns = {"math":math,"os":os,"native":self.native,"mx":self.mx,
              "PackedTokenRead":Packed,"NativeWriteBackend":Backend,
              "_validate_pinned_read_handles":lambda packed: None,
              "NativePagedReadSubmissionError":RuntimeError}
        exec(compile(module, "source_contract", "exec"),ns)
        self.read = ns["native_paged_attention_read_fp16"]
    def run_read(self):
        with patch.dict(os.environ,{"MLX2_PAGED_Q1_STOCK_SDPA":"1"}):
            return self.read(self.packed,self.backend,self.query,self.dependency,permit_candidate=True)
    def test_owned_gather_sdpa_graph_precedes_submit_and_retained_root(self):
        result=self.run_read()
        self.assertEqual(self.events,["gather_graph","sdpa_graph","submit"])
        self.assertEqual(self.backend._deferred_q1_read_roots,[result])
        self.assertEqual(self.backend.stock_sdpa_graph_calls,1)
        self.assertEqual(self.packed.state,"submitted")
    def test_sdpa_construction_failure_aborts_prepared_lease(self):
        self.mx.fast.scaled_dot_product_attention=lambda *_args,**_kwargs: (_ for _ in ()).throw(ValueError("construction"))
        with self.assertRaisesRegex(ValueError,"construction"): self.run_read()
        self.assertEqual(self.events,["gather_graph","abort"])
        self.assertEqual(self.packed.state,"closed")
        self.assertEqual(self.backend._deferred_q1_read_roots,[])
    def test_async_failure_retains_submitted_lease(self):
        self.backend.defer_staged_q1_eval=False
        self.mx.async_eval=lambda _: (_ for _ in ()).throw(RuntimeError("ambiguous"))
        with self.assertRaisesRegex(RuntimeError,"ambiguous"): self.run_read()
        self.assertEqual(self.packed.state,"submitted")
        self.assertNotIn("abort",self.events)
    def test_default_off_uses_existing_native_reader(self):
        with patch.dict(os.environ,{"MLX2_PAGED_Q1_STOCK_SDPA":"0"}):
            self.read(self.packed,self.backend,self.query,self.dependency,permit_candidate=True)
        self.assertEqual(self.events,["ordinary_read","submit"])
        self.assertEqual(self.backend.stock_sdpa_graph_calls,0)
    def test_visible_window_mask_uses_relative_gather_interval(self):
        self.spans[0].visible_bounds=lambda _: (67,100)
        self.spans[1].visible_bounds=lambda _: (31,128)
        self.run_read()
    def test_bounds_reject_before_gather_and_abort_prepared(self):
        self.spans[1].visible_bounds=lambda _: (0,129)
        with self.assertRaises(ValueError): self.run_read()
        self.assertEqual(self.events,["abort"])
    def test_native_gather_compiles_and_registers_one_exact_terminal(self):
        source=(ROOT/"native/paged_kv/arena.cpp").read_text()
        shader=re.search(r'constexpr const char\* kQ1GatherSource = R"metal\((.*?)\)metal";',source,re.S).group(1)
        with tempfile.TemporaryDirectory(prefix="q1-gather-source-") as folder:
            file=Path(folder)/"gather.metal";file.write_text(shader)
            subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal3.2","-c",str(file),"-o",str(file.with_suffix('.air'))],check=True,capture_output=True)
        primitive=source[source.index("class Q1GatherPrimitive"):source.index("class AttentionReadPrimitive")]
        self.assertIn("for (auto& output : outputs) output.set_data(mx::allocator::malloc(output.nbytes()));",primitive)
        self.assertEqual(primitive.count("addCompletedHandler"),1)
        self.assertIn("owner->read_completed(epoch",primitive)
        self.assertLess(primitive.index("addCompletedHandler"),primitive.index("encoder.dispatch_threads"))
        self.assertGreater(primitive.index("record_q1_gather_dispatch"),primitive.index("dispatched->store(true"))
    def test_runtime_import_is_blocked(self):
        with self.assertRaises(RuntimeError): __import__("mlx.core")


if __name__ == "__main__":
    unittest.main()
