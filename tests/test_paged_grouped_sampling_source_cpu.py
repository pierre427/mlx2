"""Source-isolated contracts: no MLX import, native extension, or Metal device.

Run directly with Python (tests/conftest.py imports the actual MLX runtime).
The import blocker makes accidental MLX execution fail before device creation.
"""
import ast
from contextlib import contextmanager, ExitStack
import importlib.abc
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class MetalRuntimeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("CPU source contract forbids MLX/Metal runtime import")
        return None


if __name__ != "__main__":
    raise unittest.SkipTest("run directly: source-only contracts prohibit MLX-importing conftest")

sys.meta_path.insert(0, MetalRuntimeBlocker())
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
ROOT = Path(__file__).resolve().parents[1]


def load_source(path, names, namespace, *, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef)
                    and node.name == class_name).body
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    # The source function's relative mx import is replaced with our fake mx.
    class FakeImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            assert node.module == "paged_native_continuation"
            return ast.Pass()
    module = ast.Module(body=nodes, type_ignores=[])
    module = FakeImports().visit(module)
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = ((len(values), len(values[0])) if values and
                      isinstance(values[0], list) else (len(values),))

    def __getitem__(self, index):
        if index is None:
            return Tensor([self.values])
        value = self.values[index]
        return Tensor(value) if isinstance(value, list) else SimpleNamespace(item=lambda: value)

    def astype(self, _dtype):
        return self

    def __sub__(self, _normalizer):
        return self


class FakeMx:
    float32 = "float32"
    uint32 = "uint32"

    def __init__(self, events):
        self.events = events

    def array(self, values, dtype):
        return Tensor(list(values))

    def logsumexp(self, logits, *, axis, keepdims):
        assert axis == -1 and keepdims is True
        return 0

    def eval(self, *roots):
        self.events.append(("eval", len(roots)))


class Matcher:
    _trie = (9,)

    @staticmethod
    def match(state, trie, token):
        return state + (token,), token in trie


class Continuation:
    pass


class SourceSamplingContracts(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.mx = FakeMx(self.events)
        ns = {"Any": object, "mx": self.mx, "GenerationBatch": SimpleNamespace(Response=lambda **kw: SimpleNamespace(**kw)),
              "StopSequenceMatcher": Matcher, "_invalid_output_reason": lambda token, lp: "bad" if token < 0 else None,
              "_NO_GRAPH_LOGITS": object()}
        load_source("src/mlx2/runtime/paged_native_continuation.py",
                    {"_stage_sample", "_response_from_sample", "_next_with_reader"}, ns,
                    class_name="NativeQwen3Continuation")
        for name in ("_stage_sample", "_response_from_sample", "_next_with_reader"):
            setattr(Continuation, name, ns[name])
        self.backend = SimpleNamespace(read_submissions=0, terminal_successes=0, staged_read_spans=[])
        self.candidate = SimpleNamespace(backend=self.backend, model=SimpleNamespace(layers=(0, 1)),
                                        _serving_b2=False)
        self.lanes = tuple(self.make_lane(uid) for uid in (1, 2))
        ns = {"os": os, "ExitStack": ExitStack, "mx": self.mx,
              "NativeQwen3Continuation": Continuation,
              "can_run_research_graph_b2": lambda lanes: not any(lane.closed for lane in lanes),
              "CandidateRequest": lambda *args: args, "PackedLane": lambda *args: args}
        load_source("src/mlx2/runtime/paged_native_graph_group.py", {"run_research_graph_b2"}, ns)
        self.run_group = ns["run_research_graph_b2"]
        def forward(*args, **kw):
            self.backend.read_submissions += 2
            self.backend.terminal_successes += 2
            self.backend.staged_read_spans.extend((2, 2))
            return (Tensor([1, 2]), Tensor([3, 4])), {"packed_lanes": 2}
        self.candidate.forward_staged = forward

    def make_lane(self, uid):
        lane = Continuation()
        lane.uid = uid
        lane.revision = "r1"
        lane.tokens = [7]
        lane._pending_token = 8
        lane._first_logits = None
        lane.candidate = self.candidate
        lane.count = 0
        lane.maximum = 2
        lane.matcher = Matcher()
        lane.matcher_state = ()
        lane.lane_rng = SimpleNamespace(draws=0)
        lane.research_only = True
        lane.apcv2_restored_tokens = 0
        lane.price_provenance = None
        lane.closed = False
        def processor(context, logits):
            self.events.append(("processor", uid, tuple(context.values)))
            return logits
        def sampler(logprobs):
            self.events.append(("sampler", uid))
            lane.lane_rng.draws += 1
            return Tensor([uid])
        lane.processors = [processor]
        lane.sampler = sampler
        @contextmanager
        def snapshot():
            yield SimpleNamespace(revision="r1", offset=1, generation=1, layer_owners=(0, 1))
            self.events.append(("lease_closed", uid))
        class Branch:
            layers = (SimpleNamespace(offset=2), SimpleNamespace(offset=2))
            def prepare(inner, rows):
                self.events.append(("prepare", uid, rows))
                return inner
            def publish(inner):
                self.events.append(("publish", uid))
            def rollback(inner):
                self.events.append(("rollback", uid))
        lane.owner = SimpleNamespace(snapshot=snapshot, begin=lambda request: Branch(),
                                     reap_retired=lambda: self.events.append(("reap", uid)))
        return lane

    def test_default_and_grouped_keep_independent_rng_and_response_order(self):
        with patch.dict(os.environ, {"MLX2_PAGED_GROUPED_SAMPLER": "1"}):
            responses = self.run_group(self.lanes)
        sequence = [event for event in self.events if event[0] in ("processor", "sampler", "eval")]
        self.assertEqual(sequence, [("processor", 1, (7, 8)), ("sampler", 1),
                                    ("processor", 2, (7, 8)), ("sampler", 2), ("eval", 4)])
        self.assertEqual([r.uid for r in responses], [1, 2])
        self.assertEqual([r.token for r in responses], [1, 2])
        self.assertEqual([lane.lane_rng.draws for lane in self.lanes], [1, 1])
        self.assertEqual([lane._pending_token for lane in self.lanes], [1, 2])
        self.assertTrue(all(r.mtp_receipt["grouped_sampler_eval"] for r in responses))
        self.assertEqual(self.candidate._grouped_sampler_evals, 1)
        self.setUp()
        with patch.dict(os.environ, {"MLX2_PAGED_GROUPED_SAMPLER": "0"}):
            responses = self.run_group(self.lanes)
        self.assertEqual([event for event in self.events if event[0] == "eval"], [("eval", 2), ("eval", 2)])
        self.assertEqual([r.token for r in responses], [1, 2])
        self.assertFalse(any(r.mtp_receipt["grouped_sampler_eval"] for r in responses))

    def test_stop_length_and_closed_lane(self):
        self.lanes[0].sampler = lambda lp: Tensor([9])
        self.lanes[1].maximum = 1
        with patch.dict(os.environ, {"MLX2_PAGED_GROUPED_SAMPLER": "1"}):
            responses = self.run_group(self.lanes)
        self.assertEqual([r.finish_reason for r in responses], ["stop", "length"])
        self.assertEqual([lane._pending_token for lane in self.lanes], [None, None])
        self.assertEqual([r.all_tokens for r in responses], [[7, 8], [7, 8]])
        self.assertIs(responses[1].lane_rng, self.lanes[1].lane_rng)
        self.lanes[0].closed = True
        with self.assertRaises(ValueError):
            self.run_group(self.lanes)

    def test_sampler_or_validation_failure_returns_no_cohort(self):
        for fault in ("sampler", "shape", "invalid", "eval"):
            with self.subTest(fault=fault):
                self.setUp()
                if fault == "sampler":
                    self.lanes[1].sampler = lambda lp: (_ for _ in ()).throw(RuntimeError("sampler"))
                elif fault == "shape":
                    self.lanes[1].sampler = lambda lp: Tensor([1, 2])
                elif fault == "invalid":
                    self.lanes[1].sampler = lambda lp: Tensor([-1])
                else:
                    self.mx.eval = lambda *roots: (_ for _ in ()).throw(RuntimeError("eval"))
                with patch.dict(os.environ, {"MLX2_PAGED_GROUPED_SAMPLER": "1"}):
                    with self.assertRaises(RuntimeError):
                        self.run_group(self.lanes)
                self.assertEqual(len([e for e in self.events if e[0] == "rollback"]), 4)
                self.assertFalse(any(e[0] == "reap" for e in self.events))
                self.assertEqual(getattr(self.candidate, "_grouped_sampler_evals", 0), int(fault in ("shape", "invalid")))

    def test_counter_snapshot_never_iterates_read_history(self):
        ns = {}
        load_source("src/mlx2/runtime/qwen3_paged_native_backend.py",
                    {"profile_snapshot", "profile_counters_snapshot"}, ns,
                    class_name="NativeQwen3PagedBackend")
        class History(list):
            def __iter__(self):
                raise AssertionError("timed counters copied detailed read history")
        backend = SimpleNamespace(host_profile_ns={"graph_eval": 5}, read_work=History([{"rows": 2}]),
                                  profiling_enabled=True, writer=SimpleNamespace(
                                      backend=SimpleNamespace(q1_tile_dispatch_count=lambda: 4,
                                                              grouped_q1_write_count=lambda: 4,
                                                              write_dispatch_count=lambda: 0),
                                      ledger=SimpleNamespace(completed_epoch=9)))
        backend.profile_counters_snapshot = lambda: ns["profile_counters_snapshot"](backend)
        snapshot = backend.profile_counters_snapshot()
        self.assertEqual(snapshot["read_count"], 1)
        self.assertNotIn("reads", snapshot)
        snapshot["host_ns"]["graph_eval"] = 100
        self.assertEqual(backend.host_profile_ns["graph_eval"], 5)
        backend.read_work = [{"rows": 2}]
        full = ns["profile_snapshot"](backend)
        full["reads"][0]["rows"] = 50
        self.assertEqual(backend.read_work[0]["rows"], 2)

    def test_runtime_import_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "forbids"):
            __import__("mlx.core")


if __name__ == "__main__":
    unittest.main()
