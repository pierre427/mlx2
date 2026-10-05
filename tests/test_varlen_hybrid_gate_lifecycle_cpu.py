"""Exercise the frozen hybrid gate's actual lifecycle blocks without a GPU.

The gate loads MLX and the 27B model only inside execute(). These tests compile
its bootstrap, prepare, and finalizer AST blocks and inject terminal faults.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/research/varlen_hybrid_q1_gate.py"
TREE = ast.parse(GATE.read_text())
EXECUTE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
MAIN = next(n for n in EXECUTE.body if isinstance(n, ast.Try) and n.finalbody)
BOOTSTRAP = next(n for n in MAIN.body if isinstance(n, ast.For)
                 and isinstance(n.iter, ast.Call)
                 and isinstance(n.iter.func, ast.Name)
                 and n.iter.func.id == "enumerate"
                 and isinstance(n.iter.args[0], ast.Name)
                 and n.iter.args[0].id == "prompts")
STEP = next(n for n in MAIN.body if isinstance(n, ast.For)
            and isinstance(n.iter, ast.Call)
            and isinstance(n.iter.func, ast.Name)
            and n.iter.func.id == "range")
PREPARE_START = next(i for i, n in enumerate(STEP.body)
                     if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "prepared" for t in n.targets))
PREPARE = STEP.body[PREPARE_START:PREPARE_START + 3]


def run_nodes(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=list(nodes), type_ignores=[]))
    exec(compile(module, str(GATE), "exec"), scope)


def gate_module():
    spec = importlib.util.spec_from_file_location("hybrid_gate_cpu_test", GATE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Layer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Writer:
    def __init__(self, *, pending=(), ledger=0, poisoned=False):
        self.pending_epochs = tuple(pending)
        self.ledger = SimpleNamespace(pending_count=ledger)
        self.poisoned = poisoned
        self.failed_arena_torn_down = False

    def poll_completions(self):
        return ()  # An ambiguous epoch never retires from mere polling.

    def teardown_failed_arena(self):
        assert self.poisoned and not self.pending_epochs and not self.ledger.pending_count
        self.failed_arena_torn_down = True


class Backend:
    def __init__(self):
        self._orphaned_reads = {}

    def drain_failed_read_events(self):
        return 0  # A missing callback cannot close an orphaned read.


class Candidate:
    def __init__(self, writer):
        self.writer = writer
        self.released = False

    def release_failure_roots_after_teardown(self):
        assert self.writer.failed_arena_torn_down
        self.released = True


class Adapter:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def finalizer_scope(*, writer, backend=None, layers=(), owners=(), branches=(), prepared=(),
                    reservations=(), budget=None):
    module = gate_module()
    backend = backend or Backend()
    adapter = Adapter()
    candidate = Candidate(writer)
    result = {}
    if budget is None:
        budget = {"charged": 0, "capacity": 0}
    scope = dict(
        writer=writer, backend=backend, candidate=candidate, adapter=adapter,
        owners=list(owners), branches=list(branches), prepared=list(prepared),
        unattached_layers=list(layers), reservations=list(reservations), result=result,
        budget=budget, args=SimpleNamespace(q1_joined_recurrent=False),
        RETAINED_FAILURES=[], signal=SimpleNamespace(alarm=lambda _: None),
        reap_native_request_owner=lambda owner, _writer, _backend: None,
    )
    return scope


def test_failed_bootstrap_import_keeps_unattached_layers_until_terminal():
    made = []

    def make_layer(*_args, **_kwargs):
        layer = Layer()
        made.append(layer)
        return layer

    def fail_after_first_import(*_args, **_kwargs):
        raise RuntimeError("injected import failure after first native submission")

    scope = dict(prompts=((1, 2),), depth=2, writer=object(), profile=object(),
                 PagedKVTokenOwner=make_layer, candidate=SimpleNamespace(
                     bootstrap_ordinary=fail_after_first_import),
                 reserve=lambda _: None, unattached_layers=[], owners=[],
                 result={"bootstrap_receipts": []})
    with pytest.raises(RuntimeError, match="injected import failure"):
        run_nodes((BOOTSTRAP,), scope)
    assert scope["unattached_layers"] == made

    writer = Writer(pending=(7,))
    cleanup = finalizer_scope(writer=writer, layers=made)
    with pytest.raises(RuntimeError, match="unresolved terminal leases"):
        run_nodes(MAIN.finalbody, cleanup)
    assert not any(layer.closed for layer in made)
    assert not cleanup["adapter"].closed
    assert cleanup["RETAINED_FAILURES"]

    writer.pending_epochs = ()  # Only a matched terminal event may do this.
    run_nodes(MAIN.finalbody, cleanup)
    assert all(layer.closed for layer in made)
    assert cleanup["adapter"].closed


def test_second_prepare_failure_rolls_back_first_prepared_generation():
    calls = []

    class Prepared:
        def rollback(self):
            calls.append("prepared-rollback")

        def publish(self):
            calls.append("publish")

    class Branch:
        def __init__(self, fail=False):
            self.fail = fail

        def prepare(self, rows):
            assert rows == 1
            if self.fail:
                raise RuntimeError("injected second prepare failure")
            return Prepared()

        def rollback(self):
            calls.append("branch-rollback")

    branches = [Branch(), Branch(fail=True)]
    scope = dict(branches=branches, prepared=[])
    with pytest.raises(RuntimeError, match="second prepare failure"):
        run_nodes(PREPARE, scope)
    assert len(scope["prepared"]) == 1 and calls == []

    cleanup = finalizer_scope(writer=Writer(), branches=branches,
                              prepared=scope["prepared"])
    run_nodes(MAIN.finalbody, cleanup)
    assert calls == ["prepared-rollback", "branch-rollback", "branch-rollback"]
    assert cleanup["adapter"].closed and not cleanup["RETAINED_FAILURES"]


def test_delayed_read_callback_retains_graph_roots_and_charge():
    writer = Writer(ledger=1, poisoned=True)
    backend = Backend()
    backend._orphaned_reads[17] = object()
    module = gate_module()
    budget = {"charged": 4096, "capacity": 8192}
    reservation = module.RetainedStaging(4096, budget)
    scope = finalizer_scope(writer=writer, backend=backend,
                            reservations=(reservation,), budget=budget)
    with pytest.raises(RuntimeError, match="unresolved terminal leases"):
        run_nodes(MAIN.finalbody, scope)
    assert not writer.failed_arena_torn_down
    assert not scope["candidate"].released
    assert not scope["adapter"].closed
    assert budget["charged"] == 8192

    # A late, matched read terminal closes the shared ledger and orphan map.
    writer.ledger.pending_count = 0
    backend._orphaned_reads.clear()
    run_nodes(MAIN.finalbody, scope)
    assert writer.failed_arena_torn_down
    assert scope["candidate"].released
    assert scope["adapter"].closed
    assert budget["charged"] == 4096
