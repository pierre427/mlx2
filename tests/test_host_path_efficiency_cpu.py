"""Execute real host control flow without importing a tensor framework."""

import ast
import inspect
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.runtime.call_contracts import KeywordSupportCache
from mlx2.runtime.worker_snapshot import publish_worker_snapshot

ROOT = Path(__file__).resolve().parents[1]


def external_class():
    tree = ast.parse((ROOT / "src/mlx2/runtime/external_speculative.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ExternalDraftBatchGenerator"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_propose", "_draft_accepts_processors"}
    ]

    class Row:
        def __init__(self, tokens, laws, width, features=None):
            self.tokens, self.laws, self.width = tokens, laws, width

    class Unavailable(Exception):
        pass

    ns = {
        "__package__": "mlx2.runtime",
        "HostDraftRow": Row,
        "TreeDraftRow": type("TreeRow", (Row,), {}),
        "CompactDraftRow": type("CompactRow", (), {}),
        "DraftUnavailable": Unavailable,
        "_bump": lambda d, k, v=1: d.update({k: d[k] + v}),
    }
    shell = ast.ClassDef(
        name="Executor", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    exec(  # noqa: S102 - owned AST executed with host doubles
        compile(
            ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])),
            "<external host methods>",
            "exec",
        ),
        ns,
    )
    return ns["Executor"]


class Draft:
    def __init__(self):
        self.calls = []

    def batch_caches(self, caches):
        return caches

    def draft_distributions(self, *args, **kwargs):
        self.calls.append(kwargs)
        return [[7]], [[object()]]

    def propose_tree(self, *args, **kwargs):
        self.calls.append(kwargs)
        return [([7], [object()])]


def setup_executor(kind, processors):
    batch = external_class()()
    batch.draft = Draft()
    batch.draft_topology = "tree15" if kind == "tree" else "chain"
    batch.num_draft = 1
    batch._tree_node_budget = lambda: 1
    batch._chain_draft_cap = lambda: None
    batch.pair_context_tokens = False
    batch.pairwise_selection = "batched" if kind == "pairwise" else "off"
    batch.adaptive_policy = None
    batch.continuation_policy = None
    batch.scheduler_stats = Counter(
        draft_max_width=0, external_tree_node_budget_histogram={}
    )
    batch.mx = SimpleNamespace(concatenate=lambda rows, axis: rows)
    batch._tree_forbidden = lambda lane: ()
    batch._adopt_prelaunched = lambda *args: None
    batch._tree_options = lambda **kwargs: {}
    batch._tree_clock = None
    batch._propose_pairwise = lambda *args: ([[7]], [[object()]])
    lane = SimpleNamespace(
        uid=1,
        maximum=10,
        generated=0,
        ordinary=False,
        tail=SimpleNamespace(shape=(1, 2), dtype="fake"),
        processors=processors,
        anchor=2,
        draft_cache=None,
        rng=None,
        sampling={},
        history=[1, 2],
        cancelled=False,
    )
    return batch, [lane]


@pytest.mark.parametrize(
    "kind,processors",
    [
        ("chain", []),
        ("tree", [object()]),
        ("pairwise", [SimpleNamespace(forbidden_token_ids_at_length=lambda n: ())]),
    ],
)
def test_unused_paths_never_inspect_provider_signature(monkeypatch, kind, processors):
    batch, lanes = setup_executor(kind, processors)

    def forbidden(*args, **kwargs):
        raise AssertionError("unused signature inspection")

    monkeypatch.setattr(inspect, "signature", forbidden)
    assert batch._propose(lanes)[0].tokens == [7]


def test_legacy_provider_is_inspected_once_and_receives_current_histories(monkeypatch):
    batch, lanes = setup_executor("chain", [object()])
    original = inspect.signature
    calls = []

    def inspect_once(method):
        calls.append(method)
        return original(method)

    monkeypatch.setattr(inspect, "signature", inspect_once)
    batch._propose(lanes)
    lanes[0].history.append(3)
    batch._propose(lanes)
    assert len(calls) == 1
    assert batch.draft.calls[0]["processor_histories"] == [[1, 2]]
    assert batch.draft.calls[1]["processor_histories"] == [[1, 2, 3]]


def test_explicit_capability_does_not_inspect(monkeypatch):
    batch, lanes = setup_executor("chain", [object()])
    batch.draft.supports_logits_processors = True
    monkeypatch.setattr(
        inspect, "signature", lambda *args: pytest.fail("declared provider inspected")
    )
    batch._propose(lanes)
    assert batch.draft.calls[0]["logits_processors"] == [lanes[0].processors]


def test_callable_replacement_invalidates_support():
    cache = KeywordSupportCache("logits_processors")

    def supports(*args, **kwargs):
        pass

    def rejects(*args):
        pass

    assert cache.accepts(supports)
    assert not cache.accepts(rejects)
    assert cache.inspections == 2


def test_receiver_replacement_and_signature_override_invalidate():
    cache = KeywordSupportCache("logits_processors")
    one, two = Draft(), Draft()
    assert cache.accepts(one.draft_distributions)
    assert cache.accepts(two.draft_distributions)
    assert cache.inspections == 2

    def call(*args):
        pass

    assert not cache.accepts(call)
    call.__signature__ = inspect.Signature(
        [inspect.Parameter("logits_processors", inspect.Parameter.KEYWORD_ONLY)]
    )
    assert cache.accepts(call)


def test_positional_only_parameter_is_not_a_keyword_contract():
    def positional(logits_processors, /):
        pass

    assert not KeywordSupportCache("logits_processors").accepts(positional)


def test_uninspectable_provider_requires_declaration(monkeypatch):
    cache = KeywordSupportCache("logits_processors")

    def fails(*args):
        raise ValueError("opaque callable")

    monkeypatch.setattr(inspect, "signature", fails)
    call = object()
    assert not cache.accepts(call)
    assert not cache.accepts(call)
    assert cache.inspections == 1
    assert cache.accepts(call, declared=True)


def test_snapshot_collection_keeps_admission_lock_available_and_old_view_coherent():
    lock = threading.Lock()
    snapshot = {"a": 0, "b": 0, "unrelated": 4}
    collecting, finish = threading.Event(), threading.Event()

    def collect():
        collecting.set()
        assert finish.wait(2)
        return {"a": 1, "b": 1}

    worker = threading.Thread(
        target=publish_worker_snapshot, args=(lock, snapshot, collect)
    )
    worker.start()
    try:
        assert collecting.wait(2)
        assert lock.acquire(timeout=1)
        try:
            assert snapshot == {"a": 0, "b": 0, "unrelated": 4}
        finally:
            lock.release()
    finally:
        finish.set()
        worker.join(2)
    assert not worker.is_alive()
    assert snapshot["a"] == snapshot["b"] == 1
    assert snapshot["unrelated"] == 4
    assert snapshot["host_snapshot_timing"]["collections"] == 1


def test_failed_snapshot_collection_preserves_previous_generation():
    lock, snapshot = threading.Lock(), {"generation": 1}

    def collect():
        raise ValueError("collector failed")

    with pytest.raises(ValueError, match="collector failed"):
        publish_worker_snapshot(lock, snapshot, collect)
    assert snapshot == {"generation": 1}
    assert not lock.locked()


def test_snapshot_host_timings_separate_collection_wait_and_publication():
    lock, snapshot = threading.Lock(), {}
    clock = iter([100, 150, 170, 180]).__next__
    publish_worker_snapshot(lock, snapshot, lambda: {"value": 2}, clock=clock)
    timing = snapshot["host_snapshot_timing"]
    assert timing["collection_ns"] == 50
    assert timing["publication_wait_ns"] == 20
    assert timing["publication_hold_ns"] == 10


def test_real_serving_snapshot_collectors_run_outside_admission_lock():
    source = (ROOT / "src/mlx2/serving.py").read_text()
    tree = ast.parse(source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "now - last_snapshot > 1"
    )
    lock = threading.Lock()
    calls = []

    def checked(name, value):
        assert not lock.locked(), name
        calls.append(name)
        return value

    class APC:
        @property
        def apc_stats(self):
            return checked("apc", {"hits": 1})

    self = SimpleNamespace(
        lock=lock,
        snapshot={"old": True},
        _host_memory_status=lambda: checked("host", {}),
    )
    capsules = SimpleNamespace(counters={"bytes": 4})
    ns = {
        "__package__": "mlx2",
        "now": 3,
        "last_snapshot": 0,
        "active": [],
        "self": self,
        "apc": APC(),
        "capsule_pool": capsules,
        "batch": SimpleNamespace(scheduler_stats={"rounds": 1}),
        "segmented_self_mtp_stats": lambda: checked("mtp", {}),
        "mx": SimpleNamespace(
            get_active_memory=lambda: checked("active", 0),
            get_peak_memory=lambda: checked("peak", 0),
        ),
        "physical_footprint_bytes": lambda: checked("footprint", 0),
        "_execution_diagnostics": lambda adapter: checked("adapter", {}),
        "adapter": object(),
        "admission": {},
        "deferred": [],
        "execution_headroom": lambda: checked("headroom", 1),
        "spomin_manager": None,
        "_loop_trace": SimpleNamespace(enabled=lambda: False),
    }
    exec(  # noqa: S102 - owned AST executed with host doubles
        compile(
            ast.Module(body=[node], type_ignores=[]),
            "<serving snapshot>",
            "exec",
        ),
        ns,
    )
    assert len(calls) == 8
    capsules.counters["bytes"] = 9
    assert self.snapshot["cache_capsules"] == {"bytes": 4}
    assert self.snapshot["old"] is True
    assert self.snapshot["host_snapshot_timing"]["collections"] == 1
