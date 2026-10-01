"""CPU-only tests for the sparse parent-restart tree GDN candidate.

Every Metal construction, dispatch and capability query is forbidden; the
default device is the CPU before any tensor exists. Bit parity here is
between host executors sharing one fp32 arithmetic (sparse restart vs the
all-node oracle vs the ordinary per-path chain); it says nothing about Metal
bits, registers, memory or speed.
"""

import dataclasses
import subprocess
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)  # before any tensor

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlx2.adapters import qwen38_tree_gdn as T  # noqa: E402


def _forbid(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run in CPU tests")
    return fail


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(mx.fast, "metal_kernel", _forbid("mx.fast.metal_kernel"))
    monkeypatch.setattr(mx.metal, "is_available", _forbid("mx.metal.is_available"))
    monkeypatch.setattr(mx, "device_info", _forbid("mx.device_info"))
    assert mx.default_device() == mx.cpu
    yield
    assert T._KERNELS == {}


def _inputs(width, *, hk=1, hv=2, dtype=mx.float32, seed=3):
    rng = np.random.default_rng(seed)
    d = T.HEAD_DIM
    q = rng.standard_normal((1, width, hk, d)).astype(np.float32) / np.sqrt(d)
    k = rng.standard_normal((1, width, hk, d)).astype(np.float32) / np.sqrt(d)
    v = rng.standard_normal((1, width, hv, d)).astype(np.float32)
    g = rng.uniform(0.80, 0.999, (1, width, hv)).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, (1, width, hv)).astype(np.float32)
    state = (rng.standard_normal((1, hv, d, d)) * 0.05).astype(np.float32)
    cast = lambda x: mx.array(x).astype(dtype)  # noqa: E731
    return cast(q), cast(k), cast(v), mx.array(g), mx.array(beta), mx.array(state)


def _bits(x):
    return np.array(x).view(np.uint8).tobytes() if x.dtype != mx.bfloat16 else np.array(x.view(mx.uint16)).tobytes()


def _path_to(parents, node):
    path = [node]
    while parents[path[-1]] != -1:
        path.append(parents[path[-1]])
    return path[::-1]


TOPOLOGIES = {
    "single": [-1],
    "chain": [-1, 0, 1, 2, 3, 4],
    "branch": [-1, 0, 0, 1, 1, 2, 5],
    "forest": [-1, 0, -1, 2, 0, -1, 5],
    "fan": [-1] + [0] * 7,
    "binary": [-1] + [(t - 1) // 2 for t in range(1, 15)],
    "deep_restart": [-1, 0, 1, 2, 3, 0, 5, 6, 1, 8],
}


def _random_parents(width, rng):
    parents = [-1]
    for row in range(1, width):
        parents.append(int(rng.integers(-1, row)) if rng.random() < 0.85 else row - 1)
    return parents


RANDOM = {f"random{w}": _random_parents(w, np.random.default_rng(w)) for w in (3, 9, 17, 32)}


# ---------------------------------------------------------------- plan

def test_plan_tables_for_chain_branch_forest():
    chain = T.plan_tree_restarts([-1, 0, 1, 2])
    assert chain.restart == ("entry", "carry", "carry", "carry") and chain.n_slots == 0 and not chain.has_forest
    branch = T.plan_tree_restarts([-1, 0, 0, 1, 1])
    assert branch.restart == ("entry", "carry", "slot", "slot", "slot")
    assert branch.slots == (0, 1) and branch.slot_of == (0, 1, -1, -1, -1)
    assert branch.restart_slot == (-1, -1, 0, 1, 1)
    forest = T.plan_tree_restarts([-1, 0, -1, 2, 0])
    assert forest.has_forest and forest.restart == ("entry", "carry", "entry", "carry", "slot")
    assert forest.slots == (0,)
    assert T.metal_kernel_table(branch) == (-1, 0, 0, 1, 1, -1, -1, 0, 1, 1, 0, 1, -1, -1, -1)


def test_only_nonconsecutive_parents_are_retained():
    for parents in [*TOPOLOGIES.values(), *RANDOM.values()]:
        plan = T.plan_tree_restarts(parents)
        expected = sorted({p for t, p in enumerate(parents) if p >= 0 and p != t - 1})
        assert list(plan.slots) == expected
        assert plan.n_slots <= len(parents)


@pytest.mark.parametrize("parents,message", [
    ([], "width"),
    ([-1] * 33, "width"),
    ("-1,0", "list or tuple"),
    (np.array([-1, 0]), "list or tuple"),
    ((p for p in (-1, 0)), "list or tuple"),
    ([-1, True], "plain int"),
    ([-1, 0.0], "plain int"),
    ([-1.0], "plain int"),
    ([-1, np.int64(0)], "plain int"),
    ([-1, "0"], "plain int"),
    ([-1, None], "plain int"),
    ([0], "must be -1"),
    ([-1, 1], "must be -1"),
    ([-1, 0, 2], "must be -1"),
    ([-1, -2], "must be -1"),
])
def test_malformed_parents_are_refused(parents, message):
    with pytest.raises(T.TreeGDNPlanError, match=message):
        T.plan_tree_restarts(parents)


def test_forged_plans_are_refused():
    plan = T.plan_tree_restarts([-1, 0, 0, 1])
    with pytest.raises(T.TreeGDNPlanError, match="does not match"):
        T.verified_plan(dataclasses.replace(plan, slot_of=(-1, -1, -1, -1)))
    forged = T.plan_tree_restarts([-1, 0, 0, 1])
    object.__setattr__(forged, "restart_slot", (-1, -1, 0, 0))
    with pytest.raises(T.TreeGDNPlanError, match="does not match"):
        T.verified_plan(forged)
    with pytest.raises(T.TreeGDNPlanError, match="not a TreeRestartPlan"):
        T.verified_plan(dataclasses.asdict(plan))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.parents = (-1,)


@pytest.mark.parametrize("path,message", [
    ([], "non-empty"),
    ((), "non-empty"),
    ([1, 2], "root-to-node"),
    ([0, 1, 2], "root-to-node"),
    ([0, 1, 3], "root-to-node"),
    ([0, 9], "not a row"),
    ([True], "not a row"),
    ([0.0], "not a row"),
    ("0", "non-empty list"),
])
def test_bad_accepted_paths_are_refused(path, message):
    plan = T.plan_tree_restarts([-1, 0, 0, 2])
    with pytest.raises(T.TreeGDNPlanError, match=message):
        T.accepted_path_rows(plan, path)


# ---------------------------------------------------------------- geometry

def _swap(index, value):
    def edit(arrays):
        arrays = list(arrays)
        arrays[index] = value(arrays[index])
        return arrays
    return edit


@pytest.mark.parametrize("edit,message", [
    (_swap(0, lambda q: q.astype(mx.bfloat16)), "share"),
    (_swap(2, lambda v: v.astype(mx.float16)), "share"),
    (lambda a: [x.astype(mx.float16) if i < 3 else x for i, x in enumerate(a)], "share"),
    (_swap(3, lambda g: g.astype(mx.bfloat16)), "fp32"),
    (_swap(4, lambda b: b.astype(mx.bfloat16)), "fp32"),
    (_swap(5, lambda s: s.astype(mx.bfloat16)), "per-token state cast"),
    (_swap(3, lambda g: mx.broadcast_to(g[..., None], g.shape + (128,))), "scalar gates"),
    (_swap(0, lambda q: q[..., :64]), "q and k must be"),
    (lambda a: [x[..., :64] if i in (0, 1) else x for i, x in enumerate(a)], "Dk and Dv"),
    (lambda a: [mx.concatenate([x, x]) if i < 3 else x for i, x in enumerate(a)], "B1"),
    (_swap(5, lambda s: s[:, :1]), "state must be"),
    (lambda a: [x[:, :2] if i < 5 else x for i, x in enumerate(a)], "row count"),
    (_swap(0, lambda q: q.tolist()), "mx.array"),
])
def test_out_of_bounds_geometry_is_refused(edit, message):
    arrays = edit(_inputs(3))
    with pytest.raises(T.TreeGDNGeometryError, match=message):
        T.sparse_tree_forward(*arrays, [-1, 0, 0])


def test_head_ratio_must_divide():
    q, k, v, g, beta, state = _inputs(2, hk=2, hv=3)
    with pytest.raises(T.TreeGDNGeometryError, match="Hv % Hk"):
        T.sparse_tree_forward(q, k, v, g, beta, state, [-1, 0])


# ---------------------------------------------------------------- parity

CASES = [(name, parents, dtype, heads) for name, parents in {**TOPOLOGIES, **RANDOM}.items()
         for dtype in (mx.float32, mx.bfloat16) for heads in ((1, 2), (2, 4))]


@pytest.mark.parametrize("name,parents,dtype,heads", CASES, ids=[f"{c[0]}-{c[2]}-{c[3]}" for c in CASES])
def test_sparse_matches_all_node_oracle_and_every_accepted_path(name, parents, dtype, heads):
    hk, hv = heads
    arrays = _inputs(len(parents), hk=hk, hv=hv, dtype=dtype, seed=len(parents) + hv)
    result = T.sparse_tree_forward(*arrays, parents)
    y_full, states = T.diagnostic_full_node_reference(*arrays, parents)
    assert result.y.dtype == dtype and result.y.shape == (1, len(parents), hv, T.HEAD_DIM)
    assert _bits(result.y) == _bits(y_full)
    for node in range(len(parents)):
        path = _path_to(parents, node)
        replayed = T.replay_accepted_path(result, path)
        assert replayed.dtype == mx.float32 and replayed.shape == (1, hv, T.HEAD_DIM, T.HEAD_DIM)
        assert _bits(replayed) == _bits(states[node]), (name, node)
        assert _bits(replayed) == _bits(T.ordinary_path_reference(*arrays, path))


def test_branching_paths_are_nonprefix_and_still_exact():
    parents = TOPOLOGIES["branch"]
    arrays = _inputs(len(parents))
    result = T.sparse_tree_forward(*arrays, parents)
    _, states = T.diagnostic_full_node_reference(*arrays, parents)
    path = _path_to(parents, 4)  # rows 0, 1, 4: not a prefix of row order
    assert path == [0, 1, 4]
    assert _bits(T.replay_accepted_path(result, path)) == _bits(states[4])
    assert _bits(states[4]) != _bits(states[3])  # siblings differ, so restarts matter


def test_a_wrong_restart_would_be_caught():
    """Falsifier: restarting every slot row from the previous row diverges."""
    parents = TOPOLOGIES["branch"]
    arrays = _inputs(len(parents))
    good = T.sparse_tree_forward(*arrays, parents)
    chained = T.sparse_tree_forward(*arrays, list(range(-1, len(parents) - 1)))
    assert _bits(good.y) != _bits(chained.y)


# ---------------------------------------------------------------- retention, receipts, immutability

def test_zero_and_many_retained_slots():
    chain = T.sparse_tree_forward(*_inputs(6), TOPOLOGIES["chain"])
    assert chain.receipt["retained_slots"] == 0 and chain.receipt["live_slot_peak"] == 0
    wide = [-1] + [(t - 1) // 2 for t in range(1, 32)]
    many = T.sparse_tree_forward(*_inputs(32), wide)
    expected = sorted({p for t, p in enumerate(wide) if p >= 0 and p != t - 1})
    assert many.receipt["retained_slot_rows"] == expected and many.receipt["retained_slots"] == len(expected) > 10
    assert many.receipt["entry_retained_for_forest"] is False
    forest = T.sparse_tree_forward(*_inputs(7), TOPOLOGIES["forest"])
    assert forest.receipt["entry_retained_for_forest"] is True


def test_forward_exports_no_node_states_and_mutates_nothing():
    arrays = _inputs(7)
    before = [_bits(x) for x in arrays]
    T.status(reset=True)
    result = T.sparse_tree_forward(*arrays, TOPOLOGIES["branch"])
    assert [_bits(x) for x in arrays] == before
    public = {f.name for f in dataclasses.fields(result) if not f.name.startswith("_")}
    assert public == {"plan", "geometry", "y", "receipt"}
    assert result.receipt["per_node_states_exported"] is False
    with pytest.raises(TypeError):
        result.receipt["device_engaged"] = True
    with pytest.raises(ValueError):
        result._prework.entry[0, 0, 0] = 1.0  # retained snapshots are read-only
    counters = T.status()
    assert counters["host_sparse_forwards"] == 1 and counters["device_launches_issued"] == 0
    assert counters["device_engaged"] is False and result.receipt["device_engaged"] is False
    T.replay_accepted_path(result, [0, 1, 3])
    assert T.status()["host_replays"] == 1 and T.status()["device_engaged"] is False


def test_replay_refuses_copies_and_retargeted_records():
    """Root 2868: a record retargeted to another valid same-width tree, or to
    another geometry, must not replay; neither may copies or unbound records."""
    arrays = _inputs(4)
    result = T.sparse_tree_forward(*arrays, [-1, 0, 0, 1])
    other = T.plan_tree_restarts([-1, 0, 1, 2])            # valid, same width
    attempts = [
        dataclasses.replace(result),                        # copy
        dataclasses.replace(result, plan=other),            # retargeted plan
        dataclasses.replace(result, geometry=dataclasses.replace(result.geometry, hk=2)),
        {"plan": result.plan},
    ]
    for forged in attempts:
        with pytest.raises(T.TreeGDNPlanError, match="exact record issued"):
            T.replay_accepted_path(forged, [0])
    for name, value, message in (
            ("plan", other, "no longer matches its binding"),
            ("geometry", dataclasses.replace(result.geometry, hk=2), "no longer matches its binding"),
            ("geometry", dataclasses.replace(result.geometry, activation=mx.bfloat16), "no longer matches"),
            ("geometry", dataclasses.replace(result.geometry, dk=64), "no longer matches"),   # root 2871
            ("_binding", None, "has no binding"),
            ("_binding", "0" * 64, "no longer matches its binding")):
        record = T.sparse_tree_forward(*arrays, [-1, 0, 0, 1])
        object.__setattr__(record, name, value)              # in-place forgery of an issued record
        with pytest.raises(T.TreeGDNPlanError, match=message):
            T.replay_accepted_path(record, [0])
    record = T.sparse_tree_forward(*arrays, [-1, 0, 0, 1])
    retarget = dataclasses.replace(record, plan=T.plan_tree_restarts([-1, 0, 1, 2]))   # root 2871 shape
    with pytest.raises(T.TreeGDNPlanError):
        T.replay_accepted_path(retarget, [0, 1, 2])
    reshaped = dataclasses.replace(record._prework, q=record._prework.q.reshape(4, 128))  # same bytes, new shape
    object.__setattr__(record, "_prework", reshaped)
    with pytest.raises(T.TreeGDNPlanError, match="prework changed"):
        T.replay_accepted_path(record, [0])
    record = T.sparse_tree_forward(*arrays, [-1, 0, 0, 1])
    object.__setattr__(record.plan, "slot_of", (0, -1, -1, -1))  # same parents, forged table
    with pytest.raises(T.TreeGDNPlanError, match="does not match"):
        T.replay_accepted_path(record, [0])
    assert T.replay_accepted_path(result, [0, 1, 3]).shape == (1, 2, 128, 128)  # the original still replays


def test_arbitrary_prefixes_and_non_ancestors_are_refused():
    parents = TOPOLOGIES["branch"]                          # [-1, 0, 0, 1, 1, 2, 5]
    result = T.sparse_tree_forward(*_inputs(len(parents)), parents)
    for path in ([0, 1, 2], [0, 1, 2, 3], [0, 2, 3], [1, 3], [0, 4]):
        with pytest.raises(T.TreeGDNPlanError, match="root-to-node"):
            T.replay_accepted_path(result, path)


def test_verified_plan_is_pure_and_counts_nothing():
    import threading

    T.status(reset=True)
    plan = T.plan_tree_restarts([-1, 0, 0])
    for _ in range(5):
        T.verified_plan(plan)
    assert T.status()["host_plans"] == 1

    def proposer():
        for _ in range(200):
            T.plan_tree_restarts([-1, 0, 0, 1])

    def verifier():
        for _ in range(200):
            T.verified_plan(plan)

    threads = [threading.Thread(target=f) for f in (proposer, verifier, proposer, verifier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert T.status()["host_plans"] == 1 + 400  # no increments lost, none invented


def test_host_executors_refuse_a_gpu_default_before_touching_tensors(monkeypatch):
    arrays = _inputs(3)
    result = T.sparse_tree_forward(*arrays, [-1, 0, 0])
    monkeypatch.setattr(T, "_host", _forbid("host capture"))
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    for call in (lambda: T.sparse_tree_forward(*arrays, [-1, 0, 0]),
                 lambda: T.diagnostic_full_node_reference(*arrays, [-1, 0, 0]),
                 lambda: T.ordinary_path_reference(*arrays, [0, 2]),
                 lambda: T.replay_accepted_path(result, [0, 2])):
        with pytest.raises(T.TreeGDNUnavailable, match="default_device"):
            call()


def test_lazy_or_mocked_issuance_never_counts_as_engagement(monkeypatch):
    """Root 2867: an issued (lazy) or substituted launch is not engagement,
    even if its flag reads 1 and even under a forged GPU default device."""
    T.status(reset=True)
    launches = []

    def fake_kernel(**kwargs):
        launches.append(kwargs)
        return [mx.zeros(kwargs["output_shapes"][0], dtype=kwargs["output_dtypes"][0]),
                mx.ones((1,), dtype=mx.uint32)]

    monkeypatch.setattr(T, "_admit_metal", lambda allow: None)
    monkeypatch.setattr(T, "_metal_kernel", lambda name: fake_kernel)
    arrays = _inputs(3)
    y, engaged = T.metal_sparse_tree_forward(*arrays, [-1, 0, 0], allow_unqualified_metal=True)
    state, flag = T.metal_replay_accepted_path(*arrays[1:], [-1, 0, 0], [0, 2], allow_unqualified_metal=True)
    assert len(launches) == 2 and T.status()["device_launches_issued"] == 2
    assert launches[0]["template"][-2:] == [("KSLOTS", 1), ("FOREST", 0)]
    assert launches[0]["output_shapes"] == [(1, 3, 2, 128), (1,)] and launches[0]["init_value"] == 0
    assert T.status()["device_engaged"] is False
    assert T.confirm_device_engagement(engaged) is False            # CPU default device
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert T.confirm_device_engagement(engaged) is False            # mocked kernel never registered
    assert T.confirm_device_engagement(mx.ones((1,), dtype=mx.uint32)) is False  # foreign flag
    assert T.status()["device_engagement_confirmed"] == 0 and T.status()["device_engaged"] is False


def test_pending_flags_are_weak_and_never_own_the_graph(monkeypatch):
    """Root 2872: an issued flag the caller drops leaves the pending table."""
    import gc

    def fake_kernel(**kwargs):
        return [mx.zeros(kwargs["output_shapes"][0], dtype=kwargs["output_dtypes"][0]),
                mx.ones((1,), dtype=mx.uint32)]

    monkeypatch.setattr(T, "_admit_metal", lambda allow: None)
    monkeypatch.setitem(T._KERNELS, "forward", fake_kernel)          # treated as module-built
    monkeypatch.setattr(T, "_metal_kernel", lambda name: T._KERNELS[name])
    y, engaged = T.metal_sparse_tree_forward(*_inputs(3), [-1, 0, 0], allow_unqualified_metal=True)
    key = id(engaged)
    assert T._PENDING_FLAGS.get(key) is engaged
    assert T.confirm_device_engagement(engaged) is False             # CPU: never engagement
    del y, engaged
    gc.collect()
    assert key not in T._PENDING_FLAGS and len(T._PENDING_FLAGS) == 0
    assert T.status()["device_engaged"] is False
    monkeypatch.delitem(T._KERNELS, "forward")


# ---------------------------------------------------------------- Metal definitions (never built here)

def test_metal_sources_are_sparse_and_isolated():
    src = T.SPARSE_FORWARD_SOURCE
    assert "float kept[KSLOTS][N];" in src and "MAXW" not in src and "state_out" not in src
    assert "entry[FOREST ? N : 1]" in src and "table[2 * W + row]" in src
    assert "state_out" in T.REPLAY_SOURCE and "path[p]" in T.REPLAY_SOURCE
    assert "engaged[0] = 1u;" in src and "engaged[0] = 1u;" in T.REPLAY_SOURCE
    assert "StT" not in src + T.REPLAY_SOURCE  # fp32 state only: no per-token state cast


def test_metal_entry_points_refuse_on_cpu_without_capability_queries():
    arrays = _inputs(3)
    with pytest.raises(T.TreeGDNUnavailable, match="unqualified"):
        T.metal_sparse_tree_forward(*arrays, [-1, 0, 0])
    with pytest.raises(T.TreeGDNUnavailable, match="default GPU"):
        T.metal_sparse_tree_forward(*arrays, [-1, 0, 0], allow_unqualified_metal=True)
    with pytest.raises(T.TreeGDNUnavailable, match="default GPU"):
        T.metal_replay_accepted_path(*arrays[1:], [-1, 0, 0], [0, 2], allow_unqualified_metal=True)
    with pytest.raises(T.TreeGDNUnavailable, match="unqualified"):
        T.metal_sparse_tree_forward(*arrays, [-1, 0, 0], allow_unqualified_metal=1)
    assert T.status()["device_engaged"] is False


def test_import_makes_no_metal_calls():
    code = ("import mlx.core as mx\n"
            "def boom(*a, **k): raise SystemExit(7)\n"
            "mx.metal.is_available = boom; mx.fast.metal_kernel = boom; mx.device_info = boom\n"
            "import sys, mlx2.adapters.qwen38_tree_gdn\n"
            "print(any('gated_delta' in m or 'tree_hybrid_verify' in m for m in sys.modules))\n")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0 and probe.stdout.strip() == "False", probe.stderr
