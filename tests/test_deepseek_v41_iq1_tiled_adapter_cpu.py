"""Exact CPU block-state transactions; packed Metal math remains separate."""

import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config, hash_metadata
from test_deepseek_v41_iq1_adapter_cpu import compact_fixture  # noqa: F401
from test_deepseek_v41_mlx_adapter_cpu import native_fixture  # noqa: F401

from mlx2.adapters import deepseek_v41_iq1_tiled as tiled
from mlx2.adapters.deepseek_v41_iq1 import (
    DeepSeekV41IQ1Adapter,
    execution_policy,
    optimized_policy,
)
from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
from mlx2.runtime.cache_operations import CacheOperationsUnsupported
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher

mx.set_default_device(mx.cpu)


class Store:
    atlas = None

    def __init__(self, *, hit=False, fail=False, unresolved=False):
        self._active = False
        self.hit, self.fail, self.unresolved = hit, fail, unresolved
        self.filled, self.ensures = set(), []
        self.windows = 0
        self.markers, self.deferred = [], 0

    @contextmanager
    def window(self):
        assert not self._active
        self._active = True
        self.windows += 1
        try:
            yield
        finally:
            self._active = False

    def ensure(self, prefix, ids):
        assert self._active and ids.dtype == np.uint32
        self.ensures.append((prefix, ids.copy()))
        if self.fail:
            raise OSError("injected source replacement")
        self.filled.add(prefix)

    def end_context(self):
        pass

    def defer_epoch(self, marker):
        assert self._active
        self.markers.append(marker)
        self.deferred += 1
        self._active = False

    def drain_epochs(self):
        mx.eval(self.markers)
        self.markers.clear()


class TransactionWeights(Weights):
    """Real V4.1 CPU math with only bank/kernel dispatch replaced by a fixture."""

    evaluate_block = tiled.TiledV41Weights.evaluate_block
    _missing = tiled.TiledV41Weights._missing
    _ensure_routes = tiled.TiledV41Weights._ensure_routes
    evaluate_dense_window = tiled.TiledV41Weights.evaluate_dense_window
    moe = tiled.TiledV41Weights.moe

    def __init__(
        self, args, *, hit=False, fail=False, unresolved=False, completion="sync"
    ):
        super().__init__(args)
        self.expert_stream = Store(hit=hit, fail=fail, unresolved=unresolved)
        self._block_active = False
        self.arena_completion, self._publication_values = completion, ()
        self.block_transactions = dict.fromkeys(
            (
                "blocks",
                "all_hit",
                "misses",
                "repaired",
                "route_id_readbacks",
                "flag_readbacks",
                "direct_ensured_calls",
            ),
            0,
        )

    def dense_window(self):
        return self.expert_stream.window()

    def _routed(self, x, ids, scores, prefix):
        hit = self.expert_stream.hit or prefix in self.expert_stream.filled
        hit = hit and not self.expert_stream.unresolved
        result = Weights.moe(self, x, ids, scores, prefix) if hit else mx.zeros_like(x)
        return result, mx.array(not hit, mx.bool_)


def _engram(args, weights):
    rng = np.random.default_rng(1989)
    for layer in args.engram_layer_ids:
        weights.data[f"blk.{layer}.engram_q_norm.weight"] = mx.ones(
            (4, args.hidden_size)
        )
        weights.data[f"blk.{layer}.engram_k_norm.weight"] = mx.ones(
            (4, args.hidden_size)
        )
        weights.data[f"blk.{layer}.engram_kv.weight"] = mx.array(
            rng.normal(0, 0.0004, (5 * args.hidden_size, 6144)).astype(np.float32)
        )
    metadata = hash_metadata()
    metadata["deepseek41.engram.token_map"] = [i % 5 for i in range(args.vocab_size)]

    class Rows:
        def prefetch(self, ids):
            pass

        def read(self, ids):
            return (
                np.broadcast_to(np.asarray(ids)[..., None] / 100, (*ids.shape, 256))
                .astype(np.float32)
                .copy()
            )

    return {
        "hasher": EngramHasher(metadata),
        "engram_tables": {i: Rows() for i in args.engram_layer_ids},
    }


def _models(
    *,
    hit=False,
    fail=False,
    unresolved=False,
    engram=False,
    completion="sync",
    decode=None,
):
    args = config(
        **(
            {
                "engram_layer_ids": (1, 3),
                "engram_num_embeddings": (48, 72),
                "engram_compressed_vocab_size": 5,
            }
            if engram
            else {}
        )
    )
    reference = Weights(args)
    weights = TransactionWeights(
        args, hit=hit, fail=fail, unresolved=unresolved, completion=completion
    )
    if decode is not None:
        weights.compute_policy = ":".join(tiled.numerical_profile(decode))
    options = [_engram(args, value) if engram else {} for value in (reference, weights)]
    models = [
        DeepSeekV41Model(
            args, value, identity="synthetic-tiled-transaction@cpu", **option
        )
        for value, option in zip((reference, weights), options)
    ]
    return *models, weights


def _same_state(left, right):
    for a, b in zip(left, right):
        # Storage/compute profiles deliberately have distinct cache identities.
        assert (
            a.meta_state[:1] + a.meta_state[2:] == b.meta_state[:1] + b.meta_state[2:]
        )
        assert not a.pending and not b.pending
        assert a._capacity == b._capacity
        for av, bv in zip(a.state, b.state):
            if av is None:
                assert bv is None
            else:
                np.testing.assert_array_equal(np.array(av), np.array(bv))


@pytest.mark.parametrize("hit", [False, True])
@pytest.mark.parametrize("engram", [False, True])
@pytest.mark.parametrize("completion", ["sync", "async"])
def test_block_miss_repair_preserves_outputs_taps_cache_and_single_attention(
    hit, engram, completion
):
    reference, target, weights = _models(hit=hit, engram=engram, completion=completion)
    ca, cb = reference.make_cache(), target.make_cache()
    calls = [0] * len(target.layers)
    for index, block in enumerate(target.layers):
        original = block.attention_half

        def counted(*args, _index=index, _original=original, **kwargs):
            calls[_index] += 1
            return _original(*args, **kwargs)

        block.attention_half = counted
    for tokens in ([1, 2, 3], [4], [5, 6, 7, 8, 9]):
        ids = mx.array([tokens], mx.int32)
        a, ta = reference.forward_with_taps(ids, ca, (1, 3))
        b, tb = target.forward_with_taps(ids, cb, (1, 3))
        np.testing.assert_array_equal(np.array(a), np.array(b))
        np.testing.assert_array_equal(np.array(ta), np.array(tb))
        _same_state(ca, cb)
    assert calls == [3] * 5
    assert weights.block_transactions["blocks"] == 15
    assert weights.block_transactions["repaired"] == (0 if hit else 5)
    assert weights.block_transactions["route_id_readbacks"] == (0 if hit else 5)
    assert weights.expert_stream.windows == 3 * 7  # embedding + five blocks + head
    assert weights.expert_stream.deferred == (15 if completion == "async" else 0)
    assert not weights.expert_stream.markers and not weights._publication_values


@pytest.mark.parametrize("completion", ["sync", "async"])
def test_all_hit_blocks_never_read_route_ids(monkeypatch, completion):
    _, target, weights = _models(hit=True, completion=completion)
    monkeypatch.setattr(
        tiled, "_host_route_ids", lambda ids: pytest.fail("warm route IDs reached CPU")
    )
    output = target(mx.array([[1, 2, 3]], mx.int32), target.make_cache())
    assert np.isfinite(np.array(output)).all()
    assert weights.block_transactions["all_hit"] == 5
    assert weights.block_transactions["flag_readbacks"] == 5


@pytest.mark.parametrize("failure", ["fill", "unresolved", "evaluation"])
@pytest.mark.parametrize("completion", ["sync", "async"])
def test_failed_transaction_keeps_cache_unpublishable(failure, monkeypatch, completion):
    _, target, _ = _models(
        fail=failure == "fill",
        unresolved=failure == "unresolved",
        completion=completion,
    )
    cache = target.make_cache()
    if failure == "evaluation":
        original = target._weight_eval
        calls = 0

        def evaluate(*values):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected completion failure")
            original(*values)

        monkeypatch.setattr(target, "_weight_eval", evaluate)
    with pytest.raises((OSError, RuntimeError)):
        target(mx.array([[1, 2, 3]], mx.int32), cache)
    assert all(state.pending for state in cache)
    assert not target.weights._block_active
    for state in cache:
        with pytest.raises((ValueError, CacheOperationsUnsupported)):
            state.snapshot()
    with pytest.raises(ValueError, match="frontier"):
        target(mx.array([[4]], mx.int32), cache)


@pytest.mark.parametrize("completion", ["sync", "async"])
@pytest.mark.parametrize("decode", ["row_exact", "column_f32"])
def test_serial_verify_and_snapshot_replay_use_resolved_transactions(
    completion, decode
):
    reference, target, weights = _models(
        engram=True, completion=completion, decode=decode
    )
    # The dense CPU oracle verifies serial transaction/state behavior, while
    # separate kernel tests and parent GPU evidence own the column arithmetic.
    ca, cb = reference.make_cache(), target.make_cache()
    ids = mx.array([[1, 2, 3]], mx.int32)
    reference(ids, ca)
    target(ids, cb)
    snapshots = [state.snapshot() for state in cb]
    probe = mx.array([[4, 5, 6]], mx.int32)
    a, _ = reference.verify_rows(probe, ca)
    b, _ = target.verify_rows(probe, cb)
    np.testing.assert_array_equal(np.array(a), np.array(b))
    _same_state(ca, cb)
    for state, snapshot in zip(cb, snapshots):
        state.restore(snapshot)
    replay, _ = target.verify_rows(probe, cb)
    np.testing.assert_array_equal(np.array(b), np.array(replay))
    assert weights.block_transactions["direct_ensured_calls"] == 0


def test_direct_moe_is_explicitly_ensured_and_evaluated():
    _, _, weights = _models()
    x, ids, scores = mx.ones((1, 8)), mx.array([[0, 1]], mx.uint32), mx.ones((1, 2))
    output = weights.moe(x, ids, scores, "blk.0")
    expected = Weights.moe(weights, x, ids, scores, "blk.0")
    np.testing.assert_array_equal(np.array(output), np.array(expected))
    assert weights.block_transactions["direct_ensured_calls"] == 1
    assert weights.block_transactions["route_id_readbacks"] == 1
    assert not weights.expert_stream._active


def test_completion_failure_blocks_publication_and_retains_pending_state(monkeypatch):
    _, target, weights = _models(hit=True, completion="async")

    def fail_pending():
        if weights.expert_stream.markers:
            raise RuntimeError("injected marker completion failure")

    monkeypatch.setattr(weights.expert_stream, "drain_epochs", fail_pending)
    cache = target.make_cache()
    with pytest.raises(RuntimeError, match="marker completion"):
        target(mx.array([[1, 2, 3]], mx.int32), cache)
    assert weights._publication_values and weights.expert_stream.markers
    assert all(state.pending for state in cache)
    for state in cache:
        with pytest.raises(CacheOperationsUnsupported):
            state.snapshot()


def test_final_head_evaluates_current_engram_tail_before_publication(monkeypatch):
    _, target, _ = _models(hit=True, engram=True, completion="async")
    cache = target.make_cache()
    original, boundaries = target._weight_eval, []

    def evaluate(*values):
        if len(values) == 6:
            assert values[3][0][-1] is cache[0].hash_tail
            assert all(state.pending for state in cache)
            boundaries.append("head with current frontier")
        original(*values)

    monkeypatch.setattr(target, "_weight_eval", evaluate)
    target(mx.array([[1, 2, 3]], mx.int32), cache)
    assert boundaries == ["head with current frontier"]
    assert all(not state.pending for state in cache)


def test_tiled_admission_includes_banks_and_selected_parallel_read_bound(request):
    from mlx2.runtime.streamed_load import WeightStreamRequest

    root, _, _ = request.getfixturevalue("compact_fixture")
    artifact = DeepSeekV41IQ1Adapter.artifact_inspector(root)
    adapter = object.__new__(DeepSeekV41IQ1Adapter)
    adapter.kernel_policy = execution_policy({"deepseek_expert_layout": "tiled64"})
    plans = [
        adapter._weight_admission_plan(
            artifact,
            WeightStreamRequest(
                "moe_experts",
                1 << 20,
                admission_limit_bytes=8 << 30,
                read_workers=workers,
            ),
        )
        for workers in (1, 4)
    ]
    assert (
        plans[1]["expert_staging_bound_bytes"]
        == 4 * plans[0]["expert_staging_bound_bytes"]
    )
    assert plans[0]["minimum_expert_cache_bytes"] == 2 * 3 * 256 * 50 + 2 * 4
    assert (
        plans[0]["expert_active_bound_bytes"] >= 4 * plans[0]["expert_bank_bound_bytes"]
    )


@pytest.mark.parametrize(
    "requested_workers", [None, 32], ids=["public-default", "explicit32"]
)
def test_default_auto_stream_request_uses_eight_effective_tiled_readers(
    request, requested_workers
):
    """Exercise the public default16 and serving32 through admission/factory."""
    from mlx2.runtime.streamed_load import WeightStreamRequest

    root, _, _ = request.getfixturevalue("compact_fixture")
    artifact = DeepSeekV41IQ1Adapter.artifact_inspector(root)
    streaming = WeightStreamRequest(
        mode="moe_experts",
        budget_mode="auto",
        budget_bytes=0,
        admission_limit_bytes=8 << 30,
        **({} if requested_workers is None else {"read_workers": requested_workers}),
    )
    requested = streaming.read_workers
    assert requested > 8
    adapter = object.__new__(DeepSeekV41IQ1Adapter)
    adapter.kernel_policy = execution_policy({"deepseek_expert_layout": "tiled64"})
    adapter.native_admission = adapter._weight_admission_plan(artifact, streaming)
    plan = adapter.native_admission
    assert plan["expert_requested_read_workers"] == requested
    assert plan["expert_read_workers"] == 8
    assert plan["ceiling_source"] == "auto"
    weights = adapter._construct_weights(artifact, streaming)
    try:
        assert weights.bounds["expert_read_workers"] == 8
        assert weights.bounds["expert_requested_read_workers"] == requested
        assert weights.expert_stream.read_workers == 8
        assert weights.expert_stream.requested_read_workers == requested
        assert weights.expert_stream.pool._max_workers == 8
        assert weights.expert_stream.pool._max_pending == 8
        assert weights.expert_stream.plan.staging_window == 8
        assert (
            weights.bounds["expert_staging_bound_bytes"]
            == plan["expert_staging_bound_bytes"]
        )
        assert weights.resident_bytes() == 0  # Constructor remains preallocation.
    finally:
        weights.close()


def test_layout_policy_preserves_default_and_rejects_unbound_values():
    assert execution_policy()["deepseek_expert_layout"] == "row_major"
    assert execution_policy(optimized_policy())["deepseek_expert_layout"] == "row_major"
    assert (
        execution_policy({"deepseek_expert_layout": "tiled64"})[
            "deepseek_expert_layout"
        ]
        == "tiled64"
    )
    for value in (True, "tiled", None, "auto"):
        with pytest.raises(ValueError, match="layout"):
            execution_policy({"deepseek_expert_layout": value})
    with pytest.raises(ValueError):
        execution_policy(
            {"deepseek_expert_layout": "tiled64", "deepseek_verify_rows": "layerwise"}
        )
    assert execution_policy()["deepseek_arena_completion"] == "sync"
    assert (
        execution_policy(
            {"deepseek_expert_layout": "tiled64", "deepseek_arena_completion": "async"}
        )["deepseek_arena_completion"]
        == "async"
    )
    with pytest.raises(ValueError):
        execution_policy({"deepseek_arena_completion": "async"})
    assert execution_policy()["deepseek_tiled_decode"] == "row_exact"
    for decode in ("column_f32", True, None, "column", "auto"):
        with pytest.raises(ValueError):
            execution_policy({"deepseek_tiled_decode": decode})
    assert (
        execution_policy(
            {"deepseek_expert_layout": "tiled64", "deepseek_tiled_decode": "column_f32"}
        )["deepseek_tiled_decode"]
        == "column_f32"
    )


@pytest.mark.parametrize(
    "decode,policy,scoped,rows,expected",
    [
        ("row_exact", "reference", False, 1, "row"),
        ("column_f32", "reference", False, 1, "column"),
        ("column_f32", "reference", False, 16, "column"),
        ("column_f32", "segmented_nax_fp16", False, 16, "column"),
        ("column_f32", "segmented_nax_fp16", True, 1, "column"),
        ("column_f32", "segmented_nax_fp16", True, 16, "nax"),
        ("column_f32", "segmented_nax_fp16", True, 64, "nax"),
        ("row_exact", "segmented_nax_fp16", True, 16, "nax"),
    ],
)
def test_selected_column_dispatch_never_changes_nax_scope(
    monkeypatch, decode, policy, scoped, rows, expected
):
    from contextlib import nullcontext

    from mlx2.runtime.gguf import v41_iq1_column, v41_iq1_tiled
    from mlx2.runtime.gguf.segmented_prefill import expert_prefill_context

    weights = object.__new__(tiled.TiledV41Weights)
    weights._parked = weights._closed = False
    weights._error = None
    weights.artifact = {
        "args": SimpleNamespace(
            hidden_size=256, moe_intermediate_size=256, num_experts_per_tok=2
        )
    }
    bank = SimpleNamespace(
        planes=tuple(object() for _ in range(9)),
        slot_map=mx.array([0, 1], mx.uint32),
        capacity=2,
    )
    weights.expert_stream = SimpleNamespace(bank=lambda prefix: bank)
    weights.tiled_decode, weights.expert_prefill_policy = decode, policy
    weights.scheduled_expert_calls, weights.expert_buffer_counters = {}, {}
    calls = []

    def row(x, ids, scores, planes, slot_map, **kwargs):
        assert planes is bank.planes and slot_map is bank.slot_map
        calls.append("nax" if kwargs["prefill"] else "row")
        return mx.zeros_like(x)

    def column(x, ids, scores, planes, slot_map, **kwargs):
        assert planes is bank.planes and slot_map is bank.slot_map
        assert kwargs["prefill"] is False
        calls.append("column")
        return mx.zeros_like(x)

    monkeypatch.setattr(v41_iq1_tiled, "moe", row)
    monkeypatch.setattr(v41_iq1_column, "moe", column)
    monkeypatch.setattr(
        v41_iq1_tiled,
        "missing_routes",
        lambda ids, slots, capacity: mx.zeros(ids.shape, mx.bool_),
    )
    monkeypatch.setattr(
        tiled, "_host_route_ids", lambda ids: pytest.fail("dispatch read host routes")
    )
    with expert_prefill_context(policy, tokens=rows) if scoped else nullcontext():
        output, missing = weights._routed(
            mx.ones((rows, 256)),
            mx.zeros((rows, 2), mx.uint32),
            mx.ones((rows, 2)),
            "blk.0",
        )
    assert calls == [expected]
    assert output.shape == (rows, 256) and not missing.item()
    route = {
        "row": "v41_iq1_tiled_row_exact",
        "column": "v41_iq1_tiled_column_f32",
        "nax": "v41_iq1_tiled_nax_fp16",
    }[expected]
    assert weights.scheduled_expert_calls == {route: 1}


def test_column_revision_binds_cache_identity_and_rejects_cross_profile_restore():
    args = config()
    models = []
    for decode in ("row_exact", "column_f32"):
        weights = TransactionWeights(args, hit=True)
        weights.compute_policy = ":".join(tiled.numerical_profile(decode))
        models.append(
            DeepSeekV41Model(args, weights, identity="synthetic-profile-isolation@cpu")
        )
    row, column = models
    assert row.identity != column.identity
    ca, cb = row.make_cache(), column.make_cache()
    row(mx.array([[1]], mx.int32), ca)
    column(mx.array([[1]], mx.int32), cb)
    with pytest.raises(ValueError, match="identity"):
        cb[0].restore(ca[0].snapshot())
    with pytest.raises(ValueError, match="identity"):
        column(mx.array([[2]], mx.int32), ca)


def test_tiled_adapter_selects_distinct_numerics_without_mutating_artifact_type(
    monkeypatch,
):
    def initialize(self, path, **kwargs):
        self.kernel_policy = self.policy_parser(kwargs["execution_policy"])
        self.descriptor = type(self).descriptor

    monkeypatch.setattr(DeepSeekV41MLXAdapter, "__init__", initialize)
    row = DeepSeekV41IQ1Adapter("not-opened")
    bank = DeepSeekV41IQ1Adapter(
        "not-opened", execution_policy={"deepseek_expert_layout": "tiled64"}
    )
    assert row.descriptor.model_type == bank.descriptor.model_type == "deepseek41_iq1"
    assert row.numeric_profile != bank.numeric_profile == tiled.NUMERIC_PROFILE
    assert row.numerics_revision != bank.numerics_revision
    assert bank.descriptor.metadata["expert_layout"] == "tiled64"


def test_tiled_provider_import_is_cpu_only():
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('mlx','torch','transformers','tokenizers'):
   raise RuntimeError('device import: '+fullname)
sys.meta_path.insert(0, Guard())
from mlx2.adapters.deepseek_v41_iq1_tiled import TiledV41Weights
from mlx2.adapters.deepseek_v41_iq1_tiled import numerical_profile
from mlx2.adapters.deepseek_v41_iq1 import execution_policy
assert execution_policy()['deepseek_expert_layout'] == 'row_major'
assert numerical_profile('column_f32')[0] == 'native_iq1_s_tiled_column_f32'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
