"""Synthetic CPU roots: exact ordinary math with private wrapped-ring state."""

import copy
import hashlib
import os
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_inkling_model_cpu import model

from mlx2.runtime import external_speculative as es
from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor
from mlx2.runtime.models.cache import ArraysCache, KVCache

mx.set_default_device(mx.cpu)
_ROUND = es.ExternalDraftBatchGenerator._round
_READ = es.ExternalDraftBatchGenerator._read_parked
_DECISIONS = es.ExternalDraftBatchGenerator._parked_decisions


def assert_state(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        assert type(got) is type(want)
        if isinstance(got, ArraysCache):
            pairs = zip(got.cache, want.cache, strict=True)
            assert not got.speculating and not got._rollbacks
            assert got.meta_state == want.meta_state
        else:
            assert got.offset == want.offset
            assert getattr(got, "_idx", None) == getattr(want, "_idx", None)
            pairs = zip((got.keys, got.values), (want.keys, want.values), strict=True)
        for a, b in pairs:
            assert (a is None) == (b is None)
            if a is not None:
                if type(got) is KVCache:
                    # Private anchors may lend append-only capacity; abandoned
                    # suffix writes outside the committed fill are not state.
                    a, b = a[..., : got.offset, :], b[..., : want.offset, :]
                assert bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("prefix", [2, 8])
@pytest.mark.parametrize("commit", [False, True])
def test_anchor_matches_ordinary_and_keeps_live_state_private(prefix, commit):
    target = model()
    cache = target.make_cache()
    mx.eval(target(mx.array([[1, 2] * (prefix // 2)]), cache=cache))
    before, ordinary = copy.deepcopy(cache), copy.deepcopy(cache)
    expected, expected_features = target.forward_with_taps(
        mx.array([[7]]), ordinary, [1]
    )
    executor = InklingTreeExecutor.create()
    actual, features, transaction = executor.forward_many(
        target, [[7]], [[-1]], [cache], [1], cached=False, configured_limit=1
    )
    mx.eval(actual, features, expected, expected_features)
    assert bool(mx.array_equal(actual, expected).item())
    assert bool(mx.array_equal(features, expected_features).item())
    assert_state(cache, before)
    assert not transaction.physical_packed
    assert executor.receipt()["ordinary_anchor_dispatches"] == 1
    assert executor.receipt()["descriptor_plan_captures"] == 1
    assert executor.receipt()["descriptor_plan_declines"] == 0
    assert executor.receipt()["physical_packed_dispatches"] == 0
    if commit:
        transaction.commit_paths([[0]])
        assert_state(cache, ordinary)
    else:
        transaction.abort()
        assert_state(cache, before)
    want = ordinary if commit else before
    a, b = target(mx.array([[9]]), cache=cache), target(mx.array([[9]]), cache=want)
    mx.eval(a, b)
    assert bool(mx.array_equal(a, b).item())


def test_failed_anchor_forward_never_publishes_wrapped_state(monkeypatch):
    target = model()
    cache = target.make_cache()
    mx.eval(target(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache))
    before = copy.deepcopy(cache)
    forward = target.forward_with_taps

    def fail(*args, **kwargs):
        values = forward(*args, **kwargs)
        mx.eval(values)
        raise RuntimeError("injected after private forward")

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="injected"):
        InklingTreeExecutor.create().forward_many(
            target, [[7]], [[-1]], [cache], [1], cached=False, configured_limit=1
        )
    assert_state(cache, before)


def run_pipeline(
    monkeypatch, *, pipelined=True, fail=False, cancel=False, prefix=8, width=1
):
    target = model()
    target.tree_cohort = width > 1
    draft = PromptLookupDraftModel(target, num_draft=3)
    clock = {"now": 1, "reads": 0, "failed": False}

    def round_(self, cohort, *, forced_depth=None):
        result = _ROUND(self, cohort, forced_depth=forced_depth)
        clock["now"] += 100 if forced_depth == 0 else 100_000
        return result

    def read(self, *args, **kwargs):
        clock["now"] += 100
        clock["reads"] += 1
        return _READ(self, *args, **kwargs)

    def decisions(self, *args, **kwargs):
        if fail and clock["reads"] == 8 and not clock["failed"]:
            clock["failed"] = True
            raise RuntimeError("injected after commit and following dispatch")
        return _DECISIONS(self, *args, **kwargs)

    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", round_)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_read_parked", read)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_parked_decisions", decisions)
    engine = es.ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="inkling-anchor-cpu",
        num_draft=3,
        prefill_step_size=8,
        dynamic_singleton_tree=True,
        dynamic_tree_max_width=4,
        dynamic_tree_overflow="cohort",
        tensorfold_cohort_limit=4,
        target_tree_executor_provider=InklingTreeExecutor.create(),
        cost_per_commit_handoff={
            "enabled": True,
            "reversible": True,
            "ordinary_samples": 1,
            "speculative_samples": 1,
            "loss_decisions": 1,
            "window": 2,
            "reprobe_tokens": 64,
            "reprobe_max_tokens": 64,
            "settle_rounds": 0,
            "pipelined_park": pipelined,
        },
    )
    uids = engine.insert(
        [list(range(1, prefix + row + 1)) for row in range(width)],
        max_tokens=[32] * width,
        sampling_configs=[{"sampling_temp": 0.0}] * width,
    )
    by_lane, finals = {uid: [] for uid in uids}, {}
    outputs, final = [], None
    for _ in range(200):
        try:
            _, responses = engine.next()
        except RuntimeError as error:
            if "injected after commit" not in str(error):
                raise
            continue
        for response in responses:
            outputs.append(response.token)
            by_lane[response.uid].append(response.token)
            if response.finish_reason:
                final = response
                finals[response.uid] = response
        if cancel and len(outputs) >= 12 and engine._parked_inflight is not None:
            caches = [engine.lanes[uid].cache for uid in uids]
            committed = copy.deepcopy(caches)
            engine.remove(uids)
            for cache, prior in zip(caches, committed, strict=True):
                assert_state(cache, prior)
            assert engine._parked_inflight is None
            assert engine.scheduler_stats["external_parked_pipeline_aborts"] == 1
            break
        if not engine.lanes:
            break
    else:
        raise AssertionError("Inkling parked pipeline stalled")
    if fail:
        assert clock["failed"]
    engine._test_by_lane = by_lane, finals
    return outputs, final, engine


@pytest.mark.parametrize("prefix", [3, 8])
@pytest.mark.parametrize("fail", [False, True])
def test_parked_pipeline_restores_wrapped_and_crossing_rings(monkeypatch, prefix, fail):
    ordinary, reference, _ = run_pipeline(monkeypatch, pipelined=False, prefix=prefix)
    actual, final, engine = run_pipeline(monkeypatch, fail=fail, prefix=prefix)
    assert actual == ordinary
    assert final.all_tokens == reference.all_tokens
    assert_state(final.prompt_cache, reference.prompt_cache)
    final.cache_sidecar.validate("inkling-anchor-cpu", len(final.all_tokens))
    assert engine.scheduler_stats["external_parked_pipelined_rounds"] > 10
    assert engine.scheduler_stats["external_cost_plain_probe_rounds"] >= 1
    assert engine.scheduler_stats["external_plain_anchor_provider_rounds"] > 10


def test_cancelling_unread_parked_anchor_preserves_committed_rings(monkeypatch):
    run_pipeline(monkeypatch, cancel=True)


@pytest.mark.parametrize("fail", [False, True])
def test_packed_parked_pipeline_recovers_each_lane(monkeypatch, fail):
    _, _, ordinary = run_pipeline(monkeypatch, pipelined=False, width=2, prefix=3)
    _, _, candidate = run_pipeline(monkeypatch, width=2, prefix=3, fail=fail)
    streams, finals = candidate._test_by_lane
    expected_streams, expected_finals = ordinary._test_by_lane
    assert streams == expected_streams
    assert len(finals) == 2
    for uid, response in finals.items():
        assert_state(response.prompt_cache, expected_finals[uid].prompt_cache)
        response.cache_sidecar.validate("inkling-anchor-cpu", len(response.all_tokens))
    assert candidate.scheduler_stats["external_parked_pipelined_rounds"] > 10


def test_cancelling_packed_ordinary_cohort_preserves_both_rings(monkeypatch):
    run_pipeline(monkeypatch, cancel=True, width=2, prefix=3)


def test_convolution_descriptor_clone_owns_lists_and_tensor_wrappers():
    target = model()
    cache = target.make_cache()
    mx.eval(target(mx.array([[1, 2, 3, 4, 5, 6]]), cache=cache))
    source = cache[1]
    clone = source.committed_descriptor_clone()
    assert type(clone) is type(source)
    assert clone.meta_state == source.meta_state
    assert clone.cache is not source.cache
    assert clone._checkpoints is not source._checkpoints
    before = mx.array(source.cache[0])
    clone.cache[0][..., 0] = 123
    mx.eval(clone.cache, source.cache, before)
    assert bool(mx.array_equal(source.cache[0], before).item())
    assert not bool(mx.array_equal(clone.cache[0], before).item())


def test_convolution_descriptor_plan_refuses_foreign_or_live_state():
    from mlx2.runtime.cow_cache import CommittedDescriptorPlan

    cache = model().make_cache()
    plan = CommittedDescriptorPlan(cache, "revision-a")
    assert plan.capture(cache, "revision-a") is not None
    assert plan.capture(cache, "revision-b") is None
    cache[1].start_speculation()
    assert plan.capture(cache, "revision-a") is None
    cache[1].stop_speculation()
    cache[1].foreign_state = object()
    assert plan.capture(cache, "revision-a") is None


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("commit", [False, True])
def test_packed_ordinary_cohort_owns_mixed_offset_rings(width, commit):
    target = model()
    target.tree_cohort = True
    caches = [target.make_cache() for _ in range(width)]
    for cache, prefix in zip(caches, [3, 8, 5, 9]):
        mx.eval(target(mx.array([[1] * prefix]), cache=cache))
    before, refs = copy.deepcopy(caches), copy.deepcopy(caches)
    tokens = mx.argmax(mx.eye(32)[mx.array(list(range(10, 10 + width)))], axis=1)
    expected = mx.concatenate(
        [target(mx.array([[10 + i]]), cache=cache) for i, cache in enumerate(refs)],
        axis=0,
    )
    executor = InklingTreeExecutor()
    actual, features, transaction = executor.forward_plain_anchors(
        target, tokens[:, None], caches, [1]
    )
    mx.eval(actual, features, expected)
    # CPU native GEMM has a different row reduction; Metal exact lanes are
    # checked bitwise in the component test, including successor decoding.
    assert bool(mx.allclose(actual, expected, atol=1e-5).item())
    for cache, original in zip(caches, before, strict=True):
        assert_state(cache, original)
    assert transaction.plain and transaction.physical_packed
    assert executor.receipt()["cross_request_packed_dispatches"] == 1
    if commit:
        transaction.commit(accepted_lengths=[1] * width)
    else:
        transaction.abort()
        refs = before
    for cache, ref in zip(caches, refs, strict=True):
        a, b = (
            target(mx.array([[25]]), cache=cache),
            target(mx.array([[25]]), cache=ref),
        )
        mx.eval(a, b)
        assert bool(mx.allclose(a, b, atol=1e-5).item())
        for actual_plane, expected_plane in zip(cache, ref, strict=True):
            assert actual_plane.meta_state == expected_plane.meta_state


def test_ordinary_cohort_rejects_shared_cache_descriptors():
    target = model()
    cache = target.make_cache()
    with pytest.raises(ValueError, match="independent descriptors"):
        InklingTreeExecutor().forward_plain_anchors(
            target, mx.array([[1], [2]]), [cache, cache], [1]
        )


def test_private_anchor_admission_preserves_a_parked_cohort(monkeypatch):
    from test_external_prefill_admission_cpu import _generator

    target = model()
    caches = [target.make_cache() for _ in range(2)]
    for cache in caches:
        # A substantial resident global prefix makes the false four-copy
        # reservation dominate the small one-token transient.
        for item in cache:
            if type(item) is KVCache:
                item.update_and_fetch(
                    mx.zeros((1, 1, 4096, 8)), mx.zeros((1, 1, 4096, 8))
                )
    state = SimpleNamespace(
        speculation_paused=True, settings=SimpleNamespace(deferred_context_cap=256)
    )
    lanes = [
        SimpleNamespace(
            cache=c,
            draft_cache=[],
            remaining=[1],
            tail=mx.zeros((1, 0)),
            ordinary=False,
            cancelled=False,
            processors=[],
            sampling={"sampling_temp": 0.0},
            maximum=32,
            generated=1,
            cost_handoff_state=state,
        )
        for c in caches
    ]
    engine = _generator(lambda: 1 << 40)
    engine._open = False
    engine.fly_verification = SimpleNamespace(enabled=False)
    engine.model = target
    engine.dynamic_singleton_tree = True
    engine.target_execution = "provider"
    engine.target_tree_executor_provider = InklingTreeExecutor()
    engine._ast_anchor_cohort_eligible = lambda cohort: False
    assert engine._admit(lanes, 1)
    private = engine.scheduler_stats["reservation_bytes"]
    assert engine.scheduler_stats["external_private_anchor_admission_lanes"] == 2
    reserve = engine.target_tree_executor_provider.plain_anchor_cache_reservation
    monkeypatch.setattr(
        engine.target_tree_executor_provider,
        "plain_anchor_cache_reservation",
        lambda cache: None,
    )
    assert engine._admit(lanes, 1)
    general = engine.scheduler_stats["reservation_bytes"]
    assert private < general
    engine.memory_headroom = lambda: (private + general) // 2
    assert len(engine._fit_cohort(lanes, 1, max_lanes=2)) == 1
    monkeypatch.setattr(
        engine.target_tree_executor_provider, "plain_anchor_cache_reservation", reserve
    )
    assert engine._fit_cohort(lanes, 1, max_lanes=2) == lanes

    # The rebate cannot leak into a different executed ownership contract.
    engine.memory_headroom = lambda: 1 << 40
    for field, value in [
        ("ordinary", True),
        ("processors", [object()]),
        ("maximum", 2),
    ]:
        previous = getattr(lanes[0], field)
        setattr(lanes[0], field, value)
        assert engine._admit(lanes, 1)
        assert engine.scheduler_stats["reservation_bytes"] == general
        setattr(lanes[0], field, previous)
    state.speculation_paused = False
    assert engine._admit(lanes, 1)
    assert engine.scheduler_stats["reservation_bytes"] == general
    state.speculation_paused = True
    assert engine._admit(lanes, 2)
    assert engine.scheduler_stats["reservation_bytes"] > general
    assert engine._admit(lanes, 1, prefill=True)
    assert engine.scheduler_stats["reservation_bytes"] >= 4 * sum(
        c.nbytes for cache in caches for c in cache
    )


def test_private_anchor_reservation_refuses_unknown_planes_and_prices_growth():
    cache = model().make_cache()
    executor = InklingTreeExecutor()
    assert (
        executor.plain_anchor_cache_reservation(cache) is None
    )  # no loaded global capacity
    global_kv = next(c for c in cache if type(c) is KVCache)
    global_kv.update_and_fetch(mx.zeros((1, 1, 256, 8)), mx.zeros((1, 1, 256, 8)))
    initial = executor.plain_anchor_cache_reservation(cache)
    assert initial >= 3 * global_kv.nbytes  # old prefix and two allocation quanta
    global_kv._reserve_tokens = 4096
    assert executor.plain_anchor_cache_reservation(cache) >= 16 * global_kv.nbytes
    cache[1] = ArraysCache(4)
    assert executor.plain_anchor_cache_reservation(cache) is None


@pytest.mark.parametrize("edit", ["same_size_restored_mtime", "replace", "grow"])
def test_source_guard_rejects_changed_content(tmp_path, monkeypatch, edit):
    from mlx2.runtime.inkling_tree_executor import _source_stat

    source = tmp_path / "source.py"
    source.write_bytes(b"original")
    executor = InklingTreeExecutor()
    executor._sources = (source,)
    reads = []

    def identity():
        reads.append(True)
        return ((source.name, hashlib.sha256(source.read_bytes()).hexdigest()),)

    monkeypatch.setattr(executor, "_identity", identity)
    executor.source_identity = identity()
    executor._source_stats = (_source_stat(source),)
    executor.validate_revision()
    assert len(reads) == 1
    # Metadata changes with identical bytes may be accepted after hashing.
    saved = source.stat()
    os.utime(source, ns=(saved.st_atime_ns, saved.st_mtime_ns + 10_000_000))
    executor.validate_revision()
    assert len(reads) == 2
    executor.validate_revision()
    assert len(reads) == 2
    saved = source.stat()
    if edit == "replace":
        replacement = tmp_path / "new.py"
        replacement.write_bytes(b"modified")
        replacement.replace(source)
    else:
        source.write_bytes(
            b"modified" if edit == "same_size_restored_mtime" else b"longer changed"
        )
    os.utime(source, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    with pytest.raises(RuntimeError, match="source changed"):
        executor.validate_revision()


def test_initial_plain_probe_admits_the_actual_batched_anchor(monkeypatch):
    target = model()
    target.tree_cohort = True
    engine = es.ExternalDraftBatchGenerator(
        target,
        draft_model=PromptLookupDraftModel(target, num_draft=3),
        binding="inkling-probe-admission-cpu",
        num_draft=3,
        prefill_step_size=8,
        completion_batch_size=2,
        dynamic_singleton_tree=True,
        dynamic_tree_max_width=4,
        dynamic_tree_overflow="cohort",
        tensorfold_cohort_limit=4,
        target_tree_executor_provider=InklingTreeExecutor.create(),
        cost_per_commit_handoff={
            "enabled": True, "reversible": True, "ordinary_samples": 1,
            "speculative_samples": 1, "loss_decisions": 1, "window": 2,
            "settle_rounds": 0,
        },
        memory_headroom=lambda: 1 << 40,
    )
    try:
        uids = engine.insert([[1, 2, 3], [4, 5, 6, 7]], max_tokens=[32, 32])
        lanes = [engine.lanes[uid] for uid in uids]
        for lane in lanes:
            while lane.anchor is None:
                engine._prefill(lane)
            lane.ready.clear()
        states = [lane.cost_handoff_state for lane in lanes]
        before = [state.receipt() for state in states]
        for _ in range(3):
            assert engine._plain_probe_admission_eligible(lanes)
        assert [state.receipt() for state in states] == before
        assert engine._admit(lanes, 1)
        ordinary_bytes = engine.scheduler_stats["reservation_bytes"]
        assert engine.scheduler_stats["external_private_anchor_admission_lanes"] >= 2
        append = engine._tree_node_budget(2) + 1
        assert engine._admit(lanes, append)
        speculative_bytes = engine.scheduler_stats["reservation_bytes"]
        assert ordinary_bytes < speculative_bytes
        engine.memory_headroom = lambda: (ordinary_bytes + speculative_bytes) // 2
        assert engine._fit_cohort(lanes, 1) == lanes
        assert len(engine._fit_cohort(lanes, append)) < 2
        assert [state.receipt() for state in states] == before
        # Execute the actual scheduler and private transaction, not a mocked
        # probe decision. Its tokens must equal each lane's ordinary step.
        expected = []
        for lane in lanes:
            cache = copy.deepcopy(lane.cache)
            logits = target(mx.array([[lane.anchor]]), cache=cache)
            expected.append(int(mx.argmax(logits[0, -1]).item()))
        generated = [lane.generated for lane in lanes]
        engine._decode_phase(lanes)
        assert [lane.anchor for lane in lanes] == expected
        assert [lane.generated for lane in lanes] == [n + 1 for n in generated]
        assert engine.scheduler_stats["external_plain_probe_admission_groups"] == 1
        assert engine.scheduler_stats["external_cost_plain_probe_rounds"] == 1
        assert engine.scheduler_stats["external_round_cohort_width_2"] == 1
        assert all(not state.needs_ordinary_probe(2) for state in states)
        # Established cohorts must again reserve the full speculative shape.
        assert not engine._plain_probe_admission_eligible(lanes)
    finally:
        engine.close()


@pytest.mark.parametrize("case", ["one_way", "mixed", "sampled", "processed", "final"])
def test_plain_probe_admission_does_not_underprice_other_routes(case):
    from mlx2.runtime.cost_per_commit_handoff import (
        CostPerCommitHandoffSettings,
        CostPerCommitLaneState,
    )

    engine = object.__new__(es.ExternalDraftBatchGenerator)
    engine.dynamic_singleton_tree = True
    engine._open = False
    engine.fly_verification = SimpleNamespace(enabled=False)
    engine.model = model()
    engine.target_execution = "provider"
    engine.target_tree_executor_provider = InklingTreeExecutor.create()
    engine._ast_anchor_cohort_eligible = lambda cohort: False
    settings = CostPerCommitHandoffSettings(enabled=True, reversible=case != "one_way")
    lanes = [SimpleNamespace(
        cost_handoff_state=CostPerCommitLaneState(settings, lane_uid=uid, owner_scope="test"),
        ordinary=False, cancelled=False, processors=[], sampling={"sampling_temp": 0.0},
        maximum=32, generated=1, cache=engine.model.make_cache(),
    ) for uid in range(2)]
    if case == "mixed":
        lanes[1].cost_handoff_state.needs_ordinary_probe = lambda width: False
    elif case == "sampled":
        lanes[1].sampling["sampling_temp"] = 0.7
    elif case == "processed":
        lanes[1].processors = [object()]
    elif case == "final":
        lanes[1].generated = 31
    assert not engine._plain_probe_admission_eligible(lanes)
