"""CPU admission and physical-receipt gates for the default-off B2 route."""

import json
import sys
from pathlib import Path
from contextlib import contextmanager
from threading import RLock
from types import ModuleType, SimpleNamespace

import pytest

from mlx2 import serving
from mlx2.runtime.paged_b2_research_profile import (
    CONTEXT_BOUNDS, SUSTAINED_CONTEXT_BOUNDS, FLAGS, PACKED_FLAG, SCHEMA, SCHEMA_V2, SCHEMA_V3,
    SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SUSTAINED_FLAG, SUSTAINED_TOKENS,
    COMBINED_FLAGS, DEFERRED_WRITE_EVAL_FLAG, DIRECT_FENCE_FLAG, SCHEMA_V7, STOCK_SDPA_FLAG,
    STRIPES_FLAG, VECTOR_ROPE_FLAG,
    load_b2_research_profile, validate_combined_environment)
from mlx2.server import validate_request
from mlx2.sampling_defaults import SamplingDefaults, VendorSampling, resolve_sampling
from varlen_b2_http_gate import _contexts as http_contexts
from varlen_b2_http_performance import (
    ORDER as HTTP_PERFORMANCE_ORDER, _decode_progress, _ready_ms)


def _identity():
    return {"host": "test", "hardware": "M5", "artifact_sha256": "a" * 64,
            "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
            "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
            "kernel_sha256": "e" * 64}


def _profile(identity):
    return {"schema": SCHEMA, "profile_id": "qwen3-06b-b2-63-65-research",
            "identity": identity, "ordered_context_tokens": [63, 65],
            "max_tokens": 2, "sampling": {"mode": "greedy", "processors": False},
            "required_environment": {key: "1" for key in FLAGS},
            "qualified": False, "price_usable": False,
            "serving_default": False, "warm_apcv2": False}


def _profile_v2(identity):
    profile = _profile(identity)
    profile.pop("ordered_context_tokens")
    profile.update(schema=SCHEMA_V2, profile_id="qwen3-06b-b2-32-127-research",
                   context_bounds=CONTEXT_BOUNDS)
    return profile


def _profile_v3(identity):
    profile = _profile_v2(identity)
    profile.update(schema=SCHEMA_V3, profile_id="qwen3-06b-b2-packed-32-127-research",
                   packed_prefill=True,
                   required_environment={key: "1" for key in (*FLAGS, PACKED_FLAG)})
    return profile


def _profile_v4(identity):
    profile = _profile_v3(identity)
    profile.update(schema=SCHEMA_V4, profile_id="qwen3-06b-b2-sustained-research",
                   context_bounds=SUSTAINED_CONTEXT_BOUNDS,
                   max_tokens=SUSTAINED_TOKENS, sustained_decode=True,
                   required_environment={key: "1" for key in
                                         (*FLAGS, PACKED_FLAG, SUSTAINED_FLAG)})
    return profile


def _profile_v5(identity):
    profile = _profile_v4(identity)
    profile.update(schema=SCHEMA_V5, profile_id="qwen3-06b-b2-vector-rope-research",
                   vector_q1_rope=True,
                   required_environment={key: "1" for key in
                                         (*FLAGS, PACKED_FLAG, SUSTAINED_FLAG,
                                          VECTOR_ROPE_FLAG)})
    return profile


def _profile_v6(identity):
    profile = _profile_v4(identity)
    profile.update(schema=SCHEMA_V6, profile_id="qwen3-06b-b2-direct-fence-research",
                   direct_grouped_fence=True,
                   required_environment={key: "1" for key in
                                         (*FLAGS, PACKED_FLAG, SUSTAINED_FLAG,
                                          DIRECT_FENCE_FLAG)})
    return profile


def _profile_v7(identity):
    profile = _profile_v6(identity)
    profile.update(schema=SCHEMA_V7, profile_id="qwen3-06b-b2-combined-research",
                   q1_simd_stripes=4,
                   combined_optimizations={"deferred_eval": True,
                                           "deferred_write_eval": False,
                                           "inline_metadata": True,
                                           "grouped_sampler": True,
                                           "stock_sdpa": False},
                   required_environment={**profile["required_environment"],
                                         **{key: ("0" if key == DEFERRED_WRITE_EVAL_FLAG else "1")
                                            for key in COMBINED_FLAGS},
                                         STRIPES_FLAG: "4",
                                         STOCK_SDPA_FLAG: "0"})
    return profile


def test_b2_combined_profile_binds_all_flags_and_legacy_rejects_them(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile_v7(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(path, live_identity=_identity(),
                                    context_lengths=(32, 96)) == profile
    environment = profile["required_environment"]
    assert validate_combined_environment(profile, environment)
    for key in COMBINED_FLAGS:
        with pytest.raises(ValueError, match="flags differ"):
            validate_combined_environment(profile, {
                **environment, key: "1" if environment[key] == "0" else "0"})
        with pytest.raises(ValueError, match="flags differ"):
            validate_combined_environment(_profile_v6(_identity()), {key: "1"})
    with pytest.raises(ValueError, match="stripe geometry"):
        validate_combined_environment(profile, {**environment,
                                                STRIPES_FLAG: "16"})
    with pytest.raises(ValueError, match="stock SDPA"):
        validate_combined_environment(profile, {
            **environment,
            STOCK_SDPA_FLAG: "1"})
    assert not validate_combined_environment(_profile_v6(_identity()), {})
    for options in ({"deferred_eval": True, "deferred_write_eval": False,
                     "inline_metadata": False,
                     "grouped_sampler": True, "stock_sdpa": False},
                    {"deferred_eval": True, "deferred_write_eval": False,
                     "inline_metadata": True,
                     "grouped_sampler": False, "stock_sdpa": False},
                    {"deferred_eval": False, "deferred_write_eval": False,
                     "inline_metadata": False,
                     "grouped_sampler": False, "stock_sdpa": True},
                    {"deferred_eval": False, "deferred_write_eval": True,
                     "inline_metadata": True,
                     "grouped_sampler": True, "stock_sdpa": False}):
        adjusted = {**profile, "combined_optimizations": options,
                    "required_environment": {**_profile_v6(_identity())[
                        "required_environment"], STRIPES_FLAG: "4",
                        STOCK_SDPA_FLAG: "1" if options["stock_sdpa"] else "0",
                        **{key: "1" if enabled else "0"
                            for key, enabled in zip(COMBINED_FLAGS, (
                                options["deferred_eval"], options["deferred_write_eval"],
                                options["inline_metadata"],
                                options["grouped_sampler"]))}}}
        path.write_text(json.dumps(adjusted))
        assert load_b2_research_profile(path, live_identity=_identity(),
                                        context_lengths=(32, 96)) == adjusted
        assert validate_combined_environment(adjusted, adjusted["required_environment"])
    for stripes in (8, 16, 32):
        adjusted = {**profile, "q1_simd_stripes": stripes,
                    "required_environment": {**profile["required_environment"],
                                             STRIPES_FLAG: str(stripes)}}
        path.write_text(json.dumps(adjusted))
        assert load_b2_research_profile(path, live_identity=_identity(),
                                        context_lengths=(32, 96)) == adjusted
    conflicting = {**profile,
                   "combined_optimizations": {**profile["combined_optimizations"],
                                              "deferred_write_eval": True},
                   "required_environment": {**profile["required_environment"],
                                            DEFERRED_WRITE_EVAL_FLAG: "1"}}
    path.write_text(json.dumps(conflicting))
    with pytest.raises(ValueError, match="exclusive"):
        load_b2_research_profile(path, live_identity=_identity(),
                                 context_lengths=(32, 96))
    for change in ({"combined_optimizations": {"deferred_eval": 1,
                                                "deferred_write_eval": False,
                                                "inline_metadata": True,
                                                "grouped_sampler": True,
                                                "stock_sdpa": False}},
                   {"required_environment": _profile_v6(_identity())[
                       "required_environment"]}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="scope"):
            load_b2_research_profile(path, live_identity=_identity(),
                                     context_lengths=(32, 96))


def test_b2_v3_profile_requires_explicit_packed_scope(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile_v3(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=(32, 127)) == profile
    for change in ({"packed_prefill": False}, {"qualified": True},
                   {"price_usable": True}, {"warm_apcv2": True},
                   {"required_environment": {key: "1" for key in FLAGS}},
                   {"context_bounds": {**CONTEXT_BOUNDS, "maximum": 128}}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="scope"):
            load_b2_research_profile(
                path, live_identity=_identity(), context_lengths=(32, 127))
    path.write_text(json.dumps(profile))
    for contexts in ((31, 127), (32, 128), (64, 64)):
        with pytest.raises(ValueError, match="context scope"):
            load_b2_research_profile(
                path, live_identity=_identity(), context_lengths=contexts)


def test_b2_v4_profile_requires_distinct_sustained_scope(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile_v4(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=(32, 96)) == profile
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=(32, 97)) == profile
    with pytest.raises(ValueError, match="context scope"):
        load_b2_research_profile(
            path, live_identity=_identity(), context_lengths=(32, 98))
    for change in ({"max_tokens": 2}, {"max_tokens": 31},
                   {"sustained_decode": False}, {"packed_prefill": False},
                   {"required_environment": {key: "1" for key in
                                             (*FLAGS, PACKED_FLAG)}},
                   {"qualified": True}, {"price_usable": True}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="scope"):
            load_b2_research_profile(
                path, live_identity=_identity(), context_lengths=(32, 96))


def test_sustained_context_98_refused_before_shared_arena_allocation(monkeypatch, tmp_path):
    from mlx2.runtime import paged_price_identity, qwen3_paged_graph_factory

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile_v4(_identity())))
    jobs = _jobs()
    for job, length in zip(jobs, (32, 98)):
        job.native_b2_prompt = (1000,) * length
        job.request["max_tokens"] = SUSTAINED_TOKENS
        job.effective_max_tokens = SUSTAINED_TOKENS
    for key in (*FLAGS, PACKED_FLAG, SUSTAINED_FLAG):
        monkeypatch.setenv(key, "1")
    native = ModuleType("_paged_kv_native")
    native.__file__ = str(tmp_path / "native.so")
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(paged_price_identity, "cached_live_price_identity",
                        lambda *_, **__: _identity())
    allocated = []
    monkeypatch.setattr(qwen3_paged_graph_factory,
                        "create_shared_qwen3_graph_pack",
                        lambda *_, **__: allocated.append(True))
    batch = SimpleNamespace(
        _find_uids=lambda uids: {uids[0]: (0, 0)},
        _unprocessed_sequences=[(None, None, None, None, None, None, False)])
    adapter = SimpleNamespace(identity={"path": str(tmp_path)})
    with pytest.raises(ValueError, match="context scope"):
        serving.install_explicit_native_qwen3_b2_cohort(
            batch, adapter, jobs, lifecycle_lock=RLock(),
            profile_path=profile_path, manifest_path="manifest",
            mlx_wheel_path="wheel")
    assert allocated == []


def test_b2_v5_profile_requires_explicit_vector_rope_scope(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile_v5(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=(32, 96)) == profile
    for change in ({"vector_q1_rope": False},
                   {"required_environment": {key: "1" for key in
                                             (*FLAGS, PACKED_FLAG, SUSTAINED_FLAG)}},
                   {"max_tokens": 2}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="scope"):
            load_b2_research_profile(
                path, live_identity=_identity(), context_lengths=(32, 96))


def test_b2_v6_profile_requires_explicit_direct_fence_scope(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile_v6(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=(32, 96)) == profile
    for change in ({"direct_grouped_fence": False},
                   {"required_environment": {key: "1" for key in
                                             (*FLAGS, PACKED_FLAG, SUSTAINED_FLAG)}},
                   {"max_tokens": 2}, {"qualified": True}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="scope"):
            load_b2_research_profile(
                path, live_identity=_identity(), context_lengths=(32, 96))


def _jobs():
    jobs = tuple(serving.Job({
        "paged_native_qwen3_b2": True, "skip_writing_prefix_cache": True,
        "batch_cohort": {"id": "pair", "size": 2},
        "temperature": 0, "max_tokens": 2,
        "top_p": 1, "top_k": 0, "min_p": 0,
        "repetition_penalty": 1, "presence_penalty": 0,
        "frequency_penalty": 0,
    }) for _ in range(2))
    for index, job in enumerate(jobs):
        job.uid = index + 1
        job.native_b2_prompt = tuple([1000 + index] * (63 + 2 * index))
        job.effective_max_tokens = 2
        job.effective_sampling, _ = resolve_sampling(
            job.request, None, thinking=None)
        assert job.effective_sampling == {
            "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
            "repetition_penalty": 1, "presence_penalty": 0,
            "frequency_penalty": 0,
        }
    return jobs


def test_b2_http_control_is_separate_explicit_and_cohort_bound():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    accepted = dict(base, paged_native_qwen3_b2=True,
                    skip_writing_prefix_cache=True,
                    batch_cohort={"id": "pair", "size": 2},
                    max_tokens=2, temperature=0)
    assert validate_request(accepted)["paged_native_qwen3_b2"] is True
    assert validate_request(dict(accepted, max_tokens=SUSTAINED_TOKENS))[
        "paged_native_qwen3_b2"] is True
    for bad in (dict(accepted, batch_cohort=None),
                dict(accepted, skip_writing_prefix_cache=False),
                dict(accepted, max_tokens=3),
                dict(accepted, temperature=0.1),
                dict(accepted, paged_native_qwen3=True)):
        with pytest.raises(ValueError, match="native B2"):
            validate_request(bad)


def test_b2_profile_binds_identity_and_never_grants_price(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(path, live_identity=_identity()) == profile
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=(63, 65)) == profile
    with pytest.raises(ValueError, match="identity or scope"):
        load_b2_research_profile(
            path, live_identity=_identity(), context_lengths=(64, 65))
    for change in ({"price_usable": True}, {"qualified": True},
                   {"warm_apcv2": True}, {"ordered_context_tokens": [65, 63]},
                   {"identity": {**_identity(), "kernel_sha256": "f" * 64}}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="identity or scope"):
            load_b2_research_profile(path, live_identity=_identity())


@pytest.mark.parametrize("contexts", [(32, 127), (63, 65), (64, 65), (127, 32)])
def test_b2_v2_profile_admits_distinct_bounded_contexts(tmp_path, contexts):
    path = tmp_path / "profile.json"
    profile = _profile_v2(_identity())
    path.write_text(json.dumps(profile))
    assert load_b2_research_profile(
        path, live_identity=_identity(), context_lengths=contexts) == profile


@pytest.mark.parametrize("contexts", [None, (31, 127), (32, 128), (64, 64),
                                      (True, 65), [32, 127]])
def test_b2_v2_profile_refuses_out_of_scope_contexts(tmp_path, contexts):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile_v2(_identity())))
    with pytest.raises(ValueError, match="context scope"):
        load_b2_research_profile(
            path, live_identity=_identity(), context_lengths=contexts)


def test_b2_v2_profile_refuses_qualification_and_geometry_drift(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile_v2(_identity())
    for change in ({"qualified": True}, {"price_usable": True},
                   {"warm_apcv2": True}, {"serving_default": True},
                   {"context_bounds": {**CONTEXT_BOUNDS, "maximum": 128}},
                   {"context_bounds": {**CONTEXT_BOUNDS, "distinct": 1}},
                   {"sampling": {"mode": "greedy", "processors": 0}}):
        path.write_text(json.dumps({**profile, **change}))
        with pytest.raises(ValueError, match="scope"):
            load_b2_research_profile(
                path, live_identity=_identity(), context_lengths=(32, 127))


@pytest.mark.parametrize("contexts", [[32, 127], [63, 65], [127, 32]])
def test_b2_http_gate_accepts_bounded_distinct_contexts(contexts):
    assert http_contexts({"context_tokens": contexts}) == tuple(contexts)


@pytest.mark.parametrize("contexts", [[31, 127], [32, 128], [64, 64],
                                      [True, 127], (32, 127), [32]])
def test_b2_http_gate_refuses_invalid_contexts(contexts):
    with pytest.raises(RuntimeError, match="32-127"):
        http_contexts({"context_tokens": contexts})


def test_b2_http_performance_order_and_ready_event_selection():
    assert HTTP_PERFORMANCE_ORDER == (
        "ordinary", "grouped", "grouped", "ordinary", "ordinary", "grouped")
    events = [
        {"elapsed_ms": 1, "responses": [("a", 1), ("b", 1)]},
        {"elapsed_ms": 3, "responses": [("a", 2), ("b", 2)]},
    ]
    assert _ready_ms(events, {"a", "b"}) == (3, [events[1]])
    with pytest.raises(RuntimeError, match="ready-step"):
        _ready_ms(events[:1], {"a", "b"})
    with pytest.raises(RuntimeError, match="crossed cells"):
        _ready_ms([{"elapsed_ms": 3, "responses": [("a", 2), ("old", 1)]}],
                  {"a", "b"})


def test_b2_sustained_decode_counts_only_real_paired_progress():
    events = [
        {"elapsed_ms": 4, "responses": [("a", 1), ("b", 1)]},
        {"elapsed_ms": 5, "responses": [("a", 2), ("b", 2)]},
        {"elapsed_ms": 7, "responses": [("a", 3), ("b", 3)]},
    ]
    assert _decode_progress(events, {"a", "b"}, 3, paired=True) == {
        "generator_decode_ms": 12, "generated_decode_tokens": 4,
        "generator_decode_tokens_per_second": 4000 / 12,
        "decode_events": 2}
    with pytest.raises(RuntimeError, match="lost a partner"):
        _decode_progress([*events[:2],
                          {"elapsed_ms": 7, "responses": [("a", 3)]}],
                         {"a", "b"}, 3, paired=True)
    with pytest.raises(RuntimeError, match="pinned sustained"):
        _decode_progress(events[:2], {"a", "b"}, 3, paired=True)


def test_b2_v2_greedy_keeps_vendor_filter_defaults_but_refuses_penalties():
    vendor = VendorSampling.single(SamplingDefaults(
        temperature=.6, top_p=.95, top_k=20, repetition_penalty=1,
        source="test vendor"))
    effective, record = resolve_sampling({"temperature": 0}, vendor, thinking=None)
    assert effective["temperature"] == 0
    assert (effective["top_p"], effective["top_k"], effective["min_p"]) == (.95, 20, 0)
    assert set(record["applied"]) >= {"top_p", "top_k"}
    assert effective["repetition_penalty"] == 1
    bad, _ = resolve_sampling({"temperature": 0, "repetition_penalty": 1.1},
                              vendor, thinking=None)
    assert bad["repetition_penalty"] != 1


@pytest.mark.parametrize("fault", [None, "terminal", "cancel", "second_publish"])
def test_b2_packed_queued_prefill_proves_both_before_publication(fault):
    import numpy as np
    from mlx2.adapters.qwen3_paged_candidate import Qwen3PackedCandidate
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
    from mlx2.runtime.paged_native_batch_lifecycle import (
        run_research_native_queued_qwen3_b2_packed)
    from mlx2.runtime.paged_request_transaction import CandidateRequest

    events = []
    model = SimpleNamespace(layers=(object(), object()))
    backend = SimpleNamespace(read_submissions=0, terminal_successes=0,
                              staged_read_spans=[])
    candidate = Qwen3PackedCandidate(model, backend)
    generator = object.__new__(BatchGenerator)
    generator.self_mtp = None
    generator.model = model
    generator._unprocessed_sequences = [
        (None, ((7,) * 32,), None, object(), (), None, (), None, None, None),
        (None, ((8,) * 127,), None, object(), (), None, (), None, None, None),
    ]
    generator._find_uids = lambda uids: {uid: (0, uid - 1) for uid in uids}
    owners = []
    for lane in (1, 2):
        owner = object.__new__(NativeAtomicRequestOwner)
        owner.supported_planes = ("kv",)

        @contextmanager
        def snapshot():
            yield SimpleNamespace(revision="rev", offset=0)

        class Branch:
            layers = (object(), object())

            def prepare(self, rows, lane=lane):
                events.append(("prepare", lane, rows))
                return self

            def publish(self, lane=lane):
                events.append(("publish", lane))
                if fault == "second_publish" and lane == 2:
                    raise RuntimeError("second publish failed")

            def rollback(self, lane=lane):
                events.append(("rollback", lane))

        owner.snapshot = snapshot
        owner.begin = lambda request, lane=lane, branch_type=Branch: (
            events.append(("begin", lane)) or branch_type())
        owners.append(owner)
    cancelled = [False, False]

    def forward(lanes, branches, *, permit_candidate):
        assert permit_candidate and len(lanes) == len(branches) == 2
        assert tuple(len(lane.token_ids) for lane in lanes) == (32, 127)
        events.append(("forward", 2))
        if fault == "terminal":
            raise RuntimeError("terminal failed")
        backend.read_submissions += 2
        backend.terminal_successes += 2
        backend.staged_read_spans.extend((2, 2))
        if fault == "cancel":
            cancelled[1] = True
        return np.arange(159), {"packed_lanes": 2}

    candidate.forward_staged = forward
    requests = (CandidateRequest(1, "rev", 32, ("kv",)),
                CandidateRequest(2, "rev", 127, ("kv",)))
    run = lambda: run_research_native_queued_qwen3_b2_packed(
        generator, RLock(), requests, tuple(owners), candidate,
        research_permit=True,
        cancelled=(lambda: cancelled[0], lambda: cancelled[1]))
    if fault is None:
        probes = run()
        assert tuple(len(probe.private_logits) for probe in probes) == (32, 127)
        assert all(probe.research_executed and probe.native_probe.published
                   for probe in probes)
        assert candidate._b2_prefill_proof == {
            "read_calls": 2, "terminal_successes": 2, "span_counts": [2, 2]}
        assert events.index(("forward", 2)) < events.index(("publish", 1))
        assert events.index(("prepare", 2, 127)) < events.index(("publish", 1))
    else:
        with pytest.raises((RuntimeError, ValueError), match={
            "terminal": "terminal", "cancel": "cancelled",
            "second_publish": "second publish"}[fault]):
            run()
        assert not hasattr(candidate, "_b2_prefill_proof")
        if fault != "second_publish":
            assert not any(event[0] == "publish" for event in events)
        assert ("rollback", 1) in events and ("rollback", 2) in events


def test_b2_preflight_refuses_warm_cancelled_and_wrong_shape_before_native(monkeypatch):
    jobs = _jobs()
    for mutate, reason in (
        (lambda: setattr(jobs[0], "native_b2_cached_tokens", 63), "cold"),
        (lambda: jobs[0].cancelled.set(), "cold"),
        (lambda: setattr(jobs[1], "native_b2_prompt", (1,) * 128), "32-127"),
        (lambda: setattr(jobs[1], "native_b2_prompt", (1,) * 63), "distinct"),
    ):
        jobs = _jobs()
        mutate()
        with pytest.raises(ValueError, match=reason):
            serving.install_explicit_native_qwen3_b2_cohort(
                object(), object(), jobs, lifecycle_lock=RLock(),
                profile_path="unused", manifest_path="unused", mlx_wheel_path="unused")


@pytest.mark.parametrize("mode", ["v1", "v2", "v3", "v3_terminal", "v4", "v5", "v6", "v7"])
def test_b2_second_handoff_failure_retires_uninstalled_owner(monkeypatch, tmp_path, mode):
    from mlx2.runtime import (paged_b2_research_profile,
                              paged_native_batch_lifecycle,
                              paged_price_identity, qwen3_paged_graph_factory)
    jobs = _jobs()
    if mode != "v1":
        jobs[0].native_b2_prompt = (1000,) * 32
        jobs[1].native_b2_prompt = (1001,) * 127
        for job in jobs:
            job.request.pop("top_p")
            job.request.pop("top_k")
            job.request.pop("min_p")
            job.effective_sampling.update(top_p=.95, top_k=20)
    for key in FLAGS:
        monkeypatch.setenv(key, "1")
    if mode.startswith("v3") or mode in ("v4", "v5", "v6", "v7"):
        monkeypatch.setenv(PACKED_FLAG, "1")
    if mode in ("v4", "v5", "v6", "v7"):
        monkeypatch.setenv(SUSTAINED_FLAG, "1")
        for job in jobs:
            job.request["max_tokens"] = SUSTAINED_TOKENS
            job.effective_max_tokens = SUSTAINED_TOKENS
    if mode == "v5":
        monkeypatch.setenv(VECTOR_ROPE_FLAG, "1")
    if mode in ("v6", "v7"):
        monkeypatch.setenv(DIRECT_FENCE_FLAG, "1")
        if mode == "v7":
            for key in COMBINED_FLAGS:
                monkeypatch.setenv(key, "0" if key == DEFERRED_WRITE_EVAL_FLAG else "1")
    monkeypatch.setenv(STRIPES_FLAG, "4")
    monkeypatch.setenv(STOCK_SDPA_FLAG, "0")
    native = ModuleType("_paged_kv_native")
    native.__file__ = str(tmp_path / "native.so")
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(paged_price_identity, "cached_live_price_identity",
                        lambda *_, **__: _identity())
    monkeypatch.setattr(paged_b2_research_profile, "load_b2_research_profile",
                        lambda *_, **__: {
                            "v1": _profile, "v2": _profile_v2,
                            "v3": _profile_v3,
                            "v3_terminal": _profile_v3,
                            "v4": _profile_v4, "v5": _profile_v5,
                            "v6": _profile_v6,
                            "v7": _profile_v7}[mode](_identity()))
    closed = []
    owners = tuple(SimpleNamespace(fully_retired=True,
                                   close=lambda index=index: closed.append(index))
                   for index in range(2))
    backend = SimpleNamespace(writer=SimpleNamespace())
    candidate = SimpleNamespace(backend=backend)
    monkeypatch.setattr(qwen3_paged_graph_factory, "create_shared_qwen3_graph_pack",
                        lambda *_, **__: (owners, candidate))
    monkeypatch.setattr(paged_native_batch_lifecycle,
                        "run_research_native_queued_qwen3",
                        lambda *_, **__: SimpleNamespace(
                            reason="research_executed", research_executed=True))
    monkeypatch.setattr(paged_native_batch_lifecycle,
                        "run_research_native_queued_qwen3_b2_packed",
                        lambda *_, **__: (_ for _ in ()).throw(
                            RuntimeError("paired terminal failed"))
                        if mode == "v3_terminal" else (SimpleNamespace(
                            reason="research_executed", research_executed=True),) * 2)
    monkeypatch.setattr(paged_native_batch_lifecycle,
                        "prepare_queued_native_first_response",
                        lambda *_, **__: object())
    monkeypatch.setattr("mlx2.runtime.paged_native_retirement.reap_native_request_owner",
                        lambda *_, **__: None)
    calls = []
    def install(*_, **__):
        calls.append(1)
        return {"reason": "native_installed", "selected": True} if len(calls) == 1 else {
            "reason": "cancelled", "selected": False}
    removed = []
    batch = SimpleNamespace(
        install_native_queued=install,
        remove=lambda uids: removed.append(tuple(uids)),
        _find_uids=lambda uids: {uids[0]: (0, uids[0] - 1)},
        _unprocessed_sequences=[(None, None, None, None, None, None, ())] * 2)
    adapter = SimpleNamespace(identity={"path": str(tmp_path), "fingerprint": "rev"})
    with pytest.raises((RuntimeError if mode == "v3_terminal" else ValueError),
                       match=("paired terminal" if mode == "v3_terminal"
                              else "handoff refused: cancelled")):
        serving.install_explicit_native_qwen3_b2_cohort(
            batch, adapter, jobs, lifecycle_lock=RLock(),
            profile_path="profile", manifest_path="manifest", mlx_wheel_path="wheel")
    assert calls == ([] if mode == "v3_terminal" else [1, 1])
    assert removed == ([] if mode == "v3_terminal" else [(1,)])
    assert closed == ([0, 1] if mode == "v3_terminal" else [1])
    assert not serving._NATIVE_ADMISSION_ORPHANS


def test_b2_outer_cohort_failure_finishes_both_without_response():
    jobs = _jobs()
    cohort = serving.PublishedCohort(jobs)
    active = {job.uid: job for job in jobs}
    removed, finished = [], []
    batch = SimpleNamespace(remove=lambda uids: removed.append(tuple(uids)))
    engine = object.__new__(serving.ServingEngine)
    from collections import Counter, deque
    engine.counts = Counter()
    engine.queued_jobs = 0
    engine.lock = RLock()
    engine._finish = lambda job, event: finished.append((job.id, event))
    published = deque()
    engine._fail_attaching_cohort(
        batch, active, published, cohort,
        {"error": "second handoff refused", "status": 400})
    assert removed == [(1, 2)]
    assert active == {}
    assert {job_id for job_id, _ in finished} == {job.id for job in jobs}
    assert all(event["status"] == 400 for _, event in finished)
    assert engine.counts["batch_cohort_jobs_failed_closed"] == 2
