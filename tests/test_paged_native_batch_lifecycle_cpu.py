"""CPU checks for the queued native execution seam and its serving boundary."""

import json
import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import numpy as np

from mlx2.adapters.qwen3_paged_candidate import Qwen3PackedCandidate
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
from mlx2.runtime.paged_native_batch_lifecycle import (
    NativeBatchProbeResult, prepare_queued_native_first_response,
    probe_queued_native_qwen3,
)
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner
from mlx2.runtime.paged_pack_price import PACK_STEP_UNIT, evidence_sha256, load_price
from mlx2.runtime.paged_pack_scheduler import PrefillOffer, PrefillOption
from mlx2.runtime.paged_request_route import RouteCapability
from mlx2.runtime.paged_request_transaction import CandidateRequest
from test_qwen3_paged_native_backend_cpu import (
    FakeModel, PROFILE, install_read_callbacks, setup,
)


def _uninitialized_owner():
    # These tests stop at admission, before any owner state is read or written.
    owner = object.__new__(NativeAtomicRequestOwner)
    owner.supported_planes = ("kv",)
    owner._enabled = True
    owner.snapshot = lambda: nullcontext(SimpleNamespace(offset=0, revision="r1"))
    return owner


def test_warm_apcv2_queued_probe_uses_only_suffix_and_prepares_full_history(monkeypatch):
    from mlx2.runtime import paged_native_batch_lifecycle as lifecycle

    generator = BatchGenerator(None)
    owner = _uninitialized_owner()
    public = SimpleNamespace(offset=2, revision="r1", generation=0)
    owner.snapshot = lambda: nullcontext(public)
    candidate = Qwen3PackedCandidate(None, None)
    candidate.forward = lambda lanes, **_: (np.array([[0., 1.]]), {})

    def coordinate(request, _owner, _capability, _reserved, _offers, **kwargs):
        assert request.proposed_rows == 1
        assert kwargs["context_tokens"] == (3,)
        accepted = kwargs["run_candidate"](SimpleNamespace(layers=()))
        public.offset, public.generation = 3, 1
        return SimpleNamespace(reason="accepted", published=True,
                               accepted_rows=accepted, lane_id=request.lane_id,
                               revision="r1")

    monkeypatch.setattr(lifecycle, "coordinate_paged_request", coordinate)
    try:
        uid = generator.insert([[3]], caches=[[]], all_tokens=[[1, 2]])[0]
        probe = probe_queued_native_qwen3(
            generator, threading.RLock(), CandidateRequest(uid, "r1", 1, ("kv",)),
            owner, candidate, RouteCapability(True, False, ("kv",), "p"), (),
            (PrefillOffer(uid, (PrefillOption(1, 0),)),), price=None,
            live_identity={}, context_tokens=(3,), row_capacity=1,
            free_pages=1, permit_native_probe=True)
        assert type(probe) is NativeBatchProbeResult
        prepared = prepare_queued_native_first_response(
            generator, threading.RLock(), owner, probe)
        assert prepared.reason == "native_continuation_not_installed"
        assert prepared.prompt_tokens == (1, 2, 3)
        assert prepared.first_token_logits.tolist() == [0., 1.]
    finally:
        generator.close()


def test_native_probe_is_default_off_and_keeps_ordinary_queue():
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        owner = _uninitialized_owner()
        request = CandidateRequest(uid, "r1", 3, ("kv",))
        result = probe_queued_native_qwen3(
            generator, threading.RLock(), request, owner,
            Qwen3PackedCandidate(None, None),
            RouteCapability(True, False, ("kv",), "profile"), (),
            (PrefillOffer(uid, (PrefillOption(3, 0),)),),
            price=None, live_identity={}, context_tokens=(3,),
            row_capacity=3, free_pages=1,
        )
        assert result.reason == "paged_native_probe_disabled"
        assert result.private_logits is None and result.native_probe is None
        assert not result.serving_selected and not result.serving_observed_used
        assert generator._find_uids((uid,))[uid][0] == 0
    finally:
        generator.close()


def test_native_probe_refuses_unsourced_price_and_companion_planes():
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        owner = _uninitialized_owner()
        candidate = Qwen3PackedCandidate(None, None)
        kwargs = dict(price=None, live_identity={}, context_tokens=(3,),
                      row_capacity=3, free_pages=1, permit_native_probe=True)
        offers = (PrefillOffer(uid, (PrefillOption(3, 0),)),)
        result = probe_queued_native_qwen3(
            generator, threading.RLock(), CandidateRequest(uid, "r1", 3, ("kv",)),
            owner, candidate, RouteCapability(True, False, ("kv",), "profile"),
            (), offers, **kwargs)
        assert result.reason == "source_bound_price_missing"
        assert result.native_probe is not None and not result.native_probe.selected
        assert result.private_logits is None
        companion = probe_queued_native_qwen3(
            generator, threading.RLock(), CandidateRequest(uid, "r1", 3, ("kv", "gdn")),
            owner, candidate, RouteCapability(True, False, ("kv", "gdn"), "profile"),
            (), offers, **kwargs)
        assert companion.reason == "unsupported_companion_planes"
        assert companion.native_probe is None
        assert generator._find_uids((uid,))[uid][0] == 0
    finally:
        generator.close()


def test_native_probe_refuses_missing_or_advanced_uid_before_pricing():
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        owner = _uninitialized_owner()
        request = CandidateRequest(uid, "r1", 3, ("kv",))
        generator.remove([uid])
        result = probe_queued_native_qwen3(
            generator, threading.RLock(), request, owner,
            Qwen3PackedCandidate(None, None),
            RouteCapability(True, False, ("kv",), "profile"), (),
            (PrefillOffer(uid, (PrefillOption(3, 0),)),),
            price=None, live_identity={}, context_tokens=(3,),
            row_capacity=3, free_pages=1, permit_native_probe=True,
        )
        assert result.reason == "request_not_live"
    finally:
        generator.close()


def test_native_probe_refuses_self_mtp_generator_before_native_work():
    generator = BatchGenerator(None, self_mtp={"persistent": True})
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        result = probe_queued_native_qwen3(
            generator, threading.RLock(), CandidateRequest(uid, "r1", 3, ("kv",)),
            _uninitialized_owner(), Qwen3PackedCandidate(None, None),
            RouteCapability(True, False, ("kv",), "profile"), (),
            (PrefillOffer(uid, (PrefillOption(3, 0),)),),
            price=None, live_identity={}, context_tokens=(3,),
            row_capacity=3, free_pages=1, permit_native_probe=True,
        )
        assert result.reason == "unsupported_generator_route"
        assert result.native_probe is None
        assert generator._find_uids((uid,))[uid][0] == 0
    finally:
        generator.close()


@pytest.mark.parametrize("accepted", (0, 1, 3))
def test_native_probe_runs_private_forward_without_serving_handoff(monkeypatch, tmp_path, accepted):
    # Synthetic price data is confined to this protocol test. Real admission
    # requires a source-bound complete-request measurement for the live host.
    identity = {"host": "test-host", "hardware": "cpu",
                "artifact_sha256": "a" * 64, "source_commit": "b" * 40,
                "source_tree_sha256": "c" * 64, "mlx_wheel_version": "test",
                "mlx_wheel_sha256": "d" * 64, "kernel_sha256": "e" * 64}
    price_file = tmp_path / "synthetic-protocol-price.json"
    plain = {"request_inserted": True, "admitted": True,
             "cache_transaction_published": True, "response_emitted": True,
             "sampled_output": True, "synchronized": True,
             "request_removed": True, "request_state_released": True,
             "route_receipt": {"route": "ordinary"},
             "ordinary_model_forward_calls": 1, "output_token_id": 42}
    paged = {**plain, "route_receipt": {"route": "native_qwen3_paged",
                                           "selected": True, "observed_used": True},
             "ordinary_model_forward_calls": 0, "model_layers": 1,
             "paged_read_calls": 1, "terminal_successes": 1,
             "pending_native_epochs": 0, "retained_pages": 0}
    step = {"scope": PACK_STEP_UNIT, "reserved": [], "prefill_rows": 3,
            "context_tokens": [3], "active_lane_count": 1,
            "generator_step": True, "sampled_response": True,
            "synchronized": True}
    data = {
        "schema": "mlx2.paged-pack-price.v1", "status": "measured",
        "measurement_scope": "complete_request", "gpu_executed": True,
        "price_unit": PACK_STEP_UNIT,
        "gpuq_owner": {"session": "test", "lease_id": "test-1", "pid": 1},
        "measured_route": "native_qwen3_paged",
        "identity": identity, "context_tokens": [3], "profile_id": "test-only",
        "cases": [
            {"reserved": [], "prefill_rows": 3, "pack_step_ms": [1, 1, 1],
             "ordinary_pack_step_ms": [1, 1, 1],
             "complete_native_request_ms": [2, 2, 2],
             "ordinary_request_ms": [2, 2, 2],
             "request_proofs": [
                 {"ordinary": {**plain, "scheduler_step": step,
                               "scheduler_step_ms": 1},
                  "paged": {**paged, "scheduler_step": step,
                            "scheduler_step_ms": 1}} for _ in range(3)],
             "kernel_engagement": {"paged_read_calls": 3, "terminal_successes": 3}}
        ],
    }
    data["complete_request_evidence_sha256"] = evidence_sha256(data)
    price_file.write_text(json.dumps(data))
    price = load_price(price_file, live_identity=identity, context_tokens=(3,))
    _, arena, writer, backend = setup(monkeypatch)
    install_read_callbacks(monkeypatch, arena)
    owner = NativeAtomicRequestOwner(
        "r1", tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                    for _ in range(2)), {}, supported_planes=("kv",), enabled=True)
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        result = probe_queued_native_qwen3(
            generator, threading.RLock(), CandidateRequest(uid, "r1", 3, ("kv",)),
            owner, Qwen3PackedCandidate(FakeModel(), backend),
            RouteCapability(True, False, ("kv",), "test-only"), (),
            (PrefillOffer(uid, (PrefillOption(3, 1),)),),
            price=price, live_identity=identity, context_tokens=(3,),
            row_capacity=3, free_pages=3, permit_native_probe=True,
            accept_prefix=lambda _logits, _tokens: accepted,
        )
        assert result.native_probe.published and result.native_probe.accepted_rows == accepted
        assert result.private_logits.shape == (accepted,)
        with owner.snapshot() as public:
            assert public.offset == accepted
        assert not result.serving_selected and not result.serving_observed_used
        assert generator._find_uids((uid,))[uid][0] == 0
        lock = threading.RLock()
        prepared = prepare_queued_native_first_response(generator, lock, owner, result)
        assert prepared.uid == uid
        assert prepared.serving_selected is False
        assert prepared.serving_observed_used is False
        if accepted == 3:
            assert prepared.reason == "native_continuation_not_installed"
            assert prepared.prompt_tokens == (7, 8, 9)
            assert prepared.first_token_logits is not None
            assert prepared.generation == 1
            drifted = prepare_queued_native_first_response(
                generator, lock, owner,
                replace(result, native_probe=replace(result.native_probe, revision="r2")))
            assert drifted.reason == "native_state_drifted"
            assert drifted.first_token_logits is None
        else:
            assert prepared.reason == "full_native_prompt_required"
            assert prepared.first_token_logits is None
        assert generator._find_uids((uid,))[uid][0] == 0
        cancelled = prepare_queued_native_first_response(
            generator, lock, owner, result, cancelled=lambda: True)
        assert cancelled.reason == "cancelled"
        assert cancelled.first_token_logits is None
        generator.remove([uid])
        removed = prepare_queued_native_first_response(generator, lock, owner, result)
        assert removed.reason == "request_not_live"
        assert removed.first_token_logits is None
    finally:
        generator.close()
