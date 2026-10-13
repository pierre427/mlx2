"""L2-01 (sweep 2026-10-12): the Qwen native-continuation provider bound to a
hybrid Qwen35PagedCandidate must construct its continuation (e3c19168e
required the exact Qwen3PackedCandidate type and refused every hybrid lane)."""

from types import SimpleNamespace

import mlx.core as mx
import pytest


def _hybrid_candidate():
    from mlx2.adapters.qwen35_paged_candidate import Qwen35PagedCandidate

    model = SimpleNamespace(layers=[object()])
    candidate = object.__new__(Qwen35PagedCandidate)
    candidate.model = model
    candidate.backend = SimpleNamespace(read_submissions=0, terminal_successes=0)
    candidate.state_planes = ("kv", "gdn")
    candidate.bootstrap_generation = 0
    candidate.supports_singleton = True
    candidate.owns_physical_dispatch_proof = True
    candidate.native_layer_count = 1
    candidate.logical_layer_count = 4
    return model, candidate


def _continuation_kwargs(candidate, planes=("kv", "gdn")):
    from mlx2.runtime.generate import StopSequenceMatcher
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner

    owner = object.__new__(NativeAtomicRequestOwner)
    owner.supported_planes = planes
    owner._enabled = True
    return {
        "uid": 1, "revision": "rev", "prompt_tokens": (7,),
        "first_logits": mx.zeros((4,)), "owner": owner, "candidate": candidate,
        "maximum": 4, "sampler": lambda x: x, "processors": [],
        "matcher": StopSequenceMatcher(), "lane_rng": None, "research_only": False,
    }


def test_l2_01_provider_creates_hybrid_native_continuation():
    from mlx2.adapters.native_continuation import bind_qwen_native_continuation_provider
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation
    from mlx2.runtime.paged_native_contract import supports_native_checkpoint_candidate

    model, candidate = _hybrid_candidate()
    assert supports_native_checkpoint_candidate(candidate)
    provider = bind_qwen_native_continuation_provider(
        candidate, "rev", model, "native_hybrid_paged_b2")
    continuation = provider.create_continuation(**_continuation_kwargs(candidate))
    assert type(continuation) is NativeQwen3Continuation
    assert continuation.candidate is candidate


def test_l2_01_hybrid_candidate_without_checkpoint_capability_is_refused():
    from mlx2.adapters.native_continuation import bind_qwen_native_continuation_provider

    model, candidate = _hybrid_candidate()
    candidate.owns_physical_dispatch_proof = False
    provider = bind_qwen_native_continuation_provider(
        candidate, "rev", model, "native_hybrid_paged_b2")
    with pytest.raises(ValueError, match="incomplete native continuation state"):
        provider.create_continuation(**_continuation_kwargs(candidate))
