"""Default-off native Qwen3 candidate bound to one queued BatchGenerator UID.

This is an execution probe, not a serving route. The generator still owns its
ordinary cache, sampler, and response lifecycle. No token or native page table
is handed to it, and callers must never report this receipt as serving use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ..adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
from .generate import BatchGenerator
from .paged_native_atomic_owner import NativeAtomicRequestOwner
from .paged_pack_price import (
    MeasuredPackPrice, ResearchCalibratedPrice, ResearchWarmCalibratedPrice,
)
from .paged_pack_scheduler import PrefillOffer, ReservedRows
from .paged_request_route import PagedRouteReceipt, RouteCapability, coordinate_paged_request
from .paged_request_transaction import CandidateRequest


@dataclass(frozen=True)
class NativeBatchProbeResult:
    uid: int
    reason: str
    generator_stage: str | None
    serving_selected: bool
    serving_observed_used: bool
    native_probe: PagedRouteReceipt | None
    private_logits: Any | None
    research_executed: bool = False


@dataclass(frozen=True)
class NativeQueuedHandoffPreparation:
    """Private first-token inputs, still outside the serving response path.

    A native continuation must own subsequent decode, sampler/RNG, stop state,
    APCv2, and cancellation before these logits may become a response.
    """

    uid: int
    revision: str
    generation: int | None
    prompt_tokens: tuple[int, ...]
    first_token_logits: Any | None
    reason: str
    serving_selected: bool = False
    serving_observed_used: bool = False


_STAGES = ("queued", "prefill", "decode", "plain_fallback")


def probe_queued_native_qwen3(
    generator: BatchGenerator,
    lifecycle_lock: RLock,
    request: CandidateRequest,
    owner: NativeAtomicRequestOwner,
    candidate: Qwen3PackedCandidate,
    capability: RouteCapability,
    reserved: tuple[ReservedRows, ...],
    offers: tuple[PrefillOffer, ...],
    *,
    price: MeasuredPackPrice | ResearchCalibratedPrice | ResearchWarmCalibratedPrice | None,
    live_identity: dict[str, str],
    context_tokens: tuple[int, ...],
    row_capacity: int,
    free_pages: int,
    permit_native_probe: bool = False,
    permit_research_calibration: bool = False,
    accept_prefix: Callable[[Any, tuple[int, ...]], int] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> NativeBatchProbeResult:
    """Forward a pristine queued prompt through a private native branch.

    The caller must hold the same lock around generator insert, next, and
    remove. The result keeps logits private: it never samples, advances the
    generator, or issues a serving response. The native owner is caller-owned.
    """
    if type(generator) is not BatchGenerator or type(request) is not CandidateRequest:
        raise TypeError("a BatchGenerator and CandidateRequest are required")
    if type(owner) is not NativeAtomicRequestOwner or type(candidate) is not Qwen3PackedCandidate:
        raise TypeError("a native owner and Qwen3 packed candidate are required")
    if not hasattr(lifecycle_lock, "__enter__") or not hasattr(lifecycle_lock, "__exit__"):
        raise TypeError("a lifecycle lock is required")

    def result(reason: str, stage: str | None = None,
               probe: PagedRouteReceipt | None = None,
               logits: Any | None = None) -> NativeBatchProbeResult:
        return NativeBatchProbeResult(request.lane_id, reason, stage, False, False,
                                      probe, logits)

    if not permit_native_probe:
        return result("paged_native_probe_disabled")
    if generator.self_mtp is not None:
        return result("unsupported_generator_route")
    # Dense Qwen3 has only KV state in this seam. Other companion planes need
    # an adapter that stages their exact accepted prefixes before publication.
    if request.planes != ("kv",) or capability.required_planes != ("kv",) or owner.supported_planes != ("kv",):
        return result("unsupported_companion_planes")
    with lifecycle_lock:
        found = generator._find_uids((request.lane_id,)).get(request.lane_id)
        if found is None:
            return result("request_not_live")
        stage = _STAGES[found[0]]
        if cancelled():
            return result("cancelled", stage)
        if stage != "queued":
            return result("generator_stage_not_supported", stage)
        sequence = generator._unprocessed_sequences[found[1]]
        segments, ordinary_cache, history, prefill_input = sequence[1], sequence[3], sequence[4], sequence[9]
        tokens = tuple(token for part in segments for token in part)
        prefix = tuple(history)
        full_prompt = prefix + tokens
        with owner.snapshot() as public:
            native_offset = public.offset
        if (prefill_input is not None or not tokens or
                any(type(token) is not int or token < 0 for token in full_prompt) or
                native_offset != len(prefix)):
            return result("pristine_queued_prompt_required", stage)
        # The existing ordinary cache is retained only for the reference path;
        # it is never modified or replaced by this native probe.
        if ordinary_cache is None or request.proposed_rows != len(tokens) or context_tokens != (len(full_prompt),):
            return result("live_context_mismatch", stage)
        if (reserved or len(offers) != 1 or offers[0].lane_id != request.lane_id or
                not any(option.rows == len(tokens) for option in offers[0].options)):
            return result("queued_prefill_shape_mismatch", stage)

        produced: dict[str, Any] = {}

        def run_private(branch) -> int:
            logits, _forward_receipt = candidate.forward(
                (PackedLane(tokens, branch.layers),), permit_candidate=True,
                requested_capabilities=frozenset({"text"}), atomic_branch=branch,
            )
            if getattr(logits, "shape", None) is None or logits.shape[0] != len(tokens):
                raise ValueError("native logits do not cover every proposed row")
            produced["logits"] = logits
            accepted = len(tokens) if accept_prefix is None else accept_prefix(logits, tokens)
            if type(accepted) is not int or not 0 <= accepted <= len(tokens):
                raise ValueError("accepted rows must be an integer prefix")
            produced["accepted"] = accepted
            return accepted

        probe = coordinate_paged_request(
            request, owner, capability, reserved, offers, price=price,
            live_identity=live_identity, context_tokens=context_tokens,
            row_capacity=row_capacity, free_pages=free_pages,
            run_candidate=run_private, permit_candidate=True,
            permit_research_calibration=permit_research_calibration,
            cancelled=lambda: cancelled() or request.lane_id not in generator._find_uids((request.lane_id,)),
        )
        return result(probe.reason, stage, probe,
                      produced["logits"][:produced["accepted"]] if probe.published else None)


def prepare_queued_native_first_response(
    generator: BatchGenerator,
    lifecycle_lock: RLock,
    owner: NativeAtomicRequestOwner,
    probe: NativeBatchProbeResult,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> NativeQueuedHandoffPreparation:
    """Bind a proven full native prompt to its still-queued serving UID.

    This validates the narrow handoff boundary and retains only private
    first-token logits. It never invokes a sampler, removes an ordinary lane,
    installs a native continuation, or emits a ``GenerationBatch.Response``.
    Those operations need a combined commit in ``BatchGenerator``.
    """
    if type(generator) is not BatchGenerator or type(owner) is not NativeAtomicRequestOwner:
        raise TypeError("a BatchGenerator and native owner are required")
    if type(probe) is not NativeBatchProbeResult:
        raise TypeError("a native probe result is required")
    if not hasattr(lifecycle_lock, "__enter__") or not hasattr(lifecycle_lock, "__exit__"):
        raise TypeError("a lifecycle lock is required")

    def refusal(reason: str, *, revision: str = "", generation: int | None = None,
                tokens: tuple[int, ...] = ()) -> NativeQueuedHandoffPreparation:
        return NativeQueuedHandoffPreparation(probe.uid, revision, generation,
                                              tokens, None, reason)

    with lifecycle_lock:
        if generator.self_mtp is not None:
            return refusal("unsupported_generator_route")
        if cancelled():
            return refusal("cancelled")
        found = generator._find_uids((probe.uid,)).get(probe.uid)
        if found is None:
            return refusal("request_not_live")
        if found[0] != 0:
            return refusal("generator_stage_not_supported")
        sequence = generator._unprocessed_sequences[found[1]]
        suffix = tuple(token for part in sequence[1] for token in part)
        tokens = tuple(sequence[4]) + suffix
        if (sequence[9] is not None or not suffix or
                any(type(token) is not int or token < 0 for token in tokens)):
            return refusal("pristine_queued_prompt_required", tokens=tokens)
        native = probe.native_probe
        if (native is None or not native.published or probe.private_logits is None or
                native.lane_id != probe.uid or native.accepted_rows != len(suffix)):
            return refusal("full_native_prompt_required", tokens=tokens)
        if getattr(probe.private_logits, "shape", None) is None or len(probe.private_logits) != len(suffix):
            return refusal("native_logits_shape_mismatch", tokens=tokens)
        with owner.snapshot() as public:
            if (public.revision != native.revision or
                    public.offset != len(tokens) or public.generation < 1):
                return refusal("native_state_drifted", revision=public.revision,
                               generation=public.generation, tokens=tokens)
            revision, generation = public.revision, public.generation
        if cancelled() or probe.uid not in generator._find_uids((probe.uid,)):
            return refusal("cancelled", revision=revision,
                           generation=generation, tokens=tokens)
        return NativeQueuedHandoffPreparation(
            probe.uid, revision, generation, tokens,
            probe.private_logits[-1], "native_continuation_not_installed",
        )


def run_research_native_queued_qwen3(
    generator: BatchGenerator, lifecycle_lock: RLock,
    request: CandidateRequest, owner: NativeAtomicRequestOwner,
    candidate: Qwen3PackedCandidate, *, research_permit: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> NativeBatchProbeResult:
    """Run the same native prompt transaction for offline price calibration.

    No pack price, serving route, or request response is selected here. The
    owner is request-private; the caller must close and reap it after use.
    Only actual backend submission and terminal-success counters authorize a
    research-executed receipt. This receipt cannot pass `load_price` itself.
    """
    if not research_permit:
        return NativeBatchProbeResult(request.lane_id, "research_permit_required",
                                      None, False, False, None, None)
    if (type(generator) is not BatchGenerator or type(request) is not CandidateRequest or
            type(owner) is not NativeAtomicRequestOwner or
            type(candidate) is not Qwen3PackedCandidate or
            not hasattr(lifecycle_lock, "__enter__") or
            not hasattr(lifecycle_lock, "__exit__")):
        raise TypeError("complete offline research inputs are required")
    with lifecycle_lock:
        found = generator._find_uids((request.lane_id,)).get(request.lane_id)
        if generator.self_mtp is not None or request.planes != ("kv",) or owner.supported_planes != ("kv",):
            reason = "unsupported_generator_or_planes"
        elif cancelled():
            reason = "cancelled"
        elif found is None or found[0] != 0:
            reason = "pristine_queued_prompt_required"
        else:
            sequence = generator._unprocessed_sequences[found[1]]
            tokens = tuple(token for part in sequence[1] for token in part)
            prefix = tuple(sequence[4])
            with owner.snapshot() as public:
                public_ok = (public.revision == request.revision and
                             public.offset == len(prefix))
            if (not public_ok or sequence[9] is not None or
                    not tokens or request.proposed_rows != len(tokens) or
                    any(type(token) is not int or token < 0 for token in prefix + tokens) or
                    candidate.model is not generator.model):
                reason = "pristine_queued_prompt_required"
            else:
                branch = owner.begin(request)
                prepared = None
                try:
                    backend = candidate.backend
                    before_reads = int(getattr(backend, "read_submissions", 0))
                    before_terminals = int(getattr(backend, "terminal_successes", 0))
                    logits, _forward = candidate.forward(
                        (PackedLane(tokens, branch.layers),), permit_candidate=True,
                        requested_capabilities=frozenset({"text"}), atomic_branch=branch)
                    submissions = getattr(backend, "read_submissions", 0)
                    terminals = getattr(backend, "terminal_successes", 0)
                    if (getattr(logits, "shape", None) is None or
                            logits.shape[0] != len(tokens) or
                            type(submissions) is not int or type(terminals) is not int or
                            submissions - before_reads != len(candidate.model.layers) or
                            terminals - before_terminals != len(candidate.model.layers)):
                        raise RuntimeError("research prompt lacks real terminal native reads")
                    if cancelled():
                        reason = "cancelled"
                    else:
                        prepared = branch.prepare(len(tokens))
                        if cancelled():
                            reason = "cancelled"
                        else:
                            prepared.publish()
                            receipt = PagedRouteReceipt(
                                request.lane_id, request.revision, True, False,
                                False, False, True, "research_executed", None,
                                len(tokens), None, None)
                            return NativeBatchProbeResult(
                                request.lane_id, "research_executed", "queued",
                                False, False, receipt, logits, True)
                except BaseException:
                    if prepared is not None:
                        prepared.rollback()
                    else:
                        branch.rollback()
                    raise
                if prepared is not None:
                    prepared.rollback()
                else:
                    branch.rollback()
        return NativeBatchProbeResult(request.lane_id, reason,
                                      "queued" if found and found[0] == 0 else None,
                                      False, False, None, None)


def run_research_native_queued_qwen3_b2_packed(
    generator: BatchGenerator, lifecycle_lock: RLock,
    requests: tuple[CandidateRequest, CandidateRequest],
    owners: tuple[NativeAtomicRequestOwner, NativeAtomicRequestOwner],
    candidate: Qwen3PackedCandidate, *, research_permit: bool = False,
    cancelled: tuple[Callable[[], bool], Callable[[], bool]],
) -> tuple[NativeBatchProbeResult, NativeBatchProbeResult]:
    """Prove two cold prompts with one staged graph before either response.

    This is a source-bound serving admission primitive, not a route selector.
    Its caller owns both request owners and retires both if any later
    installation fails. No generator response can run under lifecycle_lock.
    """
    if (not research_permit or type(generator) is not BatchGenerator or
            type(requests) is not tuple or len(requests) != 2 or
            any(type(request) is not CandidateRequest for request in requests) or
            type(owners) is not tuple or len(owners) != 2 or
            any(type(owner) is not NativeAtomicRequestOwner for owner in owners) or
            owners[0] is owners[1] or type(candidate) is not Qwen3PackedCandidate or
            type(cancelled) is not tuple or len(cancelled) != 2 or
            any(not callable(check) for check in cancelled)):
        raise ValueError("paired native prompt requires two exact private requests")
    if not hasattr(lifecycle_lock, "__enter__") or not hasattr(lifecycle_lock, "__exit__"):
        raise TypeError("paired native prompt requires a lifecycle lock")
    with lifecycle_lock:
        if generator.self_mtp is not None or candidate.model is not generator.model:
            raise ValueError("paired native prompt requires the live plain Qwen3 model")
        if requests[0].lane_id == requests[1].lane_id:
            raise ValueError("paired native prompt UIDs must differ")
        tokens_by_lane = []
        for request, owner, is_cancelled in zip(requests, owners, cancelled):
            found = generator._find_uids((request.lane_id,)).get(request.lane_id)
            if (is_cancelled() or found is None or found[0] != 0 or
                    request.planes != ("kv",) or
                    owner.supported_planes != ("kv",)):
                raise ValueError("paired native prompt requires pristine queued KV lanes")
            sequence = generator._unprocessed_sequences[found[1]]
            tokens = tuple(token for part in sequence[1] for token in part)
            prefix = tuple(sequence[4])
            with owner.snapshot() as public:
                valid = (public.revision == request.revision and public.offset == 0)
            if (not valid or prefix or sequence[3] is None or sequence[9] is not None or
                    not tokens or len(tokens) != request.proposed_rows or
                    any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError("paired native prompt no longer matches cold queued tokens")
            tokens_by_lane.append(tokens)
        branches, prepared = [], []
        try:
            for request, owner in zip(requests, owners):
                branches.append(owner.begin(request))
            backend = candidate.backend
            depth = len(candidate.model.layers)
            before_reads = backend.read_submissions
            before_terminals = backend.terminal_successes
            before_spans = len(backend.staged_read_spans)
            logits, receipt = candidate.forward_staged(
                tuple(PackedLane(tokens, branch.layers)
                      for tokens, branch in zip(tokens_by_lane, branches)),
                tuple(branches), permit_candidate=True)
            if (receipt.get("packed_lanes") != 2 or
                    tuple(logits.shape[:1]) != (sum(map(len, tokens_by_lane)),) or
                    backend.read_submissions - before_reads != depth or
                    backend.terminal_successes - before_terminals != depth or
                    tuple(backend.staged_read_spans[before_spans:]) != (2,) * depth):
                raise RuntimeError("paired native prompt lacks one two-span terminal per layer")
            if any(check() for check in cancelled):
                raise ValueError("paired native prompt cancelled before publication")
            for branch, tokens in zip(branches, tokens_by_lane):
                prepared.append(branch.prepare(len(tokens)))
            if any(check() for check in cancelled):
                raise ValueError("paired native prompt cancelled before publication")
            for state in prepared:
                state.publish()
            candidate._b2_prefill_proof = {
                "read_calls": depth, "terminal_successes": depth,
                "span_counts": [2] * depth,
            }
            probes = []
            begin = 0
            for request, tokens in zip(requests, tokens_by_lane):
                end = begin + len(tokens)
                route = PagedRouteReceipt(
                    request.lane_id, request.revision, True, False, False,
                    False, True, "research_executed", None, len(tokens), None, None)
                probes.append(NativeBatchProbeResult(
                    request.lane_id, "research_executed", "queued", False,
                    False, route, logits[begin:end], True))
                begin = end
            return tuple(probes)
        except BaseException:
            for state in prepared:
                state.rollback()
            for branch in branches:
                branch.rollback()
            raise


__all__ = ["NativeBatchProbeResult", "NativeQueuedHandoffPreparation",
           "probe_queued_native_qwen3", "prepare_queued_native_first_response",
           "run_research_native_queued_qwen3",
           "run_research_native_queued_qwen3_b2_packed"]
