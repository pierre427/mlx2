"""Default-off, request-owned native Qwen3 decode continuation.

Only an explicitly installed queued lane may call this. The owning
BatchGenerator serializes step, cancellation, and native owner retirement.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

from ..adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
from .generate import GenerationBatch, StopSequenceMatcher, _invalid_output_reason
from .paged_native_atomic_owner import NativeAtomicRequestOwner
from .paged_request_transaction import CandidateRequest
from .paged_native_contract import native_layer_count, supports_native_checkpoint_candidate


_NO_GRAPH_LOGITS = object()


class NativeQwen3Continuation:
    """One Qwen3 lane whose accepted KV is owned by a native atomic owner."""

    def __init__(self, *, uid: int, revision: str, prompt_tokens: tuple[int, ...],
                 first_logits: Any, owner: NativeAtomicRequestOwner,
                 candidate: Qwen3PackedCandidate, maximum: int,
                 sampler: Callable[[Any], Any], processors: list[Callable],
                 matcher: StopSequenceMatcher, lane_rng: Any = None,
                 research_only: bool = False,
                 apcv2_restored_tokens: int = 0,
                 price_provenance: str | None = None,
                 price_evidence_sha256: str | None = None) -> None:
        if (type(uid) is not int or uid < 0 or not revision or not prompt_tokens or
                type(maximum) is not int or maximum < 1 or
                type(owner) is not NativeAtomicRequestOwner or
                (type(candidate) is not Qwen3PackedCandidate and
                 not supports_native_checkpoint_candidate(candidate)) or
                not callable(sampler) or type(matcher) is not StopSequenceMatcher):
            raise ValueError("incomplete native continuation state")
        if owner.supported_planes != getattr(candidate, "state_planes", ("kv",)) or owner.atomic_publish is not True:
            raise ValueError("native continuation requires atomic KV state")
        self.uid, self.revision = uid, revision
        self.tokens = list(prompt_tokens)  # Tokens already committed in native KV.
        self._first_logits = first_logits
        self._pending_token: int | None = None
        self.owner, self.candidate = owner, candidate
        self.research_only = research_only is True
        self.apcv2_restored_tokens = apcv2_restored_tokens
        self.price_provenance = price_provenance
        self.price_evidence_sha256 = price_evidence_sha256
        self.maximum, self.sampler = maximum, sampler
        self.processors, self.matcher = list(processors), matcher
        self.matcher_state = matcher.make_state()
        self.lane_rng = lane_rng
        self.count = 0
        # Count only physical native submissions and matched terminal events
        # observed by the backend, including the preceding prompt probe.
        backend = candidate.backend
        self.native_read_calls = int(getattr(backend, "read_submissions", 0))
        self.terminal_successes = int(getattr(backend, "terminal_successes", 0))
        self.closed = False

    def _forward_pending(self):
        token = self._pending_token
        if token is None:
            raise RuntimeError("native continuation has no pending token")
        request = CandidateRequest(self.uid, self.revision, 1,
                                   getattr(self.candidate, "state_planes", ("kv",)))
        branch = self.owner.begin(request)
        prepared = None
        backend = self.candidate.backend
        before_reads = int(getattr(backend, "read_submissions", 0))
        before_terminals = int(getattr(backend, "terminal_successes", 0))
        try:
            if getattr(self.candidate, "supports_singleton", False):
                logits, _forward = self.candidate.forward_staged(
                    (self.candidate.packed_lane((token,), branch),), (branch,),
                    permit_candidate=True)
            else:
                logits, _forward = self.candidate.forward(
                    (PackedLane((token,), branch.layers),), permit_candidate=True,
                    requested_capabilities=frozenset({"text"}), atomic_branch=branch)
            if getattr(logits, "shape", None) is None or logits.shape[0] != 1:
                raise RuntimeError("native decode did not return one logits row")
            reads = int(getattr(backend, "read_submissions", 0)) - before_reads
            terminals = int(getattr(backend, "terminal_successes", 0)) - before_terminals
            if reads != native_layer_count(self.candidate) or terminals != reads:
                raise RuntimeError("native decode lacks one terminal read per layer")
            prepared = branch.prepare(1)
            prepared.publish()
            self.tokens.append(token)
            self._pending_token = None
            self.native_read_calls = int(getattr(backend, "read_submissions", 0))
            self.terminal_successes = int(getattr(backend, "terminal_successes", 0))
            if getattr(self.candidate, "owns_physical_dispatch_proof", False):
                self._last_native_graph_proof = dict(_forward)
            return logits[-1]
        except BaseException:
            if prepared is not None:
                prepared.rollback()
            else:
                branch.rollback()
            raise

    def next(self) -> GenerationBatch.Response:
        if getattr(self.candidate, 'bootstrap_generation', 1) == 0:
            from .packed_prefill_receipt import bootstrap_prefill_attribution
            bootstrap_prefill_attribution(self.candidate)
        if self.closed:
            raise RuntimeError("native continuation is closed")
        # Keep the public generation alive through the complete native read,
        # terminal callback, sampler, and response construction. A private
        # branch pins its own pages, but this lease also protects the public
        # pointer from retirement while a live serving step observes it.
        with self.owner.snapshot() as public:
            if (public.revision != self.revision or
                    public.offset != len(self.tokens) or
                    public.generation < getattr(self.candidate, "bootstrap_generation", 1) or
                    len(public.layer_owners) != native_layer_count(self.candidate)):
                raise RuntimeError("native public reader state drifted")
            response = self._next_with_reader(public.generation, public.offset)
        # Publication retires the previous generation. The lease has now
        # closed, so reclaim on every successful step instead of retaining all
        # old generations until the request terminates. A failed reap leaves
        # the owner reachable for the next step or close path to retry.
        try:
            self.owner.reap_retired()
        except Exception:
            response.mtp_receipt["native_reader_reap_deferred"] = True
        else:
            response.mtp_receipt["native_reader_reap_deferred"] = False
        return response

    def _next_with_reader(self, reader_generation: int,
                          reader_offset: int, *,
                          logits_override: Any = _NO_GRAPH_LOGITS) -> GenerationBatch.Response:
        if logits_override is _NO_GRAPH_LOGITS:
            logits = self._first_logits
            if logits is not None:
                self._first_logits = None
            else:
                logits = self._forward_pending()
        else:
            logits = logits_override
        sampled, logprobs = self._stage_sample(logits)
        mx.eval(sampled, logprobs)
        return self._response_from_sample(reader_generation, reader_offset,
                                          sampled, logprobs)

    def _stage_sample(self, logits: Any) -> tuple[Any, Any]:
        """Build one lane's existing processor and sampler graph in lane order."""
        # Keep this ordinary implementation self-contained: source-contract
        # tests execute the method body against the existing lane seam.
        sample_logits = logits[None].astype(mx.float32)
        if self.processors:
            context = mx.array(self.tokens, dtype=mx.uint32)
            for processor in self.processors:
                sample_logits = processor(context, sample_logits)
        sample_logits = sample_logits.astype(mx.float32)
        logprobs = sample_logits - mx.logsumexp(sample_logits, axis=-1,
                                                 keepdims=True)
        sampled = self.sampler(logprobs)
        return sampled, logprobs

    def _stage_sample_for_context(self, logits: Any,
                                  tokens: list[int] | tuple[int, ...]) -> tuple[Any, Any]:
        """Build the ordinary sample graph for an explicit speculative prefix.

        Online verification cannot mutate public lane history while later
        proposal rows are still private.  Supplying the exact prospective
        context keeps processors byte-for-byte on their ordinary input law.
        """
        # GenerationBatch applies processors to a one-row logits tensor after
        # the input token is in its token context. Match that order here.
        sample_logits = logits[None].astype(mx.float32)
        if self.processors:
            context = mx.array(tokens, dtype=mx.uint32)
            for processor in self.processors:
                sample_logits = processor(context, sample_logits)
        sample_logits = sample_logits.astype(mx.float32)
        logprobs = sample_logits - mx.logsumexp(sample_logits, axis=-1,
                                                 keepdims=True)
        sampled = self.sampler(logprobs)
        return sampled, logprobs

    def _response_from_sample(self, reader_generation: int, reader_offset: int,
                              sampled: Any, logprobs: Any) -> GenerationBatch.Response:
        """Validate evaluated outputs and advance only this request's state."""
        if sampled.shape != (1,):
            raise RuntimeError("native sampler must return one token")
        token = int(sampled[0].item())
        reason = _invalid_output_reason(token, logprobs[0])
        if reason is not None:
            raise RuntimeError(f"native sampled output invalid: {reason}")
        self.count += 1
        self.matcher_state, matched = StopSequenceMatcher.match(
            self.matcher_state, self.matcher._trie, token)
        finish = "stop" if matched else "length" if self.count >= self.maximum else None
        self._pending_token = token if finish is None else None
        receipt = {
            "route": getattr(self.candidate, "serving_route", ("native_qwen3_paged_b2" if getattr(
                self.candidate, "_serving_b2", False) else "native_qwen3_paged")),
            "implemented": True,
            "qualified": False, "selected": not self.research_only,
            "observed_used": (not self.research_only and self.terminal_successes > 0
                              and not getattr(self.candidate, "_serving_b2", False)),
            "research_executed": self.research_only and self.terminal_successes > 0,
            "revision": self.revision,
            "native_read_calls": self.native_read_calls,
            "terminal_successes": self.terminal_successes,
            "native_reader_generation": reader_generation,
            "native_reader_offset": reader_offset,
            "native_reader_lease": "held_through_token_step",
            "apcv2": ("warm_prefix_restored_native"
                      if self.apcv2_restored_tokens else "native_checkpoint_unavailable"),
            "apcv2_restored_tokens": self.apcv2_restored_tokens,
            "ordinary_forward_calls": 0,
        }
        if self.price_provenance is not None:
            receipt["price_provenance"] = self.price_provenance
            receipt["price_evidence_sha256"] = self.price_evidence_sha256
        if getattr(self.candidate, "_serving_b2", False):
            receipt["admission_profile"] = self.candidate._b2_profile_id
            receipt["prefill_mode"] = ("packed_staged" if getattr(
                self.candidate, "_b2_packed_prefill", False) else "serial_native")
            if getattr(self.candidate, "_b2_prefill_proof", None) is not None:
                receipt["native_prefill_proof"] = dict(self.candidate._b2_prefill_proof)
            receipt["price_usable"] = False
            receipt["native_prefill_observed_used"] = self.terminal_successes > 0
            if getattr(self.candidate, "bootstrap_generation", 1) == 0:
                from .packed_prefill_receipt import bootstrap_prefill_attribution
                receipt.update(bootstrap_prefill_attribution(self.candidate))
                receipt["state_planes"] = ["kv", "gdn"]
                receipt["native_layer_count"] = self.candidate.native_layer_count
                receipt["logical_layer_count"] = self.candidate.logical_layer_count
                receipt["singleton_mode"] = ("same_math_short_stock32_long_scalar" if
                    getattr(self.candidate, "_serving_stock_singleton", False) else "same_math_short_tile_long_scalar")
                receipt["stock_reduction_selected"] = getattr(self.candidate, "_serving_stock_reduction", False)
                receipt["stock_singleton_selected"] = getattr(self.candidate, "_serving_stock_singleton", False)
                receipt["survivor_q1_stripes"] = (32 if getattr(self.candidate, "_serving_stock_singleton", False) else self.candidate.q1_stripes)
                receipt["survivor_fallback_q1_stripes"] = self.candidate.q1_stripes
                proof = getattr(self, "_last_native_graph_proof", None)
                if proof is not None:
                    receipt["hybrid_graph_proof"] = dict(proof)
                    receipt["q1_simd_stripes"] = proof["q1_simd_stripes"]
                    receipt["stock_singleton_observed_used"] = proof.get("native_stock_singleton_dispatches", 0) > 0
                receipt["observed_used"] = not self.research_only and self.terminal_successes > 0
        return GenerationBatch.Response(
            uid=self.uid, token=token, logprobs=logprobs[0],
            finish_reason=finish, prompt_cache=None,
            all_tokens=list(self.tokens) if finish else None,
            lane_rng=self.lane_rng if finish else None,
            rng_draws=self.lane_rng.draws if finish and self.lane_rng else 0,
            mtp_receipt=receipt, execution_width=1,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._first_logits = None
        self._pending_token = None
        self.owner.close()
        self.reap()

    def reap(self) -> None:
        """Reclaim only generations whose readers and native epochs are done."""
        from .paged_native_retirement import reap_native_request_owner

        reap_native_request_owner(self.owner, self.candidate.backend.writer,
                                  self.candidate.backend)
        reap = getattr(self.candidate, "reap_serving_resources", None)
        if callable(reap): reap()

    @property
    def can_release(self) -> bool:
        # Fail closed with older owner revisions that lack this evidence.
        return self.closed and bool(getattr(self.owner, "fully_retired", False))


__all__ = ["NativeQwen3Continuation"]
