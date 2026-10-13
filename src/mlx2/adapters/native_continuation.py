"""Adapter-owned Qwen paged-native continuation lifecycle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QwenNativeContinuationProvider:
    """Bind the Qwen lifecycle and one exact candidate to an adapter revision."""

    revision: str
    model: object
    protocol: str
    candidate_type: type

    def validate_candidate(self, candidate) -> None:
        if (
            type(candidate) is not self.candidate_type
            or candidate.model is not self.model
            or getattr(candidate, "_native_continuation_provider", None) is not self
        ):
            raise TypeError("candidate is not bound to this native continuation provider")

    def create_continuation(self, **kwargs):
        self.validate_candidate(kwargs.get("candidate"))
        if kwargs.get("revision") != self.revision:
            raise ValueError("native continuation revision differs from adapter binding")
        from ..runtime.paged_native_continuation import NativeQwen3Continuation

        return NativeQwen3Continuation(**kwargs)

    def probe(self, *args, candidate, **kwargs):
        self.validate_candidate(candidate)
        from ..runtime.paged_native_batch_lifecycle import probe_queued_native_qwen3

        # ``candidate`` is the lifecycle's fifth positional parameter, and
        # serving passes capability, reserved rows and offers positionally
        # after the owner. Restore that position instead of binding it twice.
        if len(args) >= 4:
            return probe_queued_native_qwen3(*args[:4], candidate, *args[4:], **kwargs)
        return probe_queued_native_qwen3(*args, candidate=candidate, **kwargs)

    def prepare(self, *args, **kwargs):
        from ..runtime.paged_native_batch_lifecycle import prepare_queued_native_first_response

        return prepare_queued_native_first_response(*args, **kwargs)

    def run_research(self, *args, candidate, **kwargs):
        self.validate_candidate(candidate)
        from ..runtime.paged_native_batch_lifecycle import run_research_native_queued_qwen3

        return run_research_native_queued_qwen3(*args, candidate=candidate, **kwargs)

    def run_research_pair(self, *args, candidate, **kwargs):
        self.validate_candidate(candidate)
        from ..runtime.paged_native_batch_lifecycle import run_research_native_queued_qwen3_b2_packed

        return run_research_native_queued_qwen3_b2_packed(
            *args, candidate=candidate, **kwargs
        )

    def receipt(self) -> dict:
        return {
            "route": self.protocol,
            "provider": "qwen_adapter_native_continuation_v1",
            "provider_revision": self.revision,
            "implemented": True,
            "qualified": False,
        }


def bind_qwen_native_continuation_provider(
    candidate, revision: str, model, protocol: str
):
    provider = QwenNativeContinuationProvider(
        revision, model, protocol, type(candidate)
    )
    candidate._native_continuation_provider = provider
    return provider
