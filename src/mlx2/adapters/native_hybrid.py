"""Hybrid adapter's source-bound native profile and factory selection."""

from .native_cohort import NativeCohortPreparation, live_identity


class HybridNativeCohort:
    def __init__(self, kind):
        if kind not in {"hybrid_pair", "packed_pair", "packed_n"}:
            raise ValueError("unsupported hybrid native cohort kind")
        self.kind = kind

    def prepare(
        self,
        adapter,
        *,
        profile_path,
        manifest_path,
        mlx_wheel_path,
        tokens,
        environment,
        input_ids=(),
    ):
        identity = live_identity(adapter, manifest_path, mlx_wheel_path)
        counts = tuple(map(len, tokens))
        if self.kind == "packed_n":
            from ..runtime.hybrid_packed_prefill_n import load_profile

            profile = load_profile(
                profile_path,
                live_identity=identity,
                source_input_ids=input_ids,
                counts=counts,
                tokens=tokens,
                environment_values=environment,
            )
            factory = profile
        elif self.kind == "packed_pair":
            from ..runtime.paged_packed_prefill_serving_profile import (
                factory_profile,
                load_profile,
            )

            profile = load_profile(
                profile_path,
                live_identity=identity,
                context_lengths=counts,
                environment=environment,
            )
            factory = factory_profile(profile, counts)
        else:
            from ..runtime.paged_hybrid_research_profile import (
                load_hybrid_research_profile,
            )

            profile = load_hybrid_research_profile(
                profile_path,
                live_identity=identity,
                context_lengths=counts,
                environment=environment,
            )
            factory = profile
        return NativeCohortPreparation(
            profile,
            identity,
            factory,
            packed=self.kind != "hybrid_pair",
            input_ids=input_ids,
        )

    def allocate(self, adapter, requests, preparation, *, cancelled):
        if cancelled():
            raise ValueError("native cohort cancelled before allocation")
        kwargs = {
            "profile": preparation.factory_profile,
            "permit_candidate": True,
            "cancelled": cancelled,
        }
        if self.kind == "packed_n":
            return adapter.create_native_packed_prefill_n(
                requests,
                live_identity=preparation.identity,
                source_input_ids=preparation.input_ids,
                phase_boundary=getattr(adapter, "_native_n20_phase_boundary", None),
                **kwargs,
            )
        if self.kind == "packed_pair":
            return adapter.create_native_packed_prefill_b2(
                requests,
                live_identity=preparation.identity,
                **kwargs,
            )
        return adapter.create_native_paged_hybrid_b2(requests, **kwargs)
