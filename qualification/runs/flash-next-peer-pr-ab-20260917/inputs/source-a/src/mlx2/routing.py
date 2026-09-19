"""Evidence-gated capability routing."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import FIDELITY_RANK, ModelDescriptor, QualifiedProfile, RouteRequest


class RouteUnavailable(LookupError):
    """No declared and qualified route can satisfy a request."""


@dataclass(frozen=True, slots=True)
class RouteDecision:
    model: ModelDescriptor
    profile: QualifiedProfile
    receipt: str


class RoutePlanner:
    def __init__(self) -> None:
        self._models: dict[str, ModelDescriptor] = {}
        self._profiles: dict[str, list[QualifiedProfile]] = {}

    def register_model(self, descriptor: ModelDescriptor) -> None:
        if descriptor.key in self._models:
            raise ValueError(f"model descriptor already registered: {descriptor.key}")
        self._models[descriptor.key] = descriptor

    def register_profile(self, model_key: str, profile: QualifiedProfile) -> None:
        model = self._models.get(model_key)
        if model is None:
            raise KeyError(f"unknown model descriptor: {model_key}")
        missing = profile.capabilities - model.capabilities
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"profile claims undeclared capabilities: {names}")
        profiles = self._profiles.setdefault(model_key, [])
        if any(existing.name == profile.name for existing in profiles):
            raise ValueError(f"profile name already registered: {profile.name}")
        profiles.append(profile)

    def decide(self, request: RouteRequest) -> RouteDecision:
        model = self._models.get(request.model_key)
        if model is None:
            raise RouteUnavailable(f"unknown model descriptor: {request.model_key}")
        undeclared = request.required - model.capabilities
        if undeclared:
            names = ", ".join(sorted(item.value for item in undeclared))
            raise RouteUnavailable(f"model does not declare: {names}")

        candidates = [
            profile
            for profile in self._profiles.get(request.model_key, ())
            if request.required <= profile.capabilities
            and FIDELITY_RANK[profile.fidelity]
            >= FIDELITY_RANK[request.minimum_fidelity]
        ]
        if not candidates:
            raise RouteUnavailable("no qualified profile satisfies the request")
        candidates.sort(
            key=lambda profile: (
                len(profile.capabilities - request.required),
                -FIDELITY_RANK[profile.fidelity],
                profile.name,
            )
        )
        profile = candidates[0]
        requested = (
            ",".join(sorted(item.value for item in request.required)) or "ordinary"
        )
        receipt = (
            f"model={model.key};profile={profile.name};"
            f"requested={requested};fidelity={profile.fidelity.value}"
        )
        return RouteDecision(model=model, profile=profile, receipt=receipt)
