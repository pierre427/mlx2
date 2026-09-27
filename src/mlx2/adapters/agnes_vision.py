"""Agnes 3.0 embedded vision candidate; ordinary text remains the default."""

from .pinned_vlm_candidate import (
    PinnedVisionCandidateAdapter, descriptor_for, inspect_vision_artifact,
)

DESCRIPTOR = descriptor_for("agnes")


def inspect_artifact(model_path):
    return inspect_vision_artifact(model_path, expected="agnes")


class AgnesVisionCandidateAdapter(PinnedVisionCandidateAdapter):
    model_type = "agnes"
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
