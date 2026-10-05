"""Qwen2.5-VL 3B image/video candidate from a pinned mlx-vlm source."""

from dataclasses import replace
import os

from ..contracts import Capability
from .pinned_vlm_candidate import (
    PinnedVisionCandidateAdapter, descriptor_for, inspect_vision_artifact,
)
from .qwen25_rope import RequestPrivateQwen25Model
from .qwen25_vision_grouped import (
    grouped_vision_counters, install_grouped_vision_attention,
)
from .pinned_vlm_candidate import SOURCE_REVISION
from .vision_feature_reuse import certify_cold_prefill, install_qwen25

_GROUPED_VISION_ENV = "MLX2_QWEN25_GROUPED_VISION_CANDIDATE"
_FEATURE_REUSE_ENV = "MLX2_QWEN25_VISION_FEATURE_REUSE_CANDIDATE"
_TOWER_REUSE_ENV = "MLX2_QWEN25_VISION_TOWER_REUSE_CANDIDATE"

_BASE_DESCRIPTOR = descriptor_for("qwen2_5_vl")
DESCRIPTOR = replace(
    _BASE_DESCRIPTOR,
    cache_layout="qwen25-vl-request-private-mrope-v1",
    capabilities=_BASE_DESCRIPTOR.capabilities | frozenset(
        {Capability.APC_V2, Capability.PREFIX_REUSE}
    ),
    metadata={**_BASE_DESCRIPTOR.metadata,
              "execution": "mlx2.adapters.qwen25_vl.Qwen25VLCandidateAdapter"},
)


def inspect_artifact(model_path):
    return inspect_vision_artifact(model_path, expected="qwen2_5_vl")


class Qwen25VLCandidateAdapter(PinnedVisionCandidateAdapter):
    model_type = "qwen2_5_vl"
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)

    def _template_messages(self, messages, replacements):
        """Retain Qwen's vision framing from its typed chat template."""
        cursor = iter(replacements)
        missing = object()
        converted = []
        for message in messages:
            original = message.get("content")
            if not isinstance(original, list):
                converted.append(message)
                continue
            content = []
            for part in original:
                if part.get("type") == "text":
                    content.append({"type": "text", "text": part.get("text", "")})
                    continue
                replacement = next(cursor, missing)
                if replacement is missing:
                    raise RuntimeError("multimodal replacements are fewer than media parts")
                if replacement == self.processor.image_token:
                    content.append({"type": "image"})
                elif replacement == self.processor.video_token:
                    content.append({"type": "video"})
                else:
                    raise ValueError("Qwen media replacement is not a supported marker")
            converted.append({**message, "content": content})
        if next(cursor, missing) is not missing:
            raise RuntimeError("multimodal replacements outnumber media parts")
        return converted

    def __init__(self, model_path: str, *, execution_policy=None):
        # Reject bad candidate selections before loading weights or installing
        # a model-local replacement on the pinned source path.
        enabled = os.environ.get(_GROUPED_VISION_ENV, "0")
        feature_reuse = os.environ.get(_FEATURE_REUSE_ENV, "0")
        tower_reuse = os.environ.get(_TOWER_REUSE_ENV, "0")
        for name, value in ((_GROUPED_VISION_ENV, enabled),
                            (_FEATURE_REUSE_ENV, feature_reuse),
                            (_TOWER_REUSE_ENV, tower_reuse)):
            if value not in ("0", "1"):
                raise ValueError(f"{name} must be 0 or 1")
        if feature_reuse == tower_reuse == "1":
            raise ValueError("vision feature reuse scopes are mutually exclusive")
        super().__init__(model_path, execution_policy=execution_policy)
        self._grouped_vision_enabled = enabled == "1"
        if self._grouped_vision_enabled:
            install_grouped_vision_attention(self.model._model, enable=True)
        self._vision_feature_reuse_scope = (
            "tower_inputs_v1" if tower_reuse == "1" else "prompt_v1"
        )
        self._vision_feature_reuse_enabled = feature_reuse == "1" or tower_reuse == "1"
        if tower_reuse == "1":
            from .multimodal import MediaFeatureCache

            self.media_feature_cache = MediaFeatureCache(max_entries=16, max_bytes=256 << 20)
        self._vision_feature_reuse_counters = None
        if self._vision_feature_reuse_enabled:
            self._vision_feature_reuse_counters = install_qwen25(
                self.model._model, self.media_feature_cache,
                artifact_fingerprint=self.identity["fingerprint"],
                source_revision=SOURCE_REVISION,
                processor_fingerprint=self.identity["fingerprint"],
            )
        self.model = RequestPrivateQwen25Model(self.model._model)
        self.layout = self.descriptor.cache_layout

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "qwen25_grouped_vision": (
                "candidate" if self._grouped_vision_enabled else "source"
            ),
            "qwen25_grouped_vision_counts": (
                grouped_vision_counters(self.model._model)
                if self._grouped_vision_enabled else None
            ),
            "qwen25_vision_feature_reuse": (
                vars(self._vision_feature_reuse_counters).copy()
                if self._vision_feature_reuse_counters is not None else None
            ),
        }

    def execution_config(self, *, max_lanes, prefill_step):
        config = super().execution_config(
            max_lanes=max_lanes, prefill_step=prefill_step
        )
        config["grouped_vision_attention"] = (
            "candidate_v1" if self._grouped_vision_enabled else "source"
        )
        config["vision_feature_reuse"] = (
            "tower_only_candidate_v1" if self._vision_feature_reuse_enabled
            and getattr(self, "_vision_feature_reuse_scope", "prompt_v1") == "tower_inputs_v1" else
            "candidate_v1" if self._vision_feature_reuse_enabled else "source"
        )
        if getattr(self, "_vision_feature_reuse_scope", "prompt_v1") == "tower_inputs_v1":
            config["vision_feature_cache_max_bytes"] = self.media_feature_cache.max_bytes
        return config

    def validate_prefill_inputs(self, request, remaining_tokens, prefill_input):
        if not self._vision_feature_reuse_enabled:
            return prefill_input
        return certify_cold_prefill(
            request, remaining_tokens, prefill_input, family=self.model_type,
            artifact_fingerprint=self.identity["fingerprint"],
            source_revision=SOURCE_REVISION,
            processor_fingerprint=self.identity["fingerprint"],
            attest_prepared=self.has_trusted_media_preparation,
            scope=getattr(self, "_vision_feature_reuse_scope", "prompt_v1"),
        )

    def prepare_multimodal_request(self, request, *, file_loader=None):
        prepared = super().prepare_multimodal_request(
            request, file_loader=file_loader
        )
        if prepared is request:
            return request
        media_end = int(prepared["_mlx2_media_token_end"])
        if media_end >= len(prepared["_mlx2_prompt_tokens"]):
            raise ValueError("Qwen2.5-VL media placeholder cannot be the final prompt token")
        inputs = dict(prepared["_mlx2_prefill_inputs"])
        inputs["_mlx2_rope_media_end"] = media_end
        return {**prepared, "_mlx2_prefill_inputs": inputs}
