# SPDX-License-Identifier: MIT
"""Qwen3.5 122B-A10B ordinary text model using the shared hybrid MoE math.

The 48-layer/3072-width topology is selected by ModelArgs. The MTP and vision
weights in the local checkpoint are excluded before allocation and loading.
See provenance/qwen35-122b.json.
"""

from .qwen36_35b import Model as Qwen36Model, ModelArgs


class Model(Qwen36Model):
    apc_v2_layout = "qwen35-122b-a10b-hybrid-layer-segments-v1"

    @staticmethod
    def shard_prune(weights):
        return {
            key: value for key, value in weights.items()
            if not key.startswith((
                "vision_tower", "model.visual", "language_model.mtp.",
                "mtp.", "model.language_model.mtp.",
            ))
        }


class MTPModel(Model):
    """Offline candidate retaining the checkpoint's embedded prediction head."""

    @staticmethod
    def shard_prune(weights):
        return {
            key: value for key, value in weights.items()
            if not key.startswith(("vision_tower", "model.visual"))
        }
