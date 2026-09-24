"""Gemma 4 sparse and dense multimodal adapters for the MLX-VLM runtime.

The Gemma 4 model and processor remain an optional MLX-VLM dependency.  This
module owns topology, cache policy, media identity, and mlx2 route declarations;
see provenance/gemma4-mlx-vlm.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..multimodal import media_fingerprint, resolve_media
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .mlx_vlm import (
    _ids_and_kwargs,
    _LogitsModel,
    _media_token_end,
    _MLXVLMAdapter,
    _plain_messages,
    _source,
    inspect_artifact,
)

_TOPOLOGIES = {
    "26b-a4b": {"layers": 30, "hidden": 2816, "moe": True, "experts": 128},
    "31b": {"layers": 60, "hidden": 5376, "moe": False, "experts": None},
}
_SOURCE_REVISIONS = {
    "26b-a4b": ("google/gemma-4-26B-A4B", "24548b62aa021d562695c04aaf7758a1ea47990b"),
    "31b": ("google/gemma-4-31B", "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89"),
}


def inspect_gemma4_artifact(model_path):
    """Resolve these two architectures without importing or loading MLX."""
    path = Path(model_path).expanduser().resolve()
    artifact = inspect_artifact(path, expected="gemma4")
    config = artifact["config"]
    text = config.get("text_config") or {}
    topology = (
        text.get("num_hidden_layers"),
        text.get("hidden_size"),
        bool(text.get("enable_moe_block")),
        text.get("num_experts"),
    )
    variant = next(
        (
            name for name, expected in _TOPOLOGIES.items()
            if topology == (
                expected["layers"], expected["hidden"],
                expected["moe"], expected["experts"],
            )
        ),
        None,
    )
    if variant is None:
        raise ValueError(f"No mlx2 Gemma 4 adapter for topology {topology!r}")
    layers = text.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != topology[0]
            or set(layers) != {"full_attention", "sliding_attention"}
            or text.get("sliding_window") != 1024):
        raise ValueError("Gemma 4 full/sliding attention layout is unsupported")
    if config.get("audio_config") is not None or not isinstance(config.get("video_token_id"), int):
        raise ValueError("Expected vision/video Gemma 4 artifact without an audio tower")
    quant = config.get("quantization")
    if quant and any(quant.get(key) != value for key, value in
                     {"bits": 8, "group_size": 64, "mode": "affine"}.items()):
        raise ValueError("Only the local affine 8-bit Gemma 4 conversion is supported")
    provenance = path / "source-and-quantization.json"
    if provenance.is_file():
        source = json.loads(provenance.read_text())
        expected_repo, expected_rev = _SOURCE_REVISIONS[variant]
        if (source.get("source_repo"), source.get("source_revision")) != (expected_repo, expected_rev):
            raise ValueError("Gemma 4 conversion source does not match its declared model")
        artifact["source_revision"] = expected_rev
    artifact["variant"] = variant
    artifact["precision"] = "mlx-affine-8bit" if quant else "bf16"
    artifact["full_attention_layers"] = layers.count("full_attention")
    artifact["sliding_attention_layers"] = layers.count("sliding_attention")
    return artifact


class _Gemma4LogitsModel(_LogitsModel):
    """Use mlx2 cache types so batching and APCv2 can own model state."""

    def make_cache(self):
        from ..runtime.models.cache import KVCache, RotatingKVCache

        model = self._model
        text = model.config.text_config
        count = model.language_model.model.first_kv_shared_layer_idx
        return [
            KVCache() if kind == "full_attention" else
            RotatingKVCache(max_size=text.sliding_window, keep=0)
            for kind in text.layer_types[:count]
        ]


def _descriptor(variant, adapter_name):
    return ModelDescriptor(
        model_type="gemma4",
        family="gemma-4",
        variant=variant,
        state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
        capabilities=frozenset({
            Capability.TEXT, Capability.VISION, Capability.VIDEO,
            Capability.STREAMING, Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE, Capability.APC_V2,
        }),
        cache_layout=f"gemma4-{variant}-full-sliding-v1",
        metadata={
            "execution": f"mlx2.adapters.gemma4.{adapter_name}",
            "qualification": "pending",
            "required_qualification_checks": (
                "multimodal_image", "multimodal_video", "multimodal_apcv2_reuse",
            ),
            "audio_input": "unsupported_no_audio_tower",
        },
    )


GEMMA4_A4B = _descriptor("26b-a4b", "Gemma4A4BAdapter")
GEMMA4_31B = _descriptor("31b", "Gemma431BAdapter")


class _Gemma4Adapter(_MLXVLMAdapter):
    """Shared text/image/video prefill with exact media-bound feature reuse."""

    @staticmethod
    def streaming_detokenizer_class():
        from ..runtime.tokenizer_utils import SPMStreamingDetokenizer

        return SPMStreamingDetokenizer

    @staticmethod
    def _wrap_model(model):
        return _Gemma4LogitsModel(model)

    def __init__(self, model_path, *, execution_policy=None):
        artifact = inspect_gemma4_artifact(model_path)
        if artifact["variant"] != self.descriptor.variant:
            raise ValueError("Gemma 4 adapter variant does not match the artifact")
        super().__init__(model_path, execution_policy=execution_policy)
        self.identity.update({key: artifact[key] for key in (
            "variant", "precision", "full_attention_layers", "sliding_attention_layers"
        )})

    def _render(self, messages):
        # Base Gemma 4 tokenizer files have no chat template; the processor
        # handles them and preserves explicit media placeholders.
        render = getattr(self.processor, "apply_chat_template", None)
        if not callable(render):
            raise TypeError("Gemma 4 processor has no prompt renderer")
        return render(messages, tokenize=False, add_generation_prompt=True)

    def prepare_multimodal_request(self, request, *, file_loader=None):
        import mlx.core as mx
        import numpy as np

        media, replacements, images, videos, video_metadata = [], [], [], [], []
        for message in request["messages"]:
            content = message.get("content")
            for part in content if isinstance(content, list) else ():
                part_type = part.get("type")
                if part_type == "text":
                    continue
                kind = {
                    "image_url": "image", "input_image": "image",
                    "input_video": "video",
                }.get(part_type)
                if kind is None:
                    raise ValueError(f"Gemma 4 input part {part_type!r} is unsupported")
                source = _source(part, kind)
                value = resolve_media(
                    source, kind=kind, file_loader=file_loader,
                    **({"fps": 2.0, "max_frames": 32} if kind == "video" else {}),
                )
                media.append(value)
                if kind == "image":
                    images.append(value.value)
                    replacements.append(self.processor.image_token)
                else:
                    frames = np.stack(value.value).transpose(0, 3, 1, 2)
                    videos.append(frames)
                    video_metadata.append({
                        "total_num_frames": value.metadata["source_frames"],
                        "fps": value.metadata["source_fps"],
                        "frames_indices": list(value.metadata["sampled_indices"]),
                    })
                    replacements.append(self.processor.video_token)
        if not media:
            return request
        messages = _plain_messages(request["messages"], replacements)
        processed = self.processor(
            text=self._render(messages), images=images or None,
            videos=videos or None, video_metadata=video_metadata or None,
            return_tensors="np",
        )
        media_token_end = _media_token_end(processed)
        ids, kwargs = _ids_and_kwargs(processed)
        kwargs.pop("num_frames_per_video", None)
        kwargs = {
            name: mx.array(value) if isinstance(value, np.ndarray) else value
            for name, value in kwargs.items()
        }
        policy = {"family": "gemma4", "video_fps": 2.0, "video_max_frames": 32}
        fingerprint = media_fingerprint(media, policy=policy)
        kwargs["vision_cache"] = self.media_feature_cache
        if images:
            kwargs["_image_key"] = f"{self.identity['fingerprint']}:{fingerprint}:image"
        if videos:
            kwargs["_video_key"] = f"{self.identity['fingerprint']}:{fingerprint}:video"
        return {
            **request,
            "messages": messages,
            "_mlx2_prompt_tokens": ids,
            "_mlx2_prefill_inputs": kwargs,
            "_mlx2_media_token_end": media_token_end,
            "_mlx2_media_fingerprint": fingerprint,
        }

    def diagnostics(self):
        return {**super().diagnostics(),
                "variant": self.identity["variant"],
                "precision": self.identity["precision"],
                "full_attention_layers": self.identity["full_attention_layers"],
                "sliding_attention_layers": self.identity["sliding_attention_layers"]}


class Gemma4A4BAdapter(_Gemma4Adapter):
    descriptor = GEMMA4_A4B
    sampling_defaults = VendorSampling.single(
        SamplingDefaults(temperature=1.0, top_p=0.95, top_k=64, source=GENERATION_CONFIG),
        model="google/gemma-4-26B-A4B",
    )


class Gemma431BAdapter(_Gemma4Adapter):
    descriptor = GEMMA4_31B
    sampling_defaults = VendorSampling.single(
        SamplingDefaults(temperature=1.0, top_p=0.95, top_k=64, source=GENERATION_CONFIG),
        model="google/gemma-4-31B",
    )
