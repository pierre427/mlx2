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
    _require_media_markers,
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
    expected_layers = [
        "full_attention" if index % 6 == 5 else "sliding_attention"
        for index in range(topology[0])
    ]
    if (layers != expected_layers or text.get("sliding_window") != 1024
            or text.get("num_kv_shared_layers") != 0):
        raise ValueError("Gemma 4 full/sliding attention layout is unsupported")
    if (config.get("audio_config") is not None
            or (config.get("vision_config") or {}).get("model_type") != "gemma4_vision"
            or config.get("video_token_id") != 258884):
        raise ValueError("Expected vision/video Gemma 4 artifact without an audio tower")
    quant = config.get("quantization")
    if quant and any(quant.get(key) != value for key, value in
                     {"bits": 8, "group_size": 64, "mode": "affine"}.items()):
        raise ValueError("Only the local affine 8-bit Gemma 4 conversion is supported")
    provenance = path / "source-and-quantization.json"
    if quant and not provenance.is_file():
        raise ValueError("Gemma 4 conversion source provenance is required")
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


def _stamp_fps(timestamps) -> float:
    """The per-video rate whose ``j / fps`` stamps the sampled frames.

    The pinned mlx-vlm processor ignores ``video_metadata`` and labels frame
    ``j`` as ``j / fps`` (its default 2.0), so a clip longer than 16 s sampled
    to 32 frames was stamped 00:00..00:15.  Frames are sampled uniformly from
    index 0 (multimodal.uniform_frame_indices), so the span over the frame
    count gives each stamp to within one source frame.
    """
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        return 2.0  # a single frame is stamped 00:00 at any rate
    # The processor floors j / fps to whole seconds; shave one part in 1e9 so
    # an exact stamp (59.0) does not round down to 58.999...
    return (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) * (1 - 1e-9)


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

    def prefill_step_default(self):
        """Adapter-preferred prefill chunk; an explicit engine setting wins."""
        return int(type(self).default_prefill_step)

    @staticmethod
    def int8_prefill_select(scope):
        """Text decoder projections only (``language_model.model.layers.*``).

        The default path classifier would also take the quantized multimodal
        projector ``embed_vision.embedding_projection`` under ``all``.  The
        vision tower, the tied embedding head, the Gemma 3n-style per-layer
        input projections (absent on the 31B, ``hidden_size_per_layer_input``
        is 0) and the 26B router / experts stay stock.  Only declared on the
        dense 31B (``Gemma431BAdapter.int8_prefill_supported``)."""
        from ..runtime.int8_prefill import EXCLUDED_SEGMENTS

        def select(path, module):
            segments = path.split(".")
            if segments[:3] != ["language_model", "model", "layers"]:
                return False
            if any(
                seg in EXCLUDED_SEGMENTS or seg.startswith("per_layer")
                for seg in segments
            ):
                return False
            if scope == "all":
                return "self_attn" in segments or "mlp" in segments
            return "mlp" in segments

        return select

    def exact_prefix_cascade_contract(self):
        """Gemma 4 ownership boundary for continuation-prefix reuse.

        Both variants have attention-KV state only, but their full/sliding
        geometry differs in depth and the A4B variant additionally owns MoE
        tensor math.  Prefix reuse is exact only after APCv2 has restored the
        declared cache layout at one logical token boundary.  Cascade-private
        verify state is not publishable to APCv2 and ordinary decode remains
        the reference path.
        """

        variant = self.descriptor.variant
        topology = _TOPOLOGIES[variant]
        full = topology["layers"] // 6
        return {
            "schema": "mlx2.exact-prefix-cascade-contract.v1",
            "family": "gemma-4",
            "variant": variant,
            "verification_order": "longest_first",
            "invalid_sibling_pruning": True,
            "accepted_prefix_state": "exact_apcv2_restore_or_canonical_replay",
            "shared_prefix_reuse": {
                "authority": "apcv2",
                "exact_only": True,
                "recompute_common_tokens": False,
                "cache_layout": self.descriptor.cache_layout,
                "full_attention_layers": full,
                "sliding_attention_layers": topology["layers"] - full,
                "sliding_window": 1024,
            },
            "moe_tensor_math": "adapter_owned" if topology["moe"] else "not_present",
            "request_private_only": True,
            "apcv2_publication": False,
            "transactional_multirow_state_reuse": False,
            "implemented": True,
            "implementation_scope": "planner_adapter_contract_and_geometry_gate",
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }

    def exact_shared_prefix_geometry(self, caches, prefix_tokens):
        """Fail closed unless a restored cache is an exact Gemma 4 boundary.

        The check is intentionally about state geometry, not model identity:
        APCv2 already binds revision, tokenizer, media fingerprint and layout.
        A successful result allows a caller to verify only proposal suffixes;
        it does not select a serving route or authorize publication.
        """

        if type(prefix_tokens) is not int or prefix_tokens < 0:
            raise ValueError("prefix_tokens must be a nonnegative integer")
        expected = [
            "full_attention" if index % 6 == 5 else "sliding_attention"
            for index in range(_TOPOLOGIES[self.descriptor.variant]["layers"])
        ]
        caches = tuple(caches)
        if len(caches) != len(expected):
            return {
                "eligible": False,
                "reason": "cache_layer_count_mismatch",
                "expected_layers": len(expected),
                "actual_layers": len(caches),
            }
        for index, (kind, cache) in enumerate(zip(expected, caches)):
            wanted = "KVCache" if kind == "full_attention" else "RotatingKVCache"
            actual_type = type(cache)
            actual = f"{actual_type.__module__}.{actual_type.__qualname__}"
            if actual != f"mlx2.runtime.models.cache.{wanted}":
                return {
                    "eligible": False,
                    "reason": "cache_plane_type_mismatch",
                    "layer": index,
                    "expected": wanted,
                    "actual": actual_type.__name__,
                }
            if int(cache.offset) != prefix_tokens:
                return {
                    "eligible": False,
                    "reason": "cache_logical_offset_mismatch",
                    "layer": index,
                    "expected": prefix_tokens,
                    "actual": int(cache.offset),
                }
            if kind == "sliding_attention" and (
                int(cache.max_size) != 1024 or int(cache.keep) != 0
            ):
                return {
                    "eligible": False,
                    "reason": "sliding_cache_geometry_mismatch",
                    "layer": index,
                }
        return {
            "eligible": True,
            "authority": "apcv2_exact_restore",
            "prefix_tokens": prefix_tokens,
            "recompute_common_tokens": False,
            "suffix_only": True,
            "publishable": False,
        }

    def plan_exact_prefix_cascade(
        self, paths, accepted_prefix=(), *, attempted=()
    ):
        """Plan one Gemma 4 cascade stage without mutating model state."""

        from ..runtime.exact_prefix_cascade import next_cascade_stage

        return next_cascade_stage(paths, accepted_prefix, attempted=attempted)

    def _budget_prefill_step(self):
        return getattr(self, "_prefill_step", self.prefill_step_default())

    def cache_budget(self, *, mtp):
        from .mlx_vlm_memory import SlidingKVCacheBudget

        return SlidingKVCacheBudget.from_gemma4_config(
            self._text_config(), mtp=mtp,
            prefill_step=self._budget_prefill_step(),
            root_config=self.identity["config"],
        )

    def prompt_tokens(self, request):
        """Text prompts must start with ``<bos>``, as processor-built media prompts do.

        The base checkpoints ship no chat template, so neither the raw
        completions prompt nor the processor's plain-text chat fallback
        carries ``<bos>``, and the base encoder call adds no special tokens.
        Without it the 26B-A4B degenerated on the served text path ("The
        capital of France is" -> "the the the ..."; qualification/runs/
        gemma4-defaults-20260925/results/smoke-26b-q8-prefix-nobos.json).
        A prompt that already starts with ``<bos>`` (an instruct template, or
        a client that sent it) is left unchanged.
        """
        ids = super().prompt_tokens(request)
        if "_mlx2_prompt_tokens" in request:
            return ids
        bos = getattr(self.processor.tokenizer, "bos_token_id", None)
        if bos is not None and (not ids or ids[0] != bos):
            ids = [int(bos), *ids]
        return ids

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

        media, replacements, images, videos, video_fps = [], [], [], [], []
        markers = (self.processor.image_token, self.processor.video_token)
        for message in request["messages"]:
            content = message.get("content")
            for part in content if isinstance(content, list) else ():
                part_type = part.get("type")
                if part_type == "text":
                    # The processor pairs every marker with one media item;
                    # a literal marker in user text ran its replacement
                    # iterator dry (StopIteration, a server error).
                    if any(marker in str(part.get("text", "")) for marker in markers):
                        raise ValueError("Gemma 4 text must not contain a media placeholder")
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
                    video_fps.append(_stamp_fps(value.metadata["timestamps_seconds"]))
                    replacements.append(self.processor.video_token)
        if not media:
            return request
        messages = _plain_messages(request["messages"], replacements)
        prompt = self._render(messages)
        # Markers in string-content messages reach the processor too.
        _require_media_markers(prompt, "Gemma 4", (
            (self.processor.image_token, len(images)),
            (self.processor.video_token, len(videos)),
        ))
        processed = self.processor(
            text=prompt, images=images or None,
            videos=videos or None, fps=video_fps or None,
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
    # Serving prefill loop on the serving (merged Batch*) caches, 8-bit
    # artifact, ABBA medians of 4 after a discarded warm-up
    # (qualification/runs/gemma4-defaults-20260925/results/
    # prefill-26b-q8-batchcache.json): 16K 512/1024/2048/4096 ->
    # 2282/2703/2952/2831 tok/s, peak above weights 1.3/1.7/2.6/4.3 GiB.
    # Plain caches agree (prefill-26b-q8.json).  The MoE wants wide chunks;
    # 2048 is the fastest step, so the engine-wide default is kept, declared.
    default_prefill_step = 2048
    sampling_defaults = VendorSampling.single(
        SamplingDefaults(temperature=1.0, top_p=0.95, top_k=64, source=GENERATION_CONFIG),
        model="google/gemma-4-26B-A4B",
    )


class Gemma431BAdapter(_Gemma4Adapter):
    descriptor = GEMMA4_31B
    # Same harness on the serving (merged Batch*) caches
    # (results/prefill-31b-q8-batchcache.json): 16K 512/1024/2048 ->
    # 467/450/430 tok/s, peak above weights 3.4/4.3/6.3 GiB; 4K 511/488/463.
    # On plain caches (results/prefill-31b-q8.json) the order is the same out
    # to 8192 (445/434/423/388/327 tok/s at 16K).  Each sliding layer holds
    # window + chunk tokens, so on the dense model a small chunk is both
    # faster and lighter than the engine-wide 2048.
    default_prefill_step = 512
    sampling_defaults = VendorSampling.single(
        SamplingDefaults(temperature=1.0, top_p=0.95, top_k=64, source=GENERATION_CONFIG),
        model="google/gemma-4-31B",
    )

    @staticmethod
    def int8_prefill_supported():
        # Text decoder attention (q/k/v/o; full-attention layers share K and V
        # and have no v_proj) and dense MLP projections, selected by
        # ``int8_prefill_select``.  Approximate and qualification-gated like
        # every int8 prefill route; evidence in
        # qualification/runs/int8-dense8-e2e-20261009/gemma-4-31B-MLX-8bit.
        return ("mlp", "all")
