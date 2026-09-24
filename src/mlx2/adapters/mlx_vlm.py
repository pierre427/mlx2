"""Optional mlx-vlm execution adapters for Gemma 3n and MiniCPM-o.

Source mechanisms are mined from mlx-vlm revision
653f1f13e238abb313fd45071bbd04b3de414635 (MIT).  mlx2 modifications:
native video planning with timing, APCv2 media identity, isolated multimodal
prefill, MiniCPM-o slicing/vision batching/audio chunking, and fail-closed TTS.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..multimodal import media_fingerprint, resolve_media
from ..output import OutputParser
from ..sampling_defaults import (
    GENERATION_CONFIG,
    MODEL_CARD,
    SamplingDefaults,
    VendorSampling,
)
from .multimodal import (
    Gemma3nVideoPolicy,
    MediaFeatureCache,
    MiniCPMOExecutionPolicy,
    NativeVideoInput,
    install_gemma3n_vision_batching,
    install_media_feature_cache,
    install_minicpmo_vision_batching,
)

_MINICPMO_LOAD_LOCK = threading.Lock()

GEMMA3N_SAMPLING = VendorSampling.single(
    SamplingDefaults(top_p=0.95, top_k=64, source=GENERATION_CONFIG),
    model="google/gemma-3n-E2B-it",
)
MINICPMO_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=0.5,
        source=MODEL_CARD,
        note="MiniCPM-o 2.6 model-card omni inference example",
    ),
    model="openbmb/MiniCPM-o-2_6",
)


def inspect_artifact(model_path, *, expected):
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != expected:
        raise ValueError(f"expected {expected} artifact")
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("multimodal artifact has no safetensors weights")
    digest = hashlib.sha256((path / "config.json").read_bytes())
    records = []
    for item in files:
        stat = item.stat()
        record = (item.name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    text = config.get("text_config") or config
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": records,
        "model_type": expected,
        "max_context": int(text.get("max_position_embeddings", 32768)),
        "qualification": "pending",
        "supports_self_mtp": False,
        "config": config,
    }


def _source(part, kind):
    if kind == "image":
        return part.get("image_url") or part.get("file_id")
    if kind == "video":
        return part.get("video_url") or part.get("file_id")
    value = part.get("input_audio") or part.get("audio_url") or part.get("file_id")
    if isinstance(value, dict) and "data" in value:
        fmt = value.get("format", "wav").lower()
        value = f"data:audio/{fmt};base64,{value['data']}"
    return value


def _unmapped_part(family, part_type):
    """Refuse a media part the adapter has no encoder for.

    Skipping the part would leave the prompt one media replacement short, so
    every later marker would land at the wrong part.
    """
    if part_type == "input_video":
        from ..api_resources import CapabilityUnavailable

        return CapabilityUnavailable(f"{family} has no qualified video input")
    return ValueError(f"{family} does not accept {part_type!r} content parts")


def _plain_messages(messages, replacements):
    converted = []
    cursor = iter(replacements)
    missing = object()
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = []
            for part in content:
                if part["type"] == "text":
                    text.append(part.get("text", ""))
                else:
                    replacement = next(cursor, missing)
                    if replacement is missing:
                        raise RuntimeError(
                            "multimodal replacements are fewer than media parts"
                        )
                    text.append(replacement)
            content = "\n".join(value for value in text if value)
        converted.append({**message, "content": content})
    if next(cursor, missing) is not missing:
        raise RuntimeError("multimodal replacements outnumber media parts")
    return converted


def _ids_and_kwargs(processed):
    data = dict(processed)
    ids = data.pop("input_ids")
    data.pop("attention_mask", None)
    data.pop("token_type_ids", None)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids], data


def _media_token_end(processed) -> int:
    """One-past-last media placeholder required before encoder-free resume."""
    import numpy as np

    token_types = processed.get("mm_token_type_ids")
    if token_types is None:
        token_types = processed.get("token_type_ids")
    if token_types is not None:
        values = np.asarray(token_types)
        positions = np.where(values.reshape(-1) != 0)[0]
        if positions.size:
            return int(positions[-1]) + 1
    end = 0
    for name in ("image_bound", "audio_bounds"):
        collection = processed.get(name)
        if collection is None:
            continue
        for bounds in collection:
            values = np.asarray(bounds).reshape(-1, 2)
            if values.size:
                end = max(end, int(values[:, 1].max()))
    return end


class _LogitsModel:
    """Expose mlx-vlm models through mlx2's tensor-returning model contract."""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)

    def __call__(self, *args, **kwargs):
        output = self._model(*args, **kwargs)
        return getattr(output, "logits", output)


def _load_model(load, model_path, *, expected, config):
    """Apply narrowly scoped released-artifact compatibility at load time."""
    if expected == "gemma4":
        return load(str(model_path), lazy=False, strict=True, trust_remote_code=False)
    if expected != "minicpmo":
        return load(str(model_path), lazy=False)
    text = config.get("text_config") or config
    if text.get("head_dim") is not None:
        return load(str(model_path), lazy=False)
    hidden = int(text["hidden_size"])
    heads = int(text["num_attention_heads"])
    if hidden % heads:
        raise ValueError("MiniCPM-o hidden size must divide evenly across heads")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_names = index.get("weight_map", {})
    attention_bias = any(
        name.endswith("self_attn.q_proj.bias") for name in weight_names
    )

    from mlx_vlm.models.minicpmo import minicpmo as minicpmo_module
    from mlx_vlm.models.minicpmo.config import ModelConfig, TextConfig
    from mlx_vlm.models.qwen2.language import LanguageModel as Qwen2LanguageModel

    with _MINICPMO_LOAD_LOCK:
        descriptor = ModelConfig.__dict__["from_dict"]
        original = ModelConfig.from_dict
        original_text = TextConfig.from_dict
        original_language_model = minicpmo_module.LanguageModel

        def normalized_text(cls, params):
            normalized_config = original_text(params)
            normalized_config.rope_traditional = False
            return normalized_config

        def normalized(cls, params):
            nested = params.get("text_config")
            if nested:
                nested.setdefault("head_dim", hidden // heads)
                nested.setdefault("attention_bias", attention_bias)
            else:
                params.setdefault("head_dim", hidden // heads)
                params.setdefault("attention_bias", attention_bias)
            normalized_config = original(params)
            normalized_config.text_config.rope_traditional = False
            return normalized_config

        ModelConfig.from_dict = classmethod(normalized)
        TextConfig.from_dict = classmethod(normalized_text)

        class MiniCPMOLanguageModel(Qwen2LanguageModel):
            def __init__(self, args, config=None):
                super().__init__(args)

        minicpmo_module.LanguageModel = MiniCPMOLanguageModel
        try:
            return load(str(model_path), lazy=False)
        finally:
            ModelConfig.from_dict = descriptor
            del TextConfig.from_dict
            minicpmo_module.LanguageModel = original_language_model


class _MLXVLMAdapter:
    default_route = "ordinary"
    reasoning_effort_semantics = "boolean"

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError("multimodal adapters have no native MTP route")
        return "mlx-vlm-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    @staticmethod
    def streaming_detokenizer_class():
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer

        return BPEStreamingDetokenizer

    @staticmethod
    def _wrap_model(model):
        return _LogitsModel(model)

    def __init__(self, model_path, *, execution_policy=None):
        if execution_policy:
            raise ValueError("mlx-vlm adapters do not accept speculative policy")
        self.identity = inspect_artifact(model_path, expected=self.descriptor.model_type)
        self.environment = {}
        self.max_context = self.identity["max_context"]
        self.layout = self.descriptor.cache_layout
        try:
            from mlx_vlm import load
        except ImportError as error:
            raise RuntimeError("multimodal adapters require the optional mlx-vlm runtime") from error
        model, self.processor = _load_model(
            load,
            Path(model_path).resolve(),
            expected=self.descriptor.model_type,
            config=self.identity["config"],
        )
        self.model = self._wrap_model(model)
        self.media_feature_cache = MediaFeatureCache()
        tokenizer = self.processor.tokenizer
        from ..runtime.tokenizer_utils import TokenizerWrapper

        eos = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        eos = {int(eos)} if isinstance(eos, int) else {int(v) for v in (eos or ())}
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=self.streaming_detokenizer_class(),
            eos_token_ids=eos,
        )

    def _render(self, messages):
        processor_renderer = getattr(self.processor, "apply_chat_template", None)
        tokenizer_renderer = getattr(
            getattr(self.processor, "tokenizer", None), "apply_chat_template", None
        )
        if getattr(self.processor, "chat_template", None) and callable(processor_renderer):
            renderer = processor_renderer
        elif callable(tokenizer_renderer):
            renderer = tokenizer_renderer
        elif callable(processor_renderer):
            renderer = processor_renderer
        else:
            raise ValueError("multimodal processor has no chat template renderer")
        return renderer(messages, tokenize=False, add_generation_prompt=True)

    def prompt_tokens(self, request):
        prepared = request.get("_mlx2_prompt_tokens")
        if prepared is not None:
            return list(prepared)
        return self.tokenizer.encode(
            self.render_prompt(request), add_special_tokens=False
        )

    def render_prompt(self, request):
        """Text-prompt rendering; media requests carry prepared token ids."""
        if "_mlx2_prompt_tokens" in request:
            raise ValueError("prepared multimodal prompts have no text rendering")
        if "messages" in request:
            return self._render(request["messages"])
        return request["prompt"]

    def output_parser(self, request):
        return OutputParser(
            chat="messages" in request,
            thinking=bool(request.get("enable_thinking")),
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        return {
            "architecture": self.descriptor.model_type,
            "multimodal": True,
            "media_feature_cache": self.media_feature_cache.snapshot(),
        }

    def invalidate_feature_cache(self):
        self.media_feature_cache.clear()

    def close(self):
        pass


GEMMA3N = ModelDescriptor(
    model_type="gemma3n",
    family="gemma-3n",
    variant="mlx-vlm-native-video",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.VISION, Capability.VIDEO, Capability.AUDIO, Capability.STREAMING, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE, Capability.APC_V2}),
    cache_layout="gemma3n-mlx-vlm-media-v1",
    metadata={
        "execution": "mlx2.adapters.mlx_vlm.Gemma3nAdapter",
        "qualification": "pending",
        "processor_qualification": "qualification/runs/multimodal-processors-20260918/receipt.json",
        "required_qualification_checks": (
            "multimodal_image",
            "multimodal_video",
            "multimodal_audio_input",
            "multimodal_encoder_batching",
            "multimodal_apcv2_reuse",
        ),
    },
)


class Gemma3nAdapter(_MLXVLMAdapter):
    descriptor = GEMMA3N
    sampling_defaults = GEMMA3N_SAMPLING

    @staticmethod
    def streaming_detokenizer_class():
        from ..runtime.tokenizer_utils import SPMStreamingDetokenizer

        return SPMStreamingDetokenizer

    def __init__(self, model_path, *, execution_policy=None):
        super().__init__(model_path, execution_policy=execution_policy)
        self.video_policy = Gemma3nVideoPolicy()
        install_gemma3n_vision_batching(self.model, self.video_policy)
        install_media_feature_cache(self.model, self.media_feature_cache, family="gemma3n")

    def prepare_multimodal_request(self, request, *, file_loader=None):
        media, replacements, images, audios = [], [], [], []
        video_requests = video_frames = video_frame_batches = 0
        # pcm_to_float32 never resamples, so a WAV at any other rate than the
        # feature extractor's would be encoded as a time-stretched clip.
        extractor = getattr(self.processor, "feature_extractor", None)
        audio_sample_rate = int(getattr(extractor, "sampling_rate", None) or 16_000)
        for message in request["messages"]:
            for part in message.get("content", ()) if isinstance(message.get("content"), list) else ():
                kind = {"image_url": "image", "input_image": "image", "input_audio": "audio", "input_video": "video"}.get(part["type"])
                if part["type"] == "text":
                    continue
                if kind is None:
                    raise _unmapped_part("Gemma 3n", part["type"])
                value = resolve_media(_source(part, kind), kind=kind, file_loader=file_loader, fps=self.video_policy.fps, max_frames=self.video_policy.max_frames) if kind == "video" else resolve_media(_source(part, kind), kind=kind, file_loader=file_loader)
                media.append(value)
                if kind == "image":
                    images.append(value.value); replacements.append(self.processor.tokenizer.image_token)
                elif kind == "audio":
                    from ..multimodal import pcm_to_float32
                    if int(value.metadata["sample_rate"]) != audio_sample_rate:
                        raise ValueError(
                            f"Gemma 3n requires {audio_sample_rate} Hz WAV audio; explicit resampling is required"
                        )
                    audios.append(pcm_to_float32(value)); replacements.append(self.processor.tokenizer.audio_token)
                else:
                    native = NativeVideoInput.from_media(value, self.video_policy)
                    inputs = native.processor_inputs("", self.processor.tokenizer.image_token)
                    images.extend(inputs["images"]); replacements.append(inputs["text"])
                    video_requests += 1
                    video_frames += len(inputs["images"])
                    video_frame_batches += math.ceil(
                        len(inputs["images"]) / self.video_policy.frame_batch_size
                    )
        messages = _plain_messages(request["messages"], replacements)
        prompt = self._render(messages)
        processed = self.processor(text=prompt, images=images or None, audio=audios or None, sampling_rate=audio_sample_rate)
        media_token_end = _media_token_end(processed)
        ids, kwargs = _ids_and_kwargs(processed)
        if images:
            kwargs["_mlx2_vision_cache_key"] = f"{self.identity['fingerprint']}:{media_fingerprint([value for value in media if value.kind in {'image', 'video'}], policy=asdict(self.video_policy))}"
        return {**request, "messages": messages, "_mlx2_prompt_tokens": ids, "_mlx2_prefill_inputs": kwargs, "_mlx2_media_token_end": media_token_end, "_mlx2_media_fingerprint": media_fingerprint(media, policy={"family": "gemma3n", "video": asdict(self.video_policy)}), "_mlx2_multimodal_stats": {"gemma3n_video_requests": video_requests, "gemma3n_video_frames": video_frames, "gemma3n_video_frame_batches": video_frame_batches}}


MINICPMO = ModelDescriptor(
    model_type="minicpmo",
    family="minicpm-o",
    variant="mlx-vlm-optimized-media",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.VISION, Capability.AUDIO, Capability.STREAMING, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE, Capability.APC_V2}),
    cache_layout="minicpmo-mlx-vlm-media-v1",
    metadata={
        "execution": "mlx2.adapters.mlx_vlm.MiniCPMOAdapter",
        "qualification": "pending",
        "processor_qualification": "qualification/runs/multimodal-processors-20260918/receipt.json",
        "required_qualification_checks": (
            "multimodal_image",
            "multimodal_audio_input",
            "multimodal_encoder_batching",
            "multimodal_apcv2_reuse",
        ),
        "output_audio": "fail_closed_unqualified",
    },
)


class MiniCPMOAdapter(_MLXVLMAdapter):
    descriptor = MINICPMO
    sampling_defaults = MINICPMO_SAMPLING

    def __init__(self, model_path, *, execution_policy=None):
        super().__init__(model_path, execution_policy=execution_policy)
        self.media_policy = MiniCPMOExecutionPolicy.from_config(self.identity["config"])
        install_minicpmo_vision_batching(self.model, self.media_policy)
        install_media_feature_cache(self.model, self.media_feature_cache, family="minicpmo")

    def prepare_multimodal_request(self, request, *, file_loader=None):
        media, replacements, images, audios = [], [], [], []
        vision_slices = audio_chunks = 0
        for message in request["messages"]:
            for part in message.get("content", ()) if isinstance(message.get("content"), list) else ():
                kind = {"image_url": "image", "input_image": "image", "input_audio": "audio"}.get(part["type"])
                if part["type"] == "text":
                    continue
                if kind is None:
                    raise _unmapped_part("MiniCPM-o", part["type"])
                value = resolve_media(_source(part, kind), kind=kind, file_loader=file_loader)
                media.append(value)
                if kind == "image":
                    tiles = self.media_policy.slice_image(value.value)
                    images.extend(tiles)
                    replacements.append("".join("<image>" for _ in tiles))
                    vision_slices += len(tiles)
                else:
                    chunks = self.media_policy.audio_chunks(value)
                    audios.extend(chunks)
                    replacements.append("".join("<audio>" for _ in chunks))
                    audio_chunks += len(chunks)
        messages = _plain_messages(request["messages"], replacements)
        processed = self.processor(text=self._render(messages), images=images or None, audios=audios or None)
        media_token_end = _media_token_end(processed)
        ids, kwargs = _ids_and_kwargs(processed)
        if images:
            kwargs["_mlx2_vision_cache_key"] = f"{self.identity['fingerprint']}:{media_fingerprint([value for value in media if value.kind == 'image'], policy=self.media_policy.receipt())}"
        vision_batches = math.ceil(vision_slices / self.media_policy.vision_batch_size) if self.media_policy.batch_vision_input else vision_slices
        return {**request, "messages": messages, "_mlx2_prompt_tokens": ids, "_mlx2_prefill_inputs": kwargs, "_mlx2_media_token_end": media_token_end, "_mlx2_media_fingerprint": media_fingerprint(media, policy={"family": "minicpmo", **self.media_policy.receipt()}), "_mlx2_multimodal_stats": {"minicpmo_vision_batches": vision_batches, "minicpmo_vision_slices": vision_slices, "minicpmo_audio_chunks": audio_chunks}}

    def diagnostics(self):
        return {**super().diagnostics(), "media_policy": self.media_policy.receipt(), "output_audio": "unqualified_fail_closed"}
