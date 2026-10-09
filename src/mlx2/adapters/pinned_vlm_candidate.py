"""Pinned mlx-vlm candidates with CPU artifact gates and media-bound prefill."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..multimodal import media_fingerprint, resolve_media
from .mlx_vlm import (
    MediaFeatureCache, _LogitsModel, _ids_and_kwargs, _plain_messages,
    _require_media_markers, _source,
)

# Not the project pin (mlx_vlm_pin.MLX_VLM_REVISION, which pyproject installs
# and which does not descend from this revision): this adapter loads only when
# a clean checkout of SOURCE_REVISION is first on PYTHONPATH, and fails closed
# naming both revisions otherwise (sweep 2026-10-06 G2-04/G5-05).
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"

TOPOLOGIES = {
    "agnes": {
        "family": "agnes-3-flash-vision", "layers": 72, "width": 5120,
        "vision_width": 1152, "vision_depth": 27,
        "required": ("language_model.model.embed_tokens.weight",
                     "language_model.model.layers.71.global_attn.q_proj.weight",
                     "vision_tower.patch_embed.proj.weight",
                     "vision_tower.blocks.26.attn.qkv.weight"),
        "cache_layout": "agnes-vision-hybrid-candidate-v1",
    },
    "smolvlm": {
        "family": "smolvlm2-256m", "layers": 30, "width": 576,
        "vision_width": 768, "vision_depth": None,
        "required": ("model.text_model.embed_tokens.weight",
                     "model.text_model.layers.29.self_attn.q_proj.weight",
                     "model.connector.modality_projection.proj.weight",
                     "model.vision_model.embeddings.patch_embedding.weight"),
        "cache_layout": "smolvlm2-image-candidate-v1",
    },
    "qwen2_5_vl": {
        "family": "qwen2.5-vl-3b", "layers": 36, "width": 2048,
        "vision_width": 1280, "vision_depth": 32,
        "required": ("model.embed_tokens.weight", "model.layers.35.self_attn.q_proj.weight",
                     "visual.patch_embed.proj.weight", "visual.blocks.31.attn.qkv.weight"),
        "cache_layout": "qwen25-vl-mrope-candidate-v1",
    },
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _single_header(path: Path) -> set[str]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors file")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= min(64 << 20, size - 8):
            raise ValueError("invalid safetensors header")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header object")
    return set(header) - {"__metadata__"}


def inspect_vision_artifact(model_path: str | Path, *, expected: str) -> dict:
    if expected not in TOPOLOGIES:
        raise ValueError("unknown candidate topology")
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    spec = TOPOLOGIES[expected]
    text = config.get("text_config") or config
    vision = config.get("vision_config")
    if config.get("model_type") != expected or not isinstance(vision, dict):
        raise ValueError(f"expected {expected} vision checkpoint")
    if text.get("num_hidden_layers") != spec["layers"] or text.get("hidden_size") != spec["width"]:
        raise ValueError("text topology mismatch")
    if vision.get("hidden_size") != spec["vision_width"]:
        raise ValueError("vision topology mismatch")
    depth = spec["vision_depth"]
    if depth is not None and vision.get("depth") != depth:
        raise ValueError("vision depth mismatch")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = _json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("missing indexed tensors")
        names = set(weight_map)
        files = sorted(set(weight_map.values()))
    else:
        files = ["model.safetensors"]
        names = _single_header(path / files[0])
    if not set(spec["required"]) <= names:
        raise ValueError("vision or text tensors are incomplete")
    digest = hashlib.sha256()
    records = []
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "chat_template.jinja"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for name in files:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard escapes artifact")
        item = (path / name).resolve()
        hub_blob_root = path.parent.parent / "blobs" if path.parent.name == "snapshots" else None
        trusted = item.is_relative_to(path) or (
            hub_blob_root is not None and item.is_relative_to(hub_blob_root)
        )
        if not trusted or not item.is_file() or item.stat().st_size < 8:
            raise ValueError("missing weight shard")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {"path": str(path), "fingerprint": digest.hexdigest(), "files": records,
            "config": config, "tensor_count": len(names),
            "max_context": int(text.get("max_position_embeddings", 32768)),
            "qualification": "pending", "selected": False,
            "vision_present": True, "embedded_mtp": False}


def descriptor_for(expected: str) -> ModelDescriptor:
    spec = TOPOLOGIES[expected]
    planes = {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    if expected == "agnes":
        planes.add(StatePlane.RECURRENT)
    return ModelDescriptor(
        model_type=expected, family=spec["family"], variant="embedded-vision-candidate",
        state_planes=frozenset(planes),
        capabilities=frozenset({Capability.TEXT, Capability.VISION, Capability.VIDEO,
                                Capability.STREAMING}),
        cache_layout=spec["cache_layout"],
        metadata={"qualification": "pending", "source_revision": SOURCE_REVISION,
                  "required_qualification_checks": ("image_parity", "video_parity",
                       "media_token_alignment", "multimodal_state_replay",
                       "multimodal_apcv2_reuse")},
    )


class _CandidateLogitsModel(_LogitsModel):
    def make_cache(self):
        inner = getattr(self._model, "language_model", None)
        if inner is not None and hasattr(inner, "make_cache"):
            return inner.make_cache()
        from ..runtime.models.cache import make_prompt_cache
        return make_prompt_cache(self._model)


class PinnedVisionCandidateAdapter:
    default_route = "ordinary"
    model_type: str

    def __init__(self, model_path: str, *, execution_policy=None):
        # Refuse an explicitly enabled TF32 switch like the standard decoder
        # (process_env.require_process_numerics); MLX latches it.
        from ..process_env import require_process_numerics

        require_process_numerics(f"the {self.model_type} vision candidate adapter")
        if execution_policy not in (None, {}):
            raise ValueError("vision candidate has no qualified execution policy")
        self.identity = inspect_vision_artifact(model_path, expected=self.model_type)
        from .mlx_vlm_pin import mlx_vlm_runtime
        runtime = mlx_vlm_runtime()
        if runtime is None or runtime.get("revision") != SOURCE_REVISION:
            from .mlx_vlm_pin import MLX_VLM_REVISION

            raise RuntimeError(
                f"vision candidate requires mlx-vlm revision {SOURCE_REVISION}, found "
                f"{(runtime or {}).get('revision') or 'none'}; the project pin "
                f"installs {MLX_VLM_REVISION[:8]}, which is not it, so put a clean "
                f"{SOURCE_REVISION[:8]} checkout first on PYTHONPATH"
            )
        self.mlx_vlm_runtime = runtime
        self.environment = {"mlx_vlm_revision": SOURCE_REVISION}
        from mlx_vlm import load
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        model, self.processor = load(self.identity["path"], lazy=False, strict=True,
                                     trust_remote_code=False)
        from ..runtime.chat_templates import secure_model_chat_templates
        secure_model_chat_templates(self.processor)
        self.model = _CandidateLogitsModel(model)
        self.tokenizer = TokenizerWrapper(
            self.processor.tokenizer, detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=self._eos_ids(),
        )
        self.max_context = self.identity["max_context"]
        self.layout = self.descriptor.cache_layout
        self.media_feature_cache = MediaFeatureCache()
        self._media_proof_key = secrets.token_bytes(32)

    def _media_proof(self, tokens, media_fingerprint_value, media_end,
                     tower_digest=None):
        payload = json.dumps(
            [tokens, media_fingerprint_value, media_end, tower_digest],
            separators=(",", ":")
        ).encode()
        return hmac.new(self._media_proof_key, payload, hashlib.sha256).hexdigest()

    def has_trusted_media_preparation(self, request):
        tokens = request.get("_mlx2_prompt_tokens")
        media_identity = request.get("_mlx2_media_fingerprint")
        media_end = request.get("_mlx2_media_token_end")
        tower_digest = request.get("_mlx2_tower_inputs_digest")
        proof = request.get("_mlx2_media_proof")
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or not isinstance(media_identity, str) or not media_identity
                or type(media_end) is not int or not 0 < media_end <= len(tokens)
                or (tower_digest is not None and
                    (not isinstance(tower_digest, str) or len(tower_digest) != 64))
                or not isinstance(proof, str)):
            return False
        return hmac.compare_digest(
            proof, self._media_proof(tokens, media_identity, media_end, tower_digest)
        )

    def _eos_ids(self):
        from .eos import artifact_eos_token_ids

        return artifact_eos_token_ids(
            self.identity["path"], self.identity["config"],
            getattr(self.processor, "tokenizer", None),
        )

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("embedded MTP is not qualified for this vision candidate")
        return f"{self.model_type}-vision-ordinary-candidate"

    def execution_config(self, *, max_lanes, prefill_step):
        return {"persistent": True, "num_draft": 0, "backend": "ordinary",
                "rate_gate": False, "prefill_step_size": prefill_step,
                "segment_aware_live_tip": False, "segment_aware_cohort_size": max_lanes}

    def prompt_tokens(self, request):
        if "_mlx2_prompt_tokens" in request:
            return list(request["_mlx2_prompt_tokens"])
        return self.tokenizer.encode(self.render_prompt(request), add_special_tokens=False)

    def render_prompt(self, request):
        if "_mlx2_prompt_tokens" in request:
            raise ValueError("prepared media prompt has no plain text rendering")
        if "messages" in request:
            return self.processor.apply_chat_template(request["messages"],
                tokenize=False, add_generation_prompt=True)
        return request["prompt"]

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools
        if callable_tools(request):
            raise ValueError("vision candidate has no qualified tool route")
        return OutputParser(chat="messages" in request, stops=request.get("stop", ()))

    def _template_messages(self, messages, replacements):
        return _plain_messages(messages, replacements)

    def prepare_multimodal_request(self, request, *, file_loader=None):
        import numpy as np
        import mlx.core as mx

        media, replacements, images, videos = [], [], [], []
        video_metadata = []
        for message in request["messages"]:
            parts = message.get("content")
            for part in parts if isinstance(parts, list) else ():
                kind = part.get("type")
                if kind == "text":
                    continue
                if kind in ("input_image", "image_url"):
                    value = resolve_media(_source(part, "image"), kind="image", file_loader=file_loader)
                    images.append(value.value)
                    replacements.append(self.processor.image_token)
                elif kind == "input_video":
                    value = resolve_media(_source(part, "video"), kind="video",
                                          file_loader=file_loader, fps=1, max_frames=16)
                    if not value.value:
                        raise ValueError("empty decoded video")
                    frames = list(value.value)
                    shapes = {tuple(np.asarray(frame).shape) for frame in frames}
                    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
                        raise ValueError("video frames must have one consistent HWC shape")
                    if self.model_type == "smolvlm":
                        images.extend(frames)
                        replacements.append("\n".join(
                            f"[video frame {i + 1}] {self.processor.image_token}"
                            for i in range(len(frames))))
                    else:
                        videos.append(frames)
                        video_metadata.append({
                            "total_num_frames": int(value.metadata["source_frames"]),
                            "fps": float(value.metadata["source_fps"]),
                            "frames_indices": list(value.metadata["sampled_indices"]),
                        })
                        replacements.append(self.processor.video_token)
                else:
                    raise ValueError(f"vision candidate does not accept {kind!r}")
                media.append(value)
        if not media:
            return request
        if images and videos:
            raise ValueError("mixed image and video prefill is unqualified")
        messages = self._template_messages(request["messages"], replacements)
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
        _require_media_markers(prompt, "vision candidate", (
            (self.processor.image_token, len(images)),
            (getattr(self.processor, "video_token", None), len(videos)),
        ))
        processor_kwargs = {"text": prompt, "images": images or None,
                            "videos": videos or None}
        if self.model_type != "smolvlm":
            processor_kwargs["return_mm_token_type_ids"] = True
        if self.model_type == "agnes" and videos:
            processor_kwargs["video_metadata"] = video_metadata
            processor_kwargs["do_sample_frames"] = False
        processed = self.processor(**processor_kwargs)
        ids, kwargs = _ids_and_kwargs(processed)
        tower_digest = None
        if getattr(self, "_vision_feature_reuse_scope", None) == "tower_inputs_v1":
            from .vision_feature_reuse import tower_input_digest

            tower_digest = tower_input_digest(self.model_type, kwargs)
        kwargs = {key: mx.array(value) if isinstance(value, np.ndarray) else value
                  for key, value in kwargs.items()}
        media_ids = {int(self.identity["config"]["image_token_id"])}
        if self.model_type != "smolvlm":
            media_ids.add(int(self.identity["config"]["video_token_id"]))
        positions = [i for i, token in enumerate(ids) if token in media_ids]
        if not positions:
            raise ValueError("processor produced no media placeholders")
        fingerprint = media_fingerprint(
            media, policy={"family": self.model_type, "revision": SOURCE_REVISION,
                           "video_fps": 1, "video_max_frames": 16})
        media_end = positions[-1] + 1
        return {**request, "messages": messages, "_mlx2_prompt_tokens": ids,
                "_mlx2_prefill_inputs": kwargs, "_mlx2_media_token_end": positions[-1] + 1,
                "_mlx2_media_fingerprint": fingerprint,
                "_mlx2_tower_inputs_digest": tower_digest,
                "_mlx2_media_proof": self._media_proof(
                    ids, fingerprint, media_end, tower_digest)}

    def diagnostics(self):
        return {"architecture": self.model_type, "qualification": "pending",
                "mlx_vlm": self.mlx_vlm_runtime}

    def cold_media_feature_peak_increment_bytes(self) -> int:
        """Bound for one newly cache-retained feature before LRU eviction.

        Live headroom already counts current cached features.  A miss first
        computes one feature tensor and only then evicts, so reserve one full
        cache-admissible result for the opt-in tower-only cache. This does not
        bound uncached source vision activations or oversize skipped results.
        """
        if getattr(self, "_vision_feature_reuse_scope", None) != "tower_inputs_v1":
            return 0
        return int(self.media_feature_cache.max_bytes)

    def close(self):
        self.media_feature_cache.clear()
        self.model = None
        self.processor = None
        self.tokenizer = None
