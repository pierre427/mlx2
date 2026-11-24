"""Ling-3.2-air vision mode: image prompts, request-owned 3D M-RoPE, image prefill.

Opt-in (``ling_vision: true`` in the execution policy).  Off, the adapter is text-only
and unchanged.  On, the model is wrapped so every request's cache list carries one
extra plane (``LingMRoPECache``, the Qwen2.5-VL request-owned RoPE mechanism of
``qwen25_rope``) that travels through batch merge, extraction, trimming and APCv2
serialization.  A text-only request in vision mode keeps the scalar RoPE path: its
positions are ``slot + 0``.  An image request prefills once, cold and isolated,
through its last image token; afterwards its text positions are ``slot + delta``.

Laws (positions, frequencies, prompt expansion, projector) are the vendor's
(inclusionAI/Ling-3.2-air @1613b1a4, MIT); see provenance/ling32-air-vl.json.
"""

from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np

from .qwen25_rope import Qwen25RoPECache

TAG = "ling32-mrope-v1"
IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
TEMPLATE_IMAGE = VISION_START + IMAGE_PAD + VISION_END
MAX_IMAGES = 8
# Tower attention is full (non-causal) per image over its patches; bound one
# request's encoder work until the tower's memory is charged at admission.
# 8192 patches = e.g. one 2048x1024 image (vendor resize law unchanged).
MAX_PATCHES = 8192


class LingMRoPECache(Qwen25RoPECache):
    """One row per lane: ``length`` slots, text ``delta``, ``media_end``."""

    @property
    def meta_state(self):
        return (TAG, json.dumps(self.rows, separators=(",", ":")))

    @meta_state.setter
    def meta_state(self, value):
        if len(value) != 2 or value[0] != TAG:
            raise ValueError("incompatible Ling M-RoPE state")
        Qwen25RoPECache.meta_state.fset(self, ("qwen25-mrope-v1", value[1]))


class LingVisionModel:
    """Wrap a bailing_hybrid ``Model``; supply request-owned positions per forward."""

    def __init__(self, model, vision, projector, config):
        self._model = model
        self.vision_tower = vision
        self.projector = projector
        self.image_token_id = int(config["image_token_id"])
        self.vision_start_token_id = int(config["vision_start_token_id"])
        self.merge = int(config["vision_config"]["spatial_merge_size"])
        if model.args.mrope_section is None:
            raise ValueError("Ling vision mode needs the artifact's mrope_section")
        self.vision_stats = {"encoder_calls": 0, "images": 0, "patches": 0,
                             "feature_tokens": 0, "media_prefills": 0,
                             "shifted_text_forwards": 0, "plain_text_forwards": 0}

    def __getattr__(self, name):
        return getattr(self._model, name)

    def make_cache(self):
        return [*self._model.make_cache(), LingMRoPECache()]

    def image_features(self, pixel_values, grids):
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        features = self.vision_tower(mx.array(pixel_values).astype(dtype), [list(g) for g in grids])
        features = self.projector(features)
        self.vision_stats["encoder_calls"] += 1
        self.vision_stats["images"] += len(grids)
        self.vision_stats["patches"] += int(sum(math.prod(g) for g in grids))
        self.vision_stats["feature_tokens"] += int(features.shape[0])
        return features

    def _plane(self, cache):
        if cache is None or not cache or not isinstance(cache[-1], LingMRoPECache):
            raise ValueError("Ling vision mode requires a request-owned M-RoPE cache plane")
        return cache[-1]

    def __call__(self, inputs, cache=None, **media):
        from ..runtime.models.ling_mrope import find_spans, slot_positions, text_delta
        from ..runtime.prefill_checkpoint_rows import select_checkpoint_rows

        rope = self._plane(cache)
        layers = cache[:-1]
        batch, width = inputs.shape
        if len(rope.rows) != batch:
            raise ValueError("Ling M-RoPE row count mismatch")
        media_end = media.pop("_mlx2_rope_media_end", None)
        pixels = media.pop("pixel_values", None)
        grids = media.pop("image_grid_thw", None)
        if media:
            raise ValueError(f"Ling vision mode got unsupported prefill inputs {sorted(media)}")
        if pixels is not None:
            if batch != 1 or rope.rows[0]["length"]:
                raise ValueError("Ling image prefill must start at a cold isolated boundary")
            if type(media_end) is not int or not 0 < media_end <= width:
                raise ValueError("Ling image prefill must cover its media boundary")
            grids = [[int(v) for v in g] for g in (grids.tolist() if hasattr(grids, "tolist") else grids)]
            tokens = inputs[0].tolist()
            spans = find_spans(tokens, image_token_id=self.image_token_id,
                               start_token_id=self.vision_start_token_id, grids=grids,
                               merge=self.merge)
            if max(s[0] + s[2] * s[3] * s[4] for s in spans) != media_end:
                raise ValueError("Ling media boundary is not the end of the last image")
            positions = slot_positions(spans, np.arange(width))
            features = self.image_features(pixels, grids)
            where = mx.array([i for i, t in enumerate(tokens) if t == self.image_token_id], mx.int32)
            if where.size != features.shape[0]:
                raise ValueError("Ling image features and placeholder tokens differ")
            embeds = self._model.model.word_embeddings(inputs)
            embeds[0, where] = features.astype(embeds.dtype)
            hidden = self._model.model(inputs, layers, inputs_embeds=embeds,
                                       rope_positions=positions)
            rope.rows[0].update(length=width, delta=text_delta(spans, width), media=True,
                                media_end=media_end)
            self.vision_stats["media_prefills"] += 1
        else:
            if media_end is not None:
                raise ValueError("Ling media boundary without its image payload")
            if any(t == self.image_token_id for t in inputs.reshape(-1).tolist()):
                raise ValueError("Ling image tokens cannot run without their image payload")
            deltas = [row["delta"] for row in rope.rows]
            if not any(deltas):
                hidden = self._model.model(inputs, layers)
                self.vision_stats["plain_text_forwards"] += 1
            else:
                from ..runtime.models.bailing_hybrid import _cache_offset

                attention = next(c for c, layer in zip(layers, self._model.model.layers)
                                 if layer.is_global)
                offset = _cache_offset(attention)
                if isinstance(offset, int):
                    if batch != 1:
                        raise ValueError("Ling shifted text rows need per-row cache offsets")
                    positions = offset + deltas[0]
                else:
                    positions = offset + mx.array(deltas, dtype=offset.dtype)
                hidden = self._model.model(inputs, layers, rope_positions=positions)
                self.vision_stats["shifted_text_forwards"] += 1
            for row in rope.rows:
                row["length"] += width
        output = select_checkpoint_rows(hidden)
        if self._model.args.tie_word_embeddings:
            return self._model.model.word_embeddings.as_linear(output)
        return self._model.lm_head(output)

    prefill_forward = __call__


def template_messages(messages):
    """OpenAI image parts -> the template's ``{"type": "image"}`` items, in order."""
    converted, sources = [], []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif kind in {"image_url", "input_image"}:
                sources.append(part.get("image_url") or part.get("file_id"))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"Ling-3.2-air vision mode accepts still images only, not {kind!r}")
        converted.append({**message, "content": parts})
    return converted, sources


def expand_image_tokens(prompt, grids, merge):
    """The vendor ``_expand_image_tokens``: each template image becomes
    start + N pads + end + newline, N = t*h*w / merge**2, in prompt order."""
    if prompt.count(TEMPLATE_IMAGE) != len(grids) or prompt.count(IMAGE_PAD) != len(grids):
        raise ValueError("Ling-3.2-air text must not contain an image placeholder")
    pieces = prompt.split(TEMPLATE_IMAGE)
    out = pieces[0]
    for grid, rest in zip(grids, pieces[1:]):
        count = math.prod(int(v) for v in grid) // (merge * merge)
        out += VISION_START + IMAGE_PAD * count + VISION_END + "\n" + rest
    return out


def prepare(adapter, request, *, file_loader=None):
    from ..multimodal import media_fingerprint, resolve_media
    from .ling32_air_images import process_images

    if "messages" not in request:
        return request
    # The adapter's own template history first (OpenAI tool-call argument
    # strings become objects), then image parts become template image items.
    messages, sources = template_messages(adapter._template_messages(request))
    if not sources:
        return {**request, "messages": messages}
    if not adapter.vision_enabled:
        from ..api_resources import CapabilityUnavailable

        raise CapabilityUnavailable("Ling-3.2-air vision mode is off (execution policy ling_vision)")
    if len(sources) > MAX_IMAGES:
        raise ValueError(f"Ling-3.2-air vision mode accepts at most {MAX_IMAGES} images per request")
    media = [resolve_media(source, kind="image", file_loader=file_loader) for source in sources]
    pixels, grids = process_images([m.value for m in media], adapter.image_policy)
    if int(pixels.shape[0]) > MAX_PATCHES:
        raise ValueError(f"Ling-3.2-air vision mode accepts at most {MAX_PATCHES} image patches "
                         f"per request (got {int(pixels.shape[0])}); send smaller images")
    prompt = adapter.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=adapter.thinking_enabled(request), tools=adapter._template_tools(request),
    )
    expanded = expand_image_tokens(prompt, grids, adapter.image_policy["merge_size"])
    ids = adapter.tokenizer.encode(expanded, add_special_tokens=False)
    token = adapter.config["image_token_id"]
    end = max(i + 1 for i, value in enumerate(ids) if value == token)
    if end >= len(ids):
        raise ValueError("Ling-3.2-air image placeholder cannot be the final prompt token")
    identity = media_fingerprint(media, policy=adapter.image_policy)
    return {**request, "messages": messages, "_mlx2_prompt_tokens": ids,
            "_mlx2_prefill_inputs": {"pixel_values": pixels, "image_grid_thw": grids,
                                     "_mlx2_rope_media_end": end},
            "_mlx2_media_token_end": end, "_mlx2_media_fingerprint": identity}
