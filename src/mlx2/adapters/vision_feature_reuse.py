"""Candidate exact-order vision feature reuse for pinned Smol and Qwen VL.

This helper is deliberately inert until an adapter installs it and serving
certifies the entire cold media prefill.  The certificate binds model weights,
processor/source revision, ordered media, prompt tokens, and adaptation state.
The donor ``cached_image_features`` seam performs the final merge, so the
source placeholder checks and feature order remain authoritative.

Smol captures the donor's post-connector features at its merge boundary; Qwen
computes the same vision-tower call the donor would make.  Neither copies the
source model tree. Bounded M3 candidate smokes do not establish route
qualification or performance.
"""

from __future__ import annotations

import hashlib
import json
import types
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from .multimodal import MediaFeatureCache


@dataclass(frozen=True, slots=True)
class VisionFeatureCertificate:
    family: str
    artifact_fingerprint: str
    source_revision: str
    processor_fingerprint: str
    adaptation_fingerprint: str
    media_fingerprint: str
    prompt_sha256: str
    scope: str = "prompt_v1"
    tower_sha256: str | None = None

    @classmethod
    def from_prepared(cls, *, family: str, artifact_fingerprint: str,
                      source_revision: str, processor_fingerprint: str,
                      adaptation_fingerprint: str, media_fingerprint: str,
                      prompt_tokens: list[int], scope: str = "prompt_v1",
                      tower_sha256: str | None = None):
        fields = (family, artifact_fingerprint, source_revision,
                  processor_fingerprint, adaptation_fingerprint,
                  media_fingerprint)
        if any(not isinstance(value, str) or not value for value in fields):
            raise ValueError("vision feature identity fields must be nonempty strings")
        if not isinstance(prompt_tokens, list) or not prompt_tokens or any(
            type(token) is not int or token < 0 for token in prompt_tokens
        ):
            raise ValueError("vision feature prompt tokens must be nonempty CPU integers")
        prompt_hash = hashlib.sha256(
            json.dumps(prompt_tokens, separators=(",", ":")).encode()
        ).hexdigest()
        if scope not in ("prompt_v1", "tower_inputs_v1"):
            raise ValueError("unsupported vision feature reuse scope")
        if scope == "tower_inputs_v1" and (not isinstance(tower_sha256, str)
                or len(tower_sha256) != 64
                or any(char not in "0123456789abcdef" for char in tower_sha256)):
            raise ValueError("tower input digest is required for tower-only reuse")
        return cls(*fields, prompt_hash, scope, tower_sha256)


def tower_input_digest(family: str, values: dict, *, max_bytes: int = 512 << 20) -> str | None:
    """Hash exact processor tensors consumed by the pinned vision tower.

    The pinned processors can return MLX arrays despite ignoring
    ``return_tensors``. Only this default-off path materializes them on host,
    after checking their byte size. Unsupported or oversized inputs decline
    reuse and continue through the donor tower.
    """
    if family == "smolvlm":
        names = ("pixel_values", "pixel_attention_mask")
        if values.get("pixel_values") is None:
            return None
    elif family == "qwen2_5_vl":
        image = values.get("pixel_values") is not None
        video = values.get("pixel_values_videos") is not None
        if image == video:
            return None
        names = (("pixel_values", "image_grid_thw") if image else
                 ("pixel_values_videos", "video_grid_thw"))
        if values.get(names[1]) is None:
            return None
    else:
        raise ValueError("unsupported vision feature-cache family")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("tower digest byte budget must be nonnegative")
    digest = hashlib.sha256(f"mlx2-tower-inputs-v1:{family}".encode())
    total = 0
    for name in names:
        value = values.get(name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
            continue
        if isinstance(value, np.ndarray):
            array = value
            source_dtype = value.dtype.str
        elif (type(value).__name__ == "array"
              and type(value).__module__.startswith("mlx.core")):
            shape = getattr(value, "shape", None)
            nbytes = getattr(value, "nbytes", None)
            dtype = getattr(value, "dtype", None)
            if (not isinstance(shape, tuple)
                    or any(type(axis) is not int or axis < 0 for axis in shape)
                    or type(nbytes) is not int or nbytes < 0
                    or dtype is None or nbytes > max_bytes - total):
                return None
            source_dtype = str(dtype)
            try:
                array = np.asarray(value)
            except (TypeError, ValueError, RuntimeError):
                return None
            if not isinstance(array, np.ndarray) or array.shape != shape:
                return None
        else:
            return None
        if array.dtype.hasobject:
            return None
        total += int(array.nbytes)
        if total > max_bytes:
            return None
        digest.update(json.dumps((source_dtype, array.dtype.str, array.shape)).encode())
        # Streaming contiguous rows avoids a second full-size byte string.
        array = array if array.flags.c_contiguous else np.ascontiguousarray(array)
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class VisionReuseCounters:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.refusals = 0


def certify_cold_prefill(request, remaining_tokens, prefill_input, *,
                         family: str, artifact_fingerprint: str,
                         source_revision: str, processor_fingerprint: str,
                         attest_prepared=None, scope: str = "prompt_v1"):
    """Issue a private certificate only at serving's complete CPU token handoff.

    Call from an adapter's ``validate_prefill_inputs``.  A partial APCv2 hit,
    media at the reserved final decode anchor, or a caller-supplied trust flag
    receives the ordinary source path.  LoRA identity is read here because
    serving attaches it after ``prepare_multimodal_request``.
    """
    result = dict(prefill_input)
    result.pop("_mlx2_vision_feature_certificate", None)
    result.pop("_mlx2_vision_feature_verified", None)
    # The public request namespace can contain forged `_mlx2_*` keys.  A
    # trusted adapter-owned attestation must cover the generated prompt and
    # media identity before either can authorize reuse.
    if not callable(attest_prepared) or attest_prepared(request) is not True:
        return result
    expected = request.get("_mlx2_prompt_tokens")
    media_end = request.get("_mlx2_media_token_end")
    media_identity = request.get("_mlx2_media_fingerprint")
    if (not isinstance(expected, list) or len(expected) < 2
            or not isinstance(remaining_tokens, (list, tuple))
            or list(remaining_tokens) != expected
            or type(media_end) is not int or not 0 < media_end < len(expected)
            or not isinstance(media_identity, str) or not media_identity):
        return result
    if family == "smolvlm":
        if result.get("pixel_values") is None:
            return result
    elif family == "qwen2_5_vl":
        if (result.get("pixel_values") is None
                and result.get("pixel_values_videos") is None):
            return result
    else:
        raise ValueError("unsupported vision feature-cache family")
    # A selected LoRA can alter the shared vision tower.  The current adapter
    # has no revision-bound snapshot of those weights, so keep reuse base-only.
    if request.get("_mlx2_lora_fingerprint") is not None:
        return result
    adaptation = "base"
    tower_sha256 = request.get("_mlx2_tower_inputs_digest")
    if scope == "tower_inputs_v1" and (not isinstance(tower_sha256, str)
            or len(tower_sha256) != 64):
        return result
    result["_mlx2_vision_feature_certificate"] = VisionFeatureCertificate.from_prepared(
        family=family, artifact_fingerprint=artifact_fingerprint,
        source_revision=source_revision, processor_fingerprint=processor_fingerprint,
        adaptation_fingerprint=adaptation, media_fingerprint=media_identity,
        prompt_tokens=expected, scope=scope, tower_sha256=tower_sha256,
    )
    result["_mlx2_vision_feature_verified"] = True
    return result


def _key(certificate, *, family, artifact_fingerprint, source_revision,
         processor_fingerprint, model_instance, verified):
    if not verified or not isinstance(certificate, VisionFeatureCertificate):
        return None
    if (certificate.family, certificate.artifact_fingerprint,
            certificate.source_revision, certificate.processor_fingerprint) != (
            family, artifact_fingerprint, source_revision, processor_fingerprint):
        return None
    if certificate.scope == "tower_inputs_v1":
        return ("vision-features-tower-v1", model_instance,
                certificate.family, certificate.artifact_fingerprint,
                certificate.source_revision, certificate.processor_fingerprint,
                certificate.adaptation_fingerprint, certificate.media_fingerprint,
                certificate.tower_sha256)
    if certificate.scope != "prompt_v1":
        return None
    return ("vision-features-v1", model_instance, certificate)


def _eval_feature(value):
    import mlx.core as mx

    mx.eval(value)


def install_qwen25(model, cache: MediaFeatureCache, *, artifact_fingerprint: str,
                   source_revision: str, processor_fingerprint: str,
                   evaluate=None, counters=None):
    """Install on one Qwen model; source still merges image/video features."""
    if getattr(model, "_mlx2_vision_reuse_family", None) is not None:
        raise ValueError("vision feature reuse is already installed on this model")
    original = model.get_input_embeddings
    counters = counters or VisionReuseCounters()
    evaluate = evaluate or _eval_feature
    model_instance = id(model)

    def cached(bound, input_ids=None, pixel_values=None, **kwargs):
        certificate = kwargs.pop("_mlx2_vision_feature_certificate", None)
        verified = kwargs.pop("_mlx2_vision_feature_verified", False) is True
        key = _key(
            certificate, family="qwen2_5_vl",
            artifact_fingerprint=artifact_fingerprint,
            source_revision=source_revision,
            processor_fingerprint=processor_fingerprint,
            model_instance=model_instance, verified=verified,
        )
        pixels = pixel_values if pixel_values is not None else kwargs.get(
            "pixel_values_videos"
        )
        if pixels is None:
            # A restored APCv2 prefix continues as text; there is no vision
            # feature reuse attempt to accept or refuse on this call.
            return original(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        if key is None or kwargs.get("cached_image_features") is not None:
            counters.refusals += 1
            return original(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        features = cache.get(key)
        if features is None:
            counters.misses += 1
            grid = kwargs.get("image_grid_thw")
            if grid is None:
                grid = kwargs.get("video_grid_thw")
            dtype = bound.vision_tower.patch_embed.proj.weight.dtype
            features = bound.vision_tower(
                pixels.astype(dtype), grid, output_hidden_states=False
            )
            # Complete the source merge before publishing the feature result.
            result = original(
                input_ids=input_ids, pixel_values=pixel_values,
                cached_image_features=features, **kwargs,
            )
            evaluate(features)
            if cache.put(key, features):
                counters.stores += 1
            return result
        counters.hits += 1
        return original(
            input_ids=input_ids, pixel_values=pixel_values,
            cached_image_features=features, **kwargs,
        )

    model.get_input_embeddings = types.MethodType(cached, model)
    model._mlx2_vision_reuse_family = "qwen2_5_vl"
    return counters


def install_smol(model, cache: MediaFeatureCache, *, artifact_fingerprint: str,
                 source_revision: str, processor_fingerprint: str,
                 evaluate=None, counters=None):
    """Capture exact source post-connector features without copying preprocessing.

    The private capture context is per Python execution context.  Concurrent
    calls on a shared model cannot cross-wire their feature arrays.  An
    indexed-scatter proxy that calls the donor class method directly bypasses
    this helper and continues to use the source vision path.
    """
    if getattr(model, "_mlx2_vision_reuse_family", None) is not None:
        raise ValueError("vision feature reuse is already installed on this model")
    original_embeddings = model.get_input_embeddings
    original_merge = model._prepare_inputs_for_multimodal
    counters = counters or VisionReuseCounters()
    evaluate = evaluate or _eval_feature
    model_instance = id(model)
    capture_key = ContextVar(f"mlx2_smol_vision_capture_{model_instance}", default=None)

    def merge(bound, features, inputs_embeds, input_ids):
        result = original_merge(features, inputs_embeds, input_ids)
        key = capture_key.get()
        if key is not None:
            evaluate(features)
            if cache.put(key, features):
                counters.stores += 1
        return result

    def cached(bound, input_ids=None, pixel_values=None, **kwargs):
        certificate = kwargs.pop("_mlx2_vision_feature_certificate", None)
        verified = kwargs.pop("_mlx2_vision_feature_verified", False) is True
        if pixel_values is None:
            # Warm APCv2 replay and batched text decode bypass the vision
            # tower. They are ordinary continuations, not reuse refusals.
            return original_embeddings(input_ids=input_ids, pixel_values=None, **kwargs)
        key = _key(
            certificate, family="smolvlm",
            artifact_fingerprint=artifact_fingerprint,
            source_revision=source_revision,
            processor_fingerprint=processor_fingerprint,
            model_instance=model_instance, verified=verified,
        )
        if key is None or kwargs.get("cached_image_features") is not None:
            counters.refusals += 1
            return original_embeddings(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        features = cache.get(key)
        if features is not None:
            counters.hits += 1
            return original_embeddings(
                input_ids=input_ids, pixel_values=pixel_values,
                cached_image_features=features, **kwargs,
            )
        counters.misses += 1
        token = capture_key.set(key)
        try:
            return original_embeddings(
                input_ids=input_ids, pixel_values=pixel_values, **kwargs
            )
        finally:
            capture_key.reset(token)

    model._prepare_inputs_for_multimodal = types.MethodType(merge, model)
    model.get_input_embeddings = types.MethodType(cached, model)
    model._mlx2_vision_reuse_family = "smolvlm"
    return counters
