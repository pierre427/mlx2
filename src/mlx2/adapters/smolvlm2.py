"""SmolVLM2 image/video candidate with an isolated APCv2 KV lifecycle.

The pinned Idefics3 text forward asks ``create_attention_mask`` about a list
of caches, which discards the per-row mask after mlx2 merges caches for batched
decode.  Its outer forward also requires ``pixel_values`` on text-only decode.
This bridge keeps the source's embedding, tower, transformer, and head modules,
while making the cache and mask handoff explicit for each request.
"""

from dataclasses import replace

from ..contracts import Capability

from .pinned_vlm_candidate import (
    SOURCE_REVISION, PinnedVisionCandidateAdapter, _CandidateLogitsModel, descriptor_for,
    inspect_vision_artifact,
)
from .vision_feature_reuse import certify_cold_prefill, install_smol

_BASE_DESCRIPTOR = descriptor_for("smolvlm")
DESCRIPTOR = replace(
    _BASE_DESCRIPTOR,
    capabilities=_BASE_DESCRIPTOR.capabilities
    | frozenset({Capability.APC_V2, Capability.PREFIX_REUSE}),
    cache_layout="smolvlm2-image-video-apcv2-kv-v1",
    metadata={**_BASE_DESCRIPTOR.metadata,
              "required_qualification_checks": (
                  *_BASE_DESCRIPTOR.metadata["required_qualification_checks"],
                  "text_source_parity",
              ),
              "execution": "mlx2.adapters.smolvlm2.SmolVLM2CandidateAdapter"},
)


class _IndexedImageScatter:
    """Use processor-known image positions without a device-to-host index read.

    The pinned Idefics3 processor emits one row for an isolated media prefill.
    Its image features and image placeholders are both in row-major prompt
    order.  The source helper remains the fallback when that contract cannot
    be established from static shapes.
    """

    def __init__(self, model, positions, stats, scatter_backend=None):
        self.language_model = model.language_model
        self.vision_model = model.vision_model
        self.connector = model.connector
        self.config = model.config
        self._source_model = model
        self._positions = tuple(positions)
        self._stats = stats
        self._scatter_backend = scatter_backend

    def _prepare_inputs_for_multimodal(self, features, embeds, input_ids):
        positions = self._positions
        if (
            len(embeds.shape) != 3 or embeds.shape[0] != 1
            or len(features.shape) != 2
            or features.shape[0] != len(positions)
            or features.shape[1] != embeds.shape[2]
            or not positions
            or positions[0] < 0
            or positions[-1] >= embeds.shape[1]
            or any(a >= b for a, b in zip(positions, positions[1:]))
        ):
            self._stats["refusals"] += 1
            return self._source_model._prepare_inputs_for_multimodal(
                features, embeds, input_ids
            )

        mx = self._scatter_backend
        if mx is None:
            import mlx.core as mx

        indices = mx.broadcast_to(
            mx.array(positions, dtype=mx.int32)[None, :, None],
            (1, len(positions), embeds.shape[2]),
        )
        self._stats["engagements"] += 1
        return mx.put_along_axis(embeds, indices, features[None], axis=1)


def _forward_smol(model, input_ids, *, pixel_values=None, mask=None, cache=None,
                  attention_mask_factory=None, image_token_positions=None,
                  indexed_scatter=False, _mlx2_smol_positions_verified=False,
                  scatter_stats=None,
                  scatter_backend=None, **kwargs):
    """Run the pinned source layers against a request-owned KV cache.

    The mask factory receives one cache, which is the authoritative owner of
    the batch padding/offset state.  Vision features are computed only for the
    isolated cold prefill; a warm APCv2 hit resumes after every media token.
    """
    if attention_mask_factory is None:
        from mlx_vlm.models.base import create_attention_mask

        attention_mask_factory = create_attention_mask
    lm = model.language_model
    if cache is None:
        cache = [None] * len(lm.layers)
    if len(cache) != len(lm.layers):
        raise ValueError("SmolVLM2 text cache has the wrong layer count")
    if (indexed_scatter and pixel_values is not None
            and image_token_positions is not None
            and _mlx2_smol_positions_verified is True):
        proxy = _IndexedImageScatter(
            model, image_token_positions, scatter_stats,
            scatter_backend=scatter_backend,
        )
        embeds = type(model).get_input_embeddings(
            proxy, input_ids, pixel_values=pixel_values, **kwargs
        ).inputs_embeds
    else:
        if indexed_scatter and pixel_values is not None:
            scatter_stats["refusals"] += 1
        embeds = model.get_input_embeddings(
            input_ids, pixel_values=pixel_values, **kwargs
        ).inputs_embeds
    h = embeds.astype(lm.norm.weight.dtype)
    if mask is None:
        mask = attention_mask_factory(h, cache[0])
    for layer, state in zip(lm.layers, cache):
        h = layer(h, mask, state)
    return lm.lm_head(lm.norm(h))


class _SmolLogitsModel(_CandidateLogitsModel):
    def __init__(self, model, *, indexed_scatter=False):
        super().__init__(model)
        object.__setattr__(self, "indexed_scatter", indexed_scatter)
        object.__setattr__(self, "scatter_stats", {"engagements": 0, "refusals": 0})

    def make_cache(self):
        from ..runtime.models.cache import KVCache

        return [KVCache() for _ in self._model.language_model.layers]

    def __call__(self, input_ids, *, pixel_values=None, mask=None, cache=None,
                 **kwargs):
        return _forward_smol(
            self._model, input_ids, pixel_values=pixel_values, mask=mask,
            cache=cache, indexed_scatter=self.indexed_scatter,
            scatter_stats=self.scatter_stats, **kwargs
        )


def inspect_artifact(model_path):
    return inspect_vision_artifact(model_path, expected="smolvlm")


class SmolVLM2CandidateAdapter(PinnedVisionCandidateAdapter):
    model_type = "smolvlm"
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)

    def render_prompt(self, request):
        if "_mlx2_prompt_tokens" in request:
            raise ValueError("prepared media prompt has no plain text rendering")
        if "messages" in request:
            messages = self._template_messages(request["messages"], ())
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return request["prompt"]

    def _template_messages(self, messages, replacements):
        """Keep typed media parts for Smol's list-only chat template."""
        marker = self.processor.image_token
        cursor = iter(replacements)
        missing = object()
        converted = []
        for message in messages:
            original = message.get("content")
            if not isinstance(original, list):
                content = [{"type": "text", "text": str(original or "")}]
            else:
                content = []
                for part in original:
                    if part.get("type") == "text":
                        content.append({"type": "text", "text": part.get("text", "")})
                        continue
                    replacement = next(cursor, missing)
                    if replacement is missing:
                        raise RuntimeError("multimodal replacements are fewer than media parts")
                    chunks = replacement.split(marker)
                    if len(chunks) < 2:
                        raise ValueError("Smol media replacement lacks an image marker")
                    for index, chunk in enumerate(chunks):
                        if chunk:
                            content.append({"type": "text", "text": chunk})
                        if index < len(chunks) - 1:
                            content.append({"type": "image"})
            converted.append({**message, "content": content or [{"type": "text", "text": ""}]})
        if next(cursor, missing) is not missing:
            raise RuntimeError("multimodal replacements outnumber media parts")
        return converted

    def __init__(self, model_path: str, *, execution_policy=None):
        indexed_scatter = execution_policy == {"smol_image_scatter": "indexed_v1"}
        vision_reuse = execution_policy == {"vision_feature_reuse": "candidate_v1"}
        tower_reuse = execution_policy == {"vision_feature_reuse": "tower_only_candidate_v1"}
        super().__init__(
            model_path,
            execution_policy=None if indexed_scatter or vision_reuse or tower_reuse else execution_policy,
        )
        self._vision_feature_reuse_enabled = vision_reuse or tower_reuse
        self._vision_feature_reuse_scope = "tower_inputs_v1" if tower_reuse else "prompt_v1"
        if tower_reuse:
            from .multimodal import MediaFeatureCache

            self.media_feature_cache = MediaFeatureCache(max_entries=16, max_bytes=256 << 20)
        self._vision_feature_reuse_counters = None
        if self._vision_feature_reuse_enabled:
            self._vision_feature_reuse_counters = install_smol(
                self.model._model, self.media_feature_cache,
                artifact_fingerprint=self.identity["fingerprint"],
                source_revision=SOURCE_REVISION,
                processor_fingerprint=self.identity["fingerprint"],
            )
        self.model = _SmolLogitsModel(
            self.model._model, indexed_scatter=indexed_scatter
        )

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            **super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step),
            "smol_image_scatter": (
                "indexed_v1" if self.model.indexed_scatter else "source"
            ),
            "vision_feature_reuse": (
                "tower_only_candidate_v1" if self._vision_feature_reuse_enabled
                and getattr(self, "_vision_feature_reuse_scope", "prompt_v1") == "tower_inputs_v1" else
                "candidate_v1" if self._vision_feature_reuse_enabled else "source"
            ),
            **({"vision_feature_cache_max_bytes": self.media_feature_cache.max_bytes}
               if getattr(self, "_vision_feature_reuse_scope", "prompt_v1") == "tower_inputs_v1" else {}),
        }

    def prepare_multimodal_request(self, request, *, file_loader=None):
        prepared = super().prepare_multimodal_request(
            request, file_loader=file_loader
        )
        if "_mlx2_prefill_inputs" in prepared and (
            prepared["_mlx2_media_token_end"] >= len(prepared["_mlx2_prompt_tokens"])
        ):
            # BatchGenerator reserves the final prompt token for its first
            # generation forward.  Media must be consumed in isolated prefill.
            raise ValueError("SmolVLM2 media placeholder cannot be the final prompt token")
        if "_mlx2_prefill_inputs" in prepared and getattr(
            getattr(self, "model", None), "indexed_scatter", False
        ):
            token_id = int(self.identity["config"]["image_token_id"])
            prepared["_mlx2_prefill_inputs"]["image_token_positions"] = tuple(
                i for i, token in enumerate(prepared["_mlx2_prompt_tokens"])
                if token == token_id
            )
        return prepared

    def validate_prefill_inputs(self, request, remaining_tokens, prefill_input):
        """Certify the cold B=1 token span before the indexed scatter runs.

        Serving calls this at the handoff where its CPU token list is still
        available.  The private flag is never accepted from the processor or
        client.  A partial APCv2 prefix hit cannot satisfy this equality.
        """
        result = dict(prefill_input)
        result.pop("_mlx2_smol_positions_verified", None)
        if self._vision_feature_reuse_enabled:
            return certify_cold_prefill(
                request, remaining_tokens, result, family=self.model_type,
                artifact_fingerprint=self.identity["fingerprint"],
                source_revision=SOURCE_REVISION,
                processor_fingerprint=self.identity["fingerprint"],
                attest_prepared=self.has_trusted_media_preparation,
                scope=getattr(self, "_vision_feature_reuse_scope", "prompt_v1"),
            )
        if not self.model.indexed_scatter:
            return result
        if not self.has_trusted_media_preparation(request):
            return result
        expected = request.get("_mlx2_prompt_tokens")
        positions = result.get("image_token_positions")
        if not isinstance(expected, list) or not isinstance(positions, tuple):
            return result
        if list(remaining_tokens) != expected or len(expected) < 2:
            return result
        token_id = int(self.identity["config"]["image_token_id"])
        actual = tuple(
            i for i, token in enumerate(expected[:-1]) if token == token_id
        )
        if actual and positions == actual:
            result["_mlx2_smol_positions_verified"] = True
        return result

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "smol_image_scatter": (
                "indexed_v1" if self.model.indexed_scatter else "source"
            ),
            "smol_image_scatter_counts": dict(self.model.scatter_stats),
            "vision_feature_reuse": (
                vars(self._vision_feature_reuse_counters).copy()
                if self._vision_feature_reuse_counters is not None else None
            ),
        }
