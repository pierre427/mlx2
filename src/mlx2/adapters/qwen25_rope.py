"""Request-owned Qwen2.5-VL M-RoPE state for ordinary decode and APCv2.

The pinned mlx-vlm language model stores its RoPE delta on the shared model.
This plane instead travels with the KV cache through batch merge, extraction,
prefix copy, trimming, and disk serialization. Model forwards receive explicit
positions and never consult that shared field.
"""

from __future__ import annotations

import json
import threading

_BATCH_MASK_ATTENTION_TYPES = {}


class Qwen25RoPECache:
    def __init__(self, rows=None):
        self.rows = [dict(row) for row in rows] if rows is not None else [
            {"length": 0, "delta": 0, "media": False, "media_end": 0}
        ]
        self._right_padding = None

    @property
    def state(self):
        # Keep one addressable leaf for save_prompt_cache/tree_unflatten.
        # Its serializer records None as a placeholder beside meta_state.
        return (None,)

    @state.setter
    def state(self, value):
        if value != (None,) and value != [None]:
            raise ValueError("Qwen2.5 M-RoPE state must be in meta_state")

    @property
    def meta_state(self):
        return ("qwen25-mrope-v1", json.dumps(self.rows, separators=(",", ":")))

    @meta_state.setter
    def meta_state(self, value):
        if len(value) != 2 or value[0] != "qwen25-mrope-v1":
            raise ValueError("incompatible Qwen2.5 M-RoPE state")
        rows = json.loads(value[1])
        if not isinstance(rows, list) or not rows:
            raise ValueError("empty Qwen2.5 M-RoPE state")
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"length", "delta", "media", "media_end"}
                    or type(row["length"]) is not int or row["length"] < 0
                    or type(row["delta"]) is not int or type(row["media"]) is not bool
                    or type(row["media_end"]) is not int or row["media_end"] < 0
                    or row["media_end"] > row["length"]):
                raise ValueError("invalid Qwen2.5 M-RoPE row")
        self.rows = rows
        self._right_padding = None

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls()
        obj.state = state
        obj.meta_state = meta_state
        return obj

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise ValueError("cannot merge empty M-RoPE caches")
        return cls([row for cache in caches for row in cache.rows])

    def extract(self, index):
        return type(self)([self.rows[index]])

    def filter(self, indices):
        self.rows = [self.rows[index] for index in indices]

    def extend(self, other):
        self.rows.extend(dict(row) for row in other.rows)

    def prepare(self, *, right_padding=None, **_):
        if right_padding is None:
            self._right_padding = None
            return
        padding = list(right_padding)
        if (len(padding) != len(self.rows)
                or any(type(pad) is not int or pad < 0 for pad in padding)):
            raise ValueError("invalid Qwen2.5 M-RoPE right padding")
        self._right_padding = padding

    def finalize(self):
        if self._right_padding is not None:
            if len(self._right_padding) != len(self.rows):
                raise ValueError("M-RoPE padding width mismatch")
            if any(type(pad) is not int or pad < 0 or pad > row["length"]
                   for row, pad in zip(self.rows, self._right_padding)):
                raise ValueError("M-RoPE padding exceeds row length")
            if any(0 < row["length"] - pad < row["media_end"]
                   for row, pad in zip(self.rows, self._right_padding)):
                raise ValueError("M-RoPE padding enters media span")
            for row, pad in zip(self.rows, self._right_padding):
                row["length"] -= pad
                if not row["length"]:
                    row["delta"] = 0
                    row["media"] = False
                    row["media_end"] = 0
            self._right_padding = None

    def size(self):
        return max((row["length"] for row in self.rows), default=0)

    def empty(self):
        return not any(row["length"] for row in self.rows)

    @property
    def nbytes(self):
        return len(self.rows) * 24

    def is_trimmable(self):
        return True

    def supports_ragged_trim(self):
        return True

    def preflight_trim(self, count):
        """Check a uniform branch landing without mutating request-owned RoPE."""
        if type(count) is not int or count < 0:
            raise ValueError("invalid Qwen2.5 M-RoPE trim")
        return self.preflight_ragged_trim(
            [min(count, row["length"]) for row in self.rows]
        )

    def preflight_ragged_trim(self, counts, *, validate=True):
        if (len(counts) != len(self.rows) or any(type(n) is not int or n < 0
                or n > row["length"] for n, row in zip(counts, self.rows))):
            raise ValueError("invalid Qwen2.5 M-RoPE trim")
        if any(0 < row["length"] - n < row["media_end"]
               for row, n in zip(self.rows, counts)):
            raise ValueError("Qwen2.5 M-RoPE cannot trim inside media span")
        return list(counts)

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            row["length"] -= count
            if not row["length"]:
                row["delta"] = 0
                row["media"] = False
                row["media_end"] = 0
        return counts

    def trim(self, count):
        applied = self.preflight_trim(count)
        self.trim_ragged(applied)
        return min(applied, default=0)


class RequestPrivateQwen25Model:
    """Supply explicit positions to the pinned model on every forward."""

    def __init__(self, model):
        self._model = model
        self._forward_lock = threading.Lock()
        # Patch only this loaded model's attention instances. The pinned
        # implementation truncates a full KV mask to the new-key width before
        # appending the cache, which breaks mixed-length batch decode.
        language = model.language_model
        decoder = getattr(language, "model", None)
        for layer in getattr(decoder, "layers", ()):
            attention = layer.self_attn
            if not getattr(type(attention), "_mlx2_batch_mask", False):
                attention.__class__ = _batch_mask_attention(type(attention))

    def __getattr__(self, name):
        return getattr(self._model, name)

    def make_cache(self):
        inner = self._model.language_model
        if hasattr(inner, "make_cache"):
            cache = inner.make_cache()
        else:
            from ..runtime.models.cache import make_prompt_cache
            cache = make_prompt_cache(inner)
        return [*cache, Qwen25RoPECache()]

    def __call__(self, input_ids, *, cache=None, **kwargs):
        import mlx.core as mx

        media_end = kwargs.pop("_mlx2_rope_media_end", None)
        if "position_ids" in kwargs or "rope_deltas" in kwargs:
            raise ValueError("Qwen2.5-VL processor cannot override request-owned M-RoPE")
        if cache is None or not isinstance(cache[-1], Qwen25RoPECache):
            raise ValueError("Qwen2.5-VL requires a request-owned M-RoPE cache plane")
        rope = cache[-1]
        batch, width = input_ids.shape
        if len(rope.rows) != batch:
            raise ValueError("Qwen2.5-VL M-RoPE row count mismatch")
        if batch > 1 and (
            len({row["length"] for row in rope.rows}) != 1
            or (rope._right_padding is not None
                and len(set(rope._right_padding)) != 1)
        ):
            # The pinned decoder's B>1 projection/MLP arithmetic is not yet
            # numerically qualified against unequal-length B=1 decode. Both
            # prior history and current valid input width must be uniform.
            # Reject before any KV or RoPE state advances.
            raise ValueError("Qwen2.5-VL unequal-length batch is not qualified")
        media = kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None
        if media:
            if batch != 1 or rope.rows[0]["length"]:
                raise ValueError("Qwen2.5-VL media must prefill at a cold isolated boundary")
            if type(media_end) is not int or not 0 < media_end <= width:
                raise ValueError("Qwen2.5-VL media boundary is required")
            positions, delta = self._model.language_model.get_rope_index(
                input_ids, kwargs.get("image_grid_thw"), kwargs.get("video_grid_thw"),
                kwargs.get("mask"),
            )
            next_delta = int(delta.reshape(-1)[0].item())
        else:
            starts = mx.array([row["length"] + row["delta"] for row in rope.rows], dtype=mx.int32)
            positions = mx.arange(width, dtype=mx.int32)[None, :] + starts[:, None]
            positions = mx.broadcast_to(positions[None, :, :], (3, batch, width))
            next_delta = None
        if batch > 1 and kwargs.get("mask") is None:
            # The pinned Qwen language model passes the cache *list* to its
            # generic mask helper. For a one-token batch that helper returns
            # None, exposing left-padded KV slots in shorter rows. Supply the
            # first cache plane's per-row mask at the adapter boundary.
            make_mask = getattr(cache[0], "make_mask", None)
            prior_width = getattr(cache[0], "_idx", None)
            if not callable(make_mask) or type(prior_width) is not int:
                raise ValueError("Qwen2.5-VL batched decode requires a cache mask")
            kwargs["mask"] = make_mask(width, return_array=True)
        if batch > 1 and not media and kwargs.get("mask") is not None:
            from ..runtime.models.cache import BatchKVCache

            prior_width = getattr(cache[0], "_idx", None)
            mask = kwargs["mask"]
            if (any(type(plane) is not BatchKVCache
                    or plane._idx != prior_width
                    or plane.left_padding.shape[0] != batch
                    for plane in cache[:-1])
                    or type(prior_width) is not int or mask.ndim != 4
                    or mask.shape[0] != batch or mask.shape[1] != 1
                    or mask.shape[-2] != width
                    or mask.shape[-1] != prior_width + width):
                raise ValueError("Qwen2.5-VL batch mask/cache length mismatch")
        # Explicit positions bypass the pinned model's shared _rope_deltas and
        # _position_ids read path. The media call resets those fields itself.
        language = self._model.language_model
        with self._forward_lock:
            language._rope_deltas = None
            language._position_ids = None
            try:
                if batch > 1 and not media:
                    # Pinned LanguageModel.__call__ accepts ``mask`` but drops
                    # it when calling Qwen2Model. Invoke that decoder directly
                    # so merged rows cannot attend to left-padded KV slots.
                    hidden = language.model(input_ids, cache=cache[:-1],
                                            position_ids=positions, mask=kwargs["mask"])
                    output = (language.model.embed_tokens.as_linear(hidden)
                              if language.args.tie_word_embeddings
                              else language.lm_head(hidden))
                else:
                    output = self._model(input_ids, cache=cache[:-1], position_ids=positions, **kwargs)
            finally:
                language._rope_deltas = None
                language._position_ids = None
            for row in rope.rows:
                row["length"] += width
            if media:
                rope.rows[0]["delta"] = next_delta
                rope.rows[0]["media"] = True
                rope.rows[0]["media_end"] = media_end
        return getattr(output, "logits", output)


def _batch_mask_attention(base):
    if base in _BATCH_MASK_ATTENTION_TYPES:
        return _BATCH_MASK_ATTENTION_TYPES[base]

    class BatchMaskAttention(base):
        _mlx2_batch_mask = True

        def __call__(self, x, mask=None, cache=None, position_ids=None,
                     position_embeddings=None):
            if mask is None or x.shape[0] == 1:
                return super().__call__(x, mask, cache, position_ids,
                                        position_embeddings)
            from mlx_vlm.models.qwen2_5_vl.language import (
                apply_multimodal_rotary_pos_emb,
                scaled_dot_product_attention,
            )
            batch, width, _ = x.shape
            queries = self.q_proj(x).reshape(
                batch, width, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            keys = self.k_proj(x).reshape(
                batch, width, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            values = self.v_proj(x).reshape(
                batch, width, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            if position_embeddings is None:
                queries, keys = self.rotary_emb.apply_rotary(
                    queries, keys, position_ids, unsqueeze_dim=1)
            else:
                cos, sin = position_embeddings
                queries, keys = apply_multimodal_rotary_pos_emb(
                    queries, keys, cos, sin, unqueeze_dim=1)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            if mask.shape[-1] != keys.shape[-2]:
                raise ValueError("Qwen2.5-VL batch mask/cache length mismatch")
            output = scaled_dot_product_attention(
                queries, keys, values, cache, scale=self.scale, mask=mask)
            return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, width, -1))

    _BATCH_MASK_ATTENTION_TYPES[base] = BatchMaskAttention
    return BatchMaskAttention
