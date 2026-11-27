# SPDX-License-Identifier: MIT
# Mined closure; provenance/muse-dflash2.json.
from typing import List

import mlx.core as mx
import mlx.nn as nn

def swiglu(gate, value):
    return nn.silu(gate) * value


def _fold_rms_norm(norm, value):
    """RMSNorm without MLX 0.32.4's crashing fused Metal primitive."""
    dtype = value.dtype
    value = value.astype(mx.float32)
    value = value * mx.rsqrt(
        mx.mean(value * value, axis=-1, keepdims=True) + float(norm.eps)
    )
    return (value * norm.weight.astype(mx.float32)).astype(dtype)


from ..models.cache import KVCache, RotatingKVCache
def initialize_rope(*, dims, base, traditional, scaling_config):
    # Default RoPE has no position bound to configure, so the checkpoint's
    # max_position_embeddings is not taken: accepting it implied otherwise.
    if scaling_config and scaling_config.get("rope_type", "default") != "default":
        raise ValueError("DFlash2 supports default RoPE only")
    return nn.RoPE(dims, traditional=traditional, base=base)
from .base_config import DFlashConfig
from ..technologies.drafter_blocks import GatedMLP, GatedMLPSpec


def _build_rope(config: DFlashConfig):
    # Qwen-family checkpoints normally use split-half (NeoX) pairing. DSpark
    # checkpoints expose rope_is_neox_style explicitly; MLX calls the
    # interleaved GPT-J pairing "traditional".
    traditional = not bool(getattr(config, "rope_is_neox_style", True))
    return initialize_rope(
        dims=config.head_dim,
        base=config.rope_theta,
        traditional=traditional,
        scaling_config=config.rope_scaling,
    )


def append_context_kv(cache, keys, values):
    """Empty checkpoint storage must not dictate the first real KV dtype."""
    if keys.shape[2] == 0:
        if cache.offset == 0:
            return keys, values
        return cache.keys_and_values() if hasattr(cache, "keys_and_values") else cache.state
    if cache.offset == 0 and cache.keys is not None and cache.keys.shape[2] == 0:
        cache.keys = cache.values = None
    return cache.update_and_fetch(keys, values)


class _NoRope:
    """Identity rotation for draft layers declared in ``no_rope_layers``."""

    def __call__(self, x, offset=0):
        return x


NO_ROPE = _NoRope()


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.is_causal = bool(getattr(config, "is_causal", False))
        layer_types = (
            config.layer_types or ["full_attention"] * config.num_hidden_layers
        )
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        self.use_rope = layer_idx not in set(getattr(config, "no_rope_layers", None) or ())
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if getattr(config, "attention_sink_bias", False):
            # One learned logit per query head joins each softmax denominator
            # (the gpt-oss sink law MLX's fused attention implements).
            self.attention_sink_bias = mx.zeros((self.n_heads,))

    def _proposal_mask(self, prefix, rows):
        if not self.is_causal:
            return None
        query = prefix + mx.arange(rows)
        key = mx.arange(prefix + rows)
        mask = key[None, :] <= query[:, None]
        if self.is_sliding:
            mask = mask & (key[None, :] > query[:, None] - self.sliding_window)
        return mask[None, None]

    def _device_ring_attend(self, rope, entry, queries, ctx_keys, ctx_values, prop_keys,
                            prop_values):
        """One row against a ``DeviceRingKVCache`` (spec-round DC-3, default off).

        The host path's law with device offsets: context keys rotate at the
        device fill, proposals and queries at the fill after this append, and
        attention sees the ring's last ``window`` committed rows plus the
        whole (bidirectional) proposal block."""
        if self.is_causal or getattr(self, "fold_policy", None) is not None:
            raise ValueError("device ring drafter caches serve the bidirectional non-FOLD law only")
        ck = rope(ctx_keys, offset=entry.rope_offset())
        keys, values, visible = entry.append(ck, ctx_values)
        end = entry.rope_offset()
        q = rope(queries, offset=end)
        pk = rope(prop_keys, offset=end)
        if keys is None:
            # A one-token prompt has no committed features on its first
            # round. Attend the bidirectional block without allocating a
            # context ring until actual feature rows arrive.
            return mx.fast.scaled_dot_product_attention(
                q, pk, prop_values, scale=self.scale, sinks=self._sinks(q),
            )
        rows = int(queries.shape[2])
        mask = mx.concatenate([
            mx.broadcast_to(visible[None, :], (rows, int(visible.shape[0]))),
            mx.ones((rows, rows), dtype=mx.bool_),
        ], axis=1)[None, None]
        return mx.fast.scaled_dot_product_attention(
            q, mx.concatenate([keys, pk], axis=2), mx.concatenate([values, prop_values], axis=2),
            scale=self.scale, mask=mask, sinks=self._sinks(q),
        )

    def _sinks(self, queries):
        sinks = getattr(self, "attention_sink_bias", None)
        return None if sinks is None else sinks.astype(queries.dtype)

    def _projections(self, x: mx.array, x_ctx: mx.array):
        """``(queries, ctx_keys, ctx_values, prop_keys, prop_values)``.

        Context and proposal are projected separately so only context KV
        enters the cache.  A drafter may replace this per instance (DFlash2's
        one-launch path, ``dflash2.fuse_context_projections``).
        """
        fused = self.__dict__.get("_fused_projections")
        if fused is not None:
            return fused(self, x, x_ctx)
        return (
            self.q_proj(x),
            self.k_proj(x_ctx),
            self.v_proj(x_ctx),
            self.k_proj(x),
            self.v_proj(x),
        )

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache: KVCache):
        if not self.use_rope:
            # Fused installers skip NoPE layers (their kernels bake RoPE in).
            if self.__dict__.get("_fused_forward") is not None:
                raise ValueError("a fused draft attention kernel cannot serve a NoPE layer")
            rope = NO_ROPE
        fused = self.__dict__.get("_fused_forward")
        if fused is not None:
            if self.is_causal:
                raise ValueError("causal draft blocks require the mask-aware stock kernel")
            return fused(self, x, x_ctx, rope, cache)
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        queries, ctx_keys, ctx_values, prop_keys, prop_values = self._projections(
            x, x_ctx
        )
        fold = getattr(self, "fold_policy", None) is not None
        if fold and self.is_causal:
            raise ValueError("FOLD attention does not implement causal draft blocks")
        if fold and getattr(self, "attention_sink_bias", None) is not None:
            raise ValueError("FOLD attention does not implement attention sinks")
        query_rows = queries.reshape(B, L, self.n_heads, -1)
        context_key_rows = ctx_keys.reshape(B, S, self.n_kv_heads, self.head_dim)
        proposal_key_rows = prop_keys.reshape(B, L, self.n_kv_heads, -1)
        queries = (
            _fold_rms_norm(self.q_norm, query_rows)
            if fold
            else self.q_norm(query_rows)
        ).transpose(0, 2, 1, 3)
        ctx_keys = (
            _fold_rms_norm(self.k_norm, context_key_rows)
            if fold
            else self.k_norm(context_key_rows)
        ).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        prop_keys = (
            _fold_rms_norm(self.k_norm, proposal_key_rows)
            if fold
            else self.k_norm(proposal_key_rows)
        ).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        if hasattr(cache, "rows"):
            from .device_ring_kv import DeviceRingKVCache

            outputs = []
            for row, entry in enumerate(cache.rows):
                if isinstance(entry, DeviceRingKVCache):
                    outputs.append(self._device_ring_attend(
                        rope, entry, queries[row:row+1], ctx_keys[row:row+1],
                        ctx_values[row:row+1], prop_keys[row:row+1], prop_values[row:row+1]))
                    continue
                if getattr(self, "fold_policy", None) is not None:
                    from .fold import FOLDKVCache, fold_attention

                    if not isinstance(entry, FOLDKVCache):
                        raise TypeError("FOLD attention requires FOLDKVCache")
                    entry.append_raw_keys(ctx_keys[row:row+1])
                    ck = rope(ctx_keys[row:row+1], offset=entry.offset)
                    append_context_kv(entry, ck, ctx_values[row:row+1])
                    folded, used_far = fold_attention(
                        queries[row:row+1], prop_keys[row:row+1],
                        prop_values[row:row+1], rope=rope, cache=entry,
                        policy=self.fold_policy, scale=self.scale,
                        local_context=(self.sliding_window - 1 if self.is_sliding else None),
                    )
                    self.fold_calls += 1
                    self.fold_far_calls += int(used_far)
                    self.fold_max_context = max(self.fold_max_context, entry.offset)
                    outputs.append(folded)
                    continue
                q = rope(queries[row:row+1], offset=entry.offset + S)
                ck = rope(ctx_keys[row:row+1], offset=entry.offset)
                pk = rope(prop_keys[row:row+1], offset=entry.offset + S)
                keys, values = append_context_kv(entry, ck, ctx_values[row:row+1])
                if self.is_sliding and keys.shape[2] > self.sliding_window-1:
                    keys, values = keys[:, :, -(self.sliding_window-1):], values[:, :, -(self.sliding_window-1):]
                    entry.keys, entry.values, entry._idx = keys, values, keys.shape[2]
                outputs.append(mx.fast.scaled_dot_product_attention(q, mx.concatenate([keys,pk],axis=2), mx.concatenate([values,prop_values[row:row+1]],axis=2), scale=self.scale,mask=self._proposal_mask(keys.shape[2], L), sinks=self._sinks(q)))
            o = mx.concatenate(outputs,axis=0)
            return self.o_proj(o.transpose(0,2,1,3).reshape(B,L,-1))
        from .device_ring_kv import DeviceRingKVCache

        if isinstance(cache, DeviceRingKVCache):
            o = self._device_ring_attend(rope, cache, queries, ctx_keys, ctx_values,
                                         prop_keys, prop_values)
            return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
        if getattr(self, "fold_policy", None) is not None:
            from .fold import FOLDKVCache, fold_attention

            if not isinstance(cache, FOLDKVCache):
                raise TypeError("FOLD attention requires FOLDKVCache")
            cache.append_raw_keys(ctx_keys)
            ctx_keys = rope(ctx_keys, offset=cache.offset)
            append_context_kv(cache, ctx_keys, ctx_values)
            o, used_far = fold_attention(
                queries, prop_keys, prop_values, rope=rope, cache=cache,
                policy=self.fold_policy, scale=self.scale,
                local_context=(self.sliding_window - 1 if self.is_sliding else None),
            )
            self.fold_calls += 1
            self.fold_far_calls += int(used_far)
            self.fold_max_context = max(self.fold_max_context, cache.offset)
            return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = append_context_kv(cache, ctx_keys, ctx_values)
        if self.is_sliding and keys.shape[2] > self.sliding_window - 1:
            keys, values = keys[:, :, -(self.sliding_window-1):], values[:, :, -(self.sliding_window-1):]
            cache.keys, cache.values, cache._idx = keys, values, keys.shape[2]

        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        # Preserve the trained draft-block law. The default denoising block
        # is bidirectional; an explicitly admitted causal variant also masks
        # future proposals and slides visibility per query position.
        mask = self._proposal_mask(keys.shape[2] - L, L)
        o = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask,
            sinks=self._sinks(queries),
        )
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Qwen3MLP(GatedMLP):
    """Qwen3-style gated MLP (matches mlx_lm.models.qwen3.MLP weights); the
    plain ``swiglu`` law, math in technologies.drafter_blocks."""

    def __init__(self, dim, hidden_dim):
        super().__init__(dim, hidden_dim, GatedMLPSpec())


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    layer_class = DFlashDecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ["full_attention"] * self.config.num_hidden_layers
        # One row is every tap concatenated (n-stream taps for an mHC target).
        concat_dim = getattr(config, "feature_width", len(config.target_layer_ids) * config.hidden_size)
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            self.layer_class(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []

    def target_activation(self, value):
        """Optional adapter-owned draft input dtype; target arrays stay exact."""
        dtype = getattr(self, "target_activation_dtype", None)
        return value if dtype is None else value.astype(dtype)

    def bind(self, target_model) -> "DFlashDraftModel":
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(
            self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self) -> List[KVCache]:
        window = getattr(self.config, "draft_window_size", None)
        if window is not None and int(window) > 0:
            return [
                RotatingKVCache(max_size=int(window), keep=0)
                for _ in self.layers
            ]
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "DFlash draft config must define sliding_window for sliding layers."
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model) -> List[KVCache]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        return self.make_cache()

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            B = last_bonus.shape[0]
            masks = mx.full((B, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        draft_hidden = self._hidden(block, hidden, cache)
        draft_logits = self._logits(draft_hidden[:, 1:])
        return sampler(draft_logits)

    def _hidden(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
        *,
        submit: bool = False,
    ) -> mx.array:
        """Draft block hidden states.

        ``submit`` queues the projected context and each layer's output for
        evaluation as soon as its graph is built, so the device starts on
        the first layer while the host still builds the rest.  It changes
        when work is submitted, never what is computed.
        """
        h = self.target_activation(self._embed_input_tokens(inputs))
        fold = getattr(self, "fold_policy", None) is not None
        projected_context = self.fc(self.target_activation(target_hidden))
        h_ctx = (
            _fold_rms_norm(self.hidden_norm, projected_context)
            if fold
            else self.hidden_norm(projected_context)
        )
        if submit:
            mx.async_eval(h_ctx)
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
            if submit:
                mx.async_eval(h)
        return _fold_rms_norm(self.norm, h) if fold else self.norm(h)

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return self.embed_tokens(inputs) * self.embed_scale

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[KVCache],
    ) -> mx.array:
        return self._logits(self._hidden(inputs, target_hidden, cache))

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            out[k] = v
        return out


DFlashKVCache = KVCache


# -- lazy drafts (``speculative_device_round`` ``lazy_draft``; default off) -----
#
# ``DFlashDraftModel`` itself declares no ``draft_lazy``: each family's lazy
# draft mirrors one host ``draft_distributions`` (plain_dflash, xpress, dspark,
# lilicorr, dspine) and is declared on that class, so a subclass that changes
# the host draft declines (``lazy_mirrors``) instead of inheriting the wrong ops.


def lazy_anchors(anchor):
    """The one-lane anchor as an int32 ``[1]`` array (a device anchor as is)."""
    if isinstance(anchor, mx.array):
        return anchor.reshape(1).astype(mx.int32)
    return mx.array([int(anchor)], dtype=mx.int32)


def lazy_mirrors(model, owner, names=("draft_distributions",)):
    """Decline (counted ``lazy_draft_host_override``) when ``model`` no longer
    runs ``owner``'s host draft: an override of any of ``names`` on a subclass
    or on the instance means ``owner.draft_lazy`` would not mirror it."""
    from ..proposal_composition import LazyDecline

    for name in names:
        if name in model.__dict__ or getattr(type(model), name) is not getattr(owner, name):
            raise LazyDecline("lazy_draft_host_override")


def lazy_one_lane(model, hidden, cache, label):
    if hidden.shape[0] != 1:
        raise ValueError(f"{label} lazy drafts run one lane at a time")
    return model.batch_caches([cache])


def lazy_host_history(processors, history, label):
    if processors and callable(getattr(history, "apply", None)):
        raise ValueError(f"{label} lazy drafts take a host processor history")
    return list(history or ())


def lazy_prefix(history, anchors):
    """``processor_history + [anchor]`` as the host prefix (int32)."""
    return mx.concatenate([mx.array(list(history), dtype=mx.int32), anchors])


def lazy_point_mass_record(sampler, tokens, vocab):
    """Record a deterministic drafter's point-mass law through ``sampler``.

    ``device_round.sampled_drafts`` verifies a sampled lane's drafts by the
    ratio rule against the law the sampler recorded.  A point-mass drafter
    hands it logits that are 0 at its token and -inf elsewhere (scaling by
    1/T keeps that), so q is the exact point mass: the ratio rule accepts
    with probability p(token) and resamples from normalized(p - q)+, which is
    the point-mass verification law.  The sampler's own draw is the token
    with certainty; the drafted ``tokens`` are returned unchanged."""
    ids = mx.arange(int(vocab), dtype=mx.int32)
    for position in range(int(tokens.shape[0])):
        sampler(mx.where(ids == tokens[position], 0.0, -float("inf"))[None].astype(mx.float32))
    return tokens


def lazy_redo(invalid, tokens):
    """``tokens``, or ``DRAFT_LAZY_HOST_REDO`` everywhere where ``invalid``
    (a lane the host path ends early or refuses): the round is restored and
    redone on the host path, which then drafts or raises exactly as before."""
    from .dflash2 import DRAFT_LAZY_HOST_REDO

    count = int(tokens.shape[0])
    return mx.where(invalid, mx.full((count,), DRAFT_LAZY_HOST_REDO, dtype=mx.int32),
                    tokens.astype(mx.int32))
