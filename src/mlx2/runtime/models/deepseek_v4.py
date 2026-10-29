# SPDX-License-Identifier: MIT
"""Adapter-owned DeepSeek V4 ordinary text math and direct GGUF bindings.

Source equations and notice: provenance/deepseek-v4-forward.json.  All storage
is supplied by a weight provider; constructing this model never allocates a
284B-parameter floating placeholder.  Packed GGML is not MLX affine quantization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import mlx.core as mx
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from .deepseek_v4_cache import DeepSeekV4LayerCache


@dataclass
class DeepSeekV4Config:
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    scoring_func: str = "sqrtsoftplus"
    num_hash_layers: int = 3
    rms_norm_eps: float = 1e-20
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    sliding_window: int = 128
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 1048576
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    swiglu_limit: float = 10.0
    compress_ratios: tuple[int, ...] = ()
    simulate_quantization: bool = True

    @property
    def intermediate_size(self):
        """Actual expert width exposed to generic external admission geometry."""
        return self.moe_intermediate_size

    def __post_init__(self):
        if not self.compress_ratios:
            raise ValueError("DeepSeek V4 requires checkpoint compression geometry")
        self.compress_ratios = tuple(self.compress_ratios)
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError("compression geometry must match transformer layer count")
        if any(r not in (0, 4, 128) for r in self.compress_ratios):
            raise ValueError("unsupported DeepSeek V4 compression ratio")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek V4 requires one shared KV head")
        if self.num_attention_heads % self.o_groups or self.head_dim % 2:
            raise ValueError("invalid grouped-output attention dimensions")
        if self.qk_rope_head_dim > self.head_dim or self.qk_rope_head_dim % 2:
            raise ValueError("invalid rotary dimensions")
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("invalid expert count")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("unimplemented checkpoint expert scoring function")

    @classmethod
    def from_dict(cls, value):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in names})

    @classmethod
    def from_gguf(cls, metadata):
        p = "deepseek4."
        mapping = {
            "vocab_size": "vocab_size",
            "hidden_size": "embedding_length",
            "num_hidden_layers": "block_count",
            "num_attention_heads": "attention.head_count",
            "num_key_value_heads": "attention.head_count_kv",
            "head_dim": "attention.key_length",
            "q_lora_rank": "attention.q_lora_rank",
            "qk_rope_head_dim": "rope.dimension_count",
            "o_groups": "attention.output_group_count",
            "o_lora_rank": "attention.output_lora_rank",
            "n_routed_experts": "expert_count",
            "num_experts_per_tok": "expert_used_count",
            "n_shared_experts": "expert_shared_count",
            "moe_intermediate_size": "expert_feed_forward_length",
            "routed_scaling_factor": "expert_weights_scale",
            "norm_topk_prob": "expert_weights_norm",
            "num_hash_layers": "hash_layer_count",
            "rms_norm_eps": "attention.layer_norm_rms_epsilon",
            "hc_mult": "hyper_connection.count",
            "hc_sinkhorn_iters": "hyper_connection.sinkhorn_iterations",
            "hc_eps": "hyper_connection.epsilon",
            "sliding_window": "attention.sliding_window",
            "rope_theta": "rope.freq_base",
            "compress_rope_theta": "attention.compress_rope_freq_base",
            "max_position_embeddings": "context_length",
            "index_n_heads": "attention.indexer.head_count",
            "index_head_dim": "attention.indexer.key_length",
            "index_topk": "attention.indexer.top_k",
        }
        if metadata.get("general.architecture") != "deepseek4":
            raise ValueError("GGUF architecture is not deepseek4")
        values = {key: metadata[p + source] for key, source in mapping.items()}
        if metadata.get(p + "expert_gating_func") != 4:
            raise ValueError("GGUF does not declare sqrtsoftplus gating")
        # GGUF can append auxiliary next-token layer geometry after the main body.
        ratios = metadata[p + "attention.compress_ratios"]
        if len(ratios) < values["num_hidden_layers"]:
            raise ValueError("missing transformer compression geometry")
        values["compress_ratios"] = tuple(ratios[: values["num_hidden_layers"]])
        clamps = metadata[p + "swiglu_clamp_exp"][: values["num_hidden_layers"]]
        if not clamps or any(x != clamps[0] for x in clamps):
            raise ValueError(
                "per-layer SwiGLU limits require an explicit implementation"
            )
        values["swiglu_limit"] = clamps[0]
        values["rope_scaling"] = {
            "type": metadata[p + "rope.scaling.type"],
            "factor": metadata[p + "rope.scaling.factor"],
            "original_max_position_embeddings": metadata[
                p + "rope.scaling.original_context_length"
            ],
            "beta_fast": metadata[p + "rope.scaling.yarn_beta_fast"],
            "beta_slow": metadata[p + "rope.scaling.yarn_beta_slow"],
        }
        return cls(**values)


def limited_swiglu(gate, up, limit=10.0):
    """The gate clips above only; clipping its negative tail changes the model."""
    return nn.silu(mx.minimum(gate, limit)) * mx.clip(up, -limit, limit)


def route_experts(
    logits, *, top_k, scale, correction=None, token_ids=None, table=None, normalize=True
):
    scores = mx.sqrt(nn.softplus(logits.astype(mx.float32)))
    if table is not None:
        if token_ids is None:
            raise ValueError("hash routing requires original token IDs")
        indices = table[token_ids].astype(mx.int32)
    else:
        if correction is None:
            raise ValueError("learned routing requires correction bias")
        indices = mx.argpartition(-(scores + correction), kth=top_k - 1, axis=-1)[
            ..., :top_k
        ]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return indices.astype(mx.uint32), weights * scale


def sinkhorn_mix(mixes, scale, base, count, iterations, eps):
    """Always compute hyper-connection coefficients in FP32, including loaded base."""
    return _sinkhorn_mix_fp32(
        mixes.astype(mx.float32),
        scale.astype(mx.float32),
        base.astype(mx.float32),
        count,
        iterations,
        eps,
    )


def _sinkhorn_mix_fp32(z, scale, base, count, iterations, eps):
    """Same equation order with already resident FP32 static HC parameters."""
    pre = mx.sigmoid(z[..., :count] * scale[0] + base[:count]) + eps
    post = 2 * mx.sigmoid(
        z[..., count : 2 * count] * scale[1] + base[count : 2 * count]
    )
    comb = (z[..., 2 * count :] * scale[2] + base[2 * count :]).reshape(
        *z.shape[:-1], count, count
    )
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb /= mx.sum(comb, axis=-2, keepdims=True) + eps
    for _ in range(iterations - 1):
        comb /= mx.sum(comb, axis=-1, keepdims=True) + eps
        comb /= mx.sum(comb, axis=-2, keepdims=True) + eps
    return pre, post, comb


def hc_expand(branch, residual, post, comb, *, policy="reference"):
    if policy == "fused_fp32":
        from .deepseek_v4_hc import expand

        return expand(branch, residual, post, comb).astype(branch.dtype)
    value = post[..., None] * branch[..., None, :].astype(mx.float32)
    value += mx.matmul(mx.swapaxes(comb, -1, -2), residual.astype(mx.float32))
    return value.astype(branch.dtype)


def simulate_fp8_kv(value, rope_dims=64, group_size=64):
    """E4M3FN QAT roundtrip; rotary tail remains untouched (DS4 exact law)."""
    dim = value.shape[-1] - rope_dims
    if dim < 0 or dim % group_size:
        raise ValueError("FP8 KV nonrotary dimensions must align to groups64")
    if not dim:
        return value
    groups = (
        value[..., :dim]
        .astype(mx.float32)
        .reshape(*value.shape[:-1], dim // group_size, group_size)
    )
    amax = mx.maximum(mx.max(mx.abs(groups), axis=-1, keepdims=True), 1e-4)
    scale = mx.power(2.0, mx.ceil(mx.log2(amax / 448.0)))
    scaled = mx.clip(groups / scale, -448.0, 448.0)
    # E4M3FN subnormals use a fixed 2^-9 step. Normal steps have 3 fraction bits.
    exponent = mx.floor(mx.log2(mx.maximum(mx.abs(scaled), 2.0**-6)))
    step = mx.power(2.0, mx.maximum(exponent - 3, -9))
    rounded = mx.round(scaled / step) * step
    result = (rounded * scale).reshape(*value.shape[:-1], dim).astype(value.dtype)
    return mx.concatenate((result, value[..., dim:]), axis=-1)


def simulate_index_fp4(value, group_size=32):
    """Normalized Hadamard then E2M1 QAT, power-of-two scaling and even ties."""
    dim = value.shape[-1]
    if dim != 128 or dim % group_size:
        raise ValueError("DeepSeek V4 indexer QAT requires 128 coordinates")
    rotated = mx.hadamard_transform(value.astype(mx.float32), scale=dim**-0.5)
    groups = rotated.reshape(*value.shape[:-1], dim // group_size, group_size)
    amax = mx.maximum(
        mx.max(mx.abs(groups), axis=-1, keepdims=True), 7.052966104933725e-38
    )
    scale = mx.power(2.0, mx.ceil(mx.log2(amax / 6.0)))
    scaled = groups / scale
    levels = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)
    distance = mx.abs(mx.abs(scaled)[..., None] - levels)
    minimum = mx.min(distance, axis=-1, keepdims=True)
    # At exact midpoints choose the even code, rather than always the lower value.
    codes = mx.arange(8)
    priority = mx.where(codes % 2 == 0, codes, codes + 8)
    selected = mx.argmin(mx.where(distance == minimum, priority, 100), axis=-1)
    rounded = levels[selected] * mx.sign(scaled)
    return (rounded * scale).reshape(value.shape).astype(value.dtype)


class HyperConnection:
    def __init__(
        self,
        config,
        weights,
        prefix,
        *,
        head=False,
        policy="reference",
        projection_policy="reference",
    ):
        if projection_policy not in ("reference", "fixed_fp32"):
            raise ValueError("unknown DeepSeek V4 HC projection arithmetic policy")
        self.config, self.head, self.policy = config, head, policy
        self.projection_policy = projection_policy
        self.fn = weights.array(prefix + "_fn.weight")
        self.base = weights.array(prefix + "_base.weight")
        self.scale = weights.array(prefix + "_scale.weight")
        # Checkpoint-owned F16 matrices stay available for provenance/close,
        # but their exact F32 conversion is static, not token-dependent work.
        # Model.parameters/preload retain and materialize these before serving
        # publishes loaded memory or admits its first request.
        self.fn_fp32 = self.fn.astype(mx.float32)
        if projection_policy == "fixed_fp32" and (
            config.hc_mult != 4
            or config.hidden_size * config.hc_mult % 32
            or self.fn_fp32.shape
            != ((4 if head else 24), config.hidden_size * config.hc_mult)
        ):
            raise ValueError(
                "fixed FP32 HC projection requires exact HC4 matrix geometry and K multiple32"
            )
        self.base_fp32 = self.base.astype(mx.float32)
        self.scale_fp32 = self.scale.astype(mx.float32)

    def static_parameters(self):
        return {"fn": self.fn_fp32, "base": self.base_fp32, "scale": self.scale_fp32}

    def __call__(self, value):
        c = self.config
        value32 = value.astype(mx.float32)
        flat = value32.reshape(*value.shape[:-2], -1)
        normalized = mx.fast.rms_norm(flat, None, c.rms_norm_eps)
        if self.projection_policy == "fixed_fp32":
            from .deepseek_v4_hc_projection import project

            mixes = project(normalized, self.fn_fp32)
        else:
            mixes = normalized @ self.fn_fp32.T
        if self.policy == "fused_fp32":
            from .deepseek_v4_hc import head_reduce, split_sinkhorn, weighted_sum

            if self.head:
                return head_reduce(
                    value32, mixes, self.scale_fp32, self.base_fp32, eps=c.hc_eps
                ).astype(value.dtype)
            pre, post, comb = split_sinkhorn(
                mixes,
                self.scale_fp32,
                self.base_fp32,
                iterations=c.hc_sinkhorn_iters,
                eps=c.hc_eps,
            )
            return weighted_sum(value32, pre).astype(value.dtype), post, comb
        if self.head:
            pre = mx.sigmoid(mixes * self.scale_fp32 + self.base_fp32) + c.hc_eps
            return mx.sum(pre[..., None] * value32, axis=-2).astype(value.dtype)
        pre, post, comb = _sinkhorn_mix_fp32(
            mixes,
            self.scale_fp32,
            self.base_fp32,
            c.hc_mult,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )
        return mx.sum(pre[..., None] * value32, axis=-2).astype(value.dtype), post, comb


class Rotary:
    def __init__(self, dims, base, scaling=None):
        self.dims = dims
        inv = base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        if scaling and scaling.get("type") in ("yarn", "deepseek_yarn"):
            original = scaling["original_max_position_embeddings"]

            def correction(rotations):
                return (
                    dims
                    * math.log(original / (rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction(scaling.get("beta_fast", 32))), 0)
            high = min(math.ceil(correction(scaling.get("beta_slow", 1))), dims - 1)
            if high == low:
                high += 0.001
            ramp = mx.clip(
                (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1
            )
            inv = inv * (1 - ramp) + inv / scaling["factor"] * ramp
        elif scaling and scaling.get("type") not in (None, "default"):
            raise ValueError("unsupported DeepSeek V4 rotary scaling")
        self.inv = inv

    def __call__(self, x, offset=0, *, inverse=False, positions=None):
        # Pairwise/"traditional" rotation touches the LAST64 coordinates only.
        positions = (
            mx.arange(offset, offset + x.shape[-2])
            if positions is None
            else mx.array(positions)
        )
        angles = positions.astype(mx.float32)[:, None] * self.inv[None]
        if inverse:
            angles = -angles
        c, s = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        prefix, rotary = x[..., : -self.dims], x[..., -self.dims :]
        a, b = rotary[..., ::2], rotary[..., 1::2]
        rotated = mx.stack((a * c - b * s, a * s + b * c), axis=-1).reshape(
            rotary.shape
        )
        return mx.concatenate((prefix, rotated), axis=-1)


def attend(
    q, local, local_mask, *, pooled=None, pool_mask=None, indices=None, sinks=None
):
    """Ordinary stable attention; sink contributes denominator mass, never value."""
    # Bound score workspace independently of context length. Tiling changes no
    # attended set, masks, sinks or reductions across the key dimension.
    count = local.shape[2] + (
        0
        if pooled is None
        else (indices.shape[-1] if indices is not None else pooled.shape[1])
    )
    tile = max(
        1, min(128, (16 * 1024 * 1024) // max(1, q.shape[0] * q.shape[1] * count * 4))
    )
    if q.shape[2] > tile:
        pieces = []
        for start in range(0, q.shape[2], tile):
            end = start + tile
            pieces.append(
                attend(
                    q[:, :, start:end],
                    local,
                    local_mask[..., start:end, :],
                    pooled=pooled,
                    pool_mask=None
                    if pool_mask is None
                    else pool_mask[..., start:end, :],
                    indices=None if indices is None else indices[:, start:end],
                    sinks=sinks,
                )
            )
        return mx.concatenate(pieces, axis=2)
    scale = q.shape[-1] ** -0.5
    if indices is None:
        kv = (
            local
            if pooled is None
            else mx.concatenate((local, pooled[:, None]), axis=2)
        )
        mask = (
            local_mask
            if pooled is None
            else mx.concatenate((local_mask, pool_mask), axis=-1)
        )
        scores = (q.astype(mx.float32) * scale) @ mx.swapaxes(
            kv.astype(mx.float32), -1, -2
        )
        scores = mx.where(mask, scores, -mx.inf)
        normalizer = mx.logsumexp(scores, axis=-1, keepdims=True)
        if sinks is not None:
            normalizer = mx.logaddexp(
                normalizer, sinks[None, :, None, None].astype(mx.float32)
            )
        return (mx.exp(scores - normalizer) @ kv.astype(mx.float32)).astype(q.dtype)
    # Each query has its own indexer-selected compressed vectors.
    b, _heads, length, dim = q.shape
    gathered = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None], (b, length, pooled.shape[1], dim)),
        mx.broadcast_to(indices[..., None], (*indices.shape, dim)),
        axis=2,
    )
    query = q.transpose(0, 2, 1, 3).astype(mx.float32) * scale
    ls = (q.astype(mx.float32) * scale) @ mx.swapaxes(local.astype(mx.float32), -1, -2)
    ls = mx.where(local_mask, ls, -mx.inf)
    ps = (query @ mx.swapaxes(gathered.astype(mx.float32), -1, -2)).transpose(
        0, 2, 1, 3
    )
    selected_mask = mx.take_along_axis(pool_mask[:, 0], indices, axis=-1)[:, None]
    ps = mx.where(selected_mask, ps, -mx.inf)
    norm = mx.logaddexp(
        mx.logsumexp(ls, -1, keepdims=True), mx.logsumexp(ps, -1, keepdims=True)
    )
    if sinks is not None:
        norm = mx.logaddexp(norm, sinks[None, :, None, None].astype(mx.float32))
    result = mx.exp(ls - norm) @ local.astype(mx.float32)
    result += (
        mx.exp(ps - norm).transpose(0, 2, 1, 3) @ gathered.astype(mx.float32)
    ).transpose(0, 2, 1, 3)
    return result.astype(q.dtype)


def index_select(
    query,
    pooled,
    weights,
    mask,
    top_k,
    *,
    workspace_bytes=16 * 1024 * 1024,
    offset=None,
    ratio=4,
):
    """Exact flat indexer with bounded query-row workspace, no coarse pruning."""
    heads, length, dim = query.shape[1:]
    tile = max(
        1,
        min(
            128, workspace_bytes // max(1, query.shape[0] * heads * pooled.shape[1] * 4)
        ),
    )
    selected = []
    for start in range(0, length, tile):
        end = start + tile
        score = query[:, :, start:end].astype(mx.float32) @ mx.swapaxes(
            pooled[:, None].astype(mx.float32), -1, -2
        )
        score = mx.maximum(score, 0)
        weight = weights[:, start:end].transpose(0, 2, 1)[..., None]
        score = mx.sum(score * weight * dim**-0.5, axis=1)
        if mask is None:
            if type(offset) is not int or offset < 0 or ratio != 4:
                raise ValueError(
                    "flat indexer requires explicit ratio4 causal geometry"
                )
            visible = (mx.arange(start, min(end, length)) + offset + 1) // ratio
            valid = mx.arange(pooled.shape[1])[None, :] < visible[:, None]
        else:
            valid = mask[:, 0, start:end]
        score = mx.where(valid, score, -mx.inf)
        selected.append(stable_pool_topk(score, top_k))
    return mx.concatenate(selected, axis=1)


def stable_pool_topk(scores, top_k):
    """Select exact scores with a chronological cutoff tie-break and order.

    ReLU indexer scores have exact zero ties. Partition's unspecified ordering
    must not change the attended set or summation order across replay shapes.
    See provenance/deepseek-v4-audit-64k.json for the peer mechanism review.
    """
    count = scores.shape[-1]
    if type(top_k) is not int or not 1 <= top_k <= count:
        raise ValueError("pool top-k must fit the available rows")
    candidates = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    cutoff = mx.min(
        mx.take_along_axis(scores, candidates, axis=-1), axis=-1, keepdims=True
    )
    better = scores > cutoff
    tied = scores == cutoff
    remaining = top_k - mx.sum(better, axis=-1, keepdims=True)
    keep = better | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= remaining))
    positions = mx.arange(count, dtype=mx.int32)
    ordered = mx.where(keep, positions, count)
    selected = mx.argpartition(ordered, kth=top_k - 1, axis=-1)[..., :top_k]
    return mx.sort(selected, axis=-1)


class Compressor:
    def __init__(
        self,
        config,
        weights,
        prefix,
        dim,
        ratio,
        *,
        indexer=False,
        rope_qat_policy="reference",
    ):
        self.config, self.weights, self.prefix = config, weights, prefix
        self.dim, self.ratio, self.is_indexer = dim, ratio, indexer
        self.rope_qat_policy = rope_qat_policy
        self.ape = weights.array(prefix + "_ape.weight").astype(mx.float32)
        self.norm = weights.array(prefix + "_norm.weight")
        self.rope = Rotary(
            config.qk_rope_head_dim, config.compress_rope_theta, config.rope_scaling
        )

    def __call__(self, x, cache, offset, *, return_packed=False):
        append = (
            getattr(cache, "append_pooled_packed", None)
            if return_packed
            else cache.append_pooled
        )
        if not callable(append):
            raise TypeError("direct packed attention requires a packed pool cache")
        kv = self.weights.linear(self.prefix + "_kv.weight")(x).astype(mx.float32)
        gate = self.weights.linear(self.prefix + "_gate.weight")(x).astype(mx.float32)
        pooled, starts = cache.compress(kv, gate, self.ape, offset)
        if starts:
            pooled = mx.fast.rms_norm(
                pooled, self.norm, self.config.rms_norm_eps
            ).astype(x.dtype)
            if self.rope_qat_policy == "fused_fp32" and not self.is_indexer:
                from .deepseek_v4_rope_qat import rope_fp8

                pooled = rope_fp8(
                    pooled[:, None], self.rope.inv, mx.array(starts, mx.float32)
                ).squeeze(1)
                return append(pooled)
            pooled = self.rope(pooled[:, None], positions=starts).squeeze(1)
        if pooled.shape[1] and self.config.simulate_quantization:
            pooled = (
                simulate_index_fp4(pooled)
                if self.is_indexer
                else simulate_fp8_kv(pooled, self.config.qk_rope_head_dim)
            )
        return append(pooled)


class Attention:
    def __init__(
        self,
        config,
        weights,
        layer,
        *,
        attention_policy="reference",
        rope_qat_policy="reference",
    ):
        self.c, self.w, self.attention_policy = config, weights, attention_policy
        self.rope_qat_policy = rope_qat_policy
        self.prefix = f"blk.{layer}."
        self.ratio = config.compress_ratios[layer]
        self.rope = Rotary(
            config.qk_rope_head_dim,
            config.compress_rope_theta if self.ratio else config.rope_theta,
            config.rope_scaling if self.ratio else None,
        )
        self.q_norm = weights.array(self.prefix + "attn_q_a_norm.weight")
        self.kv_norm = weights.array(self.prefix + "attn_kv_a_norm.weight")
        self.sinks = weights.array(self.prefix + "attn_sinks.weight")
        if self.ratio:
            self.compressor = Compressor(
                config,
                weights,
                self.prefix + "attn_compressor",
                config.head_dim,
                self.ratio,
                rope_qat_policy=rope_qat_policy,
            )
        if self.ratio == 4:
            self.indexer = Compressor(
                config,
                weights,
                self.prefix + "indexer_compressor",
                config.index_head_dim,
                self.ratio,
                indexer=True,
            )

    def linear(self, x, suffix):
        return self.w.linear(self.prefix + suffix + ".weight")(x)

    def __call__(self, x, cache):
        c, b, length = self.c, x.shape[0], x.shape[1]
        offset = cache.offset
        qr = mx.fast.rms_norm(self.linear(x, "attn_q_a"), self.q_norm, c.rms_norm_eps)
        q = self.linear(qr, "attn_q_b").reshape(
            b, length, c.num_attention_heads, c.head_dim
        )
        q = mx.fast.rms_norm(q, None, c.rms_norm_eps).transpose(0, 2, 1, 3)
        q = self.rope(q, offset)
        kv = mx.fast.rms_norm(self.linear(x, "attn_kv"), self.kv_norm, c.rms_norm_eps)[
            :, None
        ]
        if self.rope_qat_policy == "fused_fp32":
            from .deepseek_v4_rope_qat import rope_fp8

            kv = rope_fp8(
                kv, self.rope.inv, mx.arange(offset, offset + length, dtype=mx.float32)
            )
        else:
            kv = self.rope(kv, offset)
            if c.simulate_quantization:
                kv = simulate_fp8_kv(kv, c.qk_rope_head_dim)
        local = cache.update_local(kv)
        keys = mx.arange(offset + length - local.shape[2], offset + length)
        queries = mx.arange(offset, offset + length)[:, None]
        local_mask = (
            None
            if self.attention_policy
            in ("visible_rows_fp32", "packed_visible_rows_fp32")
            else ((queries >= keys) & (queries < keys + c.sliding_window))[None, None]
        )
        pooled = pool_mask = indices = None
        direct_packed = self.attention_policy == "packed_visible_rows_fp32"
        if self.ratio:
            pooled = self.compressor(
                x, cache.compressor, offset, return_packed=direct_packed
            )
            pool_count = (
                0
                if pooled is None
                else pooled.codes.shape[1]
                if direct_packed
                else pooled.shape[1]
            )
            if pool_count:
                if self.attention_policy == "reference":
                    pool_mask = cache.compressor.make_mask(length, offset)[None, None]
                if self.ratio == 4:
                    ipool = self.indexer(x, cache.indexer, offset)
                    if pool_count > c.index_topk:
                        iq = (
                            self.linear(qr, "indexer.attn_q_b")
                            .reshape(b, length, c.index_n_heads, c.index_head_dim)
                            .transpose(0, 2, 1, 3)
                        )
                        iq = self.rope(iq, offset)
                        if c.simulate_quantization:
                            iq = simulate_index_fp4(iq)
                        iw = (
                            self.linear(x, "indexer.proj").astype(mx.float32)
                            * c.index_n_heads**-0.5
                        )
                        indices = index_select(
                            iq,
                            ipool,
                            iw,
                            pool_mask,
                            c.index_topk,
                            offset=offset,
                            ratio=self.ratio,
                        )
            elif self.ratio == 4:
                # Project/advance indexer even before the first complete pool.
                self.indexer(x, cache.indexer, offset)
        if direct_packed:
            from .deepseek_v4_packed_visible_attention import packed_visible_attention

            out = packed_visible_attention(
                q,
                local,
                pooled,
                indices,
                self.sinks,
                offset=offset,
                window=c.sliding_window,
                ratio=self.ratio,
            )
        elif self.attention_policy == "visible_rows_fp32":
            from .deepseek_v4_visible_attention import visible_attention

            out = visible_attention(
                q,
                local,
                pooled,
                indices,
                self.sinks,
                offset=offset,
                window=c.sliding_window,
                ratio=self.ratio,
            )
        else:
            out = attend(
                q,
                local,
                local_mask,
                pooled=pooled if pooled is not None and pooled.shape[1] else None,
                pool_mask=pool_mask,
                indices=indices,
                sinks=self.sinks,
            )
        out = self.rope(out, offset, inverse=True)
        # [B,H,L,D] -> [B,G,L,H/G*D], never flatten L into a head.
        out = out.reshape(
            b, c.o_groups, c.num_attention_heads // c.o_groups, length, c.head_dim
        )
        out = out.transpose(0, 3, 1, 2, 4).reshape(b, length, c.o_groups, -1)
        out = self.w.grouped(
            out, self.prefix + "attn_output_a.weight", groups=c.o_groups
        )
        out = out.reshape(b, length, c.o_groups * c.o_lora_rank)
        return self.linear(out, "attn_output_b")


class Block:
    def __init__(
        self,
        config,
        weights,
        layer,
        *,
        attention_policy="reference",
        hc_policy="reference",
        hc_projection_policy="reference",
        rope_qat_policy="reference",
    ):
        self.c, self.w, self.layer = config, weights, layer
        self.hc_policy = hc_policy
        self.prefix = f"blk.{layer}."
        self.attn = Attention(
            config,
            weights,
            layer,
            attention_policy=attention_policy,
            rope_qat_policy=rope_qat_policy,
        )
        self.attn_hc = HyperConnection(
            config,
            weights,
            self.prefix + "hc_attn",
            policy=hc_policy,
            projection_policy=hc_projection_policy,
        )
        self.ffn_hc = HyperConnection(
            config,
            weights,
            self.prefix + "hc_ffn",
            policy=hc_policy,
            projection_policy=hc_projection_policy,
        )
        self.attn_norm = weights.array(self.prefix + "attn_norm.weight")
        self.ffn_norm = weights.array(self.prefix + "ffn_norm.weight")
        self.hash_table = (
            weights.array(self.prefix + "ffn_gate_tid2eid.weight")
            if layer < config.num_hash_layers
            else None
        )
        self.correction = (
            weights.array(self.prefix + "exp_probs_b.bias")
            if layer >= config.num_hash_layers
            else None
        )

    def __call__(self, h, ids, cache):
        x, post, comb = self.attn_hc(h)
        x = self.attn(mx.fast.rms_norm(x, self.attn_norm, self.c.rms_norm_eps), cache)
        h = hc_expand(x, h, post, comb, policy=self.hc_policy)
        x, post, comb = self.ffn_hc(h)
        x = mx.fast.rms_norm(x, self.ffn_norm, self.c.rms_norm_eps)
        logits = self.w.linear(self.prefix + "ffn_gate_inp.weight")(x)
        indices, scores = route_experts(
            logits,
            top_k=self.c.num_experts_per_tok,
            scale=self.c.routed_scaling_factor,
            correction=self.correction,
            token_ids=ids,
            table=self.hash_table,
            normalize=self.c.norm_topk_prob,
        )
        if self.c.swiglu_limit != 10.0:
            raise ValueError("packed experts currently require the checkpoint clip=10")
        routed = self.w.moe(
            x.reshape(-1, x.shape[-1]),
            indices.reshape(-1, self.c.num_experts_per_tok),
            scores.reshape(-1, self.c.num_experts_per_tok),
            self.prefix[:-1],
        ).reshape(x.shape)
        names = (
            self.prefix + "ffn_gate_shexp.weight",
            self.prefix + "ffn_up_shexp.weight",
        )
        pair = getattr(self.w, "gate_up", None)
        gate, up = (
            pair(x, *names)
            if callable(pair)
            else tuple(self.w.linear(n)(x) for n in names)
        )
        shared = self.w.linear(self.prefix + "ffn_down_shexp.weight")(
            limited_swiglu(gate, up, self.c.swiglu_limit)
        )
        return hc_expand(routed + shared, h, post, comb, policy=self.hc_policy)


class DeepSeekV4Model:
    """Ordinary B1 autoregressive slice; unsupported shapes fail closed."""

    def __init__(
        self,
        config,
        weights,
        *,
        identity="unbound",
        attention_policy="reference",
        cache_policy="fp32",
        hc_policy="reference",
        hc_projection_policy="reference",
        expert_prefill_policy="reference",
        rope_qat_policy="reference",
        dense_prefill_policy="reference",
    ):
        if hc_projection_policy not in ("reference", "fixed_fp32"):
            raise ValueError("unknown DeepSeek V4 HC projection arithmetic policy")
        if hc_projection_policy == "fixed_fp32" and (
            config.hc_mult != 4 or config.hidden_size * config.hc_mult % 32
        ):
            raise ValueError("fixed FP32 HC projection requires HC4 and K multiple32")
        if dense_prefill_policy not in ("reference", "nax_fp16"):
            raise ValueError("unknown DeepSeek V4 dense prefill arithmetic policy")
        if rope_qat_policy not in ("reference", "fused_fp32"):
            raise ValueError("unknown DeepSeek V4 RoPE/QAT arithmetic policy")
        if rope_qat_policy == "fused_fp32" and (
            config.head_dim != 512
            or config.qk_rope_head_dim != 64
            or not config.simulate_quantization
        ):
            raise ValueError("fused RoPE/QAT requires D512, rotary64 and mandatory QAT")
        if expert_prefill_policy not in ("reference", "segmented_nax_fp16"):
            raise ValueError("unknown DeepSeek V4 expert prefill arithmetic policy")
        if hc_policy not in ("reference", "fused_fp32"):
            raise ValueError("unknown DeepSeek V4 hyper-connection arithmetic policy")
        if hc_policy == "fused_fp32" and config.hc_mult != 4:
            raise ValueError("fused FP32 hyper-connections require HC4")
        if cache_policy not in ("fp32", "fp8_mixed"):
            raise ValueError("unknown DeepSeek V4 physical cache policy")
        if cache_policy == "fp8_mixed" and (
            config.head_dim != 512
            or config.qk_rope_head_dim != 64
            or not config.simulate_quantization
        ):
            raise ValueError(
                "physical FP8 cache requires checkpoint D512, rotary64 and required QAT enabled"
            )
        if attention_policy not in (
            "reference",
            "visible_rows_fp32",
            "packed_visible_rows_fp32",
        ):
            raise ValueError("unknown DeepSeek V4 attention arithmetic policy")
        if (
            attention_policy == "packed_visible_rows_fp32"
            and cache_policy != "fp8_mixed"
        ):
            raise ValueError(
                "direct packed attention requires explicit fp8_mixed cache"
            )
        if attention_policy != "reference" and config.head_dim != 512:
            raise ValueError("visible attention requires checkpoint D512")
        self.args, self.weights = config, weights
        from .deepseek_v4_identity import model_execution_identity

        self.identity = model_execution_identity(
            identity,
            attention_policy=attention_policy,
            cache_policy=cache_policy,
            hc_policy=hc_policy,
            hc_projection_policy=hc_projection_policy,
            expert_prefill_policy=expert_prefill_policy,
            dense_prefill_policy=dense_prefill_policy,
            rope_qat_policy=rope_qat_policy,
        )
        self.hc_projection_policy = hc_projection_policy
        self.dense_prefill_policy = dense_prefill_policy
        self.rope_qat_policy = rope_qat_policy
        self.expert_prefill_policy = expert_prefill_policy
        self.cache_policy = cache_policy
        self.hc_policy = hc_policy
        self.attention_policy = attention_policy
        self.layers = [
            Block(
                config,
                weights,
                i,
                attention_policy=attention_policy,
                hc_policy=hc_policy,
                hc_projection_policy=hc_projection_policy,
                rope_qat_policy=rope_qat_policy,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.hc_head = HyperConnection(
            config,
            weights,
            "output_hc",
            head=True,
            policy=hc_policy,
            projection_policy=hc_projection_policy,
        )
        self.norm = weights.array("output_norm.weight")

    def make_cache(self):
        cache_type = DeepSeekV4LayerCache
        if self.cache_policy == "fp8_mixed":
            from .deepseek_v4_packed_cache import DeepSeekV4PackedLayerCache

            cache_type = DeepSeekV4PackedLayerCache
        return [
            cache_type(
                r,
                xp=mx,
                identity=self.identity,
                head_dim=self.args.head_dim,
                index_head_dim=self.args.index_head_dim,
                window_size=self.args.sliding_window,
            )
            for r in self.args.compress_ratios
        ]

    def _forward(
        self,
        inputs,
        cache=None,
        *,
        logits_to_keep=None,
        capture_layers=(),
        body_only=False,
        row_exact=False,
    ):
        if logits_to_keep is not None and (
            type(logits_to_keep) is not int or logits_to_keep < 0
        ):
            raise ValueError("logits_to_keep must be nonnegative integer or None")
        if inputs.ndim != 2 or inputs.shape[0] != 1 or not inputs.shape[1]:
            raise ValueError(
                "DeepSeek V4 ordinary reference currently requires nonempty B1 tokens"
            )
        if cache is None:
            cache = self.make_cache()
        if len(cache) != len(self.layers):
            raise ValueError("DeepSeek V4 cache layer count mismatch")
        if len({id(item) for item in cache}) != len(cache):
            raise ValueError("DeepSeek V4 requires independent cache objects per layer")
        expected_type = DeepSeekV4LayerCache
        if self.cache_policy == "fp8_mixed":
            from .deepseek_v4_packed_cache import DeepSeekV4PackedLayerCache

            expected_type = DeepSeekV4PackedLayerCache
        for ratio, item in zip(self.args.compress_ratios, cache):
            if (
                type(item) is not expected_type
                or item.identity != self.identity
                or item.ratio != ratio
                or item.head_dim != self.args.head_dim
                or item.index_head_dim != self.args.index_head_dim
                or item.window_size != self.args.sliding_window
                or item.pending
            ):
                raise ValueError(
                    "DeepSeek V4 cache identity, layout, geometry or boundary mismatch"
                )
            item.validate_state()
        offsets = {item.offset for item in cache}
        if len(offsets) != 1:
            raise ValueError("DeepSeek V4 cache layers must represent one prefix")
        x = self.weights.embedding(inputs).astype(mx.float32)
        h = mx.broadcast_to(
            x[:, :, None], (*x.shape[:2], self.args.hc_mult, self.args.hidden_size)
        )
        taps = []
        for index, (block, state) in enumerate(zip(self.layers, cache)):
            if row_exact:
                # Layer-major verification keeps each stateful operation at
                # the ordinary decode shape. No M>1 projection, compressed
                # attention reduction or HC mix may choose the target law.
                rows = [
                    block(h[:, row : row + 1], inputs[:, row : row + 1], state)
                    for row in range(inputs.shape[1])
                ]
                h = mx.concatenate(rows, axis=1)
            else:
                h = block(h, inputs, state)
            if index in capture_layers:
                # DSpark target features are the uniform mean across four
                # POST-FFN HC streams, not the learned output HC head.
                taps.append(
                    mx.concatenate(
                        [mx.mean(row.astype(mx.float32), axis=-2) for row in rows],
                        axis=1,
                    )
                    if row_exact
                    else mx.mean(h.astype(mx.float32), axis=-2)
                )
        features = mx.concatenate(taps, axis=-1) if taps else None
        if body_only:
            return None, features
        if row_exact:
            logits = []
            for row in range(inputs.shape[1]):
                hidden = mx.fast.rms_norm(
                    self.hc_head(h[:, row : row + 1]), self.norm, self.args.rms_norm_eps
                )
                logits.append(self.weights.linear("output.weight")(hidden))
            return mx.concatenate(logits, axis=1), features
        hidden = mx.fast.rms_norm(self.hc_head(h), self.norm, self.args.rms_norm_eps)
        if logits_to_keep is None:
            hidden = select_checkpoint_rows(hidden)
        elif logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return self.weights.linear("output.weight")(hidden), features

    def __call__(self, inputs, cache=None, *, logits_to_keep=None):
        return self._forward(inputs, cache, logits_to_keep=logits_to_keep)[0]

    def forward_with_taps(
        self, inputs, cache, capture_layers, *, body_only=False, return_metadata=False
    ):
        """Exact per-token POST-FFN HC-mean features for an explicitly bound drafter.

        No layer IDs are inferred from a different checkpoint or sidecar.
        Captured features retain every input position; checkpoint-head thinning
        never narrows the drafter feature history.
        """
        capture_layers = tuple(capture_layers)
        if (
            not capture_layers
            or any(type(i) is not int for i in capture_layers)
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.layers)
        ):
            raise ValueError("target tap layers must be unique ascending valid IDs")
        start = 0 if cache is None else cache[0].offset
        result = self._forward(
            inputs, cache, capture_layers=capture_layers, body_only=body_only
        )
        if not return_metadata:
            return result
        return (
            *result,
            {
                "target_identity": self.identity,
                "capture_layers": capture_layers,
                "feature_law": "post-ffn-uniform-HC-mean-fp32",
                "start_position": start,
                "end_position": start + inputs.shape[1],
                "physical_width": 1,
                "feature_width": len(capture_layers) * self.args.hidden_size,
            },
        )

    def verify_rows(self, inputs, cache, capture_layers=()):
        """Bounded causal verification with ordinary width-one math per layer.

        This is a chain executor, not a packed tree or a fused M>1 kernel.
        Prefill's numerical profile and checkpoint-head thinning never enter
        verification. Return every target logit and every requested HC tap.
        """
        capture_layers = tuple(capture_layers)
        if inputs.ndim != 2 or inputs.shape[0] != 1 or not 1 <= inputs.shape[1] <= 6:
            raise ValueError("DeepSeek verification requires bounded B1 rows1..6")
        if (
            any(type(i) is not int for i in capture_layers)
            or tuple(sorted(set(capture_layers))) != capture_layers
            or any(i < 0 or i >= len(self.layers) for i in capture_layers)
        ):
            raise ValueError("target tap layers must be unique ascending valid IDs")
        return self._forward(
            inputs,
            cache,
            logits_to_keep=0,
            capture_layers=capture_layers,
            row_exact=True,
        )

    def prefill_body(self, inputs, cache, capture_layers):
        from ..gguf.prefill_scope import dense_prefill_context
        from ..gguf.segmented_prefill import expert_prefill_context

        with (
            expert_prefill_context(self.expert_prefill_policy, tokens=inputs.shape[1]),
            dense_prefill_context(self.dense_prefill_policy, tokens=inputs.shape[1]),
        ):
            return self.forward_with_taps(
                inputs, cache, capture_layers, body_only=True
            )[1]

    def prefill_forward(self, inputs, cache=None):
        """Honor the generator's explicit checkpoint-row projection context.

        Outside that context this is a full-logit ordinary reference forward;
        verification and AST do not inherit a hidden last-only behavior.
        """
        from ..gguf.prefill_scope import dense_prefill_context
        from ..gguf.segmented_prefill import expert_prefill_context

        with (
            expert_prefill_context(self.expert_prefill_policy, tokens=inputs.shape[1]),
            dense_prefill_context(self.dense_prefill_policy, tokens=inputs.shape[1]),
        ):
            return self(inputs, cache=cache)

    def eval(self):
        """Inference-only model; materialize only already owned resident tensors."""
        mx.eval(self.parameters())
        return self

    def named_modules(self):
        """Expose the real non-replaceable root to generic serving inspection.

        Blocks and packed GGUF providers are adapter-owned Python helpers,
        not MLX nn.Module/QuantizedLinear instances. Advertising them as
        affine linears would permit an incompatible generic lane rewrite.
        Their own selected kernels and counters live in the weight provider.
        """
        yield "", self

    def parameters(self):
        native = getattr(self.weights, "_dense", {})
        packed = getattr(self.weights, "_packed", {})
        native_linear_parameters = getattr(self.weights, "native_fp32_parameters", None)
        native_linear_fp32 = (
            native_linear_parameters() if callable(native_linear_parameters) else {}
        )
        hc = {"output_hc": self.hc_head.static_parameters()}
        attention_static = {}
        for index, block in enumerate(self.layers):
            hc[f"blk.{index}.hc_attn"] = block.attn_hc.static_parameters()
            hc[f"blk.{index}.hc_ffn"] = block.ffn_hc.static_parameters()
            attn = block.attn
            values = {"rotary_inv": attn.rope.inv}
            for name in ("compressor", "indexer"):
                component = getattr(attn, name, None)
                if component is not None:
                    values[name] = {
                        "rotary_inv": component.rope.inv,
                        "ape_fp32": component.ape,
                    }
            attention_static[f"blk.{index}"] = values
        return {
            "native": dict(native),
            "packed": {k: v.data for k, v in packed.items()},
            "hc_static_fp32": hc,
            "native_linear_fp32": native_linear_fp32,
            "attention_static": attention_static,
        }

    def preload(self):
        """Own all model weights before publishing loaded-memory/admission status."""
        for name, kind in required_weights(self.args).items():
            if kind == "array":
                self.weights.array(name)
            elif kind == "expert":
                self.weights.packed(name)
            elif kind == "embedding":
                prepare = getattr(self.weights, "prepare_embedding", None)
                if callable(prepare):
                    prepare(name)
                    continue
                # Native embedding upload; no full vocabulary activation gather.
                tensor = self.weights.tensors[name]
                if tensor.ggml_type in (0, 1, 30):
                    self.weights.array(name)
                else:
                    self.weights.packed(name)
            else:
                self.weights.linear(name)
        preload_native_linear = getattr(
            self.weights, "preload_native_linear_fp32", None
        )
        if callable(preload_native_linear):
            preload_native_linear()
        mx.eval(self.parameters())
        if self.attention_policy == "visible_rows_fp32":
            from .deepseek_v4_visible_attention import self_check

            self_check()
        elif self.attention_policy == "packed_visible_rows_fp32":
            from .deepseek_v4_packed_visible_attention import self_check

            self_check()
        return self


def required_weights(config):
    """Exact ordinary-transformer names; auxiliary next-token/vision weights separate."""
    result = {
        "token_embd.weight": "embedding",
        "output.weight": "linear",
        "output_norm.weight": "array",
    }

    def hc(prefix):
        for name in ("fn", "base", "scale"):
            result[prefix + "_" + name + ".weight"] = "array"

    hc("output_hc")
    for i, ratio in enumerate(config.compress_ratios):
        p = f"blk.{i}."
        hc(p + "hc_attn")
        hc(p + "hc_ffn")
        for name in (
            "attn_q_a",
            "attn_q_b",
            "attn_kv",
            "attn_output_a",
            "attn_output_b",
            "ffn_gate_inp",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ):
            result[p + name + ".weight"] = "linear"
        for name in (
            "attn_q_a_norm",
            "attn_kv_a_norm",
            "attn_norm",
            "ffn_norm",
            "attn_sinks",
        ):
            result[p + name + ".weight"] = "array"
        result[
            p
            + (
                "ffn_gate_tid2eid.weight"
                if i < config.num_hash_layers
                else "exp_probs_b.bias"
            )
        ] = "array"
        for name in ("gate", "up", "down"):
            result[p + "ffn_" + name + "_exps.weight"] = "expert"
        if ratio:
            prefixes = [p + "attn_compressor"] + (
                [p + "indexer_compressor"] if ratio == 4 else []
            )
            for comp in prefixes:
                for name in ("kv", "gate"):
                    result[comp + "_" + name + ".weight"] = "linear"
                for name in ("norm", "ape"):
                    result[comp + "_" + name + ".weight"] = "array"
        if ratio == 4:
            result[p + "indexer.attn_q_b.weight"] = "linear"
            result[p + "indexer.proj.weight"] = "linear"
    return result


def validate_weight_inventory(config, tensors):
    """Validate presence/critical packed layout using index metadata, never payload."""
    required = required_weights(config)
    missing = sorted(set(required) - set(tensors))
    if missing:
        raise ValueError(
            "DeepSeek V4 GGUF missing ordinary tensors: " + ", ".join(missing)
        )
    c = config
    for i in range(c.num_hidden_layers):
        expected = {
            f"blk.{i}.attn_output_a.weight": (
                c.o_groups * c.o_lora_rank,
                c.num_attention_heads * c.head_dim // c.o_groups,
            ),
            f"blk.{i}.ffn_gate_exps.weight": (
                c.n_routed_experts,
                c.moe_intermediate_size,
                c.hidden_size,
            ),
            f"blk.{i}.ffn_up_exps.weight": (
                c.n_routed_experts,
                c.moe_intermediate_size,
                c.hidden_size,
            ),
            f"blk.{i}.ffn_down_exps.weight": (
                c.n_routed_experts,
                c.hidden_size,
                c.moe_intermediate_size,
            ),
        }
        for name, shape in expected.items():
            tensor = tensors[name]
            actual = (
                tuple(tensor.numpy_shape)
                if hasattr(tensor, "numpy_shape")
                else tuple(tensor["shape"])[::-1]
            )
            if actual != shape:
                raise ValueError(
                    f"DeepSeek V4 GGUF layout mismatch: {name}: {actual} != {shape}"
                )
    return {
        "required_ordinary_tensors": len(required),
        "extra_auxiliary_tensors": len(set(tensors) - set(required)),
    }
