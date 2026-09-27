# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/hy-v3.json.

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    expert_hidden_dim: int
    first_k_dense_replace: int
    rms_norm_eps: float
    rope_parameters: Dict[str, Any]
    router_scaling_factor: float = 1.0
    qk_norm: bool = True
    route_norm: bool = True
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    # The MTP (nextn) sidecar keeps the base model's full routed-expert set even
    # on REAP-pruned trunks, so its expert count can differ from num_experts.
    mtp_num_experts: Optional[int] = None
    max_position_embeddings: int = 262144
    enable_moe_fp32_combine: bool = False
    enable_lm_head_fp32: bool = False


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.use_qk_norm = args.qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            dims=self.head_dim,
            base=args.rope_parameters["rope_theta"],
            traditional=False,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def expert_select(
    gates,
    expert_bias,
    top_k,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + expert_bias

    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.route_norm
        self.routed_scaling_factor = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        result = expert_select(
            self.gate(x.astype(mx.float32)),
            self.expert_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        sink = getattr(self, "_hy3_atlas_sink", None)
        if sink is not None:
            sink(*result)
        return result


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.expert_hidden_dim,
            args.num_experts,
        )
        self.router = MoEGate(args)
        if args.num_shared_experts > 0:
            self.shared_mlp = MLP(
                args.hidden_size,
                args.expert_hidden_dim * args.num_shared_experts,
            )
        else:
            self.shared_mlp = None

        self.fp32_combine = args.enable_moe_fp32_combine
        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.router(x)
        if not self.fp32_combine:
            scores = scores.astype(x.dtype)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.shared_mlp is not None:
            y = y + self.shared_mlp(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y.astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class HYV3Model(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0])

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            h = layer(h, mask, cache=c)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class HYV3MTP(nn.Module):
    """Multi-token-prediction (nextn) sidecar — mirrors the reference HYV3
    MTP layer (vLLM hy_v3_mtp.py):

        eh_proj([enorm(embed(t_{p+1})); hnorm(hidden_p)]) -> one full
        DecoderLayer (own KV cache) -> final_layernorm -> the trunk's lm_head.

    Predicts token p+2 from the trunk's post-final-norm hidden at p and the
    committed token p+1, enabling self-speculative decoding with no external
    draft model. Built only when args.num_nextn_predict_layers > 0. Its MoE
    keeps the base model's full expert set (mtp_num_experts), which may exceed
    a REAP-pruned trunk's num_experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        mtp_args = replace(args, num_experts=args.mtp_num_experts or args.num_experts)
        # layer_idx >= first_k_dense_replace so the block is MoE.
        self.layer = DecoderLayer(mtp_args, layer_idx=args.num_hidden_layers)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


# HF/tencent checkpoints store the depth-1 MTP sidecar as an extra decoder
# layer (index num_hidden_layers). These suffixes live at the sidecar root;
# everything else belongs to its inner DecoderLayer.
_MTP_ROOT_SUFFIXES = ("eh_proj.", "enorm.", "hnorm.", "final_layernorm.")


class Model(nn.Module):
    apc_v2_layout = "hy-v3-full-kv-v1"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HYV3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # Ordinary decode does not construct the unqualified MTP sidecar.

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        return self.logits(self.model(inputs, cache))

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def logits(self, hidden: mx.array) -> mx.array:
        if self.args.enable_lm_head_fp32:
            hidden = hidden.astype(mx.float32)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def make_mtp_cache(self):
        return [KVCache()]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """One MTP forward over S positions.

        hidden: [B, S, H] post-final-norm trunk hiddens at positions p..p+S-1.
        tokens: [B, S] the committed/drafted token FOLLOWING each hidden's
        position (p+1..p+S). Returns (last_logits [B, 1, V], post_norm_hidden
        [B, S, H]) — logits only for the final position, since drafting (and
        teacher-forced context feeding) never consumes the earlier ones and
        the fp32 lm_head over the full span would dominate the head's cost.
        """
        e = self.mtp.enorm(self.model.embed_tokens(tokens))
        h = self.mtp.hnorm(hidden)
        x = self.mtp.eh_proj(mx.concatenate([e, h], axis=-1))
        mask = create_attention_mask(x, mtp_cache[0])
        x = self.mtp.layer(x, mask=mask, cache=mtp_cache[0])
        post = self.mtp.final_layernorm(x)
        return self.logits(post[:, -1:, :]), post

    def _remap_mtp_weights(self, weights):
        """Remap HF-format MTP tensors (model.layers.<n_layers + i>.*) onto the
        mtp.* module namespace. Depth-1 only: extra MTP layers are dropped."""
        n_layers = self.args.num_hidden_layers
        n_mtp = self.args.num_nextn_predict_layers
        for i in range(1, n_mtp):
            prefix = f"model.layers.{n_layers + i}."
            weights = {k: v for k, v in weights.items() if not k.startswith(prefix)}

        prefix = f"model.layers.{n_layers}."
        remapped = {}
        for k, v in weights.items():
            if not k.startswith(prefix):
                remapped[k] = v
                continue
            suffix = k[len(prefix) :]
            if suffix.startswith(_MTP_ROOT_SUFFIXES):
                remapped[f"mtp.{suffix}"] = v
            else:
                remapped[f"mtp.layer.{suffix}"] = v
        return remapped

    def sanitize(self, weights):
        n_layers = self.args.num_hidden_layers
        n_mtp = 0
        weights = {k: v for k, v in weights.items() if not k.startswith("mtp.") and not k.startswith(f"model.layers.{n_layers}.")}

        if n_mtp > 0:
            if any(k.startswith(f"model.layers.{n_layers}.") for k in weights):
                weights = self._remap_mtp_weights(weights)
            if getattr(self, "mtp", None) is not None and not any(
                k.startswith("mtp.") for k in weights
            ):
                # Config declares a nextn head but this checkpoint shipped
                # without one (the common MLX repack): drop the module so
                # strict loading stays consistent.
                self.mtp = None

        mlp_prefixes = [f"model.layers.{l}.mlp" for l in range(n_layers)]
        if getattr(self, "mtp", None) is not None:
            mlp_prefixes.append("mtp.layer.mlp")

        n_experts = self.args.num_experts
        mtp_experts = self.args.mtp_num_experts or n_experts
        for prefix in mlp_prefixes:
            bias_key = f"{prefix}.expert_bias"
            if bias_key in weights:
                weights[f"{prefix}.router.expert_bias"] = weights.pop(bias_key)

            n_e = mtp_experts if prefix.startswith("mtp.") else n_experts
            for m in ("gate_proj", "down_proj", "up_proj"):
                for k in ("weight", "scales", "biases"):
                    if f"{prefix}.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.experts.{e}.{m}.{k}")
                            for e in range(n_e)
                        ]
                        weights[f"{prefix}.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads = max(1, layer.self_attn.n_kv_heads // N)

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
            else:
                layer.mlp.sharding_group = group
                if layer.mlp.shared_mlp is not None:
                    shard_inplace(
                        layer.mlp.shared_mlp.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_mlp.down_proj,
                        "sharded-to-all",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_mlp.up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.router.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate
