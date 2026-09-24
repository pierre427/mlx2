# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/nemotron3-super-5bit.json.
import copy
from dataclasses import dataclass
from typing import Any, ClassVar

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, BatchKVCache, KVCache
from .ssm import ssm_update
from .switch_layers import SwitchMLP


@dataclass()
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    mlp_bias: bool
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    hybrid_override_pattern: list[str] | None = None
    layers_block_type: list[str] | None = None
    # nemotron_h_puzzle (Puzzle-NAS): heterogeneous MoE. Per-layer list aligned
    # with layers_block_type; MoE entries carry their own `moe_intermediate_size`
    # and `num_experts_per_tok`. Absent (plain nemotron_h) => uniform dims.
    block_configs: list[dict] | None = None
    head_dim: int | None = None
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_latent_size: int | None = None
    n_group: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    topk_group: int | None = None
    num_experts_per_tok: int | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    time_step_limit: tuple[float, float] | None = None
    time_step_min: float | None = None
    time_step_max: float | None = None
    # Multi-token-prediction head (Nemotron 3 Super checkpoints). The module
    # is built when num_nextn_predict_layers > 0 and dropped again in
    # sanitize() if the checkpoint carries no mtp.* tensors.
    num_nextn_predict_layers: int = 0
    mtp_layers_block_type: list[str] | None = None
    mtp_hybrid_override_pattern: list[str] | None = None
    mtp_block_configs: list[dict] | None = None

    # Map from layers_block_type names to single-char pattern codes
    _block_type_to_char: ClassVar[dict[str, str]] = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}

    def __post_init__(self):
        if self.time_step_limit is None:
            self.time_step_limit = (0.0, float("inf"))

        # Normalize to hybrid_override_pattern (single-char list)
        if self.hybrid_override_pattern is None and self.layers_block_type is not None:
            self.hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.layers_block_type
            ]
        if self.hybrid_override_pattern is not None:
            self.num_hidden_layers = len(self.hybrid_override_pattern)

        if (
            self.mtp_hybrid_override_pattern is None
            and self.mtp_layers_block_type is not None
        ):
            self.mtp_hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.mtp_layers_block_type
            ]
        if isinstance(self.mtp_hybrid_override_pattern, str):
            self.mtp_hybrid_override_pattern = list(self.mtp_hybrid_override_pattern)


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, group_size: int):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)
        self.group_size = group_size

    def __call__(self, x: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            x = swiglu(gate, x)
        x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.flatten(-2)


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.mamba_num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.mamba_num_heads * args.mamba_head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.mamba_head_dim
        self.time_step_limit = args.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size, projection_size, bias=args.mamba_proj_bias
        )

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        group_size = self.intermediate_size // self.n_groups
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=args.layer_norm_epsilon,
            group_size=group_size,
        )
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.mamba_proj_bias
        )

    def _conv(
        self,
        conv_input: mx.array,
        cache: ArraysCache | None,
        mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, conv_input], axis=1)
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                t = padded_input.shape[1]
                ends = mx.clip(cache.lengths, 0, t - n_keep)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(padded_input, positions, axis=1)
            else:
                cache[0] = padded_input[:, -n_keep:, :]
        else:
            padded_input = mx.pad(
                conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)]
            )

        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output), padded_input

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: ArraysCache | None,
        mask: mx.array | None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
        else:
            state = None

        y, state = ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D.astype(hidden_states.dtype),
            dt,
            self.dt_bias,
            state,
            self.time_step_limit,
            mask,
        )
        if cache:
            cache[1] = state

        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None,
        cache: ArraysCache | None = None,
        ssm_sink: list | None = None,
    ) -> mx.array:
        initial_conv = cache[0] if cache is not None else None
        initial_state = cache[1] if cache is not None else None
        projected = self.in_proj(hidden_states)

        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output, padded_input = self._conv(conv_input, cache, mask)
        hidden_states_ssm, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        if cache is not None and cache.speculating:
            steps = hidden_states_ssm.shape[1]
            # Save the post-convolution inputs and the pre-update recurrent
            # state. A rejected draft replays only the accepted prefix; merely
            # trimming the KV layers would leave Mamba on the rejected path.
            x = hidden_states_ssm.reshape(
                hidden_states_ssm.shape[0], steps, self.num_heads, self.head_dim
            )
            b = B.reshape(B.shape[0], steps, self.n_groups, self.ssm_state_size)
            c = C.reshape(C.shape[0], steps, self.n_groups, self.ssm_state_size)
            keep = self.conv_kernel_size - 1

            def rollback(prefix, *, x=x, b=b, c=c, dt=dt,
                         padded=padded_input, state=initial_state):
                _, restored = ssm_update(
                    x[:, :prefix], self.A_log, b[:, :prefix], c[:, :prefix],
                    self.D.astype(x.dtype), dt[:, :prefix], self.dt_bias,
                    state, self.time_step_limit,
                    None if mask is None else mask[:, :prefix],
                )
                return [mx.contiguous(padded[:, prefix:prefix + keep]), restored]

            cache.record_rollback(
                steps, rollback, [initial_conv, initial_state]
            )
        if ssm_sink is not None:
            # Everything needed to replay this update on an accepted prefix
            # during speculative rollback; tuple layout consumed by
            # Model.rollback_speculative_cache. Captured pre-_ssm so `state`
            # is the value before this update overwrites cache[1].
            bsz, S, _ = hidden_states_ssm.shape
            ssm_sink.append(
                (
                    hidden_states_ssm.reshape(bsz, S, self.num_heads, self.head_dim),
                    B.reshape(bsz, S, self.n_groups, self.ssm_state_size),
                    C.reshape(bsz, S, self.n_groups, self.ssm_state_size),
                    dt,
                    self.A_log,
                    self.D,
                    self.dt_bias,
                    self.time_step_limit,
                    cache[1] if cache else None,
                    mask,
                    padded_input,
                    self.conv_kernel_size,
                )
            )
        y = self._ssm(hidden_states_ssm, B, C, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate)
        return self.out_proj(y)


class NemotronHAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = (
            args.head_dim
            if args.head_dim is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        self.num_key_value_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=args.attention_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class NemotronHMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size

        self.up_proj = nn.Linear(
            args.hidden_size, intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x):
        return self.down_proj(nn.relu2(self.up_proj(x)))


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        # Puzzle configs omit group routing / scaling; default to the identity
        # (no grouping, unit scale) so group_expert_select's `n_group > 1`
        # guard doesn't hit a `None > 1` TypeError.
        self.routed_scaling_factor = config.routed_scaling_factor or 1.0
        self.n_group = config.n_group or 1
        self.topk_group = config.topk_group or 1
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class NemotronHMoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_latent_size = config.moe_latent_size

        # When latent projection is used, experts operate on the latent dim
        expert_input_dim = (
            config.moe_latent_size
            if config.moe_latent_size is not None
            else config.hidden_size
        )
        self.switch_mlp = SwitchMLP(
            expert_input_dim,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=nn.ReLU2(),
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_shared_expert_intermediate_size
            self.shared_experts = NemotronHMLP(
                config, intermediate_size=intermediate_size
            )

        # Latent projection layers for dimensionality reduction before/after experts
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )

    def __call__(self, x):
        residuals = x
        inds, scores = self.gate(x)

        if self.moe_latent_size is not None:
            x = self.fc1_latent_proj(x)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.moe_latent_size is not None:
            y = self.fc2_latent_proj(y)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(residuals)

        return y


class NemotronHBlock(nn.Module):
    def __init__(self, args: ModelArgs, block_type: str):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.block_type = block_type

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif self.block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif self.block_type == "-":
            self.mixer = NemotronHMLP(args)
        elif self.block_type == "E":
            self.mixer = NemotronHMoE(args)

    def __call__(
        self,
        x,
        mask: mx.array | None = None,
        cache: Any | None = None,
        ssm_sink: list | None = None,
    ):
        hidden_states = self.norm(x)
        if self.block_type == "M":
            hidden_states = self.mixer(
                hidden_states, mask=mask, cache=cache, ssm_sink=ssm_sink
            )
        elif self.block_type == "*":
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)

        return x + hidden_states


def _moe_layer_args(args: ModelArgs, block_cfg: dict | None) -> ModelArgs:
    """Shallow per-layer view overriding this MoE layer's heterogeneous dims."""
    if not block_cfg:
        return args
    layer_args = copy.copy(args)
    if block_cfg.get("moe_intermediate_size") is not None:
        layer_args.moe_intermediate_size = block_cfg["moe_intermediate_size"]
    if block_cfg.get("num_experts_per_tok") is not None:
        layer_args.num_experts_per_tok = block_cfg["num_experts_per_tok"]
    return layer_args


class NemotronHMTPBlock(NemotronHBlock):
    """One layer of the DeepSeek-style multi-token-prediction head shipped
    in Nemotron 3 Super checkpoints (config: num_nextn_predict_layers,
    mtp_layers_block_type). The first layer carries the embed/hidden fusion
    (eh_proj / enorm / hnorm), the last carries final_layernorm; the block
    itself is a standard NemotronHBlock."""

    def __init__(self, args: ModelArgs, block_type: str, is_first: bool, is_last: bool):
        super().__init__(args, block_type)
        eps = args.layer_norm_epsilon
        if is_first:
            self.eh_proj = nn.Linear(
                2 * args.hidden_size, args.hidden_size, bias=False
            )
            self.enorm = nn.RMSNorm(args.hidden_size, eps=eps)
            self.hnorm = nn.RMSNorm(args.hidden_size, eps=eps)
        if is_last:
            self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)


class NemotronHMTP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        pattern = args.mtp_hybrid_override_pattern
        block_configs = args.mtp_block_configs or [None] * len(pattern)
        n = len(pattern)
        self.layers = [
            NemotronHMTPBlock(
                _moe_layer_args(args, bc) if bt == "E" else args,
                bt,
                i == 0,
                i == n - 1,
            )
            for i, (bt, bc) in enumerate(zip(pattern, block_configs))
        ]


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        pattern = args.hybrid_override_pattern
        block_configs = args.block_configs or [None] * len(pattern)
        self.layers = [
            NemotronHBlock(
                _moe_layer_args(args, bc) if block_type == "E" else args,
                block_type,
            )
            for block_type, bc in zip(pattern, block_configs)
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in args.hybrid_override_pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(
        self,
        inputs,
        cache: Any | None = None,
        ssm_sink: list | None = None,
    ):
        hidden_states = self.embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        fa_cache = cache[self.fa_idx]
        if getattr(fa_cache, "_nemotron_unpadded_verify", False):
            # The B1 verifier is fully live. Use the ordinary cache's fast
            # causal form instead of a materialized batch-padding mask.
            attn_mask = None if hidden_states.shape[1] == 1 else "causal"
        else:
            attn_mask = create_attention_mask(hidden_states, fa_cache)
        ssm_cache = cache[self.ssm_idx]
        ssm_mask = None
        # A merged single-row MTP cache carries padding metadata even when
        # every position is live. Keep its recurrent update on the same
        # unmasked sequential path as ordinary decoding in that case.
        if ssm_cache is not None and hidden_states.shape[0] == 1:
            left_padding = ssm_cache._left_padding_vector()
            lengths = ssm_cache._length_vector()
            if not ((left_padding is None or left_padding[0] == 0) and (
                lengths is None or lengths[0] >= hidden_states.shape[1]
            )):
                ssm_mask = create_ssm_mask(hidden_states, ssm_cache)
        else:
            ssm_mask = create_ssm_mask(hidden_states, ssm_cache)

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c, ssm_sink=ssm_sink)

        return self.norm_f(hidden_states)


class Model(nn.Module):
    mtp_align_full_final_chunk = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type
        if args.num_nextn_predict_layers > 0 and args.mtp_hybrid_override_pattern:
            self.mtp = NemotronHMTP(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
    ):
        out = self.backbone(inputs, cache=cache)
        return self.lm_head(out)

    def mtp_backbone(self, inputs: mx.array, cache=None):
        hidden = self.backbone(inputs, cache=cache)
        return hidden, hidden

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif l.block_type == "*":
                caches.append(KVCache())
        return caches

    def logits(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def mtp_verify_backbone(self, tokens: mx.array, cache):
        """Verify B1 draft tokens with ordinary one-token state updates.

        The short block SSM and attention kernels have different rounding
        from ordinary decoding at this checkpoint's long contexts. A tied
        argmax can otherwise authorize a different token even with exact
        recurrent rollback. Batched rows retain the shared verifier path.
        """
        if tokens.shape[0] != 1 or tokens.shape[1] == 1:
            return self.mtp_backbone(tokens, cache)
        logits_hidden = []
        draft_hidden = []
        attn_caches = [entry for entry in cache if isinstance(entry, BatchKVCache)]
        try:
            for entry in attn_caches:
                if entry._right_padding is None and int(entry.left_padding[0].item()) == 0:
                    entry._nemotron_unpadded_verify = True
            for index in range(tokens.shape[1]):
                output, hidden = self.mtp_backbone(tokens[:, index:index + 1], cache)
                logits_hidden.append(output)
                draft_hidden.append(hidden)
        finally:
            for entry in attn_caches:
                entry.__dict__.pop("_nemotron_unpadded_verify", None)
        return mx.concatenate(logits_hidden, axis=1), mx.concatenate(draft_hidden, axis=1)

    @property
    def quant_predicate(self):
        if self.model_type != "nemotron_h_puzzle":
            return lambda _path, _module: True
        # Puzzle's large output projection is unusually sensitive to low-bit
        # affine quantization; keep the lm_head at its checkpoint precision.
        return lambda path, _: path != "lm_head"

    def make_mtp_cache(self):
        return [
            ArraysCache(size=2) if layer.block_type == "M" else KVCache()
            for layer in self.mtp.layers
            if layer.block_type in ("M", "*")
        ]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """One MTP forward over S positions.

        hidden: [B, S, H] post-norm_f hiddens at positions p..p+S-1 (from
        the backbone, or from a previous mtp_step when chaining draft
        depth). tokens: [B, S] the tokens at positions p+1..p+S (the
        committed or drafted token FOLLOWING each hidden's position).
        Returns (logits [B, S, V], post_final_layernorm hidden [B, S, H]).

        The MTP KV cache offset counts pairs fed, i.e. positions are
        uniformly shifted by -1 vs absolute; the attention layer only
        attends within its own cache so the shift is harmless (NemotronH
        attention uses no rope)."""
        first = self.mtp.layers[0]
        e = first.enorm(self.backbone.embeddings(tokens))
        h = first.hnorm(hidden)
        x = first.eh_proj(mx.concatenate([e, h], axis=-1))
        # The shared executor may wrap KVCache in SegmentedBatchKVCache. Use
        # the cache ABI, not a concrete cache class, to obtain its prepared
        # causal/padding mask for a batched MTP verification step.
        fa_cache = mtp_cache[0]
        mask = create_attention_mask(x, fa_cache)
        cache_index = 0
        for layer in self.mtp.layers:
            c = None
            if layer.block_type in ("M", "*"):
                c = mtp_cache[cache_index]
                cache_index += 1
            x = layer(x, mask=mask, cache=c)
        post = self.mtp.layers[-1].final_layernorm(x)
        return self.lm_head(post), post

    def rollback_speculative_cache(self, caches, ssm_states, keep, block_size):
        """Rewind target caches after a speculative verify forward of
        `block_size` tokens of which the first `keep` are kept.

        KV caches trim normally. Mamba2 caches (ArraysCache) hold a
        recurrent state that cannot trim, so they are rebuilt by replaying
        the captured verify inputs (`ssm_states`, from an `ssm_sink` passed
        to the verify forward) on the kept prefix — one ssm_update per
        Mamba layer since A_log/D/dt_bias are per-layer vectors.
        Single-sequence (B=1, unpadded) only."""
        trim = block_size - keep
        ssm_caches = []
        for c in caches:
            if c is None:
                continue
            if c.is_trimmable():
                if trim > 0:
                    c.trim(trim)
            else:
                if c.lengths is not None or c.left_padding is not None:
                    raise ValueError(
                        "rollback_speculative_cache supports single-sequence "
                        "caches only (lengths/left_padding must be None)"
                    )
                ssm_caches.append(c)
        if not ssm_caches or trim == 0:
            return
        if len(ssm_caches) != len(ssm_states):
            raise ValueError(
                f"ssm_states has {len(ssm_states)} entries for "
                f"{len(ssm_caches)} Mamba caches"
            )

        for c, st in zip(ssm_caches, ssm_states):
            x, B, C, dt, A_log, D, dt_bias, tsl, state, mask, padded, K = st
            if keep == 0:
                # Nothing kept: restore the pre-verify state verbatim.
                c[1] = state
                c[0] = padded[:, : K - 1]
                continue
            _, new_state = ssm_update(
                x[:, :keep],
                A_log,
                B[:, :keep],
                C[:, :keep],
                D.astype(x.dtype),
                dt[:, :keep],
                dt_bias,
                state,
                tsl,
                None if mask is None else mask[:, :keep],
            )
            c[1] = new_state
            c[0] = mx.contiguous(padded[:, keep : keep + K - 1])

    def sanitize(self, weights):
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)
        if not (has_mtp_weights and hasattr(self, "mtp")):
            # Checkpoint has no MTP tensors (or config declared no MTP
            # layers): drop both the weights and the module so strict
            # loading stays consistent.
            weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}
            if hasattr(self, "mtp"):
                self.mtp = None

        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        # Stack experts (backbone and mtp layers alike)
        prefixes = {
            k.rsplit(".experts.", 1)[0] for k in weights if ".experts." in k
        }
        for prefix in prefixes:
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                if f"{prefix}.experts.0.{m}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return weights

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k and "A_log" not in k

        return predicate
