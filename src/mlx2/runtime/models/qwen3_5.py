# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from .base import BaseModelArgs
from . import invariant_prefill as _invariant
from .gated_delta import gated_delta_update, normalize_gdn_qk
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated

_GDN_FUSED_MAX_ROWS = 8
# Opt-in: stage a one-graph per-row rollback that takes the accepted count as
# a device array (see GatedDeltaNet._masked_rollback).
from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)
_GDN_ARRAY_ACCEPT = os.environ.get("MLX_LM_GDN_ARRAY_ACCEPT") == "1"


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4
    mtp_num_hidden_layers: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    # Compute dtype from the artifact's text_config (e.g. "bfloat16"); the
    # dense/MoE Qwen3.8 sanitize casts float32 norm gammas to it.
    dtype: Optional[str] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)
        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.sharding_group = None

    def _normalize_qk(self, q, k):
        return normalize_gdn_qk(q, k)

    def _gated_delta_update(
        self, q, k, v, a, b, state, mask, use_kernel, *, prefill=False
    ):
        return gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=use_kernel,
            prefill_chunk_size=getattr(self, "_prefill_scan_chunk", 0)
            if prefill
            else 0,
            prefill_stats=getattr(self, "_prefill_scan_stats", None),
            prefill_segment_rows=getattr(self, "_prefill_scan_segment", 256),
            state_dtype=getattr(self, "_gdn_state_dtype", None),
        )

    def _input_projections(self, inputs: mx.array):
        if hasattr(self, "_prefill_counts"):
            from .tensorfold_prefill import grouped_input

            projected = grouped_input(self, inputs)
            if projected is not None:
                return projected
        if not hasattr(self, "in_proj_fused"):
            return (
                self.in_proj_qkv(inputs),
                self.in_proj_z(inputs),
                self.in_proj_b(inputs),
                self.in_proj_a(inputs),
            )
        fused = self.in_proj_fused
        bounds = self._gdn_fused_bounds
        if (
            inputs.shape[0] * inputs.shape[1] <= _GDN_FUSED_MAX_ROWS
            and inputs.dtype in self._gdn_fused_dtypes
        ):
            return mx.split(fused(inputs), bounds[:-1], axis=-1)
        outputs = []
        lower = 0
        matmul = mx.quantized_matmul
        if _invariant.active():
            def matmul(x, w, scales, biases, *, transpose, group_size, bits):
                return _invariant.quantized_matmul(
                    x, w, scales, biases, group_size=group_size, bits=bits
                )
        for upper in bounds:
            outputs.append(
                matmul(
                    inputs,
                    fused.weight[lower:upper],
                    fused.scales[lower:upper],
                    fused.biases[lower:upper],
                    transpose=True,
                    group_size=fused.group_size,
                    bits=fused.bits,
                )
            )
            lower = upper
        return outputs

    def _try_fused_decode(self, qkv, z, b, a, mask, cache):
        """Architecture-specific decode shortcut; stock models opt out."""
        return None

    def _masked_rollback(
        self, accepted, q, k, v, a, b, state, conv_input, use_kernel
    ):
        """Replay the full verify width with rejected steps made identities.

        ``accepted`` is a per-row count (host list, or a device array that is
        never read back). Positions ``>= accepted[row]`` get ``a = b = -inf``,
        i.e. decay ``exp(-exp(A) * softplus(-inf)) == 1`` and ``beta ==
        sigmoid(-inf) == 0``, so those steps leave the state bit-for-bit
        unchanged while the accepted prefix runs the same kernel arithmetic as
        the sliced ``q[:, :m]`` replay. One graph serves every row length.
        """
        (B, S) = a.shape[:2]
        if not isinstance(accepted, mx.array):
            accepted = mx.array([int(value) for value in accepted])
        ends = mx.broadcast_to(accepted.reshape(-1).astype(mx.int32), (B,))
        live = (mx.arange(S)[None, :] < ends[:, None])[..., None]
        a = mx.where(live, a, -mx.inf)
        b = mx.where(live, b, -mx.inf)
        (_, restored_state) = self._gated_delta_update(
            q, k, v, a, b, state, None, use_kernel
        )
        n_keep = self.conv_kernel_size - 1
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        restored_conv = mx.take_along_axis(conv_input, positions, axis=1)
        return [mx.contiguous(restored_conv), restored_state]

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        (B, S, _) = inputs.shape
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)
        (qkv, z, b, a) = self._input_projections(inputs)
        fused = self._try_fused_decode(qkv, z, b, a, mask, cache)
        if fused is not None:
            return fused
        out = self._recurrent_core(qkv, z, b, a, mask, cache, dtype=inputs.dtype)
        out = self.out_proj(out.reshape(B, S, -1))
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out

    def _recurrent_core(self, qkv, z, b, a, mask, cache, *, dtype):
        """Everything between the input and output projections.

        Shared by ``__call__`` and :meth:`mixed`, so the mixed prefill+decode
        forward runs exactly the ordinary path's conv, normalization,
        recurrence and gated norm per segment.  Returns the gated-norm output
        before ``out_proj``.
        """
        (B, S) = qkv.shape[:2]
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        (q, k, v) = [
            t.reshape(B, S, h, d)
            for (t, h, d) in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        (q, k) = self._normalize_qk(q, k)
        spans = ()
        if cache is not None:
            describe = getattr(cache, "rollback_spans", None)
            if describe is not None:
                spans = describe(S, mask)
        if (
            cache is not None
            and getattr(cache, "speculating", False)
            and (spans is not None)
        ):
            n_keep = self.conv_kernel_size - 1
            use_kernel = not self.training

            def _rollback(
                m, q=q, k=k, v=v, a=a, b=b, S0=state, ci=conv_input, nk=n_keep
            ):
                (_, s_m) = self._gated_delta_update(
                    q[:, :m],
                    k[:, :m],
                    v[:, :m],
                    a[:, :m],
                    b[:, :m],
                    S0,
                    None,
                    use_kernel,
                )
                return [mx.contiguous(ci[:, m : m + nk, :]), s_m]

            if not _GDN_ARRAY_ACCEPT:
                cache.record_rollback(S, _rollback, [conv_state, state])
            else:

                def _rollback_rows(
                    lengths, q=q, k=k, v=v, a=a, b=b, S0=state, ci=conv_input
                ):
                    return self._masked_rollback(
                        lengths, q, k, v, a, b, S0, ci, use_kernel
                    )

                def _rollback_any(m, sliced=_rollback, rows=_rollback_rows):
                    return rows(m) if isinstance(m, mx.array) else sliced(m)

                cache.record_rollback(
                    S, _rollback_any, [conv_state, state], per_row_fn=_rollback_rows
                )
        (out, state) = self._gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            state,
            mask,
            not self.training,
            prefill=S > 17 and not bool(getattr(cache, "speculating", False)),
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.norm(out, z)

    def mixed(self, inputs, parts):
        """One packed forward over several cache segments.

        ``inputs`` is the packed ``[1, N, D]`` token stream; ``parts`` is a
        sequence of ``(rows, length, start, cache, mask)`` with
        ``rows * length`` tokens starting at ``start``.  Projections run once
        on the packed stream (one weight read for every segment); the
        convolution and recurrence run per segment against that segment's
        cache.  The fused decode shortcut is not used: a decode row here
        rides inside a prefill-width matmul by design.
        """
        if self.sharding_group is not None:
            raise ValueError("mixed forward is not qualified with tensor sharding")
        (qkv, z, b, a) = self._input_projections(inputs)
        outs = []
        for rows, length, start, cache, mask in parts:
            if bool(getattr(cache, "speculating", False)):
                raise ValueError("mixed forward cannot run inside a speculative transaction")
            span = slice(start, start + rows * length)

            def seg(t):
                return t[:, span].reshape(rows, length, -1)

            out = self._recurrent_core(
                seg(qkv), seg(z), seg(b), seg(a), mask, cache, dtype=inputs.dtype
            )
            outs.append(out.reshape(1, rows * length, -1))
        return self.out_proj(mx.concatenate(outs, axis=1))
