"""Independent text-only D3 backbone; see provenance/d3-text-decisions.json."""

from copy import deepcopy

import mlx.core as mx
from mlx import nn

from ...runtime.models.gated_delta import gated_delta_update
from ...runtime.models.qwen3_5 import GatedDeltaNet, TextModelArgs
from ...runtime.models.qwen38_27b import Qwen3NextAttention


class RawRMSNorm(nn.Module):
    """HF Qwen norm: normalize and multiply by (1 + raw weight) in FP32."""

    def __init__(self, dimensions, eps):
        super().__init__()
        self.weight = mx.zeros((dimensions,))
        self.eps = eps

    def __call__(self, x):
        value = x.astype(mx.float32)
        value = value * mx.rsqrt(
            mx.mean(value * value, axis=-1, keepdims=True) + self.eps
        )
        return (value * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


class D3GatedNorm(nn.Module):
    def __init__(self, dimensions, eps):
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.eps = eps

    def __call__(self, x, gate):
        value = x.astype(mx.float32)
        value = value * mx.rsqrt(
            mx.mean(value * value, axis=-1, keepdims=True) + self.eps
        )
        value = self.weight * value.astype(x.dtype)
        return (value.astype(mx.float32) * nn.silu(gate.astype(mx.float32))).astype(
            x.dtype
        )


class D3GatedDeltaNet(GatedDeltaNet):
    """The trained reference's FP32 recurrence with explicit BF16 boundaries."""

    def __init__(self, args):
        super().__init__(args)
        self.norm = D3GatedNorm(args.linear_value_head_dim, args.rms_norm_eps)

    def _normalize_qk(self, q, k):
        q, k = q.astype(mx.float32), k.astype(mx.float32)
        q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        return q * (q.shape[-1] ** -0.5), k

    def _gated_delta_update(
        self, q, k, v, a, b, state, mask, use_kernel, *, prefill=False
    ):
        result, state = gated_delta_update(
            q,
            k,
            v.astype(mx.float32),
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=use_kernel,
            beta_input_dtype=True,
            decay_input_float32=True,
        )
        return result.astype(v.dtype), state


class D3MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class D3TextRoPE(nn.Module):
    """Text positions of the reference mRoPE, including stored-dtype cos/sin."""

    def __init__(self, dimensions, theta):
        super().__init__()
        self.dimensions = dimensions
        self.theta = theta

    def __call__(self, x):
        dimensions = self.dimensions
        inverse = self.theta ** (
            -mx.arange(0, dimensions, 2, dtype=mx.float32) / dimensions
        )
        angles = mx.arange(x.shape[-2], dtype=mx.float32)[:, None] * inverse[None, :]
        angles = mx.concatenate([angles, angles], axis=-1)
        cosine, sine = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        rotated, tail = x[..., :dimensions], x[..., dimensions:]
        half = mx.concatenate(
            [-rotated[..., dimensions // 2 :], rotated[..., : dimensions // 2]], axis=-1
        )
        return mx.concatenate([rotated * cosine + half * sine, tail], axis=-1)


class D3Layer(nn.Module):
    def __init__(self, args, kind):
        super().__init__()
        self.is_linear = kind == "linear_attention"
        if self.is_linear:
            self.linear_attn = D3GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)
            self.self_attn.q_norm = RawRMSNorm(args.head_dim, args.rms_norm_eps)
            self.self_attn.k_norm = RawRMSNorm(args.head_dim, args.rms_norm_eps)
            self.self_attn.rope = D3TextRoPE(
                int(args.head_dim * args.partial_rotary_factor), args.rope_theta
            )
        self.input_layernorm = RawRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RawRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = D3MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x):
        mixer = self.linear_attn if self.is_linear else self.self_attn
        # No cache or causal full-attention mask. GDN retains its causal scan.
        hidden = x + mixer(self.input_layernorm(x), mask=None, cache=None)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class D3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        kinds = config.get("layer_types")
        if (
            not isinstance(kinds, list)
            or len(kinds) != config.get("num_hidden_layers")
            or any(kind not in {"linear_attention", "full_attention"} for kind in kinds)
        ):
            raise ValueError(
                "D3 requires an explicit valid layer_types entry per layer"
            )
        args = TextModelArgs.from_dict(deepcopy(config))
        if args.num_experts:
            raise ValueError("D3 text adapter supports dense layers only")
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [D3Layer(args, kind) for kind in kinds]
        self.norm = RawRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, tokens):
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError(
                "D3 requires one independent nonempty question per forward"
            )
        hidden = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

    @staticmethod
    def sanitize(weights):
        result = {}
        for name, value in weights.items():
            if not name.startswith("language_model."):
                continue
            name = name.removeprefix("language_model.")
            if name.endswith("conv1d.weight"):
                if value.ndim != 3 or value.shape[1] != 1:
                    raise ValueError(
                        "D3 requires unconverted HF depthwise convolution weights"
                    )
                value = value.moveaxis(2, 1)
            result[name] = value
        return result
