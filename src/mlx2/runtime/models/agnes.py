# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/agnes-3-flash.json.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from .cache import ArraysCache, KVCache
from .pipeline import PipelineMixin
from .qwen3_5 import GatedDeltaNet
from .qwen3_next import Qwen3NextMLP
from .precise_ops import gate_sigmoid
from .rope_utils import initialize_rope

LAYER_GLOBAL = "agnes_global_attention"
LAYER_DELTA = "agnes_delta_attention"
LAYER_TYPES = (LAYER_GLOBAL, LAYER_DELTA)

@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "agnes_text"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 12288
    parallel_ffn_intermediate_size: int = 0
    vocab_size: int = 248320

    layer_types: Optional[List[str]] = None
    global_attention_interval: int = 4
    full_attention_interval: Optional[int] = None

    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: Optional[int] = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "swish"
    mamba_ssm_dtype: str = "float32"

    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False

    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        rope = dict(self.rope_parameters or self.rope_scaling or {})
        if "type" not in rope and "rope_type" in rope:
            rope["type"] = rope["rope_type"]
        self.rope_parameters = rope or None
        self.partial_rotary_factor = rope.get(
            "partial_rotary_factor", self.partial_rotary_factor
        )
        self.rope_theta = rope.get("rope_theta", self.rope_theta)
        self.rope_scaling = rope or None

        interval = self.full_attention_interval or self.global_attention_interval
        if self.layer_types is None:
            self.layer_types = [
                LAYER_GLOBAL if (i + 1) % interval == 0 else LAYER_DELTA
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "num_hidden_layers must equal the number of layer_types "
                f"({self.num_hidden_layers} != {len(self.layer_types)})"
            )
        unknown = sorted(set(self.layer_types) - set(LAYER_TYPES))
        if unknown:
            raise ValueError(
                f"layer_types entries must be in {LAYER_TYPES}, got {unknown}"
            )
        if self.hidden_act != "silu":
            raise ValueError(
                f"Agnes requires hidden_act='silu', got {self.hidden_act!r}"
            )
        if self.output_gate_type != "swish":
            raise ValueError(
                "Agnes delta attention requires output_gate_type='swish', got "
                f"{self.output_gate_type!r}"
            )
        if not self.attn_output_gate:
            raise ValueError("Agnes global attention requires attn_output_gate=true")


class AgnesMLP(Qwen3NextMLP):
    """The exact Agnes main SwiGLU plus its optional additive parallel SwiGLU."""

    def __init__(self, args: TextModelArgs, intermediate_size: int):
        super().__init__(args.hidden_size, intermediate_size)
        parallel_size = args.parallel_ffn_intermediate_size
        self.parallel_ffn = (
            Qwen3NextMLP(args.hidden_size, parallel_size) if parallel_size > 0 else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        y = super().__call__(x)
        if self.parallel_ffn is not None:
            y = y + self.parallel_ffn(x)
        return y


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * gate_sigmoid(gate))


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == LAYER_DELTA
        if self.is_linear:
            self.delta_attn = GatedDeltaNet(args)
        else:
            self.global_attn = Qwen3NextAttention(args)

        self.mlp = AgnesMLP(args, args.intermediate_size)
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
        h = self.input_layernorm(x)
        if self.is_linear:
            h = self.delta_attn(h, mask=mask, cache=cache)
        else:
            h = self.global_attn(h, mask=mask, cache=cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class AgnesTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self.fa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), None
        )

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = next(
            (i for i, layer in enumerate(self.pipeline_layers) if layer.is_linear),
            None,
        )
        self.fa_idx = next(
            (i for i, layer in enumerate(self.pipeline_layers) if not layer.is_linear),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = (
            create_attention_mask(hidden_states, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(hidden_states, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, pipeline_rank + 1)

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]
        return self.norm(hidden_states)


class TextModel(nn.Module):
    # The cache alternates recurrent GDN state and ordinary attention K/V.
    # Declaring that stable topology lets the server select APCv2, whose
    # layer-segment COW path keeps the two planes at one atomic token boundary.
    apc_v2_layout = "agnes-hybrid-layer-segments-v1"

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AgnesTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            ArraysCache(size=2) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in key and value.shape[-1] != 1
            for key, value in weights.items()
        )

        # The public Agnes implementation advertises MTP in config but does
        # not construct an MTP module and explicitly ignores ``^mtp.*``.
        # Mirror that supported runtime boundary until its weight contract is
        # published rather than guessing a Qwen-shaped draft head.
        weights = {
            key: value
            for key, value in weights.items()
            if ".mtp." not in key and not key.startswith("mtp.")
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights.pop("language_model.lm_head.weight", None)

        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(
                key.endswith(suffix) for suffix in norm_suffixes
            ):
                if value.ndim == 1:
                    weights[key] = value + 1.0
        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            return not path.endswith("A_log")

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    apc_v2_layout = "agnes-hybrid-layer-segments-v1"

    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        normalized = {}
        for key, value in weights.items():
            if (
                key.startswith("vision_tower.")
                or key.startswith("model.visual.")
                or key.startswith("model.vision_tower.")
            ):
                continue

            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif key.startswith("language_model."):
                pass
            elif key.startswith("model."):
                key = "language_model." + key
            elif key.startswith("lm_head.") or key.startswith("mtp."):
                key = "language_model." + key
            else:
                key = "language_model." + key
            normalized[key] = value
        return self.language_model.sanitize(normalized)

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
