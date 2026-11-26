"""SenseNova's fixed text loop over the shared Qwen3.5 inference components."""

from functools import partial

import mlx.core as mx
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from .base import create_attention_mask, create_ssm_mask
from .cache import ArraysCache, KVCache
from .qwen3_5 import TextModelArgs
from .qwen38_27b import DecoderLayer


def _sinkhorn_reference(logits, iterations):
    for _ in range(iterations):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


def _logsumexp_pair(a, b):
    high = mx.maximum(a, b)
    return high + mx.log1p(mx.exp(-mx.abs(a - b)))


@partial(mx.compile, shapeless=True)
def _sinkhorn_pairs(a, b, c, d, iterations):
    """Keep the two-by-two normalization as elementwise operations for fusion."""
    for _ in range(iterations):
        row0 = _logsumexp_pair(a, b)
        row1 = _logsumexp_pair(c, d)
        a, b, c, d = a - row0, b - row0, c - row1, d - row1
        col0 = _logsumexp_pair(a, c)
        col1 = _logsumexp_pair(b, d)
        a, b, c, d = a - col0, b - col1, c - col0, d - col1
    return mx.exp(a), mx.exp(b), mx.exp(c), mx.exp(d)


def _sinkhorn_two_streams(logits, iterations):
    entries = (logits[..., index] for index in range(4))
    return mx.stack(_sinkhorn_pairs(*entries, iterations), axis=-1).reshape(
        *logits.shape[:-1], 2, 2
    )


class MHCRouter(nn.Module):
    def __init__(self, width, iterations, pre_scale, compiled):
        super().__init__()
        self.weight = mx.zeros((8, 2 * width), dtype=mx.float32)
        self.bias = mx.zeros((8,), dtype=mx.float32)
        self.iterations = iterations
        self.pre_scale = pre_scale
        self.compiled = compiled

    def __call__(self, residual):
        flat = residual.flatten(-2).astype(mx.float32)
        normalized = mx.fast.rms_norm(flat, None, 1e-6)
        logits = mx.addmm(self.bias, normalized, self.weight.T)
        pre = mx.sigmoid(logits[..., :2]).astype(residual.dtype)
        post = 2 * mx.sigmoid(logits[..., 2:4])
        mix_logits = logits[..., 4:]
        if self.compiled:
            mix = _sinkhorn_two_streams(mix_logits, self.iterations)
        else:
            mix = _sinkhorn_reference(
                mix_logits.reshape(*mix_logits.shape[:-1], 2, 2), self.iterations
            )
        branch = (pre[..., None, :] @ residual).squeeze(-2)
        if self.pre_scale != 1:
            branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        mixed = mix.astype(residual.dtype) @ residual
        update = post.astype(branch.dtype)[..., None] * branch[..., None, :]
        return mixed + update


class _ReleaseRMSNorm(nn.RMSNorm):
    """The release multiplies by FP32 gamma before rounding the output."""

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(
            x.dtype
        )


class LoopLayer(DecoderLayer):
    def __init__(self, args, index, mhc, compiled):
        super().__init__(args, index)
        self.input_layernorm = _ReleaseRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = _ReleaseRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if not self.is_linear:
            self.self_attn.q_norm = _ReleaseRMSNorm(
                args.head_dim, eps=args.rms_norm_eps
            )
            self.self_attn.k_norm = _ReleaseRMSNorm(
                args.head_dim, eps=args.rms_norm_eps
            )
        self.multistream = mhc["start"] <= index < mhc["end"]
        if self.multistream:
            self.enter = index == mhc["start"]
            self.leave = index == mhc["end"] - 1
            router_args = (
                args.hidden_size,
                mhc["iterations"],
                mhc.get("pre_scale", 1.0),
                compiled,
            )
            self.mhc_attn = MHCRouter(*router_args)
            self.mhc_ffn = MHCRouter(*router_args)

    def __call__(self, x, mask=None, cache=None):
        if not self.multistream:
            return super().__call__(x, mask, cache)
        residual = mx.stack((x, x), axis=-2) if self.enter else x
        branch, post, mix = self.mhc_attn(residual)
        mixer = self.linear_attn if self.is_linear else self.self_attn
        branch = mixer(self.input_layernorm(branch), mask, cache)
        residual = MHCRouter.merge(residual, branch, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        residual = MHCRouter.merge(residual, branch, post, mix)
        return mx.mean(residual, axis=-2) if self.leave else residual


class Model(nn.Module):
    def __init__(self, config, visits, layout, *, compiled_mhc=False):
        super().__init__()
        text = dict(config["text_config"])
        text["rope_parameters"] = dict(text["rope_parameters"])
        args = TextModelArgs.from_dict(text)
        self.model_type = config["model_type"]
        self.apc_v2_layout = layout
        self.visits = visits
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LoopLayer(args, index, config["mhc"], compiled_mhc)
            for index in range(args.num_hidden_layers)
        ]
        self.norm = _ReleaseRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.linear_visits = tuple(self.layers[index].is_linear for index in visits)
        self.first_linear = self.linear_visits.index(True)
        self.first_attention = self.linear_visits.index(False)

    def make_cache(self):
        return [
            ArraysCache(size=2) if linear else KVCache()
            for linear in self.linear_visits
        ]

    def __call__(self, inputs, cache=None):
        hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = (None,) * len(self.visits)
        if len(cache) != len(self.visits):
            raise ValueError("SenseNova Loop requires one cache slot per layer visit")
        attention_mask = create_attention_mask(hidden, cache[self.first_attention])
        recurrent_mask = create_ssm_mask(hidden, cache[self.first_linear])
        for index, linear, state in zip(
            self.visits, self.linear_visits, cache, strict=True
        ):
            mask = recurrent_mask if linear else attention_mask
            hidden = self.layers[index](hidden, mask, state)
        return self.lm_head(self.norm(select_checkpoint_rows(hidden)))

    @staticmethod
    def sanitize(weights):
        """Translate raw HF releases before each shard is materialized."""
        result = {}
        for key, value in weights.items():
            # The release retains a training-time loop gate. Inference fixes the
            # visit count in config and the vendor forward never reads this tensor.
            if key == "model.language_model.loop_gate.logits" or key.startswith(
                ("model.visual.", "mtp.")
            ):
                continue
            key = key.removeprefix("model.language_model.")
            if key.endswith("conv1d.weight"):
                value = value.moveaxis(2, 1)
            elif key == "norm.weight" or key.endswith(
                (
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".q_norm.weight",
                    ".k_norm.weight",
                )
            ):
                value = 1 + value.astype(mx.float32)
            if ".mhc_" in key and value.dtype != mx.float32:
                raise ValueError(f"SenseNova router requires FP32 weights: {key}")
            result[key] = value
        return result
