"""Native MLX MOSS-Image transformer and flow-matching schedule.

The tensor names intentionally match the publisher's Diffusers checkpoint.
See provenance/moss-image-sft.json.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn


def _rms_norm(x, weight, eps=1e-6):
    return mx.fast.rms_norm(x, weight, eps).astype(x.dtype)


def _linear_input_dtype(layer):
    """Activation dtype for dense and packed MLX linear modules."""
    scales = getattr(layer, "scales", None)
    return scales.dtype if scales is not None else layer.weight.dtype


def situ(hidden, gate):
    """Publisher's bounded SiTU gated activation."""
    return (4 * mx.tanh(gate / 4) * mx.sigmoid(gate)) * (25 * mx.tanh(hidden / 25))


def rotate(x, rope):
    sine, cosine = mx.split(rope, 2, axis=-1)
    sine = mx.concatenate((sine, sine), axis=-1)[:, None]
    cosine = mx.concatenate((cosine, cosine), axis=-1)[:, None]
    a, b = mx.split(x, 2, axis=-1)
    return x * cosine + mx.concatenate((-b, a), axis=-1) * sine


def attention(q, k, v, rope, heads):
    length, width = q.shape
    shape = (1, length, heads, width // heads)
    q = rotate(q.reshape(shape), rope).transpose(0, 2, 1, 3)
    k = rotate(k.reshape(shape), rope).transpose(0, 2, 1, 3)
    v = v.reshape(shape).transpose(0, 2, 1, 3)
    result = mx.fast.scaled_dot_product_attention(
        q.astype(v.dtype), k.astype(v.dtype), v, scale=q.shape[-1] ** -0.5
    )
    return result.transpose(0, 2, 1, 3).reshape(length, width)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        return _rms_norm(x, self.weight)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def __call__(self, x, rope):
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        q = self.q_norm(q.reshape(q.shape[0], self.heads, -1)).reshape(q.shape)
        k = self.k_norm(k.reshape(k.shape[0], self.heads, -1)).reshape(k.shape)
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = [TextBlock(dim, ffn_dim, heads) for _ in range(2)]

    def __call__(self, x, rope):
        x = self.proj(x.astype(_linear_input_dtype(self.proj)))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = {
            name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")
        }
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation
        y = self.norm1(x) * (1 + scale_a) + shift_a
        dtype = _linear_input_dtype(self.proj["q"])
        y = y.astype(dtype)
        q = self.proj["q"](y)
        k = self.proj["k"](y)
        q = self.q_norm(q.reshape(q.shape[0], self.heads, -1)).reshape(q.shape)
        k = self.k_norm(k.reshape(k.shape[0], self.heads, -1)).reshape(k.shape)
        x = (
            x
            + self.proj["o"](attention(q, k, self.proj["v"](y), rope, self.heads))
            * gate_a
        )
        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = mx.split(self.proj["ffn_up_gate"](y.astype(dtype)), 2, axis=-1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f


def rotary_embedding(height, width, text_length, head_dim, base=10000.0):
    """Four-axis MOSS rotary embedding in image-then-text sequence order."""
    bands = 1 / (base ** (mx.arange(head_dim // 8, dtype=mx.float32) / (head_dim // 8)))
    rows, columns = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
    image = (
        mx.stack(
            (
                mx.zeros_like(rows),
                mx.full(rows.shape, text_length),
                rows,
                columns,
            ),
            axis=-1,
        )
        .reshape(-1, 4)
        .astype(mx.float32)
    )
    image = image.at[:, 2].add(-(height - 1) / 2)
    image = image.at[:, 3].add(-(width - 1) / 2)
    text = mx.zeros((text_length, 4), dtype=mx.float32)
    text = text.at[:, 1].add(mx.arange(text_length))
    phase = mx.concatenate((image, text), axis=0)[..., None] * bands
    return mx.concatenate((mx.sin(phase), mx.cos(phase)), axis=-2).reshape(
        height * width + text_length, head_dim
    )


def timestep_embedding(timestep, dim):
    # The reference constructs the phase in float64. Python's scalar math keeps
    # that boundary without requiring unsupported float64 Metal tensors.
    half = dim // 2
    values = [float(timestep) * (10000 ** (-i / half)) for i in range(half)]
    return mx.array(
        [[math.cos(value) for value in values] + [math.sin(value) for value in values]],
        dtype=mx.float32,
    )


class MOSSImageTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = int(config["dim"])
        ffn_dim = int(config["ffn_dim"])
        heads = int(config["num_heads"])
        self.config = dict(config)
        self.video_embedder = nn.Linear(int(config["latent_channels"]), dim)
        self.text_embedder = TextProjection(
            int(config["text_dim"]), dim, ffn_dim, heads
        )
        self.time_embedding = [
            nn.Linear(int(config["freq_dim"]), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        ]
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = [
            Block(dim, ffn_dim, heads) for _ in range(int(config["num_layers"]))
        ]
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(
            dim, int(config["latent_channels"]), bias=False
        )
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def _time(self, timestep):
        embedding = timestep_embedding(timestep, int(self.config["freq_dim"]))
        value = self.time_embedding[0](
            embedding.astype(_linear_input_dtype(self.time_embedding[0]))
        )
        return self.time_embedding[2](nn.silu(value))

    def __call__(self, latents, timestep, context, *, skip_layers=()):
        """Run one B=1 DiT forward over NHWC latent state and unpadded context."""
        if latents.ndim != 4 or latents.shape[0] != 1:
            raise ValueError("MOSS-Image native execution requires B=1 NHWC latents")
        if context.ndim != 2 or context.shape[-1] != int(self.config["text_dim"]):
            raise ValueError(
                "MOSS-Image context must be an unpadded [S, text_dim] array"
            )
        height, width = latents.shape[1:3]
        if latents.shape[-1] != int(self.config["latent_channels"]):
            raise ValueError("MOSS-Image latent channel count differs")
        skipped = frozenset(skip_layers)
        if any(
            type(index) is not int or not 0 <= index < len(self.layers)
            for index in skipped
        ):
            raise ValueError("skip_layers contains an invalid block index")
        time = self._time(timestep).astype(mx.float32)
        modulation = self.shared_block_modulation(
            time.astype(_linear_input_dtype(self.shared_block_modulation))
        ).reshape(6, -1)
        head = self.video_head_modulation(
            time.astype(_linear_input_dtype(self.video_head_modulation))
        ).reshape(2, -1)
        video = latents.reshape(-1, latents.shape[-1])
        video = self.video_embedder(video.astype(_linear_input_dtype(self.video_embedder)))
        rope = rotary_embedding(
            height,
            width,
            context.shape[0],
            int(self.config["dim"]) // int(self.config["num_heads"]),
            float(self.config.get("rope_base", 10000.0)),
        ).astype(video.dtype)
        text = self.text_embedder(context, rope[height * width :])
        hidden = mx.concatenate((video, text), axis=0)
        for index, layer in enumerate(self.layers):
            if index not in skipped:
                hidden = layer(hidden, rope, modulation)
        hidden = self.final_norm_video(hidden[: height * width])
        hidden = hidden * (1 + head[1]) + head[0]
        result = self.final_linear_video(
            hidden.astype(_linear_input_dtype(self.final_linear_video))
        )
        return result.reshape(1, height, width, -1)


def flow_schedule(steps, *, shift=5.0, training_steps=1000, training_shift=1.0):
    """Reference V2 Euler sigmas, shifted and snapped to the training grid."""
    if type(steps) is not int or steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("positive integer steps and positive shifts are required")

    def grid(count, value):
        lower = value / (training_steps + value - 1)
        base = mx.linspace(1, lower, count, dtype=mx.float32)
        return value * base / (1 + (value - 1) * base)

    requested = grid(steps, shift)
    training = grid(training_steps, training_shift)[::-1]
    # Only 50-ish scalar values cross this boundary; Python selection avoids a
    # device-side searchsorted dependency and preserves the explicit schedule.
    train = training.tolist()
    selected = []
    for sigma in requested.tolist():
        index = min(range(len(train)), key=lambda i: abs(train[i] - sigma))
        if not selected or train[index] != selected[-1]:
            selected.append(train[index])
    return mx.array(selected + [0.0], dtype=mx.float32)
