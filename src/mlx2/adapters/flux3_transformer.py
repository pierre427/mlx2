"""Native MLX FLUX 3 DiT; mined from the pinned BFL Apache-2.0 reference.

See provenance/flux3-image-klein.json. One image per call; reference tokens
share the image-conditioning stream and retain distinct fourth-axis ids.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

MODES = ("image", "image_cond", "text")


def timestep_embedding(t, dim=256):
    frequencies = mx.exp(
        -math.log(10000) * mx.arange(dim // 2, dtype=mx.float32) / (dim // 2)
    )
    # BFL multiplies in the timestep's dtype before building FP32 phases.
    scaled = 1000 * t
    phase = scaled.astype(mx.float32)[..., None] * frequencies
    return mx.concatenate((mx.cos(phase), mx.sin(phase)), axis=-1).astype(t.dtype)


def rotary_angles(ids, axes_dim=(32, 32, 32, 32), theta=10000):
    parts = []
    for axis, dim in enumerate(axes_dim):
        frequencies = theta ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        parts.append(ids[:, axis, None].astype(mx.float32) * frequencies)
    angles = mx.concatenate(parts, axis=-1)[None, None]
    return mx.cos(angles), mx.sin(angles)


def rmsnorm_rope(x, scale, frequencies):
    dtype = x.dtype
    x = x.astype(mx.float32)
    x = (
        x
        * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        * scale.astype(mx.float32)
    )
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    c, s = frequencies
    a, b = pairs[..., 0], pairs[..., 1]
    return (
        mx.stack((c * a - s * b, s * a + c * b), axis=-1).reshape(x.shape).astype(dtype)
    )


def modulate(x, scale, shift):
    return (1 + scale) * mx.fast.layer_norm(x, None, None, eps=1e-6) + shift


class RMSScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = mx.ones((dim,))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSScale(dim)
        self.key_norm = RMSScale(dim)


class Embedder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x):
        return self.out_layer(nn.silu(self.in_layer(x)))


class Modulation(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, 3 * hidden, bias=False)

    def __call__(self, vec):
        return mx.split(self.lin(nn.silu(vec)), 3, axis=-1)


class LastLayer(nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.linear = nn.Linear(hidden, channels, bias=False)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False)]

    def __call__(self, x, vec):
        shift, scale = mx.split(self.adaLN_modulation[1](nn.silu(vec)), 2, axis=-1)
        return self.linear(modulate(x, scale, shift))


class Block(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        for name in ("q_proj", "k_proj", "v_proj", "attn_out"):
            setattr(self, name, nn.Linear(hidden, hidden, bias=False))
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp_in = nn.Linear(hidden, 2 * mlp_hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def __call__(self, x, frequencies, modulation):
        shift, scale, gate = modulation
        h = modulate(x, scale, shift)

        def split(t):
            return t.reshape(1, t.shape[0], self.heads, -1).transpose(0, 2, 1, 3)

        q = rmsnorm_rope(split(self.q_proj(h)), self.norm.query_norm.scale, frequencies)
        k = rmsnorm_rope(split(self.k_proj(h)), self.norm.key_norm.scale, frequencies)
        v = split(self.v_proj(h))
        attention = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=q.shape[-1] ** -0.5
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(x.shape)
        a, b = mx.split(self.mlp_in(h), 2, axis=-1)
        return x + gate * (self.attn_out(attention) + self.mlp_out(nn.silu(a) * b))


class Flux3Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.get("flow_map"):
            raise ValueError("this adapter requires the distilled flow-map checkpoint")
        hidden, heads = config["hidden"], config["heads"]
        channels = config["in_channels"]
        self.axes_dim = tuple(config["axes_dim"])
        self.theta = config["theta"]
        self.emb_in = {m: nn.Linear(channels[m], hidden, bias=False) for m in MODES[:2]}
        self.txt_in = nn.Linear(channels["text"], hidden, bias=False)
        self.time_in = Embedder(256, hidden)
        self.ref_time_in = Embedder(256, hidden)
        self.early_stream_modulations = {
            m: Modulation(hidden) for m in ("image", "image_cond", "txt")
        }
        self.single_stream_modulations = {
            m: Modulation(hidden) for m in ("image", "image_cond", "txt")
        }

        def blocks(count):
            return [Block(hidden, heads, config["mlp_ratio"]) for _ in range(count)]

        self.content_mode_blocks = {m: blocks(config["depth"]) for m in MODES[:2]}
        self.txt_mode_blocks = blocks(config["depth"])
        self.single_blocks = blocks(config["depth_single"])
        self.final_layer = {"image": LastLayer(hidden, channels["image"])}

    def __call__(self, data, ids, timestep, target):
        # Empty streams are omitted: reference phantom tokens are never attended.
        modes = [m for m in MODES if data[m].shape[0]]
        dtype = self.txt_in.weight.dtype
        ref = self.ref_time_in(timestep_embedding(mx.array([target], dtype=dtype)))
        vec, early, single, x, frequencies = {}, {}, {}, {}, {}
        for mode in modes:
            time = timestep if mode == "image" else 0.0
            vec[mode] = (
                self.time_in(timestep_embedding(mx.array([time], dtype=dtype))) + ref
            )
            key = "txt" if mode == "text" else mode
            early[mode] = self.early_stream_modulations[key](vec[mode])
            single[mode] = self.single_stream_modulations[key](vec[mode])
            projection = self.txt_in if mode == "text" else self.emb_in[mode]
            x[mode] = projection(data[mode].astype(dtype))
            frequencies[mode] = rotary_angles(ids[mode], self.axes_dim, self.theta)
        for depth in range(len(self.txt_mode_blocks)):
            for mode in modes:
                block = (
                    self.txt_mode_blocks[depth]
                    if mode == "text"
                    else self.content_mode_blocks[mode][depth]
                )
                x[mode] = block(x[mode], frequencies[mode], early[mode])
        h = mx.concatenate([x[m] for m in modes], axis=0)
        frequencies = rotary_angles(
            mx.concatenate([ids[m] for m in modes]), self.axes_dim, self.theta
        )
        mods = tuple(
            mx.concatenate([mx.broadcast_to(single[m][i], x[m].shape) for m in modes])
            for i in range(3)
        )
        for block in self.single_blocks:
            h = block(h, frequencies, mods)
        return self.final_layer["image"](h[: data["image"].shape[0]], vec["image"])


def timesteps(steps, alpha=6.93):
    t = mx.linspace(1, 0, steps + 1, dtype=mx.float32)
    return alpha * t / (1 + (alpha - 1) * t)


def latent_ids(height, width):
    rows, cols = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
    zero = mx.zeros_like(rows)
    return mx.stack((zero, rows, cols, zero), axis=-1).reshape(-1, 4).astype(mx.float32)
