"""DeepSeek V4 ViT and aligner candidate from official inference/vision.py.

Structural MLX port only. The separate mixed-Q4 loader and image prefill
assembly are unexecuted candidates; source parity and image-aware decoder
integration have not been qualified. No serving route may
select this module until those contracts are complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VisionArgs:
    vision_n_layers: int = 32
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    dim: int = 4096


def _vision_rope(x, n_h: int, n_w: int, theta: float):
    """Apply the official 2D RoPE order to [patches, heads, head_dim]."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    if head_dim % 4 or x.shape[0] != n_h * n_w:
        raise ValueError("invalid DeepSeek V4 vision RoPE shape")
    inv_freq = 1.0 / (theta ** (mx.arange(0, half, 2, dtype=mx.float32) / half))
    hp = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    wp = mx.broadcast_to(mx.arange(n_w)[None, :], (n_h, n_w))
    pos = mx.stack((hp, wp), axis=-1).reshape(-1, 2, 1).astype(mx.float32)
    freqs = (pos * inv_freq).reshape(-1, half)
    cos, sin = mx.cos(freqs)[:, None, :], mx.sin(freqs)[:, None, :]
    left, right = mx.split(x.astype(mx.float32), 2, axis=-1)
    output = mx.concatenate((left * cos - right * sin,
                             right * cos + left * sin), axis=-1)
    return output.astype(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def __call__(self, patches):
        return self.proj(patches.reshape(patches.shape[0], -1))


class Attention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // self.n_heads
        self.rope_theta = args.vision_rope_theta
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def __call__(self, x, n_h: int, n_w: int):
        n = x.shape[0]
        q, k, v = (part.reshape(n, self.n_heads, self.head_dim)
                   for part in mx.split(self.wqkv(x), 3, axis=-1))
        q = _vision_rope(q, n_h, n_w, self.rope_theta)
        k = _vision_rope(k, n_h, n_w, self.rope_theta)
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5,
        )[0].transpose(1, 0, 2).reshape(n, -1)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.norm1 = nn.RMSNorm(args.vision_dim, eps=1e-6)
        self.attn = Attention(args)
        self.norm2 = nn.RMSNorm(args.vision_dim, eps=1e-6)
        self.mlp = MLP(args)

    def __call__(self, x, n_h: int, n_w: int):
        x = x + self.attn(self.norm1(x), n_h, n_w)
        return x + self.mlp(self.norm2(x))


class VisionTower(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.patch_embed = PatchEmbed(args)
        self.blocks = [Block(args) for _ in range(args.vision_n_layers)]
        self.norm = nn.RMSNorm(args.vision_dim, eps=1e-6)

    def __call__(self, patches, n_h: int, n_w: int):
        if patches.shape[0] != n_h * n_w:
            raise ValueError("vision patch grid mismatch")
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, n_h, n_w)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.ratio = args.vision_downsample_ratio
        self.w1 = nn.Linear(args.vision_dim * self.ratio**2, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def __call__(self, x, n_h: int, n_w: int):
        r = self.ratio
        if x.shape[0] != n_h * n_w:
            raise ValueError("aligner patch grid mismatch")
        x = x.reshape(n_h, n_w, -1)
        x = mx.pad(x, ((0, -n_h % r), (0, -n_w % r), (0, 0)))
        h, w, dim = x.shape
        # PyTorch unfold orders each patch as channel, kernel-row, kernel-col.
        x = x.reshape(h // r, r, w // r, r, dim)
        x = x.transpose(0, 2, 4, 1, 3).reshape(-1, dim * r * r)
        return self.w2(nn.gelu(self.w1(x)))


class VisionComponents(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.vision = VisionTower(args)
        self.aligner = Aligner(args)

    def __call__(self, patches, n_h: int, n_w: int):
        return self.aligner(self.vision(patches, n_h, n_w), n_h, n_w)
