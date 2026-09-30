"""RADIO math mined from pinned MIT MLX stacks; see provenance/radio-image-encoder.json."""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from ...adapters.radio_config import RadioConfig


def _interpolate_bilinear_nchw(
    x: mx.array, size: tuple[int, int], align_corners: bool = False
) -> mx.array:
    """torch.nn.functional.interpolate(mode='bilinear', antialias=False) for NCHW.

    Matches torch's `area_pixel_compute_source_index`:
      align_corners=False -> src = scale * (i + 0.5) - 0.5, clamped at 0
      align_corners=True  -> src = (in - 1) / (out - 1) * i
    """
    _b, _c, h, w = x.shape
    oh, ow = size
    if (h, w) == (oh, ow):
        return x

    def _src_index(out_len: int, in_len: int) -> mx.array:
        idx = mx.arange(out_len, dtype=mx.float32)
        if align_corners:
            scale = (in_len - 1) / (out_len - 1) if out_len > 1 else 0.0
            return idx * scale
        scale = in_len / out_len
        src = scale * (idx + 0.5) - 0.5
        return mx.maximum(src, 0.0)

    sy = _src_index(oh, h)
    sx = _src_index(ow, w)

    y0 = mx.floor(sy).astype(mx.int32)
    x0 = mx.floor(sx).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    wy = (sy - y0.astype(mx.float32)).reshape(1, 1, oh, 1)
    wx = (sx - x0.astype(mx.float32)).reshape(1, 1, 1, ow)

    # gather rows then columns
    top = x[:, :, y0, :]  # (b, c, oh, w)
    bot = x[:, :, y1, :]
    top_l = top[:, :, :, x0]
    top_r = top[:, :, :, x1]
    bot_l = bot[:, :, :, x0]
    bot_r = bot[:, :, :, x1]

    top_i = top_l + (top_r - top_l) * wx
    bot_i = bot_l + (bot_r - bot_l) * wx
    return top_i + (bot_i - top_i) * wy


@dataclass
class RadioOutput:
    summary: mx.array
    features: mx.array


class InputConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_mean = mx.zeros((3, 1, 1))
        self.norm_std = mx.ones((3, 1, 1))

    def __call__(self, x):
        return (x - self.norm_mean) / self.norm_std


class ClsToken(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int, num_registers: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_registers = num_registers
        self.token = mx.zeros((self.num_tokens + self.num_registers, embed_dim))

    @property
    def num_patches(self):
        return self.num_tokens + self.num_registers

    def __call__(self, x):
        token = mx.broadcast_to(
            self.token[None, :, :],
            (x.shape[0], self.token.shape[0], self.token.shape[1]),
        ).astype(x.dtype)
        return mx.concatenate([token, x], axis=1)


class ViTPatchGenerator(nn.Module):
    def __init__(self, config: RadioConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.align_corners = config.align_corners
        self.patch_size = config.patch_size
        self.embed_dim = embed_dim
        self.num_rows = config.max_resolution // self.patch_size
        self.num_cols = self.num_rows
        self.num_patches = self.num_rows * self.num_cols
        self.cls_token = ClsToken(
            embed_dim, config.num_cls_tokens, config.num_registers
        )
        patch_size = self.patch_size
        self.embedder = nn.Linear(3 * patch_size * patch_size, embed_dim, bias=False)
        self.pos_embed = mx.zeros((1, self.num_patches, embed_dim))

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def _im_to_patches(self, x):
        batch, channels, height, width = x.shape
        patch = self.patch_size
        patch_h = height // patch
        patch_w = width // patch
        x = x.reshape(batch, channels, patch_h, patch, patch_w, patch)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        return x.reshape(batch, patch_h * patch_w, channels * patch * patch)

    def _get_pos_embeddings(self, batch_size, input_dims):
        pe = self.pos_embed.reshape(1, self.num_rows, self.num_cols, self.embed_dim)
        pe = pe.transpose(0, 3, 1, 2).astype(mx.float32)
        extent = max(input_dims)
        pe = _interpolate_bilinear_nchw(pe, (extent, extent), self.align_corners)
        pe = pe[:, :, : input_dims[0], : input_dims[1]]
        pe = pe.reshape(1, self.embed_dim, -1).transpose(0, 2, 1)
        return mx.broadcast_to(
            pe.astype(self.pos_embed.dtype), (batch_size, pe.shape[1], self.embed_dim)
        )

    def __call__(self, x):
        patches = self.embedder(self._im_to_patches(x))
        dims = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        patches = patches + self._get_pos_embeddings(x.shape[0], dims).astype(
            patches.dtype
        )
        return self.cls_token(patches)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_hidden_dim)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RadioBackbone(nn.Module):
    def __init__(self, config: RadioConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_generator = ViTPatchGenerator(config)
        self.blocks = [
            Block(
                self.embed_dim,
                num_heads=config.num_attention_heads,
                mlp_hidden_dim=config.intermediate_size,
            )
            for _ in range(config.num_hidden_layers)
        ]

    def forward_features(self, x):
        x = self.patch_generator(x)
        for block in self.blocks:
            x = block(x)
        return x


class RadioModel(nn.Module):
    """Standalone image encoder. Input is raw RGB in [0, 1], NCHW."""

    def __init__(self, config: RadioConfig):
        super().__init__()
        self.config = config
        self.input_conditioner = InputConditioner()
        self.model = RadioBackbone(config)
        self.summary_idxs = mx.array(config.summary_idxs, dtype=mx.int32)

    def __call__(self, x):
        cfg = self.config
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[0] < 1:
            raise ValueError("RADIO expects a nonempty NCHW RGB batch")
        if any(
            d < cfg.patch_size or d > cfg.max_resolution or d % cfg.patch_size
            for d in x.shape[-2:]
        ):
            raise ValueError(
                "RADIO image dimensions must be patch-aligned and within max_resolution"
            )
        x = self.input_conditioner(x).astype(
            self.model.patch_generator.embedder.weight.dtype
        )
        y = self.model.forward_features(x)
        summary = y[:, self.summary_idxs, :].reshape(y.shape[0], -1)
        return RadioOutput(summary, y[:, cfg.num_skip :])
