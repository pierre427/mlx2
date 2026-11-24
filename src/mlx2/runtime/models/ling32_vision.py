# SPDX-License-Identifier: MIT
# Adapted from mlx-vlm; see provenance/ling32-air-vision.json and .NOTICE.
"""Ling-3.2-air vision tower (Qwen3-VL ViT without DeepStack) and projector.

The tower follows the vendor law of ``BailingMoeV3VLVisionModel``
(inclusionAI/Ling-3.2-air, read as law, never imported): Conv3d patch
embedding, bilinear position-table interpolation in merge order, pre-norm
blocks with full non-causal attention inside each image, 2D vision RoPE, and
a norm-only merger (``disable_merger_proj``). The projector is
``Linear -> exact GELU -> Linear``.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

VISION_FILE = 'vision.safetensors'

# The only geometry this port serves. VisionConfig.from_dict refuses anything
# else; direct construction (tests) still enforces the structural invariants.
LING32_VISION_GEOMETRY = {
    'model_type': 'qwen3_moe_vit',
    'depth': 27,
    'hidden_size': 1152,
    'hidden_act': 'gelu_pytorch_tanh',
    'intermediate_size': 4304,
    'num_heads': 16,
    'in_channels': 3,
    'patch_size': 16,
    'spatial_merge_size': 2,
    'temporal_patch_size': 2,
    'num_position_embeddings': 2304,
}
ROPE_THETA = 10000.0
LAYER_NORM_EPS = 1e-6

vision_stats = {'encoder_calls': 0, 'images': 0, 'patches': 0, 'feature_tokens': 0}


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    model_type: str = 'qwen3_moe_vit'
    hidden_act: str = 'gelu_pytorch_tanh'

    def __post_init__(self):
        if self.hidden_act != 'gelu_pytorch_tanh':
            raise ValueError('Ling-3.2-air vision supports only gelu_pytorch_tanh')
        if self.hidden_size % self.num_heads:
            raise ValueError('Ling-3.2-air vision hidden_size must divide by num_heads')
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 4:
            raise ValueError('Ling-3.2-air vision 2D RoPE needs head_dim divisible by 4')
        side = int(round(self.num_position_embeddings ** 0.5))
        if side * side != self.num_position_embeddings:
            raise ValueError('Ling-3.2-air vision position table must be square')

    @property
    def merged_hidden_size(self) -> int:
        return self.hidden_size * self.spatial_merge_size ** 2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2

    @classmethod
    def from_dict(cls, config: dict) -> 'VisionConfig':
        if not isinstance(config, dict):
            raise ValueError('Ling-3.2-air vision_config must be a mapping')
        for key, value in LING32_VISION_GEOMETRY.items():
            if config.get(key) != value:
                raise ValueError(
                    f'Unsupported Ling-3.2-air vision geometry: {key}={config.get(key)!r}, expected {value!r}')
        if config.get('disable_merger_proj') is not True:
            raise ValueError('Ling-3.2-air vision requires disable_merger_proj=true (norm-only merger)')
        if list(config.get('deepstack_visual_indexes') or []):
            raise ValueError('Ling-3.2-air vision has no DeepStack; deepstack_visual_indexes must be empty')
        # out_hidden_size is unused when disable_merger_proj is true (the
        # local config says 4096, the vendor's later config 4608); the served
        # width is hidden_size * merge**2 = 4608 and is checked at load.
        return cls(**{key: config[key] for key in LING32_VISION_GEOMETRY})


def _grid_list(grid_thw) -> List[List[int]]:
    if isinstance(grid_thw, mx.array):
        grid_thw = np.asarray(grid_thw)
    grid = np.asarray(grid_thw, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] < 1:
        raise ValueError('Ling-3.2-air vision grid_thw must be a non-empty list of [t, h, w]')
    if (grid < 1).any():
        raise ValueError('Ling-3.2-air vision grid entries must be positive')
    return grid.tolist()


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rotary_pos_emb_vision(tensor: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Vendor law: rotate in fp32, then cast back to the tensor dtype."""
    dtype = tensor.dtype
    x = tensor.astype(mx.float32)
    return (x * cos + rotate_half(x) * sin).astype(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                              kernel_size=kernel, stride=kernel, bias=True)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # kernel == stride over exactly one patch: the Conv3d is one matmul.
        # The vendor's bf16 path also runs it as unfold + F.linear.
        x = pixel_values.reshape(-1, self.in_channels, self.temporal_patch_size,
                                 self.patch_size, self.patch_size).moveaxis(1, 4)
        x = x.reshape(x.shape[0], -1)
        weight = self.proj.weight.reshape(self.hidden_size, -1)
        return mx.addmm(self.proj.bias, x, weight.T)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, split_points: List[int], cos: mx.array, sin: mx.array) -> mx.array:
        length = x.shape[0]
        qkv = self.qkv(x).reshape(length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = apply_rotary_pos_emb_vision(q, cos, sin)
        k = apply_rotary_pos_emb_vision(k, cos, sin)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
        if split_points:
            chunks = [mx.split(t, split_points, axis=2) for t in (q, k, v)]
            out = mx.concatenate(
                [mx.fast.scaled_dot_product_attention(qc, kc, vc, scale=self.scale)
                 for qc, kc, vc in zip(*chunks)], axis=2)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out[0].transpose(1, 0, 2).reshape(length, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=LAYER_NORM_EPS)
        self.attn = Attention(config.hidden_size, config.num_heads)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def __call__(self, x, split_points, cos, sin):
        x = x + self.attn(self.norm1(x), split_points, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """disable_merger_proj: LayerNorm(hidden) then view(-1, hidden * merge**2)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merged = config.merged_hidden_size
        self.norm = nn.LayerNorm(config.hidden_size, eps=LAYER_NORM_EPS)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x).reshape(-1, self.merged)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(round(config.num_position_embeddings ** 0.5))
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        head_dim = config.hidden_size // config.num_heads
        self._rotary_dim = head_dim // 2

    def _merge_order(self, array: np.ndarray, h: int, w: int) -> np.ndarray:
        m = self.spatial_merge_size
        tail = array.shape[2:]
        return array.reshape(h // m, m, w // m, m, *tail).transpose(
            0, 2, 1, 3, *range(4, 4 + len(tail))).reshape(h * w, *tail)

    def rot_pos_emb(self, grid: Sequence[Sequence[int]]):
        positions = []
        for t, h, w in grid:
            rows = np.broadcast_to(np.arange(h)[:, None], (h, w))
            cols = np.broadcast_to(np.arange(w)[None, :], (h, w))
            coords = self._merge_order(np.stack((rows, cols), axis=-1), h, w)
            positions.append(np.tile(coords, (t, 1)))
        positions = mx.array(np.concatenate(positions), dtype=mx.int32)
        max_hw = max(max(h, w) for _, h, w in grid)
        dim = self._rotary_dim
        inv_freq = 1.0 / (ROPE_THETA ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        table = mx.arange(max_hw, dtype=mx.float32)[:, None] * inv_freq[None, :]
        freqs = mx.concatenate((table[positions[:, 0]], table[positions[:, 1]]), axis=-1)
        freqs = mx.concatenate((freqs, freqs), axis=-1)[:, None, :]
        return mx.cos(freqs), mx.sin(freqs)

    def fast_pos_embed_interpolate(self, grid: Sequence[Sequence[int]]) -> mx.array:
        side = self.num_grid_per_side
        indices, weights = [], []
        for t, h, w in grid:
            h_idx = np.linspace(0, side - 1, h, dtype=np.float32)
            w_idx = np.linspace(0, side - 1, w, dtype=np.float32)
            h0, w0 = h_idx.astype(np.int64), w_idx.astype(np.int64)
            h1, w1 = np.minimum(h0 + 1, side - 1), np.minimum(w0 + 1, side - 1)
            dh = (h_idx - h0)[:, None]
            dw = (w_idx - w0)[None, :]
            # Vendor weight formulas (w00 = 1 - dh - w01).
            w11 = dh * dw
            w10 = dh - w11
            w01 = dw - w11
            w00 = 1 - dh - w01
            idx = np.stack([h0[:, None] * side + w0[None, :], h0[:, None] * side + w1[None, :],
                            h1[:, None] * side + w0[None, :], h1[:, None] * side + w1[None, :]], axis=-1)
            wts = np.stack(np.broadcast_arrays(w00, w01, w10, w11), axis=-1).astype(np.float32)
            idx = np.tile(self._merge_order(idx, h, w), (t, 1))
            wts = np.tile(self._merge_order(wts, h, w), (t, 1))
            indices.append(idx)
            weights.append(wts)
        dtype = self.pos_embed.weight.dtype
        indices = mx.array(np.concatenate(indices).T, dtype=mx.int32)
        weights = mx.array(np.concatenate(weights).T).astype(dtype)
        return mx.sum(self.pos_embed(indices) * weights[..., None], axis=0)

    def __call__(self, pixel_values, grid_thw) -> mx.array:
        grid = _grid_list(grid_thw)
        m = self.spatial_merge_size
        if any(h % m or w % m for _, h, w in grid):
            raise ValueError('Ling-3.2-air vision grid h and w must be multiples of the merge size')
        total = sum(t * h * w for t, h, w in grid)
        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(np.asarray(pixel_values))
        if pixel_values.ndim != 2 or pixel_values.shape[1] != self.config.patch_dim:
            raise ValueError(f'Ling-3.2-air vision pixel_values must be [N, {self.config.patch_dim}]')
        if pixel_values.shape[0] != total:
            raise ValueError(
                f'Ling-3.2-air vision pixel rows {pixel_values.shape[0]} do not match grid patches {total}')

        dtype = self.patch_embed.proj.weight.dtype
        hidden = self.patch_embed(pixel_values.astype(dtype))
        hidden = hidden + self.fast_pos_embed_interpolate(grid)
        cos, sin = self.rot_pos_emb(grid)
        lengths = [h * w for t, h, w in grid for _ in range(t)]
        split_points = np.cumsum(lengths)[:-1].tolist()
        for block in self.blocks:
            hidden = block(hidden, split_points, cos, sin)
        out = self.merger(hidden)

        vision_stats['encoder_calls'] += 1
        vision_stats['images'] += len(grid)
        vision_stats['patches'] += total
        vision_stats['feature_tokens'] += total // (m * m)
        return out


class Projector(nn.Module):
    """``linear_proj = Sequential(Linear, GELU() exact erf, Linear)``."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_proj = [nn.Linear(in_dim, out_dim, bias=True), nn.GELU(),
                            nn.Linear(out_dim, out_dim, bias=True)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.linear_proj:
            x = layer(x)
        return x


def sanitize(weights: dict, config: VisionConfig = None) -> dict:
    """Map vision.safetensors keys to ``vision_tower.*`` / ``linear_proj.*``.

    The torch Conv3d weight ``[O, C, T, P, P]`` becomes MLX ``[O, T, P, P, C]``.
    Unknown keys are refused rather than dropped.
    """
    out = {}
    for key, value in weights.items():
        if key.startswith('model.visual.'):
            name = 'vision_tower.' + key[len('model.visual.'):]
        elif key.startswith('vision_tower.') or key.startswith('linear_proj.'):
            name = key
        else:
            raise ValueError(f'Unexpected Ling-3.2-air vision weight {key!r}')
        if name == 'vision_tower.patch_embed.proj.weight':
            c = config.in_channels if config else 3
            t = config.temporal_patch_size if config else 2
            p = config.patch_size if config else value.shape[-1]
            if tuple(value.shape[1:]) == (c, t, p, p):
                value = value.transpose(0, 2, 3, 4, 1)
            elif tuple(value.shape[1:]) != (t, p, p, c):
                raise ValueError(f'Unexpected Ling-3.2-air patch_embed weight shape {value.shape}')
        out[name] = value
    return out


def _split(weights: dict, prefix: str) -> list:
    return [(k[len(prefix):], v) for k, v in weights.items() if k.startswith(prefix)]


def load_vision(model_path) -> tuple:
    path = Path(model_path)
    config = json.loads((path / 'config.json').read_text())
    vision_config = VisionConfig.from_dict(config.get('vision_config'))
    text_hidden = config.get('hidden_size')
    if isinstance(config.get('text_config'), dict):
        text_hidden = config['text_config'].get('hidden_size', text_hidden)
    if not isinstance(text_hidden, int) or text_hidden <= 0:
        raise ValueError('Ling-3.2-air config has no text hidden_size for the projector')
    weights = sanitize(mx.load(str(path / VISION_FILE)), vision_config)
    tower = VisionModel(vision_config)
    projector = Projector(vision_config.merged_hidden_size, text_hidden)
    for module, prefix in ((tower, 'vision_tower.'), (projector, '')):
        part = _split(weights, prefix) if prefix else [(k, v) for k, v in weights.items()
                                                        if k.startswith('linear_proj.')]
        expected = dict(tree_flatten(module.parameters()))
        for name, value in part:
            if name in expected and tuple(expected[name].shape) != tuple(value.shape):
                raise ValueError(f'Ling-3.2-air vision weight {name} has shape {value.shape}, '
                                 f'expected {expected[name].shape}')
        module.load_weights(part, strict=True)
    tower.eval()
    projector.eval()
    mx.eval(tower.parameters(), projector.parameters())
    return tower, projector
